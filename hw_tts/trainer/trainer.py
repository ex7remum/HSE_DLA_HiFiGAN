from collections import defaultdict
import numpy as np
import PIL
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
from hw_tts.base import BaseTrainer
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import inf_loop, MetricTracker
from hw_tts.preprocess.text import text_to_sequence
from hw_tts.model import WaveGlowInfer


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "pitch_loss", "energy_loss",
            "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.sr = 22050
        self.waveglow = WaveGlowInfer('./pretrained_models/waveglow_256channels_ljs_v2.pt', device)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["text", "src_pos", "mel_target", "mel_pos", "duration", "pitch", "energy"]:
            if tensor_for_gpu in batch:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch), start=1
        ):
            if 'error' in batch:
                continue

            if batch_idx > self.len_epoch:
                break
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0 or batch_idx == 1:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx - 1)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(batch)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        self._evaluation_epoch()

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        if is_train:
            self.optimizer.zero_grad()

        out = self.model(**batch)
        batch.update(out)

        if is_train:
            losses = self.criterion(**batch)
            batch.update(losses)
            losses["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            for loss_name, loss_value in losses.items():
                metrics.update(loss_name, loss_value.item())
            for met in self.metrics:
                metrics.update(met.name, met(**batch))
        return batch

    def _get_batch(self, name, raw_text, text_id, alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
        text = torch.tensor(text_to_sequence(raw_text, ["english_cleaners"])).long()

        batch = {
            "name": name,
            "raw_text": raw_text,
            "text_id": text_id,
            "text": text.unsqueeze(0),
            "src_pos": torch.arange(1, text.shape[0] + 1, dtype=torch.long).unsqueeze(0),
            "alpha": alpha,
            "pitch_alpha": pitch_alpha,
            "energy_alpha": energy_alpha,
        }
        return batch

    def _evaluation_epoch(self):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        test_text = [
            "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
            "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
            "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
        ]

        test_data = defaultdict(list)
        for idx, text in enumerate(test_text):
            test_data[idx].append(self._get_batch("all=0.8", text, idx, alpha=0.8, pitch_alpha=0.8, energy_alpha=0.8))
            test_data[idx].append(self._get_batch("all=1.0", text, idx, alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0))
            test_data[idx].append(self._get_batch("all=1.2", text, idx, alpha=1.2, pitch_alpha=1.2, energy_alpha=1.2))

            for parameter in ["duration", "pitch", "energy"]:
                for val in [0.8, 1.2]:
                    alpha, pitch_alpha, energy_alpha = 1.0, 1.0, 1.0
                    if parameter == "duration":
                        alpha = val
                    elif parameter == "pitch":
                        pitch_alpha = val
                    else:
                        energy_alpha = val
                    name = parameter + '=' + str(val)
                    test_data[idx].append(self._get_batch(name, text, idx, alpha, pitch_alpha, energy_alpha))

        with torch.no_grad():
            for idx, batches in tqdm(test_data.items()):
                for batch in tqdm(batches):
                    batch = self.move_batch_to_device(batch, self.device)
                    res = self.model(**batch)
                    audio = self.waveglow(res["mel_output"][0])
                    name = "text={}/{}".format(idx, batch["name"])
                    self._log_spectrogram(name + "/mel_output", res["mel_output"], 0)
                    self.writer.add_audio(name + "/audio", audio, self.sr)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(self, batch):
        if self.writer is None:
            return

        idx = np.random.choice(len(batch["text"]))
        self.writer.add_text("train/text", batch["raw_text"][idx])

        audio = self.waveglow(batch["mel_output"][idx])
        self.writer.add_audio("train/audio", audio, self.sr)
            
        audio = self.waveglow(batch["mel_target"][idx])
        self.writer.add_audio("train/audio_target", audio, self.sr)

        self._log_spectrogram("train/mel_output", batch["mel_output"], idx)
        self._log_spectrogram("train/mel_target", batch["mel_target"], idx)

    def _log_spectrogram(self, name, spectrogram_batch, idx):
        spectrogram = spectrogram_batch[idx].detach().cpu()
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image(name, ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
