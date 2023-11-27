import itertools
import numpy as np
import PIL
import os
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
import librosa
from tqdm import tqdm
from hw_nv.base import BaseTrainer
from hw_nv.logger.utils import plot_spectrogram_to_buf
from hw_nv.utils import inf_loop, MetricTracker
from hw_nv.preprocess.MelSpectrogram import MelSpectrogramConfig, MelSpectrogram


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            g_criterion,
            d_criterion,
            g_optimizer,
            d_optimizer,
            config,
            device,
            dataloaders,
            lr_g_scheduler=None,
            lr_d_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, g_criterion, d_criterion, g_optimizer, d_optimizer, config, device)
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
        self.lr_g_scheduler = lr_g_scheduler
        self.lr_d_scheduler = lr_d_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "g_loss", "msd_adv_loss", "mpd_adv_loss", "adv_loss"
            "mel_loss", "fm_mpd", "fm_msd", "fm_loss", "d_loss",
            "mpd_loss", "msd_loss", "g grad norm", "d grad norm", writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["gt_wav", "spectrogram"]:
            if tensor_for_gpu in batch:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, type):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            if type == "generator":
                clip_grad_norm_(
                    self.model.generator.parameters(), self.config["trainer"]["grad_norm_clip"]
                )
            else:
                clip_grad_norm_(
                    itertools.chain(self.model.MPD.parameters(),
                                    self.model.MSD.parameters()), self.config["trainer"]["grad_norm_clip"]
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

            if batch_idx % self.log_step == 0 or batch_idx == 1:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx - 1)
                self.logger.debug(
                    "Train Epoch: {} {} g_loss: {:.6f} d_loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["g_loss"].item(), batch["d_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate g", self.lr_g_scheduler.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "learning rate d", self.lr_d_scheduler.get_last_lr()[0]
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

        gen_out = self.model.generator_step(batch['spectrogram'])
        batch.update(gen_out)

        d_out = self.model.disrciminator_step(batch['gen_audio'].detach(),
                                              batch['gt_wav'])
        batch.update(d_out)

        self.model.freeze_discriminator(unfreeze=True)
        self.d_optimizer.zero_grad()
        losses = self.d_criterion(**batch)

        for loss_name, loss_value in losses.items():
            metrics.update(loss_name, loss_value.item())

        batch.update(losses)
        losses["d_loss"].backward()
        self._clip_grad_norm("discriminator")
        self.train_metrics.update("d grad norm", self.get_grad_norm('discriminator'))
        self.d_optimizer.step()
        if self.lr_d_scheduler is not None:
            self.lr_d_scheduler.step()

        self.model.freeze_discriminator(unfreeze=False)
        self.g_optimizer.zero_grad()
        d_out = self.model.disrciminator_step(batch['gen_audio'], batch['gt_wav'])
        batch.update(d_out)

        losses = self.g_criterion(**batch)
        batch.update(losses)
        losses["g_loss"].backward()
        self._clip_grad_norm("generator")
        self.train_metrics.update("g grad norm", self.get_grad_norm('generator'))
        self.g_optimizer.step()
        if self.lr_g_scheduler is not None:
            self.lr_g_scheduler.step()

        for loss_name, loss_value in losses.items():
            metrics.update(loss_name, loss_value.item())

        return batch

    def _evaluation_epoch(self):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        mel_gen = MelSpectrogram()
        test_data_path = './test_data'
        for filename in os.listdir(test_data_path):
            wav, _ = librosa.load(os.path.join(test_data_path, filename))
            mel = mel_gen(wav)
            with torch.no_grad():
                gen_audio = self.model(mel.unsqueeze(0))["gen_audio"]
                gen_audio = gen_audio.squeeze(0)
                self.writer.add_audio(filename + "_gen", gen_audio, MelSpectrogramConfig.sr)
                self.writer.add_audio(filename, wav, MelSpectrogramConfig.sr)

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

        idx = np.random.choice(len(batch["spectrogram"]))
        audio = batch["gen_audio"][idx]
        self.writer.add_audio("train/audio", audio, MelSpectrogramConfig.sr)
            
        audio = batch["gt_wav"][idx]
        self.writer.add_audio("train/audio_target", audio, MelSpectrogramConfig.sr)

        mel_tgt = batch['spectrogram'][idx]
        mel_gen = MelSpectrogram()
        mel_pred = mel_gen(batch['gen_audio'][idx].unsqueeze(0)).squeeze(0)
        self._log_spectrogram("train/mel_output", mel_pred)
        self._log_spectrogram("train/mel_target", mel_tgt)

    def _log_spectrogram(self, name, spectrogram):
        spectrogram = spectrogram.detach().cpu()
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image(name, ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, model_type, norm_type=2):
        if model_type == 'generator':
            parameters = self.model.generator.parameters()
        else:
            parameters = itertools.chain(self.model.MPD.parameters(),
                                         self.model.MSD.parameters())
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
