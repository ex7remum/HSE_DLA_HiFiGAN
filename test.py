import torch
from tqdm import tqdm
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import gdown
from scipy.io.wavfile import write

import hw_tts.model as module_model
from hw_tts.preprocess.text import text_to_sequence
from hw_tts.model import WaveGlowInfer
from hw_tts.utils.parse_config import ConfigParser
from hw_tts.utils import ROOT_PATH

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def get_batch(name, raw_text, text_id, alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
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


def main(config, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = config.init_obj(config["arch"], module_model)

    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    waveglow = WaveGlowInfer('./pretrained_models/waveglow_256channels_ljs_v2.pt', device)

    test_text = [
            "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
            "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
            "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
            "Vsem privet, s vami professor moriarti"
        ]

    test_data = defaultdict(list)
    for idx, text in enumerate(test_text):
        test_data[idx].append(get_batch("all=0.8", text, idx, alpha=0.8, pitch_alpha=0.8, energy_alpha=0.8))
        test_data[idx].append(get_batch("all=1.0", text, idx, alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0))
        test_data[idx].append(get_batch("all=1.2", text, idx, alpha=1.2, pitch_alpha=1.2, energy_alpha=1.2))

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
                test_data[idx].append(get_batch(name, text, idx, alpha, pitch_alpha, energy_alpha))

    with torch.no_grad():
        for idx, batches in tqdm(test_data.items()):
            for batch in tqdm(batches):
                batch["text"] = batch["text"].to(device)
                batch["src_pos"] = batch["src_pos"].to(device)
                res = model(**batch)
                audio = waveglow(res["mel_output"][0])
                name = os.path.join(out_dir, "text={}_{}.wav".format(idx, batch["name"]))
                write(name, 22050, audio.cpu().numpy())


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="test_output",
        type=str,
        help="Directory to write results",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    wg_dir = './pretrained_models'
    os.makedirs(wg_dir, exist_ok=True)

    print('Donwloading pretrained WaveGlow')
    gdown.download("https://drive.google.com/uc?export=download&id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx",
                   wg_dir + '/waveglow_256channels_ljs_v2.pt')

    model_url = 'https://drive.google.com/uc?export=download&id=1srknAuXrL0C9ULtXKmqKc2qK3txDPomy'
    model_path = './pretrained_models/model.pth'
    if not os.path.exists(model_path):
        print('Downloading FastSpeech2 model.')
        gdown.download(model_url, model_path)
        print('Downloaded FastSpeech2 model.')
    else:
        print('FastSpeech2 model already exists.')

    config_url = 'https://drive.google.com/uc?export=download&id=1GI32lN--jI8uLtX2955KNy_pwMos-9Kb'
    config_path = './pretrained_models/config.json'
    if not os.path.exists(config_path):
        print('Downloading FastSpeech2 model test config.')
        gdown.download(config_url, config_path)
        print('Downloaded FastSpeech2 model test config.')
    else:
        print('FastSpeech2 model config already exists.')

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config, args.output)
