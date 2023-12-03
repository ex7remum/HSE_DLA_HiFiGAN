import torch
import os
import json
import argparse
from pathlib import Path
import gdown
from scipy.io.wavfile import write
import librosa
import hw_nv.model as module_model
from hw_nv.utils.parse_config import ConfigParser
from hw_nv.preprocess.MelSpectrogram import MelSpectrogram, MelSpectrogramConfig
from hw_nv.utils import ROOT_PATH

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_dir, test_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

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

    mel_gen = MelSpectrogram()
    for filename in os.listdir(test_dir):
        wav, _ = librosa.load(os.path.join(test_dir, filename))
        mel = mel_gen(torch.from_numpy(wav))
        with torch.no_grad():
            gen_audio = model(mel.unsqueeze(0).to(device))["gen_audio"]
            gen_audio = gen_audio.squeeze(0).cpu()
            new_filename = 'gen_' + filename
            name = os.path.join(out_dir, new_filename)
            write(name, MelSpectrogramConfig.sr, gen_audio.cpu().numpy())


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
        "-t",
        "--test",
        default="test_data",
        type=str,
        help="Directory with test audio",
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

    model_url = 'https://drive.google.com/uc?export=download&id=1EpkdueKIsEKB1GFsrexQnpp7i-Y8UgRr'
    model_path = './pretrained_models/model.pth'
    if not os.path.exists(model_path):
        print('Downloading HiFi GAN model.')
        gdown.download(model_url, model_path)
        print('Downloaded HiFi GAN model.')
    else:
        print('HiFi GAN model already exists.')

    config_url = 'https://drive.google.com/uc?export=download&id=1HQbadWinsK7JNz8ygA2_Vfncl93AUpvq'
    config_path = './pretrained_models/config.json'
    if not os.path.exists(config_path):
        print('Downloading HiFi GAN model test config.')
        gdown.download(config_url, config_path)
        print('Downloaded HiFi GAN model test config.')
    else:
        print('HiFi GAN model config already exists.')

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

    main(config, args.output, args.test)