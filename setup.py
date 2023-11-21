import librosa
import numpy as np
import os
import pyworld
from tqdm import tqdm
import gdown
import tarfile
import wget
from hw_tts.preprocess.audio.tools import get_mel
from sklearn.preprocessing import StandardScaler


def normalize(scaler, value_dir):
    for filename in os.listdir(value_dir):
        filename = os.path.join(value_dir, filename)
        values = (np.load(filename) - scaler.mean_[0]) / scaler.scale_[0]
        np.save(filename, values, allow_pickle=False)


if __name__ == "__main__":
    out_dir = './data'
    os.makedirs(out_dir, exist_ok=False)

    mel_dir = os.path.join(out_dir, "mels")
    os.makedirs(mel_dir, exist_ok=False)

    pitch_dir = os.path.join(out_dir, "pitch")
    os.makedirs(pitch_dir, exist_ok=False)

    energy_dir = os.path.join(out_dir, "energy")
    os.makedirs(energy_dir, exist_ok=False)

    print('Donloading texts')
    gdown.download("https://drive.google.com/uc?export=download&id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx",
                   out_dir+'/train.txt')

    print('Downloading mels')
    gdown.download("https://drive.google.com/uc?export=download&id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j",
                   out_dir+'/mel.tar.gz')

    print('Extracting mels')
    with tarfile.open('./data/mel.tar.gz', 'r') as tar:
        tar.extractall(out_dir)

    print('Downloading alignments')
    filename = wget.download('https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip',
                             out=out_dir+'/alignments.zip')

    print('Downolading dataset')
    filename = wget.download('https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                             out=out_dir+'/LJSpeech-1.1.tar.bz2')

    print('Extracting audios')
    with tarfile.open('./data/LJSpeech-1.1.tar.bz2', 'r') as tar:
        tar.extractall(out_dir)

    ljspeech_dir = './data/LJSpeech-1.1'

    wg_dir = './pretrained_models'
    os.makedirs(wg_dir, exist_ok=False)

    print('Donwloading pretrained WaveGlow')
    gdown.download("https://drive.google.com/uc?export=download&id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx",
                   wg_dir+'/waveglow_256channels_ljs_v2.pt')

    pitch_scaler = StandardScaler()
    energy_scaler = StandardScaler()

    texts = []
    print('Preprocessing texts')
    with open(os.path.join(ljspeech_dir, "metadata.csv"), "r", encoding='utf-8') as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            wav_name, _, text = line.strip().split('|')
            texts.append(text)

            wav_path = os.path.join(ljspeech_dir, "wavs", wav_name + ".wav")
            wav, sr = librosa.load(wav_path)

            pitch, t = pyworld.dio(
                wav.astype(np.float64),
                sr,
                frame_period=256 / sr * 1000,
            )
            pitch = pyworld.stonemask(wav.astype(np.float64), pitch, t, sr)

            mel, energy = get_mel(wav_path)
            mel = mel.T.numpy().astype(np.float32)
            energy = energy.squeeze().numpy().astype(np.float32)

            pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            energy_scaler.partial_fit(energy.reshape((-1, 1)))

            np.save(os.path.join(mel_dir, "ljspeech-mel-%05d.npy" % (idx + 1)), mel, allow_pickle=False)
            np.save(os.path.join(pitch_dir, "ljspeech-pitch-%05d.npy" % (idx + 1)), pitch, allow_pickle=False)
            np.save(os.path.join(energy_dir, "ljspeech-energy-%05d.npy" % (idx + 1)), energy, allow_pickle=False)

    with open(os.path.join(out_dir, "train.txt"), "wb", encoding='utf-8') as f:
        f.write('\n'.join(texts))
        f.write('\n')

    normalize(pitch_scaler, pitch_dir)
    normalize(energy_scaler, energy_dir)
