import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import random
import os


class LJSpeechWAV(Dataset):
    def __init__(self, data_path, max_wav_len=None, limit=None):
        self.data_path = data_path
        self.wav_names = [filename for filename in os.listdir(data_path) if len(filename) >= 5 and
                          filename[-4:] == '.wav']

        self.max_wav_len = max_wav_len
        if limit is not None:
            self.wav_names = self.wav_names[:limit]

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        name = self.wav_names[idx]
        wav, _ = librosa.load(os.path.join(self.data_path, name))
        wav = torch.from_numpy(wav)
        if self.max_wav_len is not None:
            if len(wav) < self.max_wav_len:
                wav = torch.nn.functional.pad(wav, (0, self.max_wav_len - wav.shape[0]), 'constant')
            else:
                wav_start = random.randint(0, len(wav) - self.max_wav_len)
                wav = wav[wav_start: wav_start + self.max_wav_len]
        return {
            "gt_wav": wav
        }
