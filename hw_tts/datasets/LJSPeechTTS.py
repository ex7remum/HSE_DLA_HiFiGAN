from hw_tts.preprocess.text import text_to_sequence
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os


def get_data_to_buffer(data_path, mel_ground_truth, alignment_path,
                       pitch_path, energy_path, limit=None):
    buffer = []
    text = []
    text_cleaners = ['english_cleaners']
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            text.append(line.strip())

    size = len(text) if limit is None else min(len(text), limit)
    for i in tqdm(range(size)):
        mel_gt_target = np.load(os.path.join(mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1)))
        pitch = np.load(os.path.join(pitch_path, "ljspeech-pitch-%05d.npy" % (i + 1)))
        energy = np.load(os.path.join(energy_path, "ljspeech-energy-%05d.npy" % (i + 1)))
        duration = np.load(os.path.join(alignment_path, str(i)+".npy"))
        character = torch.tensor(text_to_sequence(text[i], text_cleaners)).long()

        batch = {
            "raw_text": text[i],
            "text": character,
            "duration": torch.from_numpy(duration).int(),
            "mel_target": torch.from_numpy(mel_gt_target).float(),
            "pitch": torch.from_numpy(pitch).float(),
            "energy": torch.from_numpy(energy).float(),
        }
        buffer.append(batch)

    return buffer


class LJSpeechTTS(Dataset):
    def __init__(self, data_path, mel_ground_truth, alignment_path, pitch, energy, limit=None):
        buffer = get_data_to_buffer(data_path, mel_ground_truth, alignment_path, pitch, energy, limit=limit)
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]
