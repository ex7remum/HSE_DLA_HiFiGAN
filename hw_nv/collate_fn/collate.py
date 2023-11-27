import logging
from typing import List
import torch
from hw_nv.preprocess.MelSpectrogram import MelSpectrogram

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    all_audios = [item['gt_wav'] for item in dataset_items]
    batch_audio = torch.stack(all_audios)
    if len(batch_audio.shape) == 1:
        batch_audio = batch_audio.unsqueeze(1)

    mel_gen = MelSpectrogram()
    batch_mel = mel_gen(batch_audio)

    return {
        "gt_wav": batch_audio,
        "spectrogram": batch_mel
    }
