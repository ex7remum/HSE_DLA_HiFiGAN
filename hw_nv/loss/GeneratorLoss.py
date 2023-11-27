import torch
import torch.nn as nn
import torch.nn.functional as F
from hw_nv.preprocess.MelSpectrogram import MelSpectrogram


class GeneratorLoss(nn.Module):
    def __init__(self, mel_coef=45, fm_coef=2):
        super(GeneratorLoss, self).__init__()
        self.mel_coef = mel_coef
        self.fm_coef = fm_coef
        self.mel_gen = MelSpectrogram()
        self.l1_loss = nn.L1Loss()

    def generator_loss(self, gen_score):
        loss = 0
        for g_score in gen_score:
            loss += torch.mean((1 - g_score) ** 2)
        return loss

    def feature_loss(self, real_features, gen_features):
        loss = 0
        for r_features, g_features in zip(real_features, gen_features):
            for r_sub_features, g_sub_features in zip(r_features, g_features):
                loss += F.l1_loss(g_sub_features, r_sub_features)
        return loss

    def forward(self, spectrogram, gen_audio, gen_mpd_score,
                gen_msd_score, real_mpd_features, gen_mpd_features,
                real_msd_features, gen_msd_features, **kwargs):

        gen_mel = self.mel_gen(gen_audio)
        mel_loss = self.l1_loss(spectrogram, gen_mel)

        msd_adv_loss = self.generator_loss(gen_msd_score)
        mpd_adv_loss = self.generator_loss(gen_mpd_score)
        adv_loss = msd_adv_loss + mpd_adv_loss

        fm_mpd = self.feature_loss(real_mpd_features, gen_mpd_features)
        fm_msd = self.feature_loss(real_msd_features, gen_msd_features)
        fm_loss = fm_mpd + fm_msd

        g_loss = self.mel_coef * mel_loss + self.fm_coef * fm_loss + adv_loss

        return {
            "g_loss": g_loss,
            "msd_adv_loss": msd_adv_loss,
            "mpd_adv_loss": mpd_adv_loss,
            "adv_loss": adv_loss,
            "mel_loss": mel_loss,
            "fm_mpd": fm_mpd,
            "fm_msd": fm_msd,
            "fm_loss": fm_loss
        }









