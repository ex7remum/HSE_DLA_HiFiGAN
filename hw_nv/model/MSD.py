import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class ScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, leaky_relu=0.1):
        super(ScaleDiscriminator, self).__init__()
        norm_msd = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_msd(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_msd(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_msd(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_msd(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_msd(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_msd(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_msd(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            norm_msd(nn.Conv1d(1024, 1, 3, 1, padding=1))
        ])
        self.leaky_relu = leaky_relu

    def forward(self, x):
        features = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i + 1 != len(self.convs):
                x = F.leaky_relu(x, self.leaky_relu)
            features.append(x)
        x = torch.flatten(x, start_dim=1)

        return x, features


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, leaky_relu=0.1):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True, leaky_relu=leaky_relu),
            ScaleDiscriminator(leaky_relu=leaky_relu),
            ScaleDiscriminator(leaky_relu=leaky_relu),
        ])
        self.pooling = nn.AvgPool1d(4, 2, padding=2)
        self.leaky_relu = leaky_relu

    def forward(self, x):
        score = []
        features = []
        x = x.view(x.shape[0], 1, -1)
        for i, discriminator in enumerate(self.discriminators):
            f_score, f_features = discriminator(x)
            score.append(f_score)
            features.append(f_features)
            if i + 1 < len(self.discriminators):
                x = self.pooling(x)

        return score, features