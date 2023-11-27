import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class PeriodDiscriminator(torch.nn.Module):
    def __init__(self, period, leaky_relu=0.1):
        super(PeriodDiscriminator, self).__init__()

        self.leaky_relu = leaky_relu
        self.period = period
        padding = (get_padding(5), 0)
        kernel_size = (5, 1)
        stride = (3, 1)
        self.convs = nn.ModuleList()
        last_n_channels = 1
        for l in range(1, 5):
            self.convs.append(weight_norm(nn.Conv2d(last_n_channels, 2**(5+l), kernel_size, stride, padding)))
            last_n_channels = 2**(5+l)
        self.convs.append(weight_norm(nn.Conv2d(last_n_channels, 1024, kernel_size, stride=1, padding=(2, 0))))
        self.convs.append(weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))))

    def forward(self, x):
        bs, channels, time = x.shape
        if time % self.period != 0:
            n_pad = self.period - (time % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            time = x.shape[-1]
        x = x.view(bs, channels, time // self.period, self.period)

        features = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i + 1 != len(self.convs):
                x = F.leaky_relu(x, self.leaky_relu)
            features.append(x)
        x = torch.flatten(x, start_dim=1)

        return x, features


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=None, leaky_relu=0.1):
        super(MultiPeriodDiscriminator, self).__init__()

        # default values in paper
        if periods is None:
            self.periods = [2, 3, 5, 7, 11]
        else:
            self.periods = periods
        self.leaky_relu = leaky_relu
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(PeriodDiscriminator(period=period, leaky_relu=leaky_relu))

    def forward(self, x):
        score = []
        features = []
        x = x.view(x.shape[0], 1, -1)
        for discriminator in self.discriminators:
            f_score, f_features = discriminator(x)
            score.append(f_score)
            features.append(f_features)

        return score, features
