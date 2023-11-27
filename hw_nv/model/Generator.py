import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from hw_nv.preprocess.MelSpectrogram import MelSpectrogramConfig


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size, dilation, lrelu_slope=0.1):
        super(ResBlock, self).__init__()

        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList()
        for i in range(len(dilation)):
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i],
                               padding=get_padding(kernel_size, dilation[i]))))

        self.convs2 = nn.ModuleList()
        for i in range(len(dilation)):
            self.convs2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt = c2(xt)
            x = xt + x
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, channels, kernel_size, resnet_kernel, resnet_dilation, lrelu_slope=0.1):
        super(GeneratorBlock, self).__init__()

        self.lrelu_slope = lrelu_slope
        padding = (kernel_size - kernel_size // 2) // 2
        self.upsample = weight_norm(nn.ConvTranspose1d(channels, channels // 2, kernel_size,
                                                       kernel_size // 2, padding=padding))

        self.MFR = nn.ModuleList()
        for i in range(len(resnet_kernel)):
            self.MFR.append(ResBlock(channels//2, resnet_kernel[i], resnet_dilation[i], lrelu_slope))

    def forward(self, x):
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.upsample(x)
        sum_x = None
        for res_block in self.MFR:
            out = res_block(x)
            sum_x = out if sum_x is None else sum_x + out
        x = sum_x / len(self.residuals)
        return x


class Generator(nn.Module):
    def __init__(self, h=512, k_u=None, k_r=None, d_r=None, lrelu_slope=0.1):
        super(Generator, self).__init__()

        # default is V1
        if k_u is None:
            self.k_u = [16, 16, 4, 4]
        else:
            self.k_u = k_u
        if k_r is None:
            self.k_r = [3, 7, 11]
        else:
            self.k_r = k_r
        if d_r is None:
            self.d_r = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        else:
            self.d_r = d_r

        self.h_channels = h
        self.k_u = k_u
        self.enter_conv = weight_norm(nn.Conv1D(in_channels=MelSpectrogramConfig.n_mels, out_channels=h,
                                      kernel_size=7, stride=1, padding=3))
        self.blocks = nn.ModuleList()

        for l in range(len(self.k_u)):
            block = GeneratorBlock(self.h_channels // 2**l, self.k_u[l], self.k_r, self.d_r, lrelu_slope)
            self.blocks.append(block)

        self.out_conv = weight_norm(nn.Conv1d(self.h_channels // 2**len(self.k_u), out_channels=1,
                                              kernel_size=7, stride=1, padding=3))

        self.lrelu_slope = lrelu_slope

    def forward(self, x):
        x = self.enter_conv(x)
        for block in self.blocks:
            x = block(x)
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.out_conv(x)
        x = torch.tanh(x).squeeze(1)
        return x
