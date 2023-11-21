import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class VariancePredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, encoder_dim,
                 duration_predictor_filter_size,
                 duration_predictor_kernel_size,
                 dropout=0.1, *args, **kwargs):
        super(VariancePredictor, self).__init__()

        self.input_size = encoder_dim
        self.filter_size = duration_predictor_filter_size
        self.kernel = duration_predictor_kernel_size
        self.conv_output_size = duration_predictor_filter_size
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
        out = self.linear_layer(encoder_output)
        out = out.squeeze(-1)
        return out


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count + k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, encoder_dim, *args, **kwargs):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VariancePredictor(encoder_dim, **kwargs)
        self.max_seq_len = kwargs['max_seq_len']

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration = torch.clamp(torch.expm1(duration_predictor_output), min=0, max=100)
            duration = ((duration + 0.5) * alpha).long()
            duration = duration * (torch.cumsum(duration, dim=-1) <= self.max_seq_len)

            if duration.sum() < 10:
                duration[0][0] = duration[0][0] + 10

            output = self.LR(x, duration)

            mel_pos = torch.arange(1, output.size(1) + 1, dtype=torch.long, device=x.device).unsqueeze(0)
            return output, mel_pos
