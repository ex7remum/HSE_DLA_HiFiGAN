import torch
from torch.nn import nn
from hw_tts.model.AdaptorBlocks import VariancePredictor, LengthRegulator


class QuantizationEmbedding(nn.Module):
    def __init__(self, n_bins, min_val, max_val, hidden, mode="linspace"):
        super().__init__()

        if mode == "linspace":
            boundaries = torch.linspace(min_val, max_val, n_bins + 1)[1:-1]
        elif mode == "logspace":
            boundaries = torch.logspace(min_val, max_val, n_bins + 1)[1:-1]
        else:
            raise ValueError()

        self.boundaries = nn.Parameter(boundaries, requires_grad=False)
        self.embedding = nn.Embedding(n_bins, hidden)

    def forward(self, x):
        return self.embedding(torch.bucketize(x, self.boundaries))


class VarianceAdaptor(nn.Module):
    def __init__(self, n_bins_energy, energy_min, energy_max,
                 encoder_dim, n_bins_pitch, pitch_min, pitch_max, *args, **kwargs):
        super().__init__()

        self.length_regulator = LengthRegulator(**kwargs)

        self.energy_predictor = VariancePredictor(**kwargs)
        self.energy_emb = QuantizationEmbedding(
            n_bins_energy,
            energy_min,
            energy_max,
            encoder_dim,
            mode="linspace",
        )

        self.pitch_predictor = VariancePredictor(**kwargs)
        self.pitch_emb = QuantizationEmbedding(
            n_bins_pitch,
            pitch_min,
            pitch_max,
            encoder_dim,
            mode="logspace",
        )

    def forward(
        self,
        x,
        alpha=1.0,
        energy_alpha=1.0,
        pitch_alpha=1.0,
        duration_target=None,
        max_len=None,
        energy_target=None,
        pitch_target=None):

        if duration_target is not None:
            x, duration_prediction = self.length_regulator(x, alpha, duration_target, max_len)

            energy_prediction = self.energy_predictor(x) * energy_alpha
            pitch_prediction = self.pitch_predictor(x) * pitch_alpha

            x = x + self.energy_emb(energy_target) + self.pitch_emb(pitch_target)

            return x, duration_prediction, energy_prediction, pitch_prediction
        else:
            x, mel_pos = self.length_regulator(x, alpha)

            energy_prediction = self.energy_predictor(x) * energy_alpha
            pitch_prediction = self.pitch_predictor(x) * pitch_alpha
            x = x + self.energy_emb(energy_prediction) + self.pitch_emb(pitch_prediction)

            return x, mel_pos
