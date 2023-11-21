from hw_tts.base import BaseModel
import torch
import torch.nn as nn
from hw_tts.model.VarianceAdaptor import VarianceAdaptor
from hw_tts.model.Blocks import Encoder, Decoder


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech2(BaseModel):
    """ FastSpeech """

    def __init__(self, num_mels, *args, **kwargs):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(**kwargs)
        self.variance_adaptor = VarianceAdaptor(**kwargs)
        self.decoder = Decoder(**kwargs)

        self.mel_linear = nn.Linear(kwargs['decoder_dim'], num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, text, src_pos, mel_pos=None, mel_max_length=None, duration=None, energy=None, pitch=None,
                alpha=1.0, energy_alpha=1.0, pitch_alpha=1.0, **kwargs):

        x, non_pad_mask = self.encoder(text, src_pos)

        if self.training:
            output, log_duration_prediction, energy_prediction, pitch_prediction = self.variance_adaptor(
                x, alpha, energy_alpha, pitch_alpha, duration, mel_max_length, energy, pitch)
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            energy_prediction.masked_fill_(mel_pos == 0, 0.0)
            pitch_prediction.masked_fill_(mel_pos == 0, 0.0)
            output = self.mel_linear(output)
            return {
                "mel_output": output,
                "log_duration_prediction": log_duration_prediction,
                "energy_prediction": energy_prediction,
                "pitch_prediction": pitch_prediction,
            }

        else:
            output, mel_pos = self.variance_adaptor(x, alpha, energy_alpha, pitch_alpha)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return {
                "mel_output": output
            }
