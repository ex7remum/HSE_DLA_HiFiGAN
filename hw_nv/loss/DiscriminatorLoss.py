import torch
import torch.nn as nn


def discriminator_loss(real_score, gen_score):
    loss = 0
    for r_score, g_score in zip(real_score, gen_score):
        loss += torch.mean((1 - r_score) ** 2) + torch.mean(g_score ** 2)
    return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, real_mpd_score, gen_mpd_score,
                real_msd_score, gen_msd_score, **kwargs):
        mpd_loss = discriminator_loss(real_mpd_score, gen_mpd_score)
        msd_loss = discriminator_loss(real_msd_score, gen_msd_score)
        d_loss = mpd_loss + msd_loss
        return {
            "d_loss": d_loss,
            "mpd_loss": mpd_loss,
            "msd_loss": msd_loss
        }
