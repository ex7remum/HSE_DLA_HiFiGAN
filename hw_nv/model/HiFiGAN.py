from hw_nv.base import BaseModel
from hw_nv.model.Generator import Generator
from hw_nv.model.MPD import MultiPeriodDiscriminator
from hw_nv.model.MSD import MultiScaleDiscriminator


class HiFiGAN(BaseModel):
    def __init__(self, channels=512, k_u=None, k_r=None, d_r=None, mpd_periods=None, leaky_relu=0.1):
        super(HiFiGAN, self).__init__()

        self.generator = Generator(channels, k_u, k_r, d_r, leaky_relu)

        self.MSD = MultiScaleDiscriminator(leaky_relu)
        self.MPD = MultiPeriodDiscriminator(mpd_periods, leaky_relu)

    def freeze_discriminator(self, unfreeze=False):
        for p in self.MSD.parameters():
            p.requires_grad = unfreeze
        for p in self.MPD.parameters():
            p.requires_grad = unfreeze

    def generator_step(self, spectrogram):
        gen_audio = self.generator(spectrogram)
        return {
            "gen_audio": gen_audio
        }

    def disrciminator_step(self, gen_audio, real_audio):
        real_mpd_score, real_mpd_features = self.MPD(real_audio)
        gen_mpd_score, gen_mpd_features = self.MPD(gen_audio)

        real_msd_score, real_msd_features = self.MSD(real_audio)
        gen_msd_score, gen_msd_features = self.MSD(gen_audio)

        return {
            "real_mpd_score": real_mpd_score,
            "real_mpd_features": real_mpd_features,
            "gen_mpd_score": gen_mpd_score,
            "gen_mpd_features": gen_mpd_features,
            "real_msd_score": real_msd_score,
            "real_msd_features": real_msd_features,
            "gen_msd_score": gen_msd_score,
            "gen_msd_features": gen_msd_features,
        }

    def forward(self, spectrogram):
        gen_audio = self.generator(spectrogram)
        return {
            "gen_audio": gen_audio
        }
