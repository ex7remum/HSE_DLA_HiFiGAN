import torch
import torch.nn as nn
from hw_tts.model.FFT import FFTBlock


def get_non_pad_mask(seq, pad):
    assert seq.dim() == 2
    return seq.ne(pad).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q, pad):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    def __init__(self, encoder_dim,
                 encoder_n_head,
                 encoder_n_layer,
                 encoder_conv1d_filter_size,
                 max_seq_len,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 vocab_size,
                 dropout=0.1,
                 *args, **kwargs):

        super(Encoder, self).__init__()

        len_max_seq = max_seq_len
        n_position = len_max_seq + 1
        n_layers = encoder_n_layer

        self.src_word_emb = nn.Embedding(
            vocab_size,
            encoder_dim,
            padding_idx=0
        )

        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=0
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            encoder_dim,
            encoder_conv1d_filter_size,
            encoder_n_head,
            encoder_dim // encoder_n_head,
            encoder_dim // encoder_n_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad=0)
        non_pad_mask = get_non_pad_mask(src_seq, pad=0)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    def __init__(self, decoder_dim,
                 decoder_n_head,
                 decoder_n_layer,
                 decoder_conv1d_filter_size,
                 max_seq_len,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout=0.1,
                 *args, **kwargs):

        super(Decoder, self).__init__()

        len_max_seq = max_seq_len
        n_position = len_max_seq + 1
        n_layers = decoder_n_layer

        self.position_enc = nn.Embedding(
            n_position,
            decoder_dim,
            padding_idx=0,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            decoder_dim,
            decoder_conv1d_filter_size,
            decoder_n_head,
            decoder_dim // decoder_n_head,
            decoder_dim // decoder_n_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, pad=0)
        non_pad_mask = get_non_pad_mask(enc_pos, pad=0)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
