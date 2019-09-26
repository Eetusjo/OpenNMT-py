""" Onmt NMT Model base class definition """
import fasttext
import torch
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, ft_embedder=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ft_embedder = ft_embedder

    def forward(self, src, tgt, lengths, bptt=False, raw=None, useft=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        if useft and self.ft_embedder and (raw is not None):
            ft_embeddings, ft_mask = self.get_ft_emb_and_mask(
                src, raw, device=src.device
            )
        else:
            ft_embeddings, ft_mask = None, None

        enc_state, memory_bank, lengths = self.encoder(
            src, lengths, ft_emb=ft_embeddings, ft_mask=ft_mask
        )

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def get_ft_emb_and_mask(self, src, raw, device):
        mask = src.clone().to(torch.float).to(device)
        mask[mask != 0] = 1

        vec_dim = self.encoder.embeddings.__dict__["word_vec_size"]
        embs = self._embed_examples(
            raw, mask,
            emb_shape=(src.size(0), src.size(1), vec_dim),
            device=device
        )

        return embs, mask

    def _embed_ft(self, word, vec_dim):
        # print(word, self.ft_embedder[word])
        # return torch.rand(1, 1, vec_dim)
        return torch.from_numpy(self.ft_embedder[word])

    def _embed_examples(self, raw, mask, emb_shape, device):
        embeddings = torch.zeros(emb_shape)
        for i, example in enumerate(raw):
            mask_vec = mask[:, i].view(-1)
            for j, ind in enumerate(mask_vec):
                if ind == 0:
                    vec = self._embed_ft(example.src[0][j], emb_shape[2])
                    # print(vec.shape)
                    embeddings[j, i] = vec

            # print(mask_vec.squeeze(), example.src[0])
        return embeddings.to(device)
