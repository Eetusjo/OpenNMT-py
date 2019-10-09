import fasttext
import torch

from onmt.utils.logging import logger


class MultiFastTextEmbedder():
    def __init__(self, models):
        self.embedders = {lang: model for (lang, model) in models}

    def get_ft_emb_and_mask(self, src, raw, vec_dim, device):
        mask = src.clone().to(torch.float).to(device)
        mask[mask != 0] = 1

        embs = self._embed_examples(
            raw, mask,
            emb_shape=(src.size(0), src.size(1), vec_dim),
            device=device
        )

        return embs, mask

    def _embed_ft(self, word, lang):
        return torch.from_numpy(self.embedders[lang][word])

    def _embed_examples(self, raw, mask, emb_shape, device):
        embeddings = torch.zeros(emb_shape)
        for i, example in enumerate(raw):
            mask_vec = mask[:, i].view(-1)
            for j, ind in enumerate(mask_vec):
                if ind == 0:
                    vec = self._embed_ft(example.src[0][j], example.srclang)
                    embeddings[j, i] = vec

        return embeddings.to(device)


def build_ft_embedder(opt):
    logger.info("Using dynamic fastText embeddings for: '{}''".format(
        opt.fasttext_langs))

    models = []
    for lang, model_path in zip(opt.fasttext_langs, opt.fasttext):
        models.append((lang, fasttext.load_model(model_path)))

    return MultiFastTextEmbedder(models)
