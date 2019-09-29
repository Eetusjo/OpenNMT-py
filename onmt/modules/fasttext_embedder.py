import torch


def get_ft_emb_and_mask(src, raw, vec_dim, embedder, device):
    mask = src.clone().to(torch.float).to(device)
    mask[mask != 0] = 1

    embs = embed_examples(
        raw, mask,
        emb_shape=(src.size(0), src.size(1), vec_dim),
        embedder=embedder,
        device=device
    )

    return embs, mask


def embed_ft(word, ft_embedder):
    return torch.from_numpy(ft_embedder[word])


def embed_examples(raw, mask, emb_shape, embedder, device):
    embeddings = torch.zeros(emb_shape)
    for i, example in enumerate(raw):
        mask_vec = mask[:, i].view(-1)
        for j, ind in enumerate(mask_vec):
            if ind == 0:
                vec = embed_ft(example.src[0][j], embedder)
                embeddings[j, i] = vec

    return embeddings.to(device)
