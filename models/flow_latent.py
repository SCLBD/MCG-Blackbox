import torch
import numpy as np


def latent_operate(latent_base, latent_vec=None, reverse=False):
    """
    Black-optimization optimize both latent and latent_vec
    """
    global ls, sizes

    if not reverse:
        ls, sizes = [latent_base.numel()],  [[-1] + list(latent_base.shape[1:])]
        ls.extend([l.numel() for l in latent_vec])
        sizes.extend([-1] + list(l.shape[1:]) for l in latent_vec)

        latent = [latent_base.reshape(1, -1)]
        latent.extend(l.reshape(1, -1) for l in latent_vec)
        return torch.cat(latent, dim=1), None
    else:
        latent_base_r = latent_base[:, :ls[0]].reshape(sizes[0])
        latent_vec_r = []
        # sum_l = ls[0] TODO could try
        sum_l = 0
        for idx in range(len(sizes)):
            l, s = ls[idx], sizes[idx]
            # print('latent op full inside dimension', idx, sum_l, l, s)
            latent_vec_r.append(latent_base[:, sum_l:sum_l + l].reshape(s))
            sum_l = sum_l + l
        return latent_base_r, latent_vec_r


def latent_initialize(image, flow_model, latent_operation):
    """
    Note:
        Normal latent_base shape (1, 48, 4, 4)
        Normal latent_vec shape [(1, 48, 4, 4), (1, 6, 16, 16), (1, 12, 8, 8)]
        Corresponding to size 768, 2304, 3072
    """
    assert len(image.shape) == 4
    init_pert, decode_logdet = flow_model.decode(image, return_prob=True, no_norm=True)

    latent, encode_logdet, latent_vec = flow_model.flow.encode(image, init_pert, return_z=True)
    latent_base = latent.clone()
    latent_vec = [lat.detach() for lat in latent_vec]

    return latent_operation(latent_base, latent_vec)


def generate_interface(G, latent_op, linf):

    def generate(images, latent):
        with torch.no_grad():
            latent, latent_vec = latent_op(latent, reverse=True)
            perturbation, _ = G.flow.decode(images, latent, zs=latent_vec)
            perturbation = torch.clamp(perturbation, min=-linf, max=linf)
        return perturbation

    return generate
