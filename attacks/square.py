import numpy as np
import torch
from .base_attack import BaseAttack


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


class SquareAttack(BaseAttack):
    def __init__(
            self,
            dataset_name,
            max_query,
            targeted,
            class_num,
            linf=0.05,
            init_epochs=500,
            p_init=0.3
    ):
        super().__init__(dataset_name, max_query, targeted, class_num, linf)
        self.init_epochs = init_epochs
        self.p_init = p_init
        self.n_features = self.c * self.h * self.w

    def attack(self, loss_func, x, y, init=None, buffer=None, **kwargs):
        if init is None:
            # [c, 1, w], i.e. vertical stripes designed in Square
            init = np.random.choice([-self.linf, self.linf], size=[x.shape[0], 3, 1, x.shape[-1]])

        x_best = torch.clamp(x + init, 0., 1.)
        output = loss_func(x_best, y, self.targeted)
        margin_min, logits_best, loss_min = output['margin'], output['logits'], output['loss']
        query_cnt = 0
        success = False

        for i_iter in range(self.max_query - 1):
            perturbation = x_best - x
            p = p_selection(self.p_init, i_iter + self.init_epochs, self.max_query)
            s = int(round(np.sqrt(p * self.n_features / self.c)))
            s = min(max(s, 1), self.h - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, self.h - s)
            center_w = np.random.randint(0, self.w - s)

            x_curr_window = x[:, :, center_h:center_h + s, center_w:center_w + s]
            x_best_curr_window = x_best[:, :, center_h:center_h + s, center_w:center_w + s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while torch.sum(torch.abs(torch.clamp(x_curr_window + perturbation[:, :, center_h:center_h + s, center_w:center_w + s], 0., 1.) - x_best_curr_window) < 10 ** -7) == self.c * s * s:
                perturbation[:, :, center_h:center_h + s, center_w:center_w + s] = torch.tensor(
                    np.random.choice([-self.linf, self.linf], size=[self.c, 1, 1])).cuda()

            x_new = torch.clamp(x + perturbation, 0., 1.)
            output = loss_func(x_new, y, self.targeted)
            margin, logits, loss = output['margin'], output['logits'], output['loss']

            if buffer is not None and query_cnt < buffer.uplimit:
                buffer.add(x_new, logits)

            if loss < loss_min:
                x_best = x_new
                margin_min = margin
                logits_best = logits
                loss_min = loss

            query_cnt += 1
            if margin_min <= 0:
                success = True
                break

        return {
            'success': success,
            'query_cnt': query_cnt,
            'adv': x_best,
            'logits_best': logits_best
        }
