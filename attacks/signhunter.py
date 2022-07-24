import numpy as np
import torch
from .base_attack import BaseAttack


class SignHunter(BaseAttack):
    def __init__(
            self,
            dataset_name,
            max_query,
            targeted,
            class_num,
            linf=0.05,
    ):
        super().__init__(dataset_name, max_query, targeted, class_num, linf)
        self.n_features = self.c * self.h * self.w

    def attack(self, loss, x, y, init=None, buffer=None, **kwargs):
        if init is None:
            init = torch.ones(self.n_features).to(x.device)
        perturbation = torch.sign(init).reshape(-1) * self.linf
        x_best = torch.clamp(x + perturbation.view(x.shape), 0., 1.)

        output = loss(x_best, y, self.targeted)
        margin_min, logits_best, loss_min = output['margin'], output['logits'], output['loss']
        query_cnt = 0
        success = False
        if buffer is not None:
            buffer.new()

        while query_cnt < self.max_query:
            for h in range(np.ceil(np.log2(self.n_features)).astype(int) + 1):
                chunk_len = np.ceil(self.n_features / 2 ** h).astype(int)
                for i in range(2 ** h):
                    istart = i * chunk_len
                    iend = min(istart + chunk_len, self.n_features)

                    perturbation[istart:iend] *= -1
                    x_new = torch.clamp(x + perturbation.view(x.shape), 0., 1.)

                    output = loss(x_new, y, self.targeted)
                    margin, logits, loss = output['margin'], output['logits'], out_dict['loss']
                    query_cnt += 1
                    query_cnt += 1

                    if loss <= loss_min:
                        x_best = x_new
                        margin_min = margin
                        logits_best = logits
                        loss_min = loss
                    else:
                        perturbation[istart:iend] *= -1

                    if buffer is not None and query_cnt < buffer.uplimit:
                        buffer.add(x_new, logits)

                    if query_cnt >= self.max_query or istart == self.n_features - 1:
                        break
                    if margin_min <= 0:
                        success = True
                        break
        return {
            'success': success,
            'query_cnt': query_cnt,
            'adv': x_best,
            'logits_best': logits_best
        }

