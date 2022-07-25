import numpy as np
import torch
import cma
from .base_attack import BaseAttack


class CGAttack(BaseAttack):

    def __init__(
            self,
            dataset_name,
            max_query,
            targeted,
            class_num,
            linf=0.05,
            popsize=20,
    ):
        super().__init__(dataset_name, max_query, targeted, class_num, linf)
        self.popsize = popsize
        self.STD = 1e5 if self.dataset == 'imagenet' else 1e15
        self.cma_configs = {
            'seed': 1919,
            'maxfevals': self.max_query,
            'popsize': self.popsize,
            'ftarget': 0,
            'tolfun': 1e-10
        }

    @staticmethod
    def generator_loss_interface(generate_function, image_loss_function, targeted):

        def loss_function(images, latent, labels):
            latent = torch.FloatTensor(latent).cuda()
            perturbation = generate_function(images, latent)
            adv_images = torch.clamp(images + perturbation.view(images.shape), 0., 1.)
            loss_output = image_loss_function(adv_images, labels, targeted=targeted)
            loss_output['adv'] = adv_images
            return loss_output

        return loss_function

    def attack(self, loss_func, x, y, init=None, buffer=None, **kwargs):
        assert 'latent' in kwargs, 'Wrong kwargs without latent'
        latent = kwargs['latent']
        query_cnt = 0
        es = cma.CMAEvolutionStrategy(latent.cpu().data.numpy().reshape(-1), self.STD, self.cma_configs)
        adv, logits = None, None
        x = x.repeat(self.popsize, 1, 1, 1)

        while not es.stop() and es.best.f > 0:
            sampled_latent = es.ask()  # get list of new solutions
            sampled_latent = np.array(sampled_latent)
            loss_output = loss_func(x, sampled_latent, y)
            score = loss_output['loss']
            # For hard attack
            score = [i.item() + 3. for i in score]
            es.tell(sampled_latent, score)  # feed values

            adv = loss_output['adv'][0].unsqueeze(0)
            logits = loss_output['logits'][0].unsqueeze(0)
            if buffer is not None and query_cnt < buffer.uplimit:
                buffer.add(adv, logits)

            query_cnt += self.popsize
        # if not success:
        success = (list(es.stop().keys())[0] == 'ftarget')
        best = es.best.get()
        latent, loss = best[0], best[1]

        return {
            'success': success,
            'query_cnt': query_cnt,
            'latent': latent,
            'adv': adv,
            'logits_best': logits
        }
