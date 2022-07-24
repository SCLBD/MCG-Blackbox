import numpy as np
import torch
import cma
from .base_attack import BaseAttack


class CGattack(BaseAttack):

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

    @staticmethod
    def generator_loss_interface(generator, image_loss_function):

        def loss_function(x, label, targeted):
            with torch.no_grad():
                logits = model(x)
                one_hot = torch.zeros([x.shape[0], class_num], dtype=torch.uint8).cuda()
                label_tensor = torch.tensor(label).reshape(-1, 1).cuda()
                one_hot.scatter_(1, label_tensor, 1)
                one_hot = one_hot.bool()

                if not targeted:
                    diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0]
                else:
                    diff = torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0] - logits[one_hot]
                margin = torch.nn.functional.relu(diff + threshold, True) - threshold

            output = {
                'margin': margin,
                'logits': torch.nn.functional.softmax(logits.clone().detach(), dim=1)
            }
            if targeted:
                logits = torch.nn.functional.softmax(logits, dim=1)
                cross_entropy_loss = torch.sum(-torch.log(logits[:, label]))
                output['loss'] = cross_entropy_loss
            else:
                output['loss'] = margin
            return output

        return loss_function

    def attack(self, loss, x, y, init=None, buffer=None, **kwargs):
        latent = kwargs['latent']
        # here init_delta is latent
        query_cnt = 0
        # imagenet target set 1e5; others try 1e15
        es = cma.CMAEvolutionStrategy(latent.cpu().data.numpy().reshape(-1), 1e15,
                                      {'seed': 1919, 'maxfevals': self.max_query, 'popsize': self.popsize, 'ftarget': 0,
                                       'tolfun': 1e-10})
        # es = cma.CMAEvolutionStrategy(latent.cpu().data.numpy().reshape(-1), 1e15, {'seed': 666, 'maxfevals': self.max_query, 'popsize': self.popsize, 'ftarget': 0, 'tolfun': 1e-10})

        while not es.stop() and es.best.f > 0:
            X = es.ask()  # get list of new solutions
            X = np.array(X)
            fit, perturbed_imgs, target_adv_logits = compound(X, return_logits=True)
            # For hard attack
            fit = [i + 3. for i in fit]

            es.tell(X, fit)  # feed values
            # print(min(fit))
            if query_cnt == 1 and attack_list_buffer is not None:
                attack_list_buffer.add(perturbed_imgs, target_adv_logits)
            query_cnt += self.popsize
        # if not success:
        success = (list(es.stop().keys())[0] == 'ftarget')
        best = es.best.get()
        latent, loss = best[0], best[1]
        return success, query_cnt, latent, loss

    def attack(self, loss, x, y, init=, flow_model, loss_function, loss_shape, latent_operation, attack_list_buffer=None):
        # here init_delta is latent

        # real_x = torch.nn.functional.interpolate(x, size=loss_shape)
        # print(real_x.shape)
        out_dict = loss_function(x, y, self.targeted)
        margin = out_dict['margin']
        query_cnt = 1
        success = False
        x_best, logits_best = None, None
        if margin <= 0:
            success = True
            return success, query_cnt, x_best, logits_best
        # latent = x.data.numpy().reshape(-1)
        # assert latent.shape[0] < 5000
        images = x.repeat(self.popsize, 1, 1, 1)
        # print('init latent', init_latent.shape)
        init_latent = init_latent.view(-1)
        es = cma.CMAEvolutionStrategy(init_latent, 1e15, {'seed': 666, 'maxfevals': self.max_query, 'popsize': self.popsize, 'ftarget': 0, 'tolfun': 1e-10})

        while not es.stop() and es.best.f > 0:
            X = es.ask()  # get list of new solutions
            X = np.array(X)

            # margin_list, logits_list, loss_list = generator_loss_function(torch.FloatTensor(X), y, self.targeted)

            latent = torch.FloatTensor(X).cuda()
            latent, latent_vec = latent_operation(latent, reverse=True)
            with torch.no_grad():
                perturbation, _ = flow_model.flow.decode(images, latent, zs=latent_vec)
            adv_images = torch.clamp(images + torch.sign(perturbation) * self.linf_limit, 0, 1)

            adv_images = torch.nn.functional.interpolate(adv_images, size=(224, 224))

            out_dict = loss_function(adv_images, y, self.targeted)
            margin, logits, loss, log = out_dict['margin'], out_dict['logit'], out_dict['loss'], out_dict['log']

            es.tell(X, loss.cpu().numpy())  # feed values
            query_cnt += self.popsize
            print(f'Query: {query_cnt}, loss: {min(loss)}')
            if torch.min(margin) <= 0:
                success = True
                break
        return success, query_cnt, x_best, logits_best

    def margin_latent_attack(self, compound, latent, attack_list_buffer=None):
        # here init_delta is latent
        query_cnt = 0
        # imagenet target set 1e5; others try 1e15
        es = cma.CMAEvolutionStrategy(latent.cpu().data.numpy().reshape(-1), 1e15, {'seed': 666, 'maxfevals': self.max_query, 'popsize': self.popsize, 'ftarget': 0, 'tolfun': 1e-10})
        # es = cma.CMAEvolutionStrategy(latent.cpu().data.numpy().reshape(-1), 1e15, {'seed': 666, 'maxfevals': self.max_query, 'popsize': self.popsize, 'ftarget': 0, 'tolfun': 1e-10})

        while not es.stop() and es.best.f > 0:
            X = es.ask()  # get list of new solutions
            X = np.array(X)
            fit, perturbed_imgs, target_adv_logits = compound(X, return_logits=True)
            # For hard attack
            fit = [i + 3. for i in fit]

            es.tell(X, fit)  # feed values
            # print(min(fit))
            if query_cnt == 1 and attack_list_buffer is not None:
                attack_list_buffer.add(perturbed_imgs, target_adv_logits)
            query_cnt += self.popsize
        # if not success:
        success = (list(es.stop().keys())[0] == 'ftarget')
        best = es.best.get()
        latent, loss = best[0], best[1]
        return success, query_cnt, latent, loss

    def latent_attack(self, compound, latent, attack_list_buffer=None):
        # here init_delta is latent
        query_cnt = 0
        success = False
        es = cma.CMAEvolutionStrategy(latent.cpu().data.numpy().reshape(-1), 1e15, {'seed': 666, 'maxfevals': self.max_query, 'popsize': self.popsize, 'ftarget': -1, 'tolfun': 1e-10})

        while not es.stop() and es.best.f > 0:
            X = es.ask()  # get list of new solutions
            X = np.array(X)
            margin, perturbed_imgs, target_adv_logits = compound(X, return_logits=True)
            es.tell(X, margin)  # feed values

            if query_cnt == 1 and attack_list_buffer is not None:
                attack_list_buffer.add(perturbed_imgs, target_adv_logits)
            query_cnt += self.popsize

            # print(min(margin))
            # print(min(label_loss))
            if min(margin) < 0:
                success = True
                break
        # if not success:
        # success = (list(es.stop().keys())[0] == 'ftarget')
        return success, query_cnt

    def criterion(self, label, logits):
        threshold = 5.
        logits_other = torch.cat((logits[:, :label], logits[:, (label + 1):]), dim=1)
        if not self.targeted:
            diff = logits[:, label].view(-1) - torch.max(logits_other, dim=1)[0]
        else:
            diff = torch.max(logits_other, dim=1)[0] - logits[:, label].view(-1)
        # print(diff.shape)
        # print(logits[:, label].shape)
        # print(torch.max(logits_other, dim=1)[0].shape)
        margin = torch.nn.functional.relu(diff + threshold, True) - threshold

        return margin.cpu().data.numpy().reshape(-1).tolist()

