import torch
import random
import numpy as np


def eliminate_error(x):
    # Not Recommend
    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    x = torch.where(torch.isinf(x), torch.zeros_like(x), x)
    return x


class TrainModelSurrogate:
    # HOGA: Class Method to train surrogate model
    def __init__(self):
        self.train_num = 0
        self.batch_size = 8
        self.lamda = 3.0
        self.d_loss_sum = 0
        self.s_loss_sum = 0
        self.forward_loss_lamda = 0.05  # 0.0005  # 0.005 will be fine
        self.backward_loss_lamda = 0.01  # 0.0005  # 0.0005 will be fine
        self.backward_loss_grad_coefficient = 0.01
        self.max_grad_clip = 0.1  # 0.0001 is fine

    def get_lamda(self):
        # Adaptive gamma in paper, Here is to get adaptive lamda
        if self.train_num > 50:
            lamda2 = self.s_loss_sum / self.d_loss_sum  # Use history s_loss sum and d_loss sum, compute lamda2
            self.lamda = self.lamda * 0.9 + lamda2 * 0.1  # Update lamda with lamda2 using momentum
        else:
            self.lamda = 3.0

    def coefficient_forward_loss(self, surrogate_models, coefficient, coefficient_optim, batch):
        images, target_logits, labels = batch[0], batch[1], batch[2]
        images = images.detach().clone()
        images.requires_grad = True
        if isinstance(labels, int):
            labels = torch.tensor(labels).cuda()
            labels = labels.view(-1).repeat(images.shape[0])
        s_score = None
        for i, model in enumerate(surrogate_models):
            with torch.no_grad():
                logits = model(images)
                prob = torch.nn.functional.softmax(logits, dim=1)
            if s_score is None:
                s_score = prob.gather(1, labels.reshape(-1, 1)) * coefficient[i]
            else:
                s_score += prob.gather(1, labels.reshape(-1, 1)) * coefficient[i]

        target_score = target_logits.gather(1, labels.reshape(-1, 1))
        # target_score = target_prob.topk(2, dim=1)[0]
        # print('s_score: ', s_score)
        # print('target_score: ', target_score)
        forward_loss = torch.nn.MSELoss()(s_score, target_score.detach())  # * self.forward_loss_lamda
        del images

        coefficient_optim.zero_grad()
        forward_loss.backward()
        coefficient_optim.step()
        print('Coefficient:', coefficient)

    # def coefficient_forward_loss(self, surrogate_models, coefficient, coefficient_optim, history_batch, current_batch):
    #     loss_list = []
    #     for batch in [history_batch, current_batch]:
    #         images, target_logits, labels = batch[0], batch[1], batch[2]
    #         images = images.detach().clone()
    #         images.requires_grad = True
    #         if isinstance(labels, int):
    #             labels = torch.tensor(labels).cuda()
    #             labels = labels.view(-1).repeat(images.shape[0])
    #         s_score = None
    #         for i, model in enumerate(surrogate_models):
    #             with torch.no_grad():
    #                 logits = model(images)
    #                 prob = torch.nn.functional.softmax(logits, dim=1)
    #             if s_score is None:
    #                 s_score = prob.gather(1, labels.reshape(-1, 1)) * coefficient[i]
    #             else:
    #                 s_score += prob.gather(1, labels.reshape(-1, 1)) * coefficient[i]
    #
    #         target_prob = torch.nn.functional.softmax(target_logits, dim=1)
    #         target_score = target_prob.gather(1, labels.reshape(-1, 1))
    #         # target_score = target_prob.topk(2, dim=1)[0]
    #         print('s_score: ', s_score)
    #         print('target_score: ', target_score)
    #         forward_loss = torch.nn.MSELoss()(s_score, target_score.detach())  # * self.forward_loss_lamda
    #         # print('Forward loss fine-tune model.', forward_loss)
    #         loss_list.append(forward_loss)
    #         del images
    #     # if abs(loss_list[1]) > 1e-8:
    #     #     alpha = loss_list[0] / loss_list[1] / 2
    #     #     beta = 1
    #     # else:
    #     #     alpha = 1
    #     #     beta = 0
    #     # print('alpha ', alpha, ' beta ', beta)
    #     loss = loss_list[0] + loss_list[1] * 0.2
    #     # print(loss)
    #     # print(coefficient_optim.state_dict())
    #     coefficient_optim.zero_grad()
    #     loss.backward()
    #     coefficient_optim.step()
    #     print('Coefficient:', coefficient)

    def lifelong_forward_loss(self, surrogate_model, optimizer, history_batch, current_batch):
        # print('lifelong finetune surrogate models')
        # batch: [images, logits, labels]
        # surrogate_model.eval()
        loss_list = []
        for batch in [history_batch, current_batch]:
            images, target_logits, labels = batch[0], batch[1], batch[2]
            images = images.detach().clone()
            images.requires_grad = True
            if isinstance(labels, int):
                labels = torch.tensor(labels).cuda()
                labels = labels.view(-1).repeat(images.shape[0])

            surrogate_logits = surrogate_model(images)
            surrogate_prob = torch.nn.functional.softmax(surrogate_logits, dim=1)
            # since attack image is the correct one, other max logit must be the top 2
            s_score = surrogate_prob.gather(1, labels.reshape(-1, 1))
            # s_score = surrogate_prob.topk(2, dim=1)[0]

            # target_prob = torch.nn.functional.softmax(target_logits, dim=1)
            target_score = target_logits.gather(1, labels.reshape(-1, 1))
            # target_score = target_prob.topk(2, dim=1)[0]

            # print('s_score: ', s_score)
            # print('target_score: ', target_score)

            forward_loss = torch.nn.MSELoss()(s_score, target_score.detach()) * self.forward_loss_lamda
            # print('Forward loss fine-tune model.', forward_loss)
            loss_list.append(forward_loss)
            del images
        # if abs(loss_list[1]) > 1e-8:
        #     alpha = loss_list[0] / loss_list[1] / 2
        #     beta = 1
        # else:
        #     alpha = 1
        #     beta = 0
        # print('loss: ', loss_list[0], ' loss: ', loss_list[1])
        # print('alpha ', alpha, ' beta ', beta)
        loss = loss_list[0] * 2 + loss_list[1]
        surrogate_model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), self.max_grad_clip)
        optimizer.step()

    def forward_loss_max(self, surrogate_model, optimizer, images, target_logits):
        images = images.detach().clone()
        images.requires_grad = True

        surrogate_logits = surrogate_model(images)
        surrogate_prob = torch.nn.functional.softmax(surrogate_logits, dim=1)
        # since attack image is the correct one, other max logit must be the top 2
        s_score = torch.max(surrogate_prob, dim=1)[0]

        target_score = torch.max(target_logits, dim=1)[0]

        forward_loss = torch.nn.MSELoss()(s_score, target_score.detach()) * self.forward_loss_lamda
        print('forward_loss_max: ', forward_loss)
        # print('Forward loss fine-tune model.', forward_loss)
        surrogate_model.zero_grad()
        forward_loss.backward()
        torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), self.max_grad_clip)
        optimizer.step()
        del images

    def forward_loss(self, surrogate_model, optimizer, images, target_logits, labels):
        images = images.detach().clone()
        images.requires_grad = True
        if isinstance(labels, int):
            labels = torch.tensor(labels).cuda()
            labels = labels.view(-1).repeat(images.shape[0])

        surrogate_logits = surrogate_model(images)
        surrogate_prob = torch.nn.functional.softmax(surrogate_logits, dim=1)
        # since attack image is the correct one, other max logit must be the top 2
        s_score = surrogate_prob.gather(1, labels.reshape(-1, 1))
        # s_score = surrogate_prob.topk(2, dim=1)[0]

        # target_prob = torch.nn.functional.softmax(target_logits, dim=1)
        # target_prob = target_logits
        target_score = target_logits.gather(1, labels.reshape(-1, 1))
        # target_score = target_prob.topk(2, dim=1)[0]
        # print('s_score: ', s_score)
        # print('target_score: ', target_score)
        # assert False
        forward_loss = torch.nn.MSELoss()(s_score, target_score.detach()) * self.forward_loss_lamda
        # print('forward_loss: ', forward_loss)
        # print('Forward loss fine-tune model.', forward_loss)
        surrogate_model.zero_grad()
        forward_loss.backward()
        torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), self.max_grad_clip)
        optimizer.step()
        del images
        # return s_score, target_score

    def forward_loss_only(self, target_model, surrogate_model, images, labels, adv_images, optimizer):
        # assert len(images.shape) == 4 and images.shape[0] > 1
        # images = (images + noise).detach().clone()
        images = images.detach().clone()
        images.requires_grad = True

        surrogate_logits = surrogate_model(images)
        surrogate_prob = torch.nn.functional.softmax(surrogate_logits, dim=1)
        # since attack image is the correct one, other max logit must be the top 2
        s_score = surrogate_prob.gather(1, labels.reshape(-1, 1))
        # s_score = surrogate_prob.topk(2, dim=1)[0]

        target_logits = target_model(images)
        target_prob = torch.nn.functional.softmax(target_logits, dim=1)
        target_score = target_prob.gather(1, labels.reshape(-1, 1))
        # target_score = target_prob.topk(2, dim=1)[0]
        # print('s_score: ', s_score)
        # print('target_score: ', target_score)
        forward_loss = torch.nn.MSELoss()(s_score, target_score.detach()) * self.forward_loss_lamda
        # print('forward_loss: ', forward_loss)

        surrogate_model.zero_grad()
        forward_loss.backward()
        torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), self.max_grad_clip)
        optimizer.step()
        del images

    def backward_loss_only(self, target_model, surrogate_model, images, labels, adv_images, optimizer):
        self.get_lamda()
        lamda = self.lamda  # torch.tensor([self.lamda]).cuda()
        self.train_num += images.shape[0]
        labels = labels.view(-1)
        images = images.detach().clone()
        images.requires_grad = True
        # TODO, keep the noise into a linf norm
        noise = adv_images - images

        surrogate_logits = surrogate_model(images)
        # Note that using cross entropy loss to train surrogate model here
        surrogate_loss = torch.nn.CrossEntropyLoss(reduction='none')(surrogate_logits, labels)
        # Create High Order Compute Graph: grad d(loss)/d(images)
        grad = torch.autograd.grad(surrogate_loss.sum(), images, create_graph=True)[0]
        # noise * s_grad: surrogate model loss with noise.
        s_loss = (noise.detach() * grad * self.backward_loss_grad_coefficient).view([images.shape[0], -1]).sum(
            dim=1)  # scalar

        with torch.no_grad():
            target_adv_logits = target_model(adv_images)
            target_adv_loss = torch.nn.CrossEntropyLoss(reduction='none')(target_adv_logits, labels)

            target_ori_logits = target_model(images)
            target_ori_loss = torch.nn.CrossEntropyLoss(reduction='none')(target_ori_logits, labels)
            # d_loss = torch.log(target_adv_loss) - torch.log(target_ori_loss)  # scalar
            d_loss = target_adv_loss - target_ori_loss  # scalar
        # print('s_loss: ', s_loss)
        # print('d_loss: ', d_loss)
        backward_loss = torch.nn.MSELoss()(s_loss / lamda, d_loss.detach()) * self.backward_loss_lamda
        print('backward loss:', backward_loss)
        surrogate_model.zero_grad()
        backward_loss.backward()
        torch.nn.utils.clip_grad_value_(surrogate_model.parameters(), self.max_grad_clip)
        optimizer.step()

        del images
        del adv_images
        self.d_loss_sum += torch.sum(d_loss).item()
        self.s_loss_sum += torch.sum(s_loss).item()

    def leba_loss(self, surrogate_model, optimizer, current_images, last_images, current_logits, last_logits, labels):
        print('Leba loss fine-tune surrogate model.')
        self.get_lamda()
        lamda = self.lamda  # torch.tensor([self.lamda]).cuda()
        self.train_num += current_images.shape[0]
        last_images = last_images.detach().clone()
        labels = labels.view(-1)
        last_images.requires_grad = True
        noise = current_images - last_images

        surrogate_logits = surrogate_model(last_images)
        surrogate_prob = torch.nn.functional.softmax(surrogate_logits, dim=1)
        s_score = surrogate_prob.topk(2, dim=1)[0]  # s_score = surrogate_prob.topk(2, dim=1)[0]
        # Note that using cross entropy loss to train surrogate model here
        # print(surrogate_logits.shape, labels.shape)
        surrogate_loss = torch.nn.CrossEntropyLoss(reduction='none')(surrogate_logits, labels)
        # Create High Order Compute Graph: grad d(loss)/d(images)
        grad = torch.autograd.grad(surrogate_loss.sum(), last_images, create_graph=True)[0]
        # noise * s_grad: surrogate model loss with noise.
        s_loss = (noise.detach() * grad * self.backward_loss_grad_coefficient).view([last_images.shape[0], -1]).sum(
            dim=1)  # scalar

        with torch.no_grad():
            target_current_loss = torch.nn.CrossEntropyLoss(reduction='none')(current_logits, labels)

            target_last_loss = torch.nn.CrossEntropyLoss(reduction='none')(last_logits, labels)
            # d_loss = torch.log(target_adv_loss) - torch.log(target_ori_loss)  # scalar
            d_loss = target_current_loss - target_last_loss  # scalar
            target_prob = torch.nn.functional.softmax(last_logits, dim=1)
            # target_score = target_prob.gather(1, labels.reshape(-1, 1))
            target_score = target_prob.topk(2, dim=1)[0]

        # print('s_loss: ', s_loss)
        # print('d_loss: ', d_loss)
        backward_loss = torch.nn.MSELoss()(s_loss / lamda, d_loss.detach()) * self.backward_loss_lamda
        forward_loss = torch.nn.MSELoss()(s_score, target_score.detach()) * self.forward_loss_lamda
        leba_loss = forward_loss + backward_loss
        print('leba loss: ', leba_loss)
        surrogate_model.zero_grad()
        leba_loss.backward()
        torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), self.max_grad_clip)
        optimizer.step()

        self.d_loss_sum += torch.sum(d_loss).item()
        self.s_loss_sum += torch.sum(s_loss).item()

    def leba_loss_(self, target_model, surrogate_model, images, labels, adv_images, optimizer):
        # Call HOGA, train surrogate_model
        # TODO, note that the images has to a batch
        """
            Args:
            noise: Current query perturbation
            query_score: Current query score with (images+noise)
            query_loss: Current query loss with (images+noise)
            last_loss: History query loss with (images)
            surrogate_model: surrogate model
            optimizer: optimizer for surrogate_model
        """
        print('Leba loss fine-tune surrogate model.')
        self.get_lamda()
        lamda = self.lamda  # torch.tensor([self.lamda]).cuda()
        self.train_num += images.shape[0]
        labels = labels.view(-1)
        images = images.detach().clone()
        images.requires_grad = True
        # TODO, keep the noise into a linf norm
        noise = adv_images - images

        surrogate_logits = surrogate_model(images)
        surrogate_prob = torch.nn.functional.softmax(surrogate_logits, dim=1)
        s_score = surrogate_prob.gather(1, labels.reshape(-1, 1))  # s_score = surrogate_prob.topk(2, dim=1)[0]
        # Note that using cross entropy loss to train surrogate model here
        surrogate_loss = torch.nn.CrossEntropyLoss(reduction='none')(surrogate_logits, labels)
        # Create High Order Compute Graph: grad d(loss)/d(images)
        grad = torch.autograd.grad(surrogate_loss.sum(), images, create_graph=True)[0]
        # noise * s_grad: surrogate model loss with noise.
        s_loss = (noise.detach() * grad * self.backward_loss_grad_coefficient).view([images.shape[0], -1]).sum(
            dim=1)  # scalar

        with torch.no_grad():
            target_adv_logits = target_model(adv_images)
            target_adv_loss = torch.nn.CrossEntropyLoss(reduction='none')(target_adv_logits, labels)

            target_ori_logits = target_model(images)
            target_ori_loss = torch.nn.CrossEntropyLoss(reduction='none')(target_ori_logits, labels)
            # d_loss = torch.log(target_adv_loss) - torch.log(target_ori_loss)  # scalar
            d_loss = target_adv_loss - target_ori_loss  # scalar

            target_prob = torch.nn.functional.softmax(target_ori_logits, dim=1)
            target_score = target_prob.gather(1, labels.reshape(-1, 1))
            # target_score = target_prob.topk(2, dim=1)[0]

        # print('s_loss: ', s_loss)
        # print('d_loss: ', d_loss)
        backward_loss = torch.nn.MSELoss()(s_loss / lamda, d_loss.detach()) * self.backward_loss_lamda
        forward_loss = torch.nn.MSELoss()(s_score, target_score.detach()) * self.forward_loss_lamda
        leba_loss = forward_loss + backward_loss
        print('leba loss: ', leba_loss)
        surrogate_model.zero_grad()
        leba_loss.backward()
        torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), self.max_grad_clip)
        optimizer.step()

        del images
        del adv_images
        self.d_loss_sum += torch.sum(d_loss).item()
        self.s_loss_sum += torch.sum(s_loss).item()

    def leba_loss_with_one_image_queries(self, surrogate_model, images, labels, noises, target_adv_logits, last_noises,
                                         target_last_adv_logits, optimizer):
        # Call HOGA, train surrogate_model
        # TODO, note that the images has to a batch
        """
            Args:
            noise: Current query perturbation
            query_score: Current query score with (images+noise)
            query_loss: Current query loss with (images+noise)
            last_loss: History query loss with (images)
            surrogate_model: surrogate model
            optimizer: optimizer for surrogate_model
        """
        print('Leba loss with one image queries fine-tune surrogate model.')
        self.get_lamda()
        lamda = self.lamda  # torch.tensor([self.lamda]).cuda()
        self.train_num += last_noises.shape[0]
        if isinstance(labels, int):
            labels = torch.tensor(labels).cuda()
        labels = labels.view(-1).repeat(last_noises.shape[0])

        last_noises = last_noises.detach().clone()
        last_noises.requires_grad = True
        last_images = torch.clamp(last_noises + images, 0., 1.)

        surrogate_logits = surrogate_model(last_images)
        surrogate_prob = torch.nn.functional.softmax(surrogate_logits, dim=1)
        s_score = surrogate_prob.topk(2, dim=1)[0]  # s_score = surrogate_prob.gather(1, labels.reshape(-1, 1))
        # Note that using cross entropy loss to train surrogate model here
        surrogate_loss = torch.nn.CrossEntropyLoss(reduction='none')(surrogate_logits, labels)
        # Create High Order Compute Graph: grad d(loss)/d(images)
        grad = torch.autograd.grad(surrogate_loss.sum(), last_images, create_graph=True)[0]
        # noise * s_grad: surrogate model loss with noise.
        s_loss = ((noises - last_noises).detach() * grad * self.backward_loss_grad_coefficient).view(
            [last_images.shape[0], -1]).sum(dim=1)  # scalar

        with torch.no_grad():
            target_adv_loss = torch.nn.CrossEntropyLoss(reduction='none')(target_adv_logits, labels)
            target_last_loss = torch.nn.CrossEntropyLoss(reduction='none')(target_last_adv_logits, labels)
            # d_loss = torch.log(target_adv_loss) - torch.log(target_ori_loss)  # scalar
            d_loss = target_adv_loss - target_last_loss  # scalar

            target_prob = torch.nn.functional.softmax(target_last_adv_logits, dim=1)
            # target_score = target_prob.gather(1, labels.reshape(-1, 1))
            target_score = target_prob.topk(2, dim=1)[0]
        # print('s_loss: ', s_loss)
        # print('d_loss: ', d_loss)
        backward_loss = torch.nn.MSELoss()(s_loss / lamda, d_loss.detach()) * self.backward_loss_lamda
        forward_loss = torch.nn.MSELoss()(s_score, target_score.detach()) * self.forward_loss_lamda
        leba_loss = forward_loss + backward_loss
        # leba_loss = eliminate_error(leba_loss)
        print('leba loss: ', leba_loss)
        # if leba_loss < 0.0001:
        surrogate_model.zero_grad()
        leba_loss.backward()
        torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), self.max_grad_clip)
        # torch.nn.utils.clip_grad_value_(surrogate_model.parameters(), self.max_grad_clip)
        optimizer.step()
        del last_noises
        del last_images
        self.d_loss_sum += torch.sum(d_loss).item()
        self.s_loss_sum += torch.sum(s_loss).item()
