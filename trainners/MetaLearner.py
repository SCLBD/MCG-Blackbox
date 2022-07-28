import copy
import os
import torch
import torch.nn as nn
import datetime
import numpy as np
from torch.utils.data import DataLoader
import time

from utils.tools import preprocess, save_model
from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval


class Trainer(object):

    def __init__(self, flow, adv_models, query_set_models, optim, train_set, valid_set, args, cuda):

        # set path and date
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "") \
            .replace(":", "") \
            .replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, args.name if args.name != '' else date)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        args_str = ''
        for key, value in sorted(vars(args).items()):
            args_str += f'{key}: {value}\n'
        with open(os.path.join(self.log_dir, 'args.txt'), 'a') as file_object:
            file_object.write(f'Name: {args.name}\n')
            file_object.write(args_str)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        # model
        self.flow = flow
        self.flow.eval()
        self.adv_models = adv_models
        self.query_set_models = query_set_models
        self.adv_model_id = 0

        self.optim = optim

        # gradient bound
        self.max_grad_clip = args.max_grad_clip
        self.max_grad_norm = args.max_grad_norm

        # data
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.trainingset_loader = DataLoader(train_set,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             drop_last=True)
        self.validset_loader = DataLoader(valid_set,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          drop_last=False)

        self.num_epochs = args.num_epochs
        self.global_step = args.num_steps
        self.label_scale = args.label_scale
        self.label_bias = args.label_bias
        self.x_bins = args.x_bins
        self.y_bins = args.y_bins
        self.margin = args.margin

        self.num_epochs = args.num_epochs
        self.log_gap = args.log_gap
        self.inference_gap = args.inference_gap
        self.test_gap = args.test_gap
        self.save_gap = args.save_gap
        self.target = args.target
        self.target_label = args.target_label
        if self.target:
            self.target_label = torch.tensor(self.target_label).cuda().unsqueeze(0).expand(self.batch_size, -1)
            print('Target Attack Training: target label: ', args.target_label)
        self.openset = args.openset
        self.openset_top5_labels_list = [41, 394, 497, 776, 911]
        self.args = args

        # device
        self.cuda = cuda
        self.label_num = 1000

        # meta training:
        self.meta_iteration = args.meta_iteration
        self.meta_test_batch = 1
        self.temp_meta_path = args.name + '.pth'

        # adversary
        self.linf = 8. / 255
        self.adversary_list = []
        self.adversary_iter_num = 10
        for i in range(len(self.adv_models)):
            self.adversary_list.append(
                LinfPGDAttack(
                    self.adv_models[i], loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8. / 255,
                    nb_iter=self.adversary_iter_num, eps_iter=2. / 255, rand_init=True, clip_min=0.0,
                    clip_max=1.0, targeted=False))
        print('Meta Learner Initialization done.')

    def adv_loss(self, y, label):
        loss = 0.0
        for adv_model in self.adv_models:
            logits = adv_model(y)

            if not self.target:
                one_hot = torch.zeros_like(logits, dtype=torch.uint8)
                label = label.reshape(-1, 1)
                one_hot.scatter_(1, label, 1)
                one_hot = one_hot.bool()
                diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0]
                margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin

            else:

                target_one_hot = torch.zeros_like(logits, dtype=torch.uint8)
                target_one_hot.scatter_(1, self.target_label, 1)
                target_one_hot = target_one_hot.bool()
                # target loss
                diff = -logits[target_one_hot]
                margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin

            loss += margin.mean()
        loss /= len(self.adv_models)

        return loss

    def augmentation(self, x, true_lab, no_adv=False):
        if self.args.adv_aug and (not no_adv):
            # x = preprocess(x, 1.0, 0.0, self.x_bins, False)
            if self.args.adv_rand:
                model_idx = np.random.randint(0, len(self.adv_models))
                model_chosen = self.adv_models[model_idx]
                iter_num = np.random.randint(0, 20 + 1)

                if iter_num > 0:
                    adversary = self.adversary_list[model_idx]
                    with ctx_noparamgrad_and_eval(model_chosen):
                        x = adversary.perturb(x, None)
                else:
                    x = preprocess(x, 1.0, 0.0, self.x_bins, False)

            elif not no_adv:
                model_idx = np.random.randint(0, len(self.adv_models))
                model_chosen = self.adv_models[model_idx]

                adversary = self.adversary_list[model_idx]

                with ctx_noparamgrad_and_eval(model_chosen):
                    x = adversary.perturb(x, None)
        else:
            x = preprocess(x, 1.0, 0.0, self.x_bins, False)
        return x

    def schedule(self, loss_prob, loss_cls, epoch):
        if loss_prob <= 0:
            return 0.01
        else:
            return 0.02

    def meta_train(self, batch_data, epoch):
        """
        Reptile meta training process
        """
        if self.args.curriculum:
            x = batch_data['cln_img']
            label = batch_data['true_lab']
        else:
            x = batch_data[0]
            label = batch_data[1]
        batch_length = len(label)
        if self.cuda:
            x = x.cuda()
            label = label.cuda()
        label = label.long()

        processed_x = self.augmentation(x, label, epoch < self.args.adv_epoch)
        _flow = copy.deepcopy(self.flow)

        self.optim = torch.optim.Adam(self.flow.parameters(), lr=0.0004, betas=self.args.betas,
                                      weight_decay=self.args.regularizer)

        for i in range(self.meta_iteration):
            y, logdet = self.flow.decode(processed_x, return_prob=True)

            loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(
                torch.clamp(torch.clamp(y, -self.linf, self.linf) + x, 0, 1), label)
            loss = loss_cls
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.max_grad_norm)
            # step
            self.optim.step()

        # meta_outer_update
        lr = self.schedule(loss_prob.data, loss_cls.data, epoch)
        dic = self.flow.state_dict()
        keys = list(dic.keys())

        meta_state = _flow.state_dict()
        for key in keys:
            dic[key] = meta_state[key] + lr / batch_length * 2 / self.meta_iteration * (dic[key] - meta_state[key])
        return loss_prob.data, loss_cls.data

    def meta_test(self):
        def check(model, image, label):
            prob_output = torch.nn.functional.softmax(model(image), dim=1)
            pred_lable = torch.argmax(prob_output, dim=1)
            print(pred_lable.item(), label)
            return pred_lable.item() != label

        mean_loss_prob, mean_loss_cls = 0, 0
        with torch.no_grad():
            for batch_data in self.validset_loader:
                images = batch_data[0]
                labels = batch_data[1]
                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                labels = labels.long()
                y, logdet = self.flow.decode(images, return_prob=True)
                loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(
                    torch.clamp(torch.sign(y) * 8. / 255 + images, 0, 1), labels)
                mean_loss_prob += loss_prob
                mean_loss_cls += loss_cls
        with open(os.path.join(self.log_dir, "meta_test_loss.txt"), "a") as f:
            f.write("mean_loss_prob: {:.5f}, mean_loss_cls: {:.5f}".format(mean_loss_prob, mean_loss_cls) + "\n")
        return mean_loss_prob, mean_loss_cls

    def train(self):
        """
        update the latent and the parameter and meta test
        Returns:
        """
        self.flow.train()
        starttime = time.time()
        # run
        num_batchs = len(self.trainingset_loader)

        total_its = self.num_epochs * num_batchs
        # parameter_loss = None
        for epoch in range(self.num_epochs):
            mean_loss_prob, mean_loss_cls = 0, 0
            for batch_id, batch in enumerate(self.trainingset_loader):
                loss_prob, loss_cls = self.meta_train(batch, epoch)
                mean_loss_prob += loss_prob
                mean_loss_cls += loss_cls
                if (self.global_step + 1) % self.test_gap == 0:
                    self.meta_test()
                # save model
                if (self.global_step + 1) % self.save_gap == 0:
                    save_model(self.flow, self.optim, self.checkpoints_dir, self.global_step + 1)
                self.global_step = self.global_step + 1
                # TODO change the print
                currenttime = time.time()
                elapsed = currenttime - starttime
                if self.global_step % 50 == 0:
                    print(
                        "Iteration: {}/{} \t Epoch: {}/{} \t Elapsed time: {:.2f} \t Meta train loss prob: {:.4f} \t loss cls: {:.4f}".format(
                            self.global_step, total_its, epoch, self.num_epochs, elapsed, loss_prob, loss_cls))

                if batch_id % self.args.log_gap == 0:
                    mean_loss_prob = float(mean_loss_prob / float(num_batchs))
                    mean_loss_cls = float(mean_loss_cls / float(num_batchs))
                    with open(os.path.join(self.log_dir, "Epoch_NLL.txt"), "a") as f:
                        currenttime = time.time()
                        elapsed = currenttime - starttime
                        f.write(
                            "epoch: {} \t iteration: {}/{} \t elapsed time: {:.2f}\t mean loss prob: {:.5f}\t mean loss cls: {:.5f}".format(
                                epoch, self.global_step, total_its, elapsed, mean_loss_prob, mean_loss_cls) + "\n")
