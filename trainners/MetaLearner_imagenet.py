import os
import random
import torch
import torch.nn as nn
import datetime
import numpy as np
from torch.utils.data import DataLoader
import utils
import time
from datasets import convert_to_img
from datasets import preprocess
from datasets import postprocess
from torchvision.utils import save_image
from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.utils import predict_from_logits
import cma


class Trainer(object):

    def __init__(self, graph, adv_models, optim, scheduler, latent, train_set, valid_set, args, cuda):

        # set path and date
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "") \
            .replace(":", "") \
            .replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, "log_" + args.name if args.name != '' else date)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        self.images_dir = os.path.join(self.log_dir, "images")
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        self.valid_samples_dir = os.path.join(self.log_dir, "valid_samples")

        if not os.path.exists(self.valid_samples_dir):
            os.makedirs(self.valid_samples_dir)

        # model
        self.graph = graph
        self.graph.eval()
        self.adv_models = adv_models
        self.adv_model_id = 0

        self.optim = optim
        self.scheduler = scheduler
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
        self.nll_gap = args.nll_gap
        self.inference_gap = args.inference_gap
        self.checkpoints_gap = args.checkpoints_gap
        self.save_gap = args.save_gap
        self.target = args.target

        self.args = args

        # device
        self.cuda = cuda
        # meta latent
        self.mean, self.logs = self.graph.prior()
        if latent is None:
            self.latent = torch.zeros(self.mean.shape).cuda()
        else:
            self.latent = latent
        self.latent_lr = 1
        self.label_num = 1000
        # CMA-ES
        self.cma_options = {'CMA_active': False,
                            'CMA_diagonal': True,
                            'popsize': self.args.cma_popsize,
                            'seed': self.args.cma_seed,
                            'ftarget': self.args.cma_ftarget,
                            'maxfevals': self.args.cma_update_max_epoch,
                            'tolfun': 1e-10,
                            'verb_disp': 0}
        # adversary
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
        # TODO, update the adv loss to be the normal and check margin whther is right or not
        loss = 0.
        # selected_class = [864, 394, 776, 911, 430, 41, 265, 988, 523, 497]
        #
        # for idx in range(len(label)):
        #     label[idx] = selected_class.index(int(label[idx]))
        # loss = None
        for adv_model in self.adv_models:
            #            loss = 0.
            logits = adv_model(y)
            # logits = logits[:, selected_class]
            if not self.target:
                # print (logits[0,:])
                one_hot = torch.zeros_like(logits, dtype=torch.uint8)
                label = label.reshape(-1, 1)
                one_hot.scatter_(1, label, 1)
                # # print ('1',~one_hot[0,:], logits[one_hot])
                one_hot = one_hot.bool()
                # # print ('2', ~one_hot[0,:], logits[one_hot])
                diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0]
                margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
                # import torch.nn.functional as F
                # print (logits.size(), label.size())
                # margin = -F.cross_entropy(logits, label.view(-1), reduction='mean')
            else:
                # diff = torch.max(torch.cat((logits[:, :label],logits[:,(label+1):]), dim=1), dim=1)[0] - logits[:, label]
                # margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
                one_hot = torch.zeros_like(logits, dtype=torch.uint8)
                label = label.reshape(-1, 1)
                one_hot.scatter_(1, label, 1)
                one_hot = one_hot.bool()
                # selected = one_hot; selected[:, selected_class] = True; selected = (~one_hot & selected)
                # diff = torch.max(logits[selected].view(len(logits),-1), dim=1)[0] - logits[one_hot]
                diff = torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0] - logits[one_hot]
                margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
                # margin = diff
                # print (logits.mean(dim=0))
                # import torch.nn.functional as F
                # margin = F.cross_entropy(logits, label, reduction='mean')
                # p = torch.softmax(logits, dim=-1)
                # margin = torch.sum(p * torch.log(p), dim=-1).mean()
            loss += margin.mean()
            # loss = margin.mean() if (loss is None or loss > margin.mean()) else loss
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
                    # print ('adv_train_rand')
                    with ctx_noparamgrad_and_eval(model_chosen):
                        x = adversary.perturb(x, None)
                else:
                    x = preprocess(x, 1.0, 0.0, self.x_bins, False)

            elif (not no_adv):
                model_idx = np.random.randint(0, len(self.adv_models))
                model_chosen = self.adv_models[model_idx]

                adversary = self.adversary_list[model_idx]

                with ctx_noparamgrad_and_eval(model_chosen):
                    x = adversary.perturb(x, None)
        else:
            x = preprocess(x, 1.0, 0.0, self.x_bins, False)
        return x

    def adv_loss_list(self, adv_images, label, adv_model_id=0):
        """
        Calculate the loss of a single adv model
        adv_images is a list of same image
        label is only one label
        Returns: loss list
        """
        logits = self.adv_models[adv_model_id](adv_images)
        # print('logits shape', logits.shape) #torch.Size([4, 1000])
        label_logits = logits[:, label]
        other_max_logits = torch.max(torch.cat((logits[:, :label], logits[:, (label + 1):]), dim=1), dim=1)[0]
        # print(label_logits)
        # print(other_max_logits)
        if not self.target:
            diff = label_logits - other_max_logits
            loss = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        else:
            diff = other_max_logits - label_logits
            loss = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        return loss

    def meta_inner_update(self, image, label):
        """
        Description:
            Nearly the same as the formal attacking process
            Use CMA-ES to update the latent and return the successful latent.
            condition: a batch with same image
            latent: cma generated queries, one success will stop
        Args:
            image:
            label:
        Returns:

        """
        # print('image shape', image.shape) #torch.Size([3, 224, 224])
        # print('label shape', label.shape) #torch.Size([])
        images = image.unsqueeze(0).repeat(self.args.cma_popsize, 1, 1, 1)
        # one_hot_labels = torch.zeros([self.args.cma_popsize, self.label_num], dtype=torch.uint8).cuda()
        # one_hot_labels.scatter_(1, labels, 1)
        # one_hot_labels = one_hot_labels.bool()
        # TODO model_id not sure how to set
        # print(self.latent.shape) torch.Size([1, 48, 4, 4])
        es = cma.CMAEvolutionStrategy(self.latent.view(-1).cpu().numpy(), self.args.cma_sigma0, self.cma_options)
        query_number = 0
        while not es.stop() and es.best.f > 0:
            noise_list = es.ask()
            latent = torch.FloatTensor(np.array(noise_list).reshape(-1, self.latent.shape[1], self.latent.shape[2], self.latent.shape[3])).cuda()
            perturbation, _ = self.graph.flow.decode(x=images, y=latent)
            # print('perturbation shape', perturbation.shape) #torch.Size([4, 3, 224, 224])
            adv_images = torch.clamp(images + torch.sign(perturbation) * 0.05, 0, 1)
            with torch.no_grad():
                loss_list = self.adv_loss_list(adv_images, label)
                loss_list = loss_list.cpu().data.numpy().reshape(-1).tolist()
            # print(len(noise_list), noise_list[0].shape)
            # print(loss_list)
            es.tell(noise_list, loss_list)
            query_number += self.args.cma_popsize
        # print('best f', es.best.f)
        return es.best.x, es.best.f, query_number

    def meta_train_latent(self, batch_data):
        """
        Update latent per image of batch_data
        """
        batch_length = len(batch_data[0])
        latent_list = []
        total_loss = 0
        total_query_number = 0
        for i in range(batch_length):
            image, label = batch_data[0][i], batch_data[1][i]
            if self.cuda:
                image = image.cuda()
                label = label.cuda()
            label = label.long()
            new_latent, loss, query_number = self.meta_inner_update(image, label)
            total_loss += loss
            total_query_number += query_number
            latent_list.append(torch.from_numpy(new_latent).cuda().view(self.latent.shape))
        # meta outer update
        total_diff = torch.zeros_like(self.latent)
        for latent in latent_list:
            total_diff += latent - self.latent
            # print(total_diff[0][0])
        # TODO now the level is around 1e4 which seems not that correct
        self.latent += self.latent_lr / batch_length * total_diff
        print('latent mean value', torch.mean(self.latent))
        return total_loss / batch_length, total_query_number / batch_length

    def schedule(self, latent_loss, parameter_loss, epoch):
        """
        update lr including latent lr and parameter lr
        """
        # TODO, now I think that the smaller the loss is, the more correct the latent is
        if latent_loss <= 0:
            self.latent_lr = 0.4
        elif latent_loss < 2:
            self.latent_lr = 0.1
        else:
            self.latent_lr = 0.01
        return

    def train_parameter(self, batch_data, epoch):
        """
        every sure epochs update the parameters use the before task
        Returns:
        """
        self.graph.train()
        self.optim.zero_grad()

        x = batch_data[0]
        label = batch_data[1]
        batch_length = len(label)
        if self.cuda:
            x = x.cuda()
            label = label.cuda()
        label = label.long()

        processed_x = self.augmentation(x, label, epoch < self.args.adv_epoch)

        history = []


        for i in range(1):
            # print(processed_x.shape) #torch.Size([4, 3, 224, 224])
            # print(self.latent.shape) #torch.Size([1, 48, 4, 4])
            y, logdet = self.graph.flow.decode(x=processed_x, y=self.latent.repeat(batch_length, 1, 1, 1))
            if self.args.tanh:
                loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(torch.tanh(y) * 8. / 255. + x, label)
            elif self.args.clamp:
                # loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(torch.clamp(torch.clamp(y,  -8. / 255.,  8. / 255.) + x, 0, 1), label)
                # loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(torch.clamp(-torch.sign(c) * 8. / 255 + x, 0, 1), label)
                # TODO now in here branch
                loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(
                    torch.clamp(torch.clamp(y, -8. / 255., 8. / 255.) + x, 0, 1), label)
            else:
                loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(
                    y / y.abs().max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1) * 8. / 255. + x, label)

            if self.args.only:
                loss = loss_cls
            else:
                loss = self.args.Lambda * loss_prob + loss_cls
            # backward
            # self.graph.zero_grad()
            self.optim.zero_grad()
            loss.backward()
            # operate grad
            if self.args.normalize_grad:
                parameters = list(filter(lambda p: (p.grad is not None), self.graph.parameters()))
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
                clip_coef = self.max_grad_norm / (total_norm + 1e-6)
                for p in parameters:
                    # print (torch.norm(p.grad.detach(), 2))
                    p.grad.detach().mul_(clip_coef)

            if self.max_grad_clip > 0:
                torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
            # step
            self.optim.step()
            print(
                "Iteration: {} \t Loss:{:.5f} \t Loss_Prob:{:.5f} \t Loss_Cls:{:.5f}".format(
                    self.global_step, loss.data, loss_prob.data, loss_cls.data))
        self.graph.eval()
        return loss.data

    def meta_test(self):
        """
        every sure epochs, Then will be made to use fine-tune to check the performance of the model
        while not update the parameters or latents
        Returns:

        """
        data_number = 0
        total_loss = 0
        total_query_number = 0
        self.cma_options['maxfevals'] = 1000
        # with torch.no_grad():
        from advertorch.attacks import L2PGDAttack, LinfPGDAttack
        adversary = LinfPGDAttack(
            self.adv_models[0], loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.06,
            nb_iter=200, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=False)

        def check(model, image, label):
            prob_output = torch.nn.functional.softmax(model(image), dim=1)
            pred_lable = torch.argmax(prob_output, dim=1)
            print(pred_lable.item(), label)
            return pred_lable.item() != label

        for batch_data in self.validset_loader:
            # TODO now I think since every batch is a task, we do not need to valid so frequently, so break once
            data_number += self.batch_size
            for i in range(self.batch_size):
                image, label = batch_data[0][i], batch_data[1][i]
                if self.cuda:
                    image = image.cuda()
                    label = label.cuda()
                label = label.long()

                image = image.unsqueeze(dim=0)

                adv_image = adversary.perturb(image)
                noise = adv_image - image

                latent, logdet, zs, y_downsample = self.graph.flow.encode(x=image, y=noise, return_z=True)
                print('latent: ', torch.sum(latent))
                adv, logdet, y_ori = self.graph.flow.decode(image, latent, zs=zs)
                # print(torch.sum(y_ori))
                # print(torch.sum(y_downsample))
                adv = torch.clamp(adv, min=-0.06, max=0.06)
                diff = torch.sum(torch.abs(noise - adv))
                print('diff: ', diff)

                if check(self.adv_models[0], torch.clamp(image + adv, min=0, max=1), label):
                    print('OK')
                else:
                    print('???????')

                # new_latent, loss, query_number = self.meta_inner_update(image, label)
                # total_loss += loss
                # total_query_number += query_number
            # break
        mean_loss = total_loss / data_number
        mean_query = total_query_number / data_number
        print("Meta-testing: Mean loss: {}, Mean query: {}".format(mean_loss, mean_query))
        # with open(os.path.join(self.log_dir, "valid_NLL.txt"), "a") as nll_file:
        #     nll_file.write(str(self.global_step) + "\t" + "{:.5f}".format(mean_loss) + "\n")
        self.cma_options['maxfevals'] = self.args.cma_update_max_epoch
        return mean_loss, mean_query

    def train(self):
        """
        update the latent and the parameter and meta test
        Returns:
        """
        self.meta_test()


        starttime = time.time()
        # run
        num_batchs = len(self.trainingset_loader)

        total_its = self.num_epochs * num_batchs
        latent_loss = None
        parameter_loss = None
        for epoch in range(self.num_epochs):
            mean_nll = 0.0
            for batch in self.trainingset_loader:
                # TODO change the training process, more data in the changing process
                with torch.no_grad():
                    latent_loss, train_query_number = self.meta_train_latent(batch)
                if False:
                    parameter_loss = self.train_parameter(batch, epoch)
                self.schedule(latent_loss, parameter_loss, epoch)
                # checkpoint
                if (self.global_step + 1) % self.checkpoints_gap == 0:
                    with torch.no_grad():
                        self.meta_test()
                # save model
                if (self.global_step + 1) % self.save_gap == 0:
                    utils.save_model(self.graph, self.optim, self.scheduler, self.checkpoints_dir, self.global_step, self.latent)
                self.global_step = self.global_step + 1
                # TODO change the print
                currenttime = time.time()
                elapsed = currenttime - starttime
                print(
                    "Iteration: {}/{} \t Epoch: {}/{} \t Elapsed time: {:.2f} \t Meta train latent loss: {:.4f} \t Query number {:.2f}".format(
                        self.global_step, total_its, epoch, self.num_epochs, elapsed, latent_loss, train_query_number))

            mean_nll = float(mean_nll / float(num_batchs))
            with open(os.path.join(self.log_dir, "Epoch_NLL.txt"), "a") as f:
                currenttime = time.time()
                elapsed = currenttime - starttime
                f.write("{} \t {:.2f}\t {:.5f}".format(epoch, elapsed, mean_nll) + "\n")


class Inferencer(object):

    def __init__(self, model, dataset, args, cuda):
        # set path and date
        self.out_root = args.out_root
        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)

        # cuda
        self.cuda = cuda
        # model
        self.model = model

        # data
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.data_loader = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      drop_last=False)

        self.label_scale = args.label_scale
        self.label_bias = args.label_bias
        self.num_labels = args.num_labels

    def sampled_based_prediction(self, n_samples):
        metrics = []
        start = time.time()
        for i_batch, batch in enumerate(self.data_loader):
            print(f"Batch IDs: {i_batch}")

            x = batch[0]
            y = x

            if self.cuda:
                x = x.cuda()
                y = y.cuda()

            sample_list = list()
            nll_list = list()
            for i in range(0, n_samples):
                print(f"Samples: {i}/{n_samples}")

                y_sample, _ = self.model(x, reverse=True)
                _, nll = self.model(x, y_sample)
                loss = torch.mean(nll)
                sample_list.append(y_sample)
                nll_list.append(loss.data.cpu().numpy())

            sample = torch.stack(sample_list)
            sample = torch.mean(sample, dim=0, keepdim=False)
            nll = np.mean(nll_list)

            sample = postprocess(sample, self.label_scale, self.label_bias)

            y_pred_imgs, y_pred_seg = convert_to_img(sample)
            y_true_imgs, y_true_seg = convert_to_img(y)

            # save trues and preds
            output = None
            for i in range(0, len(y_true_imgs)):
                true_img = y_true_imgs[i]
                pred_img = y_pred_imgs[i]
                row = torch.cat((x[i].cpu(), true_img, pred_img), dim=1)
                if output is None:
                    output = row
                else:
                    output = torch.cat((output, row), dim=2)
            save_image(output, os.path.join(self.out_root, "trues-{}.png".format(i_batch)))

            acc, acc_cls, mean_iu, fwavacc = utils.compute_accuracy(y_true_seg, y_pred_seg, self.num_labels)

            with open(os.path.join(self.out_root, "meta_list.txt"), "a") as meta_file:
                meta_file.write("NLL: {:.5f}".format(nll) + "\t")
                meta_file.write("acc: {:.8f}".format(acc) + "\t")
                meta_file.write("acc_cls: {:.8f}".format(acc_cls) + "\t")
                meta_file.write("mean_iu: {:.8f}".format(mean_iu) + "\t")
                meta_file.write("fwavacc: {:.8f}".format(fwavacc) + "\t")
                meta_file.write("\n")

            metrics.append([acc, acc_cls, mean_iu, fwavacc])
        mean_metrics = np.mean(metrics, axis=0)

        finish = time.time()
        elapsed = finish - start

        with open(os.path.join(self.out_root, "sum_meta.txt"), "w") as meta_file:
            meta_file.write("time:{:.2f}".format(elapsed) + "\t")
            meta_file.write("acc: {:.8f}".format(mean_metrics[0]) + "\t")
            meta_file.write("acc_cls: {:.8f}".format(mean_metrics[1]) + "\t")
            meta_file.write("mean_iu: {:.8f}".format(mean_metrics[2]) + "\t")
            meta_file.write("fwavacc: {:.8f}".format(mean_metrics[3]) + "\t")
