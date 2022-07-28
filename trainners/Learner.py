import os
import torch
import time
import datetime
import numpy as np
from torch.utils.data import DataLoader
from utils.tools import preprocess, save_model


class Trainer(object):
    def __init__(self, graph, optim, trainingset, validset, args, cuda):
        # set path and date
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
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

        self.images_dir = os.path.join(self.log_dir, "images")
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        self.valid_samples_dir = os.path.join(self.log_dir, "valid_samples")
        if not os.path.exists(self.valid_samples_dir):
            os.makedirs(self.valid_samples_dir)
        # model
        self.graph = graph
        self.optim = optim

        # gradient bound
        self.max_grad_clip = args.max_grad_clip
        self.max_grad_norm = args.max_grad_norm

        # data
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.trainingset_loader = DataLoader(trainingset,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             drop_last=True)

        self.validset_loader = DataLoader(validset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          drop_last=False)

        self.num_epochs = args.num_epochs
        self.global_step = args.num_steps
        self.label_scale = args.label_scale
        self.label_bias = args.label_bias
        self.x_bins = args.x_bins
        self.y_bins = args.y_bins

        self.num_epochs = args.num_epochs
        self.log_gap = args.log_gap
        self.inference_gap = args.inference_gap
        self.test_gap = args.test_gap
        self.save_gap = args.save_gap

        # device
        self.cuda = cuda
        self.lr = 0.05
        self.linf = 8. / 255

    def validate(self):
        print ("Start Validating")
        self.graph.eval()
        mean_loss = list()
        samples = list()
        with torch.no_grad():
            for i_batch, batch in enumerate(self.validset_loader):
                x, y = batch['cln_img'], batch['adv_img']
                y = y - x
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                y = preprocess(y, self.label_scale, self.label_bias, self.y_bins, True)
                # forward
                z, nll = self.graph(x,y)
                loss = torch.mean(nll)
                mean_loss.append(loss.data.cpu().item())

        # save loss
        mean = np.mean(mean_loss)
        with open(os.path.join(self.log_dir, "valid_NLL.txt"), "a") as nll_file:
            nll_file.write(str(self.global_step) + "\t" + "{:.5f}".format(mean) + "\n")
        print("Finish Validating")
        self.graph.train()

    def train(self):
        self.graph.train()
        starttime = time.time()

        # run
        num_batchs = len(self.trainingset_loader)
        total_its = self.num_epochs * num_batchs
        for epoch in range(self.num_epochs):
            mean_nll = 0.0
            for _, batch in enumerate(self.trainingset_loader):
                self.optim.zero_grad()
                x, y = batch['cln_img'], batch['adv_img']

                y = y - x
                y = y.sign() * self.linf
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()

                processed_y = preprocess(y, self.label_scale, self.label_bias, self.y_bins, False)
                processed_x = preprocess(x, 1.0, 0.0, self.x_bins, False)

                # forward
                z, nll = self.graph(processed_x, processed_y, reverse=False)

                # loss
                loss = torch.mean(nll)
                mean_nll = mean_nll + loss.data

                # backward
                self.graph.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                # operate grad
                if self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)

                # step
                self.optim.step()

                currenttime = time.time()
                elapsed = currenttime - starttime
                if self.global_step % 50 == 0:
                    print("Iteration: {}/{} \t Elapsed time: {:.2f} \t Loss:{:.5f}".format(self.global_step, total_its, elapsed, loss.data))

                if self.global_step % self.log_gap == 0:
                    with open(os.path.join(self.log_dir, "NLL.txt"), "a") as nll_file:
                        nll_file.write(str(self.global_step) + " \t " + "{:.2f} \t {:.5f}".format(elapsed, loss.data) + "\n")

                # checkpoint
                if self.global_step % self.test_gap == 0 and self.global_step > 0:
                    self.validate()

                    # # samples

                # save model
                if self.global_step % self.save_gap == 0 and self.global_step > 0:
                    save_model(self.graph, self.optim, self.checkpoints_dir, self.global_step)

                self.global_step = self.global_step + 1

            # Manually schedule
            if loss >= 17:
                lr = 0.1
            elif loss >= 10:
                lr = 0.05
            elif loss >= 5:
                lr = 0.001
            else:
                lr = 0.001

            if lr != self.lr:
                self.lr = lr
                for p in self.optim.param_groups:
                    p['lr'] = self.lr

            mean_nll = float(mean_nll / float(num_batchs))
            with open(os.path.join(self.log_dir, "Epoch_NLL.txt"), "a") as f:
                currenttime = time.time()
                elapsed = currenttime - starttime
                f.write("{} \t {:.2f}\t {:.5f}".format(epoch, elapsed, mean_nll) + "\n")

