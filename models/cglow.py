import torch
import torch.nn as nn
import numpy as np
import models.modules as modules
import torch_dct as dct
import torch.nn.functional as F


class CondGlowStep(nn.Module):

    def __init__(self, x_size, y_size, x_hidden_channels, x_hidden_size, y_hidden_channels):

        super().__init__()

        # 1. cond-actnorm
        self.actnorm = modules.CondActNorm(x_size=x_size, y_channels=y_size[0], x_hidden_channels=x_hidden_channels,
                                           x_hidden_size=x_hidden_size)

        # 2. cond-1x1conv
        self.invconv = modules.Cond1x1Conv(x_size=x_size, x_hidden_channels=x_hidden_channels,
                                           x_hidden_size=x_hidden_size, y_channels=y_size[0])

        # 3. cond-affine
        self.affine = modules.CondAffineCoupling(x_size=x_size, y_size=[y_size[0] // 2, y_size[1], y_size[2]],
                                                 hidden_channels=y_hidden_channels)

    def forward(self, x, y, logdet=None, reverse=False):

        if reverse is False:
            # 1. cond-actnorm
            y, logdet = self.actnorm(x, y, logdet, reverse=False)

            # 2. cond-1x1conv
            y, logdet = self.invconv(x, y, logdet, reverse=False)

            # 3. cond-affine
            y, logdet = self.affine(x, y, logdet, reverse=False)

            # Return
            return y, logdet

        if reverse is True:
            # 3. cond-affine
            y, logdet = self.affine(x, y, logdet, reverse=True)

            # 2. cond-1x1conv
            y, logdet = self.invconv(x, y, logdet, reverse=True)

            # 1. cond-actnorm
            y, logdet = self.actnorm(x, y, logdet, reverse=True)

            # Return
            return y, logdet


def downsample_dct_interface(scale=4.):
    # scale = 1.
    def downsample_dct(im):
        im_dct = dct.dct_3d(im)
        dim_keep = int(im.size(2) / float(scale))
        # print ('dims ', dim_keep)
        return im_dct[:, :, :dim_keep, :dim_keep]

    return downsample_dct


def upsample_dct_interface(scale=4.):
    # scale = 1.
    def upsample_dct(im):
        pad_dim = int(im.size(2) * (scale - 1.))
        # print ('dims ', pad_dim)
        im_pad = F.pad(im, (0, pad_dim, 0, pad_dim))
        im_up = dct.idct_3d(im_pad)
        return im_up

    return upsample_dct


class CondGlow(nn.Module):

    def __init__(self, x_size, y_size, x_hidden_channels, x_hidden_size, y_hidden_channels, K, L, down_sample_x,
                 down_sample_y):

        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        self.L = L
        self.down_sample = down_sample_x, down_sample_y

        C, H, W = y_size

        if self.down_sample[0] > 1:
            H_x, W_x = int(x_size[1] // self.down_sample[0]), int(x_size[2] // self.down_sample[0])
            H_x, W_x = int(2 ** np.ceil(np.log(H_x) / np.log(2))), int(2 ** np.ceil(np.log(W_x) / np.log(2)))
            scale_factor = x_size[1] / float(H_x)
            self.downsampler_x = nn.Upsample(scale_factor=1. / scale_factor, mode='bilinear', align_corners=True)
            self.upsampler_x = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            # self.downsampler_x = downsample_dct_interface(scale_factor)
            # self.upsampler_x = upsample_dct_interface(scale_factor)
            x_size = (x_size[0], H_x, W_x)
            # x_size = (3, 64, 64)
            # print ('x_size : ', x_size)
        if self.down_sample[1] > 1:
            H, W = int(H // self.down_sample[1]), int(W // self.down_sample[1])
            H, W = int(2 ** np.ceil(np.log(H) / np.log(2))), int(2 ** np.ceil(np.log(W) / np.log(2)))
            scale_factor = y_size[1] / float(H)
            # self.downsampler_y = nn.Upsample(scale_factor=1./scale_factor, mode='bilinear', align_corners=True)
            # self.upsampler_y = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.downsampler_y = downsample_dct_interface(scale_factor)
            self.upsampler_y = upsample_dct_interface(scale_factor)

        self.dimensions = H * W * C
        # print (C, H, W)

        for l in range(0, L):

            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            y_size = [C, H, W]
            self.layers.append(modules.SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K CGlowStep
            for k in range(0, K):
                self.layers.append(CondGlowStep(x_size=x_size,
                                                y_size=y_size,
                                                x_hidden_channels=x_hidden_channels,
                                                x_hidden_size=x_hidden_size,
                                                y_hidden_channels=y_hidden_channels,
                                                )
                                   )

                self.output_shapes.append([-1, C, H, W])

            # 3. Split
            if l < L - 1:
                self.layers.append(modules.Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, x, y, logdet=0.0, reverse=False, eps_std=1.0):

        if reverse == False:
            return self.encode(x, y, logdet)
        else:
            return self.decode(x, y, logdet, eps_std)

    def encode(self, x, y, logdet=0.0, return_z=False):
        if self.down_sample[0] > 1:
            x = self.downsampler_x(x)

        if self.down_sample[1] > 1:
            y_ori = y
            y = self.downsampler_y(y)

        zs = []
        for layer, shape in zip(self.layers, self.output_shapes):
            if isinstance(layer, modules.Split2d):
                y, z2, logdet = layer(y, logdet, reverse=False)
                zs.append(z2)
            elif isinstance(layer, modules.SqueezeLayer):
                y, logdet = layer(y, logdet, reverse=False)
            else:
                y, logdet = layer(x, y, logdet, reverse=False)
        if return_z:
            return y, logdet, zs
        else:
            return y, logdet

    def decode(self, x, y, logdet=0.0, eps_std=1.0, zs=None):
        if self.down_sample[0] > 1:
            x = self.downsampler_x(x)

        level_cnt = 1
        for idx, layer in enumerate(reversed(self.layers)):
            if isinstance(layer, modules.Split2d):
                if zs is None:
                    y, logdet = layer(y, logdet=logdet, reverse=True, eps_std=eps_std)
                else:
                    y, logdet = layer(y, logdet=logdet, reverse=True, eps_std=eps_std, z2=zs[-level_cnt])
                    level_cnt += 1
                # print('decode:Split2d: ', torch.sum(y))
            elif isinstance(layer, modules.SqueezeLayer):
                y, logdet = layer(y, logdet=logdet, reverse=True)
                # print('decode:SqueezeLayer: ', torch.sum(y))
            else:
                y, logdet = layer(x, y, logdet=logdet, reverse=True)
                # print('decode:CondFlowStep: ', torch.sum(y))

        if self.down_sample[1] > 1:
            self.y_ori = y
            y = self.upsampler_y(y)

        return y, logdet


class CondGlowModel(nn.Module):
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, args):
        super().__init__()
        self.flow = CondGlow(x_size=args.x_size,
                             y_size=args.y_size,
                             x_hidden_channels=args.x_hidden_channels,
                             x_hidden_size=args.x_hidden_size,
                             y_hidden_channels=args.y_hidden_channels,
                             K=args.flow_depth,
                             L=args.num_levels,
                             down_sample_x=args.down_sample_x,
                             down_sample_y=args.down_sample_y,
                             )

        self.learn_top = args.learn_top

        self.register_parameter("new_mean",
                                nn.Parameter(torch.zeros(
                                    [1,
                                     self.flow.output_shapes[-1][1],
                                     self.flow.output_shapes[-1][2],
                                     self.flow.output_shapes[-1][3]])))

        self.register_parameter("new_logs",
                                nn.Parameter(torch.zeros(
                                    [1,
                                     self.flow.output_shapes[-1][1],
                                     self.flow.output_shapes[-1][2],
                                     self.flow.output_shapes[-1][3]])))

        self.n_bins = args.y_bins

        self.args = args

        self.dimensions = self.flow.dimensions

    def prior(self):

        if self.learn_top:
            # print ('learn')
            return self.new_mean, self.new_logs
        else:
            # print ('don\'t learn')
            return torch.zeros_like(self.new_mean), torch.zeros_like(self.new_mean)

    def forward(self, x=0.0, y=None, eps_std=1.0, reverse=False):
        if reverse == False:
            # dimensions = y.size(1)*y.size(2)*y.size(3)
            dimensions = self.dimensions
            logdet = torch.zeros_like(y[:, 0, 0, 0])
            logdet += float(-np.log(self.n_bins) * dimensions)
            z, objective = self.flow(x, y, logdet=logdet, reverse=False)
            mean, logs = self.prior()
            objective += modules.GaussianDiag.logp(mean, logs, z)
            nll = -objective / float(np.log(2.) * dimensions)
            return z, nll

        else:
            with torch.no_grad():
                mean, logs = self.prior()
                if y is None:
                    y = modules.GaussianDiag.batchsample(x.size(0), mean, logs, eps_std)
                y, logdet = self.flow(x, y, eps_std=eps_std, reverse=True)
            return y, logdet

    def sample(self, x, eps_std=1.0):
        mean, logs = self.prior()
        y = modules.GaussianDiag.batchsampleR(x.size(0), mean, logs, eps_std)
        objective = modules.GaussianDiag.logp(mean, logs, y)
        y, logdet = self.flow(x, y, eps_std=eps_std, reverse=True)
        # dimensions = y.size(1)*y.size(2)*y.size(3)
        dimensions = self.dimensions
        objective += float(-np.log(self.n_bins) * dimensions)
        objective -= logdet
        objective /= float(np.log(2.) * dimensions)
        return y, objective

    def decode(self, x, mean_input=None, logs_input=None, eps_std=1.0, return_prob=False, zs=None, no_norm=False):
        mean, logs = self.prior()
        mean = mean if mean_input is None else mean_input
        logs = logs if logs_input is None else logs_input

        y = modules.GaussianDiag.batchsampleR(x.size(0), mean, logs, eps_std)
        objective = modules.GaussianDiag.logp(mean, logs, y)
        y, logdet = self.flow.decode(x, y, eps_std=eps_std, zs=zs)

        # dimensions = y.size(1)*y.size(2)*y.size(3)
        dimensions = self.dimensions
        objective += float(-np.log(self.n_bins) * dimensions)
        objective += -logdet
        objective /= float(np.log(2.) * dimensions)
        if return_prob:
            if no_norm:
                return y, objective
            if self.args.tanh:
                return torch.tanh(y), objective
            else:
                return y / y.abs().max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1), objective
        else:
            if no_norm:
                return y
            if self.args.tanh:
                return torch.tanh(y)
            else:
                return y / y.abs().max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1)

    def encode(self, x=0.0, y=None, eps_std=1.0, return_z=False):
        # dimensions = y.size(1)*y.size(2)*y.size(3)
        dimensions = self.dimensions
        logdet = torch.zeros_like(y[:, 0, 0, 0])
        logdet += float(-np.log(self.n_bins) * dimensions)
        z, objective, zs = self.flow.encode(x, y, logdet=logdet, return_z=True)

        mean, logs = self.prior()
        objective += modules.GaussianDiag.logp(mean, logs, z)
        nll = -objective / float(np.log(2.) * dimensions)

        if return_z:
            return z, objective / float(np.log(2.) * dimensions), zs
        else:
            return z, objective / float(np.log(2.) * dimensions)
