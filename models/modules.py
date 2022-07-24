import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


class ActNorm(nn.Module):

    def __init__(self, num_channels):
        super().__init__()

        size = [1, num_channels, 1, 1]

        bias = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size) * 0.05)
        logs = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size) * 0.05)
        self.register_parameter("bias", nn.Parameter(torch.Tensor(bias), requires_grad=True))
        self.register_parameter("logs", nn.Parameter(torch.Tensor(logs), requires_grad=True))

    def forward(self, input, logdet=0, reverse=False):
        dimentions = input.size(2) * input.size(3)
        if reverse == False:
            input = input + self.bias
            input = input * torch.exp(self.logs)
            dlogdet = torch.sum(self.logs) * dimentions
            logdet = logdet + dlogdet

        if reverse == True:
            input = input * torch.exp(-self.logs)
            input = input - self.bias
            dlogdet = - torch.sum(self.logs) * dimentions
            logdet = logdet + dlogdet

        return input, logdet


class Conv2dZeros(nn.Conv2d):

    def __init__(self, in_channel, out_channel, kernel_size=[3, 3], stride=[1, 1]):
        padding = (kernel_size[0] - 1) // 2
        super().__init__(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                         padding=padding)
        self.weight.data.normal_(mean=0.0, std=0.1)


class Conv2dResize(nn.Conv2d):

    def __init__(self, in_size, out_size):
        stride = [in_size[1] // out_size[1], in_size[2] // out_size[2]]
        kernel_size = Conv2dResize.compute_kernel_size(in_size, out_size, stride)
        super().__init__(in_channels=in_size[0], out_channels=out_size[0], kernel_size=kernel_size, stride=stride)
        self.weight.data.zero_()

    @staticmethod
    def compute_kernel_size(in_size, out_size, stride):
        k0 = in_size[1] - (out_size[1] - 1) * stride[0]
        k1 = in_size[2] - (out_size[2] - 1) * stride[1]
        return [k0, k1]


class Conv2dNorm(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1]):
        padding = (kernel_size[0] - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # initialize weight
        self.weight.data.normal_(mean=0.0, std=0.05)


class CondActNorm(nn.Module):

    def __init__(self, x_size, y_channels, x_hidden_channels, x_hidden_size):
        super().__init__()

        C_x, H_x, W_x = x_size

        # conditioning network
        self.x_Con = nn.Sequential(
            Conv2dResize(in_size=[C_x, H_x, W_x], out_size=[x_hidden_channels, H_x // 2, W_x // 2]),
            nn.ReLU(),
            Conv2dResize(in_size=[x_hidden_channels, H_x // 2, W_x // 2],
                         out_size=[x_hidden_channels, H_x // 4, W_x // 4]),
            nn.ReLU(),
            Conv2dResize(in_size=[x_hidden_channels, H_x // 4, W_x // 4],
                         out_size=[x_hidden_channels, H_x // 8, W_x // 8]),
            nn.ReLU()
        )

        self.x_Linear = nn.Sequential(
            LinearZeros(x_hidden_channels * H_x * W_x // (8 * 8), x_hidden_size),
            nn.ReLU(),
            LinearZeros(x_hidden_size, x_hidden_size),
            nn.ReLU(),
            LinearZeros(x_hidden_size, 2 * y_channels),
            nn.Tanh()
        )

    def forward(self, x, y, logdet=0, reverse=False):

        B, C, H, W = x.size()

        # generate weights
        x = self.x_Con(x)
        x = x.reshape(B, -1)
        x = self.x_Linear(x)
        x = x.reshape(B, -1, 1, 1)

        logs, bias = split_feature(x)
        dimentions = y.size(2) * y.size(3)

        if not reverse:
            # center and scale
            y = y + bias
            y = y * torch.exp(logs)
            dlogdet = dimentions * torch.sum(logs, dim=(1, 2, 3))
            logdet = logdet + dlogdet
        else:
            # scale and center
            y = y * torch.exp(-logs)
            y = y - bias
            dlogdet = - dimentions * torch.sum(logs, dim=(1, 2, 3))
            logdet = logdet + dlogdet

        return y, logdet


class Cond1x1Conv(nn.Module):

    def __init__(self, x_size, x_hidden_channels, x_hidden_size, y_channels):

        super().__init__()

        C_x, H_x, W_x = x_size

        # conditioning network
        self.x_Con = nn.Sequential(
            Conv2dResize(in_size=[C_x, H_x, W_x], out_size=[x_hidden_channels, H_x // 2, W_x // 2]),
            nn.ReLU(),
            Conv2dResize(in_size=[x_hidden_channels, H_x // 2, W_x // 2],
                         out_size=[x_hidden_channels, H_x // 4, W_x // 4]),
            nn.ReLU(),
            Conv2dResize(in_size=[x_hidden_channels, H_x // 4, W_x // 4],
                         out_size=[x_hidden_channels, H_x // 8, W_x // 8]),
            nn.ReLU()
        )

        self.x_Linear = nn.Sequential(
            LinearZeros(x_hidden_channels * H_x * W_x // (8 * 8), x_hidden_size),
            nn.ReLU(),
            LinearZeros(x_hidden_size, x_hidden_size),
            nn.ReLU(),
            LinearNorm(x_hidden_size, y_channels * y_channels),
            nn.Tanh()
        )

    def get_weight(self, x, y, reverse):
        y_channels = y.size(1)
        B, C, H, W = x.size()

        x = self.x_Con(x)
        x = x.reshape(B, -1)
        x = self.x_Linear(x)
        weight = x.reshape(B, y_channels, y_channels)

        dimensions = y.size(2) * y.size(3)

        # dlogdet = torch.slogdet(weight)[1] * dimensions
        # TODO
        dlogdet = torch.stack([torch.slogdet(weight[i])[1] for i in range(weight.shape[0])])
        dlogdet = dlogdet * dimensions
        # dlogdet = []
        # for i in range(B):
        #     dlogdet.append(torch.slogdet(weight[i])[-1].unsqueeze(0) * dimensions)
        # dlogdet = torch.cat(dlogdet, dim=0)

        if reverse == False:
            weight = weight.reshape(B, y_channels, y_channels, 1, 1)

        else:
            weight = torch.inverse(weight.double()).float().reshape(B, y_channels, y_channels, 1, 1)

        return weight, dlogdet

    def forward(self, x, y, logdet=None, reverse=False):

        weight, dlogdet = self.get_weight(x, y, reverse)
        B, C, H, W = y.size()
        y = y.reshape(1, B * C, H, W)
        # y = y.reshape(1, B * C, H, W)
        B_k, C_i_k, C_o_k, H_k, W_k = weight.size()
        assert B == B_k and C == C_i_k and C == C_o_k, "The input and kernel dimensions are different"
        weight = weight.reshape(B_k * C_i_k, C_o_k, H_k, W_k)

        if reverse == False:
            z = F.conv2d(y, weight, groups=B)
            z = z.reshape(B, C, H, W)
            if logdet is not None:
                logdet = logdet + dlogdet

            return z, logdet
        else:
            z = F.conv2d(y, weight, groups=B)
            z = z.reshape(B, C, H, W)

            if logdet is not None:
                logdet = logdet - dlogdet

            return z, logdet


class Conv2dNormy(nn.Conv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1]):
        padding = [(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2]
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=False)

        # initialize weight
        self.weight.data.normal_(mean=0.0, std=0.05)
        self.actnorm = ActNorm(out_channels)

    def forward(self, input):
        x = super().forward(input)
        x, _ = self.actnorm(x)
        return x


class Conv2dZerosy(nn.Conv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1]):
        padding = [(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2]
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)

        self.logscale_factor = 3.0
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.register_parameter("newbias", nn.Parameter(torch.zeros(out_channels, 1, 1)))

        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        output = output + self.newbias
        output = output * torch.exp(self.logs * self.logscale_factor)
        return output


class CondAffineCoupling(nn.Module):

    def __init__(self, x_size, y_size, hidden_channels):
        super().__init__()

        self.resize_x = nn.Sequential(
            Conv2dZeros(x_size[0], 16),
            nn.ReLU(),
            Conv2dResize((16, x_size[1], x_size[2]), out_size=y_size),
            nn.ReLU(),
            Conv2dZeros(y_size[0], y_size[0]),
            nn.ReLU()
        )

        self.f = nn.Sequential(
            Conv2dNormy(y_size[0] * 2, hidden_channels),
            nn.ReLU(),
            Conv2dNormy(hidden_channels, hidden_channels, kernel_size=[1, 1]),
            nn.ReLU(),
            Conv2dZerosy(hidden_channels, 2 * y_size[0]),
            nn.Tanh()
        )

    def forward(self, x, y, logdet=0.0, reverse=False):

        z1, z2 = split_feature(y, "split")
        x = self.resize_x(x)

        h = torch.cat((x, z1), dim=1)
        h = self.f(h)
        shift, scale = split_feature(h, "cross")
        scale = torch.sigmoid(scale + 2.)
        if reverse == False:
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=(1, 2, 3)) + logdet

        if reverse == True:
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=(1, 2, 3)) + logdet

        z = torch.cat((z1, z2), dim=1)

        return z, logdet


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = SqueezeLayer.squeeze2d(input, self.factor)
            return output, logdet
        else:
            output = SqueezeLayer.unsqueeze2d(input, self.factor)
            return output, logdet

    @staticmethod
    def squeeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        B, C, H, W = input.size()
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
        x = input.reshape(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.reshape(B, C * factor * factor, H // factor, W // factor)
        return x

    @staticmethod
    def unsqueeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        B, C, H, W = input.size()
        assert C % (factor2) == 0, "{}".format(C)
        x = input.reshape(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.reshape(B, C // (factor2), H * factor, W * factor)
        return x


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv2dZeros(num_channels // 2, num_channels),
            nn.Tanh()
        )

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, logdet=0., reverse=False, eps_std=None, z2=None):
        if not reverse:
            z1, z2 = split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = GaussianDiag.logp(mean, logs, z2) + logdet

            return z1, z2, logdet
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1)

            if z2 is None:
                z2 = GaussianDiag.sample(mean, logs, eps_std)

            logdet -= GaussianDiag.logp(mean, logs, z2)
            z = torch.cat((z1, z2), dim=1)

            return z, logdet


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        return -0.5 * (logs * 2. + ((x - mean) ** 2.) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return torch.sum(likelihood, dim=(1, 2, 3))

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        # eps = torch.normal(mean=mean,
        #                    std=logs * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def batchsample(batchsize, mean, logs, eps_std=None):
        eps_std = eps_std or 1
        sample = GaussianDiag.sample(mean, logs, eps_std)
        for i in range(1, batchsize):
            s = GaussianDiag.sample(mean, logs, eps_std)
            sample = torch.cat((sample, s), dim=0)
        return sample

    @staticmethod
    def sampleR(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        # eps = torch.normal(mean=torch.zeros_like(mean),
        #                    std=torch.ones_like(logs) * eps_std)
        # print (mean.size(), logs.size())
        eps = torch.normal(mean=mean,
                           std=logs * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def batchsampleR(batchsize, mean, logs, eps_std=None):
        eps_std = eps_std or 1
        sample = GaussianDiag.sampleR(mean, logs, eps_std)
        for i in range(1, batchsize):
            s = GaussianDiag.sampleR(mean, logs, eps_std)
            sample = torch.cat((sample, s), dim=0)
        return sample


class LinearZeros(nn.Linear):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output


class LinearNorm(nn.Linear):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.weight.data.normal_(mean=0.0, std=0.1)
        self.bias.data.normal_(mean=0.0, std=0.1)
