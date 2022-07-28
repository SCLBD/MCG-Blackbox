import os
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def preprocess(x, scale, bias, bins, noise=False):

    x = x / scale
    x = x - bias
    if noise:
        if bins == 2:
            x = x + torch.zeros_like(x).uniform_(-0.5, 0.5)
        else:
            x = x + torch.zeros_like(x).uniform_(0, 1/bins)
    return x

def save_model(model, optim, dir, iteration):
    path = os.path.join(dir, "checkpoint_{}.pth.tar".format(iteration))
    state = {}
    state["iteration"] = iteration
    state["modelname"] = model.__class__.__name__
    state["model"] = model.state_dict()
    state["optim"] = optim.state_dict()
    # if scheduler is not None:
    #     state["scheduler"] = scheduler.state_dict()
    # else:
    #     state["scheduler"] = None
    # if latent is not None:
    #     state["latent"] = latent
    torch.save(state, path)

