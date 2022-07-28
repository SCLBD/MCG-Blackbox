import h5py
import os
import numpy as np
import torch
import PIL.Image as Image
from torch.utils import data
import torchvision
import torch.nn as nn
import torch.utils.data
import json
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class AdvTrainDataset(Dataset):
    def __init__(self, root_dir):
        print('Load imagenet dataset from:', root_dir)
        self.root_dir = root_dir
        self.transform = transforms.Compose([])
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        try:
            dataset = np.load(root_dir, allow_pickle=True).item()
        except:
            dataset = np.load(root_dir, allow_pickle=True)

        self.cln_imgs = dataset['cln_img']
        self.cln_labs = dataset['cln_lab']
        self.adv_imgs = dataset['adv_img']
        self.adv_labs = dataset['adv_lab']
        self.true_labs = dataset['true_lab']

        self.num = len(self.cln_imgs)
        print('data load done.')

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        cln_img, adv_img = self.cln_imgs[idx].copy(), self.adv_imgs[idx].copy()
        cln_lab, adv_lab, true_lab = self.cln_labs[idx], self.adv_labs[idx], self.true_labs[idx]
        # transform
        if self.transform is not None:
            # print(adv_img.shape)
            adv_img = self.transform(adv_img)
            cln_img = self.transform(cln_img)
            # adv_img = (adv_img - cln_img) not efficient here using minor
            # adv_img = ((adv_img - adv_img.min()) / (adv_img.max() - adv_img.min()) - 0.5) * 2
        return {
            "adv_img": adv_img,
            "cln_img": cln_img,
            "cln_lab": cln_lab,
            'true_lab': true_lab,
            "adv_lab": adv_lab
        }


def imagenet(root, mode='train'):
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    root = os.path.join(root, mode)
    if mode == 'train':
        dataset = datasets.ImageFolder(root, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # note we only normalize before the model predicting process
            # normalize,
        ]))
    elif mode == 'validation':
        dataset = datasets.ImageFolder(root, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ]))
    else:
        raise NotImplementedError
    return dataset


def cifar10(root, mode='train'):
    if mode == 'train':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform_train)
    elif mode == 'validation':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        valid_set = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform_test)
        # Select 100 images from each class in CIFAR10
        # Note, this is the valid dataset pre_number == 100 cursor
        pre_100_cur = [960, 1100, 1000, 960, 1140, 1200, 900, 960, 950, 930]
        idx = torch.zeros(10000).bool()
        for i, pre_cur in enumerate(pre_100_cur):
            idx_i = torch.tensor(valid_set.targets) == i
            idx_i[pre_cur:] = False
            idx += idx_i
        dataset = torch.utils.data.dataset.Subset(valid_set, np.where(idx == 1)[0])
    else:
        raise NotImplementedError
    return dataset
