import os
import torch
import numpy as np

import attacks
import data.datasets as datasets
from utils.attack_count import AttackCountingFunction
from utils.load_models import load_generator, load_imagenet_model, load_cifar_model
from utils.buffer import AttackBuffer, AttackListBuffer
from utils.surrogate_trainer import TrainModelSurrogate


def seed_init():
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)


def data_init(args):
    if args.dataset_name == 'imagenet':
        valid_set = datasets.imagenet(args.dataset_root, mode="validation")
        args.x_size = (3, 224, 224)
        args.y_size = (3, 224, 224)
    elif args.dataset_name == 'cifar10':
        valid_set = datasets.cifar10(args.dataset_root, mode="validation")
        args.x_size = (3, 32, 32)
        args.y_size = (3, 32, 32)
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    return dataloader


def model_init(args):
    if args.dataset_name == 'imagenet':
        load_model = load_imagenet_model
    elif args.dataset_name == 'cifar10':
        load_model = load_cifar_model
    else:
        raise NotImplementedError

    # Target model
    T = load_model(args.target_model_name, defence_method=args.defence_method)
    T.eval()

    surrogates, surrogate_optims = [], []

    for surrogate_names in args.surrogate_model_names.split(','):
        surrogate_model, optim = load_model(surrogate_names, require_optim=True)
        # surrogate_model.train()
        surrogates.append(surrogate_model)
        surrogate_optims.append(optim)
    # Counting function
    F = AttackCountingFunction(args.max_query)
    # MCG Generator
    G = load_generator(args)
    return T, G, surrogates, surrogate_optims, F


def attacker_init(args):
    dataset = args.dataset_name
    max_query = args.max_query
    targeted = args.targeted
    class_num = 1000 if dataset == 'imagenet' else 10
    linf = 0.05 if dataset == 'imagenet' else 8. / 255
    args.class_num = class_num
    args.linf = linf

    if args.attack_method == 'square':
        attacker = attacks.SquareAttack(dataset_name=dataset, max_query=max_query, targeted=targeted, class_num=class_num, linf=linf)
    elif args.attack_method == 'signhunter':
        attacker = attacks.SignHunter(dataset_name=dataset, max_query=max_query, targeted=targeted, class_num=class_num, linf=linf)
    # elif args.attack_method == 'cg':
    #     # attacker = attacks.C(dataset_name=args.dataset_name, max_query=max_query, targeted=args.targeted, class_num=class_num, popsize=20, if_latent=True, linf_limit=linf)
    else:
        raise NotImplementedError
    return attacker


def buffer_init(args):
    mini_batch_size = args.finetune_mini_batch_size
    attack_method = args.attack_method
    buffer_limit = args.buffer_limit

    image_buffer = AttackBuffer(batch_size=mini_batch_size)
    clean_buffer = AttackBuffer(batch_size=mini_batch_size)
    adv_buffer = AttackListBuffer(attack_method=attack_method, batch_size=mini_batch_size, uplimit=buffer_limit)
    return image_buffer, clean_buffer, adv_buffer


def trainer_init(args):
    trainer = TrainModelSurrogate()
    return trainer


def log_init(args):
    if args.log_root is not None:
        log_path = args.log_root
    else:
        os.makedirs('./logs', exist_ok=True)
        targeted = 'T' if args.targeted else 'UT'
        log_path = f'./logs/{args.dataset_name}_{targeted}_{args.target_model_name}_{args.attack_method}'
    return log_path
