import pickle
import argparse
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader

from data.datasets import imagenet, cifar10
from utils.load_models import load_imagenet_model, load_cifar_model
from attacks.ifgsm_attack import IFGSM_Based_Attacker


def save_data():
    print("Save data.")
    attack_method = args.attack_method
    if args.dataset == 'imagenet':
        model_name = 'Resnet50'
        model = load_imagenet_model(model_name=model_name)
        dataset = imagenet(root=args.dataroot, mode='train')
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        linf = 0.05
    elif args.dataset == 'cifar10':
        model_name = 'Resnet18'
        model = load_cifar_model(model_name)
        dataset = cifar10(root=args.dataroot, mode='train')
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        linf = 8. / 255.
    else:
        raise NotImplementedError

    train_result = {
        'cln_img': [],
        'adv_img': [],
        'cln_lab': [],
        'adv_lab': [],
        'true_lab': []
    }

    def is_difficult(id):
        return output_label[id] == label[id] and args.difficulty_rate_lower_bound < output_logits[
            id] <= args.difficulty_rate_upper_bound

    data_loader_length = len(data_loader)
    if args.all:
        print('All data including False data.')
    print("Batch size: {}, batch number: {}, total image number {}".format(args.batch_size, data_loader_length,
                                                                           args.batch_size * data_loader_length))
    attacker = IFGSM_Based_Attacker(attack_method=attack_method, dataset=args.dataset, surrogate_model=model, args=None,
                                    linf_limit=linf, iteration=30)
    bar = tqdm.tqdm(data_loader)
    for i, batch in enumerate(bar):
        image, label = batch[0].cuda(), batch[1].cuda()
        logits = model(image)
        output = torch.nn.functional.softmax(logits, dim=1).data
        output_logits, output_label = torch.max(output, dim=1)
        if not args.all:
            difficult_id_list = list(filter(is_difficult, range(0, args.batch_size)))
            if len(difficult_id_list) == 0:
                continue
            image, label = image[difficult_id_list], label[difficult_id_list]
        adv = attacker.perturb(x=image, y=label)

        train_result['cln_img'].append(image.cpu().data)
        train_result['true_lab'].append(label.cpu().data)
        train_result['cln_lab'].append(label.cpu().data)
        train_result['adv_img'].append(adv.cpu().data)
        train_result['adv_lab'].append(label.cpu().data)

        torch.cuda.empty_cache()

    train_result['cln_img'] = torch.cat(train_result['cln_img'], dim=0).numpy()
    train_result['adv_img'] = torch.cat(train_result['adv_img'], dim=0).numpy()
    train_result['true_lab'] = torch.cat(train_result['true_lab'], dim=0).numpy()
    train_result['cln_lab'] = torch.cat(train_result['cln_lab'], dim=0).numpy()
    train_result['adv_lab'] = torch.cat(train_result['adv_lab'], dim=0).numpy()

    path = './data_{}_{}_{}.npy'.format(args.dataset, attack_method, model_name)
    with open(path, "wb") as writer:
        pickle.dump(train_result, writer, protocol=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curriculum data pre-handling')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--attack_method", type=str, default='mi')
    parser.add_argument("--dataroot", type=str)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--difficulty_rate_lower_bound", type=float, default=0.5)
    parser.add_argument("--difficulty_rate_upper_bound", type=float, default=1)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--all", action="store_true", default=False)
    args = parser.parse_args()

    save_data()
