import argparse
import ast
import copy
import sys
import torch

import utils.attack_init as attack_init
from attacks.base_attack import margin_loss_interface
from models.flow_latent import latent_operate, latent_initialize, generate_interface
from utils.finetune import finetune_latent, meta_finetune


def attack():
    attack_init.seed_init()
    dataloader = attack_init.data_init(args)
    T, G, surrogates, surrogate_optims, F = attack_init.model_init(args)
    _G = copy.deepcopy(G)
    attacker = attack_init.attacker_init(args)
    image_buffer, clean_buffer, adv_buffer = attack_init.buffer_init(args)
    trainer = attack_init.trainer_init(args)
    log_path = attack_init.log_init(args)
    loss_function = margin_loss_interface(T, class_num=args.class_num)
    generate_function = generate_interface(G, latent_operate, args.linf)
    mini_batch_size = args.finetune_mini_batch_size

    for i, (images, labels) in enumerate(dataloader):
        with torch.no_grad():
            images, labels = images.cuda(), int(labels)
            logits = torch.nn.functional.softmax(T(images), dim=1)

        correct = torch.argmax(logits, dim=1) == labels
        if not correct:
            continue

        image_buffer.add(images, labels, logits=logits, score=float(logits[:, labels].item()))

        if args.finetune_clean:
            full = clean_buffer.add(images, labels, logits)
            if full:
                batch_images, batch_logits, batch_labels = clean_buffer.make_batch()
                for idx in range(len(surrogates)):
                    trainer.forward_loss(surrogates[idx], surrogate_optims[idx], batch_images, batch_logits, batch_labels)
                clean_buffer.clear()

    clean_buffer.clear()
    print('Image buffer length: ', len(image_buffer.clean_images))

    for _i in range(len(image_buffer.clean_images)):
        torch.cuda.empty_cache()
        images = image_buffer.clean_images[_i].unsqueeze(0).cuda()
        labels = image_buffer.labels[_i]
        labels_tensor = torch.tensor(labels).view(-1).cuda()
        clean_logits = image_buffer.clean_logits[_i].cuda()

        success, query_cnt = False, 0
        if args.targeted:
            if labels == args.target_label:
                continue
            labels = args.target_label

        if args.finetune_perturbation:
            # Make the fine-tuning robust with clean images
            clean_batch_images, clean_batch_logits, clean_batch_labels = image_buffer.sample_batch(mini_batch_size - 1)

            clean_batch_images = torch.cat([clean_batch_images, images], dim=0)
            clean_batch_logits = torch.cat([clean_batch_logits, clean_logits.unsqueeze(0)], dim=0)
            clean_batch_labels = torch.cat([clean_batch_labels, labels_tensor], dim=0)

            for idx in range(len(surrogates)):
                trainer.forward_loss(
                    surrogates[idx], surrogate_optims[idx], clean_batch_images, clean_batch_logits, clean_batch_labels)

            if adv_buffer.length() > mini_batch_size:
                # Batch: images, logits, labels
                current_batch = (images, clean_logits.unsqueeze(0), labels)
                perturbation_batch = adv_buffer.sample_batch(mini_batch_size)
                for idx in range(len(surrogates)):
                    trainer.lifelong_forward_loss(
                        surrogates[idx], surrogate_optims[idx], perturbation_batch, current_batch)

        if args.finetune_perturbation:
            adv_buffer.add_clean(images, clean_logits, labels)

        latent, _ = latent_initialize(images, G, latent_operate)
        if args.finetune_latent:
            latent, _ = finetune_latent(G, surrogates, images, labels, latent, args)

        if args.finetune_glow:
            meta_finetune(G, surrogates, images, labels, latent, args, meta_iteration=2)

        # First attack attempt
        perturbation = generate_function(images, latent)
        adv_images = torch.clamp(images + perturbation.view(images.shape), 0., 1.)
        loss_output = loss_function(adv_images, labels, targeted=args.targeted)
        query_cnt += 1
        if loss_output['margin'] <= 0:
            success = True

        if not args.test_fasr and not success:
            if args.attack_method in ['cgattack']:
                generator_loss_function = attacker.generator_loss_interface(generate_function, loss_function, args.targeted)
                attack_output = attacker.attack(generator_loss_function, images, labels, init=None, buffer=adv_buffer, latent=latent)
            else:
                attack_output = attacker.attack(loss_function, images, labels, init=perturbation, buffer=adv_buffer)
            query_cnt += attack_output['query_cnt']
            success = attack_output['success']

            if args.finetune_perturbation:
                adv = attack_output['adv']
                logits = attack_output['logits_best']
                adv_buffer.add(adv, logits)

        F.add(query_cnt, success)
        log = f'image: {_i} query_cnt: {query_cnt} success: {success} Mean: {F.get_average()} Median: {F.get_median()} FASR: {F.get_first_success()} ASR: {F.get_success_rate()}\n'
        if not args.mute:
            print(log)
            sys.stdout.flush()
        with open(log_path, 'a') as f:
            f.write(log)

        if args.finetune_reload:
            G = copy.deepcopy(_G)

    final_log = f'Final Log with attack finished:\n' \
                f'Valid image number: {F.count_total}\n' \
                f'Target model: {args.target_model_name} Surrogate models: {args.surrogate_model_names}\n' \
                f'ASR: {F.get_success_rate()}\n' \
                f'FASR: {F.get_first_success()}\n' \
                f'MEAN: {F.get_average()}\n' \
                f'MEDIAN: {F.get_median()}\n'
    args_log = ''
    for arg in vars(args):
        args_log += f'{arg}: {getattr(args, arg)}\n'

    if not args.mute:
        print(final_log)
        print(args_log)

    with open(log_path, 'a') as f:
        f.write(final_log)
        f.write(args_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack')

    # input output path
    parser.add_argument("-d", "--dataset_name", type=str)
    parser.add_argument("-r", "--dataset_root", type=str)
    parser.add_argument("--generator_path", type=str, default="")
    parser.add_argument("--target_model_name", type=str)
    parser.add_argument("--surrogate_model_names", type=str)
    parser.add_argument("--buffer_limit", type=int, default=1)
    parser.add_argument("--attack_method", type=str, help='square, signhunter, cgattack')
    parser.add_argument("--defence_method", type=str, help='snd, jpeg', default=None)

    parser.add_argument("--finetune_clean", action="store_true")
    parser.add_argument("--finetune_perturbation", action="store_true")
    parser.add_argument("--finetune_glow", action="store_true")
    parser.add_argument("--finetune_reload", action="store_true")
    parser.add_argument("--finetune_latent", action="store_true")
    parser.add_argument("--test_fasr", action="store_true")
    parser.add_argument("--finetune_mini_batch_size", type=int, default=20)

    parser.add_argument("--max_query", type=int, default=10000)
    parser.add_argument("--class_num", type=int, default=1000)
    parser.add_argument("--linf", type=float, default=0.05)
    parser.add_argument("--target_label", type=int, default=1)

    # log root
    parser.add_argument("--log_root", type=str, default=None)
    parser.add_argument("--mute", action="store_true")
    # C-Glow parameters
    parser.add_argument("--x_size", type=tuple, default=(3, 224, 224))
    parser.add_argument("--y_size", type=tuple, default=(3, 224, 224))
    parser.add_argument("--x_hidden_channels", type=int, default=64)
    parser.add_argument("--x_hidden_size", type=int, default=128)
    parser.add_argument("--y_hidden_channels", type=int, default=256)
    parser.add_argument("-K", "--flow_depth", type=int, default=8)
    parser.add_argument("-L", "--num_levels", type=int, default=3)
    parser.add_argument("--learn_top", type=ast.literal_eval, default=False)

    # Dataset preprocess parameters
    parser.add_argument("--label_scale", type=float, default=1)
    parser.add_argument("--label_bias", type=float, default=0.0)
    parser.add_argument("--x_bins", type=float, default=256.0)
    parser.add_argument("--y_bins", type=float, default=2.0)

    # Optimizer parameters
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.9999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--regularizer", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--Lambda", type=float, default=1e-2)

    # Trainer parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--down_sample_x", type=int, default=8)
    parser.add_argument("--down_sample_y", type=int, default=8)
    parser.add_argument("--max_grad_clip", type=float, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--checkpoints_gap", type=int, default=1000)
    parser.add_argument("--nll_gap", type=int, default=1)
    parser.add_argument("--inference_gap", type=int, default=1000)
    parser.add_argument("--save_gap", type=int, default=1000)
    parser.add_argument("--adv_loss", type=ast.literal_eval, default=False)
    parser.add_argument("--targeted", type=ast.literal_eval, default=False)
    parser.add_argument("--tanh", type=ast.literal_eval, default=False)
    parser.add_argument("--only", type=ast.literal_eval, default=False)
    parser.add_argument("--partial", type=ast.literal_eval, default=False)
    parser.add_argument("--rand", type=ast.literal_eval, default=False)

    parser.add_argument("--clamp", type=ast.literal_eval, default=False)
    parser.add_argument("--class_size", type=int, default=-1)
    parser.add_argument("--label", type=int, default=0)

    # Adv augmentation
    parser.add_argument("--adv_aug", type=ast.literal_eval, default=False)
    parser.add_argument("--adv_rand", type=ast.literal_eval, default=False)
    parser.add_argument("--nes", type=ast.literal_eval, default=False)
    parser.add_argument("--new_form", type=ast.literal_eval, default=False)
    parser.add_argument("--normalize_grad", type=ast.literal_eval, default=False)
    parser.add_argument("--adv_epoch", type=int, default=0)

    args = parser.parse_args()
    attack()
