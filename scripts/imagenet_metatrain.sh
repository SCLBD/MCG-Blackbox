#!/bin/bash
train_dataset_root='/home/yf/adversarial/MetaAttack/imagenet/cgattack/curriculum/curriculum_train_imagenet_sgm_Resnet50_0.55_1.npy'
valid_dataset_root='/home/yf/adversarial/MetaAttack/imagenet/cgattack/data/meta_imagenet'
dataset_name='imagenet'
logroot='./save/imagenet/'
epochs=5
batchsize=16
lr=0.001
adv_loss=True
name='imagenet_metatrain'
model_path='save/imagenet/imagenet_train_forward_di/checkpoints/checkpoint_18000.pth.tar'

support_set='Resnet50'
query_set='Resnet18'
curriculum=False

CUDA_VISIBLE_DEVICES=0 \
python train_imagenet.py \
  --dataset_name=imagenet --train_dataset_root=${train_dataset_root} --valid_dataset_root=${valid_dataset_root} \
  --log_root=${logroot} --x_hidden_channels=64 --y_hidden_channels=256 \
  --x_hidden_size=128 --flow_depth=8 --num_levels=3 --num_epochs=${epochs} --batch_size=${batchsize} \
  --test_gap=10000000 --log_gap=10 --inference_gap=1000000 --lr=${lr} --max_grad_clip=0 \
  --max_grad_norm=10 --save_gap=100  --regularizer=0 --adv_loss=${adv_loss} \
  --learn_top=False --model_path=${model_path} --tanh=False --only=True --margin=5.0 --clamp=True \
  --name=${name} --support_set=${support_set} --query_set=${query_set} --down_sample_x 8 --down_sample_y 8 --meta_iteration=5 \
  --curriculum=${curriculum}

