# imagenet
CUDA_VISIBLE_DEVICES=1
python attack.py \
    --target_model_name=resnet50 \
    --dataset_name=imagenet --dataset_root='/home/yf/adversarial/MetaAttack/imagenet/cgattack/data/meta_imagenet' \
    --flow_model_path='/home/yf/adversarial/MetaAttack/imagenet/cgattack/save/imagenet/imagenet_train_reverse_mi_18000_clamp_loss/checkpoints/checkpoint_2400.pth.tar' \
    --surrogate_model_names=resnet50 \
    --attack_method=square --test_fasr  --finetune_clean --buffer_limit=1

#--finetune_latent


#    --finetune_perturbation --finetune_mini_batch_size=10
#     --finetune_glow
#'./save/imagenet/imagenet_train_reverse_mi/checkpoints/checkpoint_2300.pth.tar'
#    --start_index=100 --end_index=250
#'./save/imagenet/imagenet_train_reverse_pgd/checkpoints/imagenet_train_reverse_full_data.pth.tar'
#./save/imagenet/imagenet_train_reverse_pgd/checkpoints/imagenet_train_reverse_full_data.pth.tar

# --finetune_glow_load --defence_method=jpeg_compression

#--flow_model_path='./checkpoints/imagenet/imagenet_train_reverse_full_data.pth.tar' \
    # --finetune_perturbation --finetune_mini_batch_size=4

# Resnet18, VGG16, wrn50, InceptionV3, deno
#'save/imagenet/imagenet_train_reverse/checkpoints/checkpoint_2500.pth.tar'
#--flow_model_path='./checkpoints/imagenet/imagenet_train_reverse_full_data.pth.tar' \
    #--finetune_coefficient
    #--finetune_perturbation --finetune_mini_batch_size=20

    #--finetune_coefficient
    # --load_surrogate_models
    # --finetune_perturbation --finetune_mini_batch_size=20
