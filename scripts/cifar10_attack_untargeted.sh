
CUDA_VISIBLE_DEVICES=1

python attack.py \
    --target_model_name='norm_densenet' \
    --dataset_name=cifar10 --dataset_root='../data' \
    --generator_path='checkpoints/cifar10_mcg.pth.tar' \
    --surrogate_model_names=resnet18 \
    --max_query=1000 --class_num=10 --linf=0.0325 \
    --down_sample_x=1 --down_sample_y=1 \
    --attack_method=square --finetune_glow --finetune_reload --finetune_perturbation
