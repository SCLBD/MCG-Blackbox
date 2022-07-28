# imagenet
CUDA_VISIBLE_DEVICES=0

python attack.py \
    --target_model_name=resnet18 \
    --dataset_name=imagenet --dataset_root='./data/meta_imagenet' \
    --generator_path='checkpoints/imagenet_mcg.pth.tar' \
    --surrogate_model_names=resnet50 \
    --max_query=1000 --class_num=1000 --linf=0.05 \
    --attack_method=square --finetune_glow --finetune_reload --finetune_perturbation
