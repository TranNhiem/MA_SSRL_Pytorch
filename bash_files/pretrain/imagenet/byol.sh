python3 ../../../main_pretrain.py \
    --dataset imagenet \
    --backbone resnet50 \
    --data_dir /data1/1K_New \
    --train_dir train \
    --val_dir val \
    --max_epochs 300 \
    --gpus 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --accumulate_grad_batches 1 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 512 \
    --num_workers 8 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name byol-resnet50-imagenet-300ep \
    --entity tranrick \
    --project solo_MASSL \
    --wandb \
    --save_checkpoint \
    --method byol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier \
    --checkpoint_dir /data1/solo_baseline_ckpt \
    --dali \
    --checkpoint_frequency 10
