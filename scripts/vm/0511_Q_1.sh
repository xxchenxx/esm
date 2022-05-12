CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 1 --wandb-name S_ds_r8_s64_mixup --mixup > 0511_S_r8_s64_mixup_seed1_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 2 --wandb-name S_ds_r8_s64_mixup --mixup > 0511_S_r8_s64_mixup_seed2_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 3 --wandb-name S_ds_r8_s64_mixup --mixup > 0511_S_r8_s64_mixup_seed3_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 4 --wandb-name S_ds_r8_s64_mixup --mixup > 0511_S_r8_s64_mixup_seed4_GPU3.out &