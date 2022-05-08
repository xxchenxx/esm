# CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 --wandb-name Sr_ds_r8_s16 --sparse 16  > 0508_Sr_r8_s16_seed3_GPU0.out &

# CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --wandb-name Sr_ds_r8_s32 --sparse 32  > 0508_Sr_r8_s32_seed1_GPU4.out &

# CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 --wandb-name Sr_ds_r8_s32 --sparse 32  > 0508_Sr_r8_s32_seed2_GPU5.out &

# CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 --wandb-name Sr_ds_r8_s32 --sparse 32  > 0508_Sr_r8_s32_seed3_GPU6.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --wandb-name Sr_ds_r8_s128 --sparse 128  > 0508_Sr_r8_s128_seed1_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --wandb-name Sr_ds_r8_s64 --sparse 64 --adv --gamma 1e-5 > 0508_Sr_r8_s64_seed1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 --wandb-name Sr_ds_r8_s64 --sparse 64 --adv --gamma 1e-5 > 0508_Sr_r8_s64_seed2_GPU3.out &