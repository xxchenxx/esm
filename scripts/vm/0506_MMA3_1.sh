# CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 --noise --wandb-name Sr_noise > 0506_Sr_noise_seed3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --adv --gamma 1e-5 --wandb-name Sr_adv > 0506_Sr_adv_seed1_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 --adv --gamma 1e-5 --wandb-name Sr_adv > 0506_Sr_adv_seed2_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 --adv --gamma 1e-5 --wandb-name Sr_adv > 0506_Sr_adv_seed3_GPU3.out &

# CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 4 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --wandb-name Sr_r4 > 0506_Sr_r4_seed1_GPU4.out &

# CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 4 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 --wandb-name Sr_r4 > 0506_Sr_r4_seed2_GPU5.out &

# CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 4 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 --wandb-name Sr_r4 > 0506_Sr_r4_seed1_GPU6.out &

# CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --wandb-name Sr_r8 > 0506_Sr_r8_seed1_GPU7.out &