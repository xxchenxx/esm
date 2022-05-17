CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_full.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 1 --wandb-name 0516_S_seed1_ep10_full_GPU0 --epochs 10  > 0516_S_seed1_ep10_full_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_full.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 2 --wandb-name 0516_S_seed2_ep10_full_GPU1 --epochs 10  > 0516_S_seed2_ep10_full_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_full.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 3 --wandb-name 0516_S_seed3_ep10_full_GPU2 --epochs 10  > 0516_S_seed3_ep10_full_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_full.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 4 --wandb-name 0516_S_seed4_ep10_full_GPU3 --epochs 10  > 0516_S_seed4_ep10_full_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head_regression_full.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --wandb-name 0516_S_r_seed1_ep10_full_GPU4 --epochs 10  > 0516_S_r_seed1_ep10_full_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_regression_full.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 --wandb-name 0516_S_r_seed2_ep10_full_GPU5 --epochs 10  > 0516_S_r_seed2_ep10_full_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_regression_full.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 --wandb-name 0516_S_r_seed3_ep10_full_GPU6 --epochs 10  > 0516_S_r_seed3_ep10_full_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head_regression_full.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 4 --wandb-name 0516_S_r_seed4_ep10_full_GPU7 --epochs 10  > 0516_S_r_seed4_ep10_full_GPU7.out &
