CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 > 0506_Sr_r8_seed2_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 8 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 > 0506_Sr_r8_seed3_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 > 0506_Sr_r16_seed1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 > 0506_Sr_r16_seed2_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 > 0506_Sr_r16_seed3_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 4 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 1 > 0506_S_r4_seed1_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 4 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 2 > 0506_S_r4_seed2_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 4 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 3 > 0506_S_r4_seed1_GPU7.out &