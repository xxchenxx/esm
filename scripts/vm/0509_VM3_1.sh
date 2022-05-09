CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_4.pkl --seed 1 --wandb-name d2_4_s_sap &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_5.pkl --seed 1 --wandb-name d2_5_s_sap &

CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_6.pkl --seed 1 --wandb-name d2_6_s_sap &

CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_7.pkl --seed 1 --wandb-name d2_7_s_sap &