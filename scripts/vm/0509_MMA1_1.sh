CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2  --split_file ~/clean_datasets/d1/d1_0_classification.pkl --seed 1 --wandb-name d1_0_c_sap &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2  --split_file ~/clean_datasets/d1/d1_1_classification.pkl --seed 1 --wandb-name d1_1_c_sap &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2  --split_file ~/clean_datasets/d1/d1_2_classification.pkl --seed 1 --wandb-name d1_2_c_sap &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2  --split_file ~/clean_datasets/d1/d1_3_classification.pkl --seed 1 --wandb-name d1_3_c_sap &

CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2  --split_file ~/clean_datasets/d1/d1_4_classification.pkl --seed 1 --wandb-name d1_4_c_sap &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2  --split_file ~/clean_datasets/d1/d1_5_classification.pkl --seed 1 --wandb-name d1_5_c_sap &

CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2  --split_file ~/clean_datasets/d1/d1_6_classification.pkl --seed 1 --wandb-name d1_6_c_sap &

CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2  --split_file ~/clean_datasets/d1/d1_7_classification.pkl --seed 1 --wandb-name d1_7_c_sap &

