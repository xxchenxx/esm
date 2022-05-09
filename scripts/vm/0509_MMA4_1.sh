CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_0_classification.pkl --seed 1 --wandb-name d2_0_c_sap > 0509_d2_0_c_sap.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_1_classification.pkl --seed 1 --wandb-name d2_1_c_sap > 0509_d2_1_c_sap.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_2_classification.pkl --seed 1 --wandb-name d2_2_c_sap > 0509_d2_2_c_sap.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_3_classification.pkl --seed 1 --wandb-name d2_3_c_sap > 0509_d2_3_c_sap.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_4_classification.pkl --seed 1 --wandb-name d2_4_c_sap > 0509_d2_4_c_sap.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_5_classification.pkl --seed 1 --wandb-name d2_5_c_sap > 0509_d2_5_c_sap.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_6_classification.pkl --seed 1 --wandb-name d2_6_c_sap > 0509_d2_6_c_sap.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_7_classification.pkl --seed 1 --wandb-name d2_7_c_sap > 0509_d2_7_c_sap.out &

