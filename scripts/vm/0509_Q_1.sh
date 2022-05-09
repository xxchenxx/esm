CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_8_classification.pkl --seed 1 --wandb-name d2_8_c_sap &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file ~/clean_datasets/d2/d2_9_classification.pkl --seed 1 --wandb-name d2_9_c_sap &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2  --split_file ~/clean_datasets/d1/d1_8_classification.pkl --seed 1 --wandb-name d1_8_c_sap &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2  --split_file ~/clean_datasets/d1/d1_9_classification.pkl --seed 1 --wandb-name d1_9_c_sap &