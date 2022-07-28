CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file d2/d2_0.pkl --seed 1 --wandb-name d2_0_0_sap > d2_0_sap.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file d2/d2_1.pkl --seed 1 --wandb-name d2_1_1_sap > d2_1_sap.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file d2/d2_2.pkl --seed 1 --wandb-name d2_2_2_sap > d2_2_sap.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file d2/d2_3.pkl --seed 1 --wandb-name d2_3_3_sap > d2_3_sap.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 2e-2  --split_file d1/d1_0.pkl --seed 1 --wandb-name d1_0_0_sap > d1_0_sap.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 2e-2  --split_file d1/d1_1.pkl --seed 1 --wandb-name d1_1_1_sap > d1_1_sap.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 2e-2  --split_file d1/d1_2.pkl --seed 1 --wandb-name d1_2_2_sap > d1_2_sap.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 2e-2  --split_file d1/d1_3.pkl --seed 1 --wandb-name d1_3_3_sap > d1_3_sap.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 2e-2  --split_file d1/d1_4.pkl --seed 1 --wandb-name d1_4_0_sap > d1_4_sap.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 2e-2  --split_file d1/d1_5.pkl --seed 1 --wandb-name d1_5_1_sap > d1_5_sap.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 2e-2  --split_file d1/d1_6.pkl --seed 1 --wandb-name d1_6_2_sap > d1_6_sap.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 2e-2  --split_file d1/d1_7.pkl --seed 1 --wandb-name d1_7_3_sap > d1_7_sap.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file d2/d2_4.pkl --seed 1 --wandb-name d2_4_0_sap > d2_4_sap.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file d2/d2_5.pkl --seed 1 --wandb-name d2_5_1_sap > d2_5_sap.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file d2/d2_6.pkl --seed 1 --wandb-name d2_6_2_sap > d2_6_sap.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file d2/d2_7.pkl --seed 1 --wandb-name d2_7_3_sap > d2_7_sap.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 2e-2  --split_file d1/d1_8.pkl --seed 1 --wandb-name d1_8_0_sap > d1_8_sap.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d1 --lr 2e-2  --split_file d1/d1_9.pkl --seed 1 --wandb-name d1_9_1_sap > d1_9_sap.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file d2/d2_8.pkl --seed 1 --wandb-name d2_8_2_sap > d2_8_sap.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_sap.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2  --split_file d2/d2_7.pkl --seed 1 --wandb-name d2_9_3_sap > d2_9_sap.out &