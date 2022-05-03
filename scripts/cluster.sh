CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_0.pkl > d1_0.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_1.pkl > d1_1.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_2.pkl > d1_2.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_3.pkl > d1_3.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_4.pkl > d1_4.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_5.pkl > d1_5.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_6.pkl > d1_6.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_7.pkl > d1_7.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_8.pkl > d1_8.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_9.pkl > d1_9.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 2e-2 --split_file d2_0.pkl > d2_0.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 2e-2 --split_file d2_1.pkl > d2_1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 2e-2 --split_file d2_2.pkl > d2_2.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 2e-2 --split_file d2_3.pkl > d2_3.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 2e-2 --split_file d2_4.pkl > d2_4.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 2e-2 --split_file d2_5.pkl > d2_5.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 2e-2 --split_file d2_6.pkl > d2_6.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 2e-2 --split_file d2_7.pkl > d2_7.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 2e-2 --split_file d2_8.pkl > d2_8.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 2e-2 --split_file d2_9.pkl > d2_9.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 2e-2 --split_file d2_9.pkl > d2_9.out &

nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_0.pkl &

nohup python -u finetune_sup_head_classification_parallel.py esm1b_t33_650M_UR50S data/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1_0_clasification.pkl &