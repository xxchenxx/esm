CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_0_classification.pkl > d1_0_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_1_classification.pkl > d1_1_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_2_classification.pkl > d1_2_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_3_classification.pkl > d1_3_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_4_classification.pkl > d1_4_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_5_classification.pkl > d1_5_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_6_classification.pkl > d1_6_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_7_classification.pkl > d1_7_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_8_classification.pkl > d1_8_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_9_classification.pkl > d1_9_head_classification.out  

