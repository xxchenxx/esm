CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_0_classification.pkl > d2_0_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_1_classification.pkl > d2_1_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_2_classification.pkl > d2_2_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_3_classification.pkl > d2_3_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_4_classification.pkl > d2_4_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_5_classification.pkl > d2_5_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_6_classification.pkl > d2_6_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_7_classification.pkl > d2_7_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_8_classification.pkl > d2_8_head_classification.out  

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_9_classification.pkl > d2_9_head_classification.out  

