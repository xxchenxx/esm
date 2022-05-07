CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_9_classification.pkl --lr-factor 10 --rank 4 --sparse 16 > d2_9_rank4_s16_ssf.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_8_classification.pkl --lr-factor 10 --rank 4 --sparse 16 > d2_8_rank4_s16_ssf.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_7_classification.pkl --lr-factor 10 --rank 4 --sparse 16 > d2_7_rank4_s16_ssf.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_6_classification.pkl --lr-factor 10 --rank 4 --sparse 16 > d2_6_rank4_s16_ssf.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_5_classification.pkl --lr-factor 10 --rank 4 --sparse 16 > d2_5_rank4_s16_ssf.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_4_classification.pkl --lr-factor 10 --rank 4 --sparse 16 > d2_4_rank4_s16_ssf.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_3_classification.pkl --lr-factor 10 --rank 4 --sparse 16 > d2_3_rank4_s16_ssf.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_2_classification.pkl --lr-factor 10 --rank 4 --sparse 16 > d2_2_rank4_s16_ssf.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_1_classification.pkl --lr-factor 10 --rank 4 --sparse 16 > d2_1_rank4_s16_ssf.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_0_classification.pkl --lr-factor 10 --rank 4 --sparse 16 > d2_0_rank4_s16_ssf.out 