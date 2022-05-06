CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file d2/d2_1.pkl --rank 16 --lr-factor 10 --sparse 16 > d2_r16_s16_regression_1.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file d2/d2_2.pkl --rank 16 --lr-factor 10 --sparse 16 > d2_r16_s16_regression_2.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file d2/d2_3.pkl --rank 16 --lr-factor 10 --sparse 16 > d2_r16_s16_regression_3.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file d2/d2_4.pkl --rank 16 --lr-factor 10 --sparse 16 > d2_r16_s16_regression_4.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file d2/d2_5.pkl --rank 16 --lr-factor 10 --sparse 16 > d2_r16_s16_regression_5.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file d2/d2_6.pkl --rank 16 --lr-factor 10 --sparse 16 > d2_r16_s16_regression_6.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file d2/d2_7.pkl --rank 16 --lr-factor 10 --sparse 16 > d2_r16_s16_regression_7.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file d2/d2_8.pkl --rank 16 --lr-factor 10 --sparse 16 > d2_r16_s16_regression_8.out 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file d2/d2_9.pkl --rank 16 --lr-factor 10 --sparse 16 > d2_r16_s16_regression_9.out
