
CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file S_target_classification.pkl --rank 8 --sparse 32 > r8_1e-2_s32.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file S_target_classification.pkl --rank 8 --sparse 16 > r8_1e-2_s16.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_aadv --lr 1e-2 --split_file S_target_classification.pkl --aadv --lr-factor 100 --gamma 1e-4 > S_classification_aadv.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_0_classification.pkl  > d1_0_c.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_1_classification.pkl  > d1_1_c.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_2_classification.pkl  > d1_2_c.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_3_classification.pkl  > d1_3_c.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_4_classification.pkl  > d1_4_c.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_5_classification.pkl  > d1_5_c.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_6_classification.pkl  > d1_6_c.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_7_classification.pkl  > d1_7_c.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_8_classification.pkl  > d1_8_c.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_9_classification.pkl  > d1_9_c.out &