CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_4.pkl > d1_4_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_5.pkl > d1_5_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_6.pkl > d1_6_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_7.pkl > d1_7_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_8.pkl > d1_8_no_mix_dsee_noisy_last_two.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_9.pkl > d1_9_no_mix_dsee_noisy_last_two.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_4.pkl > d2_4_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_5.pkl > d2_5_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_6.pkl > d2_6_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_7.pkl > d2_7_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_0.pkl > d1_0_no_mix_dsee_noisy_last_two.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_1.pkl > d1_1_no_mix_dsee_noisy_last_two.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_2.pkl > d1_2_no_mix_dsee_noisy_last_two.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_3.pkl > d1_3_no_mix_dsee_noisy_last_two.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_4.pkl > d1_4_no_mix_dsee_noisy_last_two.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_5.pkl > d1_5_no_mix_dsee_noisy_last_two.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_4.pkl > d2_4_no_mix_dsee_noisy_last_two.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_5.pkl > d2_5_no_mix_dsee_noisy_last_two.out &
