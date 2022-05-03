CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 1e-2 --split_file split_2.pkl > l2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 1e-2 --split_file split_3.pkl > l3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 1e-2 --split_file split_4.pkl > l4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 1e-2 --split_file split_5.pkl > l5.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 1e-2 --split_file split_5.pkl > l5.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 1e-2 --split_file split_6.pkl > l6.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 1e-2 --split_file split_7.pkl > l7.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 1e-2 --split_file split_8.pkl > l8.out &





CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file split_2.pkl > l2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 2e-2 --split_file split_3.pkl > l3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 2e-2 --split_file split_4.pkl > l4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 2e-2 --split_file split_5.pkl > l5.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 2e-2 --split_file split_6.pkl > l6.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 2e-2 --split_file split_7.pkl > l7.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 2e-2 --split_file split_8.pkl > l8.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 2e-2 --split_file split_9.pkl > l9.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 5e-3 --split_file split_6.pkl > l6.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 5e-3 --split_file split_7.pkl > l7.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 5e-3 --split_file split_8.pkl > l8.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 5e-3 --split_file split_9.pkl > l9.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 5e-3 --split_file split_0.pkl > l0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 2e-2 --split_file split_1.pkl > l1.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 5e-3 --split_file split_2.pkl > l2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 5e-3 --split_file split_3.pkl > l3.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 5e-3 --split_file split_4.pkl > l4.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 5e-3 --split_file split_5.pkl > l5.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file split_0.pkl > l0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file split_1.pkl > l1.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file split_2.pkl > l2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file split_3.pkl > l3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file split_6.pkl > l6.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file split_7.pkl > l7.ou




CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_0.pkl > m0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_1.pkl > m1.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_2.pkl > m2.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_6.pkl > m6.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_7.pkl > m7.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_0.pkl > m0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_1.pkl > m1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_4.pkl > m4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_5.pkl > m5.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_6.pkl > m6.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_7.pkl > m7.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-5 --split_file split_6.pkl > m6.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-5 --split_file split_7.pkl > m7.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_0.pkl > l0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_1.pkl > l1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_6.pkl > l6.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_7.pkl > l7.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_8.pkl > l8.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_9.pkl > l9.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_8.pkl > l8.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_9.pkl > l9.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup.py esm1b_t33_650M_UR50S sampled sup --include mean per_tok --split_file split_0.pkl --toks_per_batch 2048 --num_classes 2 --idx sparse --lr 1e-6 > debug1.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup.py esm1b_t33_650M_UR50S sampled sup --include mean per_tok --split_file split_1.pkl --toks_per_batch 2048 --num_classes 2 --idx sparse --lr 1e-6 > debug2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_8.pkl --mix > 8.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_6.pkl --mix > 6.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_7.pkl --mix > 7.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_9.pkl --mix > 9.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_8.pkl --mix > 8.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_6.pkl > 6_no_mix.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_7.pkl > 7_no_mix.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_9.pkl > 9_no_mix.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_8.pkl > 8_no_mix.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_6.pkl > 6_no_mix_dsee.out.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_7.pkl > 7_no_mix_dsee.out.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_9.pkl > 9_no_mix_dsee.out.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_8.pkl > 8_no_mix_dsee.out.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_6.pkl > 6_no_mix.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_8.pkl > 8_no_mix.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_7.pkl > 7_no_mix.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_9.pkl > 9_no_mix.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_0.pkl > 0_no_mix.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_1.pkl > 1_no_mix.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_0.pkl --mix > 0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_1.pkl --mix > 1.out &





CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_6.pkl > d1_6_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_7.pkl > d1_7_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_9.pkl > d1_9_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_8.pkl > d1_8_no_mix_dsee.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_2.pkl > d1_2_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_3.pkl > d1_3_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_4.pkl > d1_4_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_5.pkl > d1_5_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_0.pkl > d1_0_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_1.pkl > d1_1_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_0.pkl > 0_no_mix_dsee.out.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_2.pkl > d1_2_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_3.pkl > d1_3_no_mix_dsee.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_4.pkl > d1_4_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_5.pkl > d1_5_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_6.pkl > d1_6_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_7.pkl > d1_7_no_mix_dsee.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_8.pkl > d1_8_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_9.pkl > d1_9_no_mix_dsee.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_8.pkl > d2_8_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_9.pkl > d2_9_no_mix_dsee.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_1.pkl > 1_no_mix_dsee.out.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_6.pkl > d2_6_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_7.pkl > d2_7_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_5.pkl > d2_5_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_4.pkl > d2_4_no_mix_dsee.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_3.pkl > d2_3_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_2.pkl > d2_2_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_1.pkl > d2_1_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_0.pkl > d2_0_no_mix_dsee.out &




CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_2.pkl > 2_no_mix.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_3.pkl > 3_no_mix.out &




CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_5.pkl > d2_5_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_4.pkl > d2_4_no_mix_dsee.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_3.pkl > d2_3_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_2.pkl > d2_2_no_mix_dsee.out &




CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_2.pkl > 2_no_mix_dsee.out.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_3.pkl > 3_no_mix_dsee.out.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_4.pkl > 4_no_mix_dsee.out.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_5.pkl > 5_no_mix_dsee.out.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_0.pkl > S_0_no_mix_dsee.out.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_1.pkl > S_1_no_mix_dsee.out.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_2.pkl > S_2_no_mix_dsee.out.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_3.pkl > S_3_no_mix_dsee.out.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_3.pkl > d2_3_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_2.pkl > d2_2_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_1.pkl > d2_1_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_0.pkl > d2_0_no_mix_dsee.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_3.pkl > d1_3_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_2.pkl > d1_2_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_1.pkl > d1_1_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_0.pkl > d1_0_no_mix_dsee.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_4.pkl > d1_4_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_5.pkl > d1_5_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_6.pkl > d1_6_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_7.pkl > d1_7_no_mix_dsee.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_8.pkl > d1_8_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_9.pkl > d1_9_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_8.pkl > d2_8_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_9.pkl > d2_9_no_mix_dsee.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_adaptor_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_3.pkl > d2_3_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_adaptor_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_2.pkl > d2_2_no_mix_dsee.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_1.pkl > d2_1_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_0.pkl > d2_0_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_2.pkl > d2_2_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_3.pkl > d2_3_head.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_4.pkl > d2_4_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_5.pkl > d2_5_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_6.pkl > d2_6_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_7.pkl > d2_7_head.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_1.pkl --mix > d2_1_mix_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_0.pkl --mix > d2_0_mix_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_2.pkl --mix > d2_2_mix_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_3.pkl --mix > d2_3_mix_head.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_4.pkl --mix > d2_4_mix_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_5.pkl --mix > d2_5_mix_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_6.pkl --mix > d2_6_mix_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_7.pkl --mix > d2_7_mix_head.out &





CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_4.pkl > d2_4_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_5.pkl > d2_5_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_6.pkl > d2_6_no_mix_dsee.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_7.pkl > d2_7_no_mix_dsee.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_4.pkl --mix > d1_4_mix_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_5.pkl --mix > d1_5_mix_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_6.pkl --mix > d1_6_mix_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_7.pkl --mix > d1_7_mix_head.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_0.pkl --mix > d1_0_mix_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_1.pkl --mix > d1_1_mix_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_2.pkl --mix > d1_2_mix_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_3.pkl --mix > d1_3_mix_head.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_8.pkl --mix > d1_8_mix_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_9.pkl --mix > d1_9_mix_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_8.pkl --mix > d2_8_mix_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_9.pkl --mix > d2_9_mix_head.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_0.pkl > d1_0_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_1.pkl > d1_1_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_2.pkl > d1_2_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_3.pkl > d1_3_head.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_4.pkl > d1_4_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_5.pkl > d1_5_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_6.pkl > d1_6_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_7.pkl > d1_7_head.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_0.pkl > d2_0_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_1.pkl > d2_1_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_2.pkl > d2_2_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_3.pkl > d2_3_head.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_8.pkl > d1_8_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_9.pkl > d1_9_head.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_8.pkl > d2_8_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d2_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d2_9.pkl > d2_9_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_0.pkl > S_0_head.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_1.pkl > S_1_head.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_2.pkl > S_2_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_3.pkl > S_3_head.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_4.pkl > S_4_head.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_5.pkl > S_5_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_6.pkl > S_6_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_7.pkl > S_7_head.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_8.pkl > S_8_head.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_9.pkl > S_9_head.out &