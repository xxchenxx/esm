
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_5_classification.pkl --adv --gamma 1e-5 > d1_5_advc.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_1_classification.pkl --adv --gamma 1e-5 > d1_1_advc.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_3_classification.pkl --adv --gamma 1e-5 > d1_3_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_7_classification.pkl --adv --gamma 1e-5 > d1_7_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_9_classification.pkl --adv --gamma 1e-5 > d1_9_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_1.pkl --adv --gamma 1e-5 > d1_1_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_3.pkl --adv --gamma 1e-5 > d1_3_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_5.pkl --adv --gamma 1e-5 > d1_5_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_7.pkl --adv --gamma 1e-5 > d1_7_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_9.pkl --adv --gamma 1e-5 > d1_9_advr.out &



CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_1.pkl --rank 4 --lr-factor 10  > d1_1_ds_r.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_3.pkl --rank 4 --lr-factor 10  > d1_3_ds_r.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_5.pkl --rank 4 --lr-factor 10  > d1_5_ds_r.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_7.pkl --rank 4 --lr-factor 10  > d1_7_ds_r.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_0_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_0_ds_advc.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_1_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_1_ds_advc.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_2_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_2_ds_advc.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_3_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_3_ds_advc.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_6_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_6_ds_advc.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_7_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_7_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_0_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_0_ds_advc.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_5_classification.pkl --rank 4 --lr-factor 10  > d1_5_ds_c.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_4_classification.pkl --rank 4 --lr-factor 10  > d1_4_ds_c.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_3_classification.pkl --rank 4 --lr-factor 10  > d1_3_ds_c.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_2_classification.pkl --rank 4 --lr-factor 10  > d1_2_ds_c.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_1_classification.pkl --rank 4 --lr-factor 10  > d1_1_ds_c.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_0_classification.pkl --rank 4 --lr-factor 10  > d1_0_ds_c.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_6_classification.pkl --rank 4 --lr-factor 10  > d1_6_ds_c.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_7_classification.pkl --rank 4 --lr-factor 10  > d1_7_ds_c.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_8_classification.pkl --rank 4 --lr-factor 10  > d1_8_ds_c.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_9_classification.pkl --rank 4 --lr-factor 10  > d1_9_ds_c.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_2_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_2_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_3_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_3_ds_advc.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_2_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_2_ds_advc.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_3_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_3_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_4_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_4_ds_advc.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_2_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_2_ds_advc.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_3_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_3_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_4_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_4_ds_advc.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_7.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_7_ds_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_8.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_8_ds_advr.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_7.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 --epochs 10 > d1_7_ds_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_8.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 --epochs 10 > d1_8_ds_advr.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_2.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 --epochs 10 > d1_2_ds_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_3.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 --epochs 10 > d1_3_ds_advr.out &


CUDA_VISIBLE_DEVICES=2 nohup python if1_d1.py 0 > d1_0_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d1.py 1 > d1_1_if1.out &
CUDA_VISIBLE_DEVICES=1 nohup python if1_d1.py 2 > d1_2_if1.out &

CUDA_VISIBLE_DEVICES=1 nohup python if1_d1.py 2 > d1_3_if1.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d1.py 4 > d1_4_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d1.py 5 > d1_5_if1.out &
CUDA_VISIBLE_DEVICES=1 nohup python if1_d1.py 6 > d1_6_if1.out &

CUDA_VISIBLE_DEVICES=2 nohup python if1_d1.py 7 > d1_7_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d1.py 8 > d1_8_if1.out &
CUDA_VISIBLE_DEVICES=1 nohup python if1_d1.py 9 > d1_9_if1.out &

CUDA_VISIBLE_DEVICES=2 nohup python if1_d2.py 0 > d2_0_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d2.py 1 > d2_1_if1.out &

CUDA_VISIBLE_DEVICES=1 nohup python if1_d2.py 2 > d2_2_if1.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d2.py 3 > d2_3_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d2.py 4 > d2_4_if1.out &

CUDA_VISIBLE_DEVICES=1 nohup python if1_d2.py 5 > d2_5_if1.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d2.py 6 > d2_6_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d2.py 7 > d2_7_if1.out &

CUDA_VISIBLE_DEVICES=1 nohup python if1_d2.py 8 > d2_8_if1.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d2.py 9 > d2_9_if1.out &

CUDA_VISIBLE_DEVICES=1 nohup python if1_d1_regression.py 0 > d1_r0_if1.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d1_regression.py 1 > d1_r1_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d1_regression.py 2 > d1_r2_if1.out &
CUDA_VISIBLE_DEVICES=1 nohup python if1_d1_regression.py 3 > d1_r3_if1.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d1_regression.py 4 > d1_r4_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d1_regression.py 5 > d1_r5_if1.out &
CUDA_VISIBLE_DEVICES=1 nohup python if1_d1_regression.py 6 > d1_r6_if1.out &

CUDA_VISIBLE_DEVICES=2 nohup python if1_d1_regression.py 7 > d1_r7_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d1_regression.py 8 > d1_r8_if1.out &
CUDA_VISIBLE_DEVICES=1 nohup python if1_d1_regression.py 9 > d1_r9_if1.out &


CUDA_VISIBLE_DEVICES=2 nohup python if1_d1_regression.py 7 > d1_r7_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d1_regression.py 8 > d1_r8_if1.out &
CUDA_VISIBLE_DEVICES=1 nohup python if1_d1_regression.py 9 > d1_r9_if1.out &



CUDA_VISIBLE_DEVICES=1 nohup python if1_d2_regression.py 0 > d2_r0_if1.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d2_regression.py 1 > d2_r1_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d2_regression.py 2 > d2_r2_if1.out &



CUDA_VISIBLE_DEVICES=1 nohup python if1_d2_regression.py 3 > d2_r3_if1.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d2_regression.py 4 > d2_r4_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d2_regression.py 5 > d2_r5_if1.out &

CUDA_VISIBLE_DEVICES=1 nohup python if1_d2_regression.py 6 > d2_r6_if1.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d2_regression.py 7 > d2_r7_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d2_regression.py 8 > d2_r8_if1.out &

CUDA_VISIBLE_DEVICES=3 nohup python if1_d2_regression.py 9 > d2_r9_if1.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 8 --split_file S_target_classification.pkl --seed 2 --wandb-name S_ds_r16_s64 --mix > 0509_S_r16_s64_mix_seed3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 8 --split_file S_target_classification.pkl --seed 3 --wandb-name S_ds_r16_s64 --mix > 0509_S_r16_s64_mix_seed3_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 8 --split_file S_target.pkl --seed 3 --wandb-name Sr_ds_r16_s64 --mix > 0509_Sr_r16_s64_mix_seed3_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 8 --split_file S_target.pkl --seed 2 --wandb-name Sr_ds_r16_s64 --mix > 0509_Sr_r16_s64_mix_seed2_GPU3.out &