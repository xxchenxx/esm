CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_4_classification.pkl --adv --gamma 1e-5 > d1_4_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_0_classification.pkl --adv --gamma 1e-5 > d1_0_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_2_classification.pkl --adv --gamma 1e-5 > d1_2_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_8_classification.pkl --adv --gamma 1e-5 > d1_8_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_8_classification.pkl --adv --gamma 1e-5 > d2_0_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_6_classification.pkl --adv --gamma 1e-5 > d1_6_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_0.pkl --adv --gamma 1e-5 > d1_0_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_2.pkl --adv --gamma 1e-5 > d1_2_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_4.pkl --adv --gamma 1e-5 > d1_4_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_6.pkl --adv --gamma 1e-5 > d1_6_advr.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_8.pkl --adv --gamma 1e-5 > d1_8_advr.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_0.pkl --rank 4 --lr-factor 10  > d1_0_ds_r.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_2.pkl --rank 4 --lr-factor 10  > d1_2_ds_r.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_4.pkl --rank 4 --lr-factor 10  > d1_4_ds_r.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_6.pkl --rank 4 --lr-factor 10  > d1_6_ds_r.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_8.pkl --rank 4 --lr-factor 10  > d1_8_ds_r.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_9.pkl --rank 4 --lr-factor 10  > d1_9_ds_r.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 1e-2 --split_file d2/d2_4.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5  > d2_4_ds_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 1e-2 --split_file d2/d2_5.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5  > d2_5_ds_advr.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 1e-2 --split_file d2/d2_6.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5  > d2_6_ds_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 1e-2 --split_file d2/d2_7.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5  > d2_7_ds_advr.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 1e-2 --split_file d2/d2_8.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5  > d2_8_ds_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 1e-2 --split_file d2/d2_9.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5  > d2_9_ds_advr.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_4_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_4_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_5_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_5_ds_advc.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_8_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 --seed 2 > d2_8_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_9_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 --seed 2 > d2_9_ds_advc.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_8_classification.pkl --rank 4 --lr-factor 10 > d1_8_ds_c.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_9_classification.pkl --rank 4 --lr-factor 10 > d1_9_ds_c.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_7_classification.pkl --rank 4 --lr-factor 10  > d1_7_ds_c.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_6_classification.pkl --rank 4 --lr-factor 10  > d1_6_ds_c.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_5_classification.pkl --rank 4 --lr-factor 10  > d1_5_ds_c.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_4_classification.pkl --rank 4 --lr-factor 10  > d1_4_ds_c.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_8_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_8_ds_advc.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_9_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_9_ds_advc.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_7_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_7_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_6_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_6_ds_advc.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_5_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_5_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_4_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_4_ds_advc.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_1_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_1_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_0_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_0_ds_advc.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_4_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_4_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_5_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_5_ds_advc.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_6_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_6_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_7_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_7_ds_advc.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_5_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_5_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 1e-2 --split_file d2/d2_6_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d2_6_ds_advc.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_2.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_2_ds_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_3.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_3_ds_advr.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_4.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 --epochs 10 > d1_4_ds_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_5.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 --epochs 10 > d1_5_ds_advr.out &


CUDA_VISIBLE_DEVICES=1 nohup python if1_d1.py 3 > d1_3_if1.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d1.py 4 > d1_4_if1.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d1.py 5 > d1_5_if1.out &


CUDA_VISIBLE_DEVICES=3 python -u finetune_sup_head_sap.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 2e-2 --split_file d1/d1_5.pkl 
