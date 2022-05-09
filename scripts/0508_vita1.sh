CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 1e-2 --split_file d2/d2_3.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5  > d2_3_ds_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 1e-2 --split_file d2/d2_2.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5  > d2_2_ds_advr.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 1e-2 --split_file d2/d2_1.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5  > d2_1_ds_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d2 --lr 1e-2 --split_file d2/d2_0.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5  > d2_0_ds_advr.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_7_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_7_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_6_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_6_ds_advc.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_3_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_3_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_2_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_2_ds_advc.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_6_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_6_ds_advc.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_7_classification.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-5 > d1_7_ds_advc.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_0.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-7 > d1_0_ds_advr.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S d1/d1_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx d1 --lr 1e-2 --split_file d1/d1_1.pkl --rank 4 --lr-factor 10 --adv --gamma 1e-7 > d1_1_ds_advr.out &
