
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 8 --split_file S_target_classification.pkl --seed 1 --wandb-name S_ds_r16_s64 --adv --gamma 1e-6 > 0509_S_r16_s16_seed1_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 8 --split_file S_target_classification.pkl --seed 2 --wandb-name S_ds_r16_s64 --adv --gamma 1e-6 > 0509_S_r16_s16_seed2_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 8 --split_file S_target_classification.pkl --seed 3 --wandb-name S_ds_r16_s64 --adv --gamma 1e-6 > 0509_S_r16_s16_seed3_GPU3.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 8 --split_file S_target_classification.pkl --seed 1 --wandb-name S_ds_r16_s64 > 0509_S_r16_s16_seed1_GPU0.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_sap3d.py esm1b_t33_650M_UR50S d2/d2_fasta_clean sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx d2 --lr 2e-2 --split_file d2/d2_0_classification.pkl > d2_0_3d.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 8 --split_file S_target_classification.pkl --seed 1 --wandb-name S_ds_r16_s64 --mix > 0509_S_r16_s64_mix_seed3_GPU0.out &