CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 1 --wandb-name 0509_S_ds_r16_s64_seed1_adv_1e-6_GPU0 --adv --gamma 1e-6  > 0509_S_ds_r16_s64_seed1_adv_1e-6_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 2 --wandb-name 0509_S_ds_r16_s64_seed2_adv_1e-6_GPU1 --adv --gamma 1e-6 > 0509_S_ds_r16_s64_seed2_adv_1e-6_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 3 --wandb-name 0509_S_ds_r16_s64_seed3_adv_1e-6_GPU2 --adv --gamma 1e-6 > 0509_S_ds_r16_s64_seed3_adv_1e-6_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 4 --wandb-name 0509_S_ds_r16_s64_seed4_adv_1e-6_GPU3 --adv --gamma 1e-6 > 0509_S_ds_r16_s64_seed4_adv_1e-6_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 1 --wandb-name 0509_S_ds_r16_s64_seed1_adv_5e-7_GPU0 --adv --gamma 5e-7  > 0509_S_ds_r16_s64_seed1_adv_5e-7_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 2 --wandb-name 0509_S_ds_r16_s64_seed2_adv_5e-7_GPU1 --adv --gamma 5e-7 > 0509_S_ds_r16_s64_seed2_adv_5e-7_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 3 --wandb-name 0509_S_ds_r16_s64_seed3_adv_5e-7_GPU2 --adv --gamma 5e-7 > 0509_S_ds_r16_s64_seed3_adv_5e-7_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-2 --rank 16 --lr-factor 10 --split_file ~/clean_datasets/S/S_target_classification.pkl --seed 4 --wandb-name 0509_S_ds_r16_s64_seed4_adv_5e-7_GPU3 --adv --gamma 5e-7 > 0509_S_ds_r16_s64_seed4_adv_1e-6_GPU7.out &