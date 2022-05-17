CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --rank 16 --lr-factor 1 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --wandb-name Sr_ds_16_s64_adv_5e-7_lf1_sap_corrected --adv --gamma 5e-7  > 0512_Sr_16_s64_adv_5e-7_seed1_lf1_cr_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --rank 16 --lr-factor 1 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 --wandb-name Sr_ds_16_s64_adv_5e-7_lf1_sap_corrected --adv --gamma 5e-7  > 0512_Sr_16_s64_adv_5e-7_seed2_lf1_cr_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --rank 16 --lr-factor 1 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 --wandb-name Sr_ds_16_s64_adv_5e-7_lf1_sap_corrected --adv --gamma 5e-7  > 0512_Sr_16_s64_adv_5e-7_seed3_lf1_cr_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression_dsee_sap.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 1e-3 --rank 16 --lr-factor 1 --split_file ~/clean_datasets/S/S_target.pkl --seed 4 --wandb-name Sr_ds_16_s64_adv_5e-7_lf1_sap_corrected --adv --gamma 5e-7  > 0512_Sr_16_s64_adv_5e-7_seed4_lf1_cr_GPU3.out &