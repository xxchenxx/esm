
CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target.pkl --adv --gamma 1e-6 > 0506_Sr_g1e-6_GPU7.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target.pkl --adv --gamma 1e-5 > 0506_Sr_g1e-5_GPU7.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target.pkl --adv --gamma 1e-4 > 0506_Sr_g1e-4_GPU7.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --lr-factor 10 --rank 16 --split_file ~/clean_datasets/S/S_target.pkl --adv --gamma 1e-3 > 0506_Sr_g1e-3_GPU7.out &

