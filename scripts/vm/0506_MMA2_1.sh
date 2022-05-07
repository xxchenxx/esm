
CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 > 0506_Sr_seed1_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 > 0506_Sr_seed2_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 > 0506_Sr_seed3_GPU2.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --mix > 0506_Sr_mix_seed1_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 --mix > 0506_Sr_mix_seed2_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 3 --mix > 0506_Sr_mix_seed3_GPU5.out &


CUDA_VISIBLE_DEVICES=6 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 1 --noise > 0506_Sr_noise_seed1_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S ~/clean_datasets/S/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S --lr 2e-2 --split_file ~/clean_datasets/S/S_target.pkl --seed 2 --noise > 0506_Sr_noise_seed2_GPU7.out &
