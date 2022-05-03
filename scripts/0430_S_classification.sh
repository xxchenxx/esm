CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_0 --lr 2e-2 --split_file S_target_0_classification.pkl --mix > S_0_dsee_classification.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_1 --lr 2e-2 --split_file S_target_1_classification.pkl --mix > S_1_dsee_classification.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_2 --lr 2e-2 --split_file S_target_2_classification.pkl --mix > S_2_dsee_classification.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_3 --lr 2e-2 --split_file S_target_3_classification.pkl --mix > S_3_dsee_classification.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_0_noise --lr 2e-2 --split_file S_target_0_classification.pkl --noise > S_0_noise_dsee_classification.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_1_noise --lr 2e-2 --split_file S_target_1_classification.pkl --noise > S_1_noise_dsee_classification.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_2_noise --lr 2e-2 --split_file S_target_2_classification.pkl --noise > S_2_noise_dsee_classification.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_3_noise --lr 2e-2 --split_file S_target_3_classification.pkl --noise > S_3_noise_dsee_classification.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_classification_dsee_noise --lr 2e-2 --split_file S_target_classification.pkl --noise > S_classification_dsee_noise.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_classification_dsee_mix --lr 2e-2 --split_file S_target_classification.pkl --mix > S_classification_dsee_mix.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_classification_dsee --lr 2e-2 --split_file S_target_classification.pkl > S_classification_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_classification --lr 2e-2 --split_file S_target_classification.pkl  > S_classification.out &

python -u finetune_sup_head_parallel.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_classification --lr 2e-2 --split_file S_target_classification.pkl 

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_classification --lr 2e-2 --split_file S_target_classification.pkl  > S_classification.out &

