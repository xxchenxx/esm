CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file S_target_classification.pkl --rank 24 --sparse 64 > r32_1e-2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file S_target_classification.pkl --rank 16 --sparse 32 > r16_1e-2_s32.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file S_target_classification.pkl --rank 16 --sparse 16 > r16_1e-2_s16.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --split_file S_target_classification.pkl --adv --gamma 1e-4 > S_classification_adv_new_1e-4.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 1e-2 --split_file S_target_classification.pkl --adv --gamma 2e-4 > S_classification_adv_new_2e-4.out &

