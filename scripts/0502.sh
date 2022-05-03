CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 2e-2 --split_file S_target_classification.pkl --rank 8 > r8.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 2e-2 --split_file S_target_classification.pkl --rank 16 > r16.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file S_target_classification.pkl --rank 8 > r8_1e-2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 1e-2 --split_file S_target_classification.pkl --rank 16 > r16_1e-2.out &