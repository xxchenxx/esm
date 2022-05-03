CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_5.pkl --toks_per_batch 512 --num_classes 2 --idx sparse --pruning_ratio 0.05 --checkpoint supervised-finetuned-5.pt --lr 1e-7 > 5_05.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_6.pkl --toks_per_batch 512 --num_classes 2 --idx sparse --pruning_ratio 0.05 --checkpoint supervised-finetuned-6.pt --lr 1e-7  > 6_05.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_5.pkl --toks_per_batch 512 --num_classes 2 --idx sparse --pruning_ratio 0.1 --checkpoint supervised-finetuned-5.pt --lr 1e-7 > 5_1.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_7.pkl --toks_per_batch 512 --num_classes 2 --idx sparse --pruning_ratio 0.05 --checkpoint supervised-finetuned-7.pt --lr 1e-7 > 7_05.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_6.pkl --toks_per_batch 512 --num_classes 2 --idx sparse --pruning_ratio 0.1 --checkpoint supervised-finetuned-6.pt --lr 1e-7  > 6_1.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_7.pkl --toks_per_batch 512 --num_classes 2 --idx sparse --pruning_ratio 0.1 --checkpoint supervised-finetuned-7.pt --lr 1e-7 > 7_1.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_8.pkl --toks_per_batch 512 --num_classes 2 --idx sparse --pruning_ratio 0.05 --lr 1e-6 > 8_05.out &



CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_0.pkl --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 1e-2 > l0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_1.pkl --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 1e-2 > l1.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 1 --lr 1e-2  --split_file split_9.pkl > l9.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_0.pkl --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 > l0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_1.pkl --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 > l1.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --split_file split_0.pkl --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 5e-3 > l0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok  --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 5e-3 --split_file split_2.pkl > l2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok  --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 5e-3 --split_file split_1.pkl > l1.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file split_4.pkl > l4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file split_5.pkl > l5.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file split_8.pkl > l8.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_2.pkl > m2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_3.pkl > m3.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_4.pkl > m4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_5.pkl > m5.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_8.pkl > m8.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_9.pkl > m9.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_2.pkl > m2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_3.pkl > m3.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_8.pkl > m8.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-6 --split_file split_9.pkl > m9.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-5 --split_file split_8.pkl > m8.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-5 --split_file split_9.pkl > m9.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-5 --split_file split_6.pkl > m6.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_mix.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx mix --lr 1e-5 --split_file split_7.pkl > m7.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_2.pkl > l2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_3.pkl > l3.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_4.pkl > l4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_5.pkl > l5.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_4.pkl > l4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_5.pkl > l5.out &




CUDA_VISIBLE_DEVICES=2 python -u finetune_sup_head.py esm1b_t33_650M_UR50S  sampled sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --mix --lr 2e-2 --split_file split_5.pkl 