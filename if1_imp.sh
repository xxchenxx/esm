CUDA_VISIBLE_DEVICES=0 nohup python if1_d2_tune_prune_imp.py 1 2e-2 1e-4 8 0.2 > if1_imp_1.out &
CUDA_VISIBLE_DEVICES=1 nohup python if1_d2_tune_prune_imp.py 2 2e-2 1e-4 8 0.2 > if1_imp_2.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d2_tune_prune_imp.py 3 2e-2 1e-4 8 0.2 > if1_imp_3.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d2_tune_prune_imp.py 4 2e-2 1e-4 8 0.2 > if1_imp_4.out &


CUDA_VISIBLE_DEVICES=0 nohup python if1_d2_tune_prune_imp.py 5 2e-2 1e-4 8 0.2 > if1_imp_5.out &
CUDA_VISIBLE_DEVICES=1 nohup python if1_d2_tune_prune_imp.py 6 2e-2 1e-4 8 0.2 > if1_imp_6.out &
CUDA_VISIBLE_DEVICES=2 nohup python if1_d2_tune_prune_imp.py 7 2e-2 1e-4 8 0.2 > if1_imp_7.out &
CUDA_VISIBLE_DEVICES=3 nohup python if1_d2_tune_prune_imp.py 8 2e-2 1e-4 8 0.2 > if1_imp_8.out &

CUDA_VISIBLE_DEVICES=0 nohup python if1_d2_tune_prune_imp.py 9 2e-2 1e-4 8 0.2 > if1_imp_9.out &
CUDA_VISIBLE_DEVICES=1 nohup python if1_d2_tune_prune_imp.py 0 2e-2 1e-4 8 0.2 > if1_imp_0.out &
