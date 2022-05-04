python variant-prediction/predict_all.py      --model-location esm1b_t33_650M_UR50S      --sequence "${sequence}"      --dms-input "${name}.csv"      --mutation-col pos      --dms-output "${name}"      --offset-idx 1      --scoring-strategy wt-marginals

CUDA_VISIBLE_DEVICES=3 python -u eval.py esm1b_t33_650M_UR50S fireprot sup --include mean per_tok --toks_per_batch 2048 --num_classes 1 --idx 0 --lr 2e-2 --split_file fireprot_regression_0.pkl --idx 6 

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_1.pkl > d1_1_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_0.pkl > d1_0_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_2.pkl > d1_2_head.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_3.pkl > d1_3_head.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_1.pkl --mix > d1_1_head_mix.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S d1_fasta sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file d1_0.pkl --mix > d1_0_head_mix.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_0.pkl --mix > S_0_head_mix.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_1.pkl --mix > S_1_head_mix.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_2.pkl --mix > S_2_head_mix.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx 0 --lr 2e-2 --split_file S_3.pkl --mix > S_3_head_mix.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx m4 --lr 2e-2 --split_file S_4.pkl --mix > S_4_head_mix.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx m5 --lr 2e-2 --split_file S_5.pkl --mix > S_5_head_mix.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx m6 --lr 2e-2 --split_file S_6.pkl --mix > S_6_head_mix.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 2 --idx m7 --lr 2e-2 --split_file S_7.pkl --mix > S_7_head_mix.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx m4 --lr 2e-2 --split_file S_target_4_classification.pkl --mix > S_4_head.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx m5 --lr 2e-2 --split_file S_target_5_classification.pkl --mix > S_5_head.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx m6 --lr 2e-2 --split_file S_target_6_classification.pkl --mix > S_6_head.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx m7 --lr 2e-2 --split_file S_target_7_classification.pkl --mix > S_7_head.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_4 --lr 2e-2 --split_file S_target_4_classification.pkl --mix > S_4_dsee_classification.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_4 --lr 2e-2 --split_file S_target_4_classification.pkl --mix > S_4_head_classification.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --idx S_5 --lr 2e-2 --split_file S_target_5.pkl --mix > S_5_dsee_regression.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --idx S_5 --lr 2e-2 --split_file S_target_5.pkl --mix > S_5_head_regression.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_0_head_classification --lr 2e-2 --split_file S_target_0_classification.pkl --mix > S_0_head_classification.out &

CUDA_VISIBLE_DEVICES=0 python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_0_head_classification --lr 2e-2 --split_file S_target_0_classification.pkl --mix


CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_regression_dsee_noise --lr 2e-2 --split_file S_target.pkl --noise > S_regression_dsee_noise.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_regression_dsee_mix --lr 2e-2 --split_file S_target.pkl --mix > S_regression_dsee_mix.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_regression_dsee --lr 2e-2 --split_file S_target.pkl > S_regression_dsee.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_regression --lr 2e-2 --split_file S_target.pkl  > S_regression.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 2e-2 --split_file S_target_classification.pkl --adv  > S_classification_adv.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_classification_noise --lr 2e-2 --split_file S_target_classification.pkl --noise  > S_classification_noise.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_classification_dsee_adv --lr 2e-2 --split_file S_target_classification.pkl --adv  > S_classification_dsee_adv.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_regression.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_regression_noise --lr 2e-2 --split_file S_target.pkl --noise > S_regression_noise.out &



CUDA_VISIBLE_DEVICES=3 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_adv --lr 2e-2 --split_file S_target_classification.pkl --adv  > S_classification_adv.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_sup_head.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx S_aadv --lr 2e-2 --split_file S_target_classification.pkl --aadv  > S_classification_aadv.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 5e-3 --split_file S_target_classification.pkl --rank 8 --sparse 64 > r8_5e-3.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_sup_head_dsee.py esm1b_t33_650M_UR50S data/datasets/S_target sup --include mean per_tok --toks_per_batch 2048 --num_classes 5 --idx debug --lr 5e-3 --split_file S_target_classification.pkl --rank 16 --sparse 64 > r16_5e-3.out &