with open("csv_name.txt") as f:
    names = f.readlines()

from glob import glob
for name in names:
    print(name)
    name = name.strip()
    prefix = "_".join(name.split("_")[:-1])
    align = glob(f"alignments/{prefix}*")[0]
    print(f"Match: {align}")
    #align = "alignments/PA_FLU_1_b0.5.a2m"
    with open(align) as f:
        align = f.read()
    
    seq = align.split(">")[1]
    try:
        info = seq.split("\n")[0]
        start = int(info.split("/")[1].split('-')[0])
    except:
        start = 1
    
    seq = "".join(seq.split("\n")[1:])
    import os
    command = f"CUDA_VISIBLE_DEVICES=3 python variant-prediction/predict.py --model-location esm1v_t33_650M_UR90S_1 esm1v_t33_650M_UR90S_2 esm1v_t33_650M_UR90S_3 esm1v_t33_650M_UR90S_4 esm1v_t33_650M_UR90S_5 --sequence {seq.upper()} --dms-input dms/{name} --mutation-col mutant --dms-output labeled_{name} --offset-idx {start} --scoring-strategy wt-marginals"
    print(command)
    os.system(command)
    break
