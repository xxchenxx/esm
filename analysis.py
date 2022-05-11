import pandas as pd
from glob import glob
filenames = sorted(glob("*_labeled.csv"))
import numpy as np
dtm = []
score = []
for name in filenames:
    data = pd.read_csv(name)
    sc = data['esm1v_t33_650M_UR90S_3'].to_numpy()
    tm = data['unregulated'].to_numpy()
    mask = np.isnan(tm)
    tm = tm[~mask]
    sc = sc[~mask]
    print(np.corrcoef(tm, sc)[0,1])
    dtm.append(tm)
    score.append(sc)


dtm = np.concatenate(dtm, 0)
score = np.concatenate(score, 0)

import matplotlib.pyplot as plt
plt.scatter(dtm, score)
plt.savefig("corr.png")

print(np.corrcoef(dtm, score))