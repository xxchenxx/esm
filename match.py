from glob import glob
import os
a = glob("variant_data_0415/*.fasta")
# with open("esm_missing.txt", "w") as f:
b = ','.join(glob("variant_data_0417/*.txt"))
if True:
    for name in a:
        name = name.split("/")[1].split(".")[0]
        if not name in b:
            print(name)