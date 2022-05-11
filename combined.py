from glob import glob
names = glob("pretrained/*")
names = list(names)
#print(names)

seqs = []

for name in names:
	seq = open(name).read()
	seqs.append(seq)

st = "\n".join(seqs)
print(len(st))

with open("pretrained.fasta", "w") as f:
    f.write(st)