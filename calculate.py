import sys
accs = []
for i in range(10):
    acc = 0
    with open(f"d2_{i}_if1_dense_1e-4_8_{sys.argv[1]}.out") as f:
        content = f.readlines()
    
    for line in content:
        if line.startswith("ACC"):
            number = float(line.split("(")[1].split(",")[0])
            if number > acc:
                acc = number
    accs.append(acc)

print(accs)
print(sum(accs) / 10)