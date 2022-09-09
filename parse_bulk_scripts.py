from collections import defaultdict
import sys
from collections import defaultdict
contents = open(sys.argv[1]).readlines()

commands = defaultdict(lambda: [])

for line in contents:
    if line.startswith("CUDA_VISIBLE_DEVICES"):
        commands[line[21]].append(line)

for key in commands:
    with open(sys.argv[1] + "_" + key, 'w') as f:
        f.write('\n'.join(commands[key]))