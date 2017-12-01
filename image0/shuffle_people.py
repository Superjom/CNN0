import numpy as np
images = open('people.1.txt').read().split('\n')

np.random.shuffle(images)
np.random.shuffle(images)

pos = 0
neg = 0
for line in images:
    fs = line.split()
    if len(fs) != 2: continue
    path, label = fs
    label = int(label)
    if label == 1: pos += 1
    if label == 0: neg += 1
print 'pos/neg', float(pos) / neg

with open('people.1.txt', 'w') as f:
    f.write('\n'.join(images))
