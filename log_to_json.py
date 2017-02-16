import sys
import json

def read_log(log):
    """
    reads a log file and returns lists of epoch, training loss, learning rate
    """
    epochs = []
    losses = []
    lrs = []
    with open(log, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        epoch, loss, lr, _ = line.split('\t')
        epochs.append(float(epoch))
        losses.append(float(loss))
        lrs.append(float(lr))
    return epochs, losses, lrs


epochs = []
losses = []
lrs = []
e_c = 0
for i, log_file in enumerate(sys.argv[2:]):
    e, lo, lr = read_log(log_file)
    epochs.extend([n + (e_c) for n in e])
    e_c += len(e)
    losses.extend(lo)
    lrs.extend(lr)

i = 0
lr = lrs[0]
result = {lr: [list(), list()]}
while i < len(epochs):
    if lrs[i] == lr:
        result[lr][0].append(epochs[i])
        result[lr][1].append(losses[i])
        i = i + 1
    else:
        lr = lrs[i]
        result[lr] = [list(), list()]

with open(sys.argv[1], 'w') as f:
    f.write(json.dumps(result))
    
