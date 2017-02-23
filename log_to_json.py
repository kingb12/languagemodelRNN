import sys
import json

def read_log(log):
    """
    reads a log file and returns lists of epoch, training loss, learning rate
    """
    epochs = []
    losses = []
    lrs = []
    t_perps = []
    vlosses = []
    v_perps= []
    with open(log, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        epoch, loss, lr, t_perp, vloss, v_perp, _ = line.split('\t')
        epochs.append(float(epoch))
        losses.append(float(loss))
        lrs.append(float(lr))
        vlosses.append(float(vloss))
        v_perps.append(float(v_perp))
        t_perps.append(float(t_perp))
    return epochs, losses, lrs, t_perps, vlosses, v_perps


epochs = []
losses = []
lrs = []
t_perps = []
vlosses = []
v_perps= []
e_c = 0
for i, log_file in enumerate(sys.argv[2:]):
    e, lo, lr, t_p, vlo, v_p = read_log(log_file)
    t_px = []
    for t in t_p:
        t_px.append(t)
    epochs.extend([n + (e_c) for n in e])
    e_c += len(e)
    losses.extend(lo)
    vlosses.extend(vlo)
    lrs.extend(lr)
    t_perps.extend(t_px)
    v_perps.extend(v_p)


i = 0
lr = lrs[0]
result = {lr: [list(),list(),list(),list(), list()]}
while i < len(epochs):
    if lrs[i] == lr:
        result[lr][0].append(epochs[i])
        result[lr][1].append(losses[i])
        result[lr][2].append(vlosses[i])
        result[lr][3].append(t_perps[i])
        result[lr][4].append(v_perps[i])
        i = i + 1
    else:
        lr = lrs[i]
        result[lr] = [list(), list(),list(),list(),list()]

with open(sys.argv[1], 'w') as f:
    f.write(json.dumps(result))
    
