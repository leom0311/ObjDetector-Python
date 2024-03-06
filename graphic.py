import numpy as np
import matplotlib.pyplot as plt
import time

max_mip_level = 5
data = []
for i in range(5):
    data.append([])
colors = ['c', 'g', 'b', 'y', 'm']

key = "cls-loss"

epochs = []
step = 5
with open("./log.txt", "r") as fp:
    epoch = -1
    for line in fp.readlines():
        line = str.strip(line)
        if line == "":
            continue
        if line.find("epoch") != -1:
            epoch = int(line.split("epoch ")[1].split(" ")[0])
            epochs.append(epoch)
        if epoch == -1:
            continue
        if line.find(key) == -1:
            continue
        mip = int(line.split("]")[0].split("mip")[1].split("-")[0])
        val = float(str.strip(line.split("]")[1]))
        data[mip].append(val)

print(data)
for i in range(5):
    x = epochs 
    # range(epoch + 1)
    y = data[i]
    plt.plot(x, y, color=colors[i % len(colors)], label='MIP' + str(i)) 
    plt.xticks(np.arange(min(x), max(x)+1, step))
plt.legend() 
plt.title(key)
plt.savefig("images/epoch-" + str(epoch) + ".png")
plt.show()