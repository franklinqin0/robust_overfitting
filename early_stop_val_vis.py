import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# dir = "cifar_model"
# dir = "experiments/cifar10_validation/preactresnet18"
# dir = "experiments/cifar10_wide/wideresnet_1"
# dir = "experiments/cifar10_lr/preactresnet18_manystep"
dir = "reproduce_results"

path = "val.log"

header = ""
data = []
vals = []
# whitespace = ""

with open(os.path.join(dir, path), 'r') as f:
    for line in f:
        if "Epoch" in line:
            header = line
        elif ("Namespace" in line) or ("Resuming" in line) or ("downloaded" in line):
            continue
        elif "validation" in line:
            temp = re.sub('\s+','\t', line)
            temp = temp.split('-', 1)[1].split('\t')
            temp = list(filter(None, temp))
            val = temp[-2]
            vals.append(float(val))
        else:
            temp = re.sub('\s+','\t', line)
            temp = temp.split('-', 1)[1].split('\t')
            temp = list(filter(None, temp))
            data.append(temp[:])

header = header.split('-', 1)[1]
header = re.sub('\t+', ' ', header).split("  ")
header = list(filter(None, header))

df = pd.DataFrame(data, columns=header)
df = df.rename(columns=lambda x: x.strip())

df1 = df[["Epoch", "Train Robust Loss", "Test Robust Loss"]]
df1["Epoch"] = df1["Epoch"].astype(int)
df1["Train Robust Loss"] = df1["Train Robust Loss"].astype(float)
df1["Test Robust Loss"] = df1["Test Robust Loss"].astype(float)
df1["Val Robust Loss"] = vals

temp = df1.groupby("Epoch").sum()
temp.plot()

plt.legend(loc='best')
# plt.show()
plt.savefig("try.png")
