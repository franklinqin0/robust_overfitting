import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

dir = "cifar_model"
# dir = "experiments/cifar10_validation/preactresnet18"
# dir = "experiments/cifar10_wide/wideresnet_1"
# dir = "experiments/cifar10_lr/preactresnet18_manystep"
dir = "reproduce_results"

# path = "eval.log"
path = "l2_50.log"
# path = "output.log"

# dir = "."
# path = "haha.log"

header = ""
data = []
# whitespace = ""

with open(os.path.join(dir, path), 'r') as f:
    for line in f:
        if "Epoch" in line:
            header = line
        elif ("Namespace" in line) or ("Resuming" in line) or ("downloaded" in line):
            continue
        else:
            # print('line:', line)
            temp = re.sub('\s+','\t',line)
            temp = temp.split('-', 1)[1].split('\t')
            temp = list(filter(None, temp))
            data.append(temp[:])

header = header.split('-', 1)[1]
header = re.sub('\t+', ' ', header).split("  ")
header = list(filter(None, header))

df = pd.DataFrame(data, columns=header)
df = df.rename(columns=lambda x: x.strip())

df1 = df[["Epoch", "Train Acc", "Train Robust Acc", "Test Acc", "Test Robust Acc"]]
df1["Epoch"] = df1["Epoch"].astype(int)
df1["Train Err"] = 1 - df1["Train Acc"].astype(float)
del df1["Train Acc"]
df1["Train Robust Err"] = 1 - df1["Train Robust Acc"].astype(float)
del df1["Train Robust Acc"]
df1["Test Err"] = 1 - df1["Test Acc"].astype(float)
del df1["Test Acc"]
df1["Test Robust Err"] = 1 - df1["Test Robust Acc"].astype(float)
del df1["Test Robust Acc"]

temp = df1.groupby("Epoch").sum()
temp.plot()

plt.legend(loc='best')
# plt.show()
plt.savefig("l2_50.png")
