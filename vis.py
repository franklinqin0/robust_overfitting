import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt

# dir = "experiments/cifar10_trades_l2"
# dir = "experiments/cifar10_validation/preactresnet18"
dir = "experiments/cifar10_wide/wideresnet_1"
# dir = "experiments/cifar10_lr/preactresnet18_manystep"

# path = "eval.log"
path = "output.log"

header = ""
data = []
# whitespace = ""

with open(os.path.join(dir, path), 'r') as f:
    for line in f:
        if "Epoch" in line:
            header = line
        elif "Test Acc" in line:
            temp = re.sub('\s+','\t',line)
            temp = temp.split('-', 1)[1].split('\t')
            temp = list(filter(None, temp))
            data.append(temp[:])
        else:
            continue

header = header.split('-', 1)[1]
header = re.sub('\t+', ' ', header).split("  ")
header = list(filter(None, header))

df = pd.DataFrame(data, columns=header)
df = df.rename(columns=lambda x: x.strip())

df1 = df[["Epoch", "Train Acc", "Train Robust Acc", "Test Acc", "Test Robust Acc"]]
df1["Epoch"] = df1["Epoch"].astype(int)
df1["Train Acc"] = 1 - df1["Train Acc"].astype(float)
df1["Train Robust Acc"] = 1 - df1["Train Robust Acc"].astype(float)
df1["Test Acc"] = 1 - df1["Test Acc"].astype(float)
df1["Test Robust Acc"] = 1 - df1["Test Robust Acc"].astype(float)

# print(df1.dtypes)
print(df1.head())
# print(df1["Epoch"])
# df1.plot(xticks=df1["Epoch"])
temp = df1.groupby("Epoch").sum()
temp.plot()

plt.legend(loc='best')
plt.show()