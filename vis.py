import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# dir = "cifar_model"
dir = "reproduce_results"

# path = "eval.log"
# path = "output.log"
path = "dropout0.3.log"

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
print('tbe min:', df1["Test Robust Err"].min())
print('tbe last: ', df1["Test Robust Err"].iloc[-1])
print('tbe diff:', df1["Test Robust Err"].iloc[-1] - df1["Test Robust Err"].min())

temp = df1.groupby("Epoch").sum()
temp.plot()

plt.legend(loc='best')
# plt.show()
plt.savefig(path.rsplit('.', 1)[0] + ".png")
