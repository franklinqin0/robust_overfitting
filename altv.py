import numpy as np
import os
import re
import matplotlib.pyplot as plt

dir = "experiments/cifar10_trades"
# path = "trades.out"
path = "trades_madry.out"

test_accs = []
train_accs = []
test_robust_errs = []

with open(os.path.join(dir, path), 'r') as f:
    for line in f:
        if "Test:" in line:
            test_acc = line.split('(')[1].split('%')[0]
            test_accs.append(1 - float(test_acc)/100)
        elif "Training:" in line:
            train_acc = line.split('(')[1].split('%')[0]
            train_accs.append(1 - float(train_acc)/100)
        elif "robust_err_total:" in line:
            test_robust_err = line.split('tensor(')[1].split(',')[0]
            test_robust_errs.append(float(test_robust_err)/10000)
    
plt.plot(np.arange(len(test_accs)), test_accs, label='Test Error')
plt.plot(np.arange(len(test_accs)), train_accs, label="Train Error")
plt.plot(np.arange(len(test_accs)), test_robust_errs, label="Test Robust Error")
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Error")
# plt.show()
plt.savefig("haha.png")
