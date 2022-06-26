import numpy as np
import os
import re
import matplotlib.pyplot as plt

# dir = "experiments/cifar10_trades"
# path = "trades.out"
# path = "trades_madry.out"

dir = "./"
path = "trades_4.log"
test_size = 5000

test_errs = []
train_errs = []
test_robust_errs = []

with open(os.path.join(dir, path), 'r') as f:
    for line in f:
        if "Test:" in line:
            test_err = line.split('(')[1].split('%')[0]
            test_errs.append(1 - float(test_err)/100)
        elif "Training:" in line:
            train_err = line.split('(')[1].split('%')[0]
            train_errs.append(1 - float(train_err)/100)
        elif "robust_err_total:" in line:
            test_robust_err = line.split('tensor(')[1].split(',')[0]
            test_robust_errs.append(float(test_robust_err)/test_size)

"""
print(train_errs)
print(test_errs)
print(test_robust_errs)
"""

mi = min(len(test_errs), len(train_errs), len(test_robust_errs))

# """
plt.plot(np.arange(mi), test_errs[:mi], label='Test Error')
plt.plot(np.arange(mi), train_errs[:mi], label="Train Error")
plt.plot(np.arange(mi), test_robust_errs[:mi], label="Test Robust Error")
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Error")
# plt.show()
plt.savefig("trades_4.png")
# """
