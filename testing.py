from scipy.io import loadmat
from scipy.stats import levy_stable
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch import Tensor

# contents = loadmat("plots/ram_12345678_7_10x10_2.00_1_False_False_False_False_0.01000_0.0_0.0_0.0_0.0/data_train_1.mat")
# print(contents['location'])

# def get_distances(jumps):
#
alpha = 1.45
beta = 0

plot = []

dist = levy_stable
dist.a = -1
dist.b = 1

print(dist.rvs(alpha, beta))

t = Tensor([1, 2])

# for i in range(10000):
#     # p = max(-1, min(1, levy_stable.rvs(alpha, beta)))
#     # p = math.tanh(levy_stable.rvs(alpha, beta))
#     p = dist.rvs(alpha, beta)
#     plot.append(p)

print(dist.a)

levy_array = np.array(plot)
print(levy_array)
print(np.max(levy_array))
# print("MEAN", levy_stable.mean(alpha, beta))

plt.hist(levy_array, bins="fd")
plt.show()