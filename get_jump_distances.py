from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def euclid(a, b):
    return ((a[0] - b[0])**2 + (a[1]-b[1])**2)**0.5

def load_data(path):
    points = []
    c = 180
    while True:
        try:
            fix_locs = loadmat(path + "/data_train_{}.mat".format(c))["location"]
            for i in range(1, fix_locs.shape[1]):
                batch = fix_locs[:i:]
                for j in range(batch.shape[0]):
                    points.append(fix_locs[:i:][j])
            c += 1
        except FileNotFoundError:
            break
    return points

def jump_distances(points):
    jumps = []
    for set in points:
        for i in range(set.shape[0]-1):
            jumps.append(euclid(set[i], set[i+1]))
    return jumps

path = "plots/artemis_data"
data = load_data(path)
jumps = jump_distances(data)

plt.hist(jumps, bins="fd")
plt.show()