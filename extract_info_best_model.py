# read the best model and check the epoch and best valid accuracy
import torch
import os
import glob
import numpy as np


os.chdir('/project/cortical/RVA-Fractional_motion/ckpt')

file_name = glob.glob('./*model_best.pth.tar')
epoch = np.zeros([len(file_name),1])
best_acc = np.zeros([len(file_name),1])
for i in range(len(file_name)):
	temp = torch.load(file_name[i])
	epoch[i] = temp['epoch']
	best_acc[i] = temp['best_valid_acc']
	print(file_name[i] + '_epoch_{}_best_valid_acc_{}'.format(
		temp['epoch'], temp['best_valid_acc']) + '\n')

print('The smallest epoch_{}, file_{}; the highest acc_{}, file_{}'.format(
	np.amin(epoch),file_name[np.argmin(epoch)],np.amax(best_acc),file_name[np.argmax(best_acc)]))
