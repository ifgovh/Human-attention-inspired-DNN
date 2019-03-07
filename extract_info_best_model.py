# read the best model and check the epoch and best valid accuracy
import torch
import os
import glob
import numpy as np
import scipy.io as sio

os.chdir('/project/cortical/RVA-Fractional_motion/ckpt')

file_name = glob.glob('./*ckpt.pth.tar')
best_train_acc = np.zeros([len(file_name),1])
best_valid_acc = np.zeros([len(file_name),1])
for i in range(len(file_name)):
	temp = torch.load(file_name[i])
	best_train_acc[i] = temp['best_train_acc']
	best_valid_acc[i] = temp['best_valid_acc']
	print(file_name[i] + '_epoch_{}_best_valid_acc_{}'.format(
		temp['epoch'], temp['best_valid_acc']) + '\n')

sio.savemat("/project/cortical/RVA-Fractional_motion/ckpt/best_acc.mat",
                        mdict={'best_train_acc':best_valid_acc,'patch':best_valid_acc})

print('The highest train acc_{}, file_{}; the highest valid acc_{}, file_{}'.format(
	np.amax(best_train_acc),file_name[np.argmax(best_train_acc)],np.amax(best_valid_acc),file_name[np.argmax(best_valid_acc)]))
