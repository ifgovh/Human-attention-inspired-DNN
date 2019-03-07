# read the best model and check the epoch and best valid accuracy
import torch
import os
import glob

os.chdir('/project/cortical/RVA-Fractional_motion/ckpt')

file_name = glob.glob('./*model_best.pth.tar')

for i in range(len(file_name)):
	temp = torch.load(file_name)
	print(file_name + 'epoch_{}_best_valid_acc_{}'.format(
		temp['epoch'], temp['best_valid_acc']) + '\n')