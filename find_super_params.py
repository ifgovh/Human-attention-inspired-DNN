from nevergrad import instrumentation as inst
import torch
import numpy as np
from nevergrad.optimization import optimizerlib

def find_super_params(patch_size=8, glimpse_scale=2, num_patches=1, loc_hidden=256,
	glimpse_hidden=128, num_glimpses=6, std=0.17, M=10, valid_size=0.1,
	batch_size=256, weight_decay=0, dropout_phi=0, batchnorm_phi=True, 
	dropout_l=0, batchnorm_l=True, dropout_g=0, batchnorm_g=True,
	dropout_h=0, batchnorm_h=True)

	# glimpse network params
	config.patch_size = patch_size;
	config.glimpse_scale = glimpse_scale;
	# # of downscaled patches per glimpse
	config.num_patches = num_patches;
	config.loc_hidden = loc_hidden;
	config.glimpse_hidden = glimpse_hidden;

	# core network params
	config.num_glimpses = num_glimpses;
	config.hidden_size = loc_hidden + glimpse_hidden;

	# reinforce params
	# gaussian policy standard deviation
	config.std = std;
	# Monte Carlo sampling for valid and test sets
	config.M = M;

	# data params
	config.valid_size = valid_size;
	config.batch_size = batch_size;
	# # of subprocesses to use for data loading
	config.num_workers = 4;
	config.shuffle = True;
	config.show_sample = False;
	config.dataset_name = 'CIFAR';

	# training params
	config.is_train = True;
	# # SGD
	# config.momentum = momentum;
	# config.init_lr = init_lr;
	# config.epochs = 2000;
	# # ReduceLRonPlatean
	# config.lr_patience = lr_patience;
	config.train_patience = 200;
	config.optimizer = 'Adam';
	config.loss_fun_action = 'nll';
	config.loss_fun_baseline = 'mse';
	# weight decay (L2 penalty)
	config.weight_decay = weight_decay;
	config.dropout = dropout;
	config.batchnorm = batchnorm;

	# dropout
	config.dropout_phi = dropout_phi
	config.dropout_l = dropout_l
	config.dropout_g = dropout_g
	config.dropout_h = dropout_h

	# batch normalization
	config. = batchnorm_flag_phi
	config. = batchnorm_flag_l
	config. = batchnorm_flag_g
	config. = batchnorm_flag_h
	
	# other params
	config.use_gpu = True;
	config.best = True;
	config.random_seed = 1;
	config.data_dir = './data';
	config.ckpt_dir = './ckpt';
	config.logs_dir = './logs/';
	config.use_tensorboard = False;
	config.resume = False;
	config.print_freq = 10;
	config.plot_freq = 1;
	config.PBSarray_ID = 0;

	main(config)

	# read best model
    if config.use_gpu:
	    model_name = 'ram_gpu_{0}_{1}_{2}x{3}_{4:1.2f}_{5}_{6}_{7}_{8}_{9}_{10}_{11:1.5f}_{12:1.2f}_{13:1.2f}_{14:1.2f}'.format(
	        config.PBSarray_ID, config.num_glimpses, 
            config.patch_size, config.patch_size,
            config.glimpse_scale, config.num_patches, 
            config.batchnorm_flag_phi, config.batchnorm_flag_l,
            config.batchnorm_flag_g, config.batchnorm_flag_h,
            config.weight_decay, config.dropout_phi,
            config.dropout_l, config.dropout_g,
            config.dropout_h
	    )
    else:
	    model_name = 'ram_{0}_{1}_{2}x{3}_{4:1.2f}_{5}_{6}_{7}_{8}_{9}_{10}_{11:1.5f}_{12:1.2f}_{13:1.2f}_{14:1.2f}'.format(
	        config.PBSarray_ID, config.num_glimpses, 
            config.patch_size, config.patch_size,
            config.glimpse_scale, config.num_patches, 
            config.batchnorm_flag_phi, config.batchnorm_flag_l,
            config.batchnorm_flag_g, config.batchnorm_flag_h,
            config.weight_decay, config.dropout_phi,
            config.dropout_l, config.dropout_g,
            config.dropout_h
	    )
	best_model_file = torch.load(model_name + '_model_best.pth.tar')
	return 100 - best_model_file['best_valid_acc']


patch_size = inst.var.SoftmaxCategorical(np.arange(5,20)) 
num_patches = inst.var.SoftmaxCategorical(np.arange(1,5)) 
num_glimpses = inst.var.SoftmaxCategorical(np.arange(5,15))
# glimpse_hidden = inst.var.SoftmaxCategorical(np.arange(128,5)) 
# loc_hidden = inst.var.SoftmaxCategorical(np.arange(192,15))

batchnorm_phi = inst.var.SoftmaxCategorical(["True", "False"])
batchnorm_l = inst.var.SoftmaxCategorical(["True", "False"])
batchnorm_g = inst.var.SoftmaxCategorical(["True", "False"])
batchnorm_h = inst.var.SoftmaxCategorical(["True", "False"])
 

# continuous
glimpse_scale = inst.var.Gaussian(mean=2, std=2)  
weight_decay = inst.var.Gaussian(mean=0.001, std=0.001)
dropout_phi = inst.var.Gaussian(mean=0.5, std=2)
dropout_l = inst.var.Gaussian(mean=0.5, std=2)
dropout_g = inst.var.Gaussian(mean=0.5, std=2)
dropout_h = inst.var.Gaussian(mean=0.5, std=2)




# Instrumentation
# argument transformation
"""
def find_super_params(patch_size=8, glimpse_scale=2, num_patches=1, loc_hidden=256,
	glimpse_hidden=128, num_glimpses=6, std=0.17, M=10, valid_size=0.1,
	batch_size=256, weight_decay=0, dropout_phi=0, batchnorm_phi=True, 
	dropout_l=0, batchnorm_l=True, dropout_g=0, batchnorm_g=True,
	dropout_h=0, batchnorm_h=True)

When optimizing hyperparameters as e.g. in machine learning. If you don't know what variables (see instrumentation) to use:

use SoftmaxCategorical for discrete variables
use TwoPointsDE with num_workers equal to the number of workers available to you. See the machine learning example for more.
Or if you want something more aimed at robustly outperforming random search in highly parallel settings (one-shot):

use OrderedDiscrete for discrete variables, taking care that the default value is in the middle.
Use ScrHammersleySearchPlusMiddlePoint (PlusMiddlePoint only if you have continuous parameters or good default values for discrete parameters).
"""
# dicrete
patch_size = inst.var.SoftmaxCategorical(np.arange(5,20)) 
num_patches = inst.var.SoftmaxCategorical(np.arange(1,5)) 
num_glimpses = inst.var.SoftmaxCategorical(np.arange(5,15))
# glimpse_hidden = inst.var.SoftmaxCategorical(np.arange(128,5)) 
# loc_hidden = inst.var.SoftmaxCategorical(np.arange(192,15))

batchnorm_phi = inst.var.SoftmaxCategorical(["True", "False"])
batchnorm_l = inst.var.SoftmaxCategorical(["True", "False"])
batchnorm_g = inst.var.SoftmaxCategorical(["True", "False"])
batchnorm_h = inst.var.SoftmaxCategorical(["True", "False"])
 

# continuous
glimpse_scale = inst.var.Gaussian(mean=2, std=2)  
weight_decay = inst.var.Gaussian(mean=0.001, std=0.001)
dropout_phi = inst.var.Gaussian(mean=0.5, std=2)
dropout_l = inst.var.Gaussian(mean=0.5, std=2)
dropout_g = inst.var.Gaussian(mean=0.5, std=2)
dropout_h = inst.var.Gaussian(mean=0.5, std=2)
"""
def find_super_params(patch_size=8, glimpse_scale=2, num_patches=1, loc_hidden=256,
	glimpse_hidden=128, num_glimpses=6, std=0.17, M=10, valid_size=0.1,
	batch_size=256, weight_decay=0, dropout_phi=0, batchnorm_phi=True, 
	dropout_l=0, batchnorm_l=True, dropout_g=0, batchnorm_g=True,
	dropout_h=0, batchnorm_h=True)
"""
# create the instrumented function
# put them in order, if it is discrete varible, only give the variable name; if it is continuous, give a pair;
# if it is constant, only give the constant
instrum = inst.Instrumentation(patch_size, glimpse_scale=glimpse_scale, num_patches, 256,
	128, num_glimpses, 0.17, 10, 0.1, 256, weight_decay=weight_decay, dropout_phi=dropout_phi, batchnorm_phi, 
	dropout_l=dropout_l, batchnorm_l, dropout_g=dropout_g, batchnorm_g,	dropout_h=dropout_h, batchnorm_h)

print(instrum.dimension)  

#Converts data to arguments to check
#print(instrum.data_to_arguments([1, -80, -80, 80, 3]))
# prints (args, kwargs): (('b', 'e', 'blublu'), {'value': 7})
# b is selected because 1 > 0 (the threshold is 0 here since there are 2 values.
# e is selected because proba(e) = exp(80) / (exp(80) + exp(-80) + exp(-80))
# value=7 because 3 * std + mean = 7


# create the instrumented function using the "Instrumentation" instance above
ifunc = instrum.instrument(find_super_params)
print(ifunc.dimension)  # dimensional space as above
# you can still access the instrumentation instance will ifunc.instrumentation

import pdb; pdb.set_trace()
optimizer = optimizerlib.PortfolioDiscreteOnePlusOne(dimension=ifunc.dimension, budget=48) #TwoPointsDE
recommendation = optimizer.optimize(ifunc)

# recover the arguments this way (don't forget deteriministic=True)
args, kwargs = ifunc.data_to_arguments(recommendation, deterministic=True)
print(args)    # should print ["b", "e", "blublu"]
print(kwargs)  # should print {"value": 0} because -.5 * std + mean = 0

# but be careful, since some variables are stochastic (SoftmaxCategorical ones are), setting deterministic=False may yield different results
# The following will print more information on the conversion to your arguments:
print(ifunc.get_summary(recommendation))