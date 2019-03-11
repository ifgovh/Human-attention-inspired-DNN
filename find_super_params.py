from nevergrad import instrumentation as inst
import torch
import numpy as np
def find_super_params(patch_size=8, glimpse_scale=2, num_patches=1, loc_hidden=256,
	glimpse_hidden=128, num_glimpses=6, hidden_size=384, std=0.17, M=10, valid_size=0.1,
	batch_size=256, weight_decay=0, dropout=0, batchnorm=True, PBSarray_ID)

	# glimpse network params
	config.patch_size = patch_size;
	config.glimpse_scale = glimpse_scale;
	# # of downscaled patches per glimpse
	config.num_patches = num_patches;
	config.loc_hidden = loc_hidden;
	config.glimpse_hidden = glimpse_hidden;

	# core network params
	config.num_glimpses = num_glimpses;
	config.hidden_size = hidden_size;

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
	config.PBSarray_ID = PBSarray_ID;

	main(config)

	# read best model
	    if config.use_gpu:
		    model_name = 'ram_gpu_{}_{}x{}_{}_{}'.format(
		        config.PBSarray_ID, config.num_glimpses, 
	            config.patch_size,
		        config.patch_size, config.glimpse_scale
		    )
	    else:
		    model_name = 'ram_{}_{}x{}_{}_{}'.format(
	            config.PBSarray_ID, config.num_glimpses, 
	            config.patch_size,
	            config.patch_size, config.glimpse_scale, 
	        )
	best_model_file = torch.load(model_name + '_model_best.pth.tar')
	return 100 - best_model_file['best_valid_acc']


# Instrumentation
# argument transformation
def find_super_params(patch_size=8, glimpse_scale=2, num_patches=1, loc_hidden=256,
	glimpse_hidden=128, num_glimpses=6, hidden_size=384, std=0.17, M=10, valid_size=0.1,
	batch_size=256, weight_decay=0, PBSarray_ID)
"""
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
batchnorm = inst.var.SoftmaxCategorical(["True", "False"])
arg2 = inst.var.SoftmaxCategorical(["a", "c", "e"]) 
arg2 = inst.var.SoftmaxCategorical(["a", "c", "e"]) 
arg2 = inst.var.SoftmaxCategorical(["a", "c", "e"]) 
arg2 = inst.var.SoftmaxCategorical(["a", "c", "e"]) 
arg2 = inst.var.SoftmaxCategorical(["a", "c", "e"])  

# continuous
glimpse_scale = inst.var.Gaussian(mean=2, std=2)  
weight_decay = inst.var.Gaussian(mean=2, std=2)
dropout = inst.var.Gaussian(mean=0.5, std=2)


# create the instrumented function
instrum = inst.Instrumentation(arg1, arg2, "blublu", value=value)
# the 3rd arg. is a positional arg. which will be kept constant to "blublu"
print(instrum.dimension)  # 5 dimensional space

# The dimension is 5 because:
# - the 1st discrete variable has 1 possible values, represented by a hard thresholding in
#   a 1-dimensional space, i.e. we add 1 coordinate to the continuous problem
# - the 2nd discrete variable has 3 possible values, represented by softmax, i.e. we add 3 coordinates to the continuous problem
# - the 3rd variable has no uncertainty, so it does not introduce any coordinate in the continuous problem
# - the 4th variable is a real number, represented by single coordinate.


print(instrum.data_to_arguments([1, -80, -80, 80, 3]))
# prints (args, kwargs): (('b', 'e', 'blublu'), {'value': 7})
# b is selected because 1 > 0 (the threshold is 0 here since there are 2 values.
# e is selected because proba(e) = exp(80) / (exp(80) + exp(-80) + exp(-80))
# value=7 because 3 * std + mean = 7


# create the instrumented function using the "Instrumentation" instance above
ifunc = instrum.instrument(myfunction)
print(ifunc.dimension)  # 5 dimensional space as above
# you can still access the instrumentation instance will ifunc.instrumentation

ifunc([1, -80, -80, 80, 3])  # will print "b e blublu" and return 49 = 7**2
# check the instrumentation output explanation above if this is not clear


from nevergrad.optimization import optimizerlib
optimizer = optimizerlib.OnePlusOne(dimension=ifunc.dimension, budget=100)
recommendation = optimizer.optimize(ifunc)

# recover the arguments this way (don't forget deteriministic=True)
args, kwargs = ifunc.data_to_arguments(recommendation, deterministic=True)
print(args)    # should print ["b", "e", "blublu"]
print(kwargs)  # should print {"value": 0} because -.5 * std + mean = 0

# but be careful, since some variables are stochastic (SoftmaxCategorical ones are), setting deterministic=False may yield different results
# The following will print more information on the conversion to your arguments:
print(ifunc.get_summary(recommendation))