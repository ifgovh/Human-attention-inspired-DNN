import torch
import numpy as np

from main import main

from ax.service.managed_loop import optimize

def call_rva(patch_size=8, num_patches=1, loc_hidden=256, glimpse_hidden=128, 
	num_glimpses=6, std=0.17, M=10, valid_size=0.1, batch_size=256, batchnorm_flag_phi=True,
	batchnorm_flag_l=True, batchnorm_flag_g=True, batchnorm_flag_h=True, glimpse_scale=2, weight_decay=0,
	dropout_phi=0, dropout_l=0,  dropout_g=0, dropout_h=0, alpha=1.4, gamma=0.9):

	class CONFIG:
		patch_size=8
		num_patches=1
		loc_hidden=256
		glimpse_hidden=128
		num_glimpses=6
		rnn_type='LSTMCell'
		std=0.17
		M=10
		valid_size=0.1
		batch_size=256
		batchnorm_phi=True
		batchnorm_l=True
		batchnorm_g=True
		batchnorm_h=True
		glimpse_scale=2
		weight_decay=0
		dropout_phi=0
		dropout_l=0
		dropout_g=0
		dropout_h=0
		num_workers = 4;
		shuffle = True;
		show_sample = False;
		dataset_name = 'cluttered_MNIST';
		is_train = True;
		train_patience = 200;
		optimizer = 'Adam';
		loss_fun_action = 'nll';
		loss_fun_baseline = 'mse';
		use_gpu = True;
		best = True;
		random_seed = 1;
		data_dir = './data/';
		ckpt_dir = './ckpt/';
		logs_dir = './logs/';
		use_tensorboard = False;
		resume = False;
		print_freq = 10;
		plot_freq = 1;
		PBSarray_ID = 0;
		alpha = 1.4;
		gamma = 0,9;

	config = CONFIG()
	
	# glimpse network params
	config.patch_size = patch_size;
	config.glimpse_scale = glimpse_scale;
	# # of downscaled patches per glimpse
	config.num_patches = num_patches;
	config.loc_hidden = loc_hidden;
	config.glimpse_hidden = glimpse_hidden;
	config.rnn_type = rnn_type;

	# core network params
	config.num_glimpses = num_glimpses;
	config.hidden_size = loc_hidden + glimpse_hidden;
	config.rnn_type = rnn_type;

	# reinforce params
	# gaussian policy standard deviation
	config.std = std;
	# Monte Carlo sampling for valid and test sets
	config.M = M;
	config.alpha = alpha;
	config.gamma = gamma;

	# data params
	config.valid_size = valid_size;
	config.batch_size = batch_size;
	# # of subprocesses to use for data loading
	config.num_workers = 4;
	config.shuffle = True;
	config.show_sample = False;
	config.dataset_name = dataset_name;

	# training params
	config.is_train = True;
	# SGD
	config.momentum = 0.5#momentum;
	config.init_lr = 3e-4#init_lr;	
	# ReduceLRonPlatean
	config.lr_patience = 10#lr_patience;
	config.epochs = 800;
	config.train_patience = 200;
	config.optimizer = 'Adam';
	config.loss_fun_action = 'nll';
	config.loss_fun_baseline = 'mse';
	# weight decay (L2 penalty)
	config.weight_decay = weight_decay;

	# dropout
	config.dropout_phi = dropout_phi
	config.dropout_l = dropout_l
	config.dropout_g = dropout_g
	config.dropout_h = dropout_h

	# batch normalization
	config.batchnorm_flag_phi = batchnorm_flag_phi
	config.batchnorm_flag_l = batchnorm_flag_l
	config.batchnorm_flag_g = batchnorm_flag_g
	config.batchnorm_flag_h = batchnorm_flag_h
	
	# other params
	config.use_gpu = True;
	config.best = True;
	config.random_seed = 1;
	config.data_dir = './data/';
	config.ckpt_dir = './ckpt/';
	config.logs_dir = './logs/';
	config.use_tensorboard = False;
	config.resume = False;
	config.print_freq = 10;
	config.plot_freq = 1;
	config.PBSarray_ID = 0;

	main(config)

	# read best model
	if config.use_gpu:
		model_name = 'ram_gpu_{0}_{1}_{2}x{3}_{4:1.2f}_{5}_{6}_{7}_{8}_{9}_{10:1.5f}_{11:1.1f}_{12:1.1f}_{13:1.1f}_{14:1.1f}'.format(
				config.PBSarray_ID, config.num_glimpses, 
				config.patch_size, config.patch_size,
				config.glimpse_scale, config.num_patches, 
				config.batchnorm_flag_phi, config.batchnorm_flag_l,
				config.batchnorm_flag_g, config.batchnorm_flag_h,
				config.weight_decay, config.dropout_phi,
				config.dropout_l, config.dropout_g,
				config.dropout_h)                            
	else:
		model_name = 'ram_{0}_{1}_{2}x{3}_{4:1.2f}_{5}_{6}_{7}_{8}_{9}_{10:1.5f}_{11:1.1f}_{12:1.1f}_{13:1.1f}_{14:1.1f}'.format(
				config.PBSarray_ID, config.num_glimpses, 
				config.patch_size, config.patch_size,
				config.glimpse_scale, config.num_patches, 
				config.batchnorm_flag_phi, config.batchnorm_flag_l,
				config.batchnorm_flag_g, config.batchnorm_flag_h,
				config.weight_decay, config.dropout_phi,
				config.dropout_l, config.dropout_g,
				config.dropout_h)   
	
	best_model_file = torch.load(config.ckpt_dir + model_name + '_model_best.pth.tar')

	return {"valid_accu", (100 - best_model_file['best_valid_acc'],0.0)}

# def call_rva(patch_size=8, num_patches=1, loc_hidden=256, glimpse_hidden=128, 
# 	num_glimpses=6, std=0.17, M=10, valid_size=0.1, batch_size=256, batchnorm_flag_phi=True,
# 	batchnorm_flag_l=True, batchnorm_flag_g=True, batchnorm_flag_h=True, glimpse_scale=2, weight_decay=0,
# 	dropout_phi=0, dropout_l=0,  dropout_g=0, dropout_h=0, alpha=1.4, gamma=0.9):

def find_super_params():
	best_parameters, values, experiment, model = optimize(
    parameters=[
        {
            "name": "patch_size",
            "type": "range",
            "bounds": [4, 16],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "num_patches",
            "type": "range",
            "bounds": [1, 2],
        },
        {
            "name": "loc_hidden",
            "type": "range",
            "bounds": [128, 1024],
        },
        {
            "name": "glimpse_hidden",
            "type": "range",
            "bounds": [128, 1024],
        },
        {
            "name": "num_glimpses",
            "type": "range",
            "bounds": [5, 16],
        },
        {
            "name": "batch_size",
            "type": "range",
            "bounds": [256, 1024],
        },
        {
            "name": "batchnorm_flag_phi",
            "type": "range",
            "bounds": [False, True],
            "value_type": "bool",
        },
        {
            "name": "batchnorm_flag_l",
            "type": "range",
            "bounds": [False, True],
            "value_type": "bool",
        },
        {
            "name": "batchnorm_flag_g",
            "type": "range",
            "bounds": [False, True],
            "value_type": "bool",
        },
        {
            "name": "batchnorm_flag_h",
            "type": "range",
            "bounds": [False, True],
            "value_type": "bool",
        },
        {
            "name": "glimpse_scale",
            "type": "range",
            "bounds": [1, 2],
        },        
    ],
    experiment_name="test",
    objective_name="valid_accu",
    evaluation_function=call_rva,
    minimize=True,  # Optional, defaults to False.
    total_trials=30, # Optional.
    #parameter_constraints=[" = "],  # Optional.
    #outcome_constraints=["l2norm <= 1.25"],  # Optional.    
	)

if __name__ == '__main__':
	find_super_params()
