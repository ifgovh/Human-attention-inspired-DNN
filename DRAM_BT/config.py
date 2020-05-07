import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# glimpse network params
glimpse_arg = add_argument_group('Glimpse Network Params')
glimpse_arg.add_argument('--patch_size', type=int, default=8,
                         help='size of extracted patch at highest res')

# context network params
glimpse_arg = add_argument_group('Context Network Params')
glimpse_arg.add_argument('--kernel_size', type=int, default=3,
                         help='size of kernel of convolution layers')


# core network params
core_arg = add_argument_group('Core Network Params')
core_arg.add_argument('--num_glimpses', type=int, default=6,
                      help='# of glimpses, i.e. BPTT iterations')
core_arg.add_argument('--hidden_size', type=int, default=256,
                      help='hidden size of rnn')


# reinforce params
reinforce_arg = add_argument_group('Reinforce Params')
reinforce_arg.add_argument('--std', type=float, default=0.17,
                           help='gaussian policy standard deviation')
reinforce_arg.add_argument('--M', type=float, default=10,
                           help='Monte Carlo sampling for valid and test sets')
# for symmetry stable distribution
reinforce_arg.add_argument('--alpha', type=float, default=1.4,
                           help='symmetry stable policy first shape parameter')
reinforce_arg.add_argument('--gamma', type=float, default=0.9,
                           help='symmetry stable policy scale parameter')


# bottom-up params
bot_up_arg = add_argument_group('Bottom-up Params')
bot_up_arg.add_argument('--dt', type=float, default=0.1,
                           help='integration step of Langevin equation')
bot_up_arg.add_argument('--scale_oscillation', type=float, default=0.1,
                           help='the scale of oscialltio part of Langevin equation')
bot_up_arg.add_argument('--period', type=float, default=0.2,
                           help='the period of Langevin equation')
bot_up_arg.add_argument('--threshold', type=float, default=0.05,
                           help='threshold of direction comparison of saliency increment')
bot_up_arg.add_argument('--T', type=float, default=2,
                           help='Metropolis algorithm temperature')


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_size', type=float, default=0.1,
                      help='Proportion of training set used for validation')
data_arg.add_argument('--batch_size', type=int, default=32,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=4,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')
data_arg.add_argument('--show_sample', type=str2bool, default=False,
                      help='Whether to visualize a sample grid of the data')
data_arg.add_argument('--dataset_name', type=str, default='cluttered_MNIST',
                      help='Load which dataset, MNIST, cluttered_MNIST, ImageNet, CIFAR')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--momentum', type=float, default=0.5,
                       help='Nesterov momentum value')
train_arg.add_argument('--epochs', type=int, default=200,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=3e-4,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=10,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=50,
                       help='Number of epochs to wait before stopping train')
train_arg.add_argument('--optimizer', type=str, default='Adam',
                       help='The name of optimizer')
train_arg.add_argument('--loss_fun_action', type=str, default='cross_entropy',
                       help='The name of loss fucntion of action nn')
train_arg.add_argument('--loss_fun_baseline', type=str, default='cross_entropy',
                       help='The name of loss fucntion of baseline nn')
train_arg.add_argument('--weight_decay', type=float, default='0',
                       help='weight decay (L2 penalty)')

train_arg.add_argument('--dropout', type=float, default=0,
                       help='Probability of an element to be zeroed')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=False,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data',
                      help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                      help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=False,
                      help='Whether to use tensorboard for visualization')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--print_freq', type=int, default=10,
                      help='How frequently to print training details')
misc_arg.add_argument('--plot_freq', type=int, default=1,
                      help='How frequently to plot glimpses')
misc_arg.add_argument('--PBSarray_ID', type=int, default=1,
                      help='PBSarray_ID')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
