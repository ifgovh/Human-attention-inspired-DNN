import numpy as np
import os
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset

class cluttered_MNIST(Dataset):
    # clutered MNIST
    def __init__(self, data_dir, patch_size=8):                
        # define transforms for MNIST
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        resize = transforms.Resize(patch_size)
        self.trans = transforms.Compose([resize, normalize])

        # tranfer cluttered image to tensor
        self.to_tensor = transforms.ToTensor()

        # get raw data from pytorch built-in fun
        self.dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=self.trans
            )        

    def __getitem__(self, index):
        # this is the way to access the dataset, the output is a tuple (PIL image and tensor Label)        
        image = clutter(self.dataset.__getitem__(index)[0],self.dataset)

        # returen image and label
        return (img, self.dataset.__getitem__(index)[1])

    def __len__(self):
        return len(self.dataset.__len__()) # of how many data(images?) you have

    def clutter(original_img, train_data, width=60, height=60, n_patches=4):
        """
        Inserts MNIST digits at random locations in larger blank background and
        adds 8 by 8 subpatches from other random MNIST digits.

        Args
        ----
        - original_img: a PIL image
        - train_data: the out put of dataset.MNIST.
        - width: the width of output data.
        - height: the height of output data.
        - n_patches: the number of random subpatches

        Returns
        -------   
        - cluttered_img: an 2D tensor
        """
        original_img = np.array(original_img)
        width_img = 8
        height_img = 8

        # blank background for each image 
        cluttered_img = np.zeros(height,width) 

        # sample location
        x_pos   = np.random.randint(0,width - width_img)
        y_pos   = np.random.randint(0,width - height_img)

        # insert in blank image
        cluttered_img[y_pos:y_pos+height_img, x_pos:x_pos+width_img] += original_img

        # add 8 x 8 subpatches from random other digits
        for i in range(n_patches):
            digit   = np.array(train_data.__getitem__(np.random.randint(0, train_data.shape[0]-1))[0]);
            c1, c2  = np.random.randint(0, width_img - width_sub, size=2)
            i1, i2  = np.random.randint(0, width - width_sub, size=2)
            cluttered_img[i2:i2+height_sub, i1:i1+width_sub] += digit[0, 0, c2:c2+height_sub, c1:c1+width_sub]

        cluttered_img = np.clip(cluttered_img, 0., 1.)

        return self.to_tensor(cluttered_img)


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           dataset_name,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - dataset_name: the name of dataset, can be MNIST or ImageNet
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
      In the paper, this number is set to 0.1.
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])    
    # load dataset    
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=trans
        )

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    elif dataset_name == 'cluttered_MNIST':
        dataset = cluttered_MNIST(data_dir)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
    elif dataset_name == 'ImageNet':        
        traindir = os.path.join(data_dir, 'train_sample')
        valdir = os.path.join(data_dir, 'valid_sample')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(64,scale=(0.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))    
        
        # this sampler block has bug! You cannot distribute it!
        if not shuffle:
            torch.distributed.init_process_group(backend='gloo',
                init_method='tcp://224.66.41.62:23456',
                world_size=1)
            #ref: https://github.com/pytorch/examples/blob/e0d33a69bec3eb4096c265451dbb85975eb961ea/imagenet/main.py#L113-L126
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)

        valid_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(76),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)    
    elif dataset_name == 'CIFAR':
        dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=trans
        )

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    else:
       raise ValueError("[!] Please input correct dataset_name.")
    
    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            dataset = dataset if dataset_name == 'MNIST' else train_dataset, 
            batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    # To-do: add ImageNet or other dataset for testing
    return data_loader
