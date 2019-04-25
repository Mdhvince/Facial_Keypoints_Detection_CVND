import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from transformation import *
from custom_dataset import FacialKeypointsDataset

def create_dataset(csv_file, root_dir):
    #print('Creating dataset ...')
    transform = transforms.Compose([
        Rescale(97),
        RandomCrop(96),
        Normalize(),
        ToTensor()
    ])
    
    train_set = FacialKeypointsDataset(csv_file=csv_file,
                                       root_dir=root_dir,
                                       transform=transform)
    return train_set

  
def train_valid_split(training_set, validation_size):
    """ Function that split our dataset into train and validation
        given in parameter the training set and the % of sample for 
        validation"""
    
    #print('Splitting data ...')
    # obtain training indices that will be used for validation
    num_train = len(training_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(validation_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler
      

def build_lodaers(train_set, train_sampler, valid_sampler, batch_size, valid_size, num_workers, csv_file, root_dir):
    #print('Creating loaders ...')
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=num_workers)
    #print('Done !')
    return train_loader, valid_loader