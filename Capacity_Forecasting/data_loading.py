# Data Loading File

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


# Function to create a single cluster dataset (single cluster traffic timeseries)
def create_dataset(data, chop_train, chop_val, chop_test):
    
    # Import original data (3D dataset of traffic at individual Base Stations with 2 spatial dimensions (x,y) (Base Station grid position) and 1 temporal dimension)
    file_data = np.load(data)

    
    # Aggregate data of all base station into a single cluster and leave only the temporal dimension
    #agg_data = np.sum(np.sum(file_data[:,:,:], axis=-1), axis=-1)
    agg_data=file_data[:chop_train+chop_val+chop_test]
    print(f'data_shape: {agg_data.shape}')
    
    # Create training set
    train = []
    train.append(np.array([float(i) for i in agg_data[:chop_train]]))
    print(f'train: {train[0].shape}')
    
    # Create validation set
    valid = []
    valid.append(np.array([float(i) for i in agg_data[chop_train:chop_train+chop_val]]))
    print(f'valid: {valid[0].shape}')

    # Create testing set
    test = []
    test.append(np.array([float(i) for i in agg_data[chop_train+chop_val:chop_train+chop_val+chop_test]]))
    print(f'test: {test[0].shape}')

    return train, valid, test



# Class to create Torch version of the Dataset
class Dataset(Dataset):
    
    def __init__(self, dataTrain, dataVal, dataTest, device):
        self.dataTrain = [torch.tensor(dataTrain[i]) for i in range(len(dataTrain))]
        self.dataVal = [torch.tensor(dataVal[i]) for i in range(len(dataVal))]
        self.dataTest = [torch.tensor(dataTest[i]) for i in range(len(dataTest))]
        self.device = device
        
        
    def __len__(self):
        return len(self.dataTrain)
        
    def __getitem__(self, idx):
        return self.dataTrain[idx].to(self.device), self.dataVal[idx].to(self.device), self.dataTest[idx].to(self.device), idx
