# File defining the Trainer class for TES-RNN Model

import os
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loss_modules import AlphaLoss, OverUnderProvisioning



class TESRNNTrainer(nn.Module):
    def __init__(self, model, dataloader, run_id, config):
        super(TESRNNTrainer, self).__init__()
        self.model = model.to(config['device'])
        self.config = config
        self.dl = dataloader
        self.run_id = str(run_id)
        self.epochs = 0
        self.max_epochs = config['num_of_train_epochs']
        self.csv_save_path =  os.path.join('Results', self.run_id)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_anneal_step'], gamma=config['lr_anneal_rate'])
        self.criterion = AlphaLoss(self.config['alpha'], self.config['output_size'], self.config['device'])
        self.shuffle_batch = config['shuffle_batch']



    # Training over several training epochs
    def train_epochs(self):
        
        # Start measuring total training time
        start_time = time.time()
        
        # Iterate over training epochs and save model at each iteration
        for e in range(self.max_epochs):
            self.scheduler.step()
            epoch_loss = self.train()
            self.save()
            
            # Create folder and file to store training losses over epochs
            if e == 0:
                os.makedirs(self.csv_save_path, exist_ok=True)
                file_path = os.path.join(self.csv_save_path, 'Epoch_training_losses.csv')
                with open(file_path, 'w') as f:
                    f.write('Epoch,Training_loss\n')
            
            # Store training loss of the current epoch
            with open(file_path, 'a') as f:
                f.write(','.join([str(e), str(epoch_loss)]) + '\n')
        
        # Stop measuring and save total training time
        # print('Total Training time in minutes: %5.2f' % ((time.time()-start_time)/60))
        np.save(os.path.join(self.csv_save_path, 'training_time.npy'), ((time.time()-start_time)/60))



    # Training throughout one single training epoch
    def train(self):
        self.model.train()
        epoch_loss = 0
        
        # Train over the clusters of the dataset (single cluster, single iteration)
        for clust_num, (train, val, test, idx) in enumerate(self.dl):
            loss = self.train_clust(train, val, test, idx)
            epoch_loss += loss
        epoch_loss = epoch_loss / (clust_num + 1)
        self.epochs += 1

        # Print and return training epoch loss
        # print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f' % (self.epochs, self.max_epochs, epoch_loss))
        return epoch_loss
        


    # Training over the timeseries of one single cluster
    def train_clust(self, train, val, test, idx):
        series_loss = 0
        
        # Split the timeseries in temporal batches
        temporal_batch_size = self.config['batch_size']
        maxrange = math.ceil(train.shape[1] / temporal_batch_size)
        list_of_number_of_batches = list(range(maxrange))
        if self.shuffle_batch:
            random.shuffle(list_of_number_of_batches)
        
        # Train over the temporal batches of the timeseries
        #for i in range(0,maxrange):
        for i in list_of_number_of_batches:         
            temp_train = train[:,(i*temporal_batch_size):((i+1)*temporal_batch_size)]
            loss = self.train_temporal_batch(temp_train, val, test, idx)
            series_loss += loss
        series_loss = series_loss / maxrange
        return series_loss
        
        
        
    # Training over one temporal batch    
    def train_temporal_batch(self,temp_train, val, test, idx):
        self.optimizer.zero_grad()
        
        # Perform prediction
        network_pred, network_act = self.model(temp_train, val, test, idx)
        
        # Compute loss and perform backward step
        loss = self.criterion(network_pred, network_act)
        loss.backward()
        
        nn.utils.clip_grad_value_(self.model.parameters(), self.config['gradient_clipping'])
        self.optimizer.step()
        return float(loss)
        
        
            
    # Save the model of a given training epoch
    def save(self, save_dir=''):
        file_path = os.path.join('Models', self.run_id)
        model_path = os.path.join(file_path, 'model-{}.pyt'.format(self.epochs))
        os.makedirs(file_path, exist_ok=True)
        torch.save({'state_dict': self.model.state_dict()}, model_path)