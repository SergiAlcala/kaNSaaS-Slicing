# File defining the core of TES-RNN model

import os
import torch
import torch.nn as nn
import numpy as np
from DRNN import DRNN
import sys


class TESRNN(nn.Module):
    def __init__(self, tau, maximum, num_clusters, config, run_id):
        super(TESRNN, self).__init__()
        self.config = config
        self.run_id = run_id
        
        self.add_nl_layer = self.config['add_nl_layer']
        self.nl_layer = nn.Linear(config['state_hsize'], config['state_hsize'])
        self.act = nn.Tanh()
        self.scoring = nn.Linear(config['state_hsize'], config['output_size'])
        self.logistic = nn.Sigmoid()
        self.resid_drnn = ResidualDRNN(self.config)
        
        self.num_clusters = num_clusters
        self.maximum = maximum

        
        # Set initial parameter of the Exponential Smoothing formula and make it learnable
        init_lev_sms = []
        init_lev_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
        self.init_lev_sms = nn.ParameterList(init_lev_sms)
        
        
        # Set minimum level threshold tau (expressed as fraction of maximum)
        init_tau = []
        init_tau.append(nn.Parameter(torch.Tensor([tau]), requires_grad=False))
        self.tau = init_tau
        

        
    
    def forward(self, train, val, test, idxs, validating=False, testing=False):
        # Get the pre-processing parameters previously learnt/obtained
        lev_sms = self.logistic(torch.stack([self.init_lev_sms[idx] for idx in idxs]).squeeze(1))
        tau = torch.stack([self.tau[idx] for idx in idxs]).squeeze(1)
        
        tau = tau.to('cuda')
        
        # Normalization Levels
        levs = []
        scale_levs=[]
        
        
        # Compute normalization levels with Exponential Smoothing formula for testing phase
        if testing:
           
            # Calculate initial level (first timestep)
            #print(f'if testing test_shape:{test.shape}')
            init_lev = torch.max(test[:, 0], tau * self.maximum)
            levs.append(init_lev)
            
            # Calculate level for the current timestep
            for i in range(1, test.shape[1]):
                new_lev = torch.max(lev_sms * (test[:,i]) + (1 - lev_sms)*levs[i - 1], tau * self.maximum)
                levs.append(new_lev)
                
                
                
        # Compute normalization levels with Exponential Smoothing formula for validation phase
        elif validating:
            # Calculate initial level (first timestep)
            init_lev = torch.max(val[:, 0], tau * self.maximum)
            levs.append(init_lev)
            
            # Calculate level for the current timestep
            for i in range(1, val.shape[1]):
                new_lev = torch.max(lev_sms * (val[:,i]) + (1 - lev_sms)*levs[i - 1], tau * self.maximum)
                levs.append(new_lev)
            
            
            
        
        # Compute normalization levels with Exponential Smoothing formula for training phase
        else:
            #print(f'if training train_shape:{train.shape}')
            # Calculate initial level (first timestep)
            init_lev = torch.max(train[:, 0], tau * self.maximum)
            levs.append(init_lev)
            
            # Calculate level for the current timestep
            for i in range(1, train.shape[1]):
                new_lev = torch.max(lev_sms * (train[:,i]) + (1 - lev_sms)*levs[i - 1], tau * self.maximum)
                levs.append(new_lev)
        print(f'Validating levs shape:{len(levs)}')
                
        
        # Fix dimensions of levels
        print(f'levs_ shape before transpose{levs[0].shape}')
        levs_stacked = torch.stack(levs).transpose(1, 0)
        print(f'levs_stacked_shape_sin0:  {levs_stacked.shape}')
        # Save normalization levels of the testing set
        if testing:

            for i in range(levs_stacked.shape[0]):
                np.save(os.path.join('Results', self.run_id, 'test_levels.npy'), levs_stacked[i, self.config['input_size']-1:test.shape[1]-self.config['time_admission']].cpu())
                
        # Save normalization levels of the validation set
        if validating:
            for i in range(levs_stacked.shape[0]):
                np.save(os.path.join('Results', self.run_id, 'val_levels.npy'), levs_stacked[i, self.config['input_size']-1:val.shape[1]-self.config['time_admission']].cpu())
                
        
        
                    
        # Input and output normalized windows
        window_input_list = []
        window_output_list = []
        
        
        # Input and output window normalization for testing phase
        if testing:
            for i in range(self.config['input_size'] - 1, test.shape[1]):
            #for i in range(self.config['input_size'] - 1, test.shape[1],self.config['output_size']):
                input_window_start = i + 1 - self.config['input_size']
                input_window_end = i + 1
                
                test_norm_window_input = (test[:, input_window_start:input_window_end] / levs_stacked[:, i].unsqueeze(1))
                window_input_list.append(test_norm_window_input.float())
                
                output_window_start = i + 1
                #output_window_end = i + 1 + self.config['output_size']
                output_window_end = i + 1 + self.config['time_admission']
                
                #if i < test.shape[1] - self.config['time_admission']:
                #    test_norm_window_output = (torch.max(test[:, output_window_start:output_window_end]) / levs_stacked[:, i].unsqueeze(1))
                #    #test_norm_window_output = (test[:, output_window_start:output_window_end] / levs_stacked[:, i].unsqueeze(1))
                #    window_output_list.append(test_norm_window_output)

                test_max_values=torch.empty((1,self.config['N_dec']),device=self.config['device'])
                if i < test.shape[1] - self.config['time_admission']:
                    for j in range(self.config['N_dec']):
                        test_norm_window_output = (torch.max(test[:, output_window_start+j*self.config['time_decision']:output_window_start+j*self.config['time_decision']+self.config['time_decision']]) / levs_stacked[:, i].unsqueeze(1))
                        test_max_values[0][j]=test_norm_window_output
                                                   
                    window_output_list.append(test_max_values.float())   
        
        
        
        # Input and output window normalization for validation phase
        elif validating:
            print(f'val_shape: {val.shape[1]}')
            #for i in range(self.config['input_size'] - 1, val.shape[1],self.config['output_size']):
            for i in range(self.config['input_size'] - 1, val.shape[1]):


                input_window_start = i + 1 - self.config['input_size']
                input_window_end = i + 1
                print(f'input_window_start_val:{input_window_start}')
                print(f'input_window_end_val:{input_window_end}')
                
                val_norm_window_input = (val[:, input_window_start:input_window_end] / levs_stacked[:, i].unsqueeze(1))
                window_input_list.append(val_norm_window_input.float())
                
                output_window_start = i + 1
                #output_window_end = i + 1 + self.config['output_size']
                output_window_end = i + 1 + self.config['time_admission']
                print(f'output_window_start_val:{output_window_start}')
                print(f'output_window_end_val:{output_window_end}')

                
                #if i < val.shape[1] - self.config['output_size']:
                #    val_norm_window_output = (torch.max(val[:, output_window_start:output_window_end]) / levs_stacked[:, i].unsqueeze(1))
                #    #val_norm_window_output = (val[:, output_window_start:output_window_end] / levs_stacked[:, i].unsqueeze(1))
                #    window_output_list.append(val_norm_window_output)

                val_max_values=torch.empty((1,self.config['N_dec']),device=self.config['device'])
                if i < val.shape[1] - self.config['time_admission']:
                    for j in range(self.config['N_dec']):
                        val_norm_window_output = (torch.max(val[:, output_window_start+j*self.config['time_decision']:output_window_start+j*self.config['time_decision']+self.config['time_decision']]) / levs_stacked[:, i].unsqueeze(1))
                        val_max_values[0][j]=val_norm_window_output
                                                   
                    window_output_list.append(val_max_values.float())        
        
        
        
        # Input and output window normalization for training phase
        else:
            print(f'train_shape: {train.shape[1]}')
            for i in range(self.config['input_size'] - 1, train.shape[1]):
                input_window_start = i + 1 - self.config['input_size']
                input_window_end = i + 1

                print(f'input_window_start_train:{input_window_start}')
                print(f'input_window_end_train:{input_window_end}')
               # print(f'levs_stacked : {levs_stacked[:, i].unsqueeze(1)}')
                train_norm_window_input = (train[:, input_window_start:input_window_end] / levs_stacked[:, i].unsqueeze(1))
     
                window_input_list.append(train_norm_window_input.float())

                output_window_start = i + 1
                #output_window_end = i + 1 + self.config['output_size']
                output_window_end = i + 1 + self.config['time_admission']
                print(f'output_window_start_train:{output_window_start}')
                print(f'output_window_end_train:{output_window_end}')

                #if i < train.shape[1] - self.config['output_size']:
                train_max_values=torch.empty((1,self.config['N_dec']),device=self.config['device'])
                if i < train.shape[1] - self.config['time_admission']:
                    for j in range(self.config['N_dec']):
                        train_norm_window_output = (torch.max(train[:, output_window_start+j*self.config['time_decision']:output_window_start+j*self.config['time_decision']+self.config['time_decision']]) / levs_stacked[:, i].unsqueeze(1))
                        train_max_values[0][j]=train_norm_window_output
                                                   
                    window_output_list.append(train_max_values.float())

        
        # Fix input and output window dimensions
        window_input = torch.cat([i.unsqueeze(0) for i in window_input_list], dim=0)
        window_output = torch.cat([i.unsqueeze(0) for i in window_output_list], dim=0)
        print(f'window_input_shape:{window_input.shape}')
        print(f'window_output_shape:{window_output.shape}')
        
        
        # Check if training phase
        if testing == False and validating == False:
            self.train()
        
        
        # Get neural network predictions    
        network_pred = self.series_forward(window_input[:-self.config['time_admission']])
        
        # Get actual normalized values (output window)
        network_act = window_output
        print(f'network_pred_shape:{network_pred.shape}')
        print(f'network_act_shape:{network_act.shape}')
        
        
        return network_pred, network_act
        
        
    # Forward function to get neural network predictions
    def series_forward(self, data):
        data = self.resid_drnn(data)
        if self.add_nl_layer:
            data = self.nl_layer(data)
            data = self.act(data)
        data = self.scoring(data)
        return data




# Residual DRNN class definition
class ResidualDRNN(nn.Module):
    def __init__(self, config):
        super(ResidualDRNN, self).__init__()
        self.config = config

        layers = []
        for grp_num in range(len(self.config['dilations'])):

            if grp_num == 0:
                input_size = self.config['input_size']
            else:
                input_size = self.config['state_hsize']

            l = DRNN(input_size,
                     self.config['state_hsize'],
                     n_layers=len(self.config['dilations'][grp_num]),
                     dilations=self.config['dilations'][grp_num],
                     cell_type=self.config['rnn_cell_type'])

            layers.append(l)

        self.rnn_stack = nn.Sequential(*layers)

    def forward(self, input_data):
        for layer_num in range(len(self.rnn_stack)):
            residual = input_data
            out, _ = self.rnn_stack[layer_num](input_data)
            if layer_num > 0:
                out += residual
            input_data = out
        return out