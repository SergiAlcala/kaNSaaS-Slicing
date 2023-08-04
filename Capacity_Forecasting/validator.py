# File defining the Validator class for TES-RNN Model

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from loss_modules import AlphaLoss, OverUnderProvisioning, MSEloss



class TESRNNValidator(nn.Module):
    def __init__(self, model, dataloader, run_id, config):
        super(TESRNNValidator, self).__init__()
        self.model = model.to(config['device'])
        self.config = config
        self.dl = dataloader
        self.run_id = str(run_id)
        self.csv_save_path =  os.path.join('Results', self.run_id)
        self.criterion = AlphaLoss(self.config['alpha'], self.config['output_size'], self.config['device'])
        
    
    # Actual validation
    def validating(self):
        
        # Only one iteration for the single cluster timeseries
        for clust_num, (train, val, test, idx) in enumerate(self.dl):
            
            # Evaluation mode (validation phase)
            self.model.eval()
            with torch.no_grad():
                
                # Get and store normalized predictions and normalized traffic of validation set
                predictions, actuals = self.model(train, val, test, idx, True, False)
                np.save(os.path.join(self.csv_save_path, 'val_predictions.npy'), np.squeeze(predictions.cpu()))
                np.save(os.path.join(self.csv_save_path, 'val_actuals.npy'), np.squeeze(actuals.cpu()))
                