# File defining the Tester class for TES-RNN Model

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from loss_modules import AlphaLoss, OverUnderProvisioning, MSEloss



class TESRNNTester(nn.Module):
    def __init__(self, model, dataloader, run_id, config):
        super(TESRNNTester, self).__init__()
        self.model = model.to(config['device'])
        self.config = config
        self.dl = dataloader
        self.run_id = str(run_id)
        self.csv_save_path =  os.path.join('Results', self.run_id)
        self.criterion = AlphaLoss(self.config['alpha'], self.config['output_size'], self.config['device'])
        self.evaluation = OverUnderProvisioning(self.config['device'])
        
        
    # Actual testing
    def testing(self):
        
        # print("\nTesting model ... ")
        
        # Only one iteration for the single cluster timeseries
        for clust_num, (train, val, test, idx) in enumerate(self.dl):
            
            # Evaluation mode (testing phase)
            self.model.eval()
            with torch.no_grad():
                
                # Get normalized predictions and normalized traffic of testing set
                predictions, actuals = self.model(train, val, test, idx, False, True)
                
                # Get total overprovisioning and number of sla_violations from prediction
                total_over, num_viol = self.evaluation.provisioning(predictions, actuals)
                
                # Store outputs of the tester
                np.save(os.path.join(self.csv_save_path, 'test_predictions.npy'), np.squeeze(predictions.cpu()))
                np.save(os.path.join(self.csv_save_path, 'test_actuals.npy'), np.squeeze(actuals.cpu()))
                np.save(os.path.join(self.csv_save_path, 'test_total_overprov.npy'), float(total_over))
                np.save(os.path.join(self.csv_save_path, 'test_num_viol.npy'), int(num_viol))
                # print("Test set (normalized) overprovisioning: %f " % total_over)
                # print("Test set SLA violations: %d " % num_viol)
                
