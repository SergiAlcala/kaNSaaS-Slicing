# File containing Loss modules and functions

import torch
import torch.nn as nn
import numpy as np


# Definition of the class for the normalized loss function based on alpha (with differentiability for training)
class AlphaLoss(nn.Module):

    def __init__(self, alpha, output_size, device):
        super(AlphaLoss, self).__init__()
        self.alpha = alpha
        self.output_size = output_size
        self.device = device

    def forward(self, predictions, actuals):
        
        # Function parameters
        step = self.alpha
        epsilon = 0.1 / step
        
        
        # Obtain the normalized prediction error
        error = torch.sub(predictions,actuals).to(self.device)
        
        # Obtain the zero condition
        zero_cond = torch.zeros_like(predictions).to(self.device)
        
        # Obtain the normalized overprovisioning cost (positive error with epsilon * alpha margin)
        overprov_cost = error - (epsilon * step)
        overprovisioning = torch.mul(overprov_cost, torch.gt(overprov_cost, zero_cond).type(torch.FloatTensor).to(self.device))
        
        # Obtain the normalized SLA violation cost (negative error)
        sla_cost = step - (epsilon * step) * error
        sla_violations = torch.mul(sla_cost, torch.le(error, zero_cond).type(torch.FloatTensor).to(self.device))
        
        # Obtain the normalized "differentiability" error (error between 0 and epsilon * alpha margin)
        cost = step - (1 / epsilon) * error
        diff_cost = torch.mul(cost, (torch.le(overprov_cost, zero_cond) & torch.gt(error, zero_cond)).type(torch.FloatTensor).to(self.device))
        
        # Obtaining the final loss
        loss = torch.abs(torch.add(overprovisioning, torch.add(sla_violations, diff_cost)))
        final_loss = torch.mean(loss)
        return final_loss
        


# Definition of Overprovisioning and SLA Violations (underprovisioning) class
class OverUnderProvisioning(nn.Module):
    
    def __init__(self,device):
        super(OverUnderProvisioning, self).__init__()
        self.device = device
    
    # Return total overprovisioning and number of SLA violations
    def provisioning(self, predictions, actuals):
        total_over = np.zeros((predictions.shape[1]))
        num_viol = np.zeros((predictions.shape[1]))
        
        for j in range(predictions.shape[1]):
            error = torch.sub(predictions[:,j],actuals[:,j]).to(self.device)
            zero_condition = torch.zeros_like(predictions[:,j]).to(self.device)
        
            provision = torch.mul(error, torch.ge(error,zero_condition).type(torch.FloatTensor).to(self.device))
            total_over[j] = float(torch.sum(provision))
            num_viol[j] = float(torch.sum(provision == 0))
        
        return total_over, num_viol
        
        


# Definition of MSE loss function
class MSEloss(nn.Module):

    def __init__(self, device):
        super(MSEloss, self).__init__()
        self.device = device

    def forward(self, predictions, actuals):
        diff = torch.sub(predictions, actuals).to(self.device)
        diff = torch.square(diff)
        final_loss = torch.mean(diff)
        return final_loss
        



#Definition of MAE loss function
class MAEloss(nn.Module):

    def __init__(self, device):
        super(MAEloss, self).__init__()
        self.device = device

    def forward(self, predictions, actuals):
        diff = torch.sub(predictions, actuals).to(self.device)
        diff = torch.abs(diff)
        final_loss = torch.mean(diff)
        return final_loss
        
 


# Definition of the function to evaluate overprovisioning and SLA violations costs for single cluster case
def evaluate_costs_single_clust(pred_load, real_load, traffic_peak, alpha):
    
    # Compute the difference between predicted and real load (error)
    error = pred_load - real_load
    
    # Compute overprovisioning and SLA violations
    tot_overprov = np.sum(error[np.where(error >= 0)])
    num_viol = len(error[np.where(error < 0)])
    sla_viol = np.multiply(traffic_peak,num_viol)
    sla_viol = np.multiply(sla_viol, alpha)
    
    # Compute total cost
    total_cost = sla_viol + tot_overprov
    
    return total_cost, tot_overprov, num_viol




# Definition of function to return the denormalized validation loss
def denorm_validation_loss(pred, act, lev, alpha):
    #scale_levs=[]
    #for j in range(0,len(lev),output_size):
    #            scale_levs.append(lev[j])
    #lev=scale_levs
    print(f'actuals_len:{len(act)}')
    print(f'predictions_len:{len(pred)}')
    
    # Denormalize predicted and real load and get denormalized peak
    actuals = act * lev
    preds = pred * lev
    peak = np.max(actuals)
    
    # Get the denormalized loss from overprovisioning and SLA violations
    loss, over, sla = evaluate_costs_single_clust(preds, actuals, peak, alpha)
    
    return loss