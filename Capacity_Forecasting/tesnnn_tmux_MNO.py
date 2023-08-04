

"""!pip install tensorflow==1.14"""


import math
import numpy as np
from torch.utils.data import DataLoader
from data_loading import create_dataset, Dataset
from config import get_config
from trainer import TESRNNTrainer
from validator import TESRNNValidator
from tester import TESRNNTester
from model import TESRNN
from loss_modules import *
import matplotlib.pyplot as plt
import sys
import os

#### 1 week 10080
#### 2 week 20160
#### 3 week 30240
#### 4 week 40320
#### 5 week 50400
#### 6 week 60480
#### 7 week 70560
#### 8 week 80640
#### 9 week 90720
#### 10 week 100800


# CONFIGURATION SETTINGS

device=1

torch.cuda.set_device(device)

# List of the services to be tested
emBB_filenames = []
mMTC_filenames = []
uRLLC_filenames = []

datapath='/home/user/Synthetic_Data'

for i in range(0, 7):
    if i != 6:
        emBB_filenames.append('emBB_'+str(i))
        uRLLC_filenames.append('uRLLC_'+str(i))
        mMTC_filenames.append('mMTC_'+str(i))
    else:
        emBB_filenames.append('emBB_'+str(i))
        uRLLC_filenames.append('uRLLC_'+str(i))


services = emBB_filenames +  uRLLC_filenames + mMTC_filenames



# Number of clusters
num_clusters = 1

# List of alphas to be tested

alphas = [0.75]

# Define the number of training epochs
epochs = 30

# Define the number of training batch size

batch_size= 630

# Define the number of train, validation and test samples

train_samples = 40320
val_samples = 20160
test_samples = 10080


# Define the time admission and time decision 
time_admission = 120
time_decision = 30

# Define the number of blocks in time admission
N_dec= round(time_admission/time_decision)

# Define the input size and output size of the prediction
#input_size = 180
input_size = 480

#output_size = 30
output_size = N_dec

# Golden ratio for the golden search algorithm
gratio = (math.sqrt(5) + 1) / 2

# Stopping condition value for the golden search algorithm (interval length)
stop_value = 0.01


def val_level_dimension(levels,output_size):
    arr2 = [None] * output_size
    for i in range(len(arr2)):
        arr2[i]=levels
    return np.array(arr2).transpose()

# SIMULATION RUNS
num_runs = 1

####SHUFLE BATCHES
shuffle_batch=False



# Simulations over different services
for service in services:

    # Simulations over different alpha values
    for alpha in alphas:

        # Configuration loading
        config = get_config('Traffic', epochs, num_clusters, batch_size, train_samples, val_samples, test_samples, alpha, input_size, output_size,time_admission,time_decision,N_dec,shuffle_batch)
    
        # Data loading
       
        data=f'{datapath}/{service}.npy'
        

        train, val, test = create_dataset(data, config['chop_train'], config['chop_val'], config['chop_test'])
        
        dataset = Dataset(train, val, test, config['device'])
    
        # Maximum of single cluster traffic in the training set (for normalization)
        maximum = np.max(train[0])
    

        # Running many simulations for a given service and alpha
        for i in range(1, num_runs+1):
            
            #torch.cuda.manual_seed(123)
            # Initial extremes of the interval of the Minimum Level Threshold tau (expressed as fraction of maximum)
            tau_min = 0.0
            tau_max = 1.0
    
            # Current extremes of the interval of tau
            c = tau_min
            d = tau_max
    
            # Iterations counter for golden search algorithm
            iterations = 1
    
            # Dictionary collecting denormalized validation loss values for a given tau
            val_dict = {}
    

            # Stopping condition for golden search algorithm
            while abs(tau_max - tau_min) > stop_value:
        
                # Determine current Minimum Level Threshold tau
                if (iterations%3) > 0:
                    # Try tau as left extreme    
                    if (iterations%3) == 1:
                        tau = c
                    # Try tau as right extreme
                    else:
                        tau = d
        

                # Run actual golden search algorithm 
                else:

                    # Determine the new extreme of tau interval
                    if f_c < f_d:
                        # print("\nNew right-extreme of the interval is %f" % d)
                        tau_max = d
                    else:
                        # print("\nNew left-extreme of the interval is %f" % c)
                        tau_min = c
                
                    # print("Current length of tau interval is %f \n" % abs(tau_max - tau_min))
                    c = tau_max - (tau_max - tau_min) / gratio
                    d = tau_min + (tau_max - tau_min) / gratio
                    iterations = iterations + 1
                    continue
        

                # Compute denormalized validation loss for current tau
                f_val = val_dict.get(round(tau,6))
                # print("\nSearching a threshold in the interval [%f,%f]" % (tau_min, tau_max))
                # print("Threshold for this run is %f" % tau)
        

                # Denormalized validation loss not yet calculated for current tau
                if f_val == None:
        
                    # Dataloader initialization
                    dataloader = DataLoader(dataset, batch_size=config['series_batch'], shuffle=False)

                    # Model initialization
                    #run_id = f'Tadm_Variation/{time_admission}/{service}/Alpha_{alpha}/T_dec_{time_decision}_Tadm_{time_admission}'
                    run_id =f'{service}/Alpha_{alpha}/Simulation_{i}'
                    model = TESRNN(tau = tau, maximum = maximum, num_clusters = num_clusters, config = config, run_id = run_id)

                    # Run model trainer
                    trainer = TESRNNTrainer(model, dataloader, run_id, config)
                    trainer.train_epochs()
    
                    # Run model validator
                    validator = TESRNNValidator(model, dataloader, run_id, config)
                    validator.validating()
        
                    # Compute denormalized validation loss
                    norm_preds = np.load('Results/' + run_id + '/val_predictions.npy')
                    norm_actuals = np.load('Results/' + run_id + '/val_actuals.npy')
                    levels = np.load('Results/' + run_id + '/val_levels.npy')
                    levels=val_level_dimension(levels,N_dec)
                    print('OK')
                    val_loss = denorm_validation_loss(norm_preds, norm_actuals, levels, alpha)
                    print("Denormalized validation loss for this run %f" % val_loss)
                    val_dict[round(tau,6)] = val_loss

                   
                    file_path = os.path.join('Results',run_id, 'Epoch_validation_losses.csv')
                    with open(file_path, 'w') as f:
                        f.write('Epoch,Validation_loss\n')

                        # Store Validation loss of the current epoch
                        with open(file_path, 'a') as f:
                            f.write(','.join([str(iterations), str(val_loss)]) + '\n')



                    # Set denormalized validation loss for interval extreme
                    if (iterations%3) == 1:
                        f_c = val_loss
                    else:
                        f_d = val_loss
        

                # Denormalized validation loss already calculated for current tau
                else:
                    # print("Denormalized validation loss for this run %f" % f_val)
                    # Set denormalized validation loss for interval extreme
                    if (iterations%3) == 1:
                        f_c = f_val
                    else:
                        f_d = f_val
            

                # Increase algorithm iterations
                iterations = iterations + 1

        
    
            # Get the final optimal Minimum Level Threshold tau
            tau = (tau_min + tau_max) / 2
            # print('\nFinally chosen threshold = %f\n' % tau)
            np.save('Results/' + run_id + '/optimal_tau.npy', tau)
    


            # Run the optimized model
    
            # Dataloader initialization
            dataloader = DataLoader(dataset, batch_size=config['series_batch'], shuffle=False)
    
            # Model initialization
            model = TESRNN(tau = tau, maximum = maximum, num_clusters = num_clusters, config = config, run_id = run_id)
    
            # Run model trainer
            trainer = TESRNNTrainer(model, dataloader, run_id, config)
            trainer.train_epochs()
    
            # Run model tester
            tester = TESRNNTester(model, dataloader, run_id, config)
            tester.testing()

