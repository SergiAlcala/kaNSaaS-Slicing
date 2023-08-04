# Configuration File

from math import sqrt
import torch
import os

# Get configuration parameters from main
def get_config(interval, epochs, num_clusters, batch_size, train_samples, val_samples, test_samples, alpha, input_size, output_size, time_admission, time_decision,N_dec,shuffle_batch):
    
    # Deep Learning Parameters
    config = {
        'device': ("cuda" if torch.cuda.is_available() else "cpu"),
        'percentile': 50,
        'training_percentile': 45,
        'add_nl_layer': True,
        'rnn_cell_type': 'LSTM',
        'learning_rate': 1e-3,
        'learning_rates': ((10, 1e-4)),
        'num_of_train_epochs': epochs,
        'series_batch': 1,
        'batch_size': batch_size,
        'gradient_clipping': 20,
        'c_state_penalty': 0,
        'min_learning_rate': 0.0001,
        'lr_ratio': sqrt(10),
        'lr_tolerance_multip': 1.005,
        'min_epochs_before_changing_lrate': 2,
        'lr_anneal_rate': 0.5,
        'lr_anneal_step': 5,
    }

    # Traffic Forecasting Parameters
    if interval == 'Traffic':
        config.update({
            'chop_train': train_samples,
            'chop_val': val_samples,
            'chop_test': test_samples,
            'variable': "Traffic",
            'dilations': ((1, 3), (6, 12)),
            'state_hsize': 50,
            'input_size': input_size,
            'output_size': output_size,
            'num_clusters': num_clusters,
            'alpha': alpha,
            'time_admission': time_admission,
            'time_decision': time_decision,
            'N_dec': N_dec,
            'shuffle_batch':shuffle_batch
        })    
    else:
        print("I don't have that config. :(")


    return config

            