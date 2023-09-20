

def get_config(services,Alphas,Simulations,time_admission,time_decision_NP,time_decision_SP,train_samples,val_samples,test_samples,input_size):

    config = {
        'services': services,
        'Alphas': Alphas,
        'Simulations': Simulations,
        'time_admission': time_admission,
        'time_decision_NP': time_decision_NP,
        'time_decision_SP': time_decision_SP,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
        'input_size': input_size,
        
    }
    return config

