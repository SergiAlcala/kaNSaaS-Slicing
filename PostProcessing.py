import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os




plt.style.use('/home/jupyter-salcala/Matplotlib/style.mplstyle')
import plotly.express as px


class Load_Data:
    def __init__(self, config):
        self.services = config['services']
        self.train_samples = config['train_samples']
        self.val_samples = config['val_samples']
        self.test_samples = config['test_samples']
        self.time_admission = config['time_admission']
        self.time_decision_NP = config['time_decision_NP']
        self.time_decision_SP = config['time_decision_SP']
        

        

    def load_preds(self,Results_dir,service,alpha,simulation,time_admission,time_decision,Load_type='AC_RA',Actuals=False):
        test_levels=np.load(f'{Results_dir}{service}/{alpha}/{simulation}/test_levels.npy')
        test_preds=np.load(f'{Results_dir}{service}/{alpha}/{simulation}/test_predictions.npy')
        if Actuals:
            test_preds=np.load(f'{Results_dir}{service}/{alpha}/{simulation}/test_actuals.npy')
 
        N_dec=Load_Data.N_dec(time_admission,time_decision)
        if N_dec==1:
            denorm_preds=test_preds*test_levels
        else:
            test_levels=Denormalize.val_level_dimension(test_levels,N_dec)
            denorm_preds=test_preds*test_levels
        if Load_type=='AC_RA':
            preds_times_df,preds_flat=Denormalize.select_predicted_blocks(denorm_preds,time_admission,time_decision) 
        elif Load_type=='RRA':
            preds_times_df,preds_flat=Denormalize.select_ReResource_alloctation(denorm_preds,time_decision)
            

        return preds_times_df,preds_flat

    def load_test_real(self,data_fpath):
        real_data=np.load(data_fpath)
    ## Select the time interval
        real_data=real_data[:self.train_samples+self.val_samples+ self.test_samples]
    ## Select the time interval for prediction Real Traffic
        real_data_test=real_data[- self.test_samples:]
        return real_data_test
    def N_dec(time_admision,time_decision):
        return round(time_admision/time_decision)
    def save_preds(self,Results_dir,service,alpha,simulation,time_admission,time_decision,Filesave,Load_type='AC_RA',Actuals=False):
        preds_times_df,preds_flat=self.load_preds(Results_dir,service,alpha,simulation,time_admission,time_decision,Load_type,Actuals)  
        FPath_save=f'{Filesave}/{alpha}/{simulation}/'
        if not os.path.exists(FPath_save):
            os.makedirs(FPath_save)
        if Load_type=='RRA':
            if Actuals:
                np.save(f'{FPath_save}{service}_{alpha}_Tdec_{time_decision}min_Actuals_RRA.npy',preds_flat)
            else:
                np.save(f'{FPath_save}{service}_{alpha}_Tdec_{time_decision}min_prediction_RRA.npy',preds_flat)
           
        else:
            if Actuals:
                np.save(f'{FPath_save}{service}_{alpha}_Tdec_{time_decision}min_Actuals.npy',preds_flat)
            else:
                np.save(f'{FPath_save}{service}_{alpha}_Tdec_{time_decision}min_prediction.npy',preds_flat)
        
        return f'{service}_saved correctly'


class Denormalize:
    def val_level_dimension(levels,output_size):
        arr2 = [None] * output_size
        for i in range(len(arr2)):
            arr2[i]=levels
        return np.array(arr2).transpose()
    def select_predicted_blocks(data,time_admission,time_decision):
        """Function to select the predicted blocks, taking into account the time_admission and time_decision. It gets the values from the row of i*T_admision and 
        fills the time decisions with the predicted values. We will use this created dataset to optimize the slices and retreive which slice are accepted or not. """

        df=pd.DataFrame(data)
        df_preds=pd.DataFrame()
        flat_array=np.zeros(len(df))
        for i in range(0,len(df),time_admission):
            df_preds=df_preds.append(df.iloc[i,:])
        df_preds_fin=df_preds
        df_preds=df_preds.reset_index(drop=True)
        for j in range(len(df_preds)):
            
            for k in range(round(time_admission/time_decision)):
                flat_array[(j*time_admission)+(k*time_decision):(j+1)*time_admission-time_decision*(round(time_admission/time_decision)-1-k)].fill(df_preds.iloc[j,k])
        
        return df_preds_fin,flat_array
    
    def select_ReResource_alloctation(data,time_decision):
            """Function to do the ReResource_allocation. Used in the second block of optimization. We update the values of each time decision every time decision, not time admission.
            It will be used for the second optimization problem, to retreive the alphas (% of traffic demand)"""
            df=pd.DataFrame(data)
            df_preds=pd.DataFrame()
            flat_array=np.zeros(len(df))
            for i in range(0,len(data),time_decision):
                df_preds=df_preds.append(df.iloc[i,:])
                df_preds_fin=df_preds
                df_preds=df_preds.reset_index(drop=True)
            for j in range(0,len(df_preds)):
                flat_array[j*time_decision:(j+1)*time_decision].fill(df_preds.iloc[j,0])

            
            return df_preds_fin,flat_array
    
    def dimensioning_data(data,test_samples,input_size,time_admission):
        preds_test_dimension=np.zeros(test_samples)
        preds_test_dimension[preds_test_dimension==0]=np.nan
        preds_test_dimension[input_size:-(time_admission-1)]=data
        return preds_test_dimension