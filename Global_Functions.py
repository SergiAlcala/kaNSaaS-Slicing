import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob as glob
from natsort import  natsorted
from tqdm import tqdm
import json
import os
import plotly.graph_objects as go
import plotly.express as px
# from pyscipopt import Model
# import pyscipopt
from results_json import get_3Dresults, get_results
import shutil
# from ortools.linear_solver import pywraplp
import pickle





class threeD:
    def go():
        return go
    def px():
        return px


class Functions:
    def __init__(self) -> None:
        pass

     
   
#######    SLA DEFINITIONS ##########################

    def plot_5x(alpha):
        
        df=5*alpha-4
        df[df<=-1]=-1

        return df
    
    def plot_cost(q,alpha,traffic,m=1):
        return (q*alpha*traffic)*m
    
    def plot_SLA_Cost(traffic, Beta, SLA=plot_5x):
        """function for the cost of operation"""
        sla_beta=SLA(Beta)
        subtraction=(1-sla_beta)
        result=subtraction*traffic
        return result


    def gain_lin_approx_5(alpha):

        return 5 * alpha - 4 
            
         
    def cOpex(cost_per_bps, traffic, alpha):

        """function for the cost of operation"""
        cost_list=[]
        for i in range(len(alpha)):
            cost_list.append(cost_per_bps*alpha[i]*traffic[i])
        cost= np.sum(cost_list)

        return cost

    def revenue(revenue_per_bps, traffic, alpha, SLA=lambda x:5*x-4):

        """function for the cost of operation"""
        rev_list=[]

        for i in range(len(alpha)):
            rev_list.append(revenue_per_bps*SLA(alpha[i])*traffic[i])
        revenue = np.sum(rev_list)

        return revenue

    
        
    
#######    SLA DEFINITIONS ##########################
    def namestr(obj, namespace):
        
        return [name for name in namespace if namespace[name] is obj]

   

    def select_num_slices(df, num_slices):
        return df[list(range(num_slices))]

    def load_data(Data_Fpath,typee,output_list,services_name_list,printt=False):
        for f in natsorted(glob.glob(os.path.join(Data_Fpath,typee))):
        
            output_list.append(np.load(f))
            services_name_list.append(f.split('/')[-1].split('.')[0])
            if printt == True:
                print(f)
            if not output_list:
                print ('Load Error, check Filepath Data :'+str(typee))

    def load_data_CSV(Data_Fpath,typee,output_list,services_name_list,printt=False):
        for f in natsorted(glob.glob(Data_Fpath+typee)):
            df=pd.read_csv(f)
            df['cumulative']=df['vol_up']+df['vol_dn']
            output_list.append(df.cumulative)
            services_name_list.append(f.split('/')[-1].split('.')[0])
            if printt == True:
                print(f)
            if not output_list:
        
                print ('Load Error, check Filepath Data :'+str(typee))
    
    def load_minmax(fpath):
        li=[]
        for f in natsorted(glob.glob(fpath+'*.csv')):
            df=pd.read_csv(f)
            li.append(df)
        frame=pd.concat(li,axis=0,ignore_index=True)
        frame = frame.loc[:, ~frame.columns.str.contains('^Unnamed')].T
        return frame
    def load_variables(Data_Fpath,typee,output_list,printt=False):
        
        for f in natsorted(glob.glob(Data_Fpath+typee)):

            output_list.append(np.load(f))
            if printt == True:
                print(f)
            if not output_list:
                print ('Load Error, check Filepath Variables :'+ str(typee) )

    def flatten(t):
        return [item for sublist in t for item in sublist]
    
    def flatten_slices(lists):
        output=[]
        for i in range(len(lists)):
           output.append(Functions.flatten(lists[i]))
        return output
    
    def spread_data(ls,Decision_time):
        Data_list=[[] for i in range(len(ls))]
        for i in range(len(ls)):
            for j in range(len(ls[i])):
                for k in range(len(ls[i][j])):
                    Data_list[i].append(list(np.full(Decision_time,ls[i][j][k])))
        return Data_list

    def spread_data_df(ls,Decision_time,Tslots):
        df=pd.DataFrame(ls)
        df_new=pd.DataFrame(0,index=range(Tslots),columns=range(len(df.columns)))
        for i in range(len(df.columns)):
            for j in range(len(df.index)):
                df_new.iloc[Decision_time*j: (j+1)*Decision_time,i]=df.iloc[j,i]
        return df_new
    
    def list_to_df(listt):
        df=pd.DataFrame(listt).T
        return df

    def convert0_nan(df):
        df.replace(0.0,np.nan,inplace=True)
        return df
 
    def service_accepted(df):
        service=pd.DataFrame()
        for i in range(len(df.columns)):
            service[i]=np.where(df[i]>0,i,np.nan)
        return service
    
    def total_service_accepted(df):
        service=pd.DataFrame()
        for i in range(len(df.columns)):
        
            service[i]=np.where(df[i]>0,1,np.nan)
        service['total_slices']=service.sum(axis=1)
        return service
    
    def slice_names(slices):
        Slice_list=[]
        for i in range(len(slices)):
            Slice_list.append(slices[i].split('_traff')[0])
        
        if '_' in slices[0].split('_traff')[0]:
            Slice_list=[]
            for i in range(len(slices)):
                 Slice_list.append(slices[i].split('_traff')[0].split(str(i)+'_')[-1])

        return Slice_list
    
    def savefigs(FPath_save,Date,NameFig,formatt):
        if not os.path.exists(FPath_save):
            os.makedirs(FPath_save)
        plt.savefig(FPath_save+Date+'_'+NameFig+'.'+formatt)

    def savefigs_3D(fig,FPath_save,Date,NameFig):
        if not os.path.exists(FPath_save):
            os.makedirs(FPath_save)
        fig.write_html(f'{FPath_save}/{NameFig}.html')

    def load_minmax(Fpath_Dataset):
        dfs=[]
        for f in natsorted(glob.glob(Fpath_Dataset+'*.csv')):
                    dfs.append(pd.read_csv(f))
        bigframe=pd.concat(dfs,ignore_index=True)

        #.sort_values('Slice', key=lambda col: col.str.lower()).reset_index(drop=True)
        bigframe=bigframe.loc[:,~bigframe.columns.str.match("Unnamed")]
        bigframe=bigframe.T
        return bigframe
    
    def inv_norm(norm_data,Minmax_values):

        return (Minmax_values.loc['Max']-Minmax_values.loc['Min'])*norm_data +Minmax_values.loc['Min']

    def beta_selection(alpha,TRAF,OB,SP):
        """Beta= min(1,min(TRAF,alpha*OB)/min(TRAF,SP))"""
        df1=pd.concat([TRAF,alpha*OB]).groupby(level=0).min()
        df2=pd.concat([TRAF,SP]).groupby(level=0).min()
        df3=df1/df2
        pd_1=pd.DataFrame(1,index=df1.index,columns=df1.columns)

        betta=pd.concat([pd_1,df3]).groupby(level=0).min()

        #### OB is the one that can change depending the evaluated set ######
              
        return betta

class LDV:
    def __init__(self) -> None:
        pass
    
    
    def load_dataset_variables(Fpath_Dataset,Fpath_variables,Tslots,SLA,output,n_slices=20,Ctot_ratio=1,Bss_type=False,Tdec_OB=30,Tdec_SP=120,Predictions=True,synthetic=False,q=0.5):
        from Global_Functions import Functions as F
        
        SP=[]
        OB=[]
        TRAF=[]
        SP_names=[]
        OB_names=[]
        TRAF_names=[]
        
         
        typee=['*SP_120min.npy','*OB_30min.npy','*traff.npy'] 
        Alphas=['alpha*SP*.npy','alpha*OB*.npy','alpha*traff*.npy']
        Xs=['X*SP*.npy','X*OB*.npy','X*traff*.npy']
        if Predictions==True:
            typee=['*Alpha_1*.npy','*Alpha_0*.npy','*_clean.npy'] 
            Alphas=['*alpha*Alpha_1*.npy','*alpha*Alpha_0*.npy','*alpha*clean*.npy']
            Xs=['X*Alpha_1*.npy','X*Alpha_0*.npy','X*clean*.npy']
        if synthetic==True:
            typee=['*Alpha_1*.npy','*Alpha_0*.npy','*agg_60_s*.npy']
            Alphas=['alpha*Alpha_1*.npy','*alpha*Alpha_0*.npy','*alpha*agg_60_s*.npy']
            Xs=['X*Alpha_1*.npy','X*Alpha_0*.npy','X*agg_60_s*.npy']
        
    
        F.load_data(Fpath_Dataset,typee[0],output_list=SP,services_name_list=SP_names,printt=False)
        F.load_data(Fpath_Dataset,typee[1],output_list=OB,services_name_list=OB_names,printt=False)
        F.load_data(Fpath_Dataset,typee[2],output_list=TRAF,services_name_list=TRAF_names,printt=False)
        
    
            ### Service Provider
        alpha_SP=[]
        X_SP=[]
        F.load_variables(Fpath_variables,Alphas[0],output_list=alpha_SP,printt=False)
        F.load_variables(Fpath_variables,Xs[0],output_list=X_SP,printt=False)
    
        
        ## Overbooking
        alpha_OB=[]
        X_OB=[]
        F.load_variables(Fpath_variables,Alphas[1],output_list=alpha_OB,printt=False)
        F.load_variables(Fpath_variables,Xs[1],output_list=X_OB,printt=False)


        ## Traffic
        alpha_TRAFF=[]
        X_TRAFF=[]
    
        F.load_variables(Fpath_variables,Alphas[2],output_list=alpha_TRAFF,printt=False)
        F.load_variables(Fpath_variables,Xs[2],output_list=X_TRAFF,printt=False)

        ###### CHANGE DECISION TIMES #######
        Decision_time_SP=Tdec_SP
        Decision_time_OB=Tdec_OB
        if Decision_time_OB>Decision_time_SP:
            Decision_time_SP=Decision_time_OB

        Decision_time_TRAF=1

        X_SP_flat=F.flatten_slices(F.spread_data(X_SP,Decision_time_SP))
        alpha_SP_flat=F.flatten_slices(F.spread_data(alpha_SP,Decision_time_SP))
        try:
            X_OB_flat=F.flatten_slices(F.spread_data(X_OB,Decision_time_OB))
            alpha_OB_flat=F.flatten_slices(F.spread_data(alpha_OB,Decision_time_OB))
        except:
            X_OB_flat=X_OB
            alpha_OB_flat=F.flatten_slices(F.spread_data(alpha_OB,Decision_time_OB))
            

        X_TRAF_flat=F.flatten_slices(F.spread_data(X_TRAFF,Decision_time_TRAF))
        alpha_TRAF_flat=F.flatten_slices(F.spread_data(alpha_TRAFF,Decision_time_TRAF))
        if Bss_type ==True:
            TRAF=Functions.list_to_df(TRAF)
            OB=Functions.list_to_df(OB)
            SP=Functions.list_to_df(SP)
            TRAF=TRAF.replace(np.nan,0).T
            OB=OB.replace(np.nan,0).T
            SP=SP.replace(np.nan,0).T
            TRAF=list(TRAF.values)
            OB=list(OB.values)
            SP=list(SP.values)

            X_SP_flat=Functions.list_to_df(X_SP_flat)
            X_OB_flat=Functions.list_to_df(X_OB_flat)
            X_TRAF_flat=Functions.list_to_df(X_TRAF_flat)
            X_SP_flat=X_SP_flat.replace(np.nan,0).T
            X_OB_flat=X_OB_flat.replace(np.nan,0).T
            X_TRAF_flat=X_TRAF_flat.replace(np.nan,0).T
            X_SP_flat=list(X_SP_flat.values)
            X_OB_flat=list(X_OB_flat.values)
            X_TRAF_flat=list(X_TRAF_flat.values)

            alpha_SP_flat=Functions.list_to_df(alpha_SP_flat)
            alpha_OB_flat=Functions.list_to_df(alpha_OB_flat)
            alpha_TRAF_flat=Functions.list_to_df(alpha_TRAF_flat)
            alpha_SP_flat=alpha_SP_flat.replace(np.nan,0).T
            alpha_OB_flat=alpha_OB_flat.replace(np.nan,0).T
            alpha_TRAF_flat=alpha_TRAF_flat.replace(np.nan,0).T
            alpha_SP_flat=list(alpha_SP_flat.values)
            alpha_OB_flat=list(alpha_OB_flat.values)
            alpha_TRAF_flat=list(alpha_TRAF_flat.values)


    
        Tslots=Tslots
        #Convert the traffic to a dataframe
        SP_pd=pd.DataFrame(SP).T[:Tslots]
        OB_pd=pd.DataFrame(OB).T[:Tslots]
        TRAF_pd=pd.DataFrame(TRAF).T[:Tslots]

        #Convert the alpha and X to a dataframe
        alpha_SP_flat_pd=pd.DataFrame(alpha_SP_flat).T[:Tslots]
        X_SP_flat_pd=pd.DataFrame(X_SP_flat).T[:Tslots]

        alpha_OB_flat_pd=pd.DataFrame(alpha_OB_flat).T[:Tslots]
        X_OB_flat_pd=pd.DataFrame(X_OB_flat).T[:Tslots]

        alpha_TRAF_flat_pd=pd.DataFrame(alpha_TRAF_flat).T[:Tslots]
        X_TRAF_flat_pd=pd.DataFrame(X_TRAF_flat).T[:Tslots]

        #### For beta, we change the alpha (1st parameter) and the set to evaluate (SP,OB,TRAF) in the 3rd parameter
        beta_SP_pd=F.beta_selection(alpha_SP_flat_pd,TRAF_pd,SP_pd,SP_pd)
        beta_OB_pd=F.beta_selection(alpha_OB_flat_pd,TRAF_pd,OB_pd,SP_pd)
        beta_TRAF_pd=F.beta_selection(alpha_TRAF_flat_pd,TRAF_pd,TRAF_pd,SP_pd)

        # Revenue
        value_OPT_List_SP = SP_pd*X_SP_flat_pd*SLA(beta_SP_pd)
        value_OPT_List_OB = SP_pd*X_OB_flat_pd*SLA(beta_OB_pd)
        value_OPT_List_TRAF = SP_pd*X_TRAF_flat_pd*SLA(beta_TRAF_pd)

        # SLA Cost
        SLA_Cost_SP = X_SP_flat_pd*F.plot_SLA_Cost(SP_pd,beta_SP_pd)
        SLA_Cost_OB = X_OB_flat_pd*F.plot_SLA_Cost(SP_pd,beta_OB_pd)
        SLA_Cost_TRAF = X_TRAF_flat_pd*F.plot_SLA_Cost(SP_pd,beta_TRAF_pd)
        
        # COPEX Cost
        Copex_Cost_SP = X_SP_flat_pd*F.plot_cost(q,alpha_SP_flat_pd,SP_pd)
        Copex_Cost_OB = X_OB_flat_pd*F.plot_cost(q,alpha_OB_flat_pd,OB_pd)
        Copex_Cost_TRAF = X_TRAF_flat_pd*F.plot_cost(q,alpha_TRAF_flat_pd,TRAF_pd)

        # Net Benefit
        Net_Ben_OPT_List_SP = X_SP_flat_pd*(SP_pd*SLA(beta_SP_pd)-F.plot_cost(q,alpha_SP_flat_pd,SP_pd))
        Net_Ben_OPT_List_OB = X_OB_flat_pd*(SP_pd*SLA(beta_OB_pd)-F.plot_cost(q,alpha_OB_flat_pd,OB_pd))
        Net_Ben_OPT_List_TRAF = X_TRAF_flat_pd*(SP_pd*SLA(beta_TRAF_pd)-F.plot_cost(q,alpha_TRAF_flat_pd,TRAF_pd))

        # Weight, e.g. the traffic allocated to each slice
        weight_OPT_list_SP = SP_pd*X_SP_flat_pd*alpha_SP_flat_pd
        weight_OPT_list_OB = OB_pd*X_OB_flat_pd*alpha_OB_flat_pd
        weight_OPT_list_TRAF = TRAF_pd*X_TRAF_flat_pd*alpha_TRAF_flat_pd

        # Real traffic, e.g. the traffic that actually is used for each slice
        real_traffic_OPT_List_SP = TRAF_pd*X_SP_flat_pd
        real_traffic_OPT_List_OB = TRAF_pd*X_OB_flat_pd
        real_traffic_OPT_List_TRAF = TRAF_pd*X_TRAF_flat_pd

        # Expected traffic, e.g. the traffic that is expected to be used for each slice
        expected_traffic_OPT_List_SP =SP_pd*X_SP_flat_pd
        expected_traffic_OPT_List_OB = OB_pd*X_OB_flat_pd
        expected_traffic_OPT_List_TRAF = TRAF_pd*X_TRAF_flat_pd

        # Convert all 0 to NaN to avoid problems with the metrics
        value_SP_pd = F.convert0_nan(value_OPT_List_SP)
        value_OB_pd = F.convert0_nan(value_OPT_List_OB)
        value_TRAF_pd = F.convert0_nan(value_OPT_List_TRAF)
        weight_SP_pd = F.convert0_nan(weight_OPT_list_SP)
        weight_OB_pd = F.convert0_nan(weight_OPT_list_OB)
        weight_TRAF_pd = F.convert0_nan(weight_OPT_list_TRAF)
        real_traffic_SP_pd = F.convert0_nan(real_traffic_OPT_List_SP)
        real_traffic_OB_pd = F.convert0_nan(real_traffic_OPT_List_OB)
        real_traffic_TRAF_pd = F.convert0_nan(real_traffic_OPT_List_TRAF)
        expected_traff_OPT_List_SP_pd = F.convert0_nan(expected_traffic_OPT_List_SP)
        expected_traff_OPT_List_OB_pd = F.convert0_nan(expected_traffic_OPT_List_OB)
        expected_traff_OPT_List_TRAF_pd = F.convert0_nan(expected_traffic_OPT_List_TRAF)
        service_SP_pd = F.service_accepted(weight_SP_pd)
        service_OB_pd = F.service_accepted(weight_OB_pd)
        service_TRAF_pd = F.service_accepted(weight_TRAF_pd)
        total_service_SP_pd = F.total_service_accepted(weight_SP_pd)
        total_service_OB_pd = F.total_service_accepted(weight_OB_pd)
        total_service_TRAF_pd = F.total_service_accepted(weight_TRAF_pd)
        Net_Ben_OPT_List_SP = F.convert0_nan(Net_Ben_OPT_List_SP)
        Net_Ben_OPT_List_OB = F.convert0_nan(Net_Ben_OPT_List_OB)
        Net_Ben_OPT_List_TRAF = F.convert0_nan(Net_Ben_OPT_List_TRAF)      
       


        if output== 'All_results':
            TRAF_pd=pd.DataFrame(TRAF[:n_slices]).T
            Ctot=max(TRAF_pd[:Tslots].sum(axis=1))*Ctot_ratio

            gain_list_SP, gain_SP = Metrics.rho_gain(value_SP_pd,value_TRAF_pd)
            gain_list_OB, gain_OB = Metrics.rho_gain(value_OB_pd,value_TRAF_pd)
            traffic_list_SP, traffic_SP = Metrics.rho_gain(real_traffic_SP_pd,real_traffic_TRAF_pd)
            traffic_list_OB, traffic_OB = Metrics.rho_gain(real_traffic_OB_pd,real_traffic_TRAF_pd)
            Euro_list_OB, Euro_OB = Metrics.rho_gain(Net_Ben_OPT_List_OB,weight_OB_pd)
            Euro_list_TRAF, Euro_TRAF = Metrics.rho_gain(Net_Ben_OPT_List_TRAF,weight_TRAF_pd)
            Euro_list_SP, Euro_SP = Metrics.rho_gain(Net_Ben_OPT_List_SP,weight_SP_pd)
            Eurotraf_list_OB, Eurotraf_OB = Metrics.rho_gain_TRAF(Euro_list_OB,Euro_list_TRAF)
            Eurotraf_list_SP, Eurotraf_SP = Metrics.rho_gain_TRAF(Euro_list_SP,Euro_list_TRAF)
            
            NET_GAIN_list_SP,NET_GAIN_SP = Metrics.rho_gain(Net_Ben_OPT_List_SP,Net_Ben_OPT_List_TRAF)
            NET_GAIN_list_OB,NET_GAIN_OB = Metrics.rho_gain(Net_Ben_OPT_List_OB,Net_Ben_OPT_List_TRAF)
            allocated_Traffic_list_SP,allocated_Traffic_SP = Metrics.rho_gain(weight_SP_pd,weight_TRAF_pd)
            allocated_Traffic_list_OB,allocated_Traffic_OB = Metrics.rho_gain(weight_OB_pd,weight_TRAF_pd)

            SLA_Cost_Metric_list_SP,SLA_Cost_Metric_SP = Metrics.rho_gain(SLA_Cost_SP,SLA_Cost_TRAF)
            SLA_Cost_Metric_list_OB,SLA_Cost_Metric_OB = Metrics.rho_gain(SLA_Cost_OB,SLA_Cost_TRAF)
           
            Copex_cost_Metric_list_SP,Copex_cost_Metric_SP = Metrics.rho_gain(Copex_Cost_SP,Copex_Cost_TRAF)
            Copex_cost_Metric_list_OB,Copex_cost_Metric_OB = Metrics.rho_gain(Copex_Cost_OB,Copex_Cost_TRAF)
            


            all_results = get_results(SP_pd,OB_pd,TRAF_pd,F.slice_names(TRAF_names),alpha_SP_flat_pd,X_SP_flat_pd,alpha_OB_flat_pd,X_OB_flat_pd,alpha_TRAF_flat_pd,X_TRAF_flat_pd,Ctot,
            value_SP_pd ,value_OB_pd,value_TRAF_pd,weight_SP_pd,weight_OB_pd,weight_TRAF_pd,real_traffic_SP_pd,real_traffic_OB_pd,real_traffic_TRAF_pd,expected_traff_OPT_List_SP_pd,
            expected_traff_OPT_List_OB_pd,expected_traff_OPT_List_TRAF_pd,service_SP_pd,service_OB_pd,service_TRAF_pd,total_service_SP_pd,total_service_OB_pd,
            total_service_TRAF_pd,gain_list_SP,gain_SP,gain_list_OB,gain_OB,traffic_list_SP,traffic_SP,traffic_list_OB,traffic_OB,Eurotraf_list_OB,Eurotraf_OB,Eurotraf_list_SP,Eurotraf_SP,
            Euro_list_SP,Euro_SP,Euro_list_OB,Euro_OB,Euro_list_TRAF,Euro_TRAF,beta_SP_pd,beta_OB_pd,beta_TRAF_pd,Net_Ben_OPT_List_SP ,Net_Ben_OPT_List_OB,Net_Ben_OPT_List_TRAF,NET_GAIN_list_SP
            ,NET_GAIN_SP,NET_GAIN_list_OB,NET_GAIN_OB,allocated_Traffic_list_SP,allocated_Traffic_SP,allocated_Traffic_list_OB,allocated_Traffic_OB,SLA_Cost_SP,SLA_Cost_OB,SLA_Cost_TRAF,Copex_Cost_SP,
            Copex_Cost_OB,Copex_Cost_TRAF,SLA_Cost_Metric_list_SP,SLA_Cost_Metric_SP,SLA_Cost_Metric_list_OB,SLA_Cost_Metric_OB,Copex_cost_Metric_list_SP,Copex_cost_Metric_SP
            ,Copex_cost_Metric_list_OB,Copex_cost_Metric_OB)
            return all_results

        else:
            print('Error, check output parameter, it must be "All_results"')


    def load_dataset_variables_benchmark(Fpath_Dataset,Fpath_variables,Tslots,SLA,output,n_slices=20,Ctot_ratio=1,Bss_type=True,Tdec_OB=30,Tdec_SP=120,Predictions=True,q=0.5,synthetic=False):
        from Global_Functions import Functions as F
        for f in glob.glob(Fpath_Dataset+'/*.pkl'):
            with open(f, 'rb')  as pkl_sol:
                x = pickle.load( pkl_sol)
                z = pickle.load( pkl_sol)  



        SP=[]
        OB=[]
        TRAF=[]
        SP_names=[]
        OB_names=[]
        TRAF_names=[]
        
            
        
        typee=['*Alpha_1*.npy','*Alpha_0*.npy','*_clean.npy'] 
        Alphas=['*alpha*Alpha_1*.npy','*alpha*Alpha_0*.npy','*alpha*clean*.npy']
        Xs=['X*Alpha_1*.npy','X*Alpha_0*.npy','X*clean*.npy']

        if synthetic==True:
            typee=['*Alpha_1*.npy','*Alpha_0*.npy','*agg_60_s*.npy']
            Alphas=['alpha*Alpha_1*.npy','*alpha*Alpha_0*.npy','*alpha*agg_60_s*.npy']
            Xs=['X*Alpha_1*.npy','X*Alpha_0*.npy','X*agg_60_s*.npy']
        
        F.load_data(Fpath_Dataset,typee[0],output_list=SP,services_name_list=SP_names,printt=False)
        
        F.load_data(Fpath_Dataset,typee[2],output_list=TRAF,services_name_list=TRAF_names,printt=False)

        ## Service Provider
        alpha_SP=[]
        X_SP=[]
        F.load_variables(Fpath_variables,Alphas[0],output_list=alpha_SP,printt=False)
        F.load_variables(Fpath_variables,Xs[0],output_list=X_SP,printt=False)
        
        ## Overbooking
        alpha_OB=z
        X_OB=x
        
        
        
        ## Traffic
        alpha_TRAFF=[]
        X_TRAFF=[]
        
        F.load_variables(Fpath_variables,Alphas[2],output_list=alpha_TRAFF,printt=False)
        F.load_variables(Fpath_variables,Xs[2],output_list=X_TRAFF,printt=False)

        Decision_time_SP=Tdec_SP
        Decision_time_OB=Tdec_OB
        Decision_time_TRAF=1

        X_SP_flat=F.flatten_slices(F.spread_data(X_SP,Decision_time_SP))
        alpha_SP_flat=F.flatten_slices(F.spread_data(alpha_SP,Decision_time_SP))


        alpha_OB_flat=F.spread_data_df(alpha_OB,Decision_time_OB,Tslots=Tslots)
        X_OB_flat=F.spread_data_df(X_OB,Decision_time_OB,Tslots=Tslots)



        X_TRAF_flat=F.flatten_slices(F.spread_data(X_TRAFF,Decision_time_TRAF))
        alpha_TRAF_flat=F.flatten_slices(F.spread_data(alpha_TRAFF,Decision_time_TRAF))

        
   
        SP_pd=pd.DataFrame(SP).T[:Tslots]

        TRAF_pd=pd.DataFrame(TRAF).T[:Tslots]

        alpha_SP_flat_pd=pd.DataFrame(alpha_SP_flat).T[:Tslots]
        X_SP_flat_pd=pd.DataFrame(X_SP_flat).T[:Tslots]

        alpha_OB_flat_pd=alpha_OB_flat
        X_OB_flat_pd=X_OB_flat


        alpha_TRAF_flat_pd=pd.DataFrame(alpha_TRAF_flat).T[:Tslots]
        X_TRAF_flat_pd=pd.DataFrame(X_TRAF_flat).T[:Tslots]

        beta_SP_pd=F.beta_selection(alpha_SP_flat_pd,TRAF_pd,SP_pd,SP_pd)

        beta_OB_pd=F.beta_selection(alpha_OB_flat,TRAF_pd,1,SP_pd)
        
        beta_TRAF_pd=F.beta_selection(alpha_TRAF_flat_pd,TRAF_pd,TRAF_pd,SP_pd)

        Net_Ben_OPT_List_SP=X_SP_flat_pd*(SP_pd*SLA(beta_SP_pd)-F.plot_cost(q,alpha_SP_flat_pd,SP_pd))
        Net_Ben_OPT_List_OB=X_OB_flat_pd*(SP_pd*SLA(beta_OB_pd)-F.plot_cost(q,alpha_OB_flat_pd,1))
        Net_Ben_OPT_List_TRAF=X_TRAF_flat_pd*(SP_pd*SLA(beta_TRAF_pd)-F.plot_cost(q,alpha_TRAF_flat_pd,TRAF_pd))
        weight_OPT_list_SP=SP_pd*X_SP_flat_pd*alpha_SP_flat_pd
        weight_OPT_list_OB=X_OB_flat_pd*alpha_OB_flat_pd
        weight_OPT_list_TRAF=TRAF_pd*X_TRAF_flat_pd*alpha_TRAF_flat_pd

        real_traffic_OPT_List_SP=TRAF_pd*X_SP_flat_pd
        real_traffic_OPT_List_OB=TRAF_pd*X_OB_flat_pd
        real_traffic_OPT_List_TRAF=TRAF_pd*X_TRAF_flat_pd

        
        weight_SP_pd=F.convert0_nan(weight_OPT_list_SP)
        weight_OB_pd=F.convert0_nan(weight_OPT_list_OB)
        weight_TRAF_pd=F.convert0_nan(weight_OPT_list_TRAF)
        real_traffic_SP_pd=F.convert0_nan(real_traffic_OPT_List_SP)
        real_traffic_OB_pd=F.convert0_nan(real_traffic_OPT_List_OB)
        real_traffic_TRAF_pd=F.convert0_nan(real_traffic_OPT_List_TRAF)

        service_SP_pd=F.service_accepted(weight_SP_pd)
        service_OB_pd=F.service_accepted(weight_OB_pd)
        service_TRAF_pd=F.service_accepted(weight_TRAF_pd)
        total_service_SP_pd=F.total_service_accepted(weight_SP_pd)
        total_service_OB_pd=F.total_service_accepted(weight_OB_pd)
        total_service_TRAF_pd=F.total_service_accepted(weight_TRAF_pd)
        Net_Ben_OPT_List_SP=F.convert0_nan(Net_Ben_OPT_List_SP)
        Net_Ben_OPT_List_OB=F.convert0_nan(Net_Ben_OPT_List_OB)
        Net_Ben_OPT_List_TRAF=F.convert0_nan(Net_Ben_OPT_List_TRAF)

        SLA_Cost_SP=X_SP_flat_pd*F.plot_SLA_Cost(SP_pd,beta_SP_pd)
        SLA_Cost_OB=X_OB_flat_pd*F.plot_SLA_Cost(SP_pd,beta_OB_pd)
        SLA_Cost_TRAF=X_TRAF_flat_pd*F.plot_SLA_Cost(SP_pd,beta_TRAF_pd)


        Copex_Cost_SP=X_SP_flat_pd*F.plot_cost(q,alpha_SP_flat_pd,SP_pd)
        Copex_Cost_OB=X_OB_flat_pd*F.plot_cost(q,alpha_OB_flat_pd,1)
        Copex_Cost_TRAF=X_TRAF_flat_pd*F.plot_cost(q,alpha_TRAF_flat_pd,TRAF_pd)

    
        SLA_Cost_Metric_list_SP,SLA_Cost_Metric_SP=Metrics.rho_gain(SLA_Cost_SP,SLA_Cost_TRAF)
        SLA_Cost_Metric_list_OB,SLA_Cost_Metric_OB=Metrics.rho_gain(SLA_Cost_OB,SLA_Cost_TRAF)
        
        Copex_cost_Metric_list_SP,Copex_cost_Metric_SP=Metrics.rho_gain(Copex_Cost_SP,Copex_Cost_TRAF)
        Copex_cost_Metric_list_OB,Copex_cost_Metric_OB=Metrics.rho_gain(Copex_Cost_OB,Copex_Cost_TRAF)
            



       
        if output== 'All_results':
            TRAF_pd=pd.DataFrame(TRAF[:n_slices]).T
            Ctot=max(TRAF_pd[:Tslots].sum(axis=1))*Ctot_ratio
            value_SP_pd=1
            value_TRAF_pd=1
            value_OB_pd=1
            expected_traff_OPT_List_SP_pd=1
            expected_traff_OPT_List_OB_pd=1
            expected_traff_OPT_List_TRAF_pd=1

            gain_list_SP, gain_SP=1,1
            gain_list_OB, gain_OB=1,1
           
            traffic_list_SP, traffic_SP=Metrics.rho_gain(real_traffic_SP_pd,real_traffic_TRAF_pd)
            traffic_list_OB, traffic_OB=Metrics.rho_gain(real_traffic_OB_pd,real_traffic_TRAF_pd)
            Euro_list_OB, Euro_OB=Metrics.rho_gain(Net_Ben_OPT_List_OB,weight_OB_pd)
            Euro_list_TRAF, Euro_TRAF=Metrics.rho_gain(Net_Ben_OPT_List_TRAF,weight_TRAF_pd)
            Euro_list_SP, Euro_SP=Metrics.rho_gain(Net_Ben_OPT_List_SP,weight_SP_pd)
            Eurotraf_list_OB, Eurotraf_OB=Metrics.rho_gain_TRAF(Euro_list_OB,Euro_list_TRAF)
            Eurotraf_list_SP, Eurotraf_SP=Metrics.rho_gain_TRAF(Euro_list_SP,Euro_list_TRAF)
            
            NET_GAIN_list_SP,NET_GAIN_SP=Metrics.rho_gain(Net_Ben_OPT_List_SP,Net_Ben_OPT_List_TRAF)
            NET_GAIN_list_OB,NET_GAIN_OB=Metrics.rho_gain(Net_Ben_OPT_List_OB,Net_Ben_OPT_List_TRAF)
            allocated_Traffic_list_SP,allocated_Traffic_SP=Metrics.rho_gain(weight_SP_pd,weight_TRAF_pd)
            allocated_Traffic_list_OB,allocated_Traffic_OB=Metrics.rho_gain(weight_OB_pd,weight_TRAF_pd)

            
            all_results=get_results(SP_pd,OB,TRAF_pd,F.slice_names(TRAF_names),alpha_SP_flat_pd,X_SP_flat_pd,alpha_OB_flat_pd,X_OB_flat_pd,alpha_TRAF_flat_pd,X_TRAF_flat_pd,Ctot,
            value_SP_pd ,value_OB_pd,value_TRAF_pd,weight_SP_pd,weight_OB_pd,weight_TRAF_pd,real_traffic_SP_pd,real_traffic_OB_pd,real_traffic_TRAF_pd,expected_traff_OPT_List_SP_pd,
            expected_traff_OPT_List_OB_pd,expected_traff_OPT_List_TRAF_pd,service_SP_pd,service_OB_pd,service_TRAF_pd,total_service_SP_pd,total_service_OB_pd,
            total_service_TRAF_pd,gain_list_SP,gain_SP,gain_list_OB,gain_OB,traffic_list_SP,traffic_SP,traffic_list_OB,traffic_OB,Eurotraf_list_OB,Eurotraf_OB,Eurotraf_list_SP,Eurotraf_SP,
            Euro_list_SP,Euro_SP,Euro_list_OB,Euro_OB,Euro_list_TRAF,Euro_TRAF,beta_SP_pd,beta_OB_pd,beta_TRAF_pd,Net_Ben_OPT_List_SP ,Net_Ben_OPT_List_OB,Net_Ben_OPT_List_TRAF,NET_GAIN_list_SP
            ,NET_GAIN_SP,NET_GAIN_list_OB,NET_GAIN_OB,allocated_Traffic_list_SP,allocated_Traffic_SP,allocated_Traffic_list_OB,allocated_Traffic_OB,SLA_Cost_SP,SLA_Cost_OB,SLA_Cost_TRAF,Copex_Cost_SP,
            Copex_Cost_OB,Copex_Cost_TRAF,SLA_Cost_Metric_list_SP,SLA_Cost_Metric_SP,SLA_Cost_Metric_list_OB,SLA_Cost_Metric_OB,Copex_cost_Metric_list_SP,Copex_cost_Metric_SP
            ,Copex_cost_Metric_list_OB,Copex_cost_Metric_OB)
            return all_results

       

    def load_dataset_3D_variables(Fpath_Dataset,Fpath_variables,Tslots,SLA,output,n_slices=20,Ctot_ratio=1,Tdec_OB=30,q=0.5):
        from Global_Functions import Functions as F
        
        
        SP=[]
        OB=[]
        TRAF=[]
        SP_names=[]
        OB_names=[]
        TRAF_names=[]
        
         
        typee=['*SP_120min.npy','*OB_30min.npy','*traff.npy'] 
        Alphas=['alpha*SP*.npy','alpha*OB*.npy','alpha*traff*.npy']
        Xs=['X*SP*.npy','X*OB*.npy','X*traff*.npy']
        
    
        F.load_data(Fpath_Dataset,typee[0],output_list=SP,services_name_list=SP_names,printt=False)
        F.load_data(Fpath_Dataset,typee[1],output_list=OB,services_name_list=OB_names,printt=False)
        F.load_data(Fpath_Dataset,typee[2],output_list=TRAF,services_name_list=TRAF_names,printt=False)
        
        TRAF=Functions.list_to_df(TRAF)
        OB=Functions.list_to_df(OB)
        SP=Functions.list_to_df(SP)
        TRAF=TRAF.replace(np.nan,0).T
        OB=OB.replace(np.nan,0).T
        SP=SP.replace(np.nan,0).T
        TRAF=list(TRAF.values)
        OB=list(OB.values)
        SP=list(SP.values)
           
        ## Overbooking
        alpha_OB=[]
        X_OB=[]
        F.load_variables(Fpath_variables,Alphas[1],output_list=alpha_OB,printt=False)
        F.load_variables(Fpath_variables,Xs[1],output_list=X_OB,printt=False)
    
  
         ###### CHANGE DECISION TIMES #######
     
        Decision_time_OB=Tdec_OB

        try:
            X_OB_flat=F.flatten_slices(F.spread_data(X_OB,Decision_time_OB))
            alpha_OB_flat=F.flatten_slices(F.spread_data(alpha_OB,Decision_time_OB))
        except:
            X_OB_flat=X_OB
            alpha_OB_flat=F.flatten_slices(F.spread_data(alpha_OB,Decision_time_OB))
            

        Tslots=Tslots
        SP_pd=pd.DataFrame(SP).T[:Tslots]
        OB_pd=pd.DataFrame(OB).T[:Tslots]
        TRAF_pd=pd.DataFrame(TRAF).T[:Tslots]


        alpha_OB_flat_pd=pd.DataFrame(alpha_OB_flat).T[:Tslots]
        X_OB_flat_pd=pd.DataFrame(X_OB_flat).T[:Tslots]

        alpha_TRAF_flat_pd=pd.DataFrame(1,index=range(Tslots),columns=range(len(TRAF_pd.columns)))
        

        #### For beta, we change the alpha (1st parameter) and the set to evaluate (SP,OB,TRAF) in the 3rd parameter
    
        beta_OB_pd=F.beta_selection(alpha_OB_flat_pd,TRAF_pd,OB_pd,SP_pd)
        beta_TRAF_pd=F.beta_selection(alpha_TRAF_flat_pd,TRAF_pd,TRAF_pd,SP_pd)

        value_OPT_List_OB=SP_pd*X_OB_flat_pd*SLA(beta_OB_pd)
        value_OPT_List_TRAF=SP_pd**SLA(beta_TRAF_pd)

     
        Net_Ben_OPT_List_OB=X_OB_flat_pd*(SP_pd*SLA(beta_OB_pd)-F.plot_cost(q,alpha_OB_flat_pd,OB_pd))
        Net_Ben_OPT_List_TRAF=(SP_pd*SLA(beta_TRAF_pd)-F.plot_cost(q,alpha_TRAF_flat_pd,TRAF_pd))
    
    
        weight_OPT_list_OB=OB_pd*X_OB_flat_pd*alpha_OB_flat_pd
        weight_OPT_list_TRAF=TRAF_pd
        
        real_traffic_OPT_List_OB=TRAF_pd*X_OB_flat_pd
        real_traffic_OPT_List_TRAF=TRAF_pd

        expected_traffic_OPT_List_OB=OB_pd*X_OB_flat_pd
        expected_traffic_OPT_List_TRAF=TRAF_pd

      
        value_OB_pd=F.convert0_nan(value_OPT_List_OB)
        value_TRAF_pd=F.convert0_nan(value_OPT_List_TRAF)
        
        weight_OB_pd=F.convert0_nan(weight_OPT_list_OB)
        weight_TRAF_pd=F.convert0_nan(weight_OPT_list_TRAF)
       
        real_traffic_OB_pd=F.convert0_nan(real_traffic_OPT_List_OB)
        real_traffic_TRAF_pd=F.convert0_nan(real_traffic_OPT_List_TRAF)
    
        Net_Ben_OPT_List_OB=F.convert0_nan(Net_Ben_OPT_List_OB)
        Net_Ben_OPT_List_TRAF=F.convert0_nan(Net_Ben_OPT_List_TRAF)      
       
        
       
        if output== 'All_results':
            
            TRAF_pd=pd.DataFrame(TRAF[:n_slices]).T
           
            
            traffic_list_OB, traffic_OB=Metrics.rho_gain(real_traffic_OB_pd,real_traffic_TRAF_pd)
            Euro_list_OB, Euro_OB=Metrics.rho_gain(Net_Ben_OPT_List_OB,weight_OB_pd)
            Euro_list_TRAF, Euro_TRAF=Metrics.rho_gain(Net_Ben_OPT_List_TRAF,weight_TRAF_pd)
           
            Eurotraf_list_OB, Eurotraf_OB=Metrics.rho_gain_TRAF(Euro_list_OB,Euro_list_TRAF)
            
            
           
            NET_GAIN_list_OB,NET_GAIN_OB=Metrics.rho_gain(Net_Ben_OPT_List_OB,Net_Ben_OPT_List_TRAF)
           

            all_results=get_3Dresults(SP,OB,TRAF,traffic_list_OB,traffic_OB,Eurotraf_list_OB,Eurotraf_OB,Euro_list_OB,
            Euro_OB,Euro_list_TRAF,Euro_TRAF,Net_Ben_OPT_List_OB,Net_Ben_OPT_List_TRAF,NET_GAIN_list_OB,NET_GAIN_OB)
            return all_results
        else:
            print('Error, check output parameter, it must be "All_results"')






global colors_json


with open('./Color_code_synthetic.json','r') as f:
           colors_json=json.loads(f.read())




class plt_fcn:
    def __init__(self,colors_json):
        self.colors=colors_json
        

    #### Matplotlib Style
    plt.style.use('../Matplotlib/style.mplstyle')
   
    
    ### Global functions
    def select_num_slices(df, num_slices):
        return df[list(range(num_slices))]
    
    
    
    def plot_total_acc_slices(ax,idx,df,Slice_Set=''):
        ax[idx].plot(df['total_slices'],label=Slice_Set)
   

    def single_plot_slices(axs,idx,df,slice_names, Slice_Set='',linestyle='None',marker='.'):      
        slicelists=list(df.columns)
        for i in reversed(range(len(slicelists))):
            axs[idx].plot(df[slicelists[:i+1]].sum(axis=1,min_count=1),label=slice_names[i]+''+Slice_Set,color=colors_json[slice_names[i]],linestyle=linestyle,marker=marker)
                   
    def raw_single_plot_slices(ax,df,slice_names,Slice_Set="",linestyle='None',marker='.'):
        
        slicelists=list(df.columns)
        for i in reversed(range(len(slicelists))):
            ax.plot(df[slicelists[:i+1]].sum(axis=1,min_count=1),label=slice_names[i]+''+Slice_Set,color=colors_json[slice_names[i]],linestyle=linestyle,marker=marker)
        

    def accepted_overall_slices(ax,idx,df,Slice_Set='',linestyle='None',marker='.'):
        slicelists=list(df.columns)
        ax[idx].plot(df[slicelists].sum(axis=1,min_count=1),label='Accepted '+ Slice_Set,linestyle=linestyle,marker=marker)
    

    def plot_acc_slices(ax,idx,df,slice_names,Slice_Set='',linestyle='None',marker='.'):
        slicelists=list(df.columns)
        for i in reversed(range(len(slicelists))):
            ax[idx].plot(df[slicelists[i]],label=slice_names[i]+''+Slice_Set,color=colors_json[slice_names[i]],linestyle=linestyle,marker=marker)
        ax[idx].set_yticks(slicelists)
        ax[idx].set_yticklabels(slice_names[:len(slicelists)])

    def plot_acc_slices_color(ax,idx,df,slice_names,Slice_Set='',linestyle='None',marker='.',color='blue'):
        slicelists=list(df.columns)
        df_nans=pd.DataFrame()
        for i in range(len(df.columns)):
            df_nans[i]= df[i].replace(np.nan,30,inplace=False)
            df_nans[i]= df_nans[i].replace(i,np.nan,inplace=False)
            df_nans[i]= df_nans[i].replace(30,i,inplace=False)
            ax[idx].plot(df_nans[i],label=Slice_Set,color=color,linestyle=linestyle,marker=marker)
        ax[idx].set_yticks(slicelists)
        ax[idx].set_yticklabels(slice_names[:len(slicelists)])

    def plot_acc_slices_new_color(ax,idx,df,slice_names,Tslots,Slice_Set='',linestyle='None',marker='.',color='orange'):
        slicelists=list(df.columns)
        for i in reversed(range(len(slicelists))):
            for j in range(Tslots):
                if np.isnan(df[slicelists[i]][j]):
                    ax[idx].plot(j,df[slicelists[i]][0],label=Slice_Set,color=color,linestyle=linestyle,marker=marker)
        ax[idx].set_yticks(slicelists)
        ax[idx].set_yticklabels(slice_names[:len(slicelists)])

    
    def plot_shadow(x,gain_list,label=None,color=None,linestyle='-',marker=None,daylight=False,nightlight=False,plot_Quartile=False,plot_mean=False,ax=False,idx=False):
        li=[]
        gain_list_day=[]
        gain_list_night=[]
        for i in range(len(gain_list)):
            arr=gain_list[i][0]
            li.append(arr)

        li_df=pd.DataFrame(li)
        means=np.array(li_df.T.mean())
        upperbound=li_df.T.where(li_df.T>=means)
        lowerbound=li_df.T.where(li_df.T<means)
        upper_std=upperbound.std()
        lower_std=lowerbound.std()
        
        stds=np.array(li_df.T.std())
        quarter=np.array(li_df.T.describe().loc['25%'])
        half=np.array(li_df.T.describe().loc['50%'])
        Threequarter=np.array(li_df.T.describe().loc['75%'])
        if daylight:
                for j in range(round(len(li_df.T)/1440)):
                        gain_list_day.append(pd.DataFrame(li_df.T.loc[1440*j+480:1440*j+1440]))
                alldays=pd.concat(gain_list_day)
                means=np.array(alldays.mean())
                stds=np.array(alldays.std())
                upperbound=li_df.T.where(li_df.T>=means)
                lowerbound=li_df.T.where(li_df.T<means)
                upper_std=np.array(upperbound.std())
                lower_std=np.array(lowerbound.std())
                quarter=np.array(alldays.describe().loc['25%'])
                half=np.array(alldays.describe().loc['50%'])
                Threequarter=np.array(alldays.describe().loc['75%'])
        if nightlight:
                for j in range(round(len(li_df.T)/1440)):
                        gain_list_night.append(pd.DataFrame(li_df.T.loc[1440*j:1440*j+480]))
                alldays=pd.concat(gain_list_night)
                means=np.array(alldays.mean())
                stds=np.array(alldays.std())
                upperbound=li_df.T.where(li_df.T>=means)
                lowerbound=li_df.T.where(li_df.T<means)
                upper_std=np.array(upperbound.std())
                lower_std=np.array(lowerbound.std())
                quarter=np.array(alldays.describe().loc['25%'])
                half=np.array(alldays.describe().loc['50%'])
                Threequarter=np.array(alldays.describe().loc['75%'])
        if plot_Quartile:
                plt.plot(x, half,  label=label + ' median',color=color,marker=marker,linestyle=linestyle)
                plt.fill_between(x, quarter, Threequarter, color=color, alpha=0.2,interpolate=True)
                if plot_mean:
                        plt.plot(x, means,  label=label+' mean',color=color,marker=marker,linestyle='dashed')
        elif not ax:
                plt.plot(x, means,  label=label,color=color,marker=marker,linestyle=linestyle)
                plt.fill_between(x, means- lower_std, means + upper_std, color=color, alpha=0.2)
        else:
            ax[idx].plot(x, means,  label=label,color=color,marker=marker,linestyle=linestyle)
            ax[idx].fill_between(x, means- lower_std, means + upper_std, color=color, alpha=0.2)

    def weighted_mean(df,maxs):
        
        df_means=pd.DataFrame()
        clu_maxs5=maxs.loc[maxs['cluster_list'].str.contains('5_clusters')]
        clu_maxs10=maxs.loc[maxs['cluster_list'].str.contains('10_clusters')]
        clu_maxs20=maxs.loc[maxs['cluster_list'].str.contains('20_clusters')]
        clu_maxs50=maxs.loc[maxs['cluster_list'].str.contains('50_clusters')]
        clu_maxs100=maxs.loc[maxs['cluster_list'].str.contains('100_clusters')]
        for i in range(len(df.columns)):
        #for i in range(3):
                if 'cluster_5' in df.columns[i]:
                        if 'cluster_50' in df.columns[i]:
                                df_means[df.columns[i]]=df[df.columns[i]]
                                df_means[df.columns[i]+'_mean']=(df[df.columns[i]]*clu_maxs50.max_cum_traffic.reset_index(drop=True)).sum()/clu_maxs50.max_cum_traffic.reset_index(drop=True).sum()
                                df_means[df.columns[i]+'_std']=df[df.columns[i]]-df_means[df.columns[i]+'_mean']
                        else:
                                df_means[df.columns[i]]=df[df.columns[i]]
                                df_means[df.columns[i]+'_mean']=(df[df.columns[i]]*clu_maxs5.max_cum_traffic.reset_index(drop=True)).sum()/clu_maxs5.max_cum_traffic.reset_index(drop=True).sum()
                                df_means[df.columns[i]+'_std']=df[df.columns[i]]-df_means[df.columns[i]+'_mean']
                elif 'cluster_10' in df.columns[i]:
                        if 'cluster_100' in df.columns[i]:
                                df_means[df.columns[i]]=df[df.columns[i]]
                                df_means[df.columns[i]+'_mean']=(df[df.columns[i]]*clu_maxs100.max_cum_traffic.reset_index(drop=True)).sum()/clu_maxs100.max_cum_traffic.reset_index(drop=True).sum()
                                df_means[df.columns[i]+'_std']=df[df.columns[i]]-df_means[df.columns[i]+'_mean']    
                        else:
                                df_means[df.columns[i]]=df[df.columns[i]]
                                df_means[df.columns[i]+'_mean']=(df[df.columns[i]]*clu_maxs10.max_cum_traffic.reset_index(drop=True)).sum()/clu_maxs10.max_cum_traffic.reset_index(drop=True).sum()
                                df_means[df.columns[i]+'_std']=df[df.columns[i]]-df_means[df.columns[i]+'_mean']
                elif 'cluster_20' in df.columns[i]:                        
                        df_means[df.columns[i]]=df[df.columns[i]]
                        df_means[df.columns[i]+'_mean']=(df[df.columns[i]]*clu_maxs20.max_cum_traffic.reset_index(drop=True)).sum()/clu_maxs20.max_cum_traffic.reset_index(drop=True).sum()
                        df_means[df.columns[i]+'_std']=df[df.columns[i]]-df_means[df.columns[i]+'_mean']
    
        return df_means
      
    
    
   
    
    def differentiaiton_plot(df1,df2,Slice_Set='',ax=None,idx=None,ptt=None,linestyle=None,marker='.'):
        slicelists=list(df1.columns)
        if idx:
            ax[idx].plot(df1[slicelists].sum(axis=1,min_count=1)-df2[slicelists].sum(axis=1,min_count=1),label='Difference '+ Slice_Set,linestyle=linestyle,marker=marker)
        if ptt:
            ptt.plot(df1[slicelists].sum(axis=1,min_count=1)-df2[slicelists].sum(axis=1,min_count=1),label='Difference '+ Slice_Set,linestyle=linestyle,marker=marker)
    
        else:
            plt.plot(df1[slicelists].sum(axis=1,min_count=1)-df2[slicelists].sum(axis=1,min_count=1),label='Difference '+ Slice_Set,linestyle=linestyle,marker=marker)
    

    ## Metrics
    
    def plot_metrics(rho_OB,rho_SP=None,subplots=False,label_OB='',label_SP='',title='',legend=False,grid=False,marker=None):
        if subplots== False:
            plt.figure(figsize=(13,10))
            plt.plot(rho_OB,label=label_OB,marker=marker)
            if rho_SP:
                plt.plot(rho_SP,label=label_SP,marker=marker)
            plt.xticks(np.arange(0,len(rho_OB),1),np.arange(1,len(rho_OB)+1,1))
            plt.yticks(np.arange(0,1.1,0.1))
            plt.title(title)
            plt.xlabel('Slices')
            plt.ylabel(r'$\rho$')
            plt.ylim(0.2,1.005)
           
                
            if legend== True:
                plt.legend()
            if grid == True:
                plt.grid()

class Metrics:

    
    def rho_gain(value_known_df,value_Optimal_df):
        #### Current Gain Metric: ratio of average of the optimal value to the known value
        
            rho_Tserie=value_known_df.sum(axis=1,min_count=1)/value_Optimal_df.sum(axis=1,min_count=1)
            mean_known=(1/len(value_known_df))*value_known_df.sum(axis=1,min_count=1).sum()
            mean_Optimal=(1/len(value_Optimal_df))*value_Optimal_df.sum(axis=1,min_count=1).sum()
            Ratio_known_Optimal=mean_known/mean_Optimal
    
            return rho_Tserie,Ratio_known_Optimal
    
    def rho_gain_TRAF(Eurotraf_known,Eurotraf_optimal):
        eurotaf=Eurotraf_known/Eurotraf_optimal
        eurotraf_mean=Eurotraf_known.mean()/Eurotraf_optimal.mean()
        return eurotaf, eurotraf_mean
#
    def rho_traffic(traffic_known_df,traffic_Optimal_df):
        rho_list=[]
        for i in range(len(traffic_known_df)):
            if traffic_known_df.loc[i].sum() == 0 or traffic_Optimal_df.loc[i].sum() == 0:
                rho_list.append(np.nan)
                #continue
            rho_list.append(traffic_known_df.loc[i].sum()/traffic_Optimal_df.loc[i].sum())

        return np.array(rho_list), np.nanmean(rho_list)

    def rho_traffic_mins(traffic_known_df,weight_known_df,traffic_Optimal_df,weight_Optimal_df):
        rho_list=[]
        for i in range(len(traffic_known_df)):
            if traffic_known_df.loc[i].sum() == 0 or traffic_Optimal_df.loc[i].sum() == 0:
                continue
            rho_list.append(min(traffic_known_df.loc[i].sum(),weight_known_df.loc[i].sum())/min(traffic_Optimal_df.loc[i].sum(),weight_Optimal_df.loc[i].sum()))

        return np.array(rho_list), np.nanmean(rho_list)
class Plotter:

    ## Metrics

    def plot_Shadower(listt,gain_OB_list,xlabel,gain_SP_list=None,ylabel=r'$\rho$',OB_label='Overbooking vs Optimal',SP_label='Service Provider vs Optimal',color='blue',title='',grid=False,Legend=True,marker='o',daylight=False,nightlight=False,plot_Quartile=False):
        
        plt.figure(figsize=(13,10))
        plt_fcn.plot_shadow(listt,gain_OB_list,label=OB_label+' overall',color=color[0],marker=marker,plot_Quartile=plot_Quartile)
        if daylight:
            plt_fcn.plot_shadow(listt,gain_OB_list,label=OB_label+' daylight',color=color[1],marker=marker,daylight=daylight,plot_Quartile=plot_Quartile)
        if nightlight:
            plt_fcn.plot_shadow(listt,gain_OB_list,label=OB_label+ ' nightlight',color=color[2],marker=marker,nightlight=nightlight,plot_Quartile=plot_Quartile)
        if gain_SP_list:
            plt_fcn.plot_shadow(listt,gain_SP_list,label=SP_label+' overall',color=color[3],marker=marker,plot_Quartile=plot_Quartile)
            if daylight:
                plt_fcn.plot_shadow(listt,gain_SP_list,label=SP_label+' daylight',color=color[4],marker=marker,daylight=daylight,plot_Quartile=plot_Quartile)
            if nightlight:
                plt_fcn.plot_shadow(listt,gain_SP_list,label=SP_label+' nightlight',color=color[5],marker=marker,nightlight=nightlight,plot_Quartile=plot_Quartile)
        plt.xticks(listt)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if grid:
            plt.grid(axis='y')
        if Legend:
            plt.legend()


    def plot_Ctot_ratio(Ctot,gain_SP_mean_list,gain_OB_mean_list,SP_label,OB_label,title,grid=None,Legend=None,marker=None):
        plt.figure(figsize=(13,10))
        plt.plot(Ctot,gain_SP_mean_list,label=SP_label,marker=marker)
        plt.plot(Ctot,gain_OB_mean_list,label=OB_label,marker=marker)
        #plt.xticks(np.arange(0,len(rho_OB),1),np.arange(1,len(rho_OB)+1,1))
        #plt.yticks(np.arange(0,1.1,0.1))
        plt.title(title)
        plt.xlabel('Num. Slices')
        plt.ylabel(r'$\rho$')
        plt.ylim(min(gain_SP_mean_list)-0.05,1.005)
        if grid:
            plt.grid()
        if Legend:
            plt.legend()

    def Raw_plotter(ax,Tslots,SP_norm_pd,OB_norm_pd,TRAF_norm_pd,Ctots,Slice_names):
        
        plt_fcn.raw_single_plot_slices(ax,SP_norm_pd[:Tslots],slice_names=Slice_names,Slice_Set=' Service Provider',linestyle='solid',marker='None')
        plt_fcn.raw_single_plot_slices(ax,OB_norm_pd[:Tslots],slice_names=Slice_names,Slice_Set=' Overbooking',linestyle='dotted',marker='None')
        plt_fcn.raw_single_plot_slices(ax,TRAF_norm_pd[:Tslots],slice_names=Slice_names,linestyle='solid',marker='None')
        
        plt.plot(np.full(Tslots,Ctots),linestyle='dashdot', linewidth=8,color='purple')
        for i in range(round((Tslots+120)/120)):
            plt.axvline(i*120,linestyle='dotted',linewidth=0.3,color='grey')

    def Raw_plotter_single(ax,Tslots,TRAF_norm_pd,Ctots,Slice_names):
        
        plt_fcn.raw_single_plot_slices(ax,TRAF_norm_pd[:Tslots],slice_names=Slice_names,linestyle='solid',marker='None')
        
        plt.plot(np.full(Tslots,Ctots),linestyle='dashdot', linewidth=8,color='purple')
        for i in range(round((Tslots+120)/120)):
            plt.axvline(i*120,linestyle='dotted',linewidth=0.3,color='grey')
      
        
        
    def Value_plotter(ax,value_SP_pd,value_OB_pd,value_TRAF_pd,Slice_names,Legend=None):


        ax[0].set_title('Overall Comparison')
        plt_fcn.accepted_overall_slices(ax,0,value_SP_pd,Slice_Set='SP')
        plt_fcn.accepted_overall_slices(ax,0,value_OB_pd,Slice_Set='OB')
        plt_fcn.accepted_overall_slices(ax,0,value_TRAF_pd,Slice_Set='Optimal')
        

        ##### Service Provider 

        ax[1].set_title('Service Provider')
        plt_fcn.single_plot_slices(ax,1,value_SP_pd,Slice_names,Slice_Set='SP')

        ##### Overbooking

        ax[2].set_title('Overbooking')

        plt_fcn.single_plot_slices(ax,2,value_OB_pd,Slice_names,Slice_Set='OB')

        ##### Optimal

        ax[3].set_title('Optimal')

        plt_fcn.single_plot_slices(ax,3,value_TRAF_pd,Slice_names)

        if Legend:
            ax[0].legend()
            ax[3].legend(loc='lower center', bbox_to_anchor=(0.5, -0.4),
                      ncol=6, fancybox=True, shadow=True)


                      
    def Slices_plotter(ax,total_service_SP_pd,total_service_OB_pd,total_service_TRAF_pd,
    service_SP_pd,service_OB_pd,service_TRAF_pd,Slice_names):


        ax[0].set_title('Overall Comparison')

        plt_fcn.plot_total_acc_slices(ax,0,total_service_SP_pd,Slice_Set='Service Provider')
        plt_fcn.plot_total_acc_slices(ax,0,total_service_OB_pd,Slice_Set='Overbooking')
        plt_fcn.plot_total_acc_slices(ax,0,total_service_TRAF_pd,'Optimal')

        ax[0].legend(fancybox=True,shadow=True)

        ##### Service Provider 

        ax[1].set_title('Service Provider')
        plt_fcn.plot_acc_slices(ax,1,service_SP_pd,Slice_names)

        ##### Overbooking

        ax[2].set_title('Overbooking')
        plt_fcn.plot_acc_slices(ax,2,service_OB_pd,Slice_names)
 

        ##### Optimal

        ax[3].set_title('Optimal')
        plt_fcn.plot_acc_slices(ax,3,service_TRAF_pd,Slice_names)

    def Slices_plotter_by_category(ax,total_service_SP_pd,total_service_OB_pd,total_service_TRAF_pd,
    service_SP_pd,service_OB_pd,Slice_names,Legend=None):

        ax[0].set_title('Overall Comparison')

        plt_fcn.plot_total_acc_slices(ax,0,total_service_SP_pd,Slice_Set='Service Provider')
        plt_fcn.plot_total_acc_slices(ax,0,total_service_OB_pd,Slice_Set='Infrastructure Operator')
        plt_fcn.plot_total_acc_slices(ax,0,total_service_TRAF_pd,'Oracle')

        ##### Service Provider 
        ax[1].set_title('Slices not accepted')
        plt_fcn.plot_acc_slices_color(ax,1,service_SP_pd,Slice_names,color='blue',Slice_Set='SP')
        plt_fcn.plot_acc_slices_color(ax,1,service_OB_pd,Slice_names,color='orange',Slice_Set='IO')
        ax[0].set_xlim(-100,10180)
        ax[1].set_xlim(-100,10180)
        

        if Legend:

            ax[0].legend(fancybox=True,shadow=True)
            ax[1].legend(fancybox=True,shadow=True)


    def Weight_plotter(ax,weight_SP_pd,weight_OB_pd,weight_TRAF_pd,Slice_names):
        ax[0].set_title('Overall Comparison')
        plt_fcn.accepted_overall_slices(ax,0,weight_SP_pd,Slice_Set='SP')
        plt_fcn.accepted_overall_slices(ax,0,weight_OB_pd,Slice_Set='OB')
        plt_fcn.accepted_overall_slices(ax,0,weight_TRAF_pd,Slice_Set='Optimal')
        ax[0].legend()

        ##### Service Provider 

        ax[1].set_title('Service Provider')
        plt_fcn.single_plot_slices(ax,1,weight_SP_pd,Slice_names,Slice_Set='SP')

        ##### Overbooking

        ax[2].set_title('Overbooking')

        plt_fcn.single_plot_slices(ax,2,weight_OB_pd,Slice_names,Slice_Set='OB')

        ##### Optimal

        ax[3].set_title('Optimal')

        plt_fcn.single_plot_slices(ax,3,weight_TRAF_pd,Slice_names)


        ax[3].legend(loc='lower center', bbox_to_anchor=(0.5, -0.4),
                  ncol=6, fancybox=True, shadow=True)

    def Real_traffic_plotter(fig,ax,real_traffic_SP_pd,weight_SP_pd,real_traffic_OB_pd,weight_OB_pd,real_traffic_TRAF_pd,
    weight_TRAF_pd):
        ax[0].set_title('Service Provider')
        plt_fcn.accepted_overall_slices(ax,0,real_traffic_SP_pd)
        plt_fcn.accepted_overall_slices(ax,0,weight_SP_pd)
        #ax[0].set_ylim([0,1*10**9])

        #### Overbooking
        ax[1].set_title('Overbooking')
        plt_fcn.accepted_overall_slices(ax,1,real_traffic_OB_pd)
        plt_fcn.accepted_overall_slices(ax,1,weight_OB_pd)

        #ax2=fig.add_axes(ax[1])
        #differentiaiton_plot(real_traffic_OB_pd,weight_OB_pd,ptt=ax2)

        #ax[1].set_ylim([0,1*10**9])

        #### Optimal
        ax[2].set_title('Optimal')
        plt_fcn.accepted_overall_slices(ax,2,real_traffic_TRAF_pd)
        plt_fcn.accepted_overall_slices(ax,2,weight_TRAF_pd)

        ax[2].legend(['Real Traffic','Weight'],loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=6, fancybox=True, shadow=True)

        #ax3=fig.add_axes(ax[2])
        #plt_fcn.differentiaiton_plot(real_traffic_TRAF_pd,weight_TRAF_pd,ptt=ax3)
    
    def Metrics_plotter(ax,gain_list_SP,gain_list_OB,Tslots,Legend=None,daymean=None,allmean=None,typess='p'):
        
        OB_label=r'$\bar{D}_{p}$'.replace('p',typess)
        SP_label=r'$\bar{D}_{p}^{(P)}$'.replace('p',typess)
        labels=r'$\bar{G}_{p}$'.replace('p',typess)
        ax[0].plot(gain_list_SP,label=SP_label)
        ax[0].plot(gain_list_OB,label=OB_label)
        
        
        ratioOB_SP=(gain_list_OB[:Tslots]-gain_list_SP[:Tslots])/gain_list_SP[:Tslots]
        #print(ratioOB_SP)
        mean_ratio_list=[]
        for i in range(round(len(ratioOB_SP)/1440)):
            mean_ratio_list.append(ratioOB_SP[1440*i+480:1440*i+1440])
        mean_daylight=pd.DataFrame(mean_ratio_list).T.mean().mean()
        mean_daylight_arr=np.full(Tslots,mean_daylight)

        ratioOB_SP_allday_mean=pd.DataFrame(ratioOB_SP).mean().mean()
        ratioOB_SP_allday_mean_arr=np.full(Tslots,ratioOB_SP_allday_mean)

        ax[1].plot(ratioOB_SP,label=labels)

        if daymean:
            ax[1].plot(mean_daylight_arr,label='Daylight Mean',linestyle='dashed')
        if allmean:
            ax[1].plot(ratioOB_SP_allday_mean_arr,label='Mean',linestyle='dashed')
        if Legend:
            ax[0].legend()
            ax[1].legend()
    def create_df_plotly(metric_list,Q_list,daylight=False):
        df=pd.DataFrame(metric_list).T
        df.columns=Q_list
        if daylight:
            df_daylight=pd.DataFrame()
            df_daylight=df_daylight.append(df.loc[:664])
            df_intermediate=df.iloc[664:,:]
            df_intermediate=df_intermediate.reset_index(drop=True)
            for j in range(round(len(df)/1440)):
                df_daylight=df_daylight.append(df_intermediate.loc[j*1440+480:(j+1)*1440])
            df=df_daylight.reset_index(drop=True)
        return df
    def create_plots_metrics(metric_list,Q,ylabel='',xlabel='',xtitle='',title='',Daylight=False,Distance_Optimal=False,Imrovenent_over_std_app=False,metric2nd_list=None,legend=True,ax=None,idx=None):
        df=Plotter.create_df_plotly(metric_list,Q)
        if Daylight:
            df_day=Plotter.create_df_plotly(metric_list,Q,daylight=Daylight)
        plt.title(title)
        if Distance_Optimal:
            plt.title(f'{title} - Distance Optimal')
            if metric2nd_list:
                df2nd=Plotter.create_df_plotly(metric2nd_list,Q)
                if Daylight:
                    df2nd_day=Plotter.create_df_plotly(metric2nd_list,Q,daylight=Daylight)
                    plt.plot((1-df_day.mean()-(1-df2nd_day.mean()))/(1-df_day.mean()),label='Reduction of distance to optimal Daylight')
    
                if ax:
                    ax[idx].plot((1-df.mean()-(1-df2nd.mean()))/(1-df.mean()),label='Reduction of distance to optimal')
                else:
                    plt.plot((1-df.mean()-(1-df2nd.mean()))/(1-df.mean()),label='Reduction of distance to optimal Ratio')
               
            else:
                plt.plot(1-df.mean(),label=f'{xlabel}')
                if Daylight:
                    plt.plot(1-df_day.mean(),label=f'Daylight {xlabel}')
        elif Imrovenent_over_std_app:
            if metric2nd_list:
                df2nd=Plotter.create_df_plotly(metric2nd_list,Q)
                if Daylight:
                    df2nd_day=Plotter.create_df_plotly(metric2nd_list,Q,daylight=Daylight)
                    plt.plot((df2nd_day.mean()-df_day.mean())/df_day.mean(),label='Improvement over standard approach Daylight')
                
                else:
                    if ax:
                        ax[idx].plot((df2nd.mean()-df.mean())/df.mean(),label='Improvement over standard approach')
                    else:

                        plt.plot((df2nd.mean()-df.mean())/df.mean(),label='Improvement over standard approach')
    
            else:
                plt.plot(1-df.mean(),label=f' {xlabel}')
                if Daylight:
                    plt.plot(1-df_day.mean(),label=f'Daylight {xlabel}')
        else:
            if ax:
                ax[idx].plot(df.mean(),label=f'{xlabel}')
            else:
                plt.plot(df.mean(),label=f' {xlabel}',marker='o')
            if Daylight:
                plt.plot(df_day.mean(),label=f'Daylight {xlabel}',marker='o')
        
        plt.ylabel(ylabel)
        plt.xlabel(xtitle)
        if legend:
            plt.legend()

    

class Optimizer:

    def SCIP_Optimizer_AC_RA(Tslots,traf_known,traffic_agreement,listname,SLA=Functions.gain_lin_approx_5,revenue=Functions.revenue,
    cost=Functions.cOpex,Ctot=1*10**10,save=False,Filepath='',Reserved_Time=120,Decision_Time=30,revenue_min_bps=1,cost_min_bps=0.5):
        
        I=range(len(traf_known))
        alphas_per_slice=round(Reserved_Time/Decision_Time)
        num_alphas=range(len(traf_known)*alphas_per_slice)
        print(f'TR:  {num_alphas}')

        X_lists=[[] for i in I] #Slice accepted /rejected
        alpha_lists=[[] for i in I] ## Fraction of slice accepted

        
        for block in tqdm(range(round(Tslots/Reserved_Time))):

            model=Model('Optimization')
            objvar=model.addVar(vtype='C',name='objvar')

            ## Variables
            
            X=[model.addVar(lb=0,ub=1,vtype='I',name=f'X{i}') for i in I]
            alpha=[model.addVar(lb=0.8,ub=1,vtype='C',name=f'alpha{i}') for i in num_alphas]
            model.setObjective(objvar,sense='maximize')
            
            model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
            model.hideOutput(True)


            ## Constraints, the sum of the traffic for all slices must be lower than the total capacity of the network
            for b in range(alphas_per_slice):
                constraint_eq=[]
                for s in I:
                    constraint_eq.append(X[s]*alpha[s*alphas_per_slice+b]*traf_known[s][block*Reserved_Time+b*Decision_Time])
                  
                model.addCons(np.sum(constraint_eq)<=Ctot)
            
            #Maximizer, we want to maximize the profit of the network provider
            Maxim_eq=[]
            for ma in I:
                
                Monetary_gain_per_block = revenue_min_bps * Decision_Time ## Revenue per bps for block of operator (duration = decision time)
                Monetary_cost_per_block = cost_min_bps * Decision_Time   ## Cost per bps for block of operator (duration = decision time)
                
                Agreement_traffic_per_block = traffic_agreement[ma][block*Reserved_Time:block*Reserved_Time+Reserved_Time:Decision_Time]
                Real_Traffic_per_block = traf_known[ma][block*Reserved_Time:block*Reserved_Time+Reserved_Time:Decision_Time]
                
                Real_Traffic_per_block = np.minimum(Real_Traffic_per_block,Agreement_traffic_per_block)


                """With this function, we are Maximizing the net profit of the Network Provider"""
                Maxim_eq.append(X[ma]*(revenue(Monetary_gain_per_block,Agreement_traffic_per_block, alpha[ma*alphas_per_slice:ma*alphas_per_slice+alphas_per_slice],SLA) - 
                cost(Monetary_cost_per_block, Real_Traffic_per_block, alpha[ma*alphas_per_slice:ma*alphas_per_slice+alphas_per_slice])))


            model.addCons(objvar<=np.sum(Maxim_eq))
            model.optimize()
            ## format could be .lp or .cip
            #model.writeProblem('model_new.lp')
            #model.writeProblem('model_new.cip')
            sol=model.getBestSol()
            
            #print("x: {}".format(sol[x]))
            #print("y: {}".format(sol[y]))
            #print(sol)
            #print(f'sol={sol}')
            for j in I:
                alphas_list=[]
                for alphas in range(alphas_per_slice):
                    alphas_list.append(sol[alpha[j*alphas_per_slice+alphas]])
                alpha_lists[j].append(alphas_list)
                
                X_lists[j].append(list(np.full(alphas_per_slice,round(sol[X[j]],1))))

        #save
        if save == True:

            if not Filepath:
                print('-------------')
                print('Error:')
                print('Please provide Filepath to save')
                print('-------------')

            elif not os.path.exists(Filepath):
                os.makedirs(Filepath)

                for r in I:
                    np.save(Filepath+'X_'+str(r)+'_list_'+listname[r]+'_traf_known.npy',np.array(X_lists[r]))
                    np.save(Filepath+'alpha_'+str(r)+'_list_'+listname[r]+'_traf_known.npy',np.array(alpha_lists[r]))
            else:
                for r in I:
                    np.save(Filepath+'X_'+str(r)+'_list_'+listname[r]+'_traf_known.npy',np.array(X_lists[r]))
                    np.save(Filepath+'alpha_'+str(r)+'_list_'+listname[r]+'_traf_known.npy',np.array(alpha_lists[r]))


    def     OR_Optimizer_RRA(Tslots,X,traf_known,traffic_agreement,listname,SLA=Functions.gain_lin_approx_5,revenue=Functions.revenue,cost=Functions.cOpex,Ctot=1*10**10,save=False,Filepath=''
            ,Reserved_Time=120,Decision_Time=30,revenue_min_bps=1,cost_min_bps=0.5):
        """Second Block of optimization, we will use only the accepted slices to retrieve the dyanmic resource allocation"""
        I=range(len(traf_known))
        alphas_per_slice=1
        n_alloc=round(Reserved_Time/Decision_Time)
        num_alphas=range(len(traf_known)*alphas_per_slice)
        print(f'TR:  {num_alphas}')

        
        alpha_lists=[[] for i in I] ## Fraction of slice accepted
        
        #for block in [78]:
        for block in tqdm(range(round(Tslots/Decision_Time))):
            
            solver = pywraplp.Solver.CreateSolver('GLOP')
            alpha=[solver.NumVar(0,1,f'alpha{i}') for i in num_alphas]

            constraint_eq=[]
            for s in I:
                
                Decision=X[s][block*Decision_Time]
                
                constraint_eq.append(Decision*alpha[s]*traf_known[s][block*Decision_Time])
        
            #Constraints
            solver.Add(np.sum(constraint_eq)<=Ctot)

            #Maximizer

            Maxim_eq=[]
            for ma in I:
                
                """Net profit of the Network Provider"""
                Monetary_gain_per_block = revenue_min_bps * Decision_Time ## Revenue per bps for this block
                Monetary_cost_per_block = cost_min_bps * Decision_Time   ## Cost per bps for this block


                Agreement_traffic_per_block = traffic_agreement[ma][block*Decision_Time]
                Real_Traffic_per_block = traf_known[ma][block*Decision_Time]

                # We serve what we agree
              
                Real_Traffic_per_block=np.min([Real_Traffic_per_block,Agreement_traffic_per_block])

                Decision=X[ma][block*Decision_Time]
                
                #print(f'Decision {ma}: {Decision}')
                Slice_Revenue=revenue(Monetary_gain_per_block, [Agreement_traffic_per_block], alpha[ma*alphas_per_slice:(ma+1)*alphas_per_slice])
                Slice_Cost=cost(Monetary_cost_per_block,[Real_Traffic_per_block],alpha[ma*alphas_per_slice:(ma+1)*alphas_per_slice])

              
                Maxim_eq.append(Decision*(Slice_Revenue-Slice_Cost))

            solver.Maximize(np.sum(Maxim_eq))
            solver.Solve()

            #print("x: {}".format(sol[x]))
            #print("y: {}".format(sol[y]))
            

            for j in I:
                alphas_list=[]
                for alphas in range(alphas_per_slice):

                    alphas_list.append(alpha[j*alphas_per_slice+alphas].solution_value())
                alpha_lists[j].append(alphas_list)
               
                
        if save == True:

            if not Filepath:
                print('-------------')
                print('Error:')
                print('Please provide Filepath to save')
                print('-------------')

            elif not os.path.exists(Filepath):
                os.makedirs(Filepath)

                for r in I:
                    np.save(Filepath+'X_'+str(r)+'_list_'+listname[r]+'_traf_known.npy',np.array(X[r]))
                    np.save(Filepath+'RRA_alpha_'+str(r)+'_list_'+listname[r]+'_traf_known.npy',np.array(alpha_lists[r]))
            else:
                for r in I:
                    np.save(Filepath+'X_'+str(r)+'_list_'+listname[r]+'_traf_known.npy',np.array(X[r]))
                    np.save(Filepath+'RRA_alpha_'+str(r)+'_list_'+listname[r]+'_traf_known.npy',np.array(alpha_lists[r]))

                    
     # Function to load all the data for the optimization 
    def     SCIP_Optimization_AC_RA(Fpath_Dataset,Tslots,SLA,Fpath_Save,Ctot_type='Commune',Ctot_ratio=1,save=True,Tdec_SP=120,Tdec_OB=30,predicted_traffic=True
    ,revenue_min_bps=1,cost_min_bps=0.5,synthetic_traffic=False):



        SP=[]
        OB=[]
        TRAF=[]
        SP_names=[]
        OB_names=[]
        TRAF_names=[]
    
        
        typee=['*SP_120min.npy','*OB_30min.npy','*traff.npy']
        if predicted_traffic==True:
            typee=['*Alpha_1*.npy','*Alpha_0*.npy','*clean.npy']
        elif synthetic_traffic==True:
            typee=['*Alpha_1*120min*.npy','*Alpha_0*.npy','*agg_60_s.npy']

        
        Functions.load_data(Fpath_Dataset,typee[0],output_list=SP,services_name_list=SP_names,printt=False)
        Functions.load_data(Fpath_Dataset,typee[1],output_list=OB,services_name_list=OB_names,printt=False)
        Functions.load_data(Fpath_Dataset,typee[2],output_list=TRAF,services_name_list=TRAF_names,printt=False)
    

        TRAF_pd=pd.DataFrame(TRAF).T
        if Ctot_type=='Commune':
            Ctot=max(TRAF_pd[:Tslots].sum(axis=1))*Ctot_ratio
        elif Ctot_type=='Antenna':
            TRAF=Functions.list_to_df(TRAF)
            OB=Functions.list_to_df(OB)
            SP=Functions.list_to_df(SP)
            TRAF=TRAF.replace(np.nan,0).T
            OB=OB.replace(np.nan,0).T
            SP=SP.replace(np.nan,0).T
            TRAF=list(TRAF.values)
            OB=list(OB.values)
            SP=list(SP.values)
            #Ctot=np.percentile(np.array(TRAF_pd[:Tslots].sum(axis=1)),99.8)*Ctot_ratio
            try:
                Ctot=max(TRAF_pd[:Tslots].sum(axis=1))*Ctot_ratio
            except:
                return(f'Error in : {Fpath_Dataset}')
                
        else:
            return 'Error, Check Ctot_type'

        print('Ctot: {:.2E}'.format(Ctot))

        if Tdec_OB>Tdec_SP:
            # If the decision time of the OB is longer than the SP, we need to add some slots to the SP
            Tdec_SP=Tdec_OB
            

        # We call the SCIP_Optimizer_AC_RA function to optimize the network
        print(f'Decision time: {Tdec_OB}')
        print(f'Admission time: {Tdec_SP}')
        print('SP')
        Optimizer.SCIP_Optimizer_AC_RA(Tslots,SP,SP,listname=SP_names,SLA=SLA,Ctot=Ctot,save=save,Filepath=Fpath_Save,Reserved_Time=Tdec_SP,Decision_Time=Tdec_SP,revenue_min_bps=revenue_min_bps,cost_min_bps=cost_min_bps)
        print('OB') 
        Optimizer.SCIP_Optimizer_AC_RA(Tslots,OB,SP,listname=OB_names,SLA=SLA,Ctot=Ctot,save=save,Filepath=Fpath_Save,Reserved_Time=Tdec_SP,Decision_Time=Tdec_OB,revenue_min_bps=revenue_min_bps,cost_min_bps=cost_min_bps)
        print('TRAF')            
        Optimizer.SCIP_Optimizer_AC_RA(Tslots,TRAF,SP,listname=TRAF_names,SLA=SLA,Ctot=Ctot,save=save,Filepath=Fpath_Save,Reserved_Time=Tdec_SP,Decision_Time=1,revenue_min_bps=revenue_min_bps,cost_min_bps=cost_min_bps)

    # Function to load all the data for the optimization

    def OR_Optimization_RRA(Fpath_Dataset,X,Tslots,SLA,Fpath_Save,Ctot_ratio=1,save=True,Tdec_SP=120,Tdec_OB=30,
                                predicted_traffic=True,revenue_min_bps=1,cost_min_bps=0.5,synthetic_traffic=False):
        SP=[]
        OB=[]
        TRAF=[]
        SP_names=[]
        OB_names=[]
        TRAF_names=[]
    
        
        typee=['*SP_120min.npy','*OB_30min.npy','*traff.npy']
        if predicted_traffic==True:
            typee=['*Alpha_1*.npy','*Alpha_0*.npy','*clean.npy']
        elif synthetic_traffic==True:
            #typee=['*Alpha_1*.npy','*Alpha_0*.npy','*agg_60_s.npy']
            typee=['*Alpha_1*120min*.npy','*Alpha_0*.npy','*agg_60_s.npy']
    
        Functions.load_data(Fpath_Dataset,typee[0],output_list=SP,services_name_list=SP_names,printt=False)
        Functions.load_data(Fpath_Dataset,typee[1],output_list=OB,services_name_list=OB_names,printt=False)
        Functions.load_data(Fpath_Dataset,typee[2],output_list=TRAF,services_name_list=TRAF_names,printt=False)

        

        TRAF_pd=pd.DataFrame(TRAF).T
        Ctot=max(TRAF_pd[:Tslots].sum(axis=1))*Ctot_ratio
        """We load the X array that is the decision variable"""
       
        ## RRA
        
        print('OB_RRA')

        Optimizer.OR_Optimizer_RRA(Tslots,X,OB,SP,listname=OB_names,SLA=SLA,Ctot=Ctot,save=save,Filepath=Fpath_Save,Decision_Time=Tdec_OB,
                                    revenue_min_bps=revenue_min_bps,cost_min_bps=cost_min_bps)




class DF_treatment:
    def string_to_list(string):

        string = string.replace('[','')
        string = string.replace(']','')
        string = string.replace(' ','')
        string = string.split(',')
        string = list(map(float,string))

        return string

    def create_df_Tdecs(f,column_names,cluster,exp_type):

        df=pd.read_csv(f)
        df.columns=column_names
        df_fin=pd.DataFrame(DF_treatment.string_to_list(df[f'{column_names[0]}'][0])).T
        for i in range(len(df.columns)):
            for j in range(len(df.index)):
                       
                df_intermediate=pd.DataFrame(DF_treatment.string_to_list(df[f'{column_names[i]}'][j]),columns=[f'{exp_type}_{column_names[i]}_cluster_{cluster[j]}']).T
                df_fin=df_fin.append(df_intermediate)
        df_fin=df_fin.T
        df_fin=df_fin.drop(columns=[0])

        return df_fin
    
    def create_df_Tdecs_BS(f,column_names,exp_type):
        df=pd.read_csv(f)
        if exp_type=='OP' or exp_type=='Default':
            
            df=df['0']
            df_fin=pd.DataFrame(DF_treatment.string_to_list(df[0])).T
            df_fin.columns=column_names 
            for i in df.index:
                df_intermediate=pd.DataFrame(DF_treatment.string_to_list(df[i])).T
                df_intermediate.columns=column_names
                df_fin=df_fin.append(df_intermediate)
            df_fin.reset_index(drop=True,inplace=True)
            df_fin=df_fin.iloc[1:,:]
            df_fin.reset_index(drop=True,inplace=True)
            columns=[]
            for i in range(len(df_fin.columns)):
                columns.append(f'{exp_type}_{column_names[i]}_cluster_200')
            df_fin.columns=columns
        else:
            df.columns=column_names

            df_fin=pd.DataFrame(DF_treatment.string_to_list(df[f'{column_names[0]}'][0])).T
            for j in range(len(column_names)):
                list_of_antennas_values= []
                for i in df.index:

                    list_of_antennas_values.append(DF_treatment.string_to_list(df[column_names[j]][i])[0])
                df_intermediate=pd.DataFrame(list_of_antennas_values,columns=[f'{exp_type}_{column_names[j]}_cluster_200']).T
                df_fin=df_fin.append(df_intermediate)


            df_fin=df_fin.T
            df_fin=df_fin.drop(columns=[0])
        return df_fin
    
    def create_df_Tdecs_Commune(f,column_names,exp_type, metric_type):
        df=pd.read_csv(f)
        if metric_type=='Efficiency':
            df=df.iloc[:,-1]
        elif metric_type=='Gain':
            df=df.iloc[:,-3]
        elif metric_type=='Traffic':
            df=df.iloc[:,-2]
        else:
            return 'Error, Check metric_type'
        df=pd.DataFrame(df).T.reset_index(drop=True)
        comune_names=[]
        for i in range(len(column_names)):
                comune_names.append(f'{exp_type}_{column_names[i]}_cluster_1')        
        df.columns=comune_names
        return df
    
    def create_df_Tdecs_Commune_SP(f,column_names,exp_type, metric_type):
        df=pd.read_csv(f)
        if metric_type=='Efficiency':
            df=df.iloc[:,2]
        elif metric_type=='Gain':
            df=df.iloc[:,0]
        elif metric_type=='Traffic':
            df=df.iloc[:,1]
        else:
            return 'Error, Check metric_type'
        df=pd.DataFrame(df).T.reset_index(drop=True)
        comune_names=[]
        for i in range(len(column_names)):
                comune_names.append(f'{exp_type}_{column_names[i]}_cluster_1')        
        df.columns=comune_names
        return df
    
    def add_antenna_values_cluster(commmune_df,antenna_df):
        aaa=commmune_df.T
        bbb=antenna_df.T
        ccc=bbb.append(aaa)
        ddd=ccc.T
        ddd=ddd.reindex(natsorted(ddd.columns), axis=1)
        return ddd
    
    def add_commune_value(Cluster_df, value_commune):

        Cluster_tolist=Cluster_df.values.tolist()
        listt=[[] for i in range(len(Cluster_tolist)+1)]
        listt[0]= [value_commune]
        for i in range(1,len(Cluster_tolist)+1):
            listt[i]=Cluster_tolist[i-1]
        final_df=pd.DataFrame(listt)
        return final_df
    
    def add_antenna_commune_default(commune,antenna):

        array=np.array(antenna)
        columns=[]
        commune[6]=array
        cluster=[1,5,10,20,50,100,200]
        for i in range(len(commune.columns)):
            columns.append(f'Default__cluster_{cluster[i]}')
        commune.columns=columns
        return commune
    
    def load_metrics_cluster(Fpath_Dataset,SLA,Fpathh_Variables=False,Tdec_OB=30,Tslots=10080):
        if not Fpathh_Variables:
                Fpath_Variables=f'data/OPT_{Fpath_Dataset}'
        else:
                Fpath_Variables=Fpathh_Variables
                
        gain_list_SP,gain_SP,gain_list_OB,gain_OB,traffic_list_SP,traffic_SP,traffic_list_OB,traffic_OB,Euro_list_SP,Euro_SP,Euro_list_OB,Euro_OB,Eurotraf_list_OB,Eurotraf_OB,Eurotraf_list_SP,Eurotraf_SP=LDV.load_dataset_variables(Fpath_Dataset,Fpath_Variables,Tslots,SLA,output='ALL_Metrics',Tdec_OB=Tdec_OB,Predictions=False)
        
        return gain_SP,gain_OB,traffic_SP,traffic_OB,Eurotraf_SP,Eurotraf_OB
    
class Antenna_Edge_Commune:
    
    ## Function to cluster antennas by location given the list of locations

    def Antenna_aggregation_per_app_per_cluster(Filepath_app,Filepath_cluster,Filesave):
        
        for f in natsorted(glob.glob(Filepath_app+'*.csv')):
            antenna_data_app=pd.read_csv(f)
            for i, clu in enumerate(natsorted(glob.glob(Filepath_cluster+'*.csv'))):
                antenna_cluster=pd.read_csv(clu)
                antenna_data_app_clustered=antenna_data_app[antenna_data_app.LocInfo.isin(antenna_cluster.lacinfo)]
                Filesav=f'{Filesave}cluster_{i}/'
                if not os.path.exists(Filesave):
                    os.mkdir(Filesave)
                if not os.path.exists(Filesav):
                    os.mkdir(Filesav)

                Fsave=Filesav+f.split('/')[-1].split('.')[0]+clu.split('/')[-1].split('.')[0]
                antenna_data_app_clustered.groupby(['timestamp','Time']).sum(['vol_up','vol_dn']).to_csv(Fsave+'.csv')



class RRA_postprocessing:

    def load_decision_X(Decision_X_path,Tdec_OB,predicted_traffic=True,synthetic_traffic=False):
    
        Xs = ['alpha*SP*.npy','alpha*OB*.npy','alpha*traff*.npy']
        if predicted_traffic: 
            Xs = ['X*Alpha_1*.npy','X*Alpha_0*.npy','X*clean*.npy']
        elif synthetic_traffic==True:
            Xs = ['X*Alpha_1*.npy','X*Alpha_0*.npy','X*clean*.npy']
            #Xs=['*120min*.npy','*e-05*.npy','*agg_60_s.npy']


        ## Overbooking
        X_OB = []
        Functions.load_variables(Decision_X_path,Xs[1],output_list=X_OB,printt=False)
        ## Traffic


        X_OB_flat=Functions.flatten_slices(Functions.spread_data(X_OB,Tdec_OB))
        return X_OB_flat
    
    def copy_SP_TRAFFIC(copypath,savepath,Predicted_traffic=True,synthetic_traffic=False):
        tocopy=['SP','_traff_']
        if Predicted_traffic:
            tocopy=['Alpha_1','clean']
        elif synthetic_traffic:
            #tocopy=['Alpha_1','agg_60_s']
            tocopy=['Alpha_1','*agg_60_s']
        print(copypath)
        print(savepath)
        for f in glob.glob(copypath+f'/*{tocopy[0]}*.npy'):
            
            service_name=f.split('/')[-1]
            shutil.copy(f,savepath+'/'+service_name)
            
        for f in glob.glob(copypath+f'/*{tocopy[1]}*.npy'):
            
            service_name=f.split('/')[-1]
            shutil.copy(f,savepath+'/'+service_name)
        print('Filecopy Completed')
