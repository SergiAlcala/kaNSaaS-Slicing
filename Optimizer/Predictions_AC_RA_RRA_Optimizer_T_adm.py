import sys
sys.path.append('./kaNSaaS-Slicing/')


from sympy import rational_interpolate
from Global_Functions import Functions as F
from Global_Functions import Optimizer as O
from Global_Functions import *
from Global_Functions import RRA_postprocessing as RRA_post
from multiprocessing import Pool

Tslots=9481
SLA=F.gain_lin_approx_5


Duration_idx=30

predicted_traffic=False

if predicted_traffic:
    synthetic_traffic=False
else:
    synthetic_traffic=True


revenue_min_bps = 1 ### Revenue per minute per bps ###

time_admission=[120]


time_decision_NP=[5,15,30,60,120]
Load_type=['AC_RA','RRA']
Simulations=[]
duration_list=[]
for i in range(len(time_decision_NP)):
    Simulations.append(f'T_dec_{time_decision_NP[i]}_Tadm_{time_admission[0]}')
    duration_list.append(time_decision_NP[i])

for i in range(len(Simulations)):
        

    
            ratio_cost_rev=0.9
            cost_min_bps = ratio_cost_rev * revenue_min_bps ### Cost per minute per bps ###


            Fpath_Dataset_Raw='./kaNSaaS-Slicing/Syn_Dataset/npys/Fcasting_exps_Synthetic_data_noisy/120/'
            Fpath_Dataset=f'{Fpath_Dataset_Raw}/AC_RA'

            Fpath_Dataset_RRA=f'{Fpath_Dataset_Raw}/RRA'


            Fpath_Save_Raw=f'./kaNSaaS-Slicing/Syn_Dataset/OPT_npys/Fcasting_exps_Synthetic_data_noisy/120/'
            Fpath_Save=f'{Fpath_Save_Raw}/AC_RA'
            Fpath_Save_RRA=f'{Fpath_Save_Raw}/RRA_OR'


            Fpath_Dataset=f'{Fpath_Dataset}/{Simulations[i]}/'
            Fpath_Dataset_RRA=f'{Fpath_Dataset_RRA}/{Simulations[i]}/'

            Fpath_Save=f'{Fpath_Save}/Ratio_{ratio_cost_rev}/{Simulations[i]}/'
            Fpath_Save_RRA=f'{Fpath_Save_RRA}/Ratio_{ratio_cost_rev}/{Simulations[i]}/'

         
            Decision_X_path=Fpath_Save
            # print(Fpath_Dataset)
            # print(Decision_X_path)
            ### Firstly we optimize for Admision Control and Resource Allocation, where we obtain the selected slices 
            O.SCIP_Optimization_AC_RA(Fpath_Dataset,Tslots,SLA,Fpath_Save,Ctot_ratio=1,Tdec_OB=duration_list[i],predicted_traffic=predicted_traffic
            ,revenue_min_bps=revenue_min_bps,cost_min_bps=cost_min_bps,synthetic_traffic=synthetic_traffic)

            ### Then we optimize for RRA, where we define the percentage of the slices to be used in the RRA
            Decision_X_list=RRA_post.load_decision_X(Decision_X_path,Tdec_OB=duration_list[i],predicted_traffic=predicted_traffic,synthetic_traffic=synthetic_traffic)
            O.OR_Optimization_RRA(Fpath_Dataset_RRA,Decision_X_list,Tslots,SLA,Fpath_Save_RRA,Ctot_ratio=1,Tdec_OB=duration_list[i],predicted_traffic=predicted_traffic
            ,revenue_min_bps=revenue_min_bps,cost_min_bps=cost_min_bps,synthetic_traffic=synthetic_traffic)
            RRA_post.copy_SP_TRAFFIC(Decision_X_path,Fpath_Save_RRA,Predicted_traffic=predicted_traffic,synthetic_traffic=synthetic_traffic)
