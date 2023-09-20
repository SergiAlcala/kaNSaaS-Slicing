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

NP=['075']

SP=['15']
revenue_min_bps = 1 ### Revenue per minute per bps ###


Ratio_cost_rev_list=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]




for np in NP:
    for sp in SP:
        for rat in Ratio_cost_rev_list:

    
            nps,sps,ratio_cost_rev=np,sp,rat
            cost_min_bps = ratio_cost_rev * revenue_min_bps ### Cost per minute per bps ###

            Fpath_Dataset_Raw='./kaNSaaS-Slicing/Syn_Dataset/npys/Fcasting_exps_Synthetic_data_noisy/120/'
            Fpath_Dataset=f'{Fpath_Dataset_Raw}/AC_RA'

            Fpath_Dataset_RRA=f'{Fpath_Dataset_Raw}/RRA'


            Fpath_Save_Raw=f'./kaNSaaS-Slicing/Syn_Dataset/OPT_npys/Fcasting_exps_Synthetic_data_noisy/120/'
            Fpath_Save=f'{Fpath_Save_Raw}/AC_RA'
            Fpath_Save_RRA=f'{Fpath_Save_Raw}/RRA_OR'


            Fpath_Dataset=f'{Fpath_Dataset}/T_dec_30_Tadm_120/'
            Fpath_Dataset_RRA=f'{Fpath_Dataset_RRA}/T_dec_30_Tadm_120/'

            Fpath_Save=f'{Fpath_Save}/Ratio_{ratio_cost_rev}/T_dec_30_Tadm_120/'
            Fpath_Save_RRA=f'{Fpath_Save_RRA}/Ratio_{ratio_cost_rev}/T_dec_30_Tadm_120/'

            Decision_X_path=Fpath_Save
            print(Fpath_Dataset)
            print(Decision_X_path)
            # #### Firstly we optimize for Admision Control and Resource Allocation, where we obtain the selected slices 
            O.SCIP_Optimization_AC_RA(Fpath_Dataset,Tslots,SLA,Fpath_Save,Ctot_ratio=1,Tdec_OB=Duration_idx,predicted_traffic=predicted_traffic
            ,revenue_min_bps=revenue_min_bps,cost_min_bps=cost_min_bps,synthetic_traffic=synthetic_traffic)

            ### Then we optimize for RRA, where we define the percentage of the slices to be used in the RRA
            Decision_X_list=RRA_post.load_decision_X(Decision_X_path,Tdec_OB=Duration_idx,predicted_traffic=predicted_traffic,synthetic_traffic=synthetic_traffic)
            O.OR_Optimization_RRA(Fpath_Dataset_RRA,Decision_X_list,Tslots,SLA,Fpath_Save_RRA,Ctot_ratio=1,Tdec_OB=Duration_idx,predicted_traffic=predicted_traffic
            ,revenue_min_bps=revenue_min_bps,cost_min_bps=cost_min_bps,synthetic_traffic=synthetic_traffic)
            RRA_post.copy_SP_TRAFFIC(Decision_X_path,Fpath_Save_RRA,Predicted_traffic=predicted_traffic,synthetic_traffic=synthetic_traffic)



