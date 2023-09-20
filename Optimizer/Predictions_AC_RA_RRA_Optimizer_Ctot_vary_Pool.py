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

revenue_min_bps = 1 ### Revenue per minute per bps ###
cost_min_bps = 0.9 ### Cost per minute per bps ###



predicted_traffic=False

if predicted_traffic:
    synthetic_traffic=False
else:
    synthetic_traffic=True


NP=['075']

SP=['15']
def myMultiOpt(idx):


    sp,Ctot=idx

    ### Synthetic Data ###
    Fpath_Dataset_Raw='./kaNSaaS-Slicing/Syn_Dataset/npys/Fcasting_exps_Synthetic_data_noisy/120/'
    Fpath_Dataset=f'{Fpath_Dataset_Raw}/AC_RA'

    Fpath_Dataset_RRA=f'{Fpath_Dataset_Raw}/RRA'


    Fpath_Save_Raw=f'./kaNSaaS-Slicing/Syn_Dataset/OPT_npys/Fcasting_exps_Synthetic_data_noisy/120/'
    Fpath_Save=f'{Fpath_Save_Raw}/AC_RA'
    Fpath_Save_RRA=f'{Fpath_Save_Raw}/RRA_OR'


    Fpath_Dataset=f'{Fpath_Dataset}/T_dec_30_Tadm_120/'
    Fpath_Dataset_RRA=f'{Fpath_Dataset_RRA}/T_dec_30_Tadm_120/'

    Fpath_Save=f'{Fpath_Save}/C_TOT/C_TOT_Ratio_{Ctot}/T_dec_30_Tadm_120/'
    Fpath_Save_RRA=f'{Fpath_Save_RRA}/C_TOT/C_TOT_Ratio_{Ctot}/T_dec_30_Tadm_120/'


    Decision_X_path=Fpath_Save
    print(Fpath_Dataset)
    print(Decision_X_path)

    ### Firstly we optimize for Admision Control and Resource Allocation, where we obtain the selected slices 
    O.SCIP_Optimization_AC_RA(Fpath_Dataset,Tslots,SLA,Fpath_Save,Ctot_ratio=Ctot,Tdec_OB=Duration_idx,predicted_traffic=predicted_traffic
    ,revenue_min_bps=revenue_min_bps,cost_min_bps=cost_min_bps,synthetic_traffic=synthetic_traffic)

    ### Then we optimize for RRA, where we define the percentage of the slices to be used in the RRA
    Decision_X_list=RRA_post.load_decision_X(Decision_X_path,Tdec_OB=Duration_idx,predicted_traffic=predicted_traffic,synthetic_traffic=synthetic_traffic)

    O.OR_Optimization_RRA(Fpath_Dataset_RRA,Decision_X_list,Tslots,SLA,Fpath_Save_RRA,Ctot_ratio=Ctot,Tdec_OB=Duration_idx,predicted_traffic=predicted_traffic
    ,revenue_min_bps=revenue_min_bps,cost_min_bps=cost_min_bps,synthetic_traffic=synthetic_traffic)

    RRA_post.copy_SP_TRAFFIC(Decision_X_path,Fpath_Save_RRA,Predicted_traffic=predicted_traffic,synthetic_traffic=synthetic_traffic)

Ctots=[0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
pair_list_duration_OP=[]

for sp in SP:
    for Ctot in Ctots:
        pair_list_duration_OP.append([sp,Ctot])

if __name__ == '__main__':
    with Pool(len(Ctots)) as p:
        p.map(myMultiOpt,pair_list_duration_OP)