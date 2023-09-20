# Get all results from load_variables

def get_results(SP,OB,TRAF,Slice_names,alpha_SP,X_SP,alpha_OB,X_OB,alpha_TRAFF,X_TRAFF,Ctot,value_SP_pd ,value_OB_pd,value_TRAF_pd,weight_SP_pd,weight_OB_pd,weight_TRAF_pd,real_traffic_SP_pd,real_traffic_OB_pd,real_traffic_TRAF_pd,
expected_traff_OPT_List_SP_pd,expected_traff_OPT_List_OB_pd,expected_traff_OPT_List_TRAF_pd,service_SP_pd,service_OB_pd,service_TRAF_pd,total_service_SP_pd,total_service_OB_pd,total_service_TRAF_pd,
gain_list_SP,gain_SP,gain_list_OB,gain_OB,traffic_list_SP,traffic_SP,traffic_list_OB,traffic_OB,EuroTraff_list_OB,EuroTraff_OB,EuroTraff_list_SP,EuroTraff_SP,Euro_list_SP,Euro_SP,
Euro_list_OB,Euro_OB,Euro_list_TRAF,Euro_TRAF,Beta_SP,Beta_OB,Beta_TRAF,Net_Ben_OPT_List_SP ,Net_Ben_OPT_List_OB,Net_Ben_OPT_List_TRAF,NET_GAIN_list_SP
            ,NET_GAIN_SP,NET_GAIN_list_OB,NET_GAIN_OB,allocated_Traffic_list_SP,allocated_Traffic_SP,allocated_Traffic_list_OB,allocated_Traffic_OB,
            SLA_Cost_SP,SLA_Cost_OB,SLA_Cost_TRAF,Copex_Cost_SP,Copex_Cost_OB,Copex_Cost_TRAF,SLA_Cost_Metric_SP,SLA_Cost_Metric_list_SP,SLA_Cost_Metric_list_OB,SLA_Cost_Metric_OB,Copex_cost_Metric_list_SP,Copex_cost_Metric_SP
            ,Copex_cost_Metric_list_OB,Copex_cost_Metric_OB):
    results={'SP':SP,
    'OB':OB,
    'TRAF':TRAF,
    'Slice_names':Slice_names,
    'alpha_SP':alpha_SP,'X_SP':X_SP,'alpha_OB':alpha_OB,'X_OB':X_OB,'alpha_TRAFF':alpha_TRAFF,'X_TRAFF':X_TRAFF,
    'Ctot':Ctot,
    'value_SP_pd':value_SP_pd,'value_OB_pd':value_OB_pd,'value_TRAF_pd':value_TRAF_pd,
    'weight_SP_pd':weight_SP_pd,'weight_OB_pd':weight_OB_pd,'weight_TRAF_pd':weight_TRAF_pd,
    'real_traffic_SP_pd':real_traffic_SP_pd,'real_traffic_OB_pd':real_traffic_OB_pd,'real_traffic_TRAF_pd':real_traffic_TRAF_pd,
    'expected_traff_OPT_List_SP_pd':expected_traff_OPT_List_SP_pd,'expected_traff_OPT_List_OB_pd':expected_traff_OPT_List_OB_pd,'expected_traff_OPT_List_TRAF_pd':expected_traff_OPT_List_TRAF_pd,
    'service_SP_pd':service_SP_pd,'service_OB_pd':service_OB_pd,'service_TRAF_pd':service_TRAF_pd,
    'total_service_SP_pd':total_service_SP_pd,'total_service_OB_pd':total_service_OB_pd,'total_service_TRAF_pd':total_service_TRAF_pd,
    'gain_list_SP':gain_list_SP,'gain_SP':gain_SP,'gain_list_OB':gain_list_OB,'gain_OB':gain_OB,
    'traffic_list_SP':traffic_list_SP,'traffic_SP':traffic_SP,'traffic_list_OB':traffic_list_OB,'traffic_OB':traffic_OB,
    'EuroTraff_list_OB':EuroTraff_list_OB,'EuroTraff_OB':EuroTraff_OB,'EuroTraff_list_SP':EuroTraff_list_SP,'EuroTraff_SP':EuroTraff_SP,'Euro_list_SP':Euro_list_SP,'Euro_SP':Euro_SP,
    'Euro_list_OB':Euro_list_OB,'Euro_OB':Euro_OB,'Euro_list_TRAF':Euro_list_TRAF,'Euro_TRAF':Euro_TRAF,
    'Beta_SP':Beta_SP,'Beta_OB':Beta_OB,'Beta_TRAF':Beta_TRAF,
    'Net_Ben_OPT_List_SP':Net_Ben_OPT_List_SP,'Net_Ben_OPT_List_OB':Net_Ben_OPT_List_OB,'Net_Ben_OPT_List_TRAF':Net_Ben_OPT_List_TRAF,
    'allocated_Traffic_list_SP':allocated_Traffic_list_SP,'allocated_Traffic_SP':allocated_Traffic_SP,'allocated_Traffic_list_OB':allocated_Traffic_list_OB,'allocated_Traffic_OB':allocated_Traffic_OB,
    'NET_GAIN_list_SP':NET_GAIN_list_SP,'NET_GAIN_SP':NET_GAIN_SP,'NET_GAIN_list_OB':NET_GAIN_list_OB,'NET_GAIN_OB':NET_GAIN_OB,
    'SLA_Cost_SP':SLA_Cost_SP,     'SLA_Cost_OB':SLA_Cost_OB,    'SLA_Cost_TRAF':SLA_Cost_TRAF,
    'Copex_Cost_SP':Copex_Cost_SP,     'Copex_Cost_OB':Copex_Cost_OB,     'Copex_Cost_TRAF':Copex_Cost_TRAF,
    'SLA_Cost_Metric_SP':SLA_Cost_Metric_SP, 'SLA_Cost_Metric_list_SP':SLA_Cost_Metric_list_SP,    'SLA_Cost_Metric_list_OB':SLA_Cost_Metric_list_OB,     'SLA_Cost_Metric_OB':SLA_Cost_Metric_OB,
    'Copex_cost_Metric_list_SP':Copex_cost_Metric_list_SP,     'Copex_cost_Metric_SP':Copex_cost_Metric_SP,     'Copex_cost_Metric_list_OB':Copex_cost_Metric_list_OB,     'Copex_cost_Metric_OB':Copex_cost_Metric_OB
    }
    return results

def get_3Dresults(SP,OB,TRAF,traffic_list_OB,traffic_OB,EuroTraff_list_OB,EuroTraff_OB,Euro_list_OB,Euro_OB,Euro_list_TRAF,Euro_TRAF,Net_Ben_OPT_List_OB,Net_Ben_OPT_List_TRAF,NET_GAIN_list_OB,NET_GAIN_OB):

    results={'SP':SP,
    'OB':OB,
    'TRAF':TRAF,
    
    'traffic_list_OB':traffic_list_OB,'traffic_OB':traffic_OB,
    'EuroTraff_list_OB':EuroTraff_list_OB,'EuroTraff_OB':EuroTraff_OB,
    'Euro_list_OB':Euro_list_OB,'Euro_OB':Euro_OB,'Euro_list_TRAF':Euro_list_TRAF,'Euro_TRAF':Euro_TRAF,
  
    'Net_Ben_OPT_List_OB':Net_Ben_OPT_List_OB,'Net_Ben_OPT_List_TRAF':Net_Ben_OPT_List_TRAF,
  
    'NET_GAIN_list_OB':NET_GAIN_list_OB,'NET_GAIN_OB':NET_GAIN_OB}
    return results