from copy import deepcopy
import numpy as np
from citylearn import CityLearn
from pathlib import Path

import sys
import warnings
import utils
import time

import numpy as np
import pandas as pd

if not sys.warnoptions:
    warnings.simplefilter("ignore")

## local imports
from predictor import *


class DigitalTwin(object):
    def __init__(
        self,
        SOC_bat_next: float,
        SOC_bat_current: float,
        SOC_Csto_current:float,
        SOC_Csto_previous:float,
        SOC_Csto_next:float,
        SOC_Hsto_current:float,
        SOC_Hsto_previous:float, 
        C_f_Hsto:float,
        eta_Hsto:float
        COP_C_next:float,
        C_f_Csto:float,
        eta_Csto:float
        action_bat:float,
        E_bat_max: float,
        E_grid:float,
        E_sell:float,
        E_NS:float,
        E_PV:float,
        E_PV_next:float,
        E_netgrid_next:float,
        E_NS_next:float,
        C_p_Hsto:float,
        dhw_tank_cap:float,
        eta_hp_tech: float = 0.22,   # Technical Efficiency
        t_hp_C: int = 8,             # Target temperature cooling
        eta_ehH: float = 0.9,         # Electric Heater efficiency
        C_f_bat :float = 1e-5,    # Battery capcaity loss coefficient
        C_f_bat: int = 0,         # Battery loss coefficient
        num_buildings: int,
    ):
        
    def Estimate_E_bat_max(self, SOC_bat_next: float,
                           SOC_bat_current: float,
                           C_f_bat: float):
        
        C_p_bat_est = self.Estimate_C_p_bat(action_bat, 
                        E_grid,
                        E_sell,
                        E_NS,
                        E_PV)
        
        eta_bat_est = self.Estimate_eta_bat(self, C_f_bat: float,
                        SOC_bat_current: float,
                        SOC_bat_previous, 
                        action_bat)
        
        
        E_bat_max_est = -C_p_bat_est*((SOC_bat_next - (1 - C_f_bat)*SOC_bat_current))/eta_bat_est
        
        return E_bat_max_est
    
    
    def Estimate_eta_bat(self, C_f_bat: float,
                        SOC_bat_current: float,
                        SOC_bat_previous: float 
                        action_bat: float):
        
        eta_bat_est = (SOC_bat_current - (1 - C_f_bat)*SOC_bat_previous)/action_bat
        
        return eta_bat_est
    
    
    def Estimate_max_power_bat(self, SOC_bat:float, E_bat_max: float):
        
        
        if SOC_bat<0.6:
            
            max_power_bat_est = E_bat_max
            
        else:
            
            max_power_bat_est = max(E_bat_max, (1 - SOC_bat)*C_p_bat)
            
        return max_power_bat_est
        
    
    def Estimate_C_p_bat(self, action_bat:float, 
                        E_grid:float,
                        E_sell:float,
                        E_NS:float,
                        E_PV:float):
        
        C_p_bat_est = (1/(action_bat))*(E_PV + E_grid + E_sell - E_NS)
        
        return C_p_bat_est
    
    ######## Estimate Cooling and Heating storage capacity and load estimation
    #@Zhiyao @Mingyu  # Section 3.2
    
    ## Insert Code Here
    
    # 3.2 Estimation of storage capacity
    
    
    
    
    # Estimate ratio of cooling load and cooling storage capacity
    def EstimateRatios_Cbd_CpCsto(self, SOC_Csto_current:float,
                                  C_f_Csto:float,
                                  SOC_Csto_previous:float,
                                  eta_Csto:float):
        
        Ratio_Cbd_CpCsto = -(SOC_Csto_current - (1 - C-_Csto)*SOC_Csto_previous)/eta_Csto
    
        return Ratio_Cbd_CpCsto
    
    def Estimate_C_p_Csto(self, COP_C_next:float,
                         E_PV_next:float,
                         E_netgrid_next:float,
                         E_NS_next:float,
                         SOC_Csto_previous:float, 
                         SOC_Csto_next:float,
                         SOC_Csto_current:float,
                         C_f_Csto:float,
                         eta_Csto:float
                         ):
        
        
        
        Ratio_Cbd_CpCsto = self.EstimateRatios_Cbd_CpCsto(SOC_Csto_current, C_f_csto, SOC_Csto_previous, eta_Csto)
        
        num = COP_C_next*(E_PV_next + E_netgrid_next + E_NS_next)
        den1 = (SOC_Csto_next - (1 - C_f_Csto)*SOC_Csto_current)/eta_Csto
        den2 = Ratio_Cbd_CpCsto
        
        den = den1 + den2
        
        C_p_Csto_est = num/den
        
        return C_p_Csto_est
    
    def C_p_Hsto_est(self, SOC_Hsto_next:float,
                     SOC_Hsto_current:float,
                     C_f_Hsto:float,
                     eta_Hsto:float):
        
        
        
        
    def Estimate_C_bd(self, SOC_Csto_next:float,
                     SOC_Csto_current:float,
                     C_f_Csto:float,
                     eta_Csto:float):
        
        C_bd_est = (SOC_Csto_next -(1 - C_f_Csto)*SOC_Csto_current)/eta_Csto
        
        return C_bd_est
        
        
        
    def Estimate_H_bd_t(self, eta_eh_H:float = 0.9,
                        action_bat:float,
                       E_PV_current:float,
                       E_grid_current:float,
                       E_sell_current:float, 
                       E_NS_current:float,
                       SOC_bat_current:float,
                       SOC_bat_previous:float,
                       C_f_bat:float,
                       eta_bat:float,
                       SOC_Hsto_current:float,
                       SOC_Hsto_previous:float, 
                       C_f_Hsto:float,
                       eta_Hsto:float):
        
        C_p_Hsto_est = self.Estimate_C_p_Hsto()
        C_p_bat_est = self.Estimate_C_p_bat(action_bat, 
                        E_grid_current,
                        E_sell_current,
                        E_NS_current,
                        E_PV_current)
        
        
        A1 = (SOC_bat_current - (1 - C_f_bat)*SOC_bat_previous)/eta_bat
        B1 = (SOC_Hsto_current - (1 - C_f_Hsto)*SOC_Hsto_previous)/eta_Hsto
    
        H_bd_est = eta_ehH*(E_PV_current + E_grid_current - E_NS_current - A1*C_p_bat_est) - B1*C_p_Hsto_est
        
        return H_bd_est
    
    
    ###################################
    
    
    # 3.3 Heat Pump Nominal Power
    
    def Estimate_E_hpC_max(self, eta_hp_tech:float = 0.22,
                          t_hp_C:int = 8,
                          temp_t:float,
                          C_max_est:float,
                          ):
        
        # Line 252 CityLearn.py
        COP_C_est = eta_hp_tech*((t_hp_C + 273.15)/(temp_t - t_hp_C))
        
        # from citylearn.py#L52 - We assume that the heat pump is always large enough to meet the highest heating or cooling demand of the building
        building.dhw_heating_device.nominal_power = np.array(building.sim_results['dhw_demand']/building.dhw_heating_device.cop_heating).max()
        E_hpC_max1 = building.dhw_heating_device.nominal_power    #estimated from estimated COP and observedd cooling loads
        
        E_hpC_max2 = C_max_est/20        # Maximum Cooling load
        
        E_hpC_max_est = max(E_hpC_max1, E_hpC_max2)
        
        return E_hpC_max_est
    
    
    # 3.4 Section
        
    def Estimate_E_ehH_max(self, eta_ehH: float = 0.9,
                           C_p_Hsto:float,
                           dhw_tank_cap:float
                          ):
        
        # C_p_Hsto - Heating storage capacity
        # H_max - maximum heating load
        # Getting the upper on the heating storage actio using citylearn.py line 218
        
        if dhw_tank_cap > 0.000001:
            a_high = 1/dhw_tank_cap
        
        else:
            a_high = 1
        
        # Using the relation Hmax  = C_p_Hsto* upper bound on heat store action
        H_max = C_p_Hsto*a_high

        E_ehH_max_est = H_max/eta_ehH      # Electric heater nominal power
        
        return E_ehH_max_est
    
    
    