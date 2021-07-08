
from copy import deepcopy
import numpy as np
from pathlib import Path
from Energy_Models_DigitalTwin import Battery, HeatPump, ElectricHeater, EnergyStorage, Building
from citylearn import building_loader

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
    def __init__(self, SOC_bat, SOC_Csto, SOC_Hsto,
                E_hpC_max:float, E_ehH_max:float, C_max:float, H_max:float,
                eta_bat:float, 
                C_p_Csto:float, C_p_Hsto:float, C_p_bat:float,
                time_step:int,  
                COP_C, E_grid_prev:float,
                E_grid_pkhist:float, 
                num_buildings: int,
                E_PV,
                E_NS, H_bd, C_bd, 
                save_memory = True,
                buildings_states_actions = None,
                building_attributes, 
                eta_hp_tech: float = 0.22,   # Technical Efficiency
                t_hp_C: int = 8,             # Target temperature cooling
                eta_ehH: float = 0.9,  
                C_f_Hsto:float = 0.008, C_f_Csto:float = 0.006, C_f_bat :float = 1e-5,    # Battery capacity loss coefficient 
                simulation_period = (0,8759),
                cost_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption'],
                ):
        
        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)
        
        self.buildings_states_actions = buildings_states_actions
        self.buildings_net_electricity_demand = []
        self.building_attributes = building_attributes
        self.cost_function = cost_function
        
        self.SOC_bat = SOC_bat
        self.SOC_Csto = SOC_Csto
        self.SOC_Hsto = SOC_Hsto
        self.eta_bat = eta_bat
        self.E_PV = E_PV
        self.H_bd = H_bd
        self.C_bd = C_bd
        self.E_NS = E_NS
        self.COP_C = COP_C
        
        params_loader = {'data_path':data_path,
                         'building_attributes':self.data_path / self.building_attributes,
                         'weather_file':self.data_path / self.weather_file,
                         'solar_profile':self.data_path / self.solar_profile,
                         'carbon_intensity':self.data_path / self.carbon_intensity,
                         'building_ids':building_ids,
                         'buildings_states_actions':self.buildings_states_actions,
                         'save_memory':save_memory}
        
        self.buildings, self.observation_spaces, self.action_spaces, self.observation_space, self.action_space = building_loader(**params_loader)
        
        self.simulation_period = simulation_period
        self.uid = None
        self.num_buildings = len([i for i in self.buildings])
        self.time_step = 0
        self.reset()
        
        
    def next_hour(self):
        self.time_step = next(self.hour)
        for building in self.buildings.values():
            building.time_step = self.time_step    
        
        
    def step(self, actions):
        
        self.buildings_net_electricity_demand = []
        self.current_carbon_intensity = list(self.buildings.values())[0].sim_results['carbon_intensity'][self.time_step]
        electric_demand = 0
        elec_consumption_electrical_storage = 0
        elec_consumption_dhw_storage = 0
        elec_consumption_cooling_storage = 0
        elec_consumption_dhw_total = 0
        elec_consumption_cooling_total = 0
        elec_consumption_appliances = 0
        elec_generation = 0
        
        # For the decentralized agent
        assert len(actions) == self.num_buildings, "The length of the list of actions should match the length of the list of buildings."
        
        for a, (uid, building) in zip(actions, Buildings.items()):
            
            
            if self.buildings_states_actions[uid]['actions']['electrical_storage']:
                

                if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                    
                    # Cooling
                    _electric_demand_cooling = building.set_storage_cooling(a[0], C_p_Csto, self.C_bd)
                    elec_consumption_cooling_storage += building._electric_consumption_cooling_storage
                        
                    # 'Electrical Storage' & 'Cooling Storage' & 'DHW Storage'
                    if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                        
                        # DHW
                        _electric_demand_dhw = building.set_storage_heating(a[1], E_ehH_max, C_Hsto, self.SOC_Hsto, self.H_bd)
                        elec_consumption_dhw_storage += building._electric_consumption_dhw_storage
                            
                        # Electrical
                        _electric_demand_electrical_storage = building.set_storage_electrical(a[2], C_p_bat, self.SOC_bat)
                        elec_consumption_electrical_storage += _electric_demand_electrical_storage
                            
                            # 'Electrical Storage' & 'Cooling Storage'
                    else:
                        _electric_demand_dhw = building.set_storage_heating(0.0, E_ehH_max, C_Hsto, self.SOC_Hsto, self.H_bd)
                        # Electrical
                        _electric_demand_electrical_storage = building.set_storage_electrical(a[1], C_p_bat, self.SOC_bat)
                        elec_consumption_electrical_storage += _electric_demand_electrical_storage
                else:
                    
                    _electric_demand_cooling = building.set_storage_cooling(0.0)
                        # 'Electrical Storage' & 'DHW Storage'
                    if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                        # DHW
                        _electric_demand_dhw = building.set_storage_heating(a[0], E_ehH_max, C_Hsto, self.SOC_Hsto, self.H_bd)
                        elec_consumption_dhw_storage += building._electric_consumption_dhw_storage
                            
                        # Electrical
                        _electric_demand_electrical_storage = building.set_storage_electrical(a[1], C_p_bat, self.SOC_bat)
                        elec_consumption_electrical_storage += _electric_demand_electrical_storage

                    # 'Electrical Storage'
                    else:
                        _electric_demand_dhw = building.set_storage_heating(0.0)
                        # Electrical
                        _electric_demand_electrical_storage = building.set_storage_electrical(a[0], C_p_bat, self.SOC_bat)
                        elec_consumption_electrical_storage += _electric_demand_electrical_storage
                
            
            
            else:
                
                _electric_demand_electrical_storage = 0.0
                    
                if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                    # Cooling
                    _electric_demand_cooling = building.set_storage_cooling(a[0], C_p_Csto, self.C_bd)
                    elec_consumption_cooling_storage += building._electric_consumption_cooling_storage

                    if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                        # DHW
                        _electric_demand_dhw = building.set_storage_heating(a[1], E_ehH_max, C_Hsto, self.SOC_Hsto, self.H_bd)
                        elec_consumption_dhw_storage += building._electric_consumption_dhw_storage

                    else:
                        _electric_demand_dhw = building.set_storage_heating(0.0, E_ehH_max, C_Hsto, self.SOC_Hsto, self.H_bd)

                else:
                    _electric_demand_cooling = building.set_storage_cooling(0.0, C_p_Csto, self.C_bd)
                    # DHW
                    _electric_demand_dhw = building.set_storage_heating(a[0], E_ehH_max, C_Hsto, self.SOC_Hsto, self.H_bd)
                    elec_consumption_dhw_storage += building._electric_consumption_dhw_storage

        
        
                    
        # Total heating and cooling electrical loads
        elec_consumption_cooling_total += _electric_demand_cooling
        elec_consumption_dhw_total += _electric_demand_dhw

        # Electrical appliances
        _non_shiftable_load = building.get_non_shiftable_load()
        elec_consumption_appliances += _non_shiftable_load

        # Solar generation
        _solar_generation = building.get_solar_power()
        elec_generation += _solar_generation

        # Adding loads from appliances and subtracting solar generation to the net electrical load of each building
        building_electric_demand = round(_electric_demand_electrical_storage + _electric_demand_cooling + _electric_demand_dhw + _non_shiftable_load - _solar_generation, 4)

        # Electricity consumed by every building
        building.current_net_electricity_demand = building_electric_demand
        self.buildings_net_electricity_demand.append(-building_electric_demand)    

        # Total electricity consumption
        electric_demand += building_electric_demand 
        
    self.next_hour()
        
        
    def reset(self):
        
        
        #Initialization of variables
        self.hour = iter(np.array(range(self.simulation_period[0], self.simulation_period[1] + 1)))
        self.next_hour()
            
        self.carbon_emissions = []
        self.net_electric_consumption = []
        self.net_electric_consumption_no_storage = []
        self.net_electric_consumption_no_pv_no_storage = []
        self.electric_consumption_electric_storage = []
        self.electric_consumption_dhw_storage = []
        self.electric_consumption_cooling_storage = []
        self.electric_consumption_electrical_storage = []
        self.electric_consumption_dhw = []
        self.electric_consumption_cooling = []
        self.electric_consumption_appliances = []
        self.electric_generation = []
        
        self.cumulated_reward_episode = 0
        self.current_carbon_intensity = 0
        
        self.reward_function = reward_function_ma(len(self.building_ids), self.get_building_information())
            
        self.state = []
        for uid, building in self.buildings.items():
            building.reset()
            s = []
            for state_name, value in zip(self.buildings_states_actions[uid]['states'], self.buildings_states_actions[uid]['states'].values()):
                if value == True:
                    if state_name == 'net_electricity_consumption':
                        s.append(building.current_net_electricity_demand)
                    elif (state_name != 'cooling_storage_soc') and (state_name != 'dhw_storage_soc') and (state_name != 'electrical_storage_soc'):
                        s.append(building.sim_results[state_name][self.time_step])
                    elif state_name == 'cooling_storage_soc':
                        s.append(0.0)
                    elif state_name == 'dhw_storage_soc':
                        s.append(0.0)
                    elif state_name == 'electrical_storage_soc':
                        s.append(0.0)

                self.state.append(np.array(s, dtype=np.float32))
                
            self.state = np.array(self.state, dtype='object')
            
        return self._get_ob()
        
        
        
        

    def _get_ob(self):            
        return self.state

#########################
##########################
############################