from copy import deepcopy
import numpy as np
from pathlib import Path
from Energy_Models_DigitalTwin import Battery, HeatPump, ElectricHeater, EnergyStorage, Building
from citylearn import building_loader
# from energy_models import Battery, HeatPump, ElectricHeater, EnergyStorage, Building

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
    def __init__(self,
                eta_hp_tech: float = 0.22,   # Technical Efficiency
                t_hp_C: int = 8,           # Target temperature cooling
                eta_ehH: float = 0.9,
                C_f_Hsto:float = 0.008,
                C_f_Csto:float = 0.006,
                C_f_bat :float = 1e-5,
                states,
                num_buildings: int = 9,
                save_memory = True,
                buildings_states_actions = None,simulation_period = (0,8759),cost_function =
                ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
                ):
        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)

        # States that we can directly get from the observed states
        self.E_NS = states[:,23]
        self.net_electricity_consumption = states[:,28]
        self.SOC_Csto = states[:,25]
        self.SOC_Hsto = states[:,26]
        self.SOC_bat = states[:,27]


        # States that will require a predictor/oracle
        self.E_hpC_max = E_hpC_max
        self.E_ehH_max = E_ehH_max
        self.E_bat_max = E_bat_max
        self.C_p_Csto = C_p_Csto
        self.C_p_Hsto = C_p_Hsto
        self.C_p_bat = C_p_bat
        self.eta_bat = eta_bat
        self.E_PV = E_PV
        self.H_bd = H_bd
        self.C_bd = C_bd
        self.COP_C = COP_C
        self.C_max = C_max
        self.H_max = H_max

        # Instantiating the battery, heat pump, electric heater, energy storage, and Building class

        self.battery = Battery(capacity = self.C_p_bat,
                                         capacity_loss_coef = self.C_f_bat,
                                         loss_coef = 0,
                                         efficiency = self.eta_bat,
                                         nominal_power = self.E_bat_max,
                                         power_efficiency_curve = attributes['Battery']['power_efficiency_curve'],
                                         capacity_power_curve = attributes['Battery']['capacity_power_curve'],
                                         save_memory = save_memory)

        self.heat_pump = HeatPump(nominal_power = self.E_hpC_max,
                                 eta_tech = self.eta_hp_tech,
                                 t_target_heating = attributes['Heat_Pump']['t_target_heating'],
                                 t_target_cooling = attributes['Heat_Pump']['t_target_cooling'], save_memory = save_memory)

        self.electric_heater = ElectricHeater(nominal_power = self.E_ehH_max,
                                             efficiency = self.eta_ehH, save_memory = save_memory)

        self.chilled_water_tank = EnergyStorage(capacity = self.C_p_Csto,
                                               loss_coef = self.C_f_Csto, save_memory = save_memory)

        self.dhw_tank = EnergyStorage(capacity = self.C_p_Hsto,
                                     loss_coef = self.C_f_Hsto, save_memory = save_memory)

        self.building = Building(buildingId = uid, dhw_storage = self.dhw_tank,
                            cooling_storage = self.chilled_water_tank,
                            electrical_storage = self.battery,
                            dhw_heating_device = self.electric_heater,
                            cooling_device = self.heat_pump, save_memory = save_memory)


        self.buildings = {}
        self.buildings_states_actions = buildings_states_actions
        self.buildings_net_electricity_demand = []
        self.cost_function = cost_function

        self.simulation_period = simulation_period
        self.uid = None
        self.num_buildings = 9

#     def next_hour(self, total_it):
#         self.time_step = total_it % 24
#         for building in self.buildings.values():
#             building.time_step = self.time_step


    def buildings_load(self):

        self.buildings = {}

        for uid in range(9):
            self.buildings[uid] = self.building(uid, self.dhw_tank, self.chilled_water_tank,
                                               self.battery, self.electric_heater,
                                               self.heat_pump)

        return self.buildings


    def set_state(self, states, total_it):

        # States that we can directly get from the observed states
        # Getting state for current time step and 9 buildings
        self.E_NS = states[:,23]
        self.net_electricity_consumption = states[:,28]
        self.SOC_Csto = states[:,25]
        self.SOC_Hsto = states[:,26]
        self.SOC_bat = states[:,27]


        data_est = # Predictor function will give estimates of all other states needed

        time_step = total_it % 24

        # # Getting state for current time step and 9 buildings
        self.E_hpC_max = data_est['E_hpc_max'][time_step,:]
        self.E_ehH_max = data_est['E_ehH_max'][time_step,:]
        self.E_bat_max = data_est['E_bat_max'][time_step,:]
        self.C_p_Csto = data_est['C_p_Csto'][time_step,:]
        self.C_p_Hsto = data_est['C_p_Hsto'][time_step,:]
        self.C_p_bat = data_est['C_p_bat'][time_step,:]
        self.eta_bat = data_est['eta_bat'][time_step,:]
        self.E_PV = data_est['E_PV'][time_step,:]
        self.H_bd = data_est['H_bd'][time_step,:]
        self.C_bd = data_est['C_bd'][time_step,:]
        self.COP_C = data_est['COP_C'][time_step,:]
        self.C_max = data_est['C_max'][time_step,:]
        self.H_max = data_est['H_max'][time_step,:]

        self.buildings = self.buildings_load()



    def transition(self, actions, states, total_it):


        # Initialising the next states that we will get from the digital twin
        self.buildings_net_electricity_demand = []
        electric_demand = 0
        elec_consumption_electrical_storage = 0
        elec_consumption_dhw_storage = 0
        elec_consumption_cooling_storage = 0
        elec_consumption_dhw_total = 0
        elec_consumption_cooling_total = 0
        elec_consumption_appliances = 0
        elec_generation = 0

        # Setting the current states using set_state() and also setting self.buildings
        set_state(states, total_it)


        assert len(actions) == self.num_buildings, #The length of the list of actions should match the length of the list of buildings."


        for a, (uid, building) in zip(actions, self.buildings.items()):


            if self.buildings_states_actions[uid]['actions']['electrical_storage']:


                if self.buildings_states_actions[uid]['actions']['cooling_storage']:

                    # Cooling
                    _electric_demand_cooling = self.building.set_storage_cooling(a[0], self.C_p_Csto, self.SOC_Csto, self.C_bd)
                    elec_consumption_cooling_storage += self.building._electric_consumption_cooling_storage

                    # 'Electrical Storage' & 'Cooling Storage' & 'DHW Storage'
                    if self.buildings_states_actions[uid]['actions']['dhw_storage']:

                        # DHW
                        _electric_demand_dhw = self.building.set_storage_heating(a[1], self.E_ehH_max, self.C_p_Hsto, self.SOC_Hsto, self.H_bd)
                        elec_consumption_dhw_storage += self.building._electric_consumption_dhw_storage

                        # Electrical
                        _electric_demand_electrical_storage = self.building.set_storage_electrical(a[2], self.C_p_bat, self.SOC_bat)
                        elec_consumption_electrical_storage += _electric_demand_electrical_storage

                            # 'Electrical Storage' & 'Cooling Storage'
                    else:
                        _electric_demand_dhw = self.building.set_storage_heating(0.0, self.E_ehH_max, self.C_p_Hsto, self.SOC_Hsto, self.H_bd)
                        # Electrical
                        _electric_demand_electrical_storage = self.building.set_storage_electrical(a[1], self.C_p_bat, self.SOC_bat)
                        elec_consumption_electrical_storage += _electric_demand_electrical_storage
                else:

                    _electric_demand_cooling = building.set_storage_cooling(0.0)
                        # 'Electrical Storage' & 'DHW Storage'
                    if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                        # DHW
                        _electric_demand_dhw = self.building.set_storage_heating(a[0], self.E_ehH_max, self.C_p_Hsto, self.SOC_Hsto, self.H_bd)
                        elec_consumption_dhw_storage += self.building._electric_consumption_dhw_storage

                        # Electrical
                        _electric_demand_electrical_storage = self.building.set_storage_electrical(a[1], self.C_p_bat, self.SOC_bat)
                        elec_consumption_electrical_storage += _electric_demand_electrical_storage

                    # 'Electrical Storage'
                    else:
                        _electric_demand_dhw = self.building.set_storage_heating(0.0)
                        # Electrical
                        _electric_demand_electrical_storage = self.building.set_storage_electrical(a[0], self.C_p_bat, self.SOC_bat)
                        elec_consumption_electrical_storage += _electric_demand_electrical_storage



            else:

                _electric_demand_electrical_storage = 0.0

                if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                    # Cooling
                    _electric_demand_cooling = self.building.set_storage_cooling(a[0], self.C_p_Csto, self.C_bd)
                    elec_consumption_cooling_storage += self.building._electric_consumption_cooling_storage

                    if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                        # DHW
                        _electric_demand_dhw = self.building.set_storage_heating(a[1], self.E_ehH_max, self.C_p_Hsto, self.SOC_Hsto, self.H_bd)
                        elec_consumption_dhw_storage += self.building._electric_consumption_dhw_storage

                    else:
                        _electric_demand_dhw = self.building.set_storage_heating(0.0, self.E_ehH_max, self.C_p_Hsto, SOC_Hsto, self.H_bd)

                else:
                    _electric_demand_cooling = self.building.set_storage_cooling(0.0, self.C_p_Csto, self.C_bd)
                    # DHW
                    _electric_demand_dhw = self.building.set_storage_heating(a[0], self.E_ehH_max, self.C_p_Hsto, self.SOC_Hsto, self.H_bd)
                    elec_consumption_dhw_storage += self.building._electric_consumption_dhw_storage




        # Total heating and cooling electrical loads
        elec_consumption_cooling_total += _electric_demand_cooling
        elec_consumption_dhw_total += _electric_demand_dhw

        # Electrical appliances
        _non_shiftable_load = E_NS
        elec_consumption_appliances += _non_shiftable_load

        # Solar generation
        _solar_generation = self.building.get_solar_power()
        elec_generation += _solar_generation

        # Adding loads from appliances and subtracting solar generation to the net electrical load of each building
        building_electric_demand = round(_electric_demand_electrical_storage + _electric_demand_cooling + _electric_demand_dhw + _non_shiftable_load - _solar_generation, 4)

        # Electricity consumed by every building
        self.building.current_net_electricity_demand = building_electric_demand
        self.buildings_net_electricity_demand.append(-building_electric_demand)

        # Total electricity consumption
        electric_demand += building_electric_demand


        self.state = []

        for uid, building in self.buildings.items():
                s = []
                for state_name, value in self.buildings_states_actions[uid]['states'].items():
                    if value == True:
                        if state_name == 'net_electricity_consumption':
                            s.append(self.building.current_net_electricity_demand)
                        elif (state_name != 'cooling_storage_soc') and (state_name != 'dhw_storage_soc') and (state_name != 'electrical_storage_soc'):
                            s.append(self.building.sim_results[state_name][self.time_step])
                        elif state_name == 'cooling_storage_soc':
                            s.append(self.building.SOC_Csto/C_p_Csto[uid])
                        elif state_name == 'dhw_storage_soc':
                            s.append(building.dhw_storage._soc/C_p_Hsto[uid])
                        elif state_name == 'electrical_storage_soc':
                            s.append(self.building.SOC_bat/C_p_bat[uid])

                self.state.append(np.array(s))

            self.state = np.array(self.state, dtype='object')


        # Control variables which are used to display the results and the behavior of the buildings at the district level.
        # self.carbon_emissions.append(np.float32(max(0, electric_demand)*self.current_carbon_intensity))
        self.net_electric_consumption.append(np.float32(electric_demand))
        self.electric_consumption_electric_storage.append(np.float32(elec_consumption_electrical_storage))
        self.electric_consumption_dhw_storage.append(np.float32(elec_consumption_dhw_storage))
        self.electric_consumption_cooling_storage.append(np.float32(elec_consumption_cooling_storage))
        self.electric_consumption_dhw.append(np.float32(elec_consumption_dhw_total))
        self.electric_consumption_cooling.append(np.float32(elec_consumption_cooling_total))
        self.electric_consumption_appliances.append(np.float32(elec_consumption_appliances))
        self.electric_generation.append(np.float32(elec_generation))
        self.net_electric_consumption_no_storage.append(np.float32(electric_demand-elec_consumption_cooling_storage-elec_consumption_dhw_storage-elec_consumption_electrical_storage))
        self.net_electric_consumption_no_pv_no_storage.append(np.float32(electric_demand + elec_generation - elec_consumption_cooling_storage - elec_consumption_dhw_storage-elec_consumption_electrical_storage))

        transition_digital_twin = [self._get_ob(), rewards, {}]   # self._get_ob() returns the next states

        return transition_digital_twin





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


    def cost(self):





#########################
##########################
############################
