# @Vanshaj + @Zhiyao : Implement in here
from copy import deepcopy
import numpy as np
from pathlib import Path
from Energy_Models_DigitalTwin import (
    Battery,
    HeatPump,
    ElectricHeater,
    EnergyStorage,
    Building,
)

# from citylearn import building_loader
# from energy_models import Battery, HeatPump, ElectricHeater, EnergyStorage, Building

import sys
import warnings
import utils
import time
import json

import numpy as np
import pandas as pd

if not sys.warnoptions:
    warnings.simplefilter("ignore")

## local imports
from predictor import *


class DigitalTwin:
    def __init__(
        self,
        building_ids,
        save_memory=True,
        buildings_states_actions=None,
        cost_function=[
            "ramping",
            "1-load_factor",
            "average_daily_peak",
            "peak_demand",
            "net_electricity_consumption",
        ],
        simulation_period=(0, 8759),
    ) -> None:

        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)

        # States that we can directly get from the observed states
        self.E_NS = np.zeros(9)
        self.net_electricity_consumption = np.zeros(9)
        self.SOC_Csto = np.zeros(9)
        self.SOC_Hsto = np.zeros(9)
        self.SOC_bat = np.zeros(9)
        self.solar_gen = np.zeros(9)

        # States that will require a predictor/oracle
        self.E_hpC_max = np.ones(9)*100
        self.E_ehH_max = np.ones(9)*100
        self.E_bat_max = np.ones(9)*100
        self.C_p_Csto = np.ones(9)*100
        self.C_p_Hsto = np.ones(9)*100
        self.C_p_bat = np.ones(9)*100
        self.eta_bat = np.ones(9)
        self.E_pv = np.zeros(9)
        self.H_bd = np.zeros(9)
        self.C_bd = np.zeros(9)
        self.COP_C = np.zeros(9)
        self.C_max = np.zeros(9)
        self.H_max = np.zeros(9)

        # Initialising the constant parameters
        self.eta_hp_tech: float = 0.22  # Technical Efficiency
        self.t_hp_C: int = 8  # Target temperature cooling
        self.eta_ehH: float = 0.9
        self.C_f_Hsto: float = 0.008
        self.C_f_Csto: float = 0.006
        self.C_f_bat: float = 1e-5
        self.num_buildings: int = 9

        self.buildings = {}
        self.buildings_states_actions_filename = buildings_states_actions
        self.buildings_net_electricity_demand = []
        self.cost_function = cost_function

        self.simulation_period = simulation_period
        self.uid = None
        self.num_buildings = 9
        self.save_memory = save_memory
        self.reset()

    def building_loader(self):
        """Loads the parameters for all the 9 buildings using the self.buidlng class"""
        # TODO: @Qasim - if this function is for initializing the digital twin, pls do as follows:
        # TODO: 1. import predictor
        # TODO: 2. call predictor.get_params(timestep), make sure timestep>=RBC_THRESHOLD
        # TODO: 3. the function returns a dictionary with "C_p_Csto", "C_p_Hsto", "C_p_bat", "E_bat_max"(nominal power)
        # TODO: 4. configure each parameter in data_dict below
        # TODO: 5. indexing: for example, to index nominal power for bdg 1, use E_bat_max[0]

        data_dict = {}
        for bid in range(9):
            data_dict[f'Building_{bid+1}'] = {
                "Heat_Pump": {
                    "nominal_power": self.E_hpC_max[bid],
                    "technical_efficiency": self.eta_hp_tech,
                    "t_target_heating": 45,
                    "t_target_cooling": self.t_hp_C,
                },
                "Electric_Water_Heater": {
                    "nominal_power": self.E_ehH_max[bid],
                    "efficiency": self.eta_ehH,
                },
                "Chilled_Water_Tank": {
                    "capacity": self.C_p_Csto[bid],
                    "loss_coefficient": self.C_f_Csto,
                },
                "DHW_Tank": {
                    "capacity": self.C_p_Hsto[bid],
                    "loss_coefficient": self.C_f_Csto,
                },
                "Battery": {
                    "capacity": self.C_p_bat[bid],
                    "efficiency": self.eta_bat[bid],
                    "capacity_loss_coefficient": self.C_f_bat,
                    "loss_coefficient": 0,
                    "nominal_power": self.E_bat_max[bid],
                    "power_efficiency_curve": [[0, 1], [1, 1]],
                    "capacity_power_curve": [[0, 1], [1, 0]],
                },
            }

        buildings = {}

        for uid, attributes in zip(data_dict, data_dict.values()):

            battery = Battery(
                capacity=attributes["Battery"]["capacity"],
                capacity_loss_coef=attributes["Battery"]["capacity_loss_coefficient"],
                loss_coef=attributes["Battery"]["loss_coefficient"],
                efficiency=attributes["Battery"]["efficiency"],
                nominal_power=attributes["Battery"]["nominal_power"],
                power_efficiency_curve=attributes["Battery"]["power_efficiency_curve"],
                capacity_power_curve=attributes["Battery"]["capacity_power_curve"],
                save_memory=self.save_memory,
            )

            heat_pump = HeatPump(
                nominal_power=attributes["Heat_Pump"]["nominal_power"],
                eta_tech=attributes["Heat_Pump"]["technical_efficiency"],
                t_target_heating=attributes["Heat_Pump"]["t_target_heating"],
                t_target_cooling=attributes["Heat_Pump"]["t_target_cooling"],
                save_memory=self.save_memory,
            )

            electric_heater = ElectricHeater(
                nominal_power=attributes["Electric_Water_Heater"]["nominal_power"],
                efficiency=attributes["Electric_Water_Heater"]["efficiency"],
                save_memory=self.save_memory,
            )

            chilled_water_tank = EnergyStorage(
                capacity=attributes["Chilled_Water_Tank"]["capacity"],
                loss_coef=attributes["Chilled_Water_Tank"]["loss_coefficient"],
                save_memory=self.save_memory,
            )

            dhw_tank = EnergyStorage(
                capacity=attributes["DHW_Tank"]["capacity"],
                loss_coef=attributes["DHW_Tank"]["loss_coefficient"],
                save_memory=self.save_memory,
            )

            building = Building(
                buildingId=uid,
                dhw_storage=dhw_tank,
                cooling_storage=chilled_water_tank,
                electrical_storage=battery,
                dhw_heating_device=electric_heater,
                cooling_device=heat_pump,
                save_memory=self.save_memory,
            )

            buildings[uid] = building

        self.buildings = buildings

    def set_state(self, states, total_it, memory, zeta):
        """Sets the current states to be passed to the transition function
        Also loads the buildings with the required parameters by calling
        buildings_load()"""

        # States that we can directly get from the observed states
        # Getting state for current time step and 9 buildings
        # self.E_NS = states[:, 23]
        self.net_electricity_consumption = states[:, 28]  # 9*1

        # self.solar_gen = states[:, 24]

        #         data_est = self.memory.get(-1)   # data from the predictor

        time_step = total_it % 24

        # Getting state for current time step and 9 buildings
        
        if time_step == 0:
        
            self.E_NS = memory["E_ns"][time_step, :]
            self.solar_gen = memory["E_pv"][time_step, :]
            self.E_hpC_max = memory["E_hpC_max"][time_step, :]
            self.E_ehH_max = memory["E_ehH_max"][time_step, :]
            self.E_bat_max = memory["E_bat_max"][time_step, :]
            self.C_p_Hsto = memory["C_p_Hsto"][time_step, :]
            self.C_p_Csto = memory["C_p_Csto"][time_step, :]
            self.C_p_bat = memory["C_p_bat"][time_step, :]
            self.eta_bat = zeta["eta_bat"][time_step, :]
            self.E_pv = memory["E_pv"][time_step, :]
            self.H_bd = memory["H_bd"][time_step, :]
            self.C_bd = memory["C_bd"][time_step, :]
            self.COP_C = memory["COP_C"][time_step, :]

        else:
            # Taking average of all the previous predictions to have a more robust estimate
            self.E_NS = np.mean(memory["E_ns"][:time_step, :], axis = 0)
            self.solar_gen = np.mean(memory["E_pv"][:time_step, :], axis = 0)
            self.E_hpC_max = np.mean(memory["E_hpC_max"][:time_step, :], axis = 0)
            self.E_ehH_max = np.mean(memory["E_ehH_max"][:time_step, :], axis = 0)
            self.E_bat_max = np.mean(memory["E_bat_max"][:time_step, :], axis = 0)
            self.C_p_Hsto = np.mean(memory["C_p_Hsto"][:time_step, :], axis = 0)
            self.C_p_Csto = np.mean(memory["C_p_Csto"][:time_step, :], axis = 0)
            self.C_p_bat = np.mean(memory["C_p_bat"][:time_step, :], axis = 0)
            self.eta_bat = zeta["eta_bat"][time_step, :]
            self.E_pv = np.mean(memory["E_pv"][:time_step, :], axis = 0)
            self.H_bd = np.mean(memory["H_bd"][:time_step, :], axis = 0)
            self.C_bd = np.mean(memory["C_bd"][:time_step, :], axis = 0)
            self.COP_C = np.mean(memory["COP_C"][:time_step, :], axis = 0)
            

        self.SOC_Csto = states[:, 25] *self.C_p_Csto
        self.SOC_Hsto = states[:, 26]*self.C_p_Hsto
        self.SOC_bat = states[:, 27]*self.C_p_bat

        # Load all the 9 buildings with current states
        self.building_loader()

    def transition(self, states, actions, total_it, memory: dict, zeta: dict):

        # Initialising the next states that we will get from the digital twin
        self.buildings_net_electricity_demand = []
        # electric_demand = 0
        # elec_consumption_electrical_storage = 0
        # elec_consumption_dhw_storage = 0
        # elec_consumption_cooling_storage = 0
        # elec_consumption_dhw_total = 0
        # elec_consumption_cooling_total = 0
        # elec_consumption_appliances = 0
        # elec_generation = 0

        # Setting the current states using set_state() and also setting self.buildings
        self.set_state(states, total_it, memory, zeta)

        assert (len(actions) == self.num_buildings)  # The length of the list of actions should match the length of the list of buildings."

        # Defininig dict to get access to the states from the building keys

        # dict_build = {
        #     "Building_1": 0,
        #     "Building_2": 1,
        #     "Building_3": 2,
        #     "Building_4": 3,
        #     "Building_5": 4,
        #     "Building_6": 5,
        #     "Building_7": 6,
        #     "Building_8": 7,
        #     "Building_9": 8,
        # }
        building_electric_demand = np.zeros(self.num_buildings)
        for bid in range(self.num_buildings):
            #a, (uid, building) in zip(actions, self.buildings.items()):
#             print(bid)
#             print(len(actions))
#             print(actions[0][bid])
#             print(np.shape(actions), total_it, type(actions))
            actions = np.array(actions)
            a = actions[bid,:]
            uid = f'Building_{bid+1}'
#             print(np.shape(self.E_hpC_max))
            _electric_demand_cooling = self.buildings[uid].set_storage_cooling(
                        a[0],
                        self.C_p_Csto[bid],
                        self.SOC_Csto[bid],
                        self.C_bd[bid],
                        self.COP_C[bid],
                        self.E_hpC_max[bid])
            # elec_consumption_cooling_storage += self.buildings[uid]._electric_consumption_cooling_storage


            # DHW
            _electric_demand_dhw = self.buildings[uid].set_storage_heating(
                            a[1],
                            self.E_ehH_max[bid],
                            self.C_p_Hsto[bid],
                            self.SOC_Hsto[bid],
                            self.H_bd[bid])
            # elec_consumption_dhw_storage += building._electric_consumption_dhw_storage


            # Electrical
            _electric_demand_electrical_storage = self.buildings[uid].set_storage_electrical(
                                a[2],
                                self.C_p_bat[bid],
                                self.SOC_bat[bid],)
            # elec_consumption_electrical_storage += _electric_demand_electrical_storage


            # Total heating and cooling electrical loads
            # elec_consumption_cooling_total += _electric_demand_cooling
            # elec_consumption_dhw_total += _electric_demand_dhw

            # Solar generation
            _solar_generation = self.buildings[uid].get_solar_power(self.solar_gen[bid])
            # elec_generation += _solar_generation

            # Electrical appliances
            _non_shiftable_load = self.E_NS[bid]

            building_electric_demand[bid] = _electric_demand_electrical_storage+ _electric_demand_cooling+ \
                                            _electric_demand_dhw+ _non_shiftable_load- _solar_generation





        # Adding loads from appliances and subtracting solar generation to the net electrical load of each building
        # building_electric_demand = np.round(
        #     (
        #         _electric_demand_electrical_storage
        #         + _electric_demand_cooling
        #         + _electric_demand_dhw
        #         + _non_shiftable_load
        #         - _solar_generation
        #     ).astype(np.float32),
        #     4,
        # )
        # Electricity consumed by every building
        # building.current_net_electricity_demand = building_electric_demand
        # self.buildings_net_electricity_demand.append(-building_electric_demand)

        # Total electricity consumption
        # electric_demand += building_electric_demand

        # self.state = []

        # self.net_electric_consumption.append(np.float32(electric_demand))
        # self.electric_consumption_electric_storage.append(
        #     np.float32(elec_consumption_electrical_storage)
        # )
        # self.electric_consumption_dhw_storage.append(
        #     np.float32(elec_consumption_dhw_storage)
        # )
        # self.electric_consumption_cooling_storage.append(
        #     np.float32(elec_consumption_cooling_storage)
        # )
        # self.electric_consumption_dhw.append(np.float32(elec_consumption_dhw_total))
        # self.electric_consumption_cooling.append(
        #     np.float32(elec_consumption_cooling_total)
        # )
        # self.electric_consumption_appliances.append(
        #     np.float32(elec_consumption_appliances)
        # )
        # self.electric_generation.append(np.float32(elec_generation))
        # self.net_electric_consumption_no_storage.append(
        #     np.float32(
        #         electric_demand
        #         - elec_consumption_cooling_storage
        #         - elec_consumption_dhw_storage
        #         - elec_consumption_electrical_storage
        #     )
        # )
        # self.net_electric_consumption_no_pv_no_storage.append(
        #     np.float32(
        #         electric_demand
        #         + elec_generation
        #         - elec_consumption_cooling_storage
        #         - elec_consumption_dhw_storage
        #         - elec_consumption_electrical_storage
        #     )
        # )

        # transition_digital_twin = (
        #     building_electric_demand
        # )  # self._get_ob() returns the next states
        # next_state_net_electricity_consumption = building_electric_demand.reshape((9, 1))

        next_state = np.ones((9, 30))
        next_state[:, 28] = building_electric_demand

        for bid in range(9):
            uid = f'Building_{bid+1}'
            next_state[bid, 25] = self.buildings[uid].cooling_storage._soc / self.buildings[uid].cooling_storage.capacity if self.buildings[uid].cooling_storage.capacity>0 else 0
            next_state[bid, 26] = self.buildings[uid].dhw_storage._soc / self.buildings[uid].dhw_storage.capacity if self.buildings[uid].dhw_storage.capacity>0 else 0
            next_state[bid, 27] = self.buildings[uid].electrical_storage._soc/self.buildings[uid].electrical_storage.capacity if self.buildings[uid].electrical_storage.capacity>0 else 0


        return next_state

    # def _get_ob(self):
    #     return self.state

    def reset(self):

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

        self.state = []
        for uid, building in self.buildings.items():
            building.reset()
            s = []
            for state_name, value in zip(
                self.buildings_states_actions[uid]["states"],
                self.buildings_states_actions[uid]["states"].values(),
            ):
                if value == True:
                    if state_name == "net_electricity_consumption":
                        s.append(building.current_net_electricity_demand)
                    elif (
                        (state_name != "cooling_storage_soc")
                        and (state_name != "dhw_storage_soc")
                        and (state_name != "electrical_storage_soc")
                    ):
                        s.append(0.0)
                    elif state_name == "cooling_storage_soc":
                        s.append(0.0)
                    elif state_name == "dhw_storage_soc":
                        s.append(0.0)
                    elif state_name == "electrical_storage_soc":
                        s.append(0.0)

            self.state.append(np.array(s, dtype=np.float32))

        self.state = np.array(self.state, dtype="object")

        return self.state

    # def _get_ob(self):
    #     return self.state
