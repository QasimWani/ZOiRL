from utils import ReplayBuffer, DataLoader
from collections import defaultdict
from citylearn import CityLearn
from copy import deepcopy

import numpy as np
import pandas as pd
import sys


class Oracle(DataLoader):
    """Agent with access to true environment data."""

    def __init__(self, env: CityLearn, action_space: list) -> None:
        self.action_space = action_space
        self.weather_data = self.get_weather_data(env)

    def upload_data(
        self,
        replay_buffer: ReplayBuffer,
        E_grid: list,
        actions: list,
        rewards: list,
        env: CityLearn = None,
        t_idx: int = -1,
    ):
        """Returns state information to be later added to replay buffer"""
        assert (
            env is not None and t_idx >= 0
        ), "Invalid argument passed. Missing env object and/or invalid time index passed"

        ## load current data and pass it as an argument to parse_data where data needs to be a dictionary.
        if t_idx == 2:
            E_grid_memory = [0] * len(self.action_space)
        else:
            # last hour or eod E_grid values.
            E_grid_memory = replay_buffer.get(-1)["E_grid"][-1]

        data = self.parse_data(
            replay_buffer.get_recent(),
            self.get_current_data_oracle(
                env, t_idx, E_grid, E_grid_memory, actions, rewards
            ),
        )
        replay_buffer.add(data)

    def load_data(self, state, t):
        raise NotImplementedError("Functionality not implemented")

    def get_weather_data(self, env):
        """load weather data for calculation of COP"""

        with open(env.data_path / env.weather_file) as csv_file:
            weather_data = pd.read_csv(csv_file)
        weather_data = weather_data["Outdoor Drybulb Temperature [C]"]
        return weather_data

    def parse_data(self, data: dict, current_data: dict) -> list:
        """Parses `current_data` for optimization and loads into `data`"""
        assert (
            len(current_data) == 23  # actions + rewards + E_grid_collect. Section 1.3.1
        ), f"Invalid number of parameters, found: {len(current_data)}, expected: 23. Can't run Oracle agent optimization."

        for key, value in current_data.items():
            if key not in data:
                data[key] = []
            data[key].append(value)

        return data

    def convert_to_numpy(self, params: dict):
        """Converts dic[key] to nd.array"""
        for key in params:
            if key == "c_bat_init" or key == "c_Csto_init" or key == "c_Hsto_init":
                params[key] = np.array(params[key][0])
            else:
                params[key] = np.array(params[key])

    def get_dimensions(self, data: dict):
        """Prints shape of each param"""
        for key in data.keys():
            print(data[key].shape)

    def get_building(self, data: dict, building_id: int):
        """Loads data (dict) from a particular building. 1-based indexing for building"""
        assert building_id > 0, "building_id is 1-based indexing."
        building_data = {}
        for key in data.keys():
            building_data[key] = np.array(data[key])[:, building_id - 1]
        return building_data

    def create_random_data(self, data: dict):
        """Synthetic data (Gaussian) generation"""
        for key in data:
            data[key] = np.clip(np.random.random(size=data[key].shape), 0, 1)
        return data

    def get_current_data_oracle(
        self,
        env: CityLearn,
        t: int,  # t goes from 0 - end of simulation (not 24 hour counter!)
        E_grid: list,
        E_grid_memory: list,
        actions: list = None,
        rewards: list = None,
    ):
        """
        Returns data (dict) for each building from `env` for `t` daystep
        @Params:
        1. env: CityLearn environment.
        2. t: current hour from TD3.py (agents.total_it). Goes from (0 - 4 * 365 * 24]
        3. E_grid: Current net electricity consumption, obtained from environment.
        4. E_grid_memory: Previous hours' net electricity consumption. Obtained from memory (replaybuffer).
        5. actions: set of actions for current hour. Appends to replaybuffer.
        6. rewards: per building reward. Appends to replaybuffer.
        """
        ### FB - Full batch. Trim output X[:time-step]
        ### CT - current daystep only. X = full_data[time-step], no access to full_data
        ### DP - dynamic update. time-step k = [... k], time-step k+n = [... k + n].
        ### P - constant value across all time steps. changes per building only.

        _num_buildings = len(self.action_space)  # total number of buildings in env.
        observation_data = {}

        # Loads
        E_ns = [
            env.buildings["Building_" + str(i)].sim_results["non_shiftable_load"][t]
            for i in range(1, _num_buildings + 1)
        ]  # CT
        H_bd = [
            env.buildings["Building_" + str(i)].sim_results["dhw_demand"][t]
            for i in range(1, _num_buildings + 1)
        ]  # DP
        C_bd = [
            env.buildings["Building_" + str(i)].sim_results["cooling_demand"][t]
            for i in range(1, _num_buildings + 1)
        ]  # DP
        H_max = np.max(
            [
                env.buildings["Building_" + str(i)].sim_results["dhw_demand"]
                for i in range(1, _num_buildings + 1)
            ],
            axis=1,
        )  # DP
        C_max = np.max(
            [
                env.buildings["Building_" + str(i)].sim_results["cooling_demand"]
                for i in range(1, _num_buildings + 1)
            ],
            axis=1,
        )  # DP

        # PV generations
        E_pv = [
            env.buildings["Building_" + str(i)].sim_results["solar_gen"][t]
            for i in range(1, _num_buildings + 1)
        ]  # CT

        # Heat Pump
        eta_hp = [0.22] * _num_buildings  # P
        t_C_hp = [
            8
        ] * _num_buildings  # P target cooling temperature (universal constant)

        COP_C = [None for i in range(_num_buildings)]  # DP

        E_hpC_max = [None] * _num_buildings
        for i in range(1, _num_buildings + 1):
            COP_C_t = (
                eta_hp[i - 1]
                * float(t_C_hp[i - 1] + 273.15)
                / (self.weather_data - t_C_hp[i - 1])
            )
            COP_C_t[COP_C_t < 0] = 20.0
            COP_C_t[COP_C_t > 20] = 20.0
            COP_C_t = COP_C_t.to_numpy()
            COP_C[i - 1] = COP_C_t[t]
            E_hpC_max[i - 1] = np.max(
                env.buildings["Building_" + str(i)].sim_results["cooling_demand"]
                / COP_C_t
            )

        # Electric Heater
        # replaced capacity (not avaiable in electric heater) w/ nominal_power
        E_ehH_max = [H_max[i] / 0.9 for i in range(_num_buildings)]  # P

        # Battery
        C_f_bat = [0.00001 for i in range(_num_buildings)]  # P
        C_p_bat = [60] * _num_buildings  # P (range: [20, 200])
        # current hour soc. normalized
        c_bat_init = [
            None
        ] * _num_buildings  # can't get future data since action dependent
        for i in range(1, _num_buildings + 1):
            building = env.buildings["Building_" + str(i)].electrical_storage
            try:
                c_bat_init[i - 1] = building.soc[t - 1] / building.capacity
            except:
                c_bat_init[i - 1] = 0
        # Heat (Energy->dhw) Storage
        C_f_Hsto = [0.008] * _num_buildings  # P
        C_p_Hsto = [3 * H_max[i] for i in range(_num_buildings)]  # P
        # current hour soc. normalized
        c_Hsto_init = [
            None
        ] * _num_buildings  # can't get future data since action dependent
        for i in range(1, _num_buildings + 1):
            building = env.buildings["Building_" + str(i)].dhw_storage
            try:
                c_Hsto_init[i - 1] = building.soc[t - 1] / building.capacity
            except:
                c_Hsto_init[i - 1] = 0
            # Cooling (Energy->cooling) Storage
        C_f_Csto = [0.006] * _num_buildings  # P
        C_p_Csto = [2 * C_max[i] for i in range(_num_buildings)]  # P

        # current hour soc. normalized
        c_Csto_init = [
            None
        ] * _num_buildings  # can't get future data since action dependent
        for i in range(1, _num_buildings + 1):
            building = env.buildings["Building_" + str(i)].cooling_storage
            try:
                c_Csto_init[i - 1] = building.soc[t - 1] / building.capacity
            except:
                c_Csto_init[i - 1] = 0

        # add actions - size 9 for each action
        action_H, action_C, action_bat = (
            [None] * 3 if actions is None else zip(*actions)
        )

        # fill data
        # add E-grid (part of E-grid_collect)
        observation_data["E_grid"] = (
            E_grid if E_grid is not None else [0] * _num_buildings
        )
        observation_data["E_grid_prevhour"] = E_grid_memory

        observation_data["E_ns"] = E_ns
        observation_data["H_bd"] = H_bd
        observation_data["C_bd"] = C_bd
        observation_data["H_max"] = H_max
        observation_data["C_max"] = C_max

        observation_data["E_pv"] = E_pv

        observation_data["E_hpC_max"] = E_hpC_max
        observation_data["E_ehH_max"] = E_ehH_max
        # observation_data["eta_hp"] = eta_hp # NOT NEEEDED!
        # observation_data["t_C_hp"] = t_C_hp # NOT NEEDED!
        observation_data["COP_C"] = COP_C

        # observation_data["C_f_bat"] = C_f_bat
        observation_data["C_p_bat"] = C_p_bat
        observation_data["c_bat_init"] = c_bat_init

        # observation_data["C_f_Hsto"] = C_f_Hsto
        observation_data["C_p_Hsto"] = C_p_Hsto
        observation_data["c_Hsto_init"] = c_Hsto_init

        # observation_data["C_f_Csto"] = C_f_Csto
        observation_data["C_p_Csto"] = C_p_Csto
        observation_data["c_Csto_init"] = c_Csto_init

        observation_data["action_H"] = action_H
        observation_data["action_C"] = action_C
        observation_data["action_bat"] = action_bat

        # add reward \in R^9 (scalar value for each building)
        observation_data["reward"] = rewards

        return observation_data

    def estimate_data(
        self,
        surrogate_env: CityLearn,
        data: dict,
        t_start: int,
        init_updates: dict,
        replay_buffer: ReplayBuffer,
        t_end: int = 24,
    ):
        """Returns data for hours `t_start` - 24 using `surrogate_env` running RBC `agent`"""
        for i in range(t_start % 24, t_end):
            data = self.parse_data(
                data,
                self.get_current_data_oracle(
                    surrogate_env,
                    t_start + i,
                    E_grid=None,
                    E_grid_memory=np.array(
                        [0] * len(self.action_space)
                    ),  # replay_buffer.get(-2)["E_grid"][(t_start + i) % 24],  # -2 : previous day E_grid values
                ),
            )

        return (
            self.init_values(data, init_updates)[0] if t_start % 24 == 0 else data
        )  # only load previous values at start of day

    def init_values(self, data: dict, update_values: dict = None):
        """Loads eod values for SOC and E_grid_past before(after) wiping data cache"""
        if update_values:
            # assign previous day's end socs.
            data["c_bat_init"][0] = update_values["c_bat_init"]
            data["c_Hsto_init"][0] = update_values["c_Hsto_init"]
            data["c_Csto_init"][0] = update_values["c_Csto_init"]

            # assign previous day's end E_grid.
            # data["E_grid_true"][0] = update_values["E_grid_true"]
        else:
            update_values = {
                "c_bat_init": data["c_bat_init"][-1],
                "c_Hsto_init": data["c_Hsto_init"][-1],
                "c_Csto_init": data["c_Csto_init"][-1],
                # "E_grid_true": data["E_grid_true"][-1],
            }

        return data, update_values
