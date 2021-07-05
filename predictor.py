from utils import ReplayBuffer

from citylearn import CityLearn

import numpy as np
import pandas as pd


class DataLoader:
    """Main Class for loading and uplaoding data to buffer."""

    def __init__(
        self,
        is_oracle: bool,
        action_space: list,
        env: CityLearn = None,
    ) -> None:

        self.env = env
        if is_oracle:
            self.model = Oracle(env, action_space)
        else:
            self.model = Predictor(...)

        self.data = {} if is_oracle else None

    def upload_data(
        self,
        replay_buffer: ReplayBuffer,
        action: list,
        reward: list,
        E_grid: list,
        env: CityLearn = None,
        t_idx: int = -1,  # timestep (hour) of the simulation [0 - (4years-1)]
    ):
        """Upload to memory"""
        self.model.upload_data(replay_buffer, action, reward, E_grid, env, t_idx)

    def load_data(self):
        """Sample from Memory. NOTE: Optional"""
        self.model.load_data()


class Predictor:
    """@Zhiyao + @Mingyu - estimates parameters, loads data supplied to `Actor`"""

    def __init__(self):
        raise NotImplementedError("Predictor class not implemented")

    def upload_data(self, replay_buffer: ReplayBuffer):
        raise NotImplementedError("Functionality not implemented")

    def load_data(self, state, t):
        raise NotImplementedError("Functionality not implemented")

    # TODO: implement other methods here. Make sure `DataLoader.upload_data()` and `DataLoader.load_data()` are processed correctly
    def parse_data(self, data: dict, current_data: dict) -> list:
        """Parses `current_data` for optimization and loads into `data`"""
        assert (
            len(current_data) == 30  # includes actions + rewards + E_grid_collect
        ), "Invalid number of parameters. Can't run basic (root) agent optimization"

        for key, value in current_data.items():
            if key not in data:
                data[key] = []  # [] x, 9 1, 9 -> x + 1, 9
            data[key].append(value)

        return data

    def convert_to_numpy(self, params: dict):
        """Converts dic[key] to nd.array"""
        for key in params:
            if key == "c_bat_init" or key == "c_Csto_init" or key == "c_Hsto_init":
                params[key] = np.array(params[key][0])
            else:
                params[key] = np.array(params[key])


class Oracle:
    """Agent with access to true environment data."""

    def __init__(self, env: CityLearn, action_space: list) -> None:
        self.action_space = action_space
        self.weather_data = self.get_weather_data(env)

    def upload_data(
        self,
        replay_buffer: ReplayBuffer,
        actions: list,
        rewards: list,
        E_grid: list,
        env: CityLearn = None,
        t_idx: int = -1,
    ):
        """Returns state information to be later added to replay buffer"""
        assert (
            env is not None and t_idx >= 0
        ), "Invalid argument passed. Missing env object and/or invalid time index passed"

        ## load current data and pass it as an argument to parse_data where data needs to be a dictionary.
        data = self.parse_data(
            replay_buffer.get_recent(),
            self.get_current_data_oracle(env, t_idx, actions, rewards, E_grid),
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
            len(current_data) == 33  # includes actions + rewards + E_grid_collect
        ), "Invalid number of parameters. Can't run basic (root) agent optimization"

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
        actions: list = None,
        rewards: list = None,
        E_grid: list = None,
    ):
        """Returns data (dict) for each building from `env` for `t` timestep"""
        ### FB - Full batch. Trim output X[:time-step]
        ### CT - current timestep only. X = full_data[time-step], no access to full_data
        ### DP - dynamic update. time-step k = [... k], time-step k+n = [... k + n].
        ### P - constant value across all time steps. changes per building only.

        _num_buildings = len(self.action_space)  # total number of buildings in env.
        observation_data = {}

        # p_ele = [
        #     2 if 10 <= t % 24 <= 20 else 0.2 for i in range(1, _num_buildings + 1)
        # ]  # FB -- virtual electricity price.
        p_ele = [1] * _num_buildings  # FB -- virtual electricity price.
        # can't get future data since action dependent
        E_grid_past = [
            0 if E_grid is None else E_grid[i, max(t % 24 - 1, 0)]
            for i in range(_num_buildings)
        ]  # FB -- replace w/ per building cost
        ramping_cost_coeff = [
            0.1 for i in range(_num_buildings)
        ]  # P  -- initialized to 0.1, learned through diff.

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
        E_max = np.max(
            [
                env.buildings["Building_" + str(i)].sim_results["non_shiftable_load"]
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
        eta_ehH = [0.9] * _num_buildings  # P
        # replaced capacity (not avaiable in electric heater) w/ nominal_power
        E_ehH_max = [H_max[i] / eta_ehH[i] for i in range(_num_buildings)]  # P

        # Battery
        C_f_bat = [0.0000 for i in range(_num_buildings)]  # P
        C_p_bat = [60] * _num_buildings  # P (range: [20, 200])
        eta_bat = [1] * _num_buildings  # P
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
        C_f_Hsto = [0.00] * _num_buildings  # P
        C_p_Hsto = [3 * H_max[i] for i in range(_num_buildings)]  # P
        eta_Hsto = [1] * _num_buildings  # P
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
        C_f_Csto = [0.00] * _num_buildings  # P
        C_p_Csto = [2 * C_max[i] for i in range(_num_buildings)]  # P
        eta_Csto = [1] * _num_buildings  # P
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
        observation_data["p_ele"] = p_ele
        observation_data["ramping_cost_coeff"] = ramping_cost_coeff
        observation_data["E_grid_past"] = E_grid_past
        observation_data["E_grid"] = (
            E_grid[:, t % 24] if E_grid is not None else [0] * _num_buildings
        )  # add E-grid (part of E-grid_collect)

        observation_data["E_ns"] = E_ns
        observation_data["H_bd"] = H_bd
        observation_data["C_bd"] = C_bd
        observation_data["H_max"] = H_max
        observation_data["C_max"] = C_max

        observation_data["E_pv"] = E_pv

        observation_data["E_hpC_max"] = E_hpC_max
        observation_data["eta_hp"] = eta_hp
        observation_data["t_C_hp"] = t_C_hp
        observation_data["COP_C"] = COP_C

        observation_data["eta_ehH"] = eta_ehH
        observation_data["E_ehH_max"] = E_ehH_max

        observation_data["C_f_bat"] = C_f_bat
        observation_data["C_p_bat"] = C_p_bat
        observation_data["eta_bat"] = eta_bat
        observation_data["c_bat_init"] = c_bat_init
        observation_data["c_bat_end"] = [0.1] * _num_buildings

        observation_data["C_f_Hsto"] = C_f_Hsto
        observation_data["C_p_Hsto"] = C_p_Hsto
        observation_data["eta_Hsto"] = eta_Hsto
        observation_data["c_Hsto_init"] = c_Hsto_init

        observation_data["C_f_Csto"] = C_f_Csto
        observation_data["C_p_Csto"] = C_p_Csto
        observation_data["eta_Csto"] = eta_Csto
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
        t_end: int = 24,
    ):
        """Returns data for hours `t_start` - 24 using `surrogate_env` running RBC `agent`"""
        for i in range(t_start % 24, t_end):
            data = self.parse_data(
                data, self.get_current_data_oracle(surrogate_env, t_start + i)
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
            data["E_grid_past"][0] = update_values["E_grid_past"]
        else:
            update_values = {
                "c_bat_init": data["c_bat_init"][-1],
                "c_Hsto_init": data["c_Hsto_init"][-1],
                "c_Csto_init": data["c_Csto_init"][-1],
                "E_grid_past": data["E_grid_past"][-1],
            }

        return data, update_values
