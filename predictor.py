from critic import Critic
from actor import Actor

from citylearn import CityLearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Predictor():
	""" @Zhiyao + @Mingyu - estimates parameters, loads data supplied to `Actor` """
	def __init__(...):
		pass

	### @Zhiyao: See https://github.com/QasimWani/ROLEVT/blob/main/colab_implicit_agent.ipynb for functionality on loader.
	def predict_data(self, state, ...):
		pass

	### @Qasim -- original full-agent
	def full_agent_data(self, env, t):
		pass


def weather_data(env: CityLearn) -> np.array:
    ### load weather data for calculation of COP

    with open(env.data_path / env.weather_file) as csv_file:
        weather_data = pd.read_csv(csv_file)
    weather_data = weather_data["Outdoor Drybulb Temperature [C]"]
    return weather_data


def parse_data(data: dict, current_data: dict) -> list:
    """Parses `current_data` for optimization and loads into `data`"""
    assert (
        len(current_data) == 28
    ), "Invalid number of parameters. Can't run basic (root) agent optimization"

    for key, value in current_data.items():
        if key not in data:
            data[key] = []
        data[key].append(value)

    # for key, value in current_data.items():
    #         if np.array(data[key]).shape == (1, 9):  # removes duplicates
    #                 data[key] = [value]
    return data


def get_dimensions(data: dict):
    """Gets shape of each param"""
    for key in data.keys():
        print(data[key].shape)


def get_building(data: dict, building_id: int):
    """Loads data (dict) from a particular building. 1-based indexing for building"""
    assert building_id > 0, "building_id is 1-based indexing."
    building_data = {}
    for key in data.keys():
        building_data[key] = np.array(data[key])[:, building_id - 1]
    return building_data


def convert_to_numpy(params: dict):
    """Converts dic[key] to nd.array"""
    for key in params:
        if key == "c_bat_init" or key == "c_Csto_init" or key == "c_Hsto_init":
            params[key] = np.array(params[key][0])
        else:
            params[key] = np.array(params[key])


def create_random_data(data: dict):
    """Creates random Gaussian data"""
    for key in data:
        data[key] = np.clip(np.random.random(size=data[key].shape), 0, 1)
    return data


def get_actions(
    data: dict,
    Optim: Actor,
    t,
    debug=False,
    apply_seed=False,
    lookahead=False,
    num_buildings=9,
) -> list:
    """
    Runs `Optim` for all 9 buildings per hour.
    `lookahead` once set to `True` computes day-ahead dispatch
    """
    convert_to_numpy(data)  # modify in-place
    data = create_random_data(deepcopy(data)) if apply_seed else data
    if debug:
        return [
            Optim(t, data, i, actions_spaces[i].shape[0]) for i in range(num_buildings)
        ]

    return [
        Optim(t, data, i, actions_spaces[i].shape[0]).solve(debug, lookahead)
        for i in range(9)
    ]


def get_current_data_oracle(env: CityLearn, t: int):
    """Returns data:dic for each building from `env` for `t` timestep"""
    ### FB - Full batch. Trim output X[:time-step]
    ### CT - current timestep only. X = full_data[time-step], no access to full_data
    ### DP - dynamic update. time-step k = [... k], time-step k+n = [... k + n].
    ### P - constant value across all time steps. changes per building only.

    _max_load = 2 * 168  # 2-week max load
    _num_buildings = len(actions_spaces)  # total number of buildings in env.
    _start = max(t - _max_load, 0)
    observation_data = {}

    p_ele = [
        2 if 10 <= t % 24 <= 20 else 0.2 for i in range(1, _num_buildings + 1)
    ]  # FB -- virtual electricity price.
    # can't get future data since action dependent
    E_grid_past = [
        0 for i in range(1, _num_buildings + 1)
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
    t_C_hp = [8] * _num_buildings  # P target cooling temperature (universal constant)

    COP_C = [None for i in range(_num_buildings)]  # DP

    E_hpC_max = [None] * _num_buildings
    for i in range(1, _num_buildings + 1):
        COP_C_t = (
            eta_hp[i - 1]
            * float(t_C_hp[i - 1] + 273.15)
            / (weather_data - t_C_hp[i - 1])
        )
        COP_C_t[COP_C_t < 0] = 20.0
        COP_C_t[COP_C_t > 20] = 20.0
        COP_C_t = COP_C_t.to_numpy()
        COP_C[i - 1] = COP_C_t[t]
        E_hpC_max[i - 1] = np.max(
            env.buildings["Building_" + str(i)].sim_results["cooling_demand"] / COP_C_t
        )
        # max_soc = np.max(env.buildings['Building_'+str(i)].cooling_storage.soc[:t])
        # E_hpC_max[i - 1] = max(E_hpC_max[i - 1], max_soc)

        # except ValueError:
        #     pass

    # Electric Heater
    eta_ehH = [0.9] * _num_buildings  # P
    # replaced capacity (not avaiable in electric heater) w/ nominal_power
    E_ehH_max = [H_max[i] / eta_ehH[i] for i in range(_num_buildings)]  # P

    # Battery
    C_f_bat = [0.0000 for i in range(_num_buildings)]  # P
    C_p_bat = [60] * _num_buildings  # P (range: [20, 200])
    eta_bat = [1] * _num_buildings  # P
    # current hour soc. normalized
    c_bat_init = [None] * _num_buildings  # can't get future data since action dependent
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

    ### TODO add alphas for critic params

    # fill data
    observation_data["p_ele"] = p_ele
    observation_data["ramping_cost_coeff"] = ramping_cost_coeff
    observation_data["E_grid_past"] = E_grid_past # @jinming - if E_grid from actor optim. or environment?

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

    return observation_data