# To run this example, move this file to the main directory of this repository
from citylearn import  CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import cvxpy as cp
import json
import time
from agents.rbc import RBC
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

# Select the climate zone and load environment
climate_zone = 1
sim_period = (0, 8760*4-1)
params = {'data_path':Path("data/Climate_Zone_5_Optim"),
        'building_attributes':'building_attributes.json',
        'weather_file':'weather_data.csv',
        'solar_profile':'solar_generation_1kW.csv',
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':["Building_"+str(i) for i in [1,2,3,4,5,6,7,8,9]],
        'buildings_states_actions':'buildings_state_action_space.json',
        'simulation_period': sim_period,
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'],
        'central_agent': False,
        'save_memory': False }

env = CityLearn(**params)

observations_spaces, actions_spaces = env.get_state_action_spaces()

### load weather data for calculation of COP
with open(env.data_path / env.weather_file) as csv_file:
    weather_data = pd.read_csv(csv_file)
weather_data = weather_data['Outdoor Drybulb Temperature [C]']


def parse_data(data: dict, current_data: dict):
        """ Parses `current_data` for optimization and loads into `data` """
        assert len(current_data) == 28, "Invalid number of parameters. Can't run basic (root) agent optimization"

        for key, value in current_data.items():
                if key not in data:
                        data[key] = []
                data[key].append(value)

        # for key, value in current_data.items():
        #         if np.array(data[key]).shape == (1, 9):  # removes duplicates
        #                 data[key] = [value]
        return data

def get_dimensions(data:dict):
    """ Gets shape of each param """
    for key in data.keys():
        print(data[key].shape)

def get_building(data: dict, building_id: int):
        """ Loads data (dict) from a particular building. 1-based indexing for building """
        assert building_id > 0, "building_id is 1-based indexing."
        building_data = {}
        for key in data.keys():
                building_data[key] = np.array(data[key])[:, building_id - 1]
        return building_data


def convert_to_numpy(params:dict):
    """ Converts dic[key] to nd.array """
    for key in params:
        if key == 'c_bat_init' or key == 'c_Csto_init' or key == 'c_Hsto_init':
            params[key] = np.array(params[key][0])
        else:
            params[key] = np.array(params[key])

def create_random_data(data:dict):
    """ Creates random data drawn from Gaussian. """
    for key in data:
        data[key] = np.clip(np.random.random(size=data[key].shape), 0, 1)
    return data

def get_actions(data:dict, t, debug=False, apply_seed=False, lookahead=False):
    """ Runs Optim for all 9 buildings per hour. `lookahead` once set to `True` computes day-ahead dispatch """
    convert_to_numpy(data)
    data = create_random_data(deepcopy(data)) if apply_seed else data
    if debug:
        return [Optim(t, data, i, actions_spaces[i].shape[0]) for i in range(9)]

    return [Optim(t, data, i, actions_spaces[i].shape[0]).solve(debug, lookahead) for i in range(9)]


def get_current_data_oracle(env, t):
        """ Returns data:dic for each building from `env` for `t` timestep """
        ### FB - Full batch. Trim output X[:time-step]
        ### CT - current timestep only. X = full_data[time-step], no access to full_data
        ### DP - dynamic update. time-step k = [... k], time-step k+n = [... k + n].
        ### P - constant value across all time steps. changes per building only.

        _max_load = 2 * 168  # 2-week max load
        _num_buildings = len(actions_spaces)  # total number of buildings in env.
        _start = max(t - _max_load, 0)
        observation_data = {}

        p_ele = [2 if 10 <= t % 24 <= 20 else 0.2 for i in
                 range(1, _num_buildings + 1)]  # FB -- virtual electricity price.
        # can't get future data since action dependent
        E_grid_past = [0 for i in range(1, _num_buildings + 1)]  # FB -- replace w/ per building cost
        ramping_cost_coeff = [0.1 for i in range(_num_buildings)]  # P  -- initialized to 0.1, learned through diff.

        # Loads
        E_ns = [env.buildings['Building_' + str(i)].sim_results['non_shiftable_load'][t] for i in
                range(1, _num_buildings + 1)]  # CT
        H_bd = [env.buildings['Building_' + str(i)].sim_results['dhw_demand'][t] for i in
                range(1, _num_buildings + 1)]  # DP
        C_bd = [env.buildings['Building_' + str(i)].sim_results['cooling_demand'][t] for i in
                range(1, _num_buildings + 1)]  # DP
        H_max = np.max(
                [env.buildings['Building_' + str(i)].sim_results['dhw_demand'] for i in range(1, _num_buildings + 1)],
                axis=1)  # DP
        C_max = np.max([env.buildings['Building_' + str(i)].sim_results['cooling_demand'] for i in
                        range(1, _num_buildings + 1)], axis=1)  # DP
        E_max = np.max([env.buildings['Building_' + str(i)].sim_results['non_shiftable_load'] for i in
                        range(1, _num_buildings + 1)], axis=1)  # DP
        # PV generations
        E_pv = [env.buildings['Building_' + str(i)].sim_results['solar_gen'][t] for i in
                range(1, _num_buildings + 1)]  # CT

        # Heat Pump
        eta_hp = [0.22] * _num_buildings  # P
        t_C_hp = [8] * _num_buildings  # P target cooling temperature (universal constant)

        #### CityLearn.py def. doesn't make too much sense. !!!! NOTE: THIS IS NOT UDPATED to CITYLEARN !!!!
        COP_C = [None for i in range(_num_buildings)]  # DP

        E_hpC_max = [None] * _num_buildings
        for i in range(1, _num_buildings + 1):
                COP_C_t = eta_hp[i - 1] * float(t_C_hp[i - 1] + 273.15) / (weather_data - t_C_hp[i - 1])
                COP_C_t[COP_C_t < 0] = 20.0
                COP_C_t[COP_C_t > 20] = 20.0
                COP_C_t = COP_C_t.to_numpy()
                COP_C[i - 1] = COP_C_t[t]
                E_hpC_max[i - 1] = np.max(
                        env.buildings['Building_' + str(i)].sim_results['cooling_demand'] / COP_C_t)
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
                building = env.buildings['Building_' + str(i)].electrical_storage
                try:
                        c_bat_init[i - 1] = building.soc[t-1] / building.capacity
                except:
                        c_bat_init[i - 1] = 0
        # Heat (Energy->dhw) Storage
        C_f_Hsto = [0.00] * _num_buildings  # P
        C_p_Hsto = [3 * H_max[i] for i in range(_num_buildings)]  # P
        eta_Hsto = [1] * _num_buildings  # P
        # current hour soc. normalized
        c_Hsto_init = [None] * _num_buildings  # can't get future data since action dependent
        for i in range(1, _num_buildings + 1):
                building = env.buildings['Building_' + str(i)].dhw_storage
                try:
                        c_Hsto_init[i - 1] = building.soc[t-1] / building.capacity
                except:
                        c_Hsto_init[i - 1] = 0
                # Cooling (Energy->cooling) Storage
        C_f_Csto = [0.00] * _num_buildings  # P
        C_p_Csto = [2 * C_max[i] for i in range(_num_buildings)]  # P
        eta_Csto = [1] * _num_buildings  # P
        # current hour soc. normalized
        c_Csto_init = [None] * _num_buildings  # can't get future data since action dependent
        for i in range(1, _num_buildings + 1):
                building = env.buildings['Building_' + str(i)].cooling_storage
                try:
                        c_Csto_init[i - 1] = building.soc[t-1] / building.capacity
                except:
                        c_Csto_init[i - 1] = 0
        # fill data
        observation_data['p_ele'] = p_ele
        observation_data['ramping_cost_coeff'] = ramping_cost_coeff
        observation_data['E_grid_past'] = E_grid_past

        observation_data['E_ns'] = E_ns
        observation_data['H_bd'] = H_bd
        observation_data['C_bd'] = C_bd
        observation_data['H_max'] = H_max
        observation_data['C_max'] = C_max

        observation_data['E_pv'] = E_pv

        observation_data['E_hpC_max'] = E_hpC_max
        observation_data['eta_hp'] = eta_hp
        observation_data['t_C_hp'] = t_C_hp
        observation_data['COP_C'] = COP_C

        observation_data['eta_ehH'] = eta_ehH
        observation_data['E_ehH_max'] = E_ehH_max

        observation_data['C_f_bat'] = C_f_bat
        observation_data['C_p_bat'] = C_p_bat
        observation_data['eta_bat'] = eta_bat
        observation_data['c_bat_init'] = c_bat_init
        observation_data['c_bat_end'] = [0.1] * _num_buildings

        observation_data['C_f_Hsto'] = C_f_Hsto
        observation_data['C_p_Hsto'] = C_p_Hsto
        observation_data['eta_Hsto'] = eta_Hsto
        observation_data['c_Hsto_init'] = c_Hsto_init

        observation_data['C_f_Csto'] = C_f_Csto
        observation_data['C_p_Csto'] = C_p_Csto
        observation_data['eta_Csto'] = eta_Csto
        observation_data['c_Csto_init'] = c_Csto_init

        return observation_data

class Optim:
        """ Define Differential Optimization framework for CL. """

        def __init__(self, t: int, parameters: dict, building_id: int, num_actions: int):
                """
                @Param:
                - `parameters` : data (dict) from r <= t <= T following `get_current_data` format.
                - `T` : 24 hours (constant)
                - `t` : hour to solve optimization for.
                - `building_id`: building index number (0-based)
                - `num_actions`: Number of actions for building
                    NOTE: right now, this is an integer, but will be checked programmatically.
                Solves per building as specified by `building_id`. Note: 0 based.
                """
                T = 24
                window = T - t
                self.constraints = []
                self.costs = []
                self.t = t
                self.num_actions = num_actions

                # -- define action space -- #
                bounds_high, bounds_low = np.vstack([actions_spaces[building_id].high, actions_spaces[building_id].low])
                # parse to dictionary --- temp... need to check w/ state-action-dictionary.json !!! @Zhaiyao !!!
                if len(bounds_high) == 2: #bug
                        bounds_high = {'action_C': bounds_high[0], 'action_H': None, 'action_bat': bounds_high[1]}
                        bounds_low = {'action_C': bounds_low[0], 'action_H': None, 'action_bat': bounds_low[1]}
                else:
                        bounds_high = {'action_C': bounds_high[0], 'action_H': bounds_high[1],
                                       'action_bat': bounds_high[2]}
                        bounds_low = {'action_C': bounds_low[0], 'action_H': bounds_low[1], 'action_bat': bounds_low[2]}

                # -- define action space -- #

                # define parameters and variables

                ### --- Parameters ---
                p_ele = cp.Parameter(name='p_ele', shape=(window), value=parameters['p_ele'][t:, building_id])
                E_grid_prevhour = cp.Parameter(name='E_grid_prevhour',
                                               value=0)

                E_grid_pkhist = cp.Parameter(name='E_grid_pkhist',
                                             value=0)

                # max-min normalization of ramping_cost to downplay E_grid_sell weight.
                ramping_cost_coeff = cp.Parameter(name='ramping_cost_coeff',
                                                  value=parameters['ramping_cost_coeff'][t, building_id])

                # Loads
                E_ns = cp.Parameter(name='E_ns', shape=(window), value=parameters['E_ns'][t:, building_id])
                H_bd = cp.Parameter(name='H_bd', shape=(window), value=parameters['H_bd'][t:, building_id])
                C_bd = cp.Parameter(name='C_bd', shape=(window), value=parameters['C_bd'][t:, building_id])

                # PV generations
                E_pv = cp.Parameter(name='E_pv', shape=(window), value=parameters['E_pv'][t:, building_id])

                # Heat Pump
                COP_C = cp.Parameter(name='COP_C', shape=(window), value=parameters['COP_C'][t:, building_id])
                E_hpC_max = cp.Parameter(name='E_hpC_max', value=parameters['E_hpC_max'][t, building_id])

                # Electric Heater
                eta_ehH = cp.Parameter(name='eta_ehH', value=parameters['eta_ehH'][t, building_id])
                E_ehH_max = cp.Parameter(name='E_ehH_max', value=parameters['E_ehH_max'][t, building_id])

                # Battery
                C_f_bat = cp.Parameter(name='C_f_bat', value=parameters['C_f_bat'][t, building_id])
                C_p_bat = parameters['C_p_bat'][
                        t, building_id]  # cp.Parameter(name='C_p_bat', value=parameters['C_p_bat'][t, building_id])
                eta_bat = cp.Parameter(name='eta_bat', value=parameters['eta_bat'][t, building_id])
                soc_bat_init = cp.Parameter(name='soc_bat_init', value=parameters['c_bat_init'][building_id])
                soc_bat_norm_end = cp.Parameter(name='soc_bat_norm_end', value=parameters['c_bat_end'][t,building_id])

                # Heat (Energy->dhw) Storage
                C_f_Hsto = cp.Parameter(name='C_f_Hsto', value=parameters['C_f_Hsto'][t, building_id])  # make constant.
                C_p_Hsto = cp.Parameter(name='C_p_Hsto', value=parameters['C_p_Hsto'][t, building_id])
                eta_Hsto = cp.Parameter(name='eta_Hsto', value=parameters['eta_Hsto'][t, building_id])
                soc_Hsto_init = cp.Parameter(name='soc_Hsto_init', value=parameters['c_Hsto_init'][building_id])

                # Cooling (Energy->cooling) Storage
                C_f_Csto = cp.Parameter(name='C_f_Csto', value=parameters['C_f_Csto'][t, building_id])
                C_p_Csto = cp.Parameter(name='C_p_Csto', value=parameters['C_p_Csto'][t, building_id])
                eta_Csto = cp.Parameter(name='eta_Csto', value=parameters['eta_Csto'][t, building_id])
                soc_Csto_init = cp.Parameter(name='soc_Csto_init', value=parameters['c_Csto_init'][building_id])

                ### --- Variables ---

                # relaxation variables - prevents numerical failures when solving optimization
                E_bal_relax = cp.Variable(name='E_bal_relax', shape=(window))  # electricity balance relaxation
                H_bal_relax = cp.Variable(name='H_bal_relax', shape=(window))  # heating balance relaxation
                C_bal_relax = cp.Variable(name='C_bal_relax', shape=(window))  # cooling balance relaxation

                E_grid = cp.Variable(name='E_grid', shape=(window))  # net electricity grid
                E_grid_sell = cp.Variable(name='E_grid_sell', shape=(window))  # net electricity grid

                E_hpC = cp.Variable(name='E_hpC', shape=(window))  # heat pump
                E_ehH = cp.Variable(name='E_ehH', shape=(window))  # electric heater

                SOC_bat = cp.Variable(name='SOC_bat', shape=(window))  # electric battery
                SOC_Brelax = cp.Variable(name='SOC_Brelax', shape=(
                        window))  # electrical battery relaxation (prevents numerical infeasibilities)
                action_bat = cp.Variable(name='action_bat', shape=(window))  # electric battery

                SOC_H = cp.Variable(name='SOC_H', shape=(window))  # heat storage
                SOC_Hrelax = cp.Variable(name='SOC_Hrelax',
                                         shape=(window))  # heat storage relaxation (prevents numerical infeasibilities)
                action_H = cp.Variable(name='action_H', shape=(window))  # heat storage

                SOC_C = cp.Variable(name='SOC_C', shape=(window))  # cooling storage
                SOC_Crelax = cp.Variable(name='SOC_Crelax', shape=(
                        window))  # cooling storage relaxation (prevents numerical infeasibilities)
                action_C = cp.Variable(name='action_C', shape=(window))  # cooling storage

                ### objective function
                ramping_cost = cp.abs(E_grid[0] - E_grid_prevhour) + cp.sum(
                        cp.abs(E_grid[1:] - E_grid[:-1]))  # E_grid_t+1 - E_grid_t
                peak_net_electricity_cost = cp.max(
                        cp.atoms.affine.hstack.hstack([*E_grid, E_grid_pkhist]))  # max(E_grid, E_gridpkhist)
                electricity_cost = cp.sum(p_ele * E_grid)
                selling_cost = -1e2 * cp.sum(E_grid_sell)  # not as severe as violating constraints

                ### relaxation costs - L1 norm
                # balance eq.
                E_bal_relax_cost = cp.sum(cp.abs(E_bal_relax))
                H_bal_relax_cost = cp.sum(cp.abs(H_bal_relax))
                C_bal_relax_cost = cp.sum(cp.abs(C_bal_relax))
                # soc eq.
                SOC_Brelax_cost = cp.sum(cp.abs(SOC_Brelax))
                SOC_Crelax_cost = cp.sum(cp.abs(SOC_Crelax))
                SOC_Hrelax_cost = cp.sum(cp.abs(SOC_Hrelax))

                self.costs.append(ramping_cost_coeff.value * ramping_cost +
                                  peak_net_electricity_cost + 0*electricity_cost + selling_cost +
                                  E_bal_relax_cost * 1e4 + H_bal_relax_cost * 1e4 + C_bal_relax_cost * 1e4
                                  + SOC_Brelax_cost * 1e4 + SOC_Crelax_cost * 1e4 + SOC_Hrelax_cost * 1e4)

                ### constraints
                self.constraints.append(E_grid >= 0)
                self.constraints.append(E_grid_sell <= 0)

                # energy balance constraints
                self.constraints.append(
                        E_pv + E_grid + E_grid_sell + E_bal_relax == E_ns + E_hpC + E_ehH + action_bat * C_p_bat)  # electricity balance
                self.constraints.append(E_ehH * eta_ehH + H_bal_relax == action_H * C_p_Hsto + H_bd)  # heat balance

                # !!!!! Problem Child !!!!!
                self.constraints.append(E_hpC * COP_C + C_bal_relax == action_C * C_p_Csto + C_bd)  # cooling balance
                # !!!!! Problem Child !!!!!

                # heat pump constraints
                self.constraints.append(E_hpC <= E_hpC_max)  # maximum cooling
                self.constraints.append(E_hpC >= 0)  # constraint minimum cooling to positive
                # electric heater constraints
                self.constraints.append(E_ehH >= 0)  # constraint to PD
                self.constraints.append(E_ehH <= E_ehH_max)  # maximum limit

                # electric battery constraints

                self.constraints.append(
                        SOC_bat[0] == (1 - C_f_bat) * soc_bat_init + action_bat[0] * eta_bat + SOC_Crelax[
                                0])  # initial SOC
                # soc updates
                for i in range(1, window):  # 1 = t + 1
                        self.constraints.append(
                                SOC_bat[i] == (1 - C_f_bat) * SOC_bat[i - 1] + action_bat[i] * eta_bat + SOC_Crelax[i])
                self.constraints.append(SOC_bat[-1] == soc_bat_norm_end)  # soc terminal condition
                self.constraints.append(SOC_bat >= 0)  # battery SOC bounds
                self.constraints.append(SOC_bat <= 1)  # battery SOC bounds

                # Heat Storage constraints
                self.constraints.append(
                        SOC_H[0] == (1 - C_f_Hsto) * soc_Hsto_init + action_H[0] * eta_Hsto + SOC_Hrelax[
                                0])  # initial SOC
                # soc updates
                for i in range(1, window):
                        self.constraints.append(
                                SOC_H[i] == (1 - C_f_Hsto) * SOC_H[i - 1] + action_H[i] * eta_Hsto + SOC_Hrelax[i])
                self.constraints.append(SOC_H >= 0)  # battery SOC bounds
                self.constraints.append(SOC_H <= 1)  # battery SOC bounds

                # Cooling Storage constraints
                self.constraints.append(
                        SOC_C[0] == (1 - C_f_Csto) * soc_Csto_init + action_C[0] * eta_Csto + SOC_Crelax[
                                0])  # initial SOC
                # soc updates
                for i in range(1, window):
                        self.constraints.append(
                                SOC_C[i] == (1 - C_f_Csto) * SOC_C[i - 1] + action_C[i] * eta_Csto + SOC_Crelax[i])
                self.constraints.append(SOC_C >= 0)  # battery SOC bounds
                self.constraints.append(SOC_C <= 1)  # battery SOC bounds

                #### action constraints (limit to action-space)
                # format: AS[building_id][0/1 (high/low)][heat, cool, battery]

                heat_high, cool_high, battery_high = bounds_high
                heat_low, cool_low, battery_low = bounds_low

                assert len(bounds_high) == 3, 'Invalid number of bounds for actions - see dict defined in `Optim`'

                for high, low in zip(bounds_high.items(), bounds_low.items()):
                        key, h, l = [*high, low[1]]
                        if not (h and l):  # throw DeMorgan's in!!!
                                continue

                        # heating action
                        if key == 'action_C':
                                self.constraints.append(action_C <= h)
                                self.constraints.append(action_C >= l)
                        # cooling action
                        elif key == 'action_H':
                                self.constraints.append(action_H <= h)
                                self.constraints.append(action_H >= l)
                        # Battery action
                        elif key == 'action_bat':
                                self.constraints.append(action_bat <= h)
                                self.constraints.append(action_bat >= l)

        def get_problem(self):
                """ Returns raw problem """
                # Form objective.
                obj = cp.Minimize(*self.costs)
                # Form and solve problem.
                prob = cp.Problem(obj, self.constraints)

                return prob

        def get_constraints(self):
                """ Returns constraints for problem """
                return self.constraints

        def solve(self, debug=False, dispatch=False):
                prob = self.get_problem()  # Form and solve problem
                actions = {}
                try:
                        status = prob.solve(verbose=debug)  # Returns the optimal value.
                except:
                        return [0,0,0], 0 if dispatch else None, actions
                if float('-inf') < status < float('inf'):
                        pass
                else:
                        return "Unbounded Solution"


                for var in prob.variables():
                        if dispatch:
                                actions[var.name()] = np.array(
                                        var.value)  # no need to clip... automatically restricts range
                        else:
                                actions[var.name()] = var.value[0]  # no need to clip... automatically restricts range

                # Temporary... needs fixing!
                ## compute dispatch cost
                params = {x.name(): x.value for x in prob.parameters()}

                if dispatch:
                        ramping_cost = np.sum(np.abs(actions['E_grid'][1:] + actions['E_grid_sell'][1:] -
                                                     actions['E_grid'][:-1] - actions['E_grid_sell'][:-1]))
                        net_peak_electricity_cost = np.max(actions['E_grid'])
                        virtual_electricity_cost = np.sum(params['p_ele'] * actions['E_grid'])
                        dispatch_cost = virtual_electricity_cost  # ramping_cost + net_peak_electricity_cost + virtual_electricity_cost

                if self.num_actions == 2:
                        return [actions['action_H'], actions['action_bat']], dispatch_cost if dispatch else None
                return [actions['action_C'], actions['action_H'],
                        actions['action_bat']], dispatch_cost if dispatch else None, actions

def get_idx_hour():
    # Finding which state
    with open('buildings_state_action_space.json') as file:
        actions_ = json.load(file)

    indx_hour = -1
    for obs_name, selected in list(actions_.values())[0]['states'].items():
        indx_hour += 1
        if obs_name=='hour':
            break
        assert indx_hour < len(list(actions_.values())[0]['states'].items()) - 1, "Please, select hour as a state for Building_1 to run the RBC"
    return indx_hour


def estimate_data(surrogate_env: CityLearn, data: dict, t_start: int, init_updates: dict):
        """ Returns data for hours `t_start` - 24 using `surrogate_env` running RBC `agent` """
        for i in range(0, 24):
                data = parse_data(data, get_current_data_oracle(surrogate_env, t_start + i))

        return init_values(data, init_updates)[0] if t_start == 0 else data  # only load previous values at start of day

def get_rbc_data(surrogate_env:CityLearn, state, indx_hour:int, dump_data:list, run_timesteps:int):
    """ Runs RBC for x number of timesteps """
    ## --- RBC generation ---
    for i in range(run_timesteps):
        hour_state = np.array([[state[0][indx_hour]]])
        action = agents.select_action(hour_state) #using RBC to select next action given current sate
        next_state, rewards, done, _ = surrogate_env.step(action)
        state = next_state
        dump_data.append([x[28] for x in state])


def init_values(data: dict, update_values: dict = None):
        """ Loads eod values for SOC and E_grid_past before(after) wiping data cache """
        if update_values:
                # assign previous day's end socs.
                data['c_bat_init'][0] = update_values['c_bat_init']
                data['c_Hsto_init'][0] = update_values['c_Hsto_init']
                data['c_Csto_init'][0] = update_values['c_Csto_init']

                # assign previous day's end E_grid.
                data['E_grid_past'][0] = update_values['E_grid_past']
        else:
                update_values = {'c_bat_init': data['c_bat_init'][-1], 'c_Hsto_init': data['c_Hsto_init'][-1],
                                 'c_Csto_init': data['c_Csto_init'][-1], 'E_grid_past': data['E_grid_past'][-1]}

        return data, update_values

def get_ramping_rbc(day_data):
    arr = []
    for day in range(len(day_data) // 24): #number of days
        data = day_data[day : day + 24]
        arr.append( np.sum(np.abs(data[1:] - data[:-1])) )
    return arr

def get_peak_rbc(day_data):
    arr = []
    for day in range(len(day_data) // 24): #number of days
        arr.append(np.max(day_data[day : day + 24]))
    return arr

def get_virtual_electricity_rbc(day_data):
    arr = []
    p_ele = np.array([2 if 10 <= i <= 20 else 0.2 for i in range(24)])
    for day in range(len(day_data) // 24): #number of days
        arr.append(np.sum(p_ele * day_data[day : day + 24]))
    return arr


#### accumulate
data = {}
actions_arr = []  # plot actions
E_grid = []
E_grid_pred = []
E_grid_sell_pred = []

check_data = {}
debug_item = ['E_grid','E_bal_relax','H_bal_relax','C_bal_relax','E_grid_sell','E_hpC','E_ehH','SOC_bat','SOC_Brelax','action_bat',
              'SOC_H','SOC_Hrelax','action_H','SOC_C','SOC_Crelax','action_C']
for key in debug_item:
        check_data[key] = []
check_params = {}
debug_params = ['E_ns','H_bd','C_bd']
for  key in debug_params:
        check_params[key] = []
#### accumulate


state = env.reset()  # states/building
done = False
t_idx = 0
rbc_threshold = 336  # run RBC for 2 weeks
end_time = rbc_threshold + 24 * 10

total_rewards = []  # reward for each building

agents = RBC(actions_spaces)
indx_hour = get_idx_hour()

start = time.time()

look_ahead_cost = []
RBC_Egrid = []

get_rbc_data(deepcopy(env), state, indx_hour, RBC_Egrid, end_time)

# run agent
while not done and t_idx < end_time:

        hour_state = np.array([[state[0][indx_hour]]])

        if t_idx % 24 == 0 and t_idx > rbc_threshold - 24:  # reset values every day

                _, init_updates = init_values(data)  # update 0th hour values
                data = {}

        if t_idx % 1460 < rbc_threshold:
                action = agents.select_action(hour_state)
                next_state, rewards, done, _ = env.step(action)
                E_grid_pred.append([x[28] for x in next_state])
                E_grid_sell_pred.append([0 for x in next_state])
                for key in debug_item:
                        check_data[key].append([0 for x in next_state])
                for key in debug_params:
                        check_params[key].append([0 for x in next_state])
                actions_arr.append(action)
        else:
                # day ahead dispatch.
                ### at first hour, collects the data from t = 1 to t = 24 using RBC.
                ### solves optimization for all hours, and runs corresponding actions for the next 23 hours
                if t_idx % 24 == 0:  # first hour
                        data_est = estimate_data(env, deepcopy(data), t_idx, init_updates)
                        optim_results = get_actions(data_est, t_idx % 24, lookahead=True)  # day-ahead plan
                        action_planned_day, cost_dispatch, action_planned = zip(*optim_results)

                        look_ahead_cost.append(
                                cost_dispatch)  # per day estimation cost after solving for hour 1 for hours 1-24
                # action = get_actions(data_est, t_idx % 24) #runs optimization per hour.
                assert len(action_planned_day[0][0]) == 24, 'Invalid number of observations for Optimization actions'

                actions = [np.array(action_planned_day[idx])[:, t_idx % 24] for idx in range(len(actions_spaces))]

                next_state, rewards, done, _ = env.step(actions)
                actions_arr.append(np.array(actions))
                E_grid_pred.append([x['E_grid'][t_idx % 24] for x in action_planned])
                E_grid_sell_pred.append([x['E_grid_sell'][t_idx % 24] for x in action_planned])
                for key in debug_item:
                        check_data[key].append([x[key][t_idx % 24] for x in action_planned])
                for key in debug_params:
                        check_params[key].append(data_est[key][t_idx % 24,:])
        state = next_state
        E_grid.append([x[28] for x in state])  # E_Grid

        if t_idx >= rbc_threshold - 24:  # start collecting data
                data = parse_data(data, get_current_data_oracle(env, t_idx))
        total_rewards.append(rewards)

        t_idx += 1

        print(f"\rTime step: {t_idx}", end='')

end = time.time()
print(f"\nTotal time = {end - start}")
E_grid = np.array(E_grid).T  # set per building
E_grid_pred = np.array(E_grid_pred).T  # set per building
E_grid_sell_pred = np.array(E_grid_sell_pred).T

RBC_Egrid = np.array(RBC_Egrid).T  # set per building

# plt.figure(figsize=(10, 7))

# plot predicted E_grid
week = end_time - 24*3  # plots last week of the month data
fig, axs = plt.subplots(3, 3,figsize=(15,15))
for i in range(3):
        for j in range(3):
                bid = i*3+j
                axs[i,j].set_title(f"Building {bid + 1}")
                axs[i,j].plot(E_grid[bid][week:],label='True E grid: Optim')  # plot true E grid
                axs[i,j].plot(E_grid_pred[bid][week:]+E_grid_sell_pred[bid][week:], 'gx', label='Optim predicted E grid')  # plots per month
                axs[i, j].plot(RBC_Egrid[bid][week:], label='True E grid: RBC')  # plot true E grid
                axs[i, j].grid()
                if j == 0:
                        axs[i, j].set_ylabel("E grid")
                if i == 0:
                        axs[i, j].set_xlabel("Hour")
plt.legend()
fig.savefig("images/Egrid_compare.pdf", bbox_inches='tight')

# plot optimization variables
week = end_time - 24*3  # plots last week of the month data
for key in debug_item:
        data_np = np.array(check_data[key]).T
        fig, axs = plt.subplots(3, 3,figsize=(15,15))
        for i in range(3):
                for j in range(3):
                        bid = i*3+j
                        axs[i,j].set_title(f"Building {bid + 1}: {key}")
                        axs[i,j].plot(data_np[bid][week:],label=key)  # plot true E grid
                        axs[i, j].grid()
                        if j == 0:
                                axs[i, j].set_ylabel(key)
                        if i == 0:
                                axs[i, j].set_xlabel("Hour")
        plt.legend()
        fig.savefig(f"images/{key}_plot.pdf", bbox_inches='tight')

# Plot variables Edhw, Ehp

env_comp_item = ['electric_consumption_cooling','electric_consumption_dhw']
env_comp_item_check = ['E_hpC','E_ehH']
week = end_time - 24*3  # plots last week of the month data
for key_i in range(len(env_comp_item)):
        data_np = np.array(check_data[env_comp_item_check[key_i]]).T

        fig, axs = plt.subplots(3, 3,figsize=(15,15))
        for i in range(3):
                for j in range(3):
                        bid = i*3+j
                        data_env = np.array(getattr(env.buildings['Building_' + str(bid+1)],env_comp_item[key_i]))
                        axs[i,j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
                        axs[i,j].plot(data_np[bid][week:],label="optimization")  # plot true E grid
                        axs[i, j].plot(data_env[week:], label="environment")  # plot true E grid
                        axs[i, j].grid()
                        if j == 0:
                                axs[i, j].set_ylabel(key)
                        if i == 0:
                                axs[i, j].set_xlabel("Hour")
        plt.legend()
        fig.savefig(f"images/{env_comp_item_check[key_i]}_optim_env_plot.pdf", bbox_inches='tight')

# Plot energy balance

env_comp_item = ['electrical_storage','cooling_storage','dhw_storage']
env_comp_item_check = ['action_bat','action_C','action_H']
env_comp_item_check2 = ['SOC_bat','SOC_C','SOC_H']
env_comp_item_check3 = ['C_p_bat','C_p_Csto','C_p_Hsto']

week = end_time - 24*3  # plots last week of the month data
for key_i in range(len(env_comp_item)):
        data_np = np.array(check_data[env_comp_item_check[key_i]]).T
        data_np2 = np.array(check_data[env_comp_item_check2[key_i]]).T
        fig, axs = plt.subplots(3, 3,figsize=(15,15))
        for i in range(3):
                for j in range(3):
                        bid = i*3+j
                        data_env = np.array(getattr(getattr(env.buildings['Building_' + str(bid+1)],env_comp_item[key_i]),'energy_balance'))
                        data_env2 = np.array(getattr(getattr(env.buildings['Building_' + str(bid + 1)], env_comp_item[key_i]),
                                        'soc'))
                        axs[i,j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
                        axs[i,j].plot(data_np[bid][week:]*data_est[env_comp_item_check3[key_i]][0,bid],label="optimization")  # plot true E grid
                        axs[i, j].plot(data_np2[bid][week:] *data_est[env_comp_item_check3[key_i]][0,bid],label="optimization SOC")
                        axs[i, j].plot(data_env[week:], label="environment")  # plot true E grid
                        axs[i, j].plot(data_env2[week:], label="environment SOC")  # plot true E grid

                        axs[i, j].grid()
                        if j == 0:
                                axs[i, j].set_ylabel(key)
                        if i == 0:
                                axs[i, j].set_xlabel("Hour")
        plt.legend()
        fig.savefig(f"images/{env_comp_item_check[key_i]}_optim_env_plot.pdf", bbox_inches='tight')

# Plot loads
week = end_time - 24*3  # plots last week of the month data
for key in debug_params:
        data_np = np.array(check_params[key]).T
        fig, axs = plt.subplots(3, 3,figsize=(15,15))
        for i in range(3):
                for j in range(3):
                        bid = i*3+j
                        axs[i,j].set_title(f"Building {bid + 1}: {key}")
                        axs[i,j].plot(data_np[bid][week:],label=key)  # plot true E grid
                        axs[i, j].grid()
                        if j == 0:
                                axs[i, j].set_ylabel(key)
                        if i == 0:
                                axs[i, j].set_xlabel("Hour")
        plt.legend()
        fig.savefig(f"images/{key}_plot.pdf", bbox_inches='tight')

# Get actions
actions_arr = np.array(actions_arr)

list_actions = ['action_C', 'action_H', 'action_bat']
week = end_time - 24*3  # plots last week of the month data
fig, axs = plt.subplots(3, 3,figsize=(15,15))
for i in range(3):
        for j in range(3):
                bid = i*3+j
                axs[i,j].set_title(f"Building {bid + 1}")
                for k in range(3):
                        axs[i,j].plot(actions_arr[week:,bid,k],label=list_actions[k])
                axs[i, j].grid()
                if j == 0:
                        axs[i, j].set_ylabel("Actions")
                if i == 0:
                        axs[i, j].set_xlabel("Hour")
plt.legend()
fig.savefig("images/Actions_compare.pdf", bbox_inches='tight')

# Compare the ramping, peak electricity costs
week = end_time - 24 * 10  # plots last week of the month data

ramping_cost_optim = []
ramping_cost_RBC = []
peak_electricity_cost_optim = []
peak_electricity_cost_RBC = []

for i in range((end_time-week)//24):
        t_start = week+i*24
        t_end = week+(i+1)*24
        ramping_cost_optim_t = []
        ramping_cost_RBC_t = []
        peak_electricity_cost_optim_t = []
        peak_electricity_cost_RBC_t = []
        for bid in range(9):
                E_grid_t = E_grid[bid][t_start:t_end]
                RBC_Egrid_t = RBC_Egrid[bid][t_start:t_end]
                ramping_cost_optim_t.append(np.sum(np.abs(E_grid_t[1:] - E_grid_t[:-1])))
                ramping_cost_RBC_t.append(np.sum(np.abs(RBC_Egrid_t[1:] - RBC_Egrid_t[:-1])))
                peak_electricity_cost_optim_t.append(np.max(E_grid_t))
                peak_electricity_cost_RBC_t.append(np.max(RBC_Egrid_t))
        ramping_cost_optim.append(ramping_cost_optim_t)
        ramping_cost_RBC.append(ramping_cost_RBC_t)
        peak_electricity_cost_optim.append(peak_electricity_cost_optim_t)
        peak_electricity_cost_RBC.append(peak_electricity_cost_RBC_t)

Optim_cost = {'ramping_cost':np.array(ramping_cost_optim).T,
              'peak_electricity_cost':np.array(peak_electricity_cost_optim).T,
              'total_cost':np.array(ramping_cost_optim).T+np.array(peak_electricity_cost_optim).T}
RBC_cost = {'ramping_cost':np.array(ramping_cost_RBC).T,
              'peak_electricity_cost':np.array(peak_electricity_cost_RBC).T,
            'total_cost':np.array(ramping_cost_RBC).T+np.array(peak_electricity_cost_RBC).T}

item_cost = ['ramping_cost','peak_electricity_cost','total_cost']
for k in range(len(item_cost)):
        fig, axs = plt.subplots(3, 3,figsize=(15,15))
        for i in range(3):
                for j in range(3):
                        bid = i*3+j
                        axs[i,j].set_title(f"Building {bid + 1}: {item_cost[k]}")
                        axs[i, j].plot(Optim_cost[item_cost[k]][bid,:], label=f"Optim: {item_cost[k]}")  # plot true E grid
                        axs[i, j].plot(RBC_cost[item_cost[k]][bid,:], label=f"RBC: {item_cost[k]}")
                        axs[i, j].grid()
                        if j == 0:
                                axs[i, j].set_ylabel("Cost")
                        if i == 0:
                                axs[i, j].set_xlabel("Day")
        plt.legend()
        fig.savefig(f"images/{item_cost[k]}_compare.pdf", bbox_inches='tight')

fig, axs = plt.subplots(3, 3,figsize=(15,15))
for i in range(3):
        for j in range(3):
                bid = i*3+j
                axs[i,j].set_title(f"Building {bid + 1}: total cost Optim/RBC")
                axs[i, j].plot(Optim_cost['total_cost'][bid,:]/RBC_cost['total_cost'][bid,:], label=f"Optim/RBC")  # plot true E grid
                axs[i, j].grid()
                if j == 0:
                        axs[i, j].set_ylabel("Cost (Ratio)")
                if i == 0:
                        axs[i, j].set_xlabel("Day")
plt.legend()
fig.savefig(f"images/total_cost_ratio.pdf", bbox_inches='tight')