# Run this again after editing submodules so Colab uses the updated versions
from citylearn import CityLearn
from pathlib import Path
from agent_final import Agent
from copy import deepcopy
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


import matplotlib.pyplot as plt


import utils
import time

import numpy as np
import pandas as pd
import torch


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("images/logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


sys.stdout = Logger()

# Load environment
climate_zone = 5
params = {
    "data_path": Path("data/Climate_Zone_" + str(climate_zone) + "_org"),
    "building_attributes": "building_attributes.json",
    "weather_file": "weather_data.csv",
    "solar_profile": "solar_generation_1kW.csv",
    "carbon_intensity": "carbon_intensity.csv",
    "building_ids": ["Building_" + str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]],
    "buildings_states_actions": "buildings_state_action_space.json",
    "simulation_period": (0, 8760 * 4 - 1),
    "cost_function": [
        "ramping",
        "1-load_factor",
        "average_daily_peak",
        "peak_demand",
        "net_electricity_consumption",
        "carbon_emissions",
    ],
    "central_agent": False,
    "save_memory": False,
}

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
env = CityLearn(**params)
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()


params_agent = {
    "building_ids": ["Building_" + str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]],
    "buildings_states_actions": "buildings_state_action_space.json",
    "building_info": building_info,
    "observation_spaces": observations_spaces,
    "action_spaces": actions_spaces,
}

# Instantiating the control agent(s)
agents = Agent(**params_agent)

RBC_THRESHOLD = 24 * 14
# agents = Agent(
#     num_actions=actions_spaces,
#     num_buildings=len(observations_spaces),
#     env = env, is_oracle = False,
# )

state = env.reset()
done = False

action = agents.select_action(state, False)
costs_peak_net_ele = []

t_idx = 0
# run for a month - NOTE: THIS WILL TAKE ~2 HOURS TO RUN. reduce `end_time` for quicker results.
end_time = RBC_THRESHOLD + 24 * 365

start_time = time.time()


# returns E_grid for RBC agent
E_grid_RBC = utils.RBC(actions_spaces).get_rbc_data(deepcopy(env), state, end_time)
E_grid_RBC = np.array(E_grid_RBC)

E_grid_true = []  # see comments below for more info.

while not done and env.time_step < end_time:

    next_state, reward, done, _ = env.step(action)
    action_next = agents.select_action(
        next_state
    )  # passing in environment for Oracle agent.

    #     agents.add_to_buffer_oracle(state, env, action, reward, next_state)
    agents.add_to_buffer(state, action, reward, next_state, done)
    ## add env E-grid
    E_grid_true.append([x[28] for x in state])
    state = next_state
    action = action_next

    t_idx += 1

    print(f"\rTime step: {t_idx}", end="")

print(
    f"\nTotal time (min) to run {end_time // 24} days of simulation: {round((time.time() - start_time) / 60, 3)}"
)

x = agents.logger

e_grid = []

for h in x:
    l = []
    for bid in h:
        l.append(bid["E_grid"] + bid["E_grid_sell"])
    e_grid.append(l)
e_grid = np.array(e_grid)[:-1]

tmp_e_grid_true = np.array(E_grid_true[RBC_THRESHOLD + 1 :])
tmp_rbc_e_grid = np.array(E_grid_RBC[RBC_THRESHOLD + 1 :])
np.savetxt("images/E_grid_true.csv", tmp_e_grid_true, delimiter=",")
np.savetxt("images/E_grid_true_RBC.csv", tmp_e_grid_true, delimiter=",")

np.shape(tmp_e_grid_true), np.shape(tmp_rbc_e_grid), np.shape(e_grid)


check_data = {}
debug_item = [
    "E_grid",
    "E_bal_relax",
    "H_bal_relax",
    "C_bal_relax",
    "E_grid_sell",
    "E_hpC",
    "E_ehH",
    "SOC_bat",
    "SOC_Brelax",
    "action_bat",
    "SOC_H",
    "SOC_Hrelax",
    "action_H",
    "SOC_C",
    "SOC_Crelax",
    "action_C",
]
for key in debug_item:
    check_data[key] = [[] for i in range(9)]
check_params = {}
debug_params1 = ["E_grid_prevhour", "c_bat_init", "c_Hsto_init", "c_Csto_init"]
debug_params2 = [
    "E_ns",
    "H_bd",
    "C_bd",
    "E_pv",
    "COP_C",
    "C_p_bat",
    "C_p_Hsto",
    "C_p_Csto",
]
for key in debug_params1:
    check_params[key] = [[] for i in range(9)]
for key in debug_params2:
    check_params[key] = [[] for i in range(9)]

vars_RL = agents.logger
pars_RL = agents.optim_param_logger
# collect all data
start_time = RBC_THRESHOLD
num_days = int((end_time - start_time) / 24)
for i in range(end_time - start_time):
    optim_var = vars_RL[start_time + i - RBC_THRESHOLD]
    optim_param = pars_RL[start_time + i]
    for key in debug_item:
        for bid in range(9):
            check_data[key][bid].append(optim_var[bid][key])
    for key in debug_params1:
        for bid in range(9):
            check_params[key][bid].append(optim_param[key][i % 24, bid])

for key in debug_item:
    np.savetxt(
        f"images/{key}_data.csv",
        np.array([check_data[key][bid] for bid in range(9)]).T,
        delimiter=",",
    )

for i in range(num_days):

    for key in debug_params2:
        for bid in range(9):
            val_t = [[] for _ in range(24)]
            for hour_i in range(24):
                for hour_j in range(hour_i + 1):
                    optim_param = pars_RL[start_time + i * 24 + hour_j]
                    val_t[hour_i].append(optim_param[key][hour_i, bid])
            for hour_i in range(24):
                check_params[key][bid].append(np.mean(np.array(val_t[hour_i])))

for key in debug_params2:
    for bid in range(9):
        check_params[key][bid] = np.array(check_params[key][bid])

    np.savetxt(
        f"images/{key}_param.csv",
        np.array([check_params[key][bid] for bid in range(9)]).T,
        delimiter=",",
    )

for key in debug_params1:
    for bid in range(9):
        check_params[key][bid] = np.array(check_params[key][bid])
    np.savetxt(
        f"images/{key}_param.csv",
        np.array([check_params[key][bid] for bid in range(9)]).T,
        delimiter=",",
    )

E_grid_pkhist_true = [[] for _ in range(9)]
E_grid_true_arr = np.array(E_grid_true)
check_params["E_grid_pkhist"] = [[] for _ in range(9)]
for i in range(end_time - start_time):
    optim_param = pars_RL[start_time + i]
    day_ind_start = RBC_THRESHOLD + i - (i % 24)
    for bid in range(9):
        E_grid_pkhist_t = (
            max(0, optim_param["E_grid_prevhour"][i % 24, bid])
            if (i % 24) == 0
            else np.max([0, *optim_param["E_grid"][: (i % 24 + 1), bid]])
        )
        check_params["E_grid_pkhist"][bid].append(E_grid_pkhist_t)
        if i % 24 == 0:
            E_grid_pkhist_true_t = max(0, E_grid_true_arr[day_ind_start - 1, bid])
        else:
            E_grid_pkhist_true_t = np.max(
                E_grid_true_arr[day_ind_start : (day_ind_start + (i % 24) + 1), bid]
            )
        E_grid_pkhist_true[bid].append(E_grid_pkhist_true_t)

for bid in range(9):
    check_params["E_grid_pkhist"][bid] = np.array(check_params["E_grid_pkhist"][bid])
    E_grid_pkhist_true[bid] = np.array(E_grid_pkhist_true[bid])

# plot E_grid_pkhist
end_plot = end_time - 24 * 2 - RBC_THRESHOLD
start_plot = end_plot - 24 * 10  # plots last week of the month data
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        bid = i * 3 + j
        data_vec = check_params["E_grid_pkhist"][bid]
        data_true_vec = E_grid_pkhist_true[bid]
        axs[i, j].set_title(f"Building {bid + 1}")
        axs[i, j].plot(
            np.arange(start_plot, end_plot),
            data_true_vec[start_plot:end_plot],
            "-o",
            label="True E grid pkhist",
        )  # plot true E grid
        axs[i, j].plot(
            np.arange(start_plot, end_plot),
            data_vec[start_plot:end_plot],
            label="Optim E grid pkhist",
        )  # plot true E grid
        axs[i, j].grid()
        if j == 0:
            axs[i, j].set_ylabel("E grid pkhist")
        if i == 0:
            axs[i, j].set_xlabel("Hour")
plt.legend()
fig.savefig("images/Egrid_pkhist_compare.pdf", bbox_inches="tight")

# plot E_grid for RL and RBC
end_plot = end_time - 24 * 2 - RBC_THRESHOLD
start_plot = end_plot - 24 * 10  # plots last week of the month data

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        bid = i * 3 + j
        axs[i, j].set_title(f"Building {bid + 1}")
        axs[i, j].plot(
            np.arange(start_plot, end_plot),
            tmp_e_grid_true[start_plot:end_plot, bid],
            label="True E grid",
        )  # plot true E grid
        axs[i, j].plot(
            np.arange(start_plot, end_plot),
            e_grid[start_plot:end_plot, bid],
            label="Optim E grid",
        )  # plot true E grid
        axs[i, j].plot(
            np.arange(start_plot, end_plot),
            tmp_rbc_e_grid[start_plot:end_plot, bid],
            label="RBC E grid",
        )  # plot true E grid
        axs[i, j].grid()
        if j == 0:
            axs[i, j].set_ylabel("E grid")
        if i == 0:
            axs[i, j].set_xlabel("Hour")
plt.legend()
fig.savefig("images/Egrid_compare.pdf", bbox_inches="tight")


# Debug: plot battery initial values
# unnormalized SOCs
env_comp_item = ["electrical_storage", "cooling_storage", "dhw_storage"]
data_comp_item = ["c_bat_init", "c_Csto_init", "c_Hsto_init"]
data_comp_item2 = ["C_p_bat", "C_p_Csto", "C_p_Hsto"]
start_plot = RBC_THRESHOLD + 24 * 5
end_plot = start_plot + 5 * 24
for key_i in range(len(env_comp_item)):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    data_all = []

    for i in range(3):
        for j in range(3):
            bid = i * 3 + j
            optim_data_soc1 = check_params[data_comp_item[key_i]][bid]
            optim_data_soc2 = check_params[data_comp_item2[key_i]][bid]
            optim_data_soc = optim_data_soc1 * optim_data_soc2

            data_env = np.array(
                getattr(
                    getattr(
                        env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]
                    ),
                    "soc",
                )
            )
            data_all.append(data_env)
            axs[i, j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
            axs[i, j].plot(
                np.arange(1, 1 + end_plot - start_plot),
                data_env[start_plot:end_plot],
                "-o",
                label="true SOC",
            )
            axs[i, j].plot(
                np.arange(end_plot - start_plot),
                optim_data_soc[
                    (start_plot - RBC_THRESHOLD) : (end_plot - RBC_THRESHOLD)
                ],
                label="init SOC",
            )

            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel(env_comp_item[key_i])
            if i == 0:
                axs[i, j].set_xlabel("Hour")
    np.savetxt(
        f"images/{env_comp_item[key_i]}_soc.csv", np.array(data_all).T, delimiter=","
    )

    plt.legend()
    fig.savefig(f"images/{env_comp_item[key_i]}_optim_env_SOC.pdf", bbox_inches="tight")


# normalized SOCs
for key_i in range(len(env_comp_item)):
    data_all = []
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            bid = i * 3 + j
            optim_data_soc = check_params[data_comp_item[key_i]][bid]
            data_env = np.array(
                getattr(
                    getattr(
                        env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]
                    ),
                    "soc",
                )
            ) / np.array(
                getattr(
                    getattr(
                        env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]
                    ),
                    "capacity",
                )
            )
            data_all.append(data_env)
            axs[i, j].set_title(f"Building {bid + 1}")
            axs[i, j].plot(
                np.arange(1, 1 + end_plot - start_plot),
                data_env[start_plot:end_plot],
                "-o",
                label="true SOC (nomarlized)",
            )
            axs[i, j].plot(
                np.arange(end_plot - start_plot),
                optim_data_soc[
                    (start_plot - RBC_THRESHOLD) : (end_plot - RBC_THRESHOLD)
                ],
                label="Optim SOC",
            )
            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel(env_comp_item[key_i])
            if i == 0:
                axs[i, j].set_xlabel("Hour")
    plt.legend()
    fig.savefig(
        f"images/{env_comp_item[key_i]}_optim_env_normalized.pdf", bbox_inches="tight"
    )
    np.savetxt(
        f"images/{env_comp_item[key_i]}_soc_norm.csv",
        np.array(data_all).T,
        delimiter=",",
    )

# debug: capacity
env_comp_item = ["electrical_storage", "cooling_storage", "dhw_storage"]
data_comp_item2 = ["C_p_bat", "C_p_Csto", "C_p_Hsto"]
for key_i in range(len(env_comp_item)):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            bid = i * 3 + j
            optim_data_soc = check_params[data_comp_item2[key_i]][bid]
            data_env = np.array(
                getattr(
                    getattr(
                        env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]
                    ),
                    "capacity",
                )
            )
            axs[i, j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
            axs[i, j].plot(
                np.arange(1, 1 + end_plot - start_plot),
                data_env * np.ones(end_plot - start_plot),
                "-o",
                label="true capacity",
            )
            axs[i, j].plot(
                np.arange(end_plot - start_plot),
                optim_data_soc[
                    (start_plot - RBC_THRESHOLD) : (end_plot - RBC_THRESHOLD)
                ],
                label="Optim capacity",
            )

            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel(env_comp_item[key_i])
            if i == 0:
                axs[i, j].set_xlabel("Hour")
    plt.legend()
    fig.savefig(
        f"images/{env_comp_item[key_i]}_optim_env_capacity.pdf", bbox_inches="tight"
    )

# debug: E_grid prev hour
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        bid = i * 3 + j
        optim_data_ep = check_params["E_grid_prevhour"][bid]
        data_env = E_grid_true_arr[:, bid]
        axs[i, j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
        axs[i, j].plot(
            np.arange(1, 1 + end_plot - start_plot),
            data_env[start_plot:end_plot],
            "-o",
            label="true Egrid prev hour",
        )
        axs[i, j].plot(
            np.arange(end_plot - start_plot),
            optim_data_ep[(start_plot - RBC_THRESHOLD) : (end_plot - RBC_THRESHOLD)],
            label="Optim Egrid prev hour",
        )
        axs[i, j].plot(
            np.arange(end_plot - start_plot),
            data_env[start_plot:end_plot]
            - optim_data_ep[(start_plot - RBC_THRESHOLD) : (end_plot - RBC_THRESHOLD)],
            label="Diff",
        )

        axs[i, j].grid()
        if j == 0:
            axs[i, j].set_ylabel(env_comp_item[key_i])
        if i == 0:
            axs[i, j].set_xlabel("Hour")
plt.legend()
fig.savefig(f"images/E_gridprev_optim_env.pdf", bbox_inches="tight")


env_comp_item = ["non_shiftable_load", "dhw_demand", "cooling_demand", "solar_gen"]
data_comp_item = ["E_ns", "H_bd", "C_bd", "E_pv"]
for key_i in range(len(env_comp_item)):

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            data_all = []
            bid = i * 3 + j
            optim_data_soc = check_params[data_comp_item[key_i]][bid]
            data_env = np.array(
                env.buildings["Building_" + str(bid + 1)].sim_results[
                    env_comp_item[key_i]
                ]
            )
            data_all.append(data_env[RBC_THRESHOLD:end_time])
            data_all.append(optim_data_soc[: (end_time - RBC_THRESHOLD)])
            axs[i, j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
            axs[i, j].plot(
                np.arange(1, 1 + end_plot - start_plot),
                data_env[start_plot:end_plot],
                "-o",
                label="true value",
            )
            axs[i, j].plot(
                np.arange(end_plot - start_plot),
                optim_data_soc[
                    (start_plot - RBC_THRESHOLD) : (end_plot - RBC_THRESHOLD)
                ],
                label="Optim prediction",
            )

            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel(env_comp_item[key_i])
            if i == 0:
                axs[i, j].set_xlabel("Hour")
            np.savetxt(
                f"images/{env_comp_item[key_i]}_B{bid}.csv",
                np.array(data_all).T,
                delimiter=",",
            )

    plt.legend()
    fig.savefig(f"images/{env_comp_item[key_i]}_optim_env.pdf", bbox_inches="tight")


### Testing for action clipping
env_comp_item = ["electrical_storage", "cooling_storage", "dhw_storage"]
data_comp_item = ["action_bat", "action_C", "action_H"]
for key_i in range(len(env_comp_item)):

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            data_all = []
            bid = i * 3 + j
            optim_data_soc = check_data[data_comp_item[key_i]][bid]
            data_env = np.array(
                getattr(
                    getattr(
                        env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]
                    ),
                    "energy_balance",
                )
            ) / getattr(
                getattr(
                    env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]
                ),
                "capacity",
            )
            data_all.append(data_env[RBC_THRESHOLD:end_time])
            data_all.append(optim_data_soc[: (end_time - RBC_THRESHOLD)])
            axs[i, j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
            axs[i, j].plot(
                data_env[RBC_THRESHOLD:end_time]
                - optim_data_soc[: (end_time - RBC_THRESHOLD)],
                label="true action - planned action",
            )

            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel(env_comp_item[key_i])
            if i == 0:
                axs[i, j].set_xlabel("Hour")
            np.savetxt(
                f"images/{env_comp_item[key_i]}_B{bid}_diff_action.csv",
                np.array(data_all).T,
                delimiter=",",
            )

    plt.legend()
    fig.savefig(
        f"images/{env_comp_item[key_i]}_diff_action_optim_env.pdf", bbox_inches="tight"
    )

## Plot evaluations

vars_RL = agents.logger

time_RBC = int(RBC_THRESHOLD / 24)
time_sim = int(end_time / 24) - time_RBC  # Number of days simulation
time_end = time_sim + time_RBC

all_costs = agents.all_costs
all_costs = np.mean(all_costs, axis=2)


p_ele_data = np.array(agents.p_ele_logger)

# fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# for i in range(3):
#     for j in range(3):
#         bid = i * 3 + j
#         axs[i, j].set_title(f"Building {bid + 1}")
#         axs[i, j].imshow(
#             np.squeeze(p_ele_data[:, 0, :, bid]), cmap="viridis", aspect="auto"
#         )
#         if j == 0:
#             axs[i, j].set_ylabel("theta")
#         if i == 0:
#             axs[i, j].set_xlabel("Day")
# plt.legend()
# fig.savefig("images/theta_evolve.pdf", bbox_inches="tight")

##########################

vars_RL = agents.logger

E_grid_true = np.array(E_grid_true)
E_grid_RBC = np.array(E_grid_RBC)

# 1. RBC Agent
RBC_actions_arr = []
RBC_look_ahead_cost = []
RBC_E_grid_pred = []
RBC_E_grid_sell = []
RBC_E_hpC = []
RBC_E_ehH = []
RBC_Edhw = []
RBC_SOC_bat = []
RBC_SOC_C = []
RBC_SOC_H = []
RBC_C_p_bat = []
RBC_C_p_Csto = []
RBC_C_p_Hsto = []
RBC_ramping_cost = []
RBC_peak_electricity_cost = []
RBC_total_cost = []
RBC_action_C = []
RBC_action_H = []
RBC_action_bat = []

# 2. RL Agent
RL_actions_arr = []
RL_look_ahead_cost = []
RL_E_grid_sell = []
RL_E_grid_pred = []
RL_E_hpC = []
RL_E_ehH = []
RL_Edhw = []
RL_SOC_bat = []
RL_SOC_H = []
RL_SOC_C = []
RL_C_p_bat = []
RL_C_p_Csto = []
RL_C_p_Hsto = []
RL_ramping_cost = []
RL_peak_electricity_cost = []
RL_total_cost = []
RL_action_C = []
RL_action_H = []
RL_action_bat = []


for i in range(len(vars_RL)):  # number of days of RL/NORL
    for j in range(9):
        RL_E_grid_pred.append(vars_RL[i][j]["E_grid"])
        RL_E_grid_sell.append(vars_RL[i][j]["E_grid_sell"])
        RL_E_hpC.append(vars_RL[i][j]["E_hpC"])
        RL_E_ehH.append(vars_RL[i][j]["E_ehH"])
        RL_SOC_bat.append(vars_RL[i][j]["SOC_bat"])
        RL_SOC_H.append(vars_RL[i][j]["SOC_H"])
        RL_SOC_C.append(vars_RL[i][j]["SOC_C"])
        RL_action_bat.append(vars_RL[i][j]["action_bat"])
        RL_action_C.append(vars_RL[i][j]["action_C"])
        RL_action_H.append(vars_RL[i][j]["action_H"])


# ### flatten out to get hour per building

RL_E_grid_pred = (
    np.array(RL_E_grid_pred).flatten().reshape(-1, 9)
)  # hours x num_buildings
RL_E_grid_sell = (
    np.array(RL_E_grid_sell).flatten().reshape(-1, 9)
)  # hours x num_buildings

# RL_E_grid_pred = np.array(RL_E_grid_pred).flatten().reshape(len(vars_RL), 9) # hours x num_buildings
# RL_E_grid_sell = np.array(RL_E_grid_sell).flatten().reshape(len(vars_RL), 9) # hours x num_buildings
# RL_E_hpC = np.array(RL_E_hpC).flatten().reshape(len(vars_RL), 9) # hours x num_buildings
# RL_E_ehH = np.array(RL_E_ehH).flatten().reshape(len(vars_RL), 9) # hours x num_buildings
# RL_SOC_bat = np.array(RL_SOC_bat).flatten().reshape(len(vars_RL), 9) # hours x num_buildings
# RL_SOC_H = np.array(RL_SOC_H).flatten().reshape(len(vars_RL), 9) # hours x num_buildings
# RL_SOC_C = np.array(RL_SOC_C).flatten().reshape(len(vars_RL), 9) # hours x num_buildings
# RL_action_bat = np.array(RL_action_bat).flatten().reshape(len(vars_RL), 9) # hours x num_buildings
# RL_action_C = np.array(RL_action_C).flatten().reshape(len(vars_RL), 9) # hours x num_buildings
# RL_action_H = np.array(RL_action_H).flatten().reshape(len(vars_RL), 9) # hours x num_buildings

week = 24 * 5
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        bid = i * 3 + j
        axs[i, j].set_title(f"Building {bid + 1}")
        axs[i, j].plot(
            RL_E_grid_pred[:, bid][-week:] + RL_E_grid_sell[:, bid][-week:],
            label="CEM: E_grid",
        )
        axs[i, j].plot(
            E_grid_true[RBC_THRESHOLD:, bid][-week:], label="True E_grid"
        )  # plot true E grid
        axs[i, j].plot(
            E_grid_RBC[RBC_THRESHOLD:, bid][-week:], label="RBC: E_grid"
        )  # plot true E grid
        axs[i, j].grid()
        if j == 0:
            axs[i, j].set_ylabel("E grid")
        if i == 0:
            axs[i, j].set_xlabel("Hour")
plt.legend()
fig.savefig("images/E_grid_CEM.pdf", bbox_inches="tight")

## plotting the costs

all_costs = agents.all_costs
all_costs = np.mean(all_costs, axis=2)


time_RBC = int(RBC_THRESHOLD / 24)
time_sim = int(end_time / 24) - time_RBC  # Number of days simulation
time_end = int(end_time / 24)

ramping_cost_CEM = []
ramping_cost_RBC = []
peak_electricity_cost_CEM = []
peak_electricity_cost_RBC = []


for i in range(time_sim):
    ramping_cost_CEM_t = []
    ramping_cost_RBC_t = []
    peak_electricity_cost_CEM_t = []
    peak_electricity_cost_RBC_t = []
    RL_E_grid_pred_t = E_grid_true[
        (RBC_THRESHOLD + i * 24) : (RBC_THRESHOLD + (i + 1) * 24), :
    ]
    E_grid_RBC_t = E_grid_RBC[
        (RBC_THRESHOLD + i * 24) : (RBC_THRESHOLD + (i + 1) * 24), :
    ]
    for bid in range(9):
        CEM_E_grid_t = RL_E_grid_pred_t[:, bid]
        RBC_Egrid_t = E_grid_RBC_t[:, bid]
        ramping_cost_CEM_t.append(np.sum(np.abs(CEM_E_grid_t[1:] - CEM_E_grid_t[:-1])))
        ramping_cost_RBC_t.append(np.sum(np.abs(RBC_Egrid_t[1:] - RBC_Egrid_t[:-1])))
        peak_electricity_cost_CEM_t.append(np.max(CEM_E_grid_t))
        peak_electricity_cost_RBC_t.append(np.max(RBC_Egrid_t))
    ramping_cost_CEM.append(ramping_cost_CEM_t)
    ramping_cost_RBC.append(ramping_cost_RBC_t)
    peak_electricity_cost_CEM.append(peak_electricity_cost_CEM_t)
    peak_electricity_cost_RBC.append(peak_electricity_cost_RBC_t)


CEM_cost = {
    "ramping_cost": np.array(ramping_cost_CEM).T,
    "peak_electricity_cost": np.array(peak_electricity_cost_CEM).T,
    "total_cost": np.array(ramping_cost_CEM).T + np.array(peak_electricity_cost_CEM).T,
}
RBC_cost = {
    "ramping_cost": np.array(ramping_cost_RBC).T,
    "peak_electricity_cost": np.array(peak_electricity_cost_RBC).T,
    "total_cost": np.array(ramping_cost_RBC).T + np.array(peak_electricity_cost_RBC).T,
}

item_cost = ["ramping_cost", "peak_electricity_cost", "total_cost"]
for k in range(len(item_cost)):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            bid = i * 3 + j
            axs[i, j].set_title(f"Building {bid + 1}: {item_cost[k]}")
            axs[i, j].plot(
                CEM_cost[item_cost[k]][bid, :], label=f"CEM: {item_cost[k]}"
            )  # plot true E grid
            axs[i, j].plot(RBC_cost[item_cost[k]][bid, :], label=f"RBC: {item_cost[k]}")
            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel("Cost")
            if i == 0:
                axs[i, j].set_xlabel("Day")
            print(f"Mean {item_cost[k]} ratio for building {bid+1}")
            print(
                np.mean(
                    np.array(CEM_cost[item_cost[k]][bid, :])
                    / np.array(RBC_cost[item_cost[k]][bid, :])
                )
            )
    plt.legend()
    fig.savefig(f"images/{item_cost[k]}_compare.pdf", bbox_inches="tight")

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        bid = i * 3 + j
        axs[i, j].set_title(f"Building {bid + 1}: total cost CEM/RBC")
        axs[i, j].plot(
            CEM_cost["total_cost"][bid, :] / RBC_cost["total_cost"][bid, :],
            label=f"CEM/RBC",
        )  # plot true E grid
        axs[i, j].grid()
        if j == 0:
            axs[i, j].set_ylabel("Cost (Ratio)")
        if i == 0:
            axs[i, j].set_xlabel("Day")
plt.legend()
fig.savefig(f"images/cost_ratio_compare.pdf", bbox_inches="tight")
# print(all_costs)


# aggregate cost

ramping_cost_CEM_agg = []
ramping_cost_RBC_agg = []
peak_electricity_cost_CEM_agg = []
peak_electricity_cost_RBC_agg = []
tot_cost_ratio_agg = []

for i in range(time_sim):
    CEM_E_grid_t = np.sum(
        E_grid_true[(RBC_THRESHOLD + i * 24) : (RBC_THRESHOLD + (i + 1) * 24), :],
        axis=1,
    )
    RBC_Egrid_t = np.sum(
        E_grid_RBC[(RBC_THRESHOLD + i * 24) : (RBC_THRESHOLD + (i + 1) * 24), :], axis=1
    )
    ramping_cost_CEM_agg.append(np.sum(np.abs(CEM_E_grid_t[1:] - CEM_E_grid_t[:-1])))
    ramping_cost_RBC_agg.append(np.sum(np.abs(RBC_Egrid_t[1:] - RBC_Egrid_t[:-1])))
    peak_electricity_cost_CEM_agg.append(np.max(CEM_E_grid_t))
    peak_electricity_cost_RBC_agg.append(np.max(RBC_Egrid_t))
    tot_cost_ratio_agg.append(
        0.5
        * (
            np.max(CEM_E_grid_t) / np.max(RBC_Egrid_t)
            + np.sum(np.abs(CEM_E_grid_t[1:] - CEM_E_grid_t[:-1]))
            / np.sum(np.abs(RBC_Egrid_t[1:] - RBC_Egrid_t[:-1]))
        )
    )


fig, ax1 = plt.subplots()
ax1.set_title(f"Total cost CEM/RBC")
ax1.plot(np.array(tot_cost_ratio_agg), label=f"CEM/RBC ratios")  # plot true E grid
ax1.grid()
ax1.set_ylabel("Cost (Ratio)")
ax1.set_xlabel("Day")
plt.legend()
fig.savefig(f"images/cost_ratio_aggregate.pdf", bbox_inches="tight")

print("SCS counts")
print(agents.actor.scs_cnt)
print("Total fail counts")
print(agents.actor.fail_cnt)
print("Mean cost")
print(np.mean(np.array(tot_cost_ratio_agg)))
# This is the case without c_bat_end
# cost coefficients are 1e1
