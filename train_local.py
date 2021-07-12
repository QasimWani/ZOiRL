# Run this again after editing submodules so Colab uses the updated versions
from citylearn import CityLearn
from pathlib import Path
from agents.rbc import RBC
from TD3 import TD3 as Agent
from copy import deepcopy
import sys
import warnings
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Load environment
climate_zone = 1
params = {
    "data_path": Path("data/Climate_Zone_" + str(climate_zone)),
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
# agents = Agent(**params_agent)
RBC_THRESHOLD = 48  # 2 weeks
agents = Agent(
    num_actions=actions_spaces,
    num_buildings=len(observations_spaces),
    env=env,
    rbc_threshold=RBC_THRESHOLD,
)

state = env.reset()
done = False

agents_rbc = RBC(actions_spaces)
RBC_Egrid = []
def get_idx_hour():
    # Finding which state
    with open("buildings_state_action_space.json") as file:
        actions_ = json.load(file)

    indx_hour = -1
    for obs_name, selected in list(actions_.values())[0]["states"].items():
        indx_hour += 1
        if obs_name == "hour":
            break
        assert (
            indx_hour < len(list(actions_.values())[0]["states"].items()) - 1
        ), "Please, select hour as a state for Building_1 to run the RBC"
    return indx_hour



E_grid = []
action = agents.select_action(state,env)
state_start = state

t_idx = 1
numdays = 20
end_time = RBC_THRESHOLD + 24 * numdays # run for a month

indx_hour = get_idx_hour()

def get_rbc_data(
    surrogate_env: CityLearn, state, indx_hour: int, dump_data: list, run_timesteps: int
):
    """Runs RBC for x number of timesteps"""
    ## --- RBC generation ---
    for i in range(run_timesteps):
        hour_state = np.array([[state[0][indx_hour]]])
        action = agents_rbc.select_action(
            hour_state
        )  # using RBC to select next action given current sate
        next_state, rewards, done, _ = surrogate_env.step(action)
        state = next_state
        dump_data.append([x[28] for x in state])


start_time = time.time()
cur_time = time.time()
costs_peak_net_ele = []
while not done and t_idx < end_time:
    print(f"\rTime step: {t_idx}", end="")
    next_state, reward, done, _ = env.step(action)
    action_next = agents.select_action(next_state, env)  # passing in environment for Oracle agent.

    agents.add_to_buffer_oracle(state, env, action, reward, next_state)
    # agents.add_to_buffer(state, action, reward, next_state, done)
    state = next_state
    action = action_next

    E_grid.append([x[28] for x in state])
    t_idx += 1
    print(f"Time in this step: {time.time() - cur_time}")
    cur_time = time.time()


next_state, reward, done, _ = env.step(action)
state = next_state
E_grid.append([x[28] for x in state])

print(f"Total time to run {end_time // 24} days: {time.time() - start_time}")
# env.cost()


get_rbc_data(deepcopy(env), state_start, indx_hour, RBC_Egrid, end_time)


# true E-grid values. NOTE: E_grid = E_grid_true. E_grid_pred = var["E_grid"] for RL/Optim
E_grid_true = np.array(E_grid).T

# E_grid net electricity consumption per building using RBC
RBC_Egrid = np.array(RBC_Egrid).T  # set per building


# plot E_grid for RL and RBC
plot_days = 5
week = end_time - 24 * plot_days # plots last week of the month data
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        bid = i * 3 + j
        axs[i, j].set_title(f"Building {bid + 1}")
        axs[i, j].plot(
            E_grid_true[bid][week:], label="True E grid: Optim"
        )  # plot true E grid
        axs[i, j].plot(
            RBC_Egrid[bid][week:], label="True E grid: RBC"
        )  # plot true E grid
        axs[i, j].grid()
        if j == 0:
            axs[i, j].set_ylabel("E grid")
        if i == 0:
            axs[i, j].set_xlabel("Hour")
plt.legend()
fig.savefig("images/Egrid_compare_RBC.pdf", bbox_inches="tight")


vars_RL = agents.logger

# # list of dictionary of variables generated NORL - Optim w/o any RL. See actor.py#L166 for relevant variable names. eg. vars_RL[0]['E_grid']
# vars_NORL = agents.norl_loggerd


#
# debug_item = [
#     "E_grid",
#     "E_bal_relax",
#     "H_bal_relax",
#     "C_bal_relax",
#     "E_grid_sell",
#     "E_hpC",
#     "E_ehH",
#     "SOC_bat",
#     "SOC_Brelax",
#     "action_bat",
#     "SOC_H",
#     "SOC_Hrelax",
#     "action_H",
#     "SOC_C",
#     "SOC_Crelax",
#     "action_C",
# ]
#
#
#
#
#
# check_data = {}
# for key in debug_item:
#     check_data[key] = [[] for i in range(9)]
#
# for i in range(24 * numdays):
#     optim_var = vars_RL[i]
#     for key in debug_item:
#         for bid in range(9):
#             check_data[key][bid].append(np.array(optim_var[bid][key]))
#
# check_data_param = {}
# data_param_keys = ["c_bat_init", "c_Csto_init", "c_Hsto_init", "C_p_bat", "C_p_Csto", "C_p_Hsto","E_grid","E_ns","H_bd","C_bd","E_pv","COP_C"]
# for key in data_param_keys:
#     if key in ["C_p_bat", "C_p_Csto", "C_p_Hsto"]:
#         check_data_param[key] = [[] for i in range(9)]
#     elif key in ["E_grid"]:
#         check_data_param[key] = [[] for i in range(9)]
#     else:
#         check_data_param[key] = [[] for i in range(9)]
#
# for key in data_param_keys:
#     for i in range(numdays*24):
#         for bid in range(9):
#             optim_var = agents.logger_data[i]
#             if key in ["C_p_bat", "C_p_Csto", "C_p_Hsto"]:
#                 check_data_param[key][bid].extend(np.array(optim_var[key][:,bid]))
#             elif key in ["E_grid","E_ns","H_bd","C_bd","E_pv","COP_C"]:
#                 check_data_param[key][bid].append(np.array(optim_var[key][:,bid]))
#             else:
#                 check_data_param[key][bid].append((optim_var[key][bid]))
#     check_data_param[key][bid] = np.array(check_data_param[key][bid])
#
#
# # check_params = {}
# # debug_params = ["E_ns", "H_bd", "C_bd"]
# # for key in debug_params:
# #     check_params[key] = [[] for i in range(9)]
#
#
#
#
# plot_days = 5
# env_comp_item = ["electrical_storage", "cooling_storage", "dhw_storage"]
# env_comp_item_check = ["action_bat", "action_C", "action_H"]
# env_comp_item_check2 = ["SOC_bat", "SOC_C", "SOC_H"]
# env_comp_item_check3 = ["C_p_bat", "C_p_Csto", "C_p_Hsto"]
# env_comp_item_check4 = ["c_bat_init", "c_Csto_init", "c_Hsto_init"]
#
# sthour = 24*plot_days
# week = end_time - 24 * plot_days
#
# for key_i in range(len(env_comp_item)):
#     fig, axs = plt.subplots(3, 3, figsize=(15, 15))
#     for i in range(3):
#         for j in range(3):
#             bid = i * 3 + j
#             data_env2 = np.array(
#                 getattr(getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]
#                     ),"soc"))
#             axs[i, j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
#             axs[i, j].scatter(np.arange(-1,plot_days*24-1),np.array(check_data_param[env_comp_item_check4[key_i]][bid][-sthour:])* float(check_data_param[env_comp_item_check3[key_i]][bid][0]),s=50,marker='x',
#                 label="optimization init SOC",
#             )
#             axs[i, j].plot(
#                 data_env2[-sthour:], label="environment SOC"
#             )  # plot true E grid
#
#             axs[i, j].grid()
#             if j == 0:
#                 axs[i, j].set_ylabel(env_comp_item_check2[key_i])
#             if i == 0:
#                 axs[i, j].set_xlabel("Hour")
#     plt.legend()
#     fig.savefig(
#         f"images/{env_comp_item_check[key_i]}_SOC_init_plot.pdf", bbox_inches="tight"
#     )
#
# ## plot actions comparison
#
# sthour = 24*plot_days
# true_action = {'bat':[[] for i in range(9)],'H':[[] for i in range(9)],'C':[[] for i in range(9)]}
# optim_action = {'bat':[[] for i in range(9)],'H':[[] for i in range(9)],'C':[[] for i in range(9)]}
# key_actions = ['bat','H','C']
# cap_params = ['C_p_bat','C_p_Hsto','C_p_Csto']
# cap_params2 = ["electrical_storage", "dhw_storage", "cooling_storage"]
# for key_i in range(3):
#     for bid in range(9):
#         data_env = np.array(
#             getattr(getattr(env.buildings["Building_" + str(bid + 1)], cap_params2[key_i]
#                             ), "energy_balance"))
#         true_action[key_actions[key_i]][bid] = data_env[RBC_THRESHOLD:(RBC_THRESHOLD+plot_days*24)]
#         cap_t = check_data_param[cap_params[key_i]][bid][0]
#         for t in range(plot_days*24):
#             vec = check_data[f'action_{key_actions[key_i]}'][bid][t]
#             optim_action[key_actions[key_i]][bid].append(vec[0]*cap_t)
#
#         optim_action[key_actions[key_i]][bid] = np.array(optim_action[key_actions[key_i]][bid])
#
#
# for key_i in range(3):
#     fig, axs = plt.subplots(3, 3, figsize=(15, 15))
#     for i in range(3):
#         for j in range(3):
#             bid = i * 3 + j
#             axs[i, j].set_title(f"Building {bid + 1}: {cap_params2[key_i]}")
#             axs[i, j].plot(
#                 optim_action[key_actions[key_i]][bid],
#                 label="optimization action",
#             )
#             axs[i, j].scatter(np.arange(plot_days*24),true_action[key_actions[key_i]][bid],s=50,marker='x',
#                 label="true action",
#             )
#
#             axs[i, j].grid()
#             if j == 0:
#                 axs[i, j].set_ylabel(key)
#             if i == 0:
#                 axs[i, j].set_xlabel("Hour")
#     plt.legend()
#     fig.savefig(
#         f"images/{cap_params2[key_i]}_action_plot.pdf", bbox_inches="tight"
#     )
#
#
# # Plot loads comparison
# keys_t1 = ["E_ns","H_bd","C_bd","E_pv"]
# keys_t2 = ["non_shiftable_load","dhw_demand","cooling_demand","solar_gen"]
#
# true_data = {"E_ns":[[] for _ in range(9)],
#              "H_bd":[[] for _ in range(9)],
#              "C_bd":[[] for _ in range(9)],
#              "E_pv":[[] for _ in range(9)]}
# for key_i in range(4):
#     for bid in range(9):
#         true_data[keys_t1[key_i]][bid] =env.buildings["Building_" + str(bid+1)].sim_results[keys_t2[key_i]][:24*plot_days]
#
# week = RBC_THRESHOLD  # plots last week of the month data
# for key_i in range(3):
#     fig, axs = plt.subplots(3, 3, figsize=(15, 15))
#     for i in range(3):
#         for j in range(3):
#             bid = i * 3 + j
#             axs[i, j].set_title(f"Building {bid + 1}")
#             true_data_vec = true_data[keys_t1[key_i]][bid][week:(week+24)]
#             for t in range(24):
#                 data_np = np.array(check_data_param[keys_t1[key_i]][bid][week+t-RBC_THRESHOLD])
#                 axs[i, j].plot(np.arange(t,24),
#                     true_data_vec[t:]-data_np[t:], label=f"err hour {t}"
#                 )  # plot true E grid
#             axs[i, j].grid()
#             if j == 0:
#                 axs[i, j].set_ylabel(keys_t1[key_i])
#             if i == 0:
#                 axs[i, j].set_xlabel("Hour")
#     plt.legend()
#     fig.savefig(f"images/err_{keys_t1[key_i]}_adaptive.pdf", bbox_inches="tight")
#
#
# # action compare
# for key_i in range(3):
#     fig, axs = plt.subplots(3, 3, figsize=(15, 15))
#     for i in range(3):
#         for j in range(3):
#             bid = i * 3 + j
#             axs[i, j].set_title(f"Building {bid + 1}: {cap_params2[key_i]}")
#             axs[i, j].plot(
#                 optim_action[key_actions[key_i]][bid],
#                 label="optimization action",
#             )
#             axs[i, j].scatter(np.arange(plot_days*24),true_action[key_actions[key_i]][bid],s=50,marker='x',
#                 label="true action",
#             )
#
#             axs[i, j].grid()
#             if j == 0:
#                 axs[i, j].set_ylabel(key)
#             if i == 0:
#                 axs[i, j].set_xlabel("Hour")
#     plt.legend()
#     fig.savefig(
#         f"images/{cap_params2[key_i]}_action_plot.pdf", bbox_inches="tight"
#     )
#
# sthour = 24*plot_days
# E_grid_max = [[] for i in range(9)]
# for bid in range(9):
#     for day_i in range(plot_days):
#         week_t = RBC_THRESHOLD+day_i*24
#         E_grid_max_t = [0]
#         for t in range(1,24):
#             E_grid_max_t.append(np.max(E_grid_true[bid,week_t:week_t+t]))
#         E_grid_max[bid].extend(np.array(E_grid_max_t))
#
# E_grid_max_opt = [[] for i in range(9)]
# for bid in range(9):
#     for day_i in range(plot_days):
#         week_t = day_i*24
#         E_grid_max_t = [0]
#         for t in range(1,24):
#             vec = check_data_param['E_grid'][bid][week_t+t][0:t]
#             E_grid_max_t.append(np.max(np.append(vec,0)))
#         E_grid_max_opt[bid].extend(np.array(E_grid_max_t))
#
# week = end_time - 24 * plot_days
# fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# for i in range(3):
#     for j in range(3):
#         bid = i * 3 + j
#         axs[i, j].set_title(f"Building {bid + 1}")
#         axs[i, j].plot(
#             E_grid_max[bid], label="True E grid max"
#         )  # plot true E grid
#         axs[i, j].scatter(np.arange(plot_days*24),E_grid_max_opt[bid], s=50,label="Optim E grid max"
#         )  # plot true E grid
#         axs[i, j].grid()
#         if j == 0:
#             axs[i, j].set_ylabel("E grid")
#         if i == 0:
#             axs[i, j].set_xlabel("Hour")
# plt.legend()
# fig.savefig("images/Egrid_max.pdf", bbox_inches="tight")
#
#
#
#
#
#
#
#
# # plot predicted E_grid and true E_grid
# week = RBC_THRESHOLD  # plots last week of the month data
# fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# for i in range(3):
#     for j in range(3):
#         bid = i * 3 + j
#         axs[i, j].set_title(f"Building {bid + 1}")
#
#         axs[i, j].scatter(np.arange(24),E_grid_true[bid][week:(week+24)],s=50,marker='x',c='red', label="True E grid: Optim")  # plot true E grid
#         for t in range(24):
#             data_np = np.array(check_data['E_grid'][bid][week+t-RBC_THRESHOLD]).T
#             data_np2 = np.array(check_data['E_grid_sell'][bid][week+t-RBC_THRESHOLD]).T
#             axs[i, j].plot(np.arange(t,24),
#                 data_np[:]+data_np2[:], label=f"Optim hour {t}"
#             )  # plot true E grid
#         axs[i, j].grid()
#         if j == 0:
#             axs[i, j].set_ylabel("E grid")
#         if i == 0:
#             axs[i, j].set_xlabel("Hour")
# plt.legend()
# fig.savefig("images/Egrid_compare_adaptive.pdf", bbox_inches="tight")
#


# Compare the ramping, peak electricity costs
eval_days = 20
week = end_time - 24 * eval_days  # plots last week of the month data
sthour = 24*eval_days
ramping_cost_optim = []
ramping_cost_RBC = []
peak_electricity_cost_optim = []
peak_electricity_cost_RBC = []

for i in range(eval_days):
    t_start = week + i * 24
    t_end = week + (i + 1) * 24
    ramping_cost_optim_t = []
    ramping_cost_RBC_t = []
    peak_electricity_cost_optim_t = []
    peak_electricity_cost_RBC_t = []
    for bid in range(9):
        E_grid_t = E_grid_true[bid][t_start:t_end]
        RBC_Egrid_t = RBC_Egrid[bid][t_start:t_end]
        ramping_cost_optim_t.append(np.sum(np.abs(E_grid_t[1:] - E_grid_t[:-1])))
        ramping_cost_RBC_t.append(np.sum(np.abs(RBC_Egrid_t[1:] - RBC_Egrid_t[:-1])))
        peak_electricity_cost_optim_t.append(np.max(E_grid_t))
        peak_electricity_cost_RBC_t.append(np.max(RBC_Egrid_t))
    ramping_cost_optim.append(ramping_cost_optim_t)
    ramping_cost_RBC.append(ramping_cost_RBC_t)
    peak_electricity_cost_optim.append(peak_electricity_cost_optim_t)
    peak_electricity_cost_RBC.append(peak_electricity_cost_RBC_t)

Optim_cost = {
    "ramping_cost": np.array(ramping_cost_optim).T,
    "peak_electricity_cost": np.array(peak_electricity_cost_optim).T,
    "total_cost": np.array(ramping_cost_optim).T
    + np.array(peak_electricity_cost_optim).T,
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
                Optim_cost[item_cost[k]][bid, :], label=f"Optim: {item_cost[k]}"
            )  # plot true E grid
            axs[i, j].plot(RBC_cost[item_cost[k]][bid, :], label=f"RBC: {item_cost[k]}")
            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel("Cost")
            if i == 0:
                axs[i, j].set_xlabel("Day")
    plt.legend()
    fig.savefig(f"images/{item_cost[k]}_compare.pdf", bbox_inches="tight")

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        bid = i * 3 + j
        axs[i, j].set_title(f"Building {bid + 1}: total cost Optim/RBC")
        axs[i, j].plot(
            Optim_cost["total_cost"][bid, :] / RBC_cost["total_cost"][bid, :],
            label=f"Optim/RBC",
        )  # plot true E grid
        axs[i, j].grid()
        if j == 0:
            axs[i, j].set_ylabel("Cost (Ratio)")
        if i == 0:
            axs[i, j].set_xlabel("Day")
plt.legend()
fig.savefig(f"images/total_cost_ratio.pdf", bbox_inches="tight")

fig1, ax1 = plt.subplots()
ax1.set_title(f"Total cost community Optim/RBC")
ax1.plot(np.sum(Optim_cost["total_cost"],axis=0) / np.sum(RBC_cost["total_cost"],axis=0),
            label=f"Optim/RBC")  # plot true E grid
ax1.grid()
ax1.set_ylabel("Cost (Ratio)")
ax1.set_xlabel("Day")
plt.legend()
fig1.savefig(f"images/total_all_cost_ratio.pdf", bbox_inches="tight")
