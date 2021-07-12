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
action = agents.select_action(state,env,True)
state_start = state
t_idx = 1
numdays = 20
end_time = RBC_THRESHOLD + 24 * numdays  # run for a month

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
    action_next = agents.select_action(next_state, env,True)  # passing in environment for Oracle agent.

    agents.add_to_buffer_oracle(state, env, action, reward)
    # agents.add_to_buffer(state, action, reward, next_state, done)
    state = next_state
    action = np.array(action_next)

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
week = end_time - 24 * 10  # plots last week of the month data
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

check_data = {}
check_data_param = {}

for key in debug_item:
    check_data[key] = [[] for i in range(9)]

data_param_keys = ["c_bat_init", "c_Csto_init", "c_Hsto_init", "C_p_bat", "C_p_Csto", "C_p_Hsto"]
for key in data_param_keys:
    if key in ["C_p_bat", "C_p_Csto", "C_p_Hsto"]:
        check_data_param[key] = [[] for i in range(9)]
    else:
        check_data_param[key] = [[] for i in range(9)]
check_params = {}
debug_params = ["E_ns", "H_bd", "C_bd"]
for key in debug_params:
    check_params[key] = [[] for i in range(9)]


# collect all data
start_time = end_time - 24 * numdays
for bid in range(9):
    for key in debug_item:
        for i in range(len(vars_RL)):
            optim_var = vars_RL[i]
            check_data[key][bid].extend(np.array(optim_var[bid][key]))
        check_data[key][bid] = np.array(check_data[key][bid])

for key in data_param_keys:
    for i in range(len(vars_RL)):
        for bid in range(9):
            optim_var = agents.logger_data[i]
            if key in ["C_p_bat", "C_p_Csto", "C_p_Hsto"]:
                check_data_param[key][bid].extend(np.array(optim_var[key][:,bid]))
            else:
                check_data_param[key][bid].append((optim_var[key][bid]))
    check_data_param[key][bid] = np.array(check_data_param[key][bid])

E_grid_pred = [[] for i in range(9)]
E_grid_sell_pred = [[] for i in range(9)]
for bid in range(9):
    E_grid_pred[bid] = check_data["E_grid"][bid]
    E_grid_sell_pred[bid] = check_data["E_grid_sell"][bid]
# plot predicted E_grid
sthour = 24*5  # plots last week of the month data
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        bid = i * 3 + j
        axs[i, j].set_title(f"Building {bid + 1}")
        axs[i, j].plot(
            E_grid_true[bid][-sthour:], label="True E grid: Optim"
        )  # plot true E grid
        axs[i, j].plot(
            E_grid_pred[bid][-sthour:] + E_grid_sell_pred[bid][-sthour:],
            "gx",
            label="Optim predicted E grid",
        )  # plots per month
        axs[i, j].plot(
            RBC_Egrid[bid][-sthour:], label="True E grid: RBC"
        )  # plot true E grid
        axs[i, j].grid()
        if j == 0:
            axs[i, j].set_ylabel("E grid")
        if i == 0:
            axs[i, j].set_xlabel("Hour")
plt.legend()
fig.savefig("images/Egrid_compare.pdf", bbox_inches="tight")

# plot optimization variables
for key in debug_item:
    data_np = check_data[key]
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            bid = i * 3 + j
            axs[i, j].set_title(f"Building {bid + 1}: {key}")
            axs[i, j].plot(data_np[bid][-sthour:], label=key)  # plot true E grid
            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel(key)
            if i == 0:
                axs[i, j].set_xlabel("Hour")
    plt.legend()
    fig.savefig(f"images/{key}_plot.pdf", bbox_inches="tight")

# Plot variables Edhw, Ehp

env_comp_item = ["electric_consumption_cooling", "electric_consumption_dhw"]
env_comp_item_check = ["E_hpC", "E_ehH"]
sthour = 24*7
for key_i in range(len(env_comp_item)):
    data_np = np.array(check_data[env_comp_item_check[key_i]])

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            bid = i * 3 + j
            data_env = np.array(
                getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i])
            )
            axs[i, j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
            axs[i, j].plot(
                data_np[bid][-sthour:], label="optimization"
            )  # plot true E grid
            axs[i, j].plot(data_env[-sthour:], label="environment")  # plot true E grid
            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel(env_comp_item_check[key_i])
            if i == 0:
                axs[i, j].set_xlabel("Hour")
    plt.legend()
    fig.savefig(
        f"images/{env_comp_item_check[key_i]}_optim_env_plot.pdf", bbox_inches="tight"
    )

# Plot energy balance

plot_days = 5
env_comp_item = ["electrical_storage", "cooling_storage", "dhw_storage"]
env_comp_item_check = ["action_bat", "action_C", "action_H"]
env_comp_item_check2 = ["SOC_bat", "SOC_C", "SOC_H"]
env_comp_item_check3 = ["C_p_bat", "C_p_Csto", "C_p_Hsto"]
env_comp_item_check4 = ["c_bat_init", "c_Csto_init", "c_Hsto_init"]

sthour = 24*plot_days
week = end_time - 24 * plot_days

for key_i in range(len(env_comp_item)):
    data_np = np.array(check_data[env_comp_item_check[key_i]])
    data_np2 = np.array(check_data[env_comp_item_check2[key_i]])
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            bid = i * 3 + j
            data_env = np.array(
                getattr(getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]
                    ), "energy_balance"))
            data_env2 = np.array(
                getattr(getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]
                    ),"soc"))
            axs[i, j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
            axs[i, j].plot(
                data_np[bid][-sthour:] * check_data_param[env_comp_item_check3[key_i]][bid][0],
                label="optimization",
            )  # plot true E grid
            axs[i, j].plot(
                data_np2[bid][-sthour:] * check_data_param[env_comp_item_check3[key_i]][bid][0],
                label="optimization SOC",
            )
            axs[i, j].scatter(np.arange(-1,plot_days*24-1,24),np.array(check_data_param[env_comp_item_check4[key_i]][bid][-plot_days:])* float(check_data_param[env_comp_item_check3[key_i]][bid][0]),s=50,marker='x',
                label="optimization init SOC",
            )
            axs[i, j].scatter(np.arange(0, plot_days * 24, 24),
                              (np.array(check_data_param[env_comp_item_check4[key_i]][bid][-plot_days:])+data_np[bid][np.array(range(week-RBC_THRESHOLD,end_time-RBC_THRESHOLD,24))]) * float(
                                  check_data_param[env_comp_item_check3[key_i]][bid][0]), s=50,marker='x',
                              label="optimization SOC t=1",
                              )
            axs[i, j].plot(data_env[-sthour:], label="environment")  # plot true E grid
            axs[i, j].plot(
                data_env2[-sthour:], label="environment SOC"
            )  # plot true E grid

            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel(key)
            if i == 0:
                axs[i, j].set_xlabel("Hour")
    plt.legend()
    fig.savefig(
        f"images/{env_comp_item_check[key_i]}_optim_env_plot.pdf", bbox_inches="tight"
    )


for key_i in range(len(env_comp_item)):
    data_np = np.array(check_data[env_comp_item_check[key_i]])
    data_np2 = np.array(check_data[env_comp_item_check2[key_i]])
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            bid = i * 3 + j
            data_env = np.array(
                getattr(getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]
                    ), "energy_balance"))
            data_env2 = np.array(
                getattr(getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]
                    ),"soc"))
            axs[i, j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
            axs[i, j].plot(
                data_np2[bid][-sthour:] * check_data_param[env_comp_item_check3[key_i]][bid][0]-data_env2[-sthour:],
                label="optim-environ SOC",
            )

            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel(key)
            if i == 0:
                axs[i, j].set_xlabel("Hour")
    plt.legend()
    fig.savefig(
        f"images/{env_comp_item_check[key_i]}_optim-diff-env_plot.pdf", bbox_inches="tight"
    )

# Get actions
# actions_arr = np.array(actions_arr)
#
# list_actions = ["action_C", "action_H", "action_bat"]
# week = end_time - 24 * 3  # plots last week of the month data
# fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# for i in range(3):
#     for j in range(3):
#         bid = i * 3 + j
#         axs[i, j].set_title(f"Building {bid + 1}")
#         for k in range(3):
#             axs[i, j].plot(actions_arr[week:, bid, k], label=list_actions[k])
#         axs[i, j].grid()
#         if j == 0:
#             axs[i, j].set_ylabel("Actions")
#         if i == 0:
#             axs[i, j].set_xlabel("Hour")
# plt.legend()
# fig.savefig("images/Actions_compare.pdf", bbox_inches="tight")

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
