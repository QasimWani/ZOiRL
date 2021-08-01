# Run this again after editing submodules so Colab uses the updated versions

import numpy as np
from citylearn import CityLearn
from pathlib import Path
from agents.rbc import RBC

from agent import Agent
from copy import deepcopy
import sys
import warnings
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from digital_twin import DigitalTwin

# Load environment
climate_zone = 5
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

RBC_THRESHOLD = 336
end_time = RBC_THRESHOLD + 24 * 10  # run for a month

# Instantiating the control agent(s)
# agents = Agent(**params_agent)

state = env.reset()
state_ori = deepcopy(state)
done = False


params_agent = {'building_ids': ["Building_" + str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]],
                'buildings_states_actions': 'buildings_state_action_space.json',
                'building_info': building_info,
                'observation_space': observations_spaces,
                'action_spaces': actions_spaces,
                'env': env}

# Instantiating the control agent(s)
agents = Agent(**params_agent)


action = agents.select_action_debug(state)
E_grid = []


t_idx = 0
start_time = time.time()

while not done and env.time_step < end_time:
    E_grid.append([x[28] for x in state])
    next_state, reward, done, _ = env.step(action)
    action_next = agents.select_action(
        next_state, False
    )  # passing in environment for Oracle agent.

    #     agents.add_to_buffer_oracle(state, env, action, reward, next_state)
    agents.add_to_buffer(state, action, reward, next_state, done)
    ## add env E-grid
    if t_idx >= RBC_THRESHOLD + 48:
        x = 1
    elif t_idx >= RBC_THRESHOLD + 24:
        x=1
    state = next_state
    action = action_next

    t_idx += 1
    if t_idx % 20 == 0:
        print(f"\rTime step: {t_idx}", end="")


print(f"Total time to run {end_time // 24} days: {time.time() - start_time}")
# env.cost()

E_grid_true = np.array(E_grid).T
E_grid_dt = np.array(agents.E_grid_dt).T

# plot E_grid for RL and RBC
week = end_time - 24 * 10  # plots last week of the month data
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        bid = i * 3 + j
        axs[i, j].set_title(f"Building {bid + 1}")
        axs[i, j].plot(E_grid_true[bid][week:], label="True E grid")  # plot true E grid
        axs[i, j].plot(E_grid_dt[bid][week:], label="Digital twin Egrid")  # plot true E grid
        axs[i, j].grid()
        if j == 0:
            axs[i, j].set_ylabel("E grid")
        if i == 0:
            axs[i, j].set_xlabel("Hour")
plt.legend()
fig.savefig("images/Egrid_compare_DT.pdf", bbox_inches="tight")

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        bid = i * 3 + j
        axs[i, j].set_title(f"Building {bid + 1}")
        axs[i, j].plot(E_grid_true[bid][week:]-E_grid_dt[bid][week:], label="True - DT E grid")  # plot true E grid
        axs[i, j].grid()
        if j == 0:
            axs[i, j].set_ylabel("E grid")
        if i == 0:
            axs[i, j].set_xlabel("Hour")
plt.legend()
fig.savefig("images/Egrid_compare_DT_diff.pdf", bbox_inches="tight")








############### Debug storages ###############
# unnormalized SOCs
env_comp_item = ["electrical_storage", "cooling_storage", "dhw_storage"]
week = end_time - 24 * 3  # plots last week of the month data
for key_i in range(len(env_comp_item)):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            bid = i * 3 + j
            building_data = [agents.dt_building_logger[tidx]["Building_" + str(bid + 1)] for tidx in range(week,len(agents.dt_building_logger))]
            building_data_eb = np.array([getattr(getattr(building_data[tidx], env_comp_item[key_i]),"_energy_balance") for tidx in range(len(building_data))])
            building_data_soc = np.array([getattr(getattr(building_data[tidx], env_comp_item[key_i]), "_soc") for tidx in range(len(building_data))])
            data_env = np.array(getattr(getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]),"energy_balance",))
            data_env2 = np.array(getattr(getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]),"soc",))
            axs[i, j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
            axs[i, j].plot(data_env[week:],'-o', label="true energy balance")
            axs[i, j].plot(data_env2[week:],'-o', label="true SOC")
            axs[i, j].plot(building_data_eb, label="dt energy balance")
            axs[i, j].plot(building_data_soc, label="dt SOC")

            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel(env_comp_item[key_i])
            if i == 0:
                axs[i, j].set_xlabel("Hour")
    plt.legend()
    fig.savefig(f"images/{env_comp_item[key_i]}_optim_env_plot.pdf", bbox_inches="tight")


# normalized SOCs
env_comp_item = ["electrical_storage", "cooling_storage", "dhw_storage"]
week = end_time - 24 * 3  # plots last week of the month data
soc_logger = {"electrical_storage":agents.e_soc_logger,"cooling_storage":agents.c_soc_logger,"dhw_storage":agents.h_soc_logger}
for key_i in range(len(env_comp_item)):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    soc_logger_all = np.array([soc_logger[env_comp_item[key_i]][tidx] for tidx in range(week, len(agents.dt_building_logger))]).T
    for i in range(3):
        for j in range(3):
            bid = i * 3 + j
            building_data = [agents.dt_building_logger[tidx]["Building_" + str(bid + 1)] for tidx in range(week, len(agents.dt_building_logger))]
            building_data_eb = np.array([getattr(getattr(building_data[tidx], env_comp_item[key_i]), "_energy_balance") for tidx in range(len(building_data))])/getattr(getattr(building_data[0], env_comp_item[key_i]), "capacity")
            building_data_soc = np.array([getattr(getattr(building_data[tidx], env_comp_item[key_i]), "_soc") for tidx in range(len(building_data))])/getattr(getattr(building_data[0], env_comp_item[key_i]), "capacity")
            soc_logger_t = soc_logger_all[bid,1:]

            data_env = np.array(getattr(getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]),"energy_balance",))/getattr(getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]),"capacity",)
            data_env2 = np.array(getattr(getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]),"soc",))/getattr(getattr(env.buildings["Building_" + str(bid + 1)], env_comp_item[key_i]),"capacity",)
            axs[i, j].set_title(f"Building {bid + 1}: {env_comp_item[key_i]}")
            axs[i, j].plot(data_env[week:],'-o', label="true energy balance (normalized)")
            axs[i, j].plot(data_env2[week:],'-o', label="true SOC (normalized)")
            axs[i, j].plot(building_data_eb, label="dt energy balance (normalized)")
            axs[i, j].plot(building_data_soc, label="dt SOC (normalized)")
            axs[i, j].plot(np.arange(len(soc_logger_t)),soc_logger_t,'-x', label="dt logged SOC (normalized)")

            axs[i, j].grid()
            if j == 0:
                axs[i, j].set_ylabel(env_comp_item[key_i])
            if i == 0:
                axs[i, j].set_xlabel("Hour")
    plt.legend()
    fig.savefig(f"images/{env_comp_item[key_i]}_optim_env_normalized_plot.pdf", bbox_inches="tight")

