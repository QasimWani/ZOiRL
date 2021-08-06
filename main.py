# Run this again after editing submodules so Colab uses the updated versions
from citylearn import CityLearn
from pathlib import Path
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import utils
import time
from copy import deepcopy
import torch

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
    "simulation_period": (0, 8760),
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
RBC_THRESHOLD = 24 * 14
end_time = 8760
t_idx = 0
costs_peak_net_ele = []
E_grid_true = []
start_time = time.time()

# Instantiating the control agent(s)
agents = Agent(**params_agent)

state = env.reset()
rbc_env = deepcopy(env)
done = False

action = agents.select_action(state)
# hour 1 - 24
while not done and env.time_step < end_time:
    next_state, reward, done, _ = env.step(action)
    action_next = agents.select_action(next_state)
    agents.add_to_buffer(state, action, reward, next_state, done)
    E_grid_true.append([x[28] for x in state])
    state = next_state
    action = action_next
    t_idx += 1
    print(f"\rTime step: {t_idx}", end="")

print(
    f"\nTotal time (min) to run {end_time // 24} days of simulation: {round((time.time() - start_time) / 60, 3)}"
)
env_cost = env.cost()

E_grid_RBC = utils.RBC(actions_spaces).get_rbc_data(rbc_env, state, end_time)
time_RBC = int(RBC_THRESHOLD / 24)
time_sim = int(end_time / 24) - time_RBC  # Number of days simulation
time_end = int(end_time / 24)

all_costs = agents.all_costs
all_costs = np.mean(all_costs, axis=2)

ramping_cost_CEM = []
ramping_cost_RBC = []
peak_electricity_cost_CEM = []
peak_electricity_cost_RBC = []

E_grid_true = np.array(E_grid_true)
E_grid_RBC = np.array(E_grid_RBC)

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
            print(f'Mean {item_cost[k]} ratio for building {bid+1}')
            print(np.mean(np.array(CEM_cost[item_cost[k]][bid, :])/np.array(RBC_cost[item_cost[k]][bid, :])))
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
print('Mean cost')
print(np.mean(np.array(tot_cost_ratio_agg)))
print('\n')
print(env_cost)