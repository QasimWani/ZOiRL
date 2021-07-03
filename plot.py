# Run this again after editing submodules so Colab uses the updated versions
from copy import deepcopy
import numpy as np
from citylearn import CityLearn
from pathlib import Path

from TD3 import TD3 as Agent
import utils

import sys
import warnings
import time

if not sys.warnoptions:
    warnings.simplefilter("ignore")

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


params_agent = {
    "building_ids": ["Building_" + str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]],
    "buildings_states_actions": "buildings_state_action_space.json",
    "building_info": building_info,
    "observation_spaces": observations_spaces,
    "action_spaces": actions_spaces,
}

# Instantiating the control agent(s)
# agents = Agent(**params_agent)
RBC_THRESHOLD = 336  # 2 weeks
agents = Agent(
    num_actions=actions_spaces,
    num_buildings=len(observations_spaces),
    env=env,
    rbc_threshold=RBC_THRESHOLD,
)

state = env.reset()
done = False

action = agents.select_action(state)

t_idx = 0
# run for a month - NOTE: THIS WILL TAKE ~2 HOURS TO RUN. reduce `end_time` for quicker results.
end_time = RBC_THRESHOLD + 24 * 30

start_time = time.time()

costs_peak_net_ele = []

# returns E_grid for RBC agent
E_grid_RBC = utils.RBC(actions_spaces).get_rbc_data(
    deepcopy(env), state, utils.get_idx_hour(), end_time
)

E_grid_true = []  # see comments below for more info.

while not done and t_idx <= end_time:

    ## add env E-grid
    E_grid_true.append([x[28] for x in state])

    next_state, reward, done, _ = env.step(action)
    action_next = agents.select_action(
        next_state, env
    )  # passing in environment for Oracle agent.

    agents.add_to_buffer_oracle(env, action, reward)
    # agents.add_to_buffer(state, action, reward, next_state, done)
    state = next_state
    action = action_next

    t_idx += 1

    print(f"\rTime step: {t_idx}", end="")

print(
    f"Total time (min) to run {end_time // 24} days of simulation: {round((time.time() - start_time) / 60, 3)}"
)


############## TODO: ENTER CODE HERE ##############

# list of dictionary of variables generated from RL. See actor.py#L166 for relevant variable names. eg. vars_RL[0]['E_grid']
# NOTE: dimensions of RL/NORL logger is ≠ RBC. This is because RL/NORL implicitly uses RBC. We only start collecting data from
# RL/NORL once it starts making inference, i.e, after `rbc_threshold`. Tweak the parameter above!

vars_RL = agents.logger

# list of dictionary of variables generated NORL - Optim w/o any RL. See actor.py#L166 for relevant variable names. eg. vars_RL[0]['E_grid']
vars_NORL = agents.norl_logger

# true E-grid values. NOTE: E_grid = E_grid_true. E_grid_pred = var["E_grid"] for RL/Optim
E_grid_true = np.array(E_grid_true)

# E_grid net electricity consumption per building using RBC
E_grid_RBC = np.array(E_grid_RBC)

############## TODO: ENTER CODE HERE ##############


# vars_A, vars_B, ..., vars_Z = [], [], ..., [] # RL
# vars_A, vars_B, ..., vars_Z = [], [], ..., [] # NORL
# # keys = list(...) # list of all keys from actor.py#166 (see ref.)
# for i in range(len(vars_RL)): #number of days of RL/NORL
#     vars_A.append(vars_RL[i]["key name"])
#     vars_B.append(vars_RL[i]["key name"])
#     ...
#     vars_Z.append(vars_RL[i]["key name"])

# ### flatten out to get hour per building

# vars_A = np.array(vars_A).flatten().reshape(len(vars_RL) * 24, 9) # hours x num_buildings
# vars_B = np.array(vars_B).flatten().reshape(len(vars_RL) * 24, 9) # hours x num_buildings
# ...
# vars_Z = np.array(vars_Z).flatten().reshape(len(vars_RL) * 24, 9) # hours x num_buildings

### plot
# plt.plot(vars_A_RL[:, 0])
# plt.plot(vars_A_NORL[:, 0])
# plt.plot(vars_A_RBC[:, 0])
