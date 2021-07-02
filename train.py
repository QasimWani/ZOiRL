# Run this again after editing submodules so Colab uses the updated versions
from citylearn import CityLearn
from pathlib import Path

from TD3 import TD3 as Agent

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
end_time = RBC_THRESHOLD + 24 * 30  # run for a month

start_time = time.time()

costs_peak_net_ele = []
while not done and t_idx <= end_time:

    next_state, reward, done, _ = env.step(action)
    action_next = agents.select_action(
        next_state, env
    )  # passing in environment for Oracle agent.

    # agents.action_RBC()
    # agents.action_noRL()

    # cost, e_grid, ... = agents.get_data()
    # cost, e_grid, ... = agents.RBC.get_data()
    # cost, e_grid, ... = agents.NORL.get_data()

    # agents.RBC.cost_dispatch[t]

    agents.add_to_buffer_oracle(env, action, reward)
    # agents.add_to_buffer(state, action, reward, next_state, done)
    state = next_state
    action = action_next

    t_idx += 1

    print(f"\rTime step: {t_idx}", end="")

print(f"Total time to run {end_time // 24} days: {time.time() - start_time}")
# env.cost()
