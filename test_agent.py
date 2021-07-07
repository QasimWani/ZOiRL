import sys

'''
Please use python3+ THIS HASN'T BEEN TESTED w/ python <= 2.7

Colab/Notebook : get python version

>>> from platform import python_version
>>> print(python_version())
'''


### testing package installation
try:

    import numpy as np
    import pandas as pd
    import json

    import torch

    from copy import Error, deepcopy
    from collections import defaultdict

    import time
    import warnings
    from pathlib import Path

    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer  ## COMMENT THIS IF NEEDED

    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression, QuantileRegressor    # for prediction/estimation

except ImportError as error:
    print(error.__class__.__name__ + ": " + error.message)


### testing RBC agent running
class RBC:
    def __init__(self, actions_spaces):
        """Rule based controller. Source: https://github.com/QasimWani/CityLearn/blob/master/agents/rbc.py"""
        self.actions_spaces = actions_spaces

    def select_action(self, states):
        hour_day = states[0][0]
        multiplier = 0.4
        # Daytime: release stored energy  2*0.08 + 0.1*7 + 0.09
        a = [
            [0.0 for _ in range(len(self.actions_spaces[i].sample()))]
            for i in range(len(self.actions_spaces))
        ]
        if hour_day >= 7 and hour_day <= 11:
            a = [
                [
                    -0.05 * multiplier
                    for _ in range(len(self.actions_spaces[i].sample()))
                ]
                for i in range(len(self.actions_spaces))
            ]
        elif hour_day >= 12 and hour_day <= 15:
            a = [
                [
                    -0.05 * multiplier
                    for _ in range(len(self.actions_spaces[i].sample()))
                ]
                for i in range(len(self.actions_spaces))
            ]
        elif hour_day >= 16 and hour_day <= 18:
            a = [
                [
                    -0.11 * multiplier
                    for _ in range(len(self.actions_spaces[i].sample()))
                ]
                for i in range(len(self.actions_spaces))
            ]
        elif hour_day >= 19 and hour_day <= 22:
            a = [
                [
                    -0.06 * multiplier
                    for _ in range(len(self.actions_spaces[i].sample()))
                ]
                for i in range(len(self.actions_spaces))
            ]

        # Early nightime: store DHW and/or cooling energy
        if hour_day >= 23 and hour_day <= 24:
            a = [
                [
                    0.085 * multiplier
                    for _ in range(len(self.actions_spaces[i].sample()))
                ]
                for i in range(len(self.actions_spaces))
            ]
        elif hour_day >= 1 and hour_day <= 6:
            a = [
                [
                    0.1383 * multiplier
                    for _ in range(len(self.actions_spaces[i].sample()))
                ]
                for i in range(len(self.actions_spaces))
            ]

        return np.array(a, dtype="object")


## RUN simulation
from citylearn import CityLearn  # assuming curr dir is in access to citylearn


### TEST ALL ###
if __name__ == "__main__":

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

    # Instantiating the control agent(s)
    agents = RBC(actions_spaces)

    state = env.reset()
    done = False

    action = agents.select_action(state)

    while not done:
        next_state, reward, done, _ = env.step(action)
        action_next = agents.select_action(next_state)
        state = next_state
        action = action_next

    env.cost()  # Comment it out if needed, this is guaranteed to work

    print("NO ERRORS!")
