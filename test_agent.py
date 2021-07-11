import sys

"""
Please use python3+ THIS HASN'T BEEN TESTED w/ python <= 2.7

Colab/Notebook : get python version

>>> from platform import python_version
>>> print(python_version())
"""

try:
    from platform import python_version

    print("Python Version", python_version())
except:
    print("Unable to get python version!")

### testing package installation
try:
    import numpy as np

    print("numpy version", np.__version__)
except ImportError as error:
    print(error.__class__.__name__ + ": " + error.msg)

try:
    import pandas as pd

    print("pandas version", pd.__version__)
except ImportError as error:
    print(error.__class__.__name__ + ": " + error.msg)

try:
    import json

    print("json version", json.__version__)
except ImportError as error:
    print(error.__class__.__name__ + ": " + error.msg)

try:
    import torch

    print("pytorch version", torch.__version__)
except ImportError as error:
    print(error.__class__.__name__ + ": " + error.msg)

try:
    import copy
    import collections
    import time
    import warnings
    import pathlib

except ImportError as error:
    print(error.__class__.__name__ + ": " + error.msg)

try:
    import cvxpy as cp

    print("cvxpy version", cp.__version__)
except ImportError as error:
    print(error.__class__.__name__ + ": " + error.msg)
    # from cvxpylayers.torch import CvxpyLayer  ## COMMENT THIS IF NEEDED

try:
    import sklearn

    print("sklearn version", sklearn.__version__)
except ImportError as error:
    print(error.__class__.__name__ + ": " + error.msg)

try:
    import statsmodels

    print("statsmodels version", statsmodels.__version__)
except ImportError as error:
    print(error.__class__.__name__ + ": " + error.msg)

### testing RBC agent running
class Agent:
    def __init__(self, **parameters):
        """Rule based controller. Source: https://github.com/QasimWani/CityLearn/blob/master/agents/rbc.py"""
        self.actions_spaces = parameters["action_spaces"]

    def add_to_buffer(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        coordination_vars,
        coordination_vars_next,
    ):
        pass

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

        return np.array(a, dtype="object"), None
