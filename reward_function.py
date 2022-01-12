"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified at will. This reward_function takes all the electrical demands and carbon intensity of all the buildings and turns them into one or multiple rewards for the agent(s)
"""
import numpy as np


class ElectricityRewards:
    def __init__(self) -> None:
        self.E_grid: list = []  # E_grid's previous day
        self.max_E_grid = None  # max previous day
        self.E_grid_prevhour = None  # E_grid previous hour
        self.counter: int = 0  # timestep iterator

    def eod_update(self, demand):
        """Assign E_grid_prevhour and max_E_grid values at start of new day"""
        self.E_grid_prevhour = self.E_grid[-1]
        self.max_E_grid = np.max(self.E_grid, axis=0)  # max from previous day
        self.E_grid.clear()

        self.E_grid.append(demand)

        self._invoke_counter()

    def daily_update(self, demand):
        """Append hourly electricity demand"""
        assert (
            len(self.E_grid) < 24
        ), f"REWARD: E_grid should be less than 24. Got: {len(self.E_grid)}"

        self.E_grid_prevhour = self.E_grid[-1]
        self.E_grid.append(demand)

        self._invoke_counter()

    def _invoke_counter(self):
        self.counter += 1


# Reward used in the CityLearn Challenge. Reward function for the multi-agent (decentralized) agents.
class reward_function_ma:
    def __init__(self, n_agents, building_info):
        self.n_agents = n_agents
        self.building_info = building_info
        self.elec = ElectricityRewards()

    # electricity_demand contains negative values when the building consumes more electricity than it generates
    def get_rewards(self, electricity_demand, carbon_intensity):

        # You can edit what comes next and customize it for The CityLearn Challenge
        electricity_demand = np.float32(electricity_demand)
        total_electricity_demand = 0
        for e in electricity_demand:
            total_electricity_demand += -e

        electricity_demand = np.array(electricity_demand)

        using_marlisa = False
        # Use this reward function when running the MARLISA example with information_sharing = True. The reward sent to each agent will have an individual and a collective component.
        if using_marlisa:
            return list(
                np.sign(electricity_demand)
                * 0.01
                * (
                    np.array(np.abs(electricity_demand)) ** 2
                    * max(0, total_electricity_demand)
                )
            )

        else:

            # Use this reward when running the SAC example. It assumes that the building-agents act independently of each other, without sharing information through the reward.
            if self.elec.counter < 24:
                reward_ = np.array(electricity_demand) ** 3.0
                reward_[reward_ > 0] = 0

                self.elec.E_grid.append(electricity_demand)
                self.elec.counter += 1

                return list(reward_)

            if self.elec.counter % 24 == 0:  # EOD
                self.elec.eod_update(electricity_demand)
            else:
                self.elec.daily_update(electricity_demand)

            # calculate reward
            reward_ = list(
                -np.abs(
                    np.array(self.elec.E_grid_prevhour) - np.array(electricity_demand)
                )
                - 5 * np.max(self.elec.max_E_grid, axis=0)
            )
            return reward_


# Do not use or delete
# Reward function for the centralized agent. To be used only if all the buildings receive the same reward.
def reward_function_sa(electricity_demand):

    reward_ = -np.array(electricity_demand).sum()
    reward_ = max(0, reward_)
    reward_ = reward_ ** 3.0

    return reward_
