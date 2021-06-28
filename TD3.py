import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import ReplayBuffer, RBC

## local imports
from predictor import *
from actor import Actor
from critic import Critic, Optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3(object):
    def __init__(
        self,
        num_actions: list,
        num_buildings: int,
        rbc_threshold: int = 336,  # 2 weeks by default
        env: CityLearn = None,
        is_oracle: bool = True,
        meta_episode: int = 2,
    ) -> None:
        """Initialize Actor + Critic for weekday and weekends"""
        self.buildings = num_buildings
        self.num_actions = num_actions
        self.total_it = 0
        self.rbc_threshold = rbc_threshold
        self.meta_episode = meta_episode
        self.agent_rbc = RBC(
            num_actions
        )  # runs for first 2 weeks (by default) to collect data

        self.actor = Actor(num_actions)
        self.actor_target = Actor(num_actions)

        self.critic = Critic()
        self.critic_target = Critic()

        self.memory = ReplayBuffer()

        ## initialize predictor for loading and synthesizing data passed into actor and critic
        self.data_loader = DataLoader(is_oracle, num_actions, env)

        # day-ahead dispatch actions
        self.action_planned_day = None
        self.init_updates = None

    def select_action(
        self,
        state,
        env: CityLearn = None,
        day_ahead: bool = True,
    ):
        """Returns epsilon-greedy action from RBC/Optimization"""
        if self.total_it < self.rbc_threshold:
            return self.agent_rbc.select_action(state)

        if day_ahead:
            if self.total_it % 24 == 0:
                data = deepcopy(
                    self.memory.get_recent()
                )  # should return an empty dictionary
                self.day_ahead_dispatch(env, data)

            actions = [
                np.array(self.action_planned_day[idx])[:, self.total_it % 24]
                for idx in range(len(self.num_actions))
            ]
        else:
            actions = None  # will throw an error!

        return actions

    def day_ahead_dispatch(self, env: CityLearn, data: dict):
        """Computes action for the current day (24hrs) in advance"""
        data_est = self.data_loader.model.estimate_data(
            env, data, self.total_it, self.init_updates
        )
        self.data_loader.model.convert_to_numpy(data_est)

        self.action_planned_day, cost_dispatch = zip(
            *[
                self.actor.forward(self.total_it % 24, data_est, id, dispatch=True)
                for id in range(self.buildings)
            ]
        )

        assert (
            len(self.action_planned_day[0][0]) == 24
        ), "Invalid number of observations for Optimization actions"

    def train(self):
        """Update actor and critic every meta-episode. This should be called end of each meta-episode"""
        data = deepcopy(self.memory.get(-1))  # should return an empty dictionary
        self.data_loader.model.convert_to_numpy(data)

        Q_value = [
            self.critic.forward(
                self.total_it % 24, data, id, data["rewards"][id]
            )  # add data->rewards
            for id in range(self.buildings)
        ]

    def add_to_buffer(self, state, action, reward, next_state, done):
        """Add to replay buffer"""
        raise NotImplementedError

    def add_to_buffer_oracle(self, env: CityLearn, action):
        """Add to replay buffer"""
        if (
            self.total_it % 24 == 0 and self.total_it >= self.rbc_threshold - 24
        ):  # reset values every day
            if type(self.data_loader.model) == Oracle:
                _, self.init_updates = self.data_loader.model.init_values(
                    self.memory.get(-1)
                )
            else:
                pass  # implement way to load previous eod SOC values into current days' 1st hour.

        self.data_loader.upload_data(self.memory, action, env, self.total_it)
        self.total_it += 1

        if (
            self.total_it % self.meta_episode * 24 == 0
            and self.total_it > self.rbc_threshold
        ):
            self.train()
