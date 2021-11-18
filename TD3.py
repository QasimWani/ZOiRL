from copy import deepcopy
from energy_models import Building
import numpy as np

from citylearn import CityLearn
import time
import sys, warnings
from utils import ReplayBuffer, RBC

## local imports
from predictor import Predictor as DataLoader
from actor import Actor
from critic import Critic, Optim

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class TD3(object):
    """Base Agent class"""

    def __init__(
        self,
        action_space: list,
        num_buildings: int,
        building_info: dict,
        rbc_threshold: int,
        meta_episode: int = 7,  # after how many days to train Actor-Critic
    ) -> None:
        """Initialize Actor + Critic for weekday and weekends"""
        self.buildings = num_buildings
        self.action_space = action_space
        self.total_it = 0
        self.rbc_threshold = rbc_threshold
        self.meta_episode = meta_episode

        self.agent_rbc = RBC(action_space)

        self.actor = Actor(
            action_space, num_buildings, rbc_threshold + meta_episode * 24
        )  # 1 local actor
        self.actor_target = deepcopy(self.actor)  # 1 target actor
        self.actor_norl = deepcopy(
            self.actor
        )  # NORL actor, i.e. actor whose parameters stay constant.

        self.critic = [
            Critic(num_buildings, action_space),
            Critic(num_buildings, action_space),
        ]  # 2 local critics
        self.critic_target = deepcopy(self.critic)  # 2 target critics
        self.critic_optim = Optim()

        ### --- log details ---
        self.logger = []
        self.norl_logger = []
        self.optim_param_logger = []

        self.memory: ReplayBuffer = ReplayBuffer()
        self.reward_memory: ReplayBuffer = ReplayBuffer()

        ## initialize predictor for loading and synthesizing data passed into actor and critic
        self.data_loader = DataLoader(building_info, action_space)

        # day-ahead dispatch actions
        self.action_planned_day = None
        # self.E_grid_planned_day = np.zeros(shape=(num_buildings, 24))
        # self.init_updates = None

    def select_action(
        self,
        state,
        day_ahead: bool = False,
        # env: CityLearn = None,  # use for Oracle
    ):
        """Returns action from RBC/Optimization"""
        # 3 policies:
        # 1. RBC (utils.py)
        # 2. Online Exploration. (utils.py)
        # 3. Optimization (actor.py)

        # upload state to memory
        self._add_to_buffer(state, None)

        building_parameters = None
        if self.total_it >= self.rbc_threshold:  # run Actor
            if day_ahead:
                actions, building_parameters = self.day_ahead_dispatch_pred()
            else:
                actions, building_parameters = self.adaptive_dispatch_pred()
                self.optim_param_logger.append(building_parameters)
        else:  # run RBC
            if (
                self.total_it % 24 in [22, 23, 0, 1, 2, 3, 4, 5, 6]
                and self.total_it >= 1
            ):
                actions = self.data_loader.select_action(self.total_it)
            else:
                actions = self.agent_rbc.select_action(
                    state[0][self.agent_rbc.idx_hour]
                )
            self.optim_param_logger.append([None] * self.buildings)

        # upload action to memory
        self._add_to_buffer(None, actions)
        return actions, building_parameters

    def _add_to_buffer(self, state, action):
        """Internal function for adding state & action to state_buffer and action_buffer, respectively"""
        if state is not None:
            self.data_loader.upload_state(state)

        if action is not None:
            self.data_loader.upload_action(action)
            self.total_it += 1

    def day_ahead_dispatch_pred(self):
        """Returns day-ahead dispatch"""
        data_est = None
        if self.total_it % 24 == 0:  # save actions for 24hours
            data_est = self.data_loader.estimate_data(self.memory, self.total_it)
            self.data_loader.convert_to_numpy(data_est)

            self.action_planned_day, optim_values, _ = zip(
                *[
                    self.actor.forward(self.total_it % 24, data_est, id, dispatch=True)
                    for id in range(self.buildings)
                ]
            )
            # Shape: 9, 3, 24
            self.action_planned_day = np.array(self.action_planned_day)
            self.logger.append(optim_values)  # add all variables - Optimization

        action_planned_day = self.action_planned_day[:, :, self.total_it % 24]
        return action_planned_day, data_est

    def adaptive_dispatch_pred(self):
        """Returns adaptive dispatch for current hour"""
        data_est = self.data_loader.estimate_data(
            self.memory, self.total_it, is_adaptive=True
        )
        self.data_loader.convert_to_numpy(data_est)

        action_planned_day, optim_values, _ = zip(
            *[
                self.actor.forward(self.total_it % 24, data_est, id, dispatch=False)
                for id in range(self.buildings)
            ]
        )
        self.logger.append(optim_values)  # add all variables - Optimization

        return action_planned_day, data_est

    def critic_update(self, params_1: list, params_2: list):
        """Master Critic update"""
        # pre-process each days information into numpy array and pass them to critic update

        parameters_1, rewards_1 = params_1
        parameters_2, rewards_2 = params_2

        day_params_1, day_params_2 = [], []  # parameters and rewards for each day
        for params_1, r1, params_2, r2 in zip(
            parameters_1, rewards_1, parameters_2, rewards_2
        ):
            # deepcopy to prevent overriding issues
            params_1 = deepcopy(params_1)
            params_2 = deepcopy(params_2)
            r1 = deepcopy(r1)
            r2 = deepcopy(r2)
            # parse data for critic (in-place)
            self.data_loader.convert_to_numpy(params_1)
            self.data_loader.convert_to_numpy(params_2)
            self.data_loader.convert_to_numpy(r1)
            self.data_loader.convert_to_numpy(r2)
            # add processed day info
            day_params_1.append([params_1, r1])
            day_params_2.append([params_2, r2])

        # Local Critic Update
        for id in range(self.buildings):
            # local critic backward pass
            self.critic_optim.backward(
                day_params_1,
                day_params_2,
                self.actor_target.zeta,
                id,
                self.critic,
                self.critic_target,
            )

        # Target Critic update - moving average
        for i in range(len(self.critic_target)):
            self.critic_target[i].target_update(self.critic[i].get_alphas())

        # copy problem into critic local -- for use in actor backward
        self.critic[0].prob = deepcopy(self.critic_target[0].prob)
        # self.critic[1].prob = deepcopy(self.critic_target[1].prob)

    def actor_update(self, parameters: list):
        """Master Actor update"""
        # pre-process each days information into numpy array and pass them to actor update

        day_params = []
        for params in parameters:
            # deepcopy to prevent overriding issues
            params = deepcopy(params)
            # parse data for actor (in-place)
            self.data_loader.convert_to_numpy(params)
            # add processed day info
            day_params.append(params)

        for id in range(self.buildings):
            # local actor update
            self.actor.backward(self.total_it, self.critic[0], day_params, id)

            # target actor update - moving average
            self.actor_target.target_update(self.actor.get_zeta(), id)

    def train(self):
        """Update actor and critic every meta-episode. This should be called end of each meta-episode"""

        # gather data from memory for critic update
        parameters_1, idx_1 = self.memory.sample()  # critic 1 - sequential
        rewards_1 = self.reward_memory.sample(
            sample_by_indices=idx_1
        )  # critic 1 - rewards part

        parameters_2, idx_2 = self.memory.sample(is_random=True)  # critic 2 - random
        rewards_2 = self.reward_memory.sample(
            sample_by_indices=idx_2
        )  # critic 2 - rewards part

        # local + target critic update
        self.critic_update((parameters_1, rewards_1), (parameters_2, rewards_2))

        # local + target actor update
        self.actor_update(parameters_1)

    def add_to_buffer(self, state, action, reward, next_state, done):
        """Add to replay buffer"""
        # add reward to memory
        if len(self.memory) == 0:
            return

        r = self.data_loader.parse_data(
            self.reward_memory.get_recent(), {"reward": reward}
        )
        # add to memory
        self.reward_memory.add(r)

        if (
            self.total_it % (self.meta_episode * 24) == 0
            and self.total_it >= self.rbc_threshold + self.meta_episode * 24
        ):
            start = time.time()
            self.train()
            end = time.time()
            print(f"Time taken for training: {end - start}")


class Agent(TD3):
    def __init__(self, **kwargs):
        """Initialize Agent"""
        super().__init__(
            action_space=kwargs["action_spaces"],
            num_buildings=len(kwargs["building_ids"]),
            building_info=kwargs["building_info"],
            rbc_threshold=336,
        )
