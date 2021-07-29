from copy import deepcopy
import numpy as np

import time

from utils import ReplayBuffer, RBC

## local imports
from predictor import *
from actor import Actor
from critic import Critic, Optim

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
        meta_episode: int = 50,
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

        self.actor = Actor(
            num_actions, num_buildings, rbc_threshold + meta_episode * 24
        )  # 1 local actor
        self.actor_target = deepcopy(self.actor)  # 1 target actor
        self.actor_norl = deepcopy(
            self.actor
        )  # NORL actor, i.e. actor whose parameters stay constant.

        ### --- log details ---
        self.logger = []
        self.logger_data = []
        self.norl_logger = []

        self.critic = [
            Critic(num_buildings, num_actions),
            Critic(num_buildings, num_actions),
        ]  # 2 local critic's
        self.critic_target = deepcopy(self.critic)  # 2 target critic's

        self.critic_optim = Optim()

        self.memory = ReplayBuffer()

        ## initialize predictor for loading and synthesizing data passed into actor and critic
        self.data_loader = DataLoader(is_oracle, num_actions, env)

        # day-ahead dispatch actions
        self.action_planned_day = None
        self.E_grid_planned_day = np.zeros(shape=(num_buildings, 24))
        self.init_updates = None

    def select_action(
        self,
        state,
        env: CityLearn = None,
        day_ahead: bool = False,
    ):
        """Returns epsilon-greedy action from RBC/Optimization"""

        if self.total_it >= self.rbc_threshold:  # run Actor
            # should return an empty dictionary if sod, else return up to current hour data
            if self.total_it % 24 == 0:
                data = {}
            elif self.total_it % 24 == 1:
                data = self.data_loader.model.parse_data(  # this is just to add the immediate prev hour data
                    {}, self.data_loader.model.get_current_data_oracle(
                        env, self.total_it - 1, [x[28] for x in state], [x[28] for x in state],
                        [np.zeros(3) for _ in range(9)], [0 for _ in range(9)]))
            else:
                data = deepcopy(self.memory.get_recent()) #note that the memory is lagging behind
                data = self.data_loader.model.parse_data( #this is just to add the immediate prev hour data
                data,self.data_loader.model.get_current_data_oracle(
                    env, self.total_it-1, [x[28] for x in state], [x[28] for x in state], [np.zeros(3) for _ in range(9)],[0 for _ in range(9)]))
            if day_ahead:  # run day ahead dispatch w/ true loads from the future
                if self.total_it % 24 == 0:
                    # data = {}
                    self.day_ahead_dispatch(env, data)

                actions = [
                    np.array(self.action_planned_day[idx])[:, self.total_it % 24]
                    for idx in range(len(self.num_actions))
                ]
            else:  # adaptive dispatch
                if self.total_it % 24 ==0:
                    actions = self.adaptive_dispatch(env, data)
                elif self.total_it % 24 <=8:
                    actions = [
                        np.array(self.action_planned_day[idx])[:, self.total_it % 24]
                        for idx in range(len(self.num_actions))
                    ]
                elif self.total_it % 24 in [9,11,13]:
                    actions = self.adaptive_dispatch(env, data)
                elif self.total_it % 24 in [10,12,14]:
                    actions = [
                        np.array(self.action_planned_day[idx])[:, 1]
                        for idx in range(len(self.num_actions))
                    ]
                elif self.total_it % 24 <=20:
                    actions = self.adaptive_dispatch(env, data)
                elif self.total_it % 24 in [22]:
                    actions = self.adaptive_dispatch(env, data)
                elif self.total_it % 24 in [21,23]:
                    actions = [
                        np.array(self.action_planned_day[idx])[:, 1]
                        for idx in range(len(self.num_actions))
                    ]
        else:  # run RBC
<<<<<<< Updated upstream

            actions = self.agent_rbc.select_action(state)

        self.total_it += 1
        return actions

    def adaptive_dispatch(self, env: CityLearn, data: dict):
        """Computes next action"""
        # data_t = {}
        # data_t["c_bat_init"] = [[]]
        # data_t["c_Csto_init"] = [[]]
        # data_t["c_Hsto_init"] = [[]]
        # for bid in range(9):
        #     data_t["c_Csto_init"][0].append(env.buildings[f'Building_{bid + 1}'].cooling_storage_soc[
        #                                         -1] / env.buildings[f'Building_{bid + 1}'].cooling_storage.capacity)
        #     data_t["c_Hsto_init"][0].append(env.buildings[f'Building_{bid + 1}'].dhw_storage_soc[-1] /
        #                                     env.buildings[f'Building_{bid + 1}'].dhw_storage.capacity)
        #     data_t["c_bat_init"][0].append(env.buildings[f'Building_{bid + 1}'].electrical_storage_soc[
        #                                        -1] / env.buildings[f'Building_{bid + 1}'].electrical_storage.capacity)
        #
        # if type(self.data_loader.model) == Oracle:
        #     _, self.init_updates = self.data_loader.model.init_values(
        #         data_t
        #     )
        # else:
        #     raise NotImplementedError  # implement way to load previous eod SOC values into current days' 1st hour.

        data_est = self.data_loader.model.estimate_data(
            env, data, self.total_it, self.init_updates, self.memory #note that the init_updates are useless here
        )
        data_est['c_bat_init'] = [[]]
        data_est['c_Hsto_init'] = [[]]
        data_est['c_Csto_init'] = [[]]
        for bid in range(9):
            data_est["c_Csto_init"][0].append(env.buildings[f'Building_{bid + 1}'].cooling_storage_soc[
                                                -1] / env.buildings[f'Building_{bid + 1}'].cooling_storage.capacity)
            data_est["c_Hsto_init"][0].append(env.buildings[f'Building_{bid + 1}'].dhw_storage_soc[-1] /
                                            env.buildings[f'Building_{bid + 1}'].dhw_storage.capacity)
            data_est["c_bat_init"][0].append(env.buildings[f'Building_{bid + 1}'].electrical_storage_soc[
                                               -1] / env.buildings[f'Building_{bid + 1}'].electrical_storage.capacity)

        self.data_loader.model.convert_to_numpy(data_est)
        self.logger_data.append(data_est)
        action, cost, action_planned_day = zip(
            *[
                self.actor.forward(self.total_it % 24, data_est, id, dispatch=False)
                for id in range(self.buildings)
            ]
        )
        self.action_planned_day = action_planned_day

        self.logger.append(cost)  # add all variables - RL
        # _, _, E_grid = zip(
        #     *[
        #         self.actor_target.forward(
        #             self.total_it % 24, data_est, id, dispatch=False
        #         )
        #         for id in range(self.buildings)
        #     ]
        # )
        # self.E_grid_planned_day[:, self.total_it % 24] = E_grid  # adaptive update

        ### DEBUG ###
        # gather data for NORL agent
        # _, norl_cost, _ = zip(
        #     *[
        #         self.actor_norl.forward(
        #             self.total_it % 24, data_est, id, dispatch=False
        #         )
        #         for id in range(self.buildings)
        #     ]
        # )
        # self.logger.append(cost)  # add all variables - RL
        # self.norl_logger.append(norl_cost)  # add all variables - Pure Optim
        ### DEBUG ###

        return action

    def day_ahead_dispatch(self, env: CityLearn, data: dict):
        """Computes action for the current day (24hrs) in advance"""
        if (self.total_it) % 24 == 0 and self.total_it > 0:  # reset values every day
            data_t = {}
            data_t["c_bat_init"] = [[]]
            data_t["c_Csto_init"] = [[]]
            data_t["c_Hsto_init"] = [[]]
            for bid in range(9):
                data_t["c_Csto_init"][0].append(env.buildings[f'Building_{bid + 1}'].cooling_storage_soc[
                        -1] / env.buildings[f'Building_{bid + 1}'].cooling_storage.capacity)
                data_t["c_Hsto_init"][0].append(env.buildings[f'Building_{bid + 1}'].dhw_storage_soc[-1] /
                    env.buildings[f'Building_{bid + 1}'].dhw_storage.capacity)
                data_t["c_bat_init"][0].append(env.buildings[f'Building_{bid + 1}'].electrical_storage_soc[
                        -1] /env.buildings[f'Building_{bid + 1}'].electrical_storage.capacity)


            if type(self.data_loader.model) == Oracle:
                _, self.init_updates = self.data_loader.model.init_values(
                    data_t
                )
            else:
                raise NotImplementedError  # implement way to load previous eod SOC values into current days' 1st hour.
=======
            # if (
            #     self.total_it % 24 in [22, 23, 0, 1, 2, 3, 4, 5, 6]
            #     and self.total_it >= 1
            # ):
            #     actions = self.data_loader.select_action(self.total_it)
            # else:
            #     actions = self.agent_rbc.select_action(
            #         state[0][self.agent_rbc.idx_hour]
            #     )
            actions = self.agent_rbc.select_action(
                state[0][self.agent_rbc.idx_hour])
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
>>>>>>> Stashed changes


        data_est = self.data_loader.model.estimate_data(
            env, data, self.total_it, self.init_updates, self.memory
        )
        #
        #
        # env.buildings['Building_1'].cooling_storage_soc[-1]/env.buildings['Building_1'].cooling_storage.capacity
        # x = []
        # for bid in range(9):
        #     x.append(self.init_updates['c_Csto_init'][bid]-env.buildings[f'Building_{bid+1}'].cooling_storage_soc[-1]/env.buildings[f'Building_{bid+1}'].cooling_storage.capacity)
        # y= []
        # for bid in range(9):
        #     y.append(
        #         self.init_updates['c_Hsto_init'][bid] - env.buildings[f'Building_{bid + 1}'].dhw_storage_soc[-1] /
        #         env.buildings[f'Building_{bid + 1}'].dhw_storage.capacity)
        # z = []
        # for bid in range(9):
        #     z.append(
        #         self.init_updates['c_bat_init'][bid] - env.buildings[f'Building_{bid + 1}'].electric_storage_soc[-1] /
        #         env.buildings[f'Building_{bid + 1}'].electric_storage.capacity)

        self.data_loader.model.convert_to_numpy(data_est)
        self.logger_data.append(data_est)
        self.action_planned_day, cost_dispatch, _ = zip(
            *[
                self.actor.forward(self.total_it % 24, data_est, id, dispatch=True)
                for id in range(self.buildings)
            ]
        )

<<<<<<< Updated upstream
        # compute E-grid
        # _, _, self.E_grid_planned_day = zip(
        #     *[
        #         self.actor_target.forward(
        #             self.total_it % 24, data_est, id, dispatch=True
        #         )
        #         for id in range(self.buildings)
        #     ]
        # )
        # self.E_grid_planned_day = np.array(self.E_grid_planned_day)

        ### DEBUG ###
        # gather data for NORL agent
        _, norl_cost_dispatch, _ = zip(
=======

        action_planned_day, optim_values, _ = zip(
>>>>>>> Stashed changes
            *[
                self.actor_norl.forward(self.total_it % 24, data_est, id, dispatch=True)
                for id in range(self.buildings)
            ]
        )

        self.logger.append(cost_dispatch)  # add all variables - RL
        self.norl_logger.append(norl_cost_dispatch)  # add all variables - Pure Optim
        ### DEBUG ###

        assert (
            len(self.action_planned_day[0][0]) == 24
        ), "Invalid number of observations for Optimization actions"

    def critic_update(self, parameters_1: list, parameters_2: list):
        """Master Critic update"""
        # pre-process each days information into numpy array and pass them to critic update
        day_params_1, day_params_2 = [], []
        for params_1, params_2 in zip(parameters_1, parameters_2):
            # deepcopy to prevent overriding issues
            params_1 = deepcopy(params_1)
            params_2 = deepcopy(params_2)
            # parse data for critic (in-place)
            self.data_loader.model.convert_to_numpy(params_1)
            self.data_loader.model.convert_to_numpy(params_2)
            # add processed day info
            day_params_1.append(params_1)
            day_params_2.append(params_2)

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
        self.critic[0].prob = self.critic_target[0].prob
        # self.critic[1].prob = self.critic_target[1].prob

    def actor_update(self, parameters: list):
        """Master Actor update"""
        # pre-process each days information into numpy array and pass them to actor update
        day_params = []
        for params in parameters:
            # deepcopy to prevent overriding issues
            params = deepcopy(params)
            # parse data for actor (in-place)
            self.data_loader.model.convert_to_numpy(params)
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
        parameters_1 = self.memory.sample()  # critic 1 - sequential
        parameters_2 = self.memory.sample(is_random=True)  # critic 2 - random

        # local + target critic update
        self.critic_update(parameters_1, parameters_2)

        # local + target actor update
        self.actor_update(parameters_1)

    def add_to_buffer(self, state, action, reward, next_state, done):
        """Add to replay buffer"""
        raise NotImplementedError

    def add_to_buffer_oracle(
        self, state: np.ndarray, env: CityLearn, action: list, reward: list, next_state: np.ndarray
    ):
        """Add to replay buffer"""
        # processing SOC's into suitable format
        # if (self.total_it+1) % 24 == 0 and self.total_it > 0:  # reset values every day
        #     if type(self.data_loader.model) == Oracle:
        #         _, self.init_updates = self.data_loader.model.init_values(
        #             self.memory.get(-1)
        #         )
        #     else:
        #         raise NotImplementedError  # implement way to load previous eod SOC values into current days' 1st hour.

        # upload E-grid (containarizing E-grid_collect w/ other memory for fast computational efficiency)
        self.data_loader.upload_data(
            self.memory,
            next_state[:, 28],  # current hour E_grid
            action,
            reward,
            env,
            self.total_it,
        )

        # self.total_it += 1 MJ move to select_action

        # start training after end of first meta-episode
        if (
            self.total_it % (self.meta_episode * 24) == 0
            and self.total_it > self.rbc_threshold
        ):
            start = time.time()
            self.train()  # begin critic and actor update
            print(f"\nTime taken (min): {round((time.time() - start) / 60, 3)}")
