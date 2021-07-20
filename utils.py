import numpy as np
import json

from collections import deque, namedtuple
from citylearn import CityLearn  # for RBC
import torch


# Set to cuda (gpu) instance if compute available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# source: https://gist.github.com/enochkan/56af870bd19884f189639a0cb3381ff4#file-adam_optim-py
# > w_0 = adam.update(t,w=w_0, dw=dw)
class Adam:
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, t, w, dw):
        ## dw, db are from current minibatch
        t += 1  # 0 based
        # t = max(t, 1)  # prevent zero division

        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)

        ## bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** t)

        ## update weights and biases
        w = w - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        return w


META_EPISODE = 7  # number of days in a meta-episode
MINI_BATCH = 2  # number of days to sample


class ReplayBuffer:
    """
    Implementation of a fixed size replay buffer.
    The goal of a replay buffer is to unserialize relationships between sequential experiences, gaining a better temporal understanding.
    """

    def __init__(self, buffer_size=META_EPISODE, batch_size=MINI_BATCH):
        """
        Initializes the buffer.
        @Param:
        1. action_size: env.action_space.shape[0]
        2. buffer_size: Maximum length of the buffer for extrapolating all experiences into trajectories.
        3. batch_size: size of mini-batch to train on.
        """
        self.replay_memory = deque(
            maxlen=buffer_size
        )  # Experience replay memory object
        self.batch_size = batch_size

        self.total_it = 0
        self.max_it = buffer_size * 24

    def add(self, data: dict, full_day: bool = False):
        """Adds an experience to existing memory - Oracle"""
        if self.total_it % 24 == 0:
            self.replay_memory.append({})
        self.replay_memory[-1] = data

        if full_day:
            self.total_it += 24
        else:
            self.total_it += 1

    def get_recent(self):
        """Returns most recent data from memory"""
        return (
            self.replay_memory[-1] if len(self) > 0 and self.total_it % 24 != 0 else {}
        )

    def sample(self, is_random: bool = False):
        """Picks all samples within the replay_buffer"""
        # critic 1 last n days - sequential
        # critic 2 last n days - random

        if is_random:  # critic 2
            indices = np.random.choice(
                np.arange(len(self)), size=self.batch_size, replace=False
            )

        else:  # critic 1
            indices = np.arange(len(self) - self.batch_size, len(self))

        days = [self.get(index) for index in indices]  # get all random experiences
        # combine all days together from DataLoader
        return days

    def get(self, index: int):
        """Returns an element from deque specified by `index`"""
        try:
            return self.replay_memory[index]
        except IndexError:
            print("Trying to access invalid index in replay buffer!")
            return None

    def set(self, index: int, data: dict):
        """Sets an element of replay buffer w/ dictionary"""
        try:
            self.replay_memory[index] = data
        except:
            print(
                "Trying to set replay buffer w/ either invalid index or unable to set data!"
            )
            return None

    def __len__(self):  # override default __len__ operator
        """Return the current size of internal memory."""
        return len(self.replay_memory)


class RBC:
    def __init__(self, actions_spaces: list):
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

    def get_rbc_data(
        self,
        surrogate_env: CityLearn,
        state: np.ndarray,
        indx_hour: int,
        run_timesteps: int,
    ):
        """Runs RBC for x number of timesteps"""
        ## --- RBC generation ---
        E_grid = []
        for _ in range(run_timesteps):
            hour_state = np.array([[state[0][indx_hour]]])
            action = self.select_action(
                hour_state
            )  # using RBC to select next action given current sate
            next_state, rewards, done, _ = surrogate_env.step(action)
            state = next_state
            E_grid.append([x[28] for x in state])
        return E_grid


def get_idx_hour():
    # Finding which state
    with open("buildings_state_action_space.json") as file:
        actions_ = json.load(file)

    indx_hour = -1
    for obs_name, selected in list(actions_.values())[0]["states"].items():
        indx_hour += 1
        if obs_name == "hour":
            break
        assert (
            indx_hour < len(list(actions_.values())[0]["states"].items()) - 1
        ), "Please, select hour as a state for Building_1 to run the RBC"
    return indx_hour
