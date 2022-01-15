from logger import LOG
import numpy as np
import json
from tqdm import tqdm

from collections import defaultdict, deque
from citylearn import CityLearn  # for RBC

# source: https://gist.github.com/enochkan/56af870bd19884f189639a0cb3381ff4#file-adam_optim-py
# > w_0 = adam.update(t,w=w_0, dw=dw)
class Adam:
    def __init__(self, eta=0.1, beta1=0.9, beta2=0.999, epsilon=1e-6):
        self.m_dw, self.v_dw = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, t, w, dw):
        t = max(t, 1)  # ensure no division by 1
        ## dw from current minibatch
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
        w = w + self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        return w


BUFFER_SIZE = 7  # number of days in a meta-episode
MINI_BATCH = 4  # number of days to sample


class ReplayBuffer:
    """
    Implementation of a fixed size replay buffer.
    The goal of a replay buffer is to unserialize relationships between sequential experiences, gaining a better temporal understanding.
    """

    def __init__(self, buffer_size=BUFFER_SIZE, batch_size=MINI_BATCH):
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

    def sample(self, is_random: bool = False, sample_by_indices: list = []):
        """Picks all samples within the replay_buffer"""
        # critic 1 last n days - sequential
        # critic 2 last n days - random

        if len(sample_by_indices) > 0:  # sample by pre-specified indices
            return [self.get(index) for index in sample_by_indices]

        if is_random:  # critic 2
            indices = np.random.choice(
                np.arange(len(self)), size=self.batch_size, replace=False
            )

        else:  # critic 1
            indices = np.arange(len(self) - self.batch_size, len(self))

        days = [self.get(index) for index in indices]  # get all random experiences
        # combine all days together from DataLoader
        return days, indices

    def get(self, index: int):
        """Returns an element from deque specified by `index`"""
        try:
            return self.replay_memory[index]
        except IndexError:
            LOG("Trying to access invalid index in replay buffer!")
            return None

    def clear(self):
        """Clear replay memory"""
        try:
            self.replay_memory.clear()
        except Exception as e:
            raise BufferError(f"Unable to clear replay buffer: {e}")

    def set(self, index: int, data: dict):
        """Sets an element of replay buffer w/ dictionary"""
        try:
            self.replay_memory[index] = data
        except:
            LOG(
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
        self.idx_hour = self.get_idx_hour()

    def select_action(self, states: float):
        hour_day = states
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
        run_timesteps: int,
    ):
        """Runs RBC for x number of timesteps"""
        ## --- RBC generation ---
        E_grid = []
        for _ in range(run_timesteps):
            hour_state = state[0][self.idx_hour]
            action = self.select_action(
                hour_state
            )  # using RBC to select next action given current sate
            next_state, rewards, done, _ = surrogate_env.step(action)
            state = next_state
            E_grid.append([x[28] for x in state])
        return E_grid

    def load_day_actions(self):
        """Generate template of actions for RBC for a day"""
        return np.array([self.select_action(hour) for hour in range(24)]).transpose(
            [2, 1, 0]
        )

    def get_idx_hour(self):
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


class DataLoader:
    """Base Class"""

    def __init__(self, action_space: list) -> None:
        self.action_space = action_space

    def upload_data(self) -> None:
        """Upload to memory"""
        raise NotImplementedError

    def load_data(self):
        """Optional: not directly called. Should be called within `upload_data` if used."""
        raise NotImplementedError

    def parse_data(self, data: dict, current_data: dict):
        """Parses `current_data` for optimization and loads into `data`"""
        for key, value in current_data.items():
            if key not in data:
                data[key] = []
            data[key].append(value)
        return data

    def convert_to_numpy(self, params: dict):
        """Converts dic[key] to nd.array"""
        for key in params:
            # if key == "c_bat_init" or key == "c_Csto_init" or key == "c_Hsto_init":
            #     params[key] = np.array(params[key][0])
            # else:
            params[key] = np.array(params[key])

    def get_dimensions(self, data: dict):
        """Prints shape of each param"""
        for key in data.keys():
            print(key, data[key].shape)

    def get_building(self, data: dict, building_id: int):
        """Loads data (dict) from a particular building. 1-based indexing for building"""
        assert building_id > 0, "building_id is 1-based indexing."
        building_data = {}
        for key in data.keys():
            building_data[key] = np.array(data[key])[:, building_id - 1]
        return building_data

    def create_random_data(self, data: dict):
        """Synthetic data (Gaussian) generation"""
        for key in data:
            data[key] = np.clip(np.random.random(size=data[key].shape), 0, 1)
        return data


def agent_checkpoint_cost(agents: list, env: CityLearn):
    """Runs cost analysis on a list of agents"""
    costs = defaultdict(list)
    for agent in tqdm(agents):
        state = env.reset()
        done = False
        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.add_to_buffer(state, action, reward, next_state, done)
            state = next_state
            print(f"Timestep: {env.time_step}", end="\r", flush=True)
        cost, _ = env.cost(env.simulation_period)
        # log costs
        for k, v in cost.items():
            costs[k].append(v)
    return costs
