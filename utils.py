import numpy as np

from collections import deque, namedtuple
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
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )  # standard S,A,R,S',done

    def add(self, state, action, reward, next_state, done):
        """Adds an experience to existing memory"""
        if (
            len(self.replay_memory) == 0 or len(self.replay_memory[-1]) == 24
        ):  # empty buffer or full day, create new day
            self.replay_memory.append([])

        trajectory = self.experience(state, action, reward, next_state, done)
        self.replay_memory[-1].append(trajectory)

    def sample(self, is_random=False):
        """Picks all samples within the replay_buffer"""
        # critic 1 last 4*n days - sequential
        # critic 2 last 4*n days - random
        assert (
            len(self.get(len(self) - 1)) == 24
        ), f"Meta-episode not reached yet! Current hour {len(self.replay_memory[-1])}"

        if is_random:  # critic 2
            indices = np.random.choice(
                np.arange(len(self)), size=self.batch_size, replace=False
            )
            days = [self.get(index) for index in indices]  # get all random experiences

        else:  # critic 1
            indices = np.arange(len(self) - self.batch_size, len(self))
            days = [self.get(index) for index in indices]  # get all random experiences

        states = (
            torch.from_numpy(
                np.vstack([[e.state for e in experiences] for experiences in days])
            )
            .float()
            .to(device)
        )

        actions = (
            torch.from_numpy(
                np.vstack(
                    np.vstack([[e.action for e in experiences] for experiences in days])
                )
            )
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack(
                    np.vstack([[e.reward for e in experiences] for experiences in days])
                )
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack(
                    np.vstack(
                        [[e.next_state for e in experiences] for experiences in days]
                    )
                )
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack(
                    np.vstack([[e.done for e in experiences] for experiences in days])
                ).astype(np.uint8)
            )
            .float()
            .to(device)
        )

        return states, actions, rewards, next_states, dones, indices

    def __len__(self):  # override default __len__ operator
        """Return the current size of internal memory."""
        return len(self.replay_memory)

    def get(self, index):
        """Returns an element from deque specified by `index`"""
        assert 0 <= index < len(self.replay_memory), "Invalid index specified"
        return self.replay_memory[index]
