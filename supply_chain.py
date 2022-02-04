# solving supply chain problem using Actor Critic
from collections import deque
from copy import deepcopy
import torch
import cvxpy as cp
import numpy as np
import scipy.sparse as spa
from cvxpylayers.torch import CvxpyLayer
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

""" 
Actor Critic (AC) structure
while not done:
    1. Actor.forward = env.step()
    2. replay_buffer = collects data [for first x timesteps, random actions; then optimization based actions]
    # when time to train...
    3. critic.forward = get Q values (constant action)
    4. critic.backward = solve L2 optimization by computing Q function with variables as coef 
                        of Q-function w.r.t true values gather from critic forward pass
    5. update critic alphas
    6. actor.backward = solve E2E optimization by passing in original data from replay buffer. 
                        compute gradients w.r.t differentiable parameter \in Zeta. 
                        update gradients by taking average across gradients of entire meta-episode.
    7. use Adam optimizer to update weights of actor zetas.
"""
# part 1: openAI gym environment encapsulation

### utils

# set seeds
np.random.seed(0)
torch.manual_seed(0)
# set seeds


# generate problem data
n = 4  # nodes
k = 2  # suppliers (with prices p)
c = 2  # retail (with demand d)
m = 8  # links

supply_links = [0, 1]
retail_links = [6, 7]
internode_links = [2, 3, 4, 5]

# Incidence matrices (nodes x links)
A_in = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],  # 1 (supply)
        [0, 1, 0, 0, 0, 0, 0, 0],  # 2 (supply)
        [0, 0, 1, 0, 0, 1, 0, 0],  # 3 (retail)
        [0, 0, 0, 1, 1, 0, 0, 0],  # 4 (retail)
    ]
)

A_out = np.array(
    [
        [0, 0, 1, 1, 0, 0, 0, 0],  # 1 (supply)
        [0, 0, 0, 0, 1, 0, 0, 0],  # 2 (supply)
        [0, 0, 0, 0, 0, 0, 1, 0],  # 3 (retail)
        [0, 0, 0, 0, 0, 1, 0, 1],  # 4 (retail)
    ]
)

# Prices
mu_p = torch.tensor([0, 0.1]).double()
sigma_p = torch.tensor([0.2, 0.2]).double()
mean_p = torch.exp(mu_p + sigma_p ** 2 / 2).double().view(k, 1)
var_p = (torch.exp(sigma_p ** 2) - 1) * torch.exp(2 * mean_p + sigma_p ** 2)

# Demands
mu_d = torch.tensor([0.0, 0.4]).double()
sigma_d = torch.tensor([0.2, 0.2]).double()
mean_d = torch.exp(mu_d + sigma_d ** 2 / 2).double().view(c, 1)
var_d = (torch.exp(sigma_d ** 2) - 1) * torch.exp(2 * mean_d + sigma_d ** 2)

# Uncertainty distribution (prices and demands)
w_dist = torch.distributions.log_normal.LogNormal(
    torch.cat([mu_p, mu_d], 0), torch.cat([sigma_p, sigma_d], 0)
)

# Capacities
h_max = 3.0  # Maximum capacity in every node
u_max = 2.0  # Link flow capacity

# Storage cost parameters, W(x) = alpha'x + beta'x^2 + gamma
alpha = 0.01
beta = 0.01

# Transportation cost parameters
tau = 0.05 * np.ones((m - k - c, 1))
tau_th = torch.tensor(tau, dtype=torch.double)
r = 1.3 * np.ones((k, 1))
r_th = torch.tensor(r, dtype=torch.double)


# Define linear dynamics
# x = (h, p^{wh}, d)
# u = u
# w = (p^{wh}, d)
# x_{t+1} = Ax_{t} + Bu_{t} + w
A_d = np.bmat(
    [
        [np.eye(n), np.zeros((n, k + c))],
        [np.zeros((k + c, n)), np.zeros((k + c, k + c))],
    ]
)
A_d_th = torch.tensor(A_d, dtype=torch.double)
B_d = np.vstack([A_in - A_out, np.zeros((k + c, m))])
B_d_th = torch.tensor(B_d, dtype=torch.double)
n_x, n_u = B_d.shape

# Setup policy
# Parameters
P_sqrt = cp.Parameter((n, n))  # 4x4
q = cp.Parameter((n, 1))  # 4x1
x = cp.Parameter((n_x, 1))  # 8x1
h, p, d = x[:n], x[n : n + k], x[n + k :]

# Variables
u = cp.Variable((n_u, 1))
h_next = cp.Variable((n, 1))

# Cvxpy Layer
stage_cost = cp.vstack([p, tau, -r]).T @ u
next_stage_cost = cp.sum_squares(P_sqrt @ h_next) + q.T @ h_next
constraints = [
    h_next == h + (A_in - A_out) @ u,
    h_next <= h_max,
    0 <= u,
    u <= u_max,
    A_out @ u <= h,
    u[retail_links] <= d,
]
prob = cp.Problem(cp.Minimize(stage_cost + next_stage_cost), constraints)
policy = CvxpyLayer(prob, [x, P_sqrt, q], [u, h_next])

## functions
def stage_cost(x, u):
    assert x.shape[0] == u.shape[0]
    batch_size = x.shape[0]
    r_batch = r_th.repeat(batch_size, 1, 1)
    tau_batch = tau_th.repeat(batch_size, 1, 1)

    h, p, dh = x[:, :n], x[:, n : n + k], x[:, n + k :]

    m = len(u)

    # Selling + buying + shipping cost
    s_vec = torch.cat([p, tau_batch, -r_batch], 1).double()
    S = torch.bmm(s_vec.transpose(1, 2), u)
    H = alpha * h + beta * (h ** 2)  # Storage cost

    return torch.sum(S, 1) + torch.sum(H, 1)


def simulate(x, u):
    assert x.shape[0] == u.shape[0]
    batch_size = x.shape[0]

    A_batch = A_d_th.repeat(batch_size, 1, 1)
    B_batch = B_d_th.repeat(batch_size, 1, 1)

    zer = torch.zeros(batch_size, n, 1).double()
    w = w_dist.sample((batch_size,)).double().view((batch_size, k + c, 1))
    w_batch = torch.cat([zer, w], 1).double()

    return torch.bmm(A_batch, x) + torch.bmm(B_batch, u) + w_batch


def loss(policy, params, time_horizon, batch_size, seed=None):
    P_sqrt, q = params
    if seed is not None:
        torch.manual_seed(seed)

    # Batchify input
    x_b_0 = h_max * torch.rand(batch_size, n, 1).double()
    w_0 = w_dist.sample((batch_size,)).double().view((batch_size, k + c, 1))
    x_batch = torch.cat([x_b_0, w_0], 1).double()

    # Repeat parameter values
    P_sqrt_batch = P_sqrt.repeat(batch_size, 1, 1)
    q_batch = q.repeat(batch_size, 1, 1)

    cost = 0.0
    x_t = x_batch
    x_hist = [x_batch]
    u_hist = []
    for t in range(time_horizon):
        u_t = policy(
            x_t, P_sqrt_batch, q_batch, solver_args={"acceleration_lookback": 0}
        )[0]

        x_t = simulate(x_t, u_t)
        cost += stage_cost(x_t, u_t).mean() / time_horizon
        x_hist.append(x_t)
        u_hist.append(u_t)

    return cost, x_hist, u_hist


def loss_ac(policy, params, time_horizon, batch_size=1, seed=None):
    """Actor Critic loss"""
    P_sqrt, q = params
    if seed is not None:
        torch.manual_seed(seed)

    # Batchify input
    x_b_0 = h_max * torch.rand(batch_size, n, 1).double()
    w_0 = w_dist.sample((batch_size,)).double().view((batch_size, k + c, 1))
    x_batch = torch.cat([x_b_0, w_0], 1).double()

    # Repeat parameter values
    P_sqrt_batch = P_sqrt.repeat(batch_size, 1, 1)
    q_batch = q.repeat(batch_size, 1, 1)

    cost = []
    x_t = x_batch
    x_hist = []
    u_hist = []
    h_next_hist = []
    for t in range(time_horizon):
        u_t, h_next = policy(
            x_t, P_sqrt_batch, q_batch, solver_args={"acceleration_lookback": 0}
        )

        x_t = simulate(x_t, u_t)

        _c = stage_cost(x_t, u_t).mean() / time_horizon
        cost.append(_c + torch.randn_like(_c))

        x_hist.append(x_t)
        u_hist.append(u_t)
        h_next_hist.append(h_next)

    return cost, x_hist, u_hist, h_next_hist


def monte_carlo(policy, params, time_horizon, batch_size=1, trials=10, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    results = []
    x = []
    u = []

    for i in range(trials):
        cost, x_hist, u_hist = loss(
            policy, params, time_horizon, batch_size=batch_size, seed=seed
        )
        results.append(cost.item())
        x.append(x_hist)
        u.append(u_hist)
    return results, x, u


def get_baseline_params():
    torch.manual_seed(0)
    P_sqrt_baseline = torch.eye(n, dtype=torch.double)
    q_baseline = -h_max * torch.ones(n, 1, dtype=torch.double)
    return [P_sqrt_baseline + torch.rand(n), q_baseline + torch.rand(1)]


### utils


class ReplayBuffer:
    """
    Implementation of a fixed size replay buffer.
    The goal of a replay buffer is to unserialize relationships between sequential experiences, gaining a better temporal understanding.
    """

    def __init__(self, buffer_size: int, batch_size: int):
        """
        Initializes the buffer.
        @Param:
        1. buffer_size: Maximum length of the buffer for extrapolating all experiences into trajectories. Max episode capacity.
        2. batch_size: size of mini-batch to train on. Sampling episode capcity.
        3. time_horizon: number of experiences per episode. Number of exeriences per episode.
        """
        self.replay_memory = deque(
            maxlen=buffer_size
        )  # Experience replay memory object
        self.batch_size = batch_size

        self.max_capacity = buffer_size

    def _is_empty(self):
        """Is buffer empty"""
        return len(self) == 0

    def add(self, data: list):
        """Adds an experience to existing memory"""
        self.replay_memory.append(data)

    def get_recent(self) -> list:
        """Returns most recent data from memory"""
        return self.replay_memory[-1]

    def sample(self, is_random: bool = False):
        """Picks all samples within the replay_buffer"""
        # critic 1 last n timesteps - sequential
        # critic 2 last n timesteps - random
        assert (
            len(self.replay_memory) >= self.batch_size
        ), f"Current replay buffer size: {len(self.replay_memory)}. Expected at least {self.batch_size}."

        if is_random:  # critic 2
            indices = np.random.choice(
                np.arange(len(self)), size=self.batch_size, replace=False
            )

        else:  # critic 1
            indices = np.arange(len(self) - self.batch_size, len(self))

        data = [self.get(index) for index in indices]  # get all random experiences
        return data, indices

    def get(self, index: int):
        """Returns an element from deque specified by `index`"""
        try:
            return self.replay_memory[index]
        except IndexError:
            print("Trying to access invalid index in replay buffer!")
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
            print(
                "Trying to set replay buffer w/ either invalid index or unable to set data!"
            )
            return None

    def __len__(self):  # override default __len__ operator
        """Return the current size of internal memory."""
        return len(self.replay_memory)


class Actor:
    def __init__(
        self,
        policy: CvxpyLayer,
        init_params: list,
        time_horizon: int,
        batch_size: int,
        lr: float,
        rho: float = 0.8,
    ) -> None:
        self.policy = policy
        self.params = init_params
        self.time_horizon = time_horizon
        self.batch_size = batch_size
        self.rho = rho

        self.params_progression = [deepcopy(self.params)]

        self.optim = torch.optim.Adam(self.params, lr=lr)

    def get_params(self):
        return [p.clone().detach() for p in self.params]

    def forward(self, seed: int = 0):
        """Forward pass for actor module. returns next action"""
        with torch.no_grad():
            rewards, x, u, h_next = loss_ac(
                self.policy,
                self.params,
                self.time_horizon,
                self.batch_size,
                seed=seed,
            )
        # format data
        rewards = torch.tensor(rewards).double()
        x = torch.stack(x).squeeze(1)
        u = torch.stack(u).squeeze(1)
        h_next = torch.stack(h_next).squeeze(1)

        return rewards, x, u, h_next

    def backward(self, batch_parameters, critic):
        """
        Takes in meta-episode worth of data to compute the gradients for zetas
        @Param:
        1. batch_parameters: list of tuples of (rewards, x, u, h_next) per episode
        2. critic: local Critic - 1
        """
        m = len(batch_parameters)  # number of episodes
        self.optim.zero_grad()

        cost = torch.tensor(0.0, requires_grad=True)
        for i in range(m):

            _, X, U, H_NEXT = batch_parameters[i]

            for j in range(self.time_horizon):
                # unload parameters
                P_sqrt, q = self.params

                # assert P_sqrt.requires_grad, "P_sqrt requires grad"
                # assert q.requires_grad, "q requires grad"

                P_sqrt.requires_grad = True
                q.requires_grad = True

                # unroll parameters based on timestep j
                x = X[j]
                u = U[j]
                h_next = H_NEXT[j]

                h, p, d = x[:n], x[n : n + k], x[n + k :]

                # Cvxpy Layer
                stage_cost = (
                    critic.alpha_stage_cost * cp.vstack([p, tau, -r]).T @ u
                ).value.item()

                next_stage_cost = (
                    torch.tensor(critic.alpha_next_stage_cost)
                    * torch.sum((P_sqrt @ h_next) ** 2)
                    + q.T @ h_next
                )

                # update cost
                cost = cost + (
                    -stage_cost - next_stage_cost - torch.tensor(critic.alpha_bias)
                ) / (self.time_horizon * m)

        # aggregate loss across episodes
        cost.backward()
        self.optim.step()

        # progression of parameters
        self.params_progression.append(self.params)

    def target_update(self, params: list):
        """Update target actor zeta params using zetas from local actor"""
        for i in range(len(self.params)):
            self.params[i] = self.rho * self.params[i] + (1 - self.rho) * params[i]


class Critic:
    def __init__(self, rho: float = 0.8) -> None:
        # constant
        self.rho = rho

        # define coefficients to learn
        self.alpha_stage_cost = -1.0
        self.alpha_next_stage_cost = -1.0
        self.alpha_bias = -1.0

    def get_alphas(self):
        """Returns alpha coef for critic optimization model"""
        return [self.alpha_stage_cost, self.alpha_next_stage_cost, self.alpha_bias]

    def set_alphas(self, sc, nsc, b):
        self.alpha_stage_cost = sc
        self.alpha_next_stage_cost = nsc
        self.alpha_bias = b

    def target_update(self, alphas: list):
        """Update critic target alphas using local critic alphas"""
        assert (
            len(alphas) == 3
        ), f"Incorrect dimension passed. Alpha tuple should be of size 3. found {len(alphas)}"

        ### main target update -- alphas_new comes from LS optim sol.
        sc, nsc, b = self.get_alphas()  # stage cost, next stage cost, bias
        alpha_sc = self.rho * sc + (1 - self.rho) * alphas[0]
        alpha_nsc = self.rho * nsc + (1 - self.rho) * alphas[1]
        alpha_b = self.rho * b + (1 - self.rho) * alphas[2]

        # update target critic alphas
        self.set_alphas(alpha_sc, alpha_nsc, alpha_b)


class CriticOptim:
    """Computes critic updates via L2 optimization"""

    def __init__(
        self,
        policy: CvxpyLayer,
        time_horizon: int,
        lambda_: float = 0.9,
        rho: float = 0.8,
    ) -> None:
        self.policy = policy

        # constants
        self.lambda_ = lambda_
        self.rho = rho
        self.time_horizon = time_horizon

    def reward_warping_layer(
        self, timestep: int, data: list, critic: Critic, actor_zeta: list
    ):
        """
        Uses cost function from optimization to generate Q-value
        @Param:
        - data: one individual experience in an episode of the format `reward, x, u, h_next`
        """
        # unload parameters
        P_sqrt, q = actor_zeta

        reward, x, u, h_next = data

        # unwrap `data` based on `timestep`
        reward = reward[timestep]
        x = x[timestep]
        u = u[timestep]
        h_next = h_next[timestep]

        h, p, d = x[:n], x[n : n + k], x[n + k :]

        # Cvxpy Layer

        stage_cost = (
            critic.alpha_stage_cost * cp.vstack([p, tau, -r]).T @ u
        ).value.item()

        next_stage_cost = (
            critic.alpha_next_stage_cost * cp.sum_squares(P_sqrt @ h_next)
            + q.T @ h_next
        ).value.item()

        Q_value = -stage_cost - next_stage_cost - critic.alpha_bias
        self.debug = stage_cost, next_stage_cost, Q_value

        return Q_value, reward, stage_cost, next_stage_cost

    def obtain_target_Q(
        self,
        batch_1_parameters,
        batch_2_parameters,
        critic_1: Critic,
        critic_2: Critic,
        actor_zeta,
        timestep: int,
        is_random: bool,
    ):
        """Computes target Q to build up `y_r` for L2 optimization model. Seeks data per episode"""

        Q1, r1, *costs1 = self.reward_warping_layer(
            timestep, batch_1_parameters, critic_1, actor_zeta
        )
        Q2, r2, *costs2 = self.reward_warping_layer(
            timestep, batch_2_parameters, critic_2, actor_zeta
        )

        is_terminal = int(timestep + 1 == self.time_horizon)

        if is_random:  # critic 2
            return costs2, r2 + self.lambda_ * (1 - is_terminal) * min(Q1, Q2)  # y_r

        return costs1, r1 + self.lambda_ * (1 - is_terminal) * min(Q1, Q2)  # y_r

    def gather_target_Q(
        self,
        batch_1_parameters,
        batch_2_parameters,
        critic_1: Critic,
        critic_2: Critic,
        actor_zeta,
        is_random: bool,
    ):
        """Gathers output of `obtain_target_Q` for all episodes"""
        assert len(batch_1_parameters) == len(
            batch_2_parameters
        ), "Data must be same length!"
        y_values = []
        stage_costs = []
        next_stage_costs = []

        m, n = (
            len(batch_1_parameters),
            self.time_horizon,
        )  # number of episodes, time horizon

        # get target Q for meta-episode
        for episode in range(len(batch_1_parameters)):
            for r in range(self.time_horizon):
                # get target Q for each timestep per episode
                costs, y_r = self.obtain_target_Q(
                    batch_1_parameters[episode],
                    batch_2_parameters[episode],
                    critic_1,
                    critic_2,
                    actor_zeta,
                    r,
                    is_random,
                )

                # append data
                y_values.append(y_r)
                stage_costs.append(costs[0])
                next_stage_costs.append(costs[1])
        return (
            np.reshape(y_values, (m, n)),
            np.reshape(stage_costs, (m, n)),
            np.reshape(next_stage_costs, (m, n)),
        )

    def L2_optimization(
        self,
        batch_parameters1,
        batch_parameters2,
        critic: "list[Critic]",
        actor_zeta,
        is_random: bool,
    ):
        """Computes L2 optimization for critic alphas"""
        critic_1, critic_2 = critic  # unpack critics

        m = len(batch_parameters1)  # number of episodes

        # define optimization model
        alpha_stage_cost = cp.Variable()
        alpha_next_stage_cost = cp.Variable()
        alpha_bias = cp.Variable()

        y, sc, nsc = self.gather_target_Q(
            batch_parameters1,
            batch_parameters2,
            critic_1,
            critic_2,
            actor_zeta,
            is_random,
        )

        # define cost
        costs = []
        for i in range(m):
            costs.append(
                self.rho
                * cp.sum(
                    cp.square(
                        -(
                            alpha_stage_cost * sc[i]
                            + alpha_next_stage_cost * nsc[i]
                            + alpha_bias
                        )
                        - y[i]
                    )
                    + (1 - self.rho)
                    * (
                        cp.square(alpha_stage_cost)
                        + cp.square(alpha_next_stage_cost)
                        + cp.square(alpha_bias)
                    )
                )
            )

        # define objective
        objective = cp.Minimize(cp.sum(costs))

        # define problem
        prob = cp.Problem(objective)

        # solve problem
        solution = prob.solve(solver=cp.SCS, verbose=False)

        assert (
            float("-inf") < solution < float("inf")
        ), "Unbounded solution/primal infeasable"

        # return new parameters
        return alpha_stage_cost.value, alpha_next_stage_cost.value, alpha_bias.value

    def backward(
        self,
        batch_1_parameters: list,
        batch_2_parameters: list,
        zeta: list,
        critic_local: "list[Critic]",
        critic_target: "list[Critic]",
    ) -> None:
        """Runs L2 optimization to get best estimates for Q-function"""
        # extract critic
        critic_local_1, critic_local_2 = critic_local

        # Compute L2 Optimization for Critic Local 1 (using sequential data) and Critic Local 2 (using random data) using Critic Target 1 and 2
        local_1_solution = self.L2_optimization(
            batch_1_parameters, batch_2_parameters, critic_target, zeta, False
        )

        local_2_solution = self.L2_optimization(
            batch_1_parameters, batch_2_parameters, critic_target, zeta, True
        )

        # update alphas for local
        critic_local_1.alpha_stage_cost = local_1_solution[0]
        critic_local_1.alpha_next_stage_cost = local_1_solution[1]
        critic_local_1.alpha_bias = local_1_solution[2]

        critic_local_2.alpha_stage_cost = local_2_solution[0]
        critic_local_2.alpha_next_stage_cost = local_2_solution[1]
        critic_local_2.alpha_bias = local_2_solution[2]


class Agent:
    def __init__(
        self,
        policy: CvxpyLayer,
        time_horizon: int,
        batch_size: int,  # internal to cost calculation
        replay_buffer_size: int,  # replay buffer max buffer size
        replay_batch_size: int,  # replay buffer training sample
        lr: float,
    ) -> None:
        """Initialization of AC model"""

        # define actor modules
        init_params = get_baseline_params()

        self.actor = Actor(policy, init_params, time_horizon, batch_size, lr)
        self.actor_target = deepcopy(self.actor)

        # define critic modules
        self.critic_optim = CriticOptim(policy, time_horizon)
        self.critic = [Critic(), Critic()]
        self.critic_target = [Critic(), Critic()]

        # define replay buffer
        self.memory = ReplayBuffer(replay_buffer_size, replay_batch_size)

        self.total_it = 0
        self.meta_episode = replay_batch_size

    def next(self):
        """Simulate next action"""
        reward, x, u, h_next = self.actor.forward()  # gather next step data

        self.memory.add([reward, x, u, h_next])  # add to memory

        self.total_it += 1

        if self.total_it > 0 and self.total_it % self.meta_episode == 0:
            self.train()

    def train(self):
        """Trains actor and critic model"""
        if len(self.memory) < self.memory.batch_size:
            return

        (
            paramerers_1,
            paramerers_2,
        ) = self.gather_data()  # gather meta-episode data (sequential and shuffled)

        # update critic
        self.critic_update(paramerers_1, paramerers_2)

        # update alphas
        self.actor_update(paramerers_1)

    def gather_data(self):
        """Format meta-episode data to be used by actor and critic"""
        p1, _ = self.memory.sample()  # sample from replay buffer
        p2, _ = self.memory.sample(is_random=True)  # sample from replay buffer
        return p1, p2

    def critic_update(self, critic_data1, critic_data2):
        """Perform critic update using meta-episode data"""

        # local critic update
        self.critic_optim.backward(
            critic_data1,
            critic_data2,
            self.actor_target.get_params(),
            self.critic,
            self.critic_target,
        )

        # Target Critic update - moving average
        for i in range(len(self.critic_target)):
            self.critic_target[i].target_update(self.critic[i].get_alphas())

    def actor_update(self, data):
        """Perform actor update using meta-episode `data`"""
        self.actor.backward(
            data, self.critic[0]
        )  # local actor update -- supply local critic 1
        self.actor_target.target_update(self.actor.get_params())  # target actor update


if __name__ == "__main__":
    # parameters
    time_horizon = 10
    epochs = 50  # number of episodes to run for
    cost_batch_size = 1  # batch size for cost calculation
    replay_buffer_size = 10  # max memory size
    replay_batch_size = 1  # training sample size -- train every `meta_episode`
    lr = 0.05  # adam learning rate

    agent = Agent(
        policy,
        time_horizon,
        cost_batch_size,
        replay_buffer_size,
        replay_batch_size,
        lr,
    )

    # get baseline cost
    P_sqrt_baseline = torch.eye(n, dtype=torch.double)
    q_baseline = -h_max * torch.ones(n, 1, dtype=torch.double)

    baseline_params = [P_sqrt_baseline, q_baseline]

    cost, _, _ = monte_carlo(
        policy,
        baseline_params,
        time_horizon,
        batch_size=1,
        trials=10,
        seed=0,
    )
    baseline_cost = np.mean(cost)
    print("Baseline cost: ", round(baseline_cost, 3))

    # perform training
    print("Training...")

    costs = []
    pbar = tqdm(range(epochs))

    for i in pbar:

        torch.manual_seed(i)

        agent.next()
        cost, _, _ = monte_carlo(
            agent.actor.policy,
            agent.actor.get_params(),
            time_horizon,
            batch_size=1,
            trials=10,
            seed=0,
        )
        costs.append(np.mean(cost))

        # update progress bar
        pbar.set_description("cost: " + str(round(costs[-1], 3)))

    print("Final cost: ", round(costs[-1], 3))

    improvement = 100 * np.abs(costs[-1] - baseline_cost) / np.abs(baseline_cost)
    print("Performance improvement: ", round(improvement, 3))

    # plot cost
    plt.plot(costs, c="k", label="Loss")
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.tight_layout()
    plt.savefig("iAC_supply_chain_training.pdf")
    plt.show()
