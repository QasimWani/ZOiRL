import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import ReplayBuffer

## local imports
from predictor import Predictor
from actor import Actor
from critic import Critic, OptimCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        meta_episode=2,
    ):
        self.total_it = 0

        self.actor = Actor(
            ...
        )  # instead of initializing actor class every iteration, add setter within Actor
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(...)
        self.critic_optimizer = OptimCritic(...)
        self.critic_target = copy.deepcopy(self.critic)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.meta_episode = meta_episode

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        ### supply w/ param and building info before calling forward
        return self.actor.forward()

    def train(self, replay_buffer: ReplayBuffer, batch_size=2 * 24):
        # Methods called only on Delayed policy updates
        self.total_it += 1

        if self.total_it % self.meta_episode != 0:
            return

        # Sample replay buffer
        (
            state_1,
            action_1,
            next_state_1,
            reward_1,
            not_done_1,
        ) = replay_buffer.sample()  # critic 1
        (state_2, action_2, next_state_2, reward_2, not_done_2,) = replay_buffer.sample(
            True
        )  # critic 2

        with torch.no_grad():
            ### @jinming - Are we using noise?
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = self.actor_target(next_state) + noise

            self.critic.forward(replay_buffer, self.total_it)
            alphas = self.critic_optimizer.backward(
                ..., self.critic, self.critic_target
            )  # end of critic update

        # Compute actor loss
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
