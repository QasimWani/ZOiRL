# solving supply chain problem using Actor Critic

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


class Actor:
    def __init__(self) -> None:
        pass

    def forward(self):
        """Forward pass for actor module. Used in env.step"""
        pass

    def backward(self):
        """Takes in meta-episode worth of data to compute the gradients for zetas"""
        pass

    def target_update(self):
        """Update target actor zeta params using zetas from local actor"""
        pass


class Critic:
    def __init__(self) -> None:
        pass

    def get_alphas(self):
        """Returns alpha coef for critic optimization model"""
        pass

    def forward(self):
        """Critic optimization forward pass. supply it with current action."""
        pass

    def Q_function(self):
        """Calculation of Q-function using critic alphas"""
        pass

    def target_update(self):
        """Update critic target alphas using local critic alphas"""
        pass


class CriticOptim:
    """Computes critic updates via L2 optimization"""

    def __init__(self) -> None:
        pass

    def backward(self, critic: "list[Critic]"):
        """Runs L2 optimization to get best estimates for Q-function"""
        pass


class Agent:
    def __init__(self) -> None:
        """Initialization of AC model"""

        # define actor modules
        self.actor = Actor
        self.actor_target = Actor

        # define critic modules
        self.critic_optim = CriticOptim
        self.critic = [Critic, Critic]
        self.critic_target = [Critic, Critic]

    def train(self):
        """Trains actor and critic model"""
        data = self.gather_data(...)  # gather meta-episode data
        # update critic
        self.critic_update(data)
        # update alphas
        self.actor_update(data)

    def gather_data(self):
        """Format meta-episode data to be used by actor and critic"""
        pass

    def critic_update(self, data):
        """Perform critic update using meta-episode `data`"""
        self.critic_optim.backward(self.critic)  # local critic update
        # Target Critic update - moving average
        for i in range(len(self.critic_target)):
            self.critic_target[i].target_update(self.critic[i].get_alphas())

    def actor_update(self, data):
        """Perform actor update using meta-episode `data`"""
        self.actor.backward(data)  # local actor update
        self.actor_target.backward(data, self.actor)  # target actor update
