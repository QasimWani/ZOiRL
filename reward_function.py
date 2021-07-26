"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified at will. This reward_function takes all the electrical demands and carbon intensity of all the buildings and turns them into one or multiple rewards for the agent(s)
"""
import numpy as np

class DemandBuffer:
    """
    Implementation of a fixed size replay buffer.
    """

    def __init__(self, buffer_size:int = 7, electricity_demand):
        """
        Initializes the buffer.
        """
        self.electricity_demand = electricity_demand
        self.replay_memory = deque([electric_demand])  # Experience replay memory object
        
        self.total_it = 0
        self.max_it = buffer_size * 24
        

    def add(self, data: dict):
        """Adds an experience to existing memory - Oracle"""
        if self.total_it % 24 == 0:
            self.replay_memory.append({})
        self.replay_memory[-1] = data
        self.total_it += 1

    def get_recent(self):
        """Returns most recent data from memory"""
        return (
            self.replay_memory[-1] if len(self) > 0 and self.total_it % 24 != 0 else {}
        )

    def sample(self, is_random=False):
        """Picks all samples within the replay_buffer"""
        # critic 1 last n days - sequential
        # critic 2 last n days - random

        if is_random:  
            indices = np.random.choice(
                np.arange(len(self)), size=self.batch_size, replace=False
            )

        else:  
            indices = np.arange(len(self) - self.batch_size, len(self))

        days = [self.get(index) for index in indices]  # get all random experiences
        # combine all days together from DataLoader
        return days

    def get(self, index):
        """Returns an element from deque specified by `index`"""
        try:
            return self.replay_memory[index]
        except IndexError:
            print("Trying to access invalid index in replay buffer!")
            return None

    def set(self, index, data: dict):
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


    


class reward_function_ma:
    def __init__(self, n_agents, building_info):
        self.n_agents = n_agents
        self.building_info = building_info

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
            return list(np.sign(electricity_demand)*0.01*(np.array(np.abs(electricity_demand))**2 * max(0, total_electricity_demand)))
            
            # Use this reward when running the SAC example. It assumes that the building-agents act independently of each other, without sharing information through the reward.
        
        # Need to include the function to get current hour
        
        alpha = 5  
        # Define ramping cost
        ramping_cost = electricity_demand - DemandBuffer.replay_memory[-1]
        
        DemandBuffer.replay_memory.append(electricty_demand)
        
        # Getting if at the middle of the day or the end of the day
        hour = DemandBuffer.total_it % 24
        
        if hour == 23:
            r_ = -ramping_cost - alpha*max(DemandBuffer.replay_mamory[-24:])**2
        else:
            r_ = -ramping_cost 
            
        reward_ = reward_.append(r_)
        
        reward_[reward_>0] = 0
        return list(reward_)
       
        
        
        
        
        
# Do not use or delete
# Reward function for the centralized agent. To be used only if all the buildings receive the same reward.
def reward_function_sa(electricity_demand):

    reward_ = -np.array(electricity_demand).sum()
    reward_ = max(0, reward_)
    reward_ = reward_**3.0
    
    return reward_