"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified at will. This reward_function takes all the electrical demands and carbon intensity of all the buildings and turns them into one or multiple rewards for the agent(s)
"""
import numpy as np
from utils import ReplayBuffer

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
        
        buffer_size = 24*7
        demand_buffer = ReplayBuffer(buffer_size)
        
        if hour == 0:
            ramping_cost = electricity_demand
        else:
            ramping_cost = electricity_demand - demand_buffer.get_recent()
        
        demand_buffer.append(electricty_demand)
        
        alpha = 5   
            
        # We have access to previous E_grids from the buffer
        # Get current hour

            
        if hour <= 23:
            r_ = -ramping_cost
            
        else:
            r_ = -ramping_cost - alpha*(E_grid_pkhst)**2
                
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