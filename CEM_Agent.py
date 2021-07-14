import numpy as np
from actor import Actor


import time
from copy import deepcopy

from utils import ReplayBuffer, RBC

## local imports
from predictor import *


class CEM_Agent(object):
    
    def __init__(self,
        num_actions: list,
        num_buildings: int = 9,
        rbc_threshold: int = 24*1,  # 2 weeks by default
        env: CityLearn = None,
        is_oracle: bool = True,):
                 
#                  building_ids,
#             buildings_states_actions,
#             building_info,
#             observation_spaces,
#             action_spaces,
#             num_actions:list,
#             num_buildings:int,
#             env: CityLearn = None):
        
        self.buildings = num_buildings
        self.N_samples = 10
        self.K = 5    # size of elite set
        self.K_keep = 3
        self.k = 1   # Initial sample index
        self.flag = 0
#       self.env = env

        # Instantiating the actor class
        self.actor = Actor(
            num_actions, num_buildings, rbc_threshold
        )
        self.actor_norl = deepcopy(
            self.actor
        )  # NORL actor, i.e. actor whose parameters stay constant.
        
        ## initialize predictor for loading and synthesizing data passed into actor and critic
        self.data_loader = DataLoader(is_oracle, num_actions, env)
        
        ### --- log details ---
        self.logger = []
        self.logger_data = []
        self.norl_logger = []
        
        #Observed states initialisation
        self.E_netelectric_hist = []
        self.E_NS_hist = []
        self.C_bd_hist = []
        self.H_bd_hist = []
        self.eta_ehH_hist = []
        self.COP_C_hist = []
        self.outputs = {'E_netelectric_hist': self.E_netelectric_hist, 'E_NS_hist': self.E_NS_hist, 'C_bd_hist': self.C_bd_hist, 'H_bd_hist': self.H_bd_hist, 'COP_C_hist' : self.COP_C_hist}  # List for observed states for the last 24 hours

        self.zeta = []    # zeta for 9 buidling for 24 hours - list of 24 lists of size 9
#         self.zeta_k = []  # all zetas after the end of the day

        self.zeta_keys = set(
            [
                "p_ele",
#                 "ramping_cost_coeff",  # won't be used because constant due to DPP
                "eta_ehH",
                "eta_bat",
                "c_bat_end",
                "eta_Hsto",
                "eta_Csto",
            ]
        )
    
        
        self.zeta_eta_bat = np.ones(((1,24,9)))
        self.zeta_eta_Hsto = np.ones(((1,24,9)))
        self.zeta_eta_Csto = np.ones(((1,24,9)))
        self.zeta_eta_ehH = 0.9
        self.zeta_c_bat_end = 0.1
        
        self.mean_p_ele = [1]*9
        self.std_p_ele = [0.2]*9
        self.range_p_ele = [0.1, 5]
        

        # Initialising the elite sets
        self.elite_set = []     # Storing best 5 zetas i.e. a list of 5 lists which are further a list of 24 lists of size 9
        self.elite_set_prev = []  # Same format as elite_set

        # Initialising the list of costs after using certain params zetas
        self.costs = []

#         self.building_ids = building_ids
#         self.buildings_states_actions = buildings_states_actions
#         self.building_info = building_info
#         self.obsevation_spaces = observation_spaces
#         self.action_spaces = action_spaces

#         self.time_step = 1     #Initial hour of the day
        self.buildings = num_buildings
        self.num_actions = num_actions
        self.total_it = 0      # Total iterations
        self.rbc_threshold = rbc_threshold
        
        self.agent_rbc = RBC(
            num_actions
        )  # runs for first 2 weeks (by default) to collect data
        
        self.memory = ReplayBuffer()
        self.action_planned_day = None
        self.init_updates = None
        
        
        
        #########################################################################

    def get_zeta(self):    # Getting zeta for the 9 buildings for 24 hours

        # Getting the elite_set and elite_set_prev
        elite_set_eliteset_prev = self.set_EliteSet_EliteSetPrev()
        
        elite_set = elite_set_eliteset_prev[0]
        elite_set_prev = elite_set_eliteset_prev[1]


        if len(self.elite_set_prev) and self.k <= self.K_keep:

            self.zeta = self.elite_set_prev[-1]    # k-th best from elite_set_prev - zeta for 9 buildings

            zeta_k = self.zeta     # zeta for 9 buildings for 24 hours

        else:

            # Initialising parameters for the rest of the day for 24 hrs for 9 buildings
            zeta_p_ele = np.zeros(((1,24,9)))
#             zeta_eta_ehH = np.zeros(((1,24,9)))
#             zeta_eta_bat = np.zeros(((1,24,9)))
#             zeta_c_bat_end = np.zeros(((1,24,9)))
#             zeta_eta_Hsto = np.zeros(((1,24, 9)))
#             zeta_eta_Csto = np.zeros(((1,24,9)))

            mean_sigma_range = self.get_mean_sigma_range()    # Getting a list of lists for mean, std and ranges
            
            for i in range(9):
                
            
                zeta_p_ele[:,:,i] = np.clip(np.random.normal(mean_sigma_range[0][i],mean_sigma_range[1][i],24), mean_sigma_range[2][0], mean_sigma_range[2][1])
#                 zeta_eta_ehH[:,:,i] = np.clip(np.random.normal(mean_sigma_range[1][0][i],mean_sigma_range[1][1][i],24), mean_sigma_range[1][2][0], mean_sigma_range[1][2][1])
#                 zeta_eta_bat[:,:,i] = np.clip(np.random.normal(mean_sigma_range[2][0][i],mean_sigma_range[2][1][i],24), mean_sigma_range[2][2][0], mean_sigma_range[2][2][1])
#                 zeta_c_bat_end[:,:,i] = np.clip(np.random.normal(mean_sigma_range[3][0][i],mean_sigma_range[3][1][i],24), mean_sigma_range[3][2][0], mean_sigma_range[3][2][1])
#                 zeta_eta_Hsto[:,:,i] = np.clip(np.random.normal(mean_sigma_range[4][0][i],mean_sigma_range[4][1][i],24), mean_sigma_range[4][2][0], mean_sigma_range[4][2][1])
#                 zeta_eta_Csto[:,:,i] = np.clip(np.random.normal(mean_sigma_range[5][0][i],mean_sigma_range[5][1][i],24), mean_sigma_range[5][2][0], mean_sigma_range[5][2][1])

#             self.zeta = np.vstack((((((zeta_p_ele,
#                                     zeta_eta_bat,
#                                     zeta_eta_Hsto,
#                                     zeta_eta_Csto,
#                                     zeta_eta_ehH,
#                                     zeta_c_bat_end))))))

            self.zeta = zeta_p_ele

            zeta_k = self.zeta   # will set this zeta for the rest of the day

        return zeta_k

    ###########################################################################################
    
    
    def get_mean_sigma_range(self):

#         if self.flag == 0:
            
#             mean_p_ele = [1]*9
#             std_p_ele = [0.2]*9
#             range_p_ele = [0.1, 5]

#             mean_eta_ehH = [0.9]*9
#             std_eta_ehH = [0.1]*9
#             range_eta_ehH = [0.7, 1.3]

#             mean_eta_bat = [1]*9
#             std_eta_bat = [0.2]*9
#             range_eta_bat = [0.7, 1.3]

#             mean_c_bat_end = [0.1]*9
#             std_c_bat_end = [0.1]*9
#             range_c_bat_end = [0.01, 0.5]

#             mean_eta_Hsto = [1]*9
#             std_eta_Hsto = [0.2]*9
#             range_eta_Hsto = [0.7, 1.3]

#             mean_eta_Csto = [1]*9
#             std_eta_Csto = [0.2]*9
#             range_eta_Csto = [0.7, 1.3]

        mean_sigma_range = [self.mean_p_ele, self.std_p_ele, self.range_p_ele]

#         mean_sigma_range = [[mean_p_ele, std_p_ele, range_p_ele],
#                             [mean_eta_ehH, std_eta_ehH, range_eta_ehH],
#                             [mean_eta_bat, std_eta_bat, range_eta_bat],
#                             [mean_c_bat_end, std_c_bat_end, range_c_bat_end],
#                             [mean_eta_Hsto, std_eta_Hsto, range_eta_Hsto],
#                             [mean_eta_Csto, std_eta_Csto, range_eta_Csto]]

        return mean_sigma_range

    def select_action(self, state, env: CityLearn = None, day_ahead: bool = True):
        
        
        if self.total_it >= self.rbc_threshold:
            
            if self.total_it % 24 == 0:           # Get zeta at the starting of the day to be used for the rest of the day
                
                # Setting 9 zetas for 9 different buildings
                zeta_k = self.get_zeta()
                
                self.elite_set.append(zeta_k)
                
                print(np.shape(self.elite_set))
                
                for i in range(9):
                    zeta_tuple = (zeta_k[0,:,i] , self.zeta_eta_bat[:,:,i], self.zeta_eta_Hsto[:,:,i], self.zeta_eta_Csto[:,:,i], self.zeta_eta_ehH, self.zeta_c_bat_end)
                    self.actor.set_zeta(zeta_tuple, i)
                
#                 global eta_ehH0, eta_ehH1, eta_ehH2, eta_ehH3, eta_ehH4, eta_ehH5, eta_ehH6, eta_ehH7, eta_ehH8
                
                
#                 eta_ehH0 = zeta_k[4,:,0].tolist()
#                 eta_ehH1 = zeta_k[4,:,1].tolist()
#                 eta_ehH2 = zeta_k[4,:,2].tolist()
#                 eta_ehH3 = zeta_k[4,:,3].tolist()
#                 eta_ehH4 = zeta_k[4,:,4].tolist()
#                 eta_ehH5 = zeta_k[4,:,5].tolist()                              # Dimension 24
#                 eta_ehH6 = zeta_k[4,:,6].tolist()
#                 eta_ehH7 = zeta_k[4,:,7].tolist()
#                 eta_ehH8 = zeta_k[4,:,8].tolist()
                


            if self.total_it % 24 == 0:   # 24 th hour
                data = {}
            elif self.total_it % 24 == 1:   # Hour 1
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
                    global data_output
                    data_output = self.day_ahead_dispatch(env, data)
#                     print(data_output['COP_C'])
#                 print(type(self.action_planned_day))
#                 print(np.shape(self.action_planned_day))
#                 print(self.action_planned_day)
                
                actions = [
                    np.array(self.action_planned_day[idx])[:, self.total_it % 24]
                    for idx in range(len(self.num_actions))
                ]
        
            else:

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

            actions = self.agent_rbc.select_action(state)
      
        
        
        
        
#         zeta_t = self.get_zeta()          # Getting zet to be used for the rest of the day

#         zeta_t_dict = {'p_ele': zeta_t[0], 'eta_ehH': zeta_t[1], 'eta_bat': zeta_t[2], 'c_bat_end': zeta_t[3],
#                       'eta_Hsto': zeta_t[4], 'eta_Csto': zeta_t[5]}
        

        if self.total_it >= self.rbc_threshold and self.total_it % 24 <= 23:     # Storing the value of variables to calculate end of the day cost

            # Getting current states for all the 9 buildings
            
            
            E_observed = state[:,28]     # For 9 buildings and 24 hours - list of 24 lists
            
            
            E_NS_t = state[:,23]        # For 9 buildings and 24 hours - list of 24 lists
            
            
            
            self.C_bd_hist = data_output['C_bd']       # For 9 buildings and 24 hours - np.array size - 24*9
            
            self.H_bd_hist = data_output['H_bd']       # For 9 buildings and 24 hours - np.array size - 24*9
            
            self.COP_C_hist = data_output['COP_C']     # For 9 buildings and 24 hours - np.array size - 24*9
            
            self.eta_ehH_hist = [0.9]*9                # For 9 buildings and 24 hours - list of 9 lists of size 24

            # Appending the current states to the day history list of states
            self.E_netelectric_hist.append(E_observed)    # List of 24 lists each list size 9
            self.E_NS_hist.append(E_NS_t)                 # List of 24 lists

        if self.total_it >= self.rbc_threshold and self.total_it % 24 == 1:    # Calculate cost at the end of the day

            self.outputs = {'E_netelectric_hist': self.E_netelectric_hist, 'E_NS_hist': self.E_NS_hist, 'C_bd_hist': self.C_bd_hist, 'H_bd_hist': self.H_bd_hist, 'COP_C_hist' : self.COP_C_hist, 'eta_ehH_hist': self.eta_ehH_hist}  # List for observed states for the last 24 hours for the 9 buildings

            cost = self.get_cost_day_end()   # Calculating cost at the end of the day

            self.costs.append(cost)

            self.k = self.k + 1
            
            self.C_bd_hist = []    
            
            self.E_netelectric_hist = []
            
            self.H_bd_hist = []       
            
            self.COP_C_hist = [] 
            
            self.E_NS_hist = []

        self.total_it += 1
            
        return actions
    
    
    def get_cost_day_end(self):

        # outputs act as the next_state that we get after taking actions
       #  outputs = {'E_netelectric_hist': E_netelectric_hist, 'E_NS_hist': E_NS_hist, 'C_bd_hist': C_bd_hist, 'H_bd_hist': H_bd_hist}
       # outputs includes the history of all observed states during the day
        
        
        cost = np.zeros((1,9))
        self.outputs['E_netelectric_hist'] = np.array(self.outputs['E_netelectric_hist'])  # size 24*9
        print(np.shape(self.outputs['E_netelectric_hist']))
        self.outputs['E_NS_hist'] = np.array(self.outputs['E_NS_hist'])            # size 2*9
        print(np.shape(self.outputs['E_NS_hist']))
        self.outputs['eta_ehH_hist'] = np.array(self.outputs['eta_ehH_hist'])   # size 9*24
        
        
        
        for i in range(9):
            
            num = np.max(self.outputs['E_netelectric_hist'][:,i])
            
            C_bd_div_COP_C = np.divide(self.outputs['C_bd_hist'][:,i], self.outputs['COP_C_hist'][:,i])

            H_bd_div_eta_ehH = self.outputs['H_bd_hist'][:,i]/self.zeta_eta_ehH

            print(np.shape(self.outputs['E_NS_hist']))
#             E_NS_history = np.ones((24,1))
            
            den = np.max(self.outputs['E_NS_hist'][1,i]*np.ones((24,1)) + C_bd_div_COP_C + H_bd_div_eta_ehH)

            cost[:,i] = num/den

        return cost
    
    
    
    
    
#     def adaptive_dispatch(self, env: CityLearn, data: dict):  
        
#         """Computes next action"""
        
#         data_est = self.data_loader.model.estimate_data(
#             env, data, self.total_it, self.init_updates, self.memory #note that the init_updates are useless here
#         )
#         data_est['c_bat_init'] = [[]]
#         data_est['c_Hsto_init'] = [[]]
#         data_est['c_Csto_init'] = [[]]
#         for bid in range(9):
#             data_est["c_Csto_init"][0].append(env.buildings[f'Building_{bid + 1}'].cooling_storage_soc[
#                                                 -1] / env.buildings[f'Building_{bid + 1}'].cooling_storage.capacity)
#             data_est["c_Hsto_init"][0].append(env.buildings[f'Building_{bid + 1}'].dhw_storage_soc[-1] /
#                                             env.buildings[f'Building_{bid + 1}'].dhw_storage.capacity)
#             data_est["c_bat_init"][0].append(env.buildings[f'Building_{bid + 1}'].electrical_storage_soc[
#                                                -1] / env.buildings[f'Building_{bid + 1}'].electrical_storage.capacity)

#         self.data_loader.model.convert_to_numpy(data_est)
#         self.logger_data.append(data_est)
#         action, cost, action_planned_day = zip(
#             *[
#                 self.actor.forward(self.total_it % 24, data_est, id, dispatch=False)
#                 for id in range(self.buildings)
#             ]
#         )
#         self.action_planned_day = action_planned_day

#         self.logger.append(cost) 
        
#         return action
        
##########################################################################

    def set_EliteSet_EliteSetPrev(self):

        if self.k == 1:

            self.elite_set_prev = self.elite_set
            self.elite_set = []


#             elif self.k < self.N_samples:

#                 self.elite_set.append(self.get_zeta())

        if self.k > self.N_samples:      # Enough samples of zeta collected

            # Finding best k samples according to cost y_k
            self.costs = np.array(self.costs)  # Converting self.costs to np.array   dimensions = k*1*9
#             print(np.shape(self.costs))
            best_zeta_args = np.zeros((self.k - 1, 9))
    
            elite_set_dummy = self.elite_set
        
#             print(np.shape(self.elite_set))
#             print(np.shape(self.elite_set[0]))
            
            for i in range(9):
                best_zeta_args[:,i] = np.argsort(self.costs[:,:,i]).reshape(-1)     # Arranging costs for the i-th building

                # Finding the best K samples from the elite set
                for Kbest in range(self.K):
                    a = best_zeta_args[:,i][Kbest].astype(np.int32)
                    self.elite_set[Kbest][:,:,i] = elite_set_dummy[a][:,:,i]  
                    
            
             
            self.elite_set = self.elite_set[0:self.K]
            
            self.mean_p_ele = [[]]*9
            self.std_p_ele = [[]]*9

#             mean_eta_ehH = [[]]*9
#             std_eta_ehH = [[]]*9

#             mean_eta_bat = [[]]*9
#             std_eta_bat = [[]]*9

#             mean_c_bat_end = [[]]*9
#             std_c_bat_end = [[]]*9

#             mean_eta_Hsto = [[]]*9
#             std_eta_Hsto = [[]]*9

#             mean_eta_Csto = [[]]*9
#             std_eta_Csto = [[]]*9
                    
            # Fitting mean and standard deviation to the the elite set
            
            A = np.hstack(self.elite_set)
            
            for i in range(9):
                
                self.mean_p_ele[i] = np.mean(A[:,:,i], axis = 1)
                self.std_p_ele[i] = np.std(A[:,:,i], axis = 1)
                
            
            
            self.flag = 1

            self.k = 1    # Reset the sample index

            self.costs = []

        elite_set = self.elite_set
        elite_set_prev = self.elite_set_prev

        eliteSet_eliteSetPrev = [elite_set, elite_set_prev]

        return eliteSet_eliteSetPrev
    
    
    
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

        global data_est
        data_est = self.data_loader.model.estimate_data(
            env, data, self.total_it, self.init_updates, self.memory)
            
        self.data_loader.model.convert_to_numpy(data_est)
#         print(data_est['COP_C'])
        self.logger_data.append(data_est)
        self.action_planned_day, cost_dispatch, _ = zip(
            *[
                self.actor.forward(self.total_it % 24, data_est, id, dispatch=True)
                for id in range(self.buildings)
            ]
        )
        
        # gather data for NORL agent
        _, norl_cost_dispatch, _ = zip(
            *[
                self.actor_norl.forward(self.total_it % 24, data_est, id, dispatch=True)
                for id in range(self.buildings)
            ]
        )

        self.logger.append(cost_dispatch)  # add all variables - RL
        self.norl_logger.append(norl_cost_dispatch)  # add all variables - Pure Optim
        
        return data_est
     

