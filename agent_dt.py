
# ### Feel free to edit this file at will, but make sure it runs properly when we execute the main.py or main.ipynb file that is provided. You can't change the main file, only to the submission files.

# '''Import any packages here'''
# import json
# import torch

# class Agent:
#     def __init__(self, building_ids, buildings_states_actions, building_info):     
#         with open(buildings_states_actions) as json_file:
#             self.buildings_states_actions = json.load(json_file)
            
#         '''Initialize the class and define any hyperparameters of the controller'''
        
            
#     def select_action(self, states):
        
#         '''Action selection algorithm. You can set coordination_vars = None if you do not want to use this variable '''
            
#         return actions, coordination_vars
                
        
#     def add_to_buffer(self, states, actions, rewards, next_states, done, coordination_vars=None, coordination_vars_next=None):
        
#         '''Make any updates to your policy, you don't have to use all the variables above (you can leave the coordination
#         variables empty if you wish, or use them to share information among your different agents). You can add a counter
#         within this function to compute the time-step of the simulation, since it will be called once per time-step'''
        
        

from copy import deepcopy
from TD3 import TD3
from digital_twin import DigitalTwin
from oracle import Oracle

import numpy as np

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class Agent(TD3):
    """CEM Agent - inherits TD3 as agent"""

    def __init__(self, **kwargs):
        """Initialize Agent"""
        super().__init__(
            num_actions=kwargs["action_spaces"],
            num_buildings=len(kwargs["building_ids"]),
            rbc_threshold=336
        )

        observation_space = kwargs["observation_space"]
        self.env = kwargs["env"]

        self.oracle = Oracle(self.env,kwargs["action_spaces"])


        self.state_hist = []
        self.E_grid_dt = []       # For debugging purposes
        # CEM Specific parameters
        self.N_samples = 10
        self.K = 5  # size of elite set
        self.K_keep = 3
        self.k = 1  # Initial sample index
        self.flag = 0
        self.all_costs = []
        self.DT_costs = []     
        self.sod = np.ones((9,30))

        self.p_ele_logger = []
        self.mean_elite_set = []
        self.loads = {'E_ns':[],
                      'C_bd': [],
                      'H_bd':[],
                      'E_ns_dt': [],
                      'C_bd_dt': [],
                      'H_bd_dt': []
                      }
        # Observed states initialisation
        self.E_netelectric_hist = []
        self.E_NS_hist = []
        self.C_bd_hist = []
        self.H_bd_hist = []
        self.eta_ehH_hist = []
        self.COP_C_hist = []
        self.outputs = {
            "E_netelectric_hist": self.E_netelectric_hist,
            "E_NS_hist": self.E_NS_hist,
            "C_bd_hist": self.C_bd_hist,
            "H_bd_hist": self.H_bd_hist,
            "COP_C_hist": self.COP_C_hist,
        }  # List for observed states for the last 24 hours

        self.zeta = []  # zeta for all buidling for 24 hours (1*24x9)

        self.zeta_eta_bat = np.ones(((1, 24, self.buildings)))
        self.zeta_eta_Hsto = np.ones(((1, 24, self.buildings)))
        self.zeta_eta_Csto = np.ones(((1, 24, self.buildings)))
        self.zeta_eta_ehH = 0.9
        self.zeta_c_bat_end = 0.1

        self.mean_p_ele = [np.ones(24)] * self.buildings          # Having mean and range for each of the hour
        self.std_p_ele = [0.2*np.ones(24)] * self.buildings
        self.range_p_ele = [0.1, 5]

        # Initialising the elite sets
        self.elite_set = []  # Storing best 5 zetas i.e. a list of 5 lists which are further a list of 24 lists of size 9
        self.elite_set_prev = []  # Same format as elite_set

        # Initialising the list of costs after using certain params zetas
        self.costs = []

        # Digital Twin specific parameters

        self.Digital_Twin = DigitalTwin(
            ["Building_" + str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]],
            save_memory=True,
            buildings_states_actions="buildings_state_action_space.json",
            cost_function=[
                "ramping",
                "1-load_factor",
                "average_daily_peak",
                "peak_demand",
                "net_electricity_consumption",
            ],
            simulation_period=(0, 8759),
        )

        # create Digital Twin specific actor
        self.actor_digital_twin = deepcopy(self.actor)

        # Store state for duration of day for digital twin Zeta evaluation
        self.day_data = [None] * 24

        # 4 zetas defined to be evaluated in Digital TWin after each metaepisode 
        self.zeta_k_list = np.ones(
            ((4, 1, 24, len(observation_space)))
        )  # 4 different Zetas.

        self.zeta_k_list[1,:,0:13, :] = 0.2
        self.zeta_k_list[1,:,13:19, :] = 5
        self.zeta_k_list[1,:,19:23, :] = 0.2

        self.zeta_k_list[2,:,0:6, :] = 0.2
        self.zeta_k_list[2,:,7:19, :] = 5
        self.zeta_k_list[2,:,20:23, :] = 0.2

        self.zeta_k_list[3,:,0:5, :] = 0.2
        self.zeta_k_list[3,:,11:17, :] = 2
        self.zeta_k_list[3,:,22:23, :] = 0.2
        
        # For debugging purposes
        self.dt_building_logger = []
        self.e_soc_logger = []
        self.h_soc_logger = []
        self.c_soc_logger = []

    def get_zeta(self):  # Getting zeta for the 9 buildings for 24 hours
        """This function is used to get zeta for the actor. We set the zeta for the actor and do the forward pass to get actions. In our case
        we will only have p_ele as the zeta parameter. This get_zeta function calls the set_EliteSet_EliteSetPrev
        to get the elite_set and then selects zeta from that. Elite set stores the best zetas."""

        # Setting the elite_set and elite_set_prev
        elite_set_eliteset_prev = self.set_EliteSet_EliteSetPrev()

        if len(self.elite_set_prev) and self.k <= self.K_keep:

            # k-th best from elite_set_prev - zeta for all buildings
            self.zeta = self.elite_set_prev[0]

            zeta_k = self.zeta  # zeta for 9 buildings for 24 hours

        else:

            # Initialising parameters for the rest of the day for 24 hrs for 9 buildings
            zeta_p_ele = np.zeros(((1, 24, self.buildings)))

            mean_sigma_range = (
                self.get_mean_sigma_range()
            )  # Getting a list of lists for mean, std and ranges
            
            # Add condition to only take zeta at the start of the day
            if self.total_it % 24 == 0: 
                
                for i in range(self.buildings):
                    for t in range(24):

                        zeta_p_ele[:, t, i] = np.clip(
                            np.random.normal(
                                mean_sigma_range[0][i][t], mean_sigma_range[1][i][t], 1
                            ),
                            mean_sigma_range[2][0],
                            mean_sigma_range[2][1],
                        )

                self.zeta = zeta_p_ele

#         self.elite_set.append(self.zeta)
        return self.zeta

    def get_mean_sigma_range(self):
        """This function is called to get the current mean, standard deviation and allowed range for the
        parameter p_ele. We can access these 3 quantities by calling this function."""

        # ADD ALL PARAMS
        mean_sigma_range = [self.mean_p_ele, self.std_p_ele, self.range_p_ele]

        return mean_sigma_range
    
    def get_cem_daily_cost(self, E_grid_data: np.ndarray):
        """Computes cost ratios of zeta and rbc from E_grid_data for 9 buildings. Instead of using get_cost_day_end()"""
        
        if isinstance(E_grid_data, list):
            E_grid_data = np.array(E_grid_data)

        
        ramping_cost = []
        peak_electricity_cost = []

        for bid in range(9):
            ramping_cost_t = []
            peak_electricity_cost_t = []
            E_grid_t = E_grid_data[:, bid]  #  24*1

            ramping_cost.append(np.sum(np.abs(E_grid_t[1:] - E_grid_t[:-1])))  # Size 9
            peak_electricity_cost.append(np.max(E_grid_t))  # Size 9

        total_cost = np.array(ramping_cost) + np.array(peak_electricity_cost)  # Size 9
        
        total_cost = np.reshape(total_cost, (1, self.buildings))
        
        E_grid_rbc_data = []

        # get RBC cost for doing rbc actions for one day to take ratios with the total cost observed above
        cs = deepcopy(self.sod)
        
        for t in range(24):
            next_state = self.Digital_Twin.transition(
                cs,
                self.agent_rbc.select_action(t),
                self.total_it - 24 + t,
                self.day_data[t],
                self.actor_digital_twin.zeta,
            )

            E_grid_rbc_data.append(
                next_state[:, 28]
            )  # Apeending Electricity demand to the E_grid_data

            cs = next_state
            
        if isinstance(E_grid_rbc_data, list):
            E_grid_rbc_data = np.array(E_grid_rbc_data)
        
#         print('rbc Egrid = ', E_grid_rbc_data)
        ramping_cost_rbc = []
        peak_electricity_cost_rbc = []

        for bid in range(9):
            ramping_cost_t_rbc = []
            peak_electricity_cost_t_rbc = []
            E_grid_t_rbc = E_grid_rbc_data[:, bid]  #  24*1

            ramping_cost_rbc.append(np.sum(np.abs(E_grid_t_rbc[1:] - E_grid_t_rbc[:-1])))  # Size 9
            peak_electricity_cost_rbc.append(np.max(E_grid_t_rbc))  # Size 9

        rbc_cost = np.array(ramping_cost_rbc) + np.array(peak_electricity_cost_rbc)  # Size 9
        
        rbc_cost = np.reshape(rbc_cost, (1, self.buildings))
        
        
        cost = np.divide(total_cost, rbc_cost)   # Size 9

        return cost
    

    def set_EliteSet_EliteSetPrev(self):
        """This function is called by get_zeta() - see first line in get_zeta(). After this function is called inside
        get_zeta, it updates the self.elite_set according to the value of self.k. Once the elite_set is updated inside this
        function, get_zeta can use self.elite_set to get the zeta- p_ele to be passed through the actor."""

        if self.k == 1:

            self.elite_set_prev = self.elite_set
            self.elite_set = []

        if self.k > self.N_samples:  # Enough samples of zeta collected

            # Finding best k samples according to cost y_k
            self.costs = np.array(
                self.costs
            )  # Converting self.costs to np.array   dimensions = k*1*9
            #             print(np.shape(self.costs))
            best_zeta_args = np.zeros(
                (self.k - 1, self.buildings)
            )  # Will store the arguments of the sort

            elite_set_dummy = self.elite_set

            for i in range(self.buildings):
                best_zeta_args[:, i] = np.argsort(self.costs[:, :, i], axis=0).reshape(
                    -1
                )  # Arranging costs for the i-th building

                # Finding the best K samples from the elite set
                for Kbest in range(self.K):
                    a = best_zeta_args[:, i][Kbest].astype(np.int32)
                    self.elite_set[Kbest][:, :, i] = elite_set_dummy[a][:, :, i]

            self.elite_set = self.elite_set[0 : self.K]

            self.mean_p_ele = [[]] * self.buildings
            self.std_p_ele = [[]] * self.buildings
            
            A = np.vstack(self.elite_set)
#             print(np.shape(A))
            

            for i in range(self.buildings):
                self.mean_p_ele[i] = np.mean(A[:, :, i], axis=0)
                #                 print('A = ',A[:,:,i])
                #                 print('A_mean = ', self.mean_p_ele[i])
                self.std_p_ele[i] = np.std(A[:, :, i], axis=0)
            
#             print(self.mean_p_ele)
            self.elite_set_prev = self.elite_set
            self.elite_set = []

            self.k = 1  # Reset the sample index

            self.costs = []

        elite_set = self.elite_set
        elite_set_prev = self.elite_set_prev

        eliteSet_eliteSetPrev = [elite_set, elite_set_prev]

        return eliteSet_eliteSetPrev

    def evaluate_cost(self, state):
        """Evaluate cost computed from current set of state and action using set of zetas previously supplied"""
        if self.total_it <= self.rbc_threshold:
            return
        
        if self.total_it % 24 == 0:
            self.sod = state

        E_observed = state[:, 28]  # Storing E_grid for all buildings

#         E_NS_t = state[:, 23]  # For all buildings

#         data_output = self.memory.get(-1)

#         C_bd_hist = data_output["C_bd"][
#             self.total_it % 24, :
#         ]  # For 9 buildings and current hour - np.array size - 1*9

#         H_bd_hist = data_output["H_bd"][
#             self.total_it % 24, :
#         ]  # For 9 buildings and 24 hours - np.array size - 1*9

#         COP_C_hist = data_output["COP_C"][
#             self.total_it % 24, :
#         ]  # For 9 buildings and 24 hours - np.array size - 1*9

#         self.eta_ehH_hist = [
#             0.9
#         ] * self.buildings  # For 9 buildings and 24 hours - list of 9 lists of size 24

        # Appending the current states to the day history list of states
        self.E_netelectric_hist.append(E_observed)  # List of 24 lists each list size 9

        if self.total_it % 24 == 0:  # Calculate cost at the end of the day

            cost = self.get_cem_daily_cost(self.E_netelectric_hist)
    
            self.costs.append(cost)

            self.all_costs.append(cost)
            
            self.elite_set.append(self.zeta)
#             print('Elite set shape = ', np.shape(self.elite_set))
            
            self.p_ele_logger.append(self.zeta)
            
            self.k = self.k + 1

            self.E_netelectric_hist = []

        self.mean_elite_set.append(self.mean_p_ele)

    def set_zeta(self, zeta=None):
        """Update zeta which will be supplied to `select_action`"""
        if zeta is None:
            zeta = self.get_zeta()  # put into actor

        if self.total_it >= self.rbc_threshold and self.total_it % 24 == 0:
#             self.elite_set.append(zeta)
#             print('Elite set shape = ', np.shape(self.elite_set))
            for i in range(self.buildings):
                zeta_tuple = (
                    zeta[0, :, i],
                    self.zeta_eta_bat[:, :, i],
                    self.zeta_eta_Hsto[:, :, i],
                    self.zeta_eta_Csto[:, :, i],
                    self.zeta_eta_ehH,
                    self.zeta_c_bat_end,
                )
                self.actor.set_zeta(zeta_tuple, i)   # Setting zeta for all the buildings
                

    def select_action(self, state, day_ahead: bool = False):
        """Overrides from `TD3`. Utilizes CEM and Digital Twin computations"""
        # update zeta
        self.set_zeta()
        
        # run forward pass
        actions, parameters = super().select_action(state, day_ahead)
        
        # For updating sod inside get_cem_daily_cost() as it is also using the dt to get the cost ratios
        self.cem_cost_debug(state, parameters)
        
        # evaluate agent
        self.evaluate_cost(state)
        
#         digital twin oracle testing for debugging
#         if self.total_it%24 >= self.rbc_threshold:
#             items = ["E_hpC_max", "E_ehH_max", "E_bat_max", "C_p_Hsto", "C_p_bat", "C_p_Csto", "E_pv", "H_bd", "C_bd",
#                      "COP_C", "C_max", "H_max", "E_ns", "E_pv"]
#             data_orc = self.oracle.get_current_data_oracle(self.env, self.total_it, None, None)  # Use this instead of params
#             for item in items:
#                 parameters[item] = np.zeros((24, 9))

#                 parameters[item][self.total_it % 24, :] = np.array(data_orc[item]) 
                
        # digital twin
        self.digital_twin_interface(state, parameters)
    
        return actions

    def select_action_debug(self, state, day_ahead: bool = False):

        """Overrides from `TD3`. Utilizes CEM and Digital Twin computations"""
        self.state_hist.append(state)
        # update zeta
        self.set_zeta()
        # run forward pass
        # actions, parameters = super().select_action(state, day_ahead)

        parameters = {}
        items = ["E_hpC_max", "E_ehH_max", "E_bat_max", "C_p_Hsto", "C_p_bat", "C_p_Csto", "E_pv", "H_bd", "C_bd",
                 "COP_C", "C_max", "H_max", "E_ns", "E_pv"]
        data_orc = self.oracle.get_current_data_oracle(self.env, self.total_it, None, None)  # Use this instead of params
        for item in items:
            parameters[item] = np.zeros((24, 9))
            if item == "E_bat_max":
                parameters[item][self.total_it % 24, :] = np.array(data_orc["C_p_bat"])
            else:
                parameters[item][self.total_it % 24, :] = np.array(data_orc[item])

        # DEBUGGING PURPOSE
        indx_hour = 2
        hour_state = np.array([[state[0][indx_hour]]])
        actions = self.agent_rbc.select_action(hour_state)
        # actions[2, :] = 0
        # actions[3, :] = 0
        # actions *= 0.1
        # evaluate agent
        # self.evaluate_cost(state)



        # digital twin
        if self.total_it >= self.rbc_threshold + 48 and self.total_it % 24 == 0:  # end of day, rerun with the digital twin for the past day
            initial_state = self.state_hist[self.total_it - 24]

            # get RBC cost for doing rbc actions for one day
            cs = deepcopy(initial_state)

            for t in range(24):
                self.E_grid_dt.append(
                    cs[:, 28]
                )
                actions_dt = self.agent_rbc.select_action(t + 1)
                # actions_dt[2, :] = 0
                # actions_dt[3, :] = 0
                next_state = self.Digital_Twin.transition(
                    cs,
                    actions_dt,
                    self.total_it - 24 + t,
                    self.day_data[t],
                    self.actor_digital_twin.zeta,
                )

                # self.E_grid_dt.append(
                #     next_state[:, 28]
                # )  # Appending Electricity demand to the E_grid_data
                self.dt_building_logger.append(self.Digital_Twin.buildings)
                self.c_soc_logger.append(cs[:, 25])
                self.h_soc_logger.append(cs[:, 26])
                self.e_soc_logger.append(cs[:, 27])

                cs = next_state

        elif self.total_it < self.rbc_threshold + 24:
            self.E_grid_dt.append(
                np.zeros(9)
            )
            self.dt_building_logger.append([])
            self.e_soc_logger.append([])
            self.h_soc_logger.append([])
            self.c_soc_logger.append([])


        self.update_hour_of_day_data(parameters, (self.total_it) % 24)

        self.total_it += 1

        return actions         
    
    # --------------------------- METHODS FOR DIGITAL TWIN ------------------------------------------------------------ #
    def update_hour_of_day_data(self, parameters: dict, t: int):
        """Updates state to start of day state. Function called only when start of day. Handled within `digital_twin_interface`"""
        self.day_data[t] = parameters
        
        
    def get_cost(self, E_grid_data: np.ndarray):
            """Computes cost from E_grid_data for 9 buildings for Digital Twin"""
            if isinstance(E_grid_data, list):
                E_grid_data = np.array(E_grid_data)

#             E_grid_data = E_grid_data[1:23,:]
    #         print(E_grid_data)
            ramping_cost = []
            peak_electricity_cost = []

            for bid in range(9):
                ramping_cost_t = []
                peak_electricity_cost_t = []
                E_grid_t = E_grid_data[:, bid]  #  24*1

                ramping_cost.append(np.sum(np.abs(E_grid_t[1:] - E_grid_t[:-1])))  # Size 9
                peak_electricity_cost.append(np.max(E_grid_t))  # Size 9


            total_cost = np.array(ramping_cost) + np.array(peak_electricity_cost)  # Size 9


            cost = total_cost          # Array of size 9
            
#             print('zeta_cost = ', cost)

            return cost

        
    
    def get_dt_cost(self, E_grid_data: np.ndarray):
        """This function is for debugging the DT using the data_oracle. It gives the
        aggregate ramping and peak electric costs for all the buildings combined. Not used for
        the CEM agent implementation"""
        if isinstance(E_grid_data, list):
            E_grid_data = np.array(E_grid_data)
        
        E_grid_t = np.sum(E_grid_data,
        axis=1,
    )
        
        ramping_cost = np.sum(np.abs(E_grid_t[1:] - E_grid_t[:-1]))
        peak_electricity_cost = np.max(E_grid_t)
        
#         print('dt ramping cost = ', ramping_cost)
#         print('dt peak_electricity_cost =  ', peak_electricity_cost)

        return ramping_cost, peak_electricity_cost
    

    def evaluate_zeta(self, current_state):
        """Main function to evaluate different values of Zeta for after the meta-episode is 
        complete"""

        E_grid_rbc_data = []

        # get RBC cost for doing rbc actions for one day
        cs = deepcopy(current_state)
        for t in range(24):
            next_state = self.Digital_Twin.transition(
                cs,
                self.agent_rbc.select_action(t),
                self.total_it - 24 + t,
                self.day_data[t],
                self.actor_digital_twin.zeta,
            )

            E_grid_rbc_data.append(
                next_state[:, 28]
            )  # Appending Electricity demand to the E_grid_data

            cs = next_state
            
#         self.E_grid_dt.append(E_grid_zeta_data)        # For debugging
        rbc_cost = self.get_cost(E_grid_rbc_data)

#         rbc_ramping_cost, rbc_peak_electricity_cost = self.get_dt_cost(E_grid_rbc_data)  # For debugging purposes
        

        # keep track of Optim/RBC ratios
        ratios = []       # Stroing ratios of zeta/RBC costs
#         ratios_dt = []    # For debugging
        
        all_zeta_costs = []
        
#         Evaluation of 4 different defined zetas using the DT costs which are the total_zeta_cost/RBC_cost
#         for zeta in self.zeta_k_list:
        ######################################################
        # Zeta 1
        E_grid_zeta_data = []
        # aggregate data for 24 hour and store in E_grid_zeta_data
        cs = deepcopy(current_state)

        for t in range(24):
            self.set_zeta(self.zeta_k_list[0,:,:,:])

            actions, optim_values, _ = zip(
                *[
                    self.actor_digital_twin.forward(
                        t,
                        self.day_data[t],  # next_state
                        id,
                        dispatch=False,
                    )
                    for id in range(self.buildings)
                ]
            )

            next_state = self.Digital_Twin.transition(
                cs,
                actions,
                self.total_it - 24 + t,
                self.day_data[t],
                self.actor_digital_twin.zeta,
            )

            E_grid_zeta_data.append(
                next_state[:, 28]
            )  # Appending Electricity demand to the E_grid_data

            cs = next_state

#             print('zeta E_grid shape = ', np.shape(E_grid_zeta_data))
        zeta_cost = self.get_cost(E_grid_zeta_data)

        all_zeta_costs.append(zeta_cost)

        ratios.append(np.divide(zeta_cost, rbc_cost))     # Appending the ratio of costs for 9 buildings
        
        # Zeta 2
        E_grid_zeta_data = []
        # aggregate data for 24 hour and store in E_grid_zeta_data
        cs = deepcopy(current_state)

        for t in range(24):
            self.set_zeta(self.zeta_k_list[1,:,:,:])

            actions, optim_values, _ = zip(
                *[
                    self.actor_digital_twin.forward(
                        t,
                        self.day_data[t],  # next_state
                        id,
                        dispatch=False,
                    )
                    for id in range(self.buildings)
                ]
            )

            next_state = self.Digital_Twin.transition(
                cs,
                actions,
                self.total_it - 24 + t,
                self.day_data[t],
                self.actor_digital_twin.zeta,
            )

            E_grid_zeta_data.append(
                next_state[:, 28]
            )  # Appending Electricity demand to the E_grid_data

            cs = next_state

#             print('zeta E_grid shape = ', np.shape(E_grid_zeta_data))
        zeta_cost = self.get_cost(E_grid_zeta_data)

        all_zeta_costs.append(zeta_cost)

        ratios.append(np.divide(zeta_cost, rbc_cost))
                                
        # Zeta 3                        
        E_grid_zeta_data = []
        # aggregate data for 24 hour and store in E_grid_zeta_data
        cs = deepcopy(current_state)

        for t in range(24):
            self.set_zeta(self.zeta_k_list[2,:,:,:])

            actions, optim_values, _ = zip(
                *[
                    self.actor_digital_twin.forward(
                        t,
                        self.day_data[t],  # next_state
                        id,
                        dispatch=False,
                    )
                    for id in range(self.buildings)
                ]
            )

            next_state = self.Digital_Twin.transition(
                cs,
                actions,
                self.total_it - 24 + t,
                self.day_data[t],
                self.actor_digital_twin.zeta,
            )

            E_grid_zeta_data.append(
                next_state[:, 28]
            )  # Appending Electricity demand to the E_grid_data

            cs = next_state

#             print('zeta E_grid shape = ', np.shape(E_grid_zeta_data))
        zeta_cost = self.get_cost(E_grid_zeta_data)

        all_zeta_costs.append(zeta_cost)

        ratios.append(np.divide(zeta_cost, rbc_cost))
        
        # Zeta 4
        E_grid_zeta_data = []
        # aggregate data for 24 hour and store in E_grid_zeta_data
        cs = deepcopy(current_state)

        for t in range(24):
            self.set_zeta(self.zeta_k_list[3,:,:,:])

            actions, optim_values, _ = zip(
                *[
                    self.actor_digital_twin.forward(
                        t,
                        self.day_data[t],  # next_state
                        id,
                        dispatch=False,
                    )
                    for id in range(self.buildings)
                ]
            )

            next_state = self.Digital_Twin.transition(
                cs,
                actions,
                self.total_it - 24 + t,
                self.day_data[t],
                self.actor_digital_twin.zeta,
            )

            E_grid_zeta_data.append(
                next_state[:, 28]
            )  # Appending Electricity demand to the E_grid_data

            cs = next_state

#             print('zeta E_grid shape = ', np.shape(E_grid_zeta_data))
        zeta_cost = self.get_cost(E_grid_zeta_data)

        all_zeta_costs.append(zeta_cost)

        ratios.append(np.divide(zeta_cost, rbc_cost))
        
        
        
        ########################################################## 4 zetas evaluation done
            
        # Evaluating for the current zeta i.e. self.zeta
        
        E_grid_zeta_data = []
        # aggregate data for 24 hour and store in E_grid_zeta_data
        cs = deepcopy(current_state)

        for t in range(24):
            self.set_zeta(self.zeta)

            actions, optim_values, _ = zip(
                *[
                    self.actor_digital_twin.forward(
                        t,
                        self.day_data[t],  # next_state
                        id,
                        dispatch=False,
                    )
                    for id in range(self.buildings)
                ]
            )

            next_state = self.Digital_Twin.transition(
                cs,
                actions,
                self.total_it - 24 + t,
                self.day_data[t],
                self.actor_digital_twin.zeta,
            )

            E_grid_zeta_data.append(
                next_state[:, 28]
            )  # Appending Electricity demand to the E_grid_data

            cs = next_state
            
#         self.E_grid_dt.append(E_grid_zeta_data)        # For debugging
        
#         zeta_ramping_cost, zeta_peak_electricity_cost = self.get_dt_cost(E_grid_zeta_data)

#         ratios_dt.append(
#             0.5
#             * (
#                 zeta_peak_electricity_cost / rbc_peak_electricity_cost
#                 + zeta_ramping_cost/ rbc_ramping_cost
#             )
#         )
        
#         self.DT_costs.append(ratios_dt)       # Getting DT costs for comparison with the costs from the true environment
        
        zeta_cost = self.get_cost(E_grid_zeta_data)

        all_zeta_costs.append(zeta_cost)

        ratios.append(np.divide(zeta_cost, rbc_cost))     # Appending the ratio of costs for 9 buildings

        
        ratios = np.array(ratios)
        
        zeta_args_best = np.zeros(self.buildings)  
        
        zeta = np.zeros(((1,24,9)))
        
        cost_ratios = np.zeros(9)

        for i in range(self.buildings):
            
            zeta_args_best[i] = np.argmin(ratios[:,i])
            
            a = int(zeta_args_best[i])
            zeta[:,:,i] = self.zeta_k_list[a,:,:,i] 
            
            cost_ratios[i] = np.min(ratios[:,i])
            
        return zeta, cost_ratios
    
    def cem_cost_debug(self, current_state, parameters):
        '''To debug the Nonetype object error encountered and defining sod when using dt
        to get cem daily costs'''
        if self.total_it <= self.rbc_threshold:
            return

        # if self.total_it % 24 == 1:  # start of day
        self.update_hour_of_day_data(parameters, (self.total_it - 1) % 24)
        
    
    def digital_twin_interface(self, current_state, parameters):
        """Main interface for utilizing Digital Twin"""
        if self.total_it <= self.rbc_threshold:
            return

        # if self.total_it % 24 == 1:  # start of day
        self.update_hour_of_day_data(parameters, (self.total_it - 1) % 24)
        if self.total_it % 120 == 0:  # Change this on the basis of meta episode selected
            zeta, cost = self.evaluate_zeta(current_state)  
            
            for i in range(self.buildings):
                
                if cost[i] < self.all_costs[-1].squeeze(0)[i]:  # update zeta
                    self.zeta[:,:,i] = zeta[:,:,i]    
            
            self.set_zeta(self.zeta)

    # --------------------------- METHODS FOR DIGITAL TWIN ------------------------------------------------------------ #
