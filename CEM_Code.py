import numpy as np
import pandas as pd
import math
from collections import deque
from copy import deepcopy
from citylearn import CityLearn
from pathlib import Path

import sys
import warnings
import utils
import time

import matplotlib as plot
import torch

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from actor import Actor

class CEM_actions(object):
    def __init__(self, N_samples:int = 10, K: int = 5, K_keep: int = 3,
                 k: int = 1, # Initial sample index
                 num_days: int = 30, flag:int = 0):

        self.zeta = np.zeros((6,24))
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
        self.elite_set = []
        self.elite_set_prev = []
        self.costs = []



    def CEM_Code(self):

        done = False



        for j in range(self.num_days):   # Loop for 30 days

            # Initialize mean and standard deviation-sigma for the actor parameters zeta
            if self.flag == 0:
                mean_p_ele = 1
                std_p_ele = 0.2
                range_p_ele = [0.1, 5]

                mean_eta_ehH = 0.9
                std_eta_ehH = 0.1
                range_eta_ehH = [0.7, 1.3]

                mean_eta_bat = 1
                std_eta_bat = 0.2
                range_eta_bat = [0.7, 1.3]

                mean_c_bat_end = 0.1
                std_c_bat_end = 0.1
                range_c_bat_end = [0.01, 0.5]

                mean_eta_Hsto = 1
                std_eta_Hsto = 0.2
                range_eta_Hsto = [0.7, 1.3]

                mean_eta_Csto = 1
                std_eta_Csto = 0.2
                range_eta_Csto = [0.7, 1.3]

            if len(self.elite_set_prev) and k <= K_keep:

                self.zeta = self.elite_set_prev[-1]    # k-th best from elite_set_prev

            else:

                # Initialising parameters for the rest of the day for 24 hrs.
                zeta_p_ele = np.zeros((1,24))
                zeta_eta_ehH = np.zeros((1,24))
                zeta_eta_bat = np.zeros((1,24))
                zeta_c_bat_end = np.zeros((1,24))
                zeta_eta_Hsto = np.zeros((1,24))
                zeta_eta_Csto = np.zeros((1,24))

                zeta_p_ele = np.clip(np.random.normal(mean_p_ele,std_p_ele,24), range_p_ele[0], range_p_ele[1])
                zeta_eta_ehH = np.clip(np.random.normal(mean_eta_ehH,std_eta_ehH,10), range_eta_ehH[0], range_eta_ehH[1])
                zeta_eta_bat = np.clip(np.random.normal(mean_eta_bat,std_eta_bat,10), range_eta_bat[0], range_eta_bat[1])
                zeta_c_bat_end = np.clip(np.random.normal(mean_c_bat_end,std_c_bat_end,10), range_c_bat_end[0], range_c_bat_end[1])
                zeta_eta_Hsto = np.clip(np.random.normal(mean_eta_Hsto,std_eta_Hsto,10), range_eta_Hsto[0], range_eta_Hsto[1])
                zeta_eta_Csto = np.clip(np.random.normal(mean_eta_Csto,std_eta_Csto,10), range_eta_Csto[0], range_eta_Csto[1])

                self.zeta = np.vstack((((((zeta_p_ele,
                                        zeta_eta_ehH,
                                        zeta_eta_bat,
                                        zeta_c_bat_end,
                                        zeta_eta_Hsto,
                                        zeta_eta_Csto))))))

                self.elite_set.append(self.zeta)

            # Initializing the set of states that will be observed after taking 24 actions during the day
            # Size of these lists will be 24 # Observations
            E_netelectric_hist = []
            E_NS_hist= []
            C_bd_hist = []
            H_bd_hist = []


            if j == 0:

                initial_state = env.reset()

                E_netelectric_hist.append(initial_state['electric_demand'])
                E_NS_hist.append(initial_state['E_NS'])
                C_bd_hist.append(initial_state['C_bd'])
                H_bd_hist,append(initial_state['H_bd'])

            else:

                initial_state = state_24  # last state observed at the end of the day, calculated from the upcoming hours for loop

            actions = [[] for d in range(24)]    # Initializing a list of lists for the actions

            actions_day_hist = []    # Initializing a list of lists for the actions for storage

            for t in range(24):

                action, _, _ = zip(*[Actor.forward(t, self.zeta, id, dispatch=False)
                for id in range(self.buildings)])

                next_state, reward, done, _ = env.step(action)

                E_netelectric_hist.append(next_state['electric_demand'])
                E_NS_hist.append(next_state['E_NS'])
                C_bd_hist.append(next_state['C_bd'])
                H_bd_hist,append(next_state['H_bd'])

                actions.append(action)

                if t == 23:   # Saving the states observed at the end of the day

                    state_24 = next_state


            actions_day_hist.append(actions)
            actions = [[] for d in range(24)]

            outputs = {'E_netelectric_hist': E_netelectric_hist, 'E_NS_hist': E_NS_hist, 'C_bd_hist': C_bd_hist, 'H_bd_hist': H_bd_hist}

            num = max(outputs['E_netelectric_hist'])

            C_bd_div_COP_C = [i/j for i,j in zip(outputs['C_bd_hist'], outputs['COP_C_hist'])]
            H_bd_div_eta_ehH = [i/j for i,j in zip(outputs['H_bd_hist'], outputs['eta_ehH_hist'])]


            den = outputs['E_NS_hist'] + C_bd_div_COP_C + H_bd_div_eta_ehH

            cost = num/den

            self.costs.append(cost)

            self.k = self.k + 1

            if self.k > self.N_samples:

                # Finding best k samples according to cost y_k

                self.costs = np.array(self.costs)
                best_zeta_args = np.argsort(self.costs)

                # Finding the best K samples from the elite set
                for Kbest in range(self.K):
                    elite_set_dummy = self.elite_set
                    self.elite_set[Kbest] = elite_set_dummy[best_zeta_args[Kbest]]


                self.elite_set = np.hstack(((((self.elite_set[0], self.elite_set[1], self.elite_set[2], self.elite_set[3], self.elite_set[4])))))
                # Fitting mean and standard deviation to the the elite set
                mean_p_ele = np.mean(self.elite_set[0,:], axis = 1)
                std_p_ele = np.std(self.elite_set[0,:], axis = 1)

                mean_eta_ehH = np.mean(self.elite_set[1,:], axis = 1)
                std_eta_ehH = np.std(self.elite_set[1,:], axis = 1)

                mean_eta_bat = np.mean(self.elite_set[2,:], axis = 1)
                std_eta_bat = np.std(self.elite_set[2,:], axis = 1)

                mean_c_bat_end = np.mean(self.elite_set[3,:], axis = 1)
                std_c_bat_end = np.std(self.elite_set[3,:], axis = 1)

                mean_eta_Hsto = np.mean(self.elite_set[4,:], axis = 1)
                std_eta_Hsto = np.std(self.elite_set[4,:], axis = 1)

                mean_eta_Csto = np.mean(self.elite_set[5,:], axis = 1)
                std_eta_Csto = np.std(self.elite_set[5,:], axis = 1)

                self.flag = 1

                self.k = 1    # Reset the sample index
                self.elite_set_prev = self.elite_set
                self.elite_set = []

        elite_set_actions_hist = [self.elite_set, actions_day_hist]

        return elite_set_actions


    def get_zeta(self, state):

        elite_set_actions_hist = self.CEM_code()

        elite_set = elite_set_actions_hist[0]
        zeta = elite_set[1]

        return zeta


    def get_actions(self, state, zeta):

        elite_set_actions_hist = self.CEM_code()

        actions = elite_set_actions_hist[1]

        return actions
