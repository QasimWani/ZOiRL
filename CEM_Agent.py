from actor import Actor

class CEM_Agent(object):
    def __init__(self, self,building_ids,
            buildings_states_actions,
            building_info,
            observation_spaces,
            action_spaces):

        self.N_samples = 10
        self.K = 5    # size of elite set
        self.K_keep = 3
        self.k = 1   # Initial sample index
        self.flag = 0
        self.env = env


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

        self.building_ids = building_ids
        self.buildings_states_actions = buildings_states_actions
        self.building_info = building_info
        self.obsevation_spaces = observation_spaces
        self.action_spaces = action_spaces

        self.total_it = 0
        self.rbc_threshold = rbc_threshold

        def get_zeta(self):

            if len(self.elite_set_prev) and self.k <= self.K_keep:

                self.zeta = self.elite_set_prev[-1]    # k-th best from elite_set_prev

            else:

                # Initialising parameters for the rest of the day for 24 hrs.
                zeta_p_ele = np.zeros((1,24))
                zeta_eta_ehH = np.zeros((1,24))
                zeta_eta_bat = np.zeros((1,24))
                zeta_c_bat_end = np.zeros((1,24))
                zeta_eta_Hsto = np.zeros((1,24))
                zeta_eta_Csto = np.zeros((1,24))

                mean_sigma_range = get_mean_sigma_range()

                zeta_p_ele = np.clip(np.random.normal(mean_sigma_range[0][0],mean_sigma_range[0][1],24), mean_sigma_range[0][2][0], mean_sigma_range[0][2][1])
                zeta_eta_ehH = np.clip(np.random.normal(mean_sigma_range[1][0],mean_sigma_range[1][1],24), mean_sigma_range[1][2][0], mean_sigma_range[1][2][1])
                zeta_eta_bat = np.clip(np.random.normal(mean_sigma_range[2][0],mean_sigma_range[2][1],24), mean_sigma_range[2][2][0], mean_sigma_range[2][2][1])
                zeta_c_bat_end = np.clip(np.random.normal(mean_sigma_range[3][0],mean_sigma_range[3][1],24), mean_sigma_range[3][2][0], mean_sigma_range[3][2][1])
                zeta_eta_Hsto = np.clip(np.random.normal(mean_sigma_range[4][0],mean_sigma_range[4][1],24), mean_sigma_range[4][2][0], mean_sigma_range[4][2][1])
                zeta_eta_Csto = np.clip(np.random.normal(mean_sigma_range[5][0],mean_sigma_range[5][1],24), mean_sigma_range[5][2][0], mean_sigma_range[5][2][1])

                self.zeta = np.vstack((((((zeta_p_ele,
                                        zeta_eta_ehH,
                                        zeta_eta_bat,
                                        zeta_c_bat_end,
                                        zeta_eta_Hsto,
                                        zeta_eta_Csto))))))

                self.elite_set.append(self.zeta)

                zeta_k = self.zeta

            return zeta_k

        def get_mean_sigma_range(self):

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

            else:
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


            mean_sigma_range = [[mean_p_ele, std_p_ele, range_p_ele],
                                [mean_eta_ehH, std_eta_ehH, range_eta_ehH],
                                [mean_eta_bat, std_eta_bat, range_eta_bat],
                                [mean_c_bat_end, std_c_bat_end, range_c_bat_end],
                                [mean_eta_Hsto, std_eta_Hsto, range_eta_Hsto],
                                [mean_eta_Csto, std_eta_Csto, range_eta_Csto]]

            return mean_sigma_range

        def select_action(self,state):

            t =    # Figure this out

            zeta_k = self.get_zeta()

            zeta_k_dict = {'p_ele': zeta_k[0,t-1], 'eta_ehH': zeta_k[1,t-1], 'eta_bat': zeta_k[2,t-1], 'c_bat_end': zeta_k[3,t-1],
                          'eta_Hsto': zeta_k[4,t-1], 'eta_Csto': zeta_k[5,t-1]}

            action = Actor.forward(t, zeta_k_dict, building_id, debug=False, dispatch=False,)

            return action

        def cost_design(self, outputs):

            # outputs act as the next_state that we get after taking actions
#             outputs = {'E_netelectric_hist': E_netelectric_hist, 'E_NS_hist': E_NS_hist, 'C_bd_hist': C_bd_hist, 'H_bd_hist': H_bd_hist}

            num = max(outputs['E_netelectric_hist'])

            C_bd_div_COP_C = [i/j for i,j in zip(outputs['C_bd_hist'], outputs['COP_C_hist'])]
            H_bd_div_eta_ehH = [i/j for i,j in zip(outputs['H_bd_hist'], outputs['eta_ehH_hist'])]


            den = outputs['E_NS_hist'] + C_bd_div_COP_C + H_bd_div_eta_ehH

            cost = num/den

            self.costs.append(cost)

            self.k = self.k + 1

            return costs

        def set_EliteSet_EliteSetPrev(self, ):

            if self.k == 1:

                self.elite_set_prev = elite_set
                self.elite_set = []

            elif self.k < self.N_samples:

                elite_set.append(self.get_zeta)

            else:

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

            return elite_set
