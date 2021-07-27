from copy import deepcopy
from TD3 import TD3
from digital_twin import DigitalTwin

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
            rbc_threshold=336,
        )

        observation_space = kwargs["observation_space"]

        # CEM Specific parameters
        self.N_samples = 10
        self.K = 5  # size of elite set
        self.K_keep = 3
        self.k = 1  # Initial sample index
        self.flag = 0
        self.all_costs = []

        self.p_ele_logger = []
        self.mean_elite_set = []

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

        self.zeta = []  # zeta for all buidling for 24 hours (24x9)

        self.zeta_eta_bat = np.ones(((1, 24, self.buildings)))
        self.zeta_eta_Hsto = np.ones(((1, 24, self.buildings)))
        self.zeta_eta_Csto = np.ones(((1, 24, self.buildings)))
        self.zeta_eta_ehH = 0.9
        self.zeta_c_bat_end = 0.1

        self.mean_p_ele = [[1]] * self.buildings
        self.std_p_ele = [[0.2]] * self.buildings
        self.range_p_ele = [0.1, 5]

        # Initialising the elite sets
        self.elite_set = (
            []
        )  # Storing best 5 zetas i.e. a list of 5 lists which are further a list of 24 lists of size 9
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

        # get actions from RBC at start of day
        self.rbc_actions = self.agent_rbc.select_action(0)
        # Store state for duration of day for digital twin Zeta evaluation
        self.day_data = [None] * 24

        # @Vanshaj, make sure you define this!
        self.zeta_k_list = np.ones(
            ((4, 1, 24, len(observation_space)))
        )  # 4 different Zetas.

        self.zeta_k_list[1, 0:13, :] = 0.2
        self.zeta_k_list[1, 13:19, :] = 5
        self.zeta_k_list[1, 19:23, :] = 0.2

        self.zeta_k_list[2, 0:6, :] = 0.2
        self.zeta_k_list[2, 7:19, :] = 5
        self.zeta_k_list[2, 20:23, :] = 0.2

        self.zeta_k_list[3, 0:5, :] = 0.2
        self.zeta_k_list[3, 11:17, :] = 2
        self.zeta_k_list[3, 22:23, :] = 0.2

    def get_zeta(self):  # Getting zeta for the 9 buildings for 24 hours
        """This function is used to get zeta for the actor. We set the zeta for the actor and do the forward pass to get actions. In our case
        we will only have p_ele as the zeta parameter. This get_zeta function calls the set_EliteSet_EliteSetPrev
        to get the elite_set and then selects zeta from that. Elite set stores the best zetas."""

        # Getting the elite_set and elite_set_prev
        elite_set_eliteset_prev = self.set_EliteSet_EliteSetPrev()

        if len(self.elite_set_prev) and self.k <= self.K_keep:

            # k-th best from elite_set_prev - zeta for all buildings
            self.zeta = self.elite_set_prev[-1]

            zeta_k = self.zeta  # zeta for 9 buildings for 24 hours

        else:

            # Initialising parameters for the rest of the day for 24 hrs for 9 buildings
            zeta_p_ele = np.zeros(((1, 24, self.buildings)))
            #             zeta_eta_ehH = np.zeros(((1, 24, self.buildings)))
            #             zeta_eta_bat = np.zeros(((1, 24, self.buildings)))
            #             zeta_c_bat_end = np.zeros(((1, 24, self.buildings)))
            #             zeta_eta_Hsto = np.zeros(((1, 24, self.buildings)))
            #             zeta_eta_Csto = np.zeros(((1, 24, self.buildings)))

            mean_sigma_range = (
                self.get_mean_sigma_range()
            )  # Getting a list of lists for mean, std and ranges

            for i in range(self.buildings):

                zeta_p_ele[:, :, i] = np.clip(
                    np.random.normal(
                        mean_sigma_range[0][i], mean_sigma_range[1][i], 24
                    ),
                    mean_sigma_range[2][0],
                    mean_sigma_range[2][1],
                )

            #                 self.zeta = np.vstack(
            #                     (
            #                         zeta_p_ele,
            #                         zeta_eta_bat,
            #                         zeta_eta_Hsto,
            #                         zeta_eta_Csto,
            #                         zeta_eta_ehH,
            #                         zeta_c_bat_end,
            #                     )
            #                 )

            self.zeta = zeta_p_ele

            zeta_k = self.zeta  # will set this zeta for the rest of the day

        self.p_ele_logger.append(zeta_k)
        self.elite_set.append(zeta_k)

        return zeta_k

    def get_mean_sigma_range(self):
        """This function is called to get the current mean, standard deviation and allowed range for the
        parameter p_ele. We can access these 3 quantities by calling this function."""

        # ADD ALL PARAMS
        mean_sigma_range = [self.mean_p_ele, self.std_p_ele, self.range_p_ele]

        return mean_sigma_range

    def get_cost_day_end(self):
        """This function calculates the cost at the end of each day after using certain zeta.
        This function is called at the end of each day. Cost is calculated using the recorded
        outputs/states from the environment in the past 24 hours using a certain value of zeta- p_ele."""

        # outputs act as the next_state that we get after taking actions
        #  outputs = {'E_netelectric_hist': E_netelectric_hist, 'E_NS_hist': E_NS_hist, 'C_bd_hist': C_bd_hist, 'H_bd_hist': H_bd_hist}
        # outputs includes the history of all observed states during the day

        cost = np.zeros((1, self.buildings))
        self.outputs["E_netelectric_hist"] = np.array(
            self.outputs["E_netelectric_hist"]
        )  # size 24*9
        self.outputs["E_NS_hist"] = np.array(self.outputs["E_NS_hist"])  # size 2*9
        self.outputs["eta_ehH_hist"] = np.array(
            self.outputs["eta_ehH_hist"]
        )  # size 9*24

        self.C_bd_hist = np.vstack(self.C_bd_hist)
        self.H_bd_hist = np.vstack(self.H_bd_hist)
        self.COP_C_hist = np.vstack(self.COP_C_hist)

        self.outputs["C_bd_hist"] = np.array(self.outputs["C_bd_hist"])
        self.outputs["H_bd_hist"] = np.array(self.outputs["H_bd_hist"])
        self.outputs["COP_C_hist"] = np.array(self.outputs["COP_C_hist"])

        for i in range(self.buildings):
            num = np.max(self.outputs["E_netelectric_hist"][:, i])

            C_bd_div_COP_C = np.divide(
                self.outputs["C_bd_hist"][:, i], self.outputs["COP_C_hist"][:, i]
            )

            H_bd_div_eta_ehH = self.outputs["H_bd_hist"][:, i] / self.zeta_eta_ehH

            den = np.max(
                self.outputs["E_NS_hist"][1, i] * np.ones((24, 1))
                + C_bd_div_COP_C
                + H_bd_div_eta_ehH
            )

            cost[:, i] = num / den

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

            A = np.hstack(self.elite_set)

            for i in range(self.buildings):
                self.mean_p_ele[i] = np.mean(A[:, :, i], axis=1)
                #                 print('A = ',A[:,:,i])
                #                 print('A_mean = ', self.mean_p_ele[i])
                self.std_p_ele[i] = np.std(A[:, :, i], axis=1)

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

        E_observed = state[:, 28]  # For all buildings

        E_NS_t = state[:, 23]  # For all buildings

        data_output = self.memory.get(-1)

        C_bd_hist = data_output["C_bd"][
            self.total_it % 24, :
        ]  # For 9 buildings and current hour - np.array size - 1*9

        H_bd_hist = data_output["H_bd"][
            self.total_it % 24, :
        ]  # For 9 buildings and 24 hours - np.array size - 1*9

        COP_C_hist = data_output["COP_C"][
            self.total_it % 24, :
        ]  # For 9 buildings and 24 hours - np.array size - 1*9

        self.eta_ehH_hist = [
            0.9
        ] * self.buildings  # For 9 buildings and 24 hours - list of 9 lists of size 24

        # Appending the current states to the day history list of states
        self.E_netelectric_hist.append(E_observed)  # List of 24 lists each list size 9
        self.E_NS_hist.append(E_NS_t)  # List of 24 lists each list of size 9
        self.C_bd_hist.append(C_bd_hist)
        self.H_bd_hist.append(H_bd_hist)
        self.COP_C_hist.append(COP_C_hist)

        if self.total_it % 24 == 23:  # Calculate cost at the end of the day

            self.outputs = {
                "E_netelectric_hist": self.E_netelectric_hist,
                "E_NS_hist": self.E_NS_hist,
                "C_bd_hist": self.C_bd_hist,
                "H_bd_hist": self.H_bd_hist,
                "COP_C_hist": self.COP_C_hist,
                "eta_ehH_hist": self.eta_ehH_hist,
            }  # List for observed states for the last 24 hours for the 9 buildings

            cost = self.get_cost_day_end()  # Calculating cost at the end of the day

            self.costs.append(cost)

            self.all_costs.append(cost)

            self.k = self.k + 1

            self.C_bd_hist = []

            self.E_netelectric_hist = []

            self.H_bd_hist = []

            self.COP_C_hist = []

            self.E_NS_hist = []

        self.mean_elite_set.append(self.mean_p_ele)

    def set_zeta(self, zeta=None):
        """Update zeta which will be supplied to `select_action`"""
        if zeta is None:
            zeta = self.get_zeta()  # put into actor

        if self.total_it >= self.rbc_threshold and self.total_it % 24 == 0:
            self.elite_set.append(zeta)
            for i in range(self.buildings):
                zeta_tuple = (
                    zeta[0, :, i],
                    self.zeta_eta_bat[:, :, i],
                    self.zeta_eta_Hsto[:, :, i],
                    self.zeta_eta_Csto[:, :, i],
                    self.zeta_eta_ehH,
                    self.zeta_c_bat_end,
                )
                self.actor.set_zeta(zeta_tuple, i)

    def select_action(self, state, day_ahead: bool = False):
        """Overrides from `TD3`. Utilizes CEM and Digital Twin computations"""
        # update zeta
        self.set_zeta()
        # run forward pass
        actions, parameters = super().select_action(state, day_ahead)
        # evaluate agent
        self.evaluate_cost(state)
        # digital twin
        self.digital_twin_interface(state, parameters)
        return actions

    # --------------------------- METHODS FOR DIGITAL TWIN ------------------------------------------------------------ #
    def update_hour_of_day_data(self, parameters: dict, t: int):
        """Updates state to start of day state. Function called only when start of day. Handled within `digital_twin_interface`"""
        self.day_data[t] = parameters

    def get_cost(self, E_grid_data: np.ndarray):
        """Computes cost from E_grid_data for 9 buildings"""
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

        cost = np.mean(total_cost)

        return cost

    def evaluate_zeta(self, current_state):
        """Main function to evaluate different values of Zeta for."""

        E_grid_rbc_data = []

        # get RBC cost for doing rbc actions for one day
        for t in range(24):

            next_state = self.Digital_Twin.transition(
                current_state,
                self.agent_rbc.select_action(t),
                self.total_it - 24 + t,
                self.day_data[t],
                self.actor_digital_twin.zeta,
            )

            E_grid_rbc_data.append(
                next_state[:, 28]
            )  # Apeending Electricity demand to the E_grid_data

            current_state = next_state

        rbc_cost = self.get_cost(E_grid_rbc_data)

        # keep track of Optim/RBC ratios
        ratios = []
        E_grid_zeta_data = []

        for zeta in self.zeta_k_list:
            # aggregate data for 24 hour and store in E_grid_zeta_data
            for t in range(24):
                self.set_zeta(zeta)
                actions, optim_values, _ = zip(
                    *[
                        self.actor_digital_twin.forward(
                            t,
                            self.day_data[t],
                            id,
                            dispatch=False,
                        )
                        for id in range(self.buildings)
                    ]
                )

                next_state = self.Digital_Twin.transition(
                    current_state,
                    actions,
                    self.total_it - 24 + t,
                    self.day_data[t],
                    self.actor_digital_twin.zeta,
                )

                E_grid_zeta_data.append(
                    next_state[:, 28]
                )  # Appending Electricity demand to the E_grid_data

                current_state = next_state

            zeta_cost = self.get_cost(E_grid_zeta_data)

            ratios.append(zeta_cost / rbc_cost)

            E_grid_zeta_data = []  # To store E_grids for the new zeta

        return self.zeta_k_list[np.argmin(ratios)], min(ratios)

    def digital_twin_interface(self, current_state, parameters):
        """Main interface for utilizing Digital Twin"""
        if self.total_it <= self.rbc_threshold:
            return

        # if self.total_it % 24 == 1:  # start of day
        self.update_hour_of_day_data(parameters, (self.total_it - 1) % 24)
        if self.total_it % 24 == 0:  # end of day
            zeta, cost = self.evaluate_zeta(current_state)
            if cost < self.all_costs[-1].mean():  # update zeta
                # all_costs, costs
                self.set_zeta(zeta)

    # --------------------------- METHODS FOR DIGITAL TWIN ------------------------------------------------------------ #
