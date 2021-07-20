from TD3 import TD3
import numpy as np


class Agent(TD3):
    """CEM Agent - inherits TD3 as agent"""

    def __init__(self, **kwargs):
        """Initialize Agent"""
        super().__init__(
            num_actions=kwargs["action_spaces"],
            num_buildings=len(kwargs["building_ids"]),
            rbc_threshold=336,
        )

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

        # Initialising the elite sets
        self.elite_set = (
            []
        )  # Storing best 5 zetas i.e. a list of 5 lists which are further a list of 24 lists of size 9
        self.elite_set_prev = []  # Same format as elite_set

        # Initialising the list of costs after using certain params zetas
        self.costs = []

    def get_zeta(self):  # Getting zeta for the 9 buildings for 24 hours

        if len(self.elite_set_prev) and self.k <= self.K_keep:

            # k-th best from elite_set_prev - zeta for all buildings
            self.zeta = self.elite_set_prev[self.k]

            zeta_k = self.zeta  # zeta for 9 buildings for 24 hours

        else:

            # Initialising parameters for the rest of the day for 24 hrs for 9 buildings
            zeta_p_ele = np.zeros(((1, 24, 9)))
            zeta_eta_ehH = np.zeros(((1, 24, 9)))
            zeta_eta_bat = np.zeros(((1, 24, 9)))
            zeta_c_bat_end = np.zeros(((1, 24, 9)))
            zeta_eta_Hsto = np.zeros(((1, 24, 9)))
            zeta_eta_Csto = np.zeros(((1, 24, 9)))

            mean_sigma_range = (
                self.get_mean_sigma_range()
            )  # Getting a list of lists for mean, std and ranges

            for i in range(9):

                zeta_p_ele[:, :, i] = np.clip(
                    np.random.normal(
                        mean_sigma_range[0][i], mean_sigma_range[1][i], 24
                    ),
                    mean_sigma_range[2][0],
                    mean_sigma_range[2][1],
                )

                zeta_eta_ehH[:, :, i] = np.clip(
                    np.random.normal(
                        mean_sigma_range[1][0][i], mean_sigma_range[1][1][i], 24
                    ),
                    mean_sigma_range[1][2][0],
                    mean_sigma_range[1][2][1],
                )
                zeta_eta_bat[:, :, i] = np.clip(
                    np.random.normal(
                        mean_sigma_range[2][0][i], mean_sigma_range[2][1][i], 24
                    ),
                    mean_sigma_range[2][2][0],
                    mean_sigma_range[2][2][1],
                )
                zeta_c_bat_end[:, :, i] = np.clip(
                    np.random.normal(
                        mean_sigma_range[3][0][i], mean_sigma_range[3][1][i], 24
                    ),
                    mean_sigma_range[3][2][0],
                    mean_sigma_range[3][2][1],
                )
                zeta_eta_Hsto[:, :, i] = np.clip(
                    np.random.normal(
                        mean_sigma_range[4][0][i], mean_sigma_range[4][1][i], 24
                    ),
                    mean_sigma_range[4][2][0],
                    mean_sigma_range[4][2][1],
                )
                zeta_eta_Csto[:, :, i] = np.clip(
                    np.random.normal(
                        mean_sigma_range[5][0][i], mean_sigma_range[5][1][i], 24
                    ),
                    mean_sigma_range[5][2][0],
                    mean_sigma_range[5][2][1],
                )

                self.zeta = np.vstack(
                    (
                        zeta_p_ele,
                        zeta_eta_bat,
                        zeta_eta_Hsto,
                        zeta_eta_Csto,
                        zeta_eta_ehH,
                        zeta_c_bat_end,
                    )
                )

            self.zeta = zeta_p_ele

            zeta_k = self.zeta  # will set this zeta for the rest of the day

        self.p_ele_logger.append(zeta_k)

        return zeta_k

    def get_mean_sigma_range(self):

        if self.flag == 0:

            mean_p_ele = [1] * 9
            std_p_ele = [0.2] * 9
            range_p_ele = [0.1, 5]

            mean_eta_ehH = [0.9] * 9
            std_eta_ehH = [0.1] * 9
            range_eta_ehH = [0.7, 1.3]

            mean_eta_bat = [1] * 9
            std_eta_bat = [0.2] * 9
            range_eta_bat = [0.7, 1.3]

            mean_c_bat_end = [0.1] * 9
            std_c_bat_end = [0.1] * 9
            range_c_bat_end = [0.01, 0.5]

            mean_eta_Hsto = [1] * 9
            std_eta_Hsto = [0.2] * 9
            range_eta_Hsto = [0.7, 1.3]

            mean_eta_Csto = [1] * 9
            std_eta_Csto = [0.2] * 9
            range_eta_Csto = [0.7, 1.3]

        mean_sigma_range = [self.mean_p_ele, self.std_p_ele, self.range_p_ele]

        #         mean_sigma_range = [[mean_p_ele, std_p_ele, range_p_ele],
        #                             [mean_eta_ehH, std_eta_ehH, range_eta_ehH],
        #                             [mean_eta_bat, std_eta_bat, range_eta_bat],
        #                             [mean_c_bat_end, std_c_bat_end, range_c_bat_end],
        #                             [mean_eta_Hsto, std_eta_Hsto, range_eta_Hsto],
        #                             [mean_eta_Csto, std_eta_Csto, range_eta_Csto]]

        return mean_sigma_range
