from os import name
from predictor import DataLoader, Oracle, Predictor
import numpy as np
import cvxpy as cp

from utils import *


class Critic:
    def __init__(
        self,
        num_buildings: int,
        lambda_: float = 0.9,
        rho: float = 0.1,
        n_step: int = 2,
    ):
        """One-time initialization. Need to call `create_problem` to initialize optimization model with params."""
        self.lambda_ = lambda_
        self.rho = rho  # critic update step size
        self.n = n_step  # td lambda action-value step size
        # Optim specific
        self.constraints = []
        self.costs = []
        self.alpha_ramp = [1] * num_buildings
        self.alpha_peak1 = [1] * num_buildings
        self.alpha_peak2 = [1] * num_buildings

    def create_problem(self, t: int, parameters: dict, building_id: int):
        """
        Solves reward warping layer per building as specified by `building_id`. Note: 0 based.
        -> Internal function. Used by `forward()` for solution to Reward Wrapping Layer (RWL).
        @Param:
        - `t` : hour to solve convex optimization for.
        - `parameters` : data (dict) from r <= t <= T following `get_current_data` format.
        - `building_id`: building index number (0-based).
        """
        # reset data every call.
        t = np.clip(t, 0, 23)
        # assert (
        #     0 <= t < 24
        # ), "Invalid timestep. Need to be a valid hour of the day [0 - 23]"
        T = 24
        window = T - t
        self.constraints = []
        self.costs = []
        self.t = t

        ### --- Parameters ---

        # Weights
        alpha_ramp = cp.Parameter(name="ramp", value=self.alpha_ramp[building_id])

        alpha_peak1 = cp.Parameter(name="peak1", value=self.alpha_peak1[building_id])

        alpha_peak2 = cp.Parameter(name="peak2", value=self.alpha_peak1[building_id])

        # Electricity grid
        E_grid_prevhour = cp.Parameter(
            name="E_grid_prevhour",
            value=parameters["E_grid"][max(t - 1, 0), building_id]
            if "E_gird" in parameters and len(parameters["E_grid"].shape) == 2
            else 0,
        )

        E_grid_pkhist = cp.Parameter(
            name="E_grid_pkhist",
            value=np.max(parameters["E_grid"][:, building_id])
            if "E_grid" in parameters and len(parameters["E_grid"].shape) == 2
            else 0,
        )

        # Loads
        E_ns = cp.Parameter(
            name="E_ns", shape=(window), value=parameters["E_ns"][t:, building_id]
        )
        H_bd = cp.Parameter(
            name="H_bd", shape=(window), value=parameters["H_bd"][t:, building_id]
        )
        C_bd = cp.Parameter(
            name="C_bd", shape=(window), value=parameters["C_bd"][t:, building_id]
        )

        # PV generations
        E_pv = cp.Parameter(
            name="E_pv", shape=(window), value=parameters["E_pv"][t:, building_id]
        )

        # Heat Pump
        COP_C = cp.Parameter(
            name="COP_C", shape=(window), value=parameters["COP_C"][t:, building_id]
        )
        E_hpC_max = cp.Parameter(
            name="E_hpC_max", value=parameters["E_hpC_max"][t, building_id]
        )

        # Electric Heater
        eta_ehH = cp.Parameter(
            name="eta_ehH", value=parameters["eta_ehH"][t, building_id]
        )
        E_ehH_max = cp.Parameter(
            name="E_ehH_max", value=parameters["E_ehH_max"][t, building_id]
        )

        # Battery
        C_f_bat = cp.Parameter(
            name="C_f_bat", value=parameters["C_f_bat"][t, building_id]
        )
        C_p_bat = parameters["C_p_bat"][t, building_id]
        eta_bat = cp.Parameter(
            name="eta_bat", value=parameters["eta_bat"][t, building_id]
        )
        soc_bat_init = cp.Parameter(
            name="soc_bat_init", value=parameters["c_bat_init"][building_id]
        )
        soc_bat_norm_end = cp.Parameter(
            name="soc_bat_norm_end", value=parameters["c_bat_end"][t, building_id]
        )

        # Heat (Energy->dhw) Storage
        C_f_Hsto = cp.Parameter(
            name="C_f_Hsto", value=parameters["C_f_Hsto"][t, building_id]
        )  # make constant.
        C_p_Hsto = cp.Parameter(
            name="C_p_Hsto", value=parameters["C_p_Hsto"][t, building_id]
        )
        eta_Hsto = cp.Parameter(
            name="eta_Hsto", value=parameters["eta_Hsto"][t, building_id]
        )
        soc_Hsto_init = cp.Parameter(
            name="soc_Hsto_init", value=parameters["c_Hsto_init"][building_id]
        )

        # Cooling (Energy->cooling) Storage
        C_f_Csto = cp.Parameter(
            name="C_f_Csto", value=parameters["C_f_Csto"][t, building_id]
        )
        C_p_Csto = cp.Parameter(
            name="C_p_Csto", value=parameters["C_p_Csto"][t, building_id]
        )
        eta_Csto = cp.Parameter(
            name="eta_Csto", value=parameters["eta_Csto"][t, building_id]
        )
        soc_Csto_init = cp.Parameter(
            name="soc_Csto_init", value=parameters["c_Csto_init"][building_id]
        )
        ### --- Variables ---

        # relaxation variables - prevents numerical failures when solving optimization
        E_bal_relax = cp.Variable(
            name="E_bal_relax", shape=(window)
        )  # electricity balance relaxation
        H_bal_relax = cp.Variable(
            name="H_bal_relax", shape=(window)
        )  # heating balance relaxation
        C_bal_relax = cp.Variable(
            name="C_bal_relax", shape=(window)
        )  # cooling balance relaxation

        E_grid = cp.Variable(name="E_grid", shape=(window))  # net electricity grid
        E_grid_sell = cp.Variable(
            name="E_grid_sell", shape=(window)
        )  # net electricity grid

        E_hpC = cp.Variable(name="E_hpC", shape=(window))  # heat pump
        E_ehH = cp.Variable(name="E_ehH", shape=(window))  # electric heater

        SOC_bat = cp.Variable(name="SOC_bat", shape=(window))  # electric battery
        SOC_Brelax = cp.Variable(
            name="SOH_Brelax", shape=(window)
        )  # electrical battery relaxation (prevents numerical infeasibilities)

        action_bat = cp.Variable(name="action_bat", shape=(window))
        action_H = cp.Variable(name="action_H", shape=(window))
        action_C = cp.Variable(name="action_C", shape=(window))

        current_action_bat = parameters["action_bat"][
            t, building_id
        ]  # electric battery
        current_action_H = parameters["action_H"][t, building_id]  # heat storage
        current_action_C = parameters["action_C"][t, building_id]  # cooling storage

        SOC_H = cp.Variable(name="SOC_H", shape=(window))  # heat storage
        SOC_Hrelax = cp.Variable(
            name="SOH_Crelax", shape=(window)
        )  # heat storage relaxation (prevents numerical infeasibilities)

        SOC_C = cp.Variable(name="SOC_C", shape=(window))  # cooling storage
        SOC_Crelax = cp.Variable(
            name="SOC_Crelax", shape=(window)
        )  # cooling storage relaxation (prevents numerical infeasibilities)

        ### objective function

        ramping_cost = cp.abs(E_grid[0] - E_grid_prevhour) + cp.sum(
            cp.abs(E_grid[1:] - E_grid[:-1])
        )  # E_grid_t+1 - E_grid_t

        peak_net_electricity_cost = cp.max(
            cp.atoms.affine.hstack.hstack([*E_grid, E_grid_pkhist])
        )  # max(E_grid, E_gridpkhist)
        # peak_net_electricity_cost = cp.atoms.elementwise.maximum.maximum(
        #     *E_grid, E_grid_pkhist
        # )

        # https://docs.google.com/document/d/1QbqCQtzfkzuhwEJeHY1-pQ28disM13rKFGTsf8dY8No/edit?disco=AAAAMzPtZMU
        reward_func = (
            -alpha_ramp.value * ramping_cost
            - alpha_peak1.value * peak_net_electricity_cost
            - alpha_peak2.value * peak_net_electricity_cost
        )

        ### relaxation costs - L1 norm
        # balance eq.
        E_bal_relax_cost = cp.sum(cp.abs(E_bal_relax))
        H_bal_relax_cost = cp.sum(cp.abs(H_bal_relax))
        C_bal_relax_cost = cp.sum(cp.abs(C_bal_relax))
        # soc eq.
        SOC_Brelax_cost = cp.sum(cp.abs(SOC_Brelax))
        SOC_Crelax_cost = cp.sum(cp.abs(SOC_Crelax))
        SOC_Hrelax_cost = cp.sum(cp.abs(SOC_Hrelax))

        self.costs.append(
            reward_func
            - E_bal_relax_cost * 1e4
            - H_bal_relax_cost * 1e4
            - C_bal_relax_cost * 1e4
            - SOC_Brelax_cost * 1e4
            - SOC_Crelax_cost * 1e4
            - SOC_Hrelax_cost * 1e4
        )

        ### constraints

        # action constraints
        self.constraints.append(action_bat[0] == current_action_bat)
        self.constraints.append(action_H[0] == current_action_H)
        self.constraints.append(action_C[0] == current_action_C)

        # Net electricity consumption constraints
        self.constraints.append(E_grid >= 0)
        self.constraints.append(E_grid_sell <= 0)

        # energy balance constraints
        self.constraints.append(
            E_pv + E_grid + E_grid_sell + E_bal_relax
            == E_ns + E_hpC + E_ehH + action_bat * C_p_bat
        )  # electricity balance
        self.constraints.append(
            E_ehH * eta_ehH + H_bal_relax == action_H * C_p_Hsto + H_bd
        )  # heat balance

        self.constraints.append(
            E_hpC * COP_C + C_bal_relax == action_C * C_p_Csto + C_bd
        )  # cooling balance

        # heat pump constraints
        self.constraints.append(E_hpC <= E_hpC_max)  # maximum cooling
        self.constraints.append(E_hpC >= 0)  # constraint minimum cooling to positive
        # electric heater constraints
        self.constraints.append(E_ehH >= 0)  # constraint to PD
        self.constraints.append(E_ehH <= E_ehH_max)  # maximum limit

        # electric battery constraints
        self.constraints.append(
            SOC_bat[0]
            == (1 - C_f_bat) * soc_bat_init
            + current_action_bat * eta_bat
            + SOC_Crelax[0]
        )  # initial SOC
        # soc updates
        for i in range(1, window):  # 1 = t + 1
            self.constraints.append(
                SOC_bat[i]
                == (1 - C_f_bat) * SOC_bat[i - 1]
                + action_bat[i] * eta_bat
                + SOC_Crelax[i]
            )
        self.constraints.append(
            SOC_bat[-1] == soc_bat_norm_end
        )  # soc terminal condition
        self.constraints.append(SOC_bat >= 0)  # battery SOC bounds
        self.constraints.append(SOC_bat <= 1)  # battery SOC bounds

        # Heat Storage constraints
        self.constraints.append(
            SOC_H[0]
            == (1 - C_f_Hsto) * soc_Hsto_init
            + current_action_H * eta_Hsto
            + SOC_Hrelax[0]
        )  # initial SOC
        # soc updates
        for i in range(1, window):
            self.constraints.append(
                SOC_H[i]
                == (1 - C_f_Hsto) * SOC_H[i - 1]
                + action_H[i] * eta_Hsto
                + SOC_Hrelax[i]
            )
        self.constraints.append(SOC_H >= 0)  # battery SOC bounds
        self.constraints.append(SOC_H <= 1)  # battery SOC bounds

        # Cooling Storage constraints
        self.constraints.append(
            SOC_C[0]
            == (1 - C_f_Csto) * soc_Csto_init
            + current_action_C * eta_Csto
            + SOC_Crelax[0]
        )  # initial SOC
        # soc updates
        for i in range(1, window):
            self.constraints.append(
                SOC_C[i]
                == (1 - C_f_Csto) * SOC_C[i - 1]
                + action_C[i] * eta_Csto
                + SOC_Crelax[i]
            )
        self.constraints.append(SOC_C >= 0)  # battery SOC bounds
        self.constraints.append(SOC_C <= 1)  # battery SOC bounds

    def get_problem(self):
        """Returns raw problem. Need to call `create_problem` first. Returns error otherwise"""
        # Form objective.
        assert (
            len(self.costs) == 1
        ), "Objective function not/ill-defined. Need to call `create_problem` before running forward pass"
        obj = cp.Maximize(*self.costs)
        # Form problem.
        prob = cp.Problem(obj, self.constraints)

        return prob

    def get_constraints(self):
        """Returns constraints for problem"""
        return self.constraints

    def set_alphas(self, ramp, pk1, pk2):
        """Setter target alphas"""
        self.alpha_ramp = ramp
        self.alpha_peak1 = pk1
        self.alpha_peak2 = pk2

    def get_alphas(self):
        """Getter target alphas"""
        return np.array([self.alpha_ramp, self.alpha_peak1, self.alpha_peak2])

    def solve(self, t: int, parameters: dict, building_id: int, debug: bool = False):
        """Computes optimal Q-value using RWL as objective function"""
        self.create_problem(
            t, parameters, building_id
        )  # computes Q-value for n-step in the future
        prob = self.get_problem()  # Form and solve problem
        Q_reward_warping_output = prob.solve(
            verbose=debug
        )  # output of reward warping function

        print("optimal solution", Q_reward_warping_output)
        if float("-inf") < Q_reward_warping_output < float("inf"):
            return Q_reward_warping_output  # , prob.variables()["E_grid"]
        return "Unbounded Solution"

    def forward(
        self,
        t: int,
        parameters: dict,
        building_id: int,
        debug=False,
    ):
        """Uses result of RWL to compute for clipped Q values"""

        Gt_tn = 0.0
        rewards = parameters["reward"][:, building_id]
        for n in range(1, 24 - t):
            Gt_tn += np.sum(rewards[t + 1 : t + n + 1]) + self.solve(
                t + n, parameters, building_id, debug
            )

        # compute TD(\lambda) rewards
        Gt_lambda = (1 - self.lambda_) * Gt_tn + np.sum(
            rewards[t + 1 :]
        ) * self.lambda_ ** (24 - t)

        return Gt_lambda

    def target_update(self, alphas_local: list):
        """Updates alphas given from least squares optimization"""
        assert (
            len(alphas_local) == 3
        ), f"Incorrect dimension passed. Alpha tuple should be of size 3. found {len(alphas_local)}"

        ### main target update
        alpha_ramp, alpha_peak1, alpha_peak2 = (
            self.rho * np.array(self.get_alphas())
            + (1 - self.rho) * alphas_local  # alphas_new comes from LS optim sol.
        )

        self.set_alphas(
            alpha_ramp, alpha_peak1, alpha_peak2
        )  # updated alphas! -- end of critic update


class Optim:
    """Performs Critic Update"""

    def __init__(self) -> None:
        pass

    def obtain_target_Q(
        self,
        critic_target_1: Critic,
        critic_target_2: Critic,
        t: int,
        parameters_1: dict,
        parameters_2: dict,
        building_id: int,
        debug: bool = False,
    ):
        """Computes min Q"""
        # shared data. difference only in alphas
        Q1 = critic_target_1.forward(t, parameters_1, building_id, debug)
        Q2 = critic_target_2.forward(t, parameters_2, building_id, debug)
        return min(Q1, Q2)  # y_r

    def backward(
        self,
        parameters_1: dict,  # data collected within actor forward pass - Critic 1 (sequential)
        parameters_2: dict,  # data collected within actor forward pass - Critic 2 (random)
        t: int,
        building_id: int,
        critic_local: list,
        critic_target: list,
        debug: bool = False,
    ):
        """
        Define least-squares optimization for generating optimal values for alpha_ramp,peak1/2.
        NOTE: Called only for local critic update
        """
        # extrapolate Critics
        critic_local_1, critic_local_2 = critic_local
        critic_target_1, critic_target_2 = critic_target

        ### variables
        alpha_ramp = cp.Variable(name="ramp")
        alpha_peak1 = cp.Variable(name="peak1")
        alpha_peak2 = cp.Variable(name="peak2")

        ### parameters
        E_grid = cp.Parameter(
            name="E_grid", shape=(24 - t), value=parameters_1["E_grid"][t:, building_id]
        )
        E_grid_pkhist = cp.Parameter(
            name="E_grid_pkhist",
            value=np.max(parameters_1["E_grid"][:, building_id]),
        )
        E_grid_prevhour = cp.Parameter(
            name="E_grid_prevhour",
            value=parameters_1["E_grid"][max(t - 1, 0), building_id],
        )

        y_r = cp.Parameter(
            name="y_r",
            value=self.obtain_target_Q(
                critic_target_1,
                critic_target_2,
                t,
                parameters_1,
                parameters_2,
                building_id,
                debug,
            ),
        )

        #### cost
        ramping_cost = cp.abs(E_grid[0] - E_grid_prevhour) + cp.sum(
            cp.abs(E_grid[1:] - E_grid[:-1])
        )  # E_grid_t+1 - E_grid_t

        peak_net_electricity_cost = cp.max(
            cp.atoms.affine.hstack.hstack([*E_grid, E_grid_pkhist])
        )  # max(E_grid, E_gridpkhist)

        # https://docs.google.com/document/d/1QbqCQtzfkzuhwEJeHY1-pQ28disM13rKFGTsf8dY8No/edit?disco=AAAAMzPtZMU
        self.cost = [
            cp.sum(
                cp.square(
                    alpha_ramp * ramping_cost
                    + alpha_peak1 * peak_net_electricity_cost
                    + alpha_peak2 * cp.square(peak_net_electricity_cost)
                    - y_r
                )
            )
        ]

        #### constraints
        self.constraints = []

        # alpha-peak
        self.constraints.append(alpha_ramp <= 2)
        self.constraints.append(alpha_ramp >= 0.1)

        # alpha-peak
        self.constraints.append(alpha_peak1 <= 2)
        self.constraints.append(alpha_peak1 >= 0.1)

        # alpha-peak
        self.constraints.append(alpha_peak2 <= 2)
        self.constraints.append(alpha_peak2 >= 0.1)

        # Form objective.
        obj = cp.Minimize(*self.cost)
        # Form and solve problem.
        prob = cp.Problem(obj, self.constraints)

        optim_solution = prob.solve()

        assert (
            float("-inf") < optim_solution < float("inf")
        ), "Unbounded solution/primal infeasable"

        solution = {}
        for var in prob.variables():
            solution[var.name()] = var.value

        critic_local_1.alpha_ramp = solution["ramp"]
        critic_local_1.alpha_peak1 = solution["peak1"]
        critic_local_1.alpha_peak2 = solution["peak2"]

        critic_local_2.alpha_ramp = solution["ramp"]
        critic_local_2.alpha_peak1 = solution["peak1"]
        critic_local_2.alpha_peak2 = solution["peak2"]
