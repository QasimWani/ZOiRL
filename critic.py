import numpy as np
import cvxpy as cp

from copy import deepcopy
from collections import defaultdict

from utils import *


class Critic:  # Centralized for now.
    def __init__(
        self,
        num_buildings: int,
        num_actions: list,
        lambda_: float = 0.9,
        rho: float = 0.01,
    ):
        """One-time initialization. Need to call `create_problem` to initialize optimization model with params."""
        self.lambda_ = lambda_
        self.rho = rho  # critic update step size
        # Optim specific
        self.num_actions = num_actions
        self.constraints = []
        self.costs = []
        self.alpha_ramp = [1] * num_buildings
        self.alpha_peak1 = [1] * num_buildings
        self.alpha_peak2 = [1] * num_buildings
        # define problem - forward pass
        self.prob = [None] * 24  # template for each hour

    def create_problem(
        self, t: int, parameters: dict, zeta_target: dict, building_id: int
    ):
        """
        Solves reward warping layer per building as specified by `building_id`. Note: 0 based.
        -> Internal function. Used by `forward()` for solution to Reward Wrapping Layer (RWL).
        @Param:
        - `t` : hour to solve convex optimization for.
        - `parameters` : data (dict) from r <= t <= T following `Oracle.get_current_data_oracle` format.
        - `zeta_target` : set of differentiable parameters from actor target.
        - `building_id`: building index number (0-based).
        """
        t = np.clip(t, 0, 23)
        T = 24
        window = T - t
        # Reset data
        self.constraints = []
        self.costs = []
        self.t = t
        # -- define action space -- #
        bounds_high, bounds_low = np.vstack(
            [self.num_actions[building_id].high, self.num_actions[building_id].low]
        )
        if len(bounds_high) == 2:  # bug
            bounds_high = {
                "action_C": bounds_high[0],
                "action_H": None,
                "action_bat": bounds_high[1],
            }
            bounds_low = {
                "action_C": bounds_low[0],
                "action_H": None,
                "action_bat": bounds_low[1],
            }
        else:
            bounds_high = {
                "action_C": bounds_high[0],
                "action_H": bounds_high[1],
                "action_bat": bounds_high[2],
            }
            bounds_low = {
                "action_C": bounds_low[0],
                "action_H": bounds_low[1],
                "action_bat": bounds_low[2],
            }

        # -- define action space -- #

        # define parameters and variables

        ### --- Parameters ---
        p_ele = cp.Parameter(
            name="p_ele", shape=(window), value=zeta_target["p_ele"][t:, building_id]
        )

        E_grid_prevhour = cp.Parameter(
            name="E_grid_prevhour",
            value=parameters["E_grid_past"][t, building_id]
            if "E_gird" in parameters and len(parameters["E_grid"].shape) == 2
            else 0,  # used in day ahead dispatch, default E-grid okay
        )

        E_grid_pkhist = cp.Parameter(
            name="E_grid_pkhist",
            value=np.max(parameters["E_grid"][:t, building_id])
            if t > 0 and "E_grid" in parameters and len(parameters["E_grid"].shape) == 2
            else 0,
        )  # used in day ahead dispatch, default E-grid okay

        # max-min normalization of ramping_cost to downplay E_grid_sell weight.
        ramping_cost_coeff = cp.Parameter(
            name="ramping_cost_coeff",
            value=zeta_target["ramping_cost_coeff"][t, building_id],
        )

        # Loads
        E_ns = cp.Parameter(
            name="E_ns", shape=window, value=parameters["E_ns"][t:, building_id]
        )
        H_bd = cp.Parameter(
            name="H_bd", shape=window, value=parameters["H_bd"][t:, building_id]
        )
        C_bd = cp.Parameter(
            name="C_bd", shape=window, value=parameters["C_bd"][t:, building_id]
        )

        # PV generations
        E_pv = cp.Parameter(
            name="E_pv", shape=window, value=parameters["E_pv"][t:, building_id]
        )

        # Heat Pump
        COP_C = cp.Parameter(
            name="COP_C", shape=window, value=parameters["COP_C"][t:, building_id]
        )
        E_hpC_max = cp.Parameter(
            name="E_hpC_max", value=parameters["E_hpC_max"][t, building_id]
        )

        # Electric Heater
        eta_ehH = cp.Parameter(
            name="eta_ehH", value=zeta_target["eta_ehH"][t, building_id]
        )
        E_ehH_max = cp.Parameter(
            name="E_ehH_max", value=parameters["E_ehH_max"][t, building_id]
        )

        # Battery
        C_f_bat = cp.Parameter(
            name="C_f_bat", value=parameters["C_f_bat"][t, building_id]
        )
        C_p_bat = cp.Parameter(
            name="C_p_bat", value=zeta_target["C_p_bat"][t, building_id]
        )
        eta_bat = cp.Parameter(
            name="eta_bat", value=zeta_target["eta_bat"][t, building_id]
        )
        soc_bat_init = cp.Parameter(
            name="c_bat_init", value=zeta_target["c_bat_init"][building_id]
        )
        soc_bat_norm_end = cp.Parameter(
            name="c_bat_end", value=zeta_target["c_bat_end"][t, building_id]
        )

        # Heat (Energy->dhw) Storage
        C_f_Hsto = cp.Parameter(
            name="C_f_Hsto", value=parameters["C_f_Hsto"][t, building_id]
        )
        C_p_Hsto = cp.Parameter(
            name="C_p_Hsto", value=zeta_target["C_p_Hsto"][t, building_id]
        )
        eta_Hsto = cp.Parameter(
            name="eta_Hsto", value=zeta_target["eta_Hsto"][t, building_id]
        )
        soc_Hsto_init = cp.Parameter(
            name="c_Hsto_init", value=zeta_target["c_Hsto_init"][building_id]
        )

        # Cooling (Energy->cooling) Storage
        C_f_Csto = cp.Parameter(
            name="C_f_Csto", value=parameters["C_f_Csto"][t, building_id]
        )
        C_p_Csto = cp.Parameter(
            name="C_p_Csto", value=zeta_target["C_p_Csto"][t, building_id]
        )
        eta_Csto = cp.Parameter(
            name="eta_Csto", value=zeta_target["eta_Csto"][t, building_id]
        )
        soc_Csto_init = cp.Parameter(
            name="c_Csto_init", value=zeta_target["c_Csto_init"][building_id]
        )

        ### current actions
        current_action_bat = cp.Parameter(
            name="current_action_bat", value=parameters["action_bat"][t, building_id]
        )  # electric battery
        current_action_H = cp.Parameter(
            name="current_action_H", value=parameters["action_H"][t, building_id]
        )  # heat storage
        current_action_C = cp.Parameter(
            name="current_action_C", value=parameters["action_C"][t, building_id]
        )  # cooling storage
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
            name="SOC_Brelax", shape=(window)
        )  # electrical battery relaxation (prevents numerical infeasibilities)
        action_bat = cp.Variable(
            name="action_bat", shape=(window - 1)
        )  # electric battery

        SOC_H = cp.Variable(name="SOC_H", shape=(window))  # heat storage
        SOC_Hrelax = cp.Variable(
            name="SOC_Hrelax", shape=(window)
        )  # heat storage relaxation (prevents numerical infeasibilities)
        action_H = cp.Variable(name="action_H", shape=(window - 1))  # heat storage

        SOC_C = cp.Variable(name="SOC_C", shape=(window))  # cooling storage
        SOC_Crelax = cp.Variable(
            name="SOC_Crelax", shape=(window)
        )  # cooling storage relaxation (prevents numerical infeasibilities)
        action_C = cp.Variable(name="action_C", shape=(window - 1))  # cooling storage

        ### objective function
        ramping_cost = cp.abs(E_grid[0] - E_grid_prevhour) + cp.sum(
            cp.abs(E_grid[1:] - E_grid[:-1])
        )  # E_grid_t+1 - E_grid_t
        peak_net_electricity_cost = cp.max(
            cp.atoms.affine.hstack.hstack([*E_grid, E_grid_pkhist])
        )  # max(E_grid, E_gridpkhist)
        electricity_cost = cp.sum(p_ele * E_grid)
        selling_cost = -1e2 * cp.sum(
            E_grid_sell
        )  # not as severe as violating constraints

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
            ramping_cost_coeff.value * ramping_cost
            + 5 * peak_net_electricity_cost
            + electricity_cost
            + selling_cost
            + E_bal_relax_cost * 1e4
            + H_bal_relax_cost * 1e4
            + C_bal_relax_cost * 1e4
            + SOC_Brelax_cost * 1e4
            + SOC_Crelax_cost * 1e4
            + SOC_Hrelax_cost * 1e4
        )

        ### constraints

        # action constraints
        # self.constraints.append(action_bat[0] == current_action_bat)
        # self.constraints.append(action_H[0] == current_action_H)
        # self.constraints.append(action_C[0] == current_action_C)

        self.constraints.append(E_grid >= 0)
        self.constraints.append(E_grid_sell <= 0)

        # energy balance constraints

        # electricity balance
        self.constraints.append(
            E_pv[1:] + E_grid[1:] + E_grid_sell[1:] + E_bal_relax[1:]
            == E_ns[1:] + E_hpC[1:] + E_ehH[1:] + action_bat * C_p_bat
        )
        self.constraints.append(
            E_pv[0] + E_grid[0] + E_grid_sell[0] + E_bal_relax[0]
            == E_ns[0] + E_hpC[0] + E_ehH[0] + current_action_bat * C_p_bat
        )

        # heat balance
        self.constraints.append(
            E_ehH[1:] * eta_ehH + H_bal_relax[1:] == action_H * C_p_Hsto + H_bd[1:]
        )
        self.constraints.append(
            E_ehH[0] * eta_ehH + H_bal_relax[0] == current_action_H * C_p_Hsto + H_bd[0]
        )

        # cooling balance
        self.constraints.append(
            E_hpC[1:] * COP_C[1:] + C_bal_relax[1:] == action_C * C_p_Csto + C_bd[1:]
        )
        self.constraints.append(
            E_hpC[0] * COP_C[0] + C_bal_relax[0]
            == current_action_C * C_p_Csto + C_bd[0]
        )

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
        for i in range(1, window):
            self.constraints.append(
                SOC_bat[i]
                == (1 - C_f_bat) * SOC_bat[i - 1]
                + action_bat[i - 1] * eta_bat
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
                + action_H[i - 1] * eta_Hsto
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
                + action_C[i - 1] * eta_Csto
                + SOC_Crelax[i]
            )
        self.constraints.append(SOC_C >= 0)  # battery SOC bounds
        self.constraints.append(SOC_C <= 1)  # battery SOC bounds

        #### action constraints (limit to action-space)
        assert (
            len(bounds_high) == 3
        ), "Invalid number of bounds for actions - see dict defined in `Optim`"

        for high, low in zip(bounds_high.items(), bounds_low.items()):
            key, h, l = [*high, low[1]]
            if not (h and l):
                continue

            # heating action
            if key == "action_C":
                self.constraints.append(action_C <= h)
                self.constraints.append(action_C >= l)
            # cooling action
            elif key == "action_H":
                self.constraints.append(action_H <= h)
                self.constraints.append(action_H >= l)
            # Battery action
            elif key == "action_bat":
                self.constraints.append(action_bat <= h)
                self.constraints.append(action_bat >= l)

    def get_problem(
        self,
        t: int,
        parameters: dict,
        zeta_target: dict,
        building_id: int,
        debug: bool = False,
    ):
        """Returns raw problem. Calls `create_problem` if problem not defined. DPP for speed-up"""
        # Form objective.
        if self.prob[t % 24] is None:  # create problem
            self.create_problem(t, parameters, zeta_target, building_id)
            obj = cp.Minimize(*self.costs)
            # Form problem.
            self.prob[t % 24] = cp.Problem(obj, self.constraints)
        else:  # DPP
            self.inject_params(t, parameters, zeta_target, building_id)

    def inject_params(
        self, t: int, parameters: dict, zeta_target: dict, building_id: int
    ):
        """Sets parameter values - DPP"""
        assert (
            self.prob[t % 24] is not None
        ), "Problem must be defined to be able to use DPP."
        problem_parameters = self.prob[t % 24].param_dict

        ### --- Parameters ---
        problem_parameters["p_ele"].value = zeta_target["p_ele"][t:, building_id]

        # used in day ahead dispatch, default E-grid okay
        problem_parameters["E_grid_prevhour"].value = (
            parameters["E_grid_past"][t, building_id]
            if "E_gird" in parameters and len(parameters["E_grid"].shape) == 2
            else 0
        )

        problem_parameters["E_grid_pkhist"].value = (
            np.max(parameters["E_grid"][:t, building_id])
            if t > 0 and "E_grid" in parameters and len(parameters["E_grid"].shape) == 2
            else 0
        )  # used in day ahead dispatch, default E-grid okay

        # max-min normalization of ramping_cost to downplay E_grid_sell weight.
        self.prob[t % 24].constants()[0] = zeta_target["ramping_cost_coeff"][
            t, building_id
        ]

        # Loads
        problem_parameters["E_ns"].value = parameters["E_ns"][t:, building_id]
        problem_parameters["H_bd"].value = parameters["H_bd"][t:, building_id]
        problem_parameters["C_bd"].value = parameters["C_bd"][t:, building_id]

        # PV generations
        problem_parameters["E_pv"].value = parameters["E_pv"][t:, building_id]

        # Heat Pump
        problem_parameters["COP_C"].value = parameters["COP_C"][t:, building_id]
        problem_parameters["E_hpC_max"].value = parameters["E_hpC_max"][t, building_id]

        # Electric Heater
        problem_parameters["eta_ehH"].value = zeta_target["eta_ehH"][t, building_id]
        problem_parameters["E_ehH_max"].value = parameters["E_ehH_max"][t, building_id]

        # Battery
        problem_parameters["C_f_bat"].value = parameters["C_f_bat"][t, building_id]

        problem_parameters["C_p_bat"].value = zeta_target["C_p_bat"][t, building_id]

        problem_parameters["eta_bat"].value = zeta_target["eta_bat"][t, building_id]

        problem_parameters["c_bat_init"].value = zeta_target["c_bat_init"][building_id]

        problem_parameters["c_bat_end"].value = zeta_target["c_bat_end"][t, building_id]

        # Heat (Energy->dhw) Storage
        problem_parameters["C_f_Hsto"].value = parameters["C_f_Hsto"][t, building_id]

        problem_parameters["C_p_Hsto"].value = zeta_target["C_p_Hsto"][t, building_id]

        problem_parameters["eta_Hsto"].value = zeta_target["eta_Hsto"][t, building_id]

        problem_parameters["c_Hsto_init"].value = zeta_target["c_Hsto_init"][
            building_id
        ]

        # Cooling (Energy->cooling) Storage
        problem_parameters["C_f_Csto"].value = parameters["C_f_Csto"][t, building_id]

        problem_parameters["C_p_Csto"].value = zeta_target["C_p_Csto"][t, building_id]

        problem_parameters["eta_Csto"].value = zeta_target["eta_Csto"][t, building_id]

        problem_parameters["c_Csto_init"].value = zeta_target["c_Csto_init"][
            building_id
        ]

        ### current actions

        # electric battery
        problem_parameters["current_action_bat"].value = parameters["action_bat"][
            t, building_id
        ]

        # heat storage
        problem_parameters["current_action_H"].value = parameters["action_H"][
            t, building_id
        ]

        # cooling storage
        problem_parameters["current_action_C"].value = parameters["action_C"][
            t, building_id
        ]

        ## Update Parameters
        for key, prob_val in problem_parameters.items():
            self.prob[t % 24].param_dict[key].value = prob_val.value

    def get_constraints(self):
        """Returns constraints for problem"""
        return self.constraints

    def set_alphas(self, ramp, pk1, pk2):
        """Setter target alphas"""
        self.alpha_ramp = np.clip(ramp, 0.1, 2)
        self.alpha_peak1 = np.clip(pk1, 0.1, 2)
        self.alpha_peak2 = np.clip(pk2, 0.1, 2)

    def get_alphas(self):
        """Getter target alphas"""
        return np.array([self.alpha_ramp, self.alpha_peak1, self.alpha_peak2])

    def solve(
        self,
        t: int,
        parameters: dict,
        zeta_target: dict,
        building_id: int,
        debug: bool = False,
    ):
        """Computes optimal Q-value using RWL as objective function"""
        # computes Q-value for n-step in the future
        # Form and solve problem - automatically assigns to self.prob (DPP if problem already exists)
        self.get_problem(t, parameters, zeta_target, building_id)
        status = self.prob[t % 24].solve(
            verbose=debug, max_iters=1000
        )  # output of reward warping function
        if float("-inf") < status < float("inf"):
            return [
                self.prob[t % 24].var_dict["E_grid"].value,
                self.prob[t % 24].param_dict["E_grid_pkhist"].value,
                self.prob[t % 24].param_dict["E_grid_prevhour"].value,
            ]
        raise ValueError(f"Unbounded solution with status - {status}")

    def reward_warping_layer(
        self, timestep: int, optimal_values: list, building_id: int
    ):
        """Calculates Q-value"""
        (
            E_grid,
            E_grid_pkhist,
            E_grid_prevhour,
        ) = optimal_values  # building specific values

        peak_hist_cost = np.max([E_grid[timestep:].max(), E_grid_pkhist])
        ramping_cost = np.sum(E_grid[timestep:] - E_grid_prevhour)

        Q_value = (
            -self.alpha_ramp[building_id] * ramping_cost
            - self.alpha_peak1[building_id] * peak_hist_cost
            - self.alpha_peak2[building_id] * np.square(peak_hist_cost)
        )
        return Q_value

    def forward(
        self,
        t: int,
        parameters: dict,
        zeta_target: dict,
        building_id: int,
        debug=False,
    ):
        """Uses result of RWL to compute for clipped Q values"""

        Gt_tn = 0.0
        rewards = parameters["reward"][:, building_id]
        solution = self.solve(t, parameters, zeta_target, building_id, debug)
        for n in range(1, 24 - t):
            Q_value = self.reward_warping_layer(t + n, solution, building_id)
            Gt_tn += np.sum(rewards[t + 1 : t + n + 1]) + Q_value

        # compute TD(\lambda) rewards
        Gt_lambda = (1 - self.lambda_) * Gt_tn + np.sum(rewards[t:]) * self.lambda_ ** (
            24 - t - 1
        )

        return Gt_lambda

    def target_update(self, alphas_local: list):
        """Updates alphas given from L1 optimization"""
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
        parameters: dict,
        zeta_target: dict,
        building_id: int,
        debug: bool = False,
    ):
        """Computes min Q"""
        # shared zeta-target across both target critic
        Q1 = critic_target_1.forward(t, parameters, zeta_target, building_id, debug)
        Q2 = critic_target_2.forward(t, parameters, zeta_target, building_id, debug)
        return min(Q1, Q2)  # y_r

    def least_absolute_optimization(
        self,
        parameters: list,  # data collected within actor forward pass for MINI_BATCH (utils.py) number of updates
        zeta_target: dict,
        building_id: int,
        critic_target: list,
        debug: bool = False,
    ):
        """Define least-absolute optimization for generating optimal values for alpha_ramp,peak1/2."""
        # extract target Critic
        critic_target_1, critic_target_2 = critic_target

        ### variables
        alpha_ramp = cp.Variable(name="ramp")
        alpha_peak1 = cp.Variable(name="peak1")
        alpha_peak2 = cp.Variable(name="peak2")

        clipped_values = []  # length will be MINI_BATCH * 24. reshapes it to per day
        data = defaultdict(list)  # E_grid, E_gridpkhist, E_grid_prevhour over #days

        for day_params in parameters:
            # append daily data
            data["E_grid"].append(day_params["E_grid"][:, building_id])
            data["E_grid_pkhist"].append(0)  # pkhist at 0th hour is 0.
            data["E_grid_prevhour"].append(day_params["E_grid_past"][0, building_id])

            for r in range(24):
                y_r = self.obtain_target_Q(
                    critic_target_1,
                    critic_target_2,
                    r,
                    day_params,
                    zeta_target,
                    building_id,
                    debug,
                )
                clipped_values.append(y_r)

        # convert to ndarray
        clipped_values = np.array(clipped_values).reshape(
            len(parameters), 24
        )  # number of days, 24 hours
        data["E_grid"] = np.array(data["E_grid"]).reshape(clipped_values.shape)

        ### parameters
        E_grid = cp.Parameter(
            name="E_grid", shape=(clipped_values.shape), value=data["E_grid"]
        )
        E_grid_pkhist = cp.Parameter(
            name="E_grid_pkhist", shape=len(parameters), value=data["E_grid_pkhist"]
        )
        E_grid_prevhour = cp.Parameter(
            name="E_grid_prevhour",
            shape=len(parameters),
            value=data["E_grid_prevhour"],
        )

        y_t = cp.Parameter(
            name="y_r",
            shape=(clipped_values.shape),
            value=clipped_values,
        )

        #### cost
        self.cost = []
        for i in range(len(parameters)):
            ramping_cost = cp.abs(E_grid[i][0] - E_grid_prevhour[i]) + cp.sum(
                cp.abs(E_grid[i][1:] - E_grid[i][:-1])
            )  # E_grid_t+1 - E_grid_t

            peak_net_electricity_cost = cp.max(
                cp.atoms.affine.hstack.hstack([*E_grid[i], E_grid_pkhist[i]])
            )  # max(E_grid, E_gridpkhist)

            # L1 norm https://docs.google.com/document/d/1QbqCQtzfkzuhwEJeHY1-pQ28disM13rKFGTsf8dY8No/edit?disco=AAAAMzPtZMU
            self.cost.append(
                cp.sum(
                    cp.abs(
                        alpha_ramp * ramping_cost
                        + alpha_peak1 * peak_net_electricity_cost
                        + alpha_peak2 * cp.square(peak_net_electricity_cost)
                        - y_t[i]
                    )
                )
            )

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
        obj = cp.Minimize(cp.sum(self.cost))
        # Form and solve problem.
        prob = cp.Problem(obj, self.constraints)

        self.debug_l1 = prob

        optim_solution = prob.solve()

        assert (
            float("-inf") < optim_solution < float("inf")
        ), "Unbounded solution/primal infeasable"

        solution = {}
        for var in prob.variables():
            solution[var.name()] = var.value

        return solution

    # LOCAL critic update
    def backward(
        self,
        batch_parameters_1: list,  # data collected within actor forward pass - Critic 1 (sequential)
        batch_parameters_2: list,  # data collected within actor forward pass - Critic 2 (random)
        zeta_target: dict,
        t: int,
        building_id: int,
        critic_local: list,
        critic_target: list,
        debug: bool = False,
    ):
        # extract local critic
        critic_local_1, critic_local_2 = critic_local
        # Compute L1 Optimization for Critic Local 1 (using sequential data) and Critic Local 2 (using random data) using Critic Target 1 and 2
        local_1_solution = self.least_absolute_optimization(
            batch_parameters_1, zeta_target, building_id, critic_target
        )
        local_2_solution = self.least_absolute_optimization(
            batch_parameters_2, zeta_target, building_id, critic_target
        )
        # update alphas for local
        critic_local_1.alpha_ramp[building_id] = local_1_solution["ramp"]
        critic_local_1.alpha_peak1[building_id] = local_1_solution["peak1"]
        critic_local_1.alpha_peak2[building_id] = local_1_solution["peak2"]

        critic_local_2.alpha_ramp[building_id] = local_2_solution["ramp"]
        critic_local_2.alpha_peak1[building_id] = local_2_solution["peak1"]
        critic_local_2.alpha_peak2[building_id] = local_2_solution["peak2"]
