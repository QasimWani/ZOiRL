from logger import LOG
import numpy as np
import cvxpy as cp


from collections import defaultdict

from utils import *


class Critic:  # decentralized version
    def __init__(
        self,
        num_buildings: int,
        num_actions: list,
        lambda_: float = 0.8,
        rho: float = 0.8,
    ):
        """One-time initialization. Need to call `create_problem` to initialize optimization model with params."""
        self.lambda_ = lambda_
        self.rho = rho  # critic update step size
        # Optim specific
        self.num_actions = num_actions
        self.constraints = []
        self.cost = None  # created at every call to `create_problem`. not used in DPP.

        self.alpha_ramp = [1] * num_buildings
        self.alpha_peak1 = [1] * num_buildings
        self.alpha_elec = [
            [1] * 24 for _ in range(num_buildings)
        ]  # virtual electricity cost

        # define problem - forward pass
        self.prob = [None] * 24  # template for each hour

        # q value for latest building recycled everyday - see `least_absolute_optimization`.
        # note that we don't need to creat 9 `Q_value` because we're clearing the data out
        # per building in `least_absolute_optimization` which is called from `Optim.backward()`
        self.Q_value = [None] * 24

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
        # t = np.clip(t, 0, 23)
        assert (
            0 <= t < 24
        ), f"timestep invalid range. needs to be 0 <= t < 24. found {t}"

        T = 24
        window = T - t
        # Reset data
        self.constraints = []
        # self.cost = None ### reassign to NONE. not needed.

        ### define constants
        C_f_bat = 0.00001
        C_f_Csto = 0.006
        C_f_Hsto = 0.008

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
            name="E_grid_prevhour", value=parameters["E_grid_prevhour"][t, building_id]
        )

        E_grid_pkhist = cp.Parameter(
            name="E_grid_pkhist",
            value=np.max([0, *parameters["E_grid"][:t, building_id]])
            if t > 0
            else max(E_grid_prevhour.value, 0),
        )  # used in day ahead dispatch, default E-grid okay

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
            name="eta_ehH", value=zeta_target["eta_ehH"][building_id]
        )
        E_ehH_max = cp.Parameter(
            name="E_ehH_max", value=parameters["E_ehH_max"][t, building_id]
        )

        # Battery
        C_p_bat = cp.Parameter(
            name="C_p_bat", value=parameters["C_p_bat"][t, building_id]
        )
        eta_bat = cp.Parameter(
            name="eta_bat", shape=window, value=zeta_target["eta_bat"][t:, building_id]
        )
        soc_bat_init = cp.Parameter(
            name="c_bat_init", value=parameters["c_bat_init"][t, building_id]
        )
        soc_bat_norm_end = cp.Parameter(
            name="c_bat_end", value=zeta_target["c_bat_end"][building_id]
        )

        # Heat (Energy->dhw) Storage
        C_p_Hsto = cp.Parameter(
            name="C_p_Hsto", value=parameters["C_p_Hsto"][t, building_id]
        )
        eta_Hsto = cp.Parameter(
            name="eta_Hsto",
            shape=window,
            value=zeta_target["eta_Hsto"][t:, building_id],
        )
        soc_Hsto_init = cp.Parameter(
            name="c_Hsto_init", value=parameters["c_Hsto_init"][t, building_id]
        )

        # Cooling (Energy->cooling) Storage
        C_p_Csto = cp.Parameter(
            name="C_p_Csto", value=parameters["C_p_Csto"][t, building_id]
        )
        eta_Csto = cp.Parameter(
            name="eta_Csto",
            shape=window,
            value=zeta_target["eta_Csto"][t:, building_id],
        )
        soc_Csto_init = cp.Parameter(
            name="c_Csto_init", value=parameters["c_Csto_init"][t, building_id]
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

        action_bat = cp.Variable(name="action_bat", shape=(window))  # electric battery

        SOC_H = cp.Variable(name="SOC_H", shape=(window))  # heat storage
        SOC_Hrelax = cp.Variable(
            name="SOC_Hrelax", shape=(window)
        )  # heat storage relaxation (prevents numerical infeasibilities)

        action_H = cp.Variable(name="action_H", shape=(window))  # heat storage

        SOC_C = cp.Variable(name="SOC_C", shape=(window))  # cooling storage
        SOC_Crelax = cp.Variable(
            name="SOC_Crelax", shape=(window)
        )  # cooling storage relaxation (prevents numerical infeasibilities)

        action_C = cp.Variable(name="action_C", shape=(window))  # cooling storage

        ### objective function
        ramping_cost = cp.abs(E_grid[0] - E_grid_prevhour)
        if window > 1:  # not at eod
            ramping_cost += cp.sum(
                cp.abs(E_grid[1:] - E_grid[:-1])
            )  # E_grid_t+1 - E_grid_t

        peak_net_electricity_cost = cp.max(
            cp.atoms.affine.hstack.hstack([*E_grid, E_grid_pkhist])
        )  # max(E_grid, E_gridpkhist)
        electricity_cost = cp.sum(p_ele * E_grid)
        selling_cost = -1e2 * cp.sum(
            E_grid_sell
        )  # not as severe as violating constraints

        ### relaxation costs - L2 norm
        # balance eq.
        E_bal_relax_cost = cp.sum(cp.abs(E_bal_relax))
        H_bal_relax_cost = cp.sum(cp.abs(H_bal_relax))
        C_bal_relax_cost = cp.sum(cp.abs(C_bal_relax))
        # soc eq.
        SOC_Brelax_cost = cp.sum(cp.abs(SOC_Brelax))
        SOC_Crelax_cost = cp.sum(cp.abs(SOC_Crelax))
        SOC_Hrelax_cost = cp.sum(cp.abs(SOC_Hrelax))

        self.cost = (
            0.1 * ramping_cost
            + 5 * peak_net_electricity_cost
            + electricity_cost
            + selling_cost
            + E_bal_relax_cost * 1e4
            + H_bal_relax_cost * 1e4
            + C_bal_relax_cost * 1e4
            + SOC_Brelax_cost * 1e4
            + SOC_Crelax_cost * 1e4
            + SOC_Hrelax_cost * 1e4
            + 1e-6
            * cp.sum(
                cp.square(E_bal_relax)
                + cp.square(H_bal_relax)
                + cp.square(C_bal_relax)
                + cp.square(E_grid)
                + cp.square(E_grid_sell)
                + cp.square(E_hpC)
                + cp.square(E_ehH)
                + cp.square(SOC_bat)
                + cp.square(SOC_Brelax)
                + cp.square(action_bat)
                + cp.square(SOC_H)
                + cp.square(SOC_Hrelax)
                + cp.square(action_H)
                + cp.square(SOC_C)
                + cp.square(SOC_Crelax)
                + cp.square(action_C)
            )
        )

        ### constraints

        # action constraints
        self.constraints.append(action_bat[0] == current_action_bat)
        self.constraints.append(action_H[0] == current_action_H)
        self.constraints.append(action_C[0] == current_action_C)

        self.constraints.append(E_grid >= 0)
        self.constraints.append(E_grid_sell <= 0)

        # energy balance constraints

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
            == (1 - C_f_bat) * soc_bat_init + action_bat[0] * eta_bat[0] + SOC_Brelax[0]
        )  # initial SOC
        # soc updates
        for i in range(1, window):
            self.constraints.append(
                SOC_bat[i]
                == (1 - C_f_bat) * SOC_bat[i - 1]
                + action_bat[i] * eta_bat[i]
                + SOC_Brelax[i]
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
            + action_H[0] * eta_Hsto[0]
            + SOC_Hrelax[0]
        )  # initial SOC
        # soc updates
        for i in range(1, window):
            self.constraints.append(
                SOC_H[i]
                == (1 - C_f_Hsto) * SOC_H[i - 1]
                + action_H[i] * eta_Hsto[i]
                + SOC_Hrelax[i]
            )
        self.constraints.append(SOC_H >= 0)  # battery SOC bounds
        self.constraints.append(SOC_H <= 1)  # battery SOC bounds

        # Cooling Storage constraints
        self.constraints.append(
            SOC_C[0]
            == (1 - C_f_Csto) * soc_Csto_init
            + action_C[0] * eta_Csto[0]
            + SOC_Crelax[0]
        )  # initial SOC
        # soc updates
        for i in range(1, window):
            self.constraints.append(
                SOC_C[i]
                == (1 - C_f_Csto) * SOC_C[i - 1]
                + action_C[i] * eta_Csto[i]
                + SOC_Crelax[i]
            )
        self.constraints.append(SOC_C >= 0)  # battery SOC bounds
        self.constraints.append(SOC_C <= 1)  # battery SOC bounds

        #### action constraints (limit to action-space)
        assert (
            len(bounds_high) == 3
        ), "Invalid number of bounds for actions - see dict defined in `Optim`"

        if window <= 1:  # eod. no actions to consider
            return

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
        return_prob: bool = False,
    ):
        """Returns raw problem. Calls `create_problem` if problem not defined. DPP for speed-up"""
        # Form objective.
        if self.prob[t % 24] is None:  # create problem
            self.create_problem(t, parameters, zeta_target, building_id)
            obj = cp.Minimize(self.cost)
            # Form problem.
            if return_prob:
                return cp.Problem(obj, self.constraints)

            self.prob[t % 24] = cp.Problem(obj, self.constraints)
        else:  # DPP
            prob = self.inject_params(t, parameters, zeta_target, building_id)
            if return_prob:
                return prob
            self.prob[t % 24] = prob

        assert self.prob[t % 24].is_dpp()

    def inject_params(
        self,
        t: int,
        parameters: dict,
        zeta_target: dict,
        building_id: int,
    ):
        """Sets parameter values - DPP"""
        assert (
            self.prob[t % 24] is not None
        ), "Problem must be defined to be able to use DPP."
        problem_parameters = self.prob[t % 24].param_dict

        ### --- Parameters ---
        problem_parameters["p_ele"].value = zeta_target["p_ele"][t:, building_id]

        problem_parameters["E_grid_prevhour"].value = parameters["E_grid_prevhour"][
            t, building_id
        ]

        problem_parameters["E_grid_pkhist"].value = (
            np.max([0, *parameters["E_grid"][:t, building_id]])
            if t > 0
            else max(0, parameters["E_grid_prevhour"][t, building_id])
        )

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
        problem_parameters["eta_ehH"].value = zeta_target["eta_ehH"][building_id]
        problem_parameters["E_ehH_max"].value = parameters["E_ehH_max"][t, building_id]

        # Battery
        problem_parameters["C_p_bat"].value = parameters["C_p_bat"][t, building_id]
        problem_parameters["eta_bat"].value = zeta_target["eta_bat"][t:, building_id]
        problem_parameters["c_bat_init"].value = parameters["c_bat_init"][
            t, building_id
        ]
        problem_parameters["c_bat_end"].value = zeta_target["c_bat_end"][building_id]

        # Heat (Energy->dhw) Storage
        problem_parameters["C_p_Hsto"].value = parameters["C_p_Hsto"][t, building_id]
        problem_parameters["eta_Hsto"].value = zeta_target["eta_Hsto"][t:, building_id]
        problem_parameters["c_Hsto_init"].value = parameters["c_Hsto_init"][
            t, building_id
        ]

        # Cooling (Energy->cooling) Storage
        problem_parameters["C_p_Csto"].value = parameters["C_p_Csto"][t, building_id]
        problem_parameters["eta_Csto"].value = zeta_target["eta_Csto"][t:, building_id]
        problem_parameters["c_Csto_init"].value = parameters["c_Csto_init"][
            t, building_id
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
        prob = self.prob[t % 24]
        for key, prob_val in problem_parameters.items():
            prob.param_dict[key].value = prob_val.value
        return prob

    def get_constraints(self):
        """Returns constraints for problem"""
        return self.constraints

    def set_alphas(self, ramp, pk1, elec):
        """Setter target alphas"""
        self.alpha_ramp = ramp
        self.alpha_peak1 = pk1
        self.alpha_elec = elec

    def get_alphas(self):
        """Getter target alphas"""
        return (
            np.array(self.alpha_ramp),
            np.array(self.alpha_peak1),
            np.array(self.alpha_elec),
        )

    def get(self, index):
        """Returns an element from Q-value array specified by `index`"""
        return self.Q_value[index]

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
        try:
            status = self.prob[t % 24].solve(
                solver="SCS",
                verbose=debug,  # max_iters=1_000_000
            )  # Returns the optimal value.
            assert float("-inf") < status < float("inf"), "Problem is infeasible."
        except:  # try another solver

            LOG(
                f"\nSolving critic using MAX_ITERS at t = {t} for building {building_id}"
            )
            status = self.prob[t].solve(
                solver="SCS", verbose=debug, max_iters=10_000_000
            )  # Returns the optimal value.

        if float("-inf") < status < float("inf"):
            return (
                self.prob[t % 24].var_dict["E_grid"].value,  # from Optim
                parameters["E_grid"],  # from env
                parameters["E_grid_prevhour"],  # from env
            )  # building specific sol.

        raise ValueError(f"Unbounded solution with status - {status}")

    def reward_warping_layer(
        self, timestep: int, parameters_E_grid: dict, building_id: int
    ):
        """Calculates Q-value"""
        E_grid, E_grid_true, E_grid_prevhour = parameters_E_grid

        E_grid_prevhour = E_grid_prevhour[timestep, building_id]
        E_grid_pkhist = (
            np.max([0, *E_grid_true[:timestep, building_id]])
            if timestep > 0
            else max(E_grid_prevhour, 0)
        )

        peak_hist_cost = np.max([*E_grid, E_grid_pkhist])
        ramping_cost = np.abs(E_grid[0] - E_grid_prevhour)
        electricity_cost = np.sum(self.alpha_elec[building_id][timestep:] * E_grid)

        if len(E_grid) > 1:  # not at eod
            ramping_cost += np.sum(np.abs(E_grid[1:] - E_grid[:-1]))

        Q_value = (
            -self.alpha_ramp[building_id] * ramping_cost
            - self.alpha_peak1[building_id] * peak_hist_cost
            - electricity_cost  # add virtual elec cost
        )

        # called only if no Q value exists for current timestep
        self.Q_value[timestep] = Q_value

        return Q_value

    def forward(
        self,
        t: int,
        parameters: dict,
        rewards: dict,
        zeta_target: dict,
        building_id: int,
        debug=False,
    ):
        """Uses result of RWL to compute for clipped Q values"""

        # TEMP
        Q_value = self.get(t)
        if Q_value is None:
            solution = self.solve(t, parameters, zeta_target, building_id, debug)
            Q_value = self.reward_warping_layer(t, solution, building_id)
        return Q_value
        # TEMP

        Gt_tn = 0.0
        rewards = rewards["reward"][:, building_id]

        for n in range(1, 24 - t):
            # first check if we need to compute it at all. --> Saves computation

            # if Q_value := self.get(t + n) is not None: # will break < 3.8.5
            #     return Q_value

            Q_value = self.get(t + n)  # NOTE: n is an index from 0 - 23

            if Q_value is None:  # doesn't exist, solve
                solution = self.solve(
                    t + n, parameters, zeta_target, building_id, debug
                )
                Q_value = self.reward_warping_layer(t + n, solution, building_id)

            Gt_tn += (np.sum(rewards[t + 1 : t + n]) + Q_value) * self.lambda_ ** (
                n - 1
            )

        # compute TD(\lambda) rewards
        Gt_lambda = (1 - self.lambda_) * Gt_tn + np.sum(rewards[t:]) * self.lambda_ ** (
            24 - t - 1
        )

        return Gt_lambda

    def target_update(self, alphas_local: np.ndarray):
        """Updates alphas given from L2 optimization"""
        assert (
            len(alphas_local) == 3
        ), f"Incorrect dimension passed. Alpha tuple should be of size 3. found {len(alphas_local)}"

        ### main target update -- alphas_new comes from LS optim sol.
        r, p, e = self.get_alphas()  # ramp, peak, elec
        alpha_ramp = self.rho * r + (1 - self.rho) * alphas_local[0]
        alpha_peak1 = self.rho * p + (1 - self.rho) * alphas_local[1]
        alpha_elec = self.rho * e + (1 - self.rho) * alphas_local[2]

        self.set_alphas(
            alpha_ramp, alpha_peak1, alpha_elec
        )  # updated alphas! -- end of critic update


class Optim:
    """Performs Critic Update"""

    def __init__(self, rho=0.9) -> None:
        self.L2_scores = defaultdict(list)  # list of L2 scores over iterations
        self.rho = rho  # regularization term used in L2 optimization

    def obtain_target_Q(
        self,
        critic_target_1: Critic,
        critic_target_2: Critic,
        t: int,
        parameters: dict,  # both batch_1 and batch_2
        rewards: dict,  # both batch_1 and batch_2
        zeta_target: dict,
        building_id: int,
        debug: bool = False,
    ):
        """Computes min Q"""
        # shared zeta-target across both target critic
        parameters_1, parameters_2 = parameters
        rewards_1, rewards_2 = rewards

        Q1 = critic_target_1.forward(
            t, parameters_1, rewards_1, zeta_target, building_id, debug
        )
        Q2 = critic_target_2.forward(
            t, parameters_2, rewards_2, zeta_target, building_id, debug
        )

        if min(Q1, Q2) == Q1:  # sequential choice
            return (
                rewards_1["reward"][t, building_id]
                + critic_target_1.lambda_ * (1 - int(t == 23)) * Q1
            )

        return (
            rewards_2["reward"][t, building_id]
            + critic_target_2.lambda_ * (1 - int(t == 23)) * Q2
        )

        # return rewards["reward"][t, building_id] + critic_target_1.lambda_ * (
        #     1 - int(t == 23)
        # ) * min(Q1, Q2)
        # # TEMP
        # return min(Q1, Q2)  # y_r

    def log_L2_optimization_scores(self):
        """Records MSE from L2 optimization"""
        pass

    def least_absolute_optimization(
        self,
        parameters: list,  # data collected within actor forward pass for MINI_BATCH (utils.py) number of updates (contains params + rewards) -- batch_params_1, batch_params_2
        zeta_target: dict,
        building_id: int,
        critic_target: list,
        is_random: bool,  # true indicates shuffled data i.e. critic_target_2
        debug: bool = False,
    ):
        """Define least-absolute optimization for generating optimal values for alpha_ramp,peak1/2."""
        # extract target Critic
        critic_target_1: Critic = critic_target[0]
        critic_target_2: Critic = critic_target[1]

        clipped_values = []  # length will be MINI_BATCH * 24. reshapes it to per day
        data = defaultdict(list)  # E_grid, E_gridpkhist, E_grid_prevhour over #days

        parameters_1, parameters_2 = parameters
        NUM_DAYS = len(parameters_1)  # meta-episode duration

        for i in range(NUM_DAYS):
            day_params_1, day_rewards_1 = parameters_1[i]
            day_params_2, day_rewards_2 = parameters_2[i]

            # append daily data
            if is_random:
                data["E_grid"].append(day_params_2["E_grid"][:, building_id])
            else:
                data["E_grid"].append(day_params_1["E_grid"][:, building_id])

            # clear Q-buffer at each day
            critic_target_1.Q_value = [None] * 24
            critic_target_2.Q_value = [None] * 24

            for r in range(24):
                LOG(f"L2 Optim\tBuilding: {building_id}\tHour: {str(r).zfill(2)}")

                y_r = self.obtain_target_Q(
                    critic_target_1,
                    critic_target_2,
                    r,
                    (day_params_1, day_params_2),
                    (day_rewards_1, day_rewards_2),
                    zeta_target,
                    building_id,
                    debug,
                )

                if is_random:
                    data["E_grid_prevhour"].append(
                        day_params_2["E_grid_prevhour"][r, building_id]
                    )
                    data["E_grid_pkhist"].append(
                        max(0, day_params_2["E_grid_prevhour"][r, building_id])
                        if r == 0
                        else np.max([0, *day_params_2["E_grid"][:r, building_id]])
                    )  # pkhist at 0th hour is 0.
                else:
                    data["E_grid_prevhour"].append(
                        day_params_1["E_grid_prevhour"][r, building_id]
                    )
                    data["E_grid_pkhist"].append(
                        max(0, day_params_1["E_grid_prevhour"][r, building_id])
                        if r == 0
                        else np.max([0, *day_params_1["E_grid"][:r, building_id]])
                    )  # pkhist at 0th hour is 0.

                clipped_values.append(y_r)

        # convert to ndarray
        clipped_values = np.array(clipped_values, dtype=float).reshape(
            NUM_DAYS, 24
        )  # number of days, 24 hours

        data["E_grid"] = np.array(data["E_grid"], dtype=float).reshape(
            clipped_values.shape
        )

        data["E_grid_prevhour"] = np.array(
            data["E_grid_prevhour"], dtype=float
        ).reshape(clipped_values.shape)

        data["E_grid_pkhist"] = np.array(data["E_grid_pkhist"], dtype=float).reshape(
            clipped_values.shape
        )

        ### variables
        alpha_ramp = cp.Variable(name="ramp")
        alpha_peak1 = cp.Variable(name="peak1")
        alpha_elec = cp.Variable(name="elec", shape=(24,))  # virtual electricity cost

        ### parameters
        E_grid = cp.Parameter(
            name="E_grid", shape=(clipped_values.shape), value=data["E_grid"]
        )
        E_grid_pkhist = cp.Parameter(
            name="E_grid_pkhist",
            shape=(clipped_values.shape),
            value=data["E_grid_pkhist"],
        )
        E_grid_prevhour = cp.Parameter(
            name="E_grid_prevhour",
            shape=(clipped_values.shape),
            value=data["E_grid_prevhour"],
        )

        y_t = cp.Parameter(
            name="y_r",
            shape=(clipped_values.shape),
            value=clipped_values,
        )

        #### cost & constraints
        cost = []
        constraints = []

        self.debug = defaultdict(list)

        for i in range(NUM_DAYS):
            ramping_cost = cp.abs(E_grid[i][0] - E_grid_prevhour[i]) + cp.sum(
                cp.abs(E_grid[i][1:] - E_grid[i][:-1])
            )  # E_grid_t+1 - E_grid_t
            peak_net_electricity_cost = cp.atoms.elementwise.maximum.maximum(
                E_grid[i], E_grid_pkhist[i]
            )  # element-wise max(E_grid, E_gridpkhist)

            electricity_cost = cp.sum(alpha_elec * E_grid[i])

            # append ramping and peak net electricity cost to debug
            self.debug["ramping_cost"].append(ramping_cost)
            self.debug["peak_net_electricity_cost"].append(peak_net_electricity_cost)
            self.debug["electricity_cost"].append(E_grid[i])

            # L2 norm https://docs.google.com/document/d/1QbqCQtzfkzuhwEJeHY1-pQ28disM13rKFGTsf8dY8No/edit?disco=AAAAMzPtZMU
            cost.append(
                self.rho
                * cp.sum(
                    cp.square(
                        -alpha_ramp * ramping_cost
                        - alpha_peak1 * peak_net_electricity_cost
                        - electricity_cost  #  add virtual elec cost
                        - y_t[i]
                    )
                )
                + (1 - self.rho)
                * (
                    cp.square(alpha_peak1)
                    + cp.square(alpha_ramp)
                    + cp.sum(
                        cp.square(
                            E_grid[i]
                            * (critic_target_1.alpha_elec[building_id] - alpha_elec)
                        )
                    )
                )
            )

            # Ensure that Q value is negative.
            # for j in range(24):
            #     rwl = (
            #         -alpha_ramp * ramping_cost[j]
            #         - alpha_peak1 * peak_net_electricity_cost[j]
            #         - alpha_elec[j] * E_grid[i][j]
            #     )
            #     constraints.append(rwl <= 0.0)

            # check eigen values of A.T @ T. condition number is bounded. if Q value is not large enough, rank defficient. cannot invert.
            self.peak = list(peak_net_electricity_cost.value)
            self.ramping = list(ramping_cost.value)
            self.E_grid = list(E_grid[i].value)
            self.y = y_t[i].value

        # low = 0.1
        # high = 2
        # # alpha-ramp
        # constraints.append(alpha_ramp <= high)
        # constraints.append(alpha_ramp >= low)

        # # alpha-peak –– OBSERVATION: usually takes the highest variance
        # constraints.append(alpha_peak1 <= high)
        # constraints.append(alpha_peak1 >= low)

        # Form objective.
        obj = cp.Minimize(cp.sum(cost))
        # Form and solve problem.
        prob = cp.Problem(obj, constraints)

        self.problem = prob

        try:
            optim_solution = prob.solve(
                verbose=debug,  # max_iters=1_000_000
            )  # Returns the optimal value.
            assert (
                float("-inf") < optim_solution < float("inf")
            ), "Optimization failed! Trying SCS..."

        except:  # try another solver

            LOG(
                f"\nSolving L2 optimization using SCS solver for building {building_id}"
            )

            optim_solution = prob.solve(
                solver="SCS",
                verbose=debug,  # max_iters=1_000_000
            )  # Returns the optimal value.

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
        building_id: int,
        critic_local: list,
        critic_target: list,
        debug: bool = False,
    ):

        # extract local critic
        critic_local_1, critic_local_2 = critic_local
        # Compute L2 Optimization for Critic Local 1 (using sequential data) and Critic Local 2 (using random data) using Critic Target 1 and 2
        local_1_solution = self.least_absolute_optimization(
            (batch_parameters_1, batch_parameters_2),
            zeta_target,
            building_id,
            critic_target,
            False,
            debug,
        )

        local_2_solution = self.least_absolute_optimization(
            (batch_parameters_2, batch_parameters_1),
            zeta_target,
            building_id,
            critic_target,
            True,
            debug,
        )
        # update alphas for local
        critic_local_1.alpha_ramp[building_id] = local_1_solution["ramp"]
        critic_local_1.alpha_peak1[building_id] = local_1_solution["peak1"]
        critic_local_1.alpha_elec[building_id] = local_1_solution["elec"]

        critic_local_2.alpha_ramp[building_id] = local_2_solution["ramp"]
        critic_local_2.alpha_peak1[building_id] = local_2_solution["peak1"]
        critic_local_2.alpha_elec[building_id] = local_2_solution["elec"]
