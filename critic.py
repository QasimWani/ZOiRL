import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp


class Critic:
    def __init__(self, t: int, parameters: dict, building_id: int, num_actions: int):
        """
        @Param:
        - `parameters` : data (dict) from r <= t <= T following `get_current_data` format.
        - `t` : hour to solve convex optimization for.
        - `building_id`: building index number (0-based).
        - `num_actions`: Number of actions for building.
            NOTE: right now, this is an integer, but will be checked programmatically later.
            Solves reward warping layer per building as specified by `building_id`. Note: 0 based.
            Call `forward()` for solution to RWL.
        """

        T = 24
        window = T - t
        self.constraints = []
        self.costs = []
        self.t = t
        self.num_actions = num_actions

        ### critic params
        self.lambda_ = 0.9

        ### --- Parameters share with the actor
        p_ele = cp.Parameter(
            name="p_ele", shape=(window), value=parameters["p_ele"][t:, building_id]
        )
        E_grid_prevhour = cp.Parameter(
            name="E_grid_prevhour",
            value=parameters["E_grid_past"][min(t - 1, 0), building_id],
        )

        E_grid_pkhist = cp.Parameter(
            name="E_grid_pkhist",
            value=parameters["E_grid_past"][: (t + 1), building_id].max(),
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
            name="soc_bat_init", value=parameters["c_bat_end"][0, building_id]
        )
        soc_bat_norm_end = cp.Parameter(
            name="soc_bat_norm_end", value=parameters["c_bat_end"][1, building_id]
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
            name="soc_Hsto_init", value=parameters["c_Hsto_end"][0, building_id]
        )
        soc_Hsto_norm_end = cp.Parameter(
            name="soc_Hsto_norm_end", value=parameters["c_Hsto_end"][1, building_id]
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
            name="soc_Csto_init", value=parameters["c_Csto_end"][0, building_id]
        )
        soc_Csto_norm_end = cp.Parameter(
            name="soc_Csto_norm_end", value=parameters["c_Csto_end"][1, building_id]
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
        action_bat = cp.Parameter(
            name="action_bat",
            value=parameters["action_bat"][:, building_id],
            shape=(window),
        )  # electric battery

        SOC_H = cp.Variable(name="SOC_H", shape=(window))  # heat storage
        SOC_Hrelax = cp.Variable(
            name="SOH_Crelax", shape=(window)
        )  # heat storage relaxation (prevents numerical infeasibilities)
        action_H = cp.Parameter(
            name="action_H",
            value=parameters["action_H"][:, building_id],
            shape=(window),
        )  # heat storage

        SOC_C = cp.Variable(name="SOC_C", shape=(window))  # cooling storage
        SOC_Crelax = cp.Variable(
            name="SOC_Crelax", shape=(window)
        )  # cooling storage relaxation (prevents numerical infeasibilities)
        action_C = cp.Parameter(
            name="action_C",
            value=parameters["action_C"][:, building_id],
            shape=(window),
        )  # cooling storage

        ### objective function

        ### NEEDS CHANGING ###

        ramping_cost = cp.abs(E_grid[0] - E_grid_prevhour) + cp.sum(
            cp.abs(E_grid[1:] - E_grid[:-1])
        )  # E_grid_t+1 - E_grid_t
        peak_net_electricity_cost = cp.max(
            cp.atoms.affine.hstack.hstack([*E_grid, E_grid_pkhist])
        )  # max(E_grid, E_gridpkhist)

        # https://docs.google.com/document/d/1QbqCQtzfkzuhwEJeHY1-pQ28disM13rKFGTsf8dY8No/edit?disco=AAAAMzPtZMU
        reward_func = (
            alpha_ramp.value * ramping_cost
            + alpha_peak1.value * peak_net_electricity_cost
            + alpha_peak2.value * cp.square(peak_net_electricity_cost)
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

        ### @jinming --- do we need relaxation costs? or is `reward_func` guaranteed to produce feasable solutions?
        self.costs.append(
            reward_func
            + E_bal_relax_cost * 1e4
            + H_bal_relax_cost * 1e4
            + C_bal_relax_cost * 1e4
            + SOC_Brelax_cost * 1e4
            + SOC_Crelax_cost * 1e4
            + SOC_Hrelax_cost * 1e4
        )

        ### NEEDS CHANGING ###

        ### constraints
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

        #!!!!! Problem Child !!!!!
        self.constraints.append(
            E_hpC * COP_C + C_bal_relax == action_C * C_p_Csto + C_bd
        )  # cooling balance
        #!!!!! Problem Child !!!!!

        # heat pump constraints
        self.constraints.append(E_hpC <= E_hpC_max)  # maximum cooling
        self.constraints.append(E_hpC >= 0)  # constraint minimum cooling to positive
        # electric heater constraints
        self.constraints.append(E_ehH >= 0)  # constraint to PD
        self.constraints.append(E_ehH <= E_ehH_max)  # maximum limit

        # electric battery constraints

        self.constraints.append(
            SOC_bat[0]
            == (1 - C_f_bat) * soc_bat_init + action_bat[0] * eta_bat + SOC_Crelax[0]
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
            == (1 - C_f_Hsto) * soc_Hsto_init + action_H[0] * eta_Hsto + SOC_Hrelax[0]
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
            == (1 - C_f_Csto) * soc_Csto_init + action_C[0] * eta_Csto + SOC_Crelax[0]
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
        """Returns raw problem"""
        # Form objective.
        obj = cp.Minimize(*self.costs)
        # Form and solve problem.
        prob = cp.Problem(obj, self.constraints)

        return prob

    def get_constraints(self):
        """Returns constraints for problem"""
        return self.constraints

    def forward(self, debug=False, dispatch=False):
        prob = self.get_problem()  # Form and solve problem
        Q_reward_warping_output = prob.solve(
            verbose=debug
        )  # output of reward warping function

        if float("-inf") < Q_reward_warping_output < float("inf"):
            G_t_tPlusn = self.get_past_rewards(t) + [
                Q_reward_warping_output
            ]  ## takes past rewards nd.array of shape T - t - 1 {! need to define this !}
            G_t_lambda = (1 - self.lambda_) * np.sum(
                [
                    self.lambda_ ** (i - 1) * G_t_tPlusn[t : t + i]
                    for i in range(len(G_t_tPlusn))
                ]
            ) + self.lambda_ ** (24 - t - 1) * G_t_tPlusn[t]
            return G_t_lambda, prob.variables()["E_grid"]

        else:
            return "Unbounded Solution"

    def get_past_rewards(self, t_end, size=2 * 24):
        """samples past `t` rewards"""
        rewards = replay_buffer.n_sample(t_end - size, t_end)[2]
        return rewards

    def backward(self, data, t, building_id):
        """Define least-squares optimization for generating values for alpha_ramp,peak1/2"""
        y_r, E_grid = obtain_target_Q(
            Q1, Q2
        )  # note: Q1 and Q2 are functions defined below!
        ### variables
        alpha_ramp = cp.Variable(name="ramp", shape=(24 - t))
        alpha_peak1 = cp.Variable(name="peak1", shape=(24 - t))
        alpha_peak2 = cp.Variable(name="peak2", shape=(24 - t))

        ### parameters
        E_grid = cp.Parameter(
            name="E_grid", shape=(24 - t), value=E_grid[t:, building_id]
        )
        E_grid_pk_hist = cp.Parameter(
            name="E_grid_pk_hist",
            shape=(24 - t),
            value=data["E_grid_pk_hist"][:, building_id],
        )
        y_r = cp.Parameter(
            name="y_r", value=y_r[t, building_id]
        )  ### defined in terms of `obtain_target_Q`

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
        obj = cp.Minimize(*self.costs)
        # Form and solve problem.
        prob = cp.Problem(obj, self.constraints)

        optim_solution = prob.solve()

        assert (
            float("-inf") < optim_solution < float("inf")
        ), "Unbounded solution/primal infeasable"

        solution = {}
        for var in prob.variables():
            solution[var.name()] = var.value

        return solution  ### returns optimal values for 3 alphas. LOCAL UPDATE
        ## after this, we call Actor.backward() to get actor update and pass in data from critic update (alphas).

    def target_update(self, data_target, data_local):
        _alpha_ramp, _alpha_peak1, _alpha_peak2 = zip(
            *self.backward(data_local, t, building_id).values()
        )  ## local update after Least square optim.
        ### main target update
        alpha_ramp, alpha_peak1, alpha_peak2 = get_current_target_alphas(data_target)
        alpha_ramp, alpha_peak1, alpha_peak2 = rho * np.array(
            [_alpha_ramp, _alpha_peak1, _alpha_peak2]
        ) + (1 - rho) * np.array([alpha_ramp, alpha_peak1, alpha_peak2])

        return (
            alpha_ramp,
            alpha_peak1,
            alpha_peak2,
        )  ### updated target Qx (x = 1/2 depending on param. need to call this function twice)
