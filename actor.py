from copy import deepcopy

from utils import Adam
import numpy as np
import torch

from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp


class Actor:
    def __init__(self, num_actions: list, num_buildings: int, rho: float = 0.9):
        """One-time initialization. Need to call `create_problem` to initialize optimization model with params."""
        self.num_actions = num_actions
        self.num_buildings = num_buildings
        self.rho = rho
        # Optim specific
        self.constraints = []
        self.costs = []

        self.zeta = None
        self.params = None

        self.optim = Adam()

    def create_problem(self, t: int, parameters: dict, building_id: int):
        """
        @Param:
        - `t` : hour to solve optimization for.
        - `parameters` : data (dict) from r <= t <= T following `get_current_data` format.
        - `building_id`: building index number (0-based)
        - `action_spaces`: action space for agent in CL evn. Changes over time.
        NOTE: right now, this is an integer, but will be checked programmatically.
        Solves per building as specified by `building_id`. Note: 0 based.
        """
        T = 24
        window = T - t
        # Reset data
        self.constraints = []
        self.costs = []
        self.t = t

        # assign Zeta
        if self.zeta is None:
            self.zeta = deepcopy(
                parameters
            )  # will be changed automatically after forward pass.

        self.set_parameters(
            params=deepcopy(parameters)
        )  # update parameters (constants + predictors)

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
            name="p_ele", shape=(window), value=self.zeta["p_ele"][t:, building_id]
        )

        E_grid_prevhour = (
            self.params["E_grid_past"][t, building_id]
            if "E_gird" in self.params
            and len(self.params["E_grid"].shape)
            == 2  # E-grid exists means E_grid_past valid
            else 0,  # used in day ahead dispatch, default E-grid okay
        )

        E_grid_pkhist = (
            np.max(self.params["E_grid"][:t, building_id])
            if "E_grid" in self.params and len(self.params["E_grid"].shape) == 2
            else 0,  # used in day ahead dispatch, default E-grid okay
        )

        # max-min normalization of ramping_cost to downplay E_grid_sell weight.
        ramping_cost_coeff = cp.Parameter(
            name="ramping_cost_coeff",
            value=self.zeta["ramping_cost_coeff"][t, building_id],
        )

        # Loads
        E_ns = self.params["E_ns"][t:, building_id]
        H_bd = self.params["H_bd"][t:, building_id]
        C_bd = self.params["C_bd"][t:, building_id]

        # PV generations
        E_pv = self.params["E_pv"][t:, building_id]

        # Heat Pump
        COP_C = self.params["COP_C"][t:, building_id]
        E_hpC_max = self.params["E_hpC_max"][t, building_id]

        # Electric Heater
        eta_ehH = cp.Parameter(
            name="eta_ehH", value=self.zeta["eta_ehH"][t, building_id]
        )
        E_ehH_max = self.params["E_ehH_max"][t, building_id]

        # Battery
        C_f_bat = self.params["C_f_bat"][t, building_id]
        C_p_bat = cp.Parameter(
            name="C_p_bat", value=self.zeta["C_p_bat"][t, building_id]
        )
        eta_bat = cp.Parameter(
            name="eta_bat", value=self.zeta["eta_bat"][t, building_id]
        )
        soc_bat_init = cp.Parameter(
            name="c_bat_init", value=self.zeta["c_bat_init"][building_id]
        )
        soc_bat_norm_end = cp.Parameter(
            name="c_bat_end", value=self.zeta["c_bat_end"][t, building_id]
        )

        # Heat (Energy->dhw) Storage
        C_f_Hsto = self.params["C_f_Hsto"][t, building_id]
        C_p_Hsto = cp.Parameter(
            name="C_p_Hsto", value=self.zeta["C_p_Hsto"][t, building_id]
        )
        eta_Hsto = cp.Parameter(
            name="eta_Hsto", value=self.zeta["eta_Hsto"][t, building_id]
        )
        soc_Hsto_init = cp.Parameter(
            name="c_Hsto_init", value=self.zeta["c_Hsto_init"][building_id]
        )

        # Cooling (Energy->cooling) Storage
        C_f_Csto = self.params["C_f_Csto"][t, building_id]
        C_p_Csto = cp.Parameter(
            name="C_p_Csto", value=self.zeta["C_p_Csto"][t, building_id]
        )
        eta_Csto = cp.Parameter(
            name="eta_Csto", value=self.zeta["eta_Csto"][t, building_id]
        )
        soc_Csto_init = cp.Parameter(
            name="c_Csto_init", value=self.zeta["c_Csto_init"][building_id]
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
            == (1 - C_f_bat) * soc_bat_init + action_bat[0] * eta_bat + SOC_Crelax[0]
        )  # initial SOC
        # soc updates
        for i in range(1, window):
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

    def forward(
        self,
        t: int,
        parameters: dict,
        building_id: int,
        debug=False,
        dispatch=False,
    ):
        """Actor Optimization"""
        self.create_problem(
            t, parameters, building_id
        )  # problem formulation for Actor optimizaiton
        prob = self.get_problem()  # Form and solve problem
        actions = {}
        status = prob.solve(verbose=debug)  # Returns the optimal value.
        if float("-inf") < status < float("inf"):
            pass
        else:
            return [0, 0, 0], 0, 0 if dispatch else None, actions, None

        for var in prob.variables():
            if dispatch:
                actions[var.name()] = np.array(
                    var.value
                )  # no need to clip... automatically restricts range
            else:
                actions[var.name()] = var.value[
                    0
                ]  # no need to clip... automatically restricts range

        # prune out zeta - set zeta values for later use in backward pass.
        self.prune_update_zeta(prob.param_dict, building_id)

        ## compute dispatch cost
        if dispatch:
            ramping_cost = np.sum(
                np.abs(
                    actions["E_grid"][1:]
                    + actions["E_grid_sell"][1:]
                    - actions["E_grid"][:-1]
                    - actions["E_grid_sell"][:-1]
                )
            )
            net_peak_electricity_cost = np.max(actions["E_grid"])
            virtual_electricity_cost = np.sum(
                self.zeta["p_ele"][:, building_id] * actions["E_grid"]
            )
            dispatch_cost = (
                ramping_cost + net_peak_electricity_cost + virtual_electricity_cost
            )

        if self.num_actions[building_id].shape[0] == 2:
            return (
                [
                    actions["action_H"],
                    actions["action_bat"],
                ],
                dispatch_cost if dispatch else None,
                actions["E_grid"],  # + actions["E_grid_sell"]
            )
        return (
            [
                actions["action_C"],
                actions["action_H"],
                actions["action_bat"],
            ],
            dispatch_cost if dispatch else None,
            actions["E_grid"],  # + actions["E_grid_sell"]
        )

    def get_parameters(self):
        """Returns parameters (constant + predictors) and zeta"""
        return self.zeta, self.params

    def set_parameters(self, zeta: dict = None, params: dict = None):
        """Sets values for parameters (constant + predictors) and/or zeta"""
        assert (
            zeta or params
        ), "Missing data. Please pass at least params or zeta params to update actor network."
        if zeta:
            self.zeta = zeta
        if params:
            self.params = params

    def prune_update_zeta(self, params: dict, building_id: int):
        """Deep Update Zeta"""
        for key, value in params.items():
            if key in self.zeta:
                if len(self.zeta[key].shape) == 2:  # 2d array
                    self.zeta[key][:, building_id] = value.value
                else:
                    self.zeta[key][building_id] = value.value
            else:
                self.zeta[key] = np.array([0] * self.num_buildings)
                self.zeta[key][building_id] = value.value

        if len(self.zeta.keys()) > len(params.keys()):
            # prune out other types of params
            for key in deepcopy(list(self.zeta.keys())):
                if (
                    key not in params and key != "ramping_cost_coeff"
                ):  # not used directly in `create_problem`
                    self.zeta.pop(key, None)  # is guaranteed to exist in Zeta

    def target_update(self, local_zeta: dict, building_id: int):
        """Update rule for Target Actor."""
        assert (
            self.zeta is not None
        ), "Zeta must be pre-initialized to local actor network. See `pre_initialize_target_params`"

        for param, value in self.zeta.items():
            if len(self.zeta[param].shape) == 2:
                self.zeta[param][:, building_id] = (
                    self.rho * value[:, building_id]
                    + (1 - self.rho) * local_zeta[param][:, building_id]
                )
            else:
                self.zeta[param][building_id] = (
                    self.rho * value[building_id]
                    + (1 - self.rho) * local_zeta[param][building_id]
                )

    def pre_initialize_target_params(self, local_zeta: dict, local_params: dict):
        """Initializes target actor parameters for the very first run."""
        if self.zeta is None:
            self.zeta = local_zeta
        if self.params is None:
            self.params = local_params

    def backward(
        self,
        t: int,
        alphas: list,
        building_id: int,
        is_local: bool,
    ):
        """
        Computes the gradient first for optimization given parameters `params`.
        Updates actor parameters accordingly.

        This function calculates math: \nabla_\zeta Q(s, \mu(s, \zeta), w) - see section 1.3.1
        Step 1. Solve Actor optimization w/ actions this time as parameters, whose values were obtained in Actor forward pass.
        Step 2. Use reward warping function w/ E^grid given from (1) and perform forward pass.
        Step 3. Take backward pass of (2) with parameters \zeta.
        """
        self.create_problem(
            t, self.params, building_id
        )  # problem formulation for Actor optimizaiton

        prob = self.get_problem()
        (zeta, variables_actor,) = (
            prob.param_dict,
            prob.var_dict,
        )

        def convert_to_torch_tensor(params: dict):
            """Converts cp.param to torch.tensor"""
            param_arr = []
            for value in params.values():
                param_arr.append(torch.from_numpy(np.array(value.value)).float())
            return param_arr

        # Actor forward pass - Step 1
        fit_actor = CvxpyLayer(
            prob,
            parameters=list(zeta.values()),
            variables=list(variables_actor.values()),
        )
        # fetch params in loss calculation

        E_grid_prevhour = self.params["E_grid_past"][t, building_id]
        E_grid_pkhist = self.params["E_grid_pkhist"][t, building_id]

        # typecast each param to tensor for autograd later
        zeta_tensor = convert_to_torch_tensor(zeta)
        E_grid, *_ = fit_actor(*zeta_tensor)

        # Q function forward pass - Step 2
        alpha_ramp, alpha_peak1, alpha_peak2 = torch.from_numpy(alphas).float()

        ramping_cost = torch.abs(E_grid[0] - E_grid_prevhour) + torch.sum(
            torch.abs(E_grid[1:] - E_grid[:-1])
        )  # E_grid_t+1 - E_grid_t
        peak_net_electricity_cost = torch.max(E_grid.max(), torch.tensor(E_grid_pkhist))

        reward_warping_loss = (
            -alpha_ramp[building_id] * ramping_cost
            - alpha_peak1[building_id] * peak_net_electricity_cost
            - alpha_peak2[building_id] * torch.square(peak_net_electricity_cost)
        )
        reward_warping_loss.requires_grad = True

        # Gradient w.r.t parameters (math: \zeta) - Step 3
        reward_warping_loss.backward()
        grad = reward_warping_loss.grad.item()
        if is_local:
            # updated via Adam
            for i, param in enumerate(zeta.keys()):
                param_update = self.optim.update(t, zeta_tensor[i].numpy(), grad)
                zeta[param].value = param_update

            self.prune_update_zeta(zeta, building_id)  # update zeta

        return zeta
