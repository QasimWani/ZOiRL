from collections import defaultdict
from copy import Error, deepcopy
from critic import Critic
from logger import LOG

from utils import Adam, RBC, normalize
import numpy as np
import torch

from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp


class Actor:
    def __init__(
        self,
        num_actions: list,
        num_buildings: int,
        offset: int,
        rho: float = 0.75,  # the higher the value, the more weight is given to the previous state
    ):
        """One-time initialization. Need to call `create_problem` to initialize optimization model with params."""
        self.num_actions = num_actions
        self.num_buildings = num_buildings
        self.rho = rho
        # Optim specific
        self.constraints = []
        self.scs_cnt = [0 for _ in range(9)]
        self.fail_cnt = [0 for _ in range(9)]

        self.cost = None  # created at every call to `create_problem`. not used in DPP.
        # list of parameter names for Zeta
        zeta_keys = set(
            [
                "p_ele",
                "eta_ehH",
                "eta_bat",
                "c_bat_end",
                "eta_Hsto",
                "eta_Csto",
            ]
        )

        self.zeta = self.initialize_zeta()  # initialize zeta w/ default values

        self.optim = []

        for _ in range(num_buildings):
            self.optim.append({key: Adam() for key in zeta_keys})

        self.adam_offset = offset  # (t - offset + 1) see Adam.update

        # define problem - forward pass
        self.prob = [None] * 24  # template for each hour

        ### RBC deviation
        a, b, c = RBC(num_actions).load_day_actions()
        # a, b, c = np.zeros((3, self.num_buildings, 24))
        self.rbc_actions = {"action_C": a, "action_H": b, "action_bat": c}

        # Logger
        self._grads = [
            defaultdict(list) for _ in range(self.num_buildings)
        ]  # zeta grads over time

        self._losses = defaultdict(list)  # zeta losses over time

    def initialize_zeta(
        self,
        p_ele: float = 1.0,
        eta_ehH: float = 0.9,
        eta_bat: float = 1.0,
        eta_Hsto: float = 1.0,
        eta_Csto: float = 1.0,
        c_bat_end: float = 0.1,
        c_Csto_end: float = 0.1,  # constant
    ):
        """Initialize differentiable parameters, zeta with default values"""
        zeta = {}  # 6 parameters learned via differentiation

        zeta["p_ele"] = np.full((24, self.num_buildings), p_ele)

        zeta["eta_bat"] = np.full((24, self.num_buildings), eta_bat)
        zeta["eta_Hsto"] = np.full((24, self.num_buildings), eta_Hsto)
        zeta["eta_Csto"] = np.full((24, self.num_buildings), eta_Csto)

        zeta["eta_ehH"] = np.full(9, eta_ehH)
        zeta["c_bat_end"] = np.full(9, c_bat_end)
        zeta["c_Csto_end"] = np.full(9, c_Csto_end)

        return zeta

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
        # self.cost = None ### reassign to NONE. not needed.
        self.t = t

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
            name="p_ele", shape=(window), value=self.zeta["p_ele"][t:, building_id]
        )

        E_grid_prevhour = cp.Parameter(
            name="E_grid_prevhour", value=parameters["E_grid_prevhour"][t, building_id]
        )

        E_grid_pkhist = cp.Parameter(
            name="E_grid_pkhist",
            value=np.max([0, *parameters["E_grid"][:t, building_id]])
            if t > 0
            else max(E_grid_prevhour.value, 0),
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
        eta_ehH = cp.Parameter(name="eta_ehH", value=self.zeta["eta_ehH"][building_id])
        E_ehH_max = cp.Parameter(
            name="E_ehH_max", value=parameters["E_ehH_max"][t, building_id]
        )

        # Battery
        C_p_bat = cp.Parameter(
            name="C_p_bat", value=parameters["C_p_bat"][t, building_id]
        )
        eta_bat = cp.Parameter(
            name="eta_bat", shape=window, value=self.zeta["eta_bat"][t:, building_id]
        )
        soc_bat_init = cp.Parameter(
            name="c_bat_init", value=parameters["c_bat_init"][t, building_id]
        )
        soc_bat_norm_end = cp.Parameter(
            name="c_bat_end", value=self.zeta["c_bat_end"][building_id]
        )

        # Heat (Energy->dhw) Storage
        C_p_Hsto = cp.Parameter(
            name="C_p_Hsto", value=parameters["C_p_Hsto"][t, building_id]
        )
        eta_Hsto = cp.Parameter(
            name="eta_Hsto",
            shape=window,
            value=self.zeta["eta_Hsto"][t:, building_id],
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
            value=self.zeta["eta_Csto"][t:, building_id],
        )
        soc_Csto_init = cp.Parameter(
            name="c_Csto_init", value=parameters["c_Csto_init"][t, building_id]
        )
        soc_Csto_norm_end = cp.Parameter(
            name="c_Csto_end", value=self.zeta["c_Csto_end"][building_id]
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

        ### relaxation costs - L1 norm
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
            + cp.sum(cp.abs(action_bat)) * 1e1
            + cp.sum(cp.abs(action_C)) * 1e1
            + cp.sum(cp.abs(action_H)) * 1e1
        )

        ### constraints
        self.constraints.append(E_grid >= 0)
        self.constraints.append(E_grid_sell <= 0)

        # energy balance constraints
        self.constraints.append(
            E_pv + E_grid + E_grid_sell + E_bal_relax
            == E_ns
            + E_hpC
            + E_ehH
            + (action_bat + self.rbc_actions["action_bat"][building_id, T - window :])
            * C_p_bat
        )  # electricity balance
        self.constraints.append(
            E_ehH * eta_ehH + H_bal_relax
            == (action_H + self.rbc_actions["action_H"][building_id, T - window :])
            * C_p_Hsto
            + H_bd
        )  # heat balance

        self.constraints.append(
            E_hpC * COP_C + C_bal_relax
            == (action_C + self.rbc_actions["action_C"][building_id, T - window :])
            * C_p_Csto
            + C_bd
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
            + (action_bat[0] + self.rbc_actions["action_bat"][building_id, T - window])
            * eta_bat[0]
            + SOC_Brelax[0]
        )  # initial SOC
        # soc updates
        for i in range(1, window):
            self.constraints.append(
                SOC_bat[i]
                == (1 - C_f_bat) * SOC_bat[i - 1]
                + (
                    action_bat[i]
                    + self.rbc_actions["action_bat"][building_id, T - window + i]
                )
                * eta_bat[i]
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
            + (action_H[0] + self.rbc_actions["action_H"][building_id, T - window])
            * eta_Hsto[0]
            + SOC_Hrelax[0]
        )  # initial SOC
        # soc updates
        for i in range(1, window):
            self.constraints.append(
                SOC_H[i]
                == (1 - C_f_Hsto) * SOC_H[i - 1]
                + (
                    action_H[i]
                    + self.rbc_actions["action_H"][building_id, T - window + i]
                )
                * eta_Hsto[i]
                + SOC_Hrelax[i]
            )
        self.constraints.append(SOC_H >= 0)  # battery SOC bounds
        self.constraints.append(SOC_H <= 1)  # battery SOC bounds

        # Cooling Storage constraints
        self.constraints.append(
            SOC_C[0]
            == (1 - C_f_Csto) * soc_Csto_init
            + (action_C[0] + self.rbc_actions["action_C"][building_id, T - window])
            * eta_Csto[0]
            + SOC_Crelax[0]
        )  # initial SOC
        # soc updates
        for i in range(1, window):
            self.constraints.append(
                SOC_C[i]
                == (1 - C_f_Csto) * SOC_C[i - 1]
                + (
                    action_C[i]
                    + self.rbc_actions["action_C"][building_id, T - window + i]
                )
                * eta_Csto[i]
                + SOC_Crelax[i]
            )
        self.constraints.append(SOC_C[-1] == soc_Csto_norm_end)
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
                self.constraints.append(
                    action_C + self.rbc_actions["action_C"][building_id, T - window :]
                    <= h
                )
                self.constraints.append(
                    action_C + self.rbc_actions["action_C"][building_id, T - window :]
                    >= l
                )
            # cooling action
            elif key == "action_H":
                self.constraints.append(
                    action_H + self.rbc_actions["action_H"][building_id, T - window :]
                    <= h
                )
                self.constraints.append(
                    action_H + self.rbc_actions["action_H"][building_id, T - window :]
                    >= l
                )
            # Battery action
            elif key == "action_bat":
                self.constraints.append(
                    action_bat
                    + self.rbc_actions["action_bat"][building_id, T - window :]
                    <= h
                )
                self.constraints.append(
                    action_bat
                    + self.rbc_actions["action_bat"][building_id, T - window :]
                    >= l
                )

    def get_problem(self, t: int, parameters: dict, building_id: int):
        """Returns raw problem"""
        assert 0 <= t < 24, f"Invalid range for t. Found {t}, needs to be (0, 24]"
        # Form objective.
        if self.prob[t] is None:
            self.create_problem(
                t, parameters, building_id
            )  # problem formulation for Actor optimizaiton
            obj = cp.Minimize(self.cost)
            # Form problem.
            self.prob[t] = cp.Problem(obj, self.constraints)
            assert self.prob[t].is_dpp()
        else:  # DPP
            self.inject_params(t, parameters, building_id)

    def inject_params(self, t: int, parameters: dict, building_id: int):
        """Sets parameter values for problem. DPP"""
        assert (
            self.prob[t] is not None
        ), "Problem must be defined to be able to use DPP."
        problem_parameters = self.prob[t].param_dict

        ### --- Parameters ---
        problem_parameters["p_ele"].value = self.zeta["p_ele"][t:, building_id]

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
        problem_parameters["eta_ehH"].value = self.zeta["eta_ehH"][building_id]
        problem_parameters["E_ehH_max"].value = parameters["E_ehH_max"][t, building_id]

        # Battery
        problem_parameters["C_p_bat"].value = parameters["C_p_bat"][t, building_id]
        problem_parameters["eta_bat"].value = self.zeta["eta_bat"][t:, building_id]
        problem_parameters["c_bat_init"].value = parameters["c_bat_init"][
            t, building_id
        ]
        problem_parameters["c_bat_end"].value = self.zeta["c_bat_end"][building_id]

        # Heat (Energy->dhw) Storage
        problem_parameters["C_p_Hsto"].value = parameters["C_p_Hsto"][t, building_id]
        problem_parameters["eta_Hsto"].value = self.zeta["eta_Hsto"][t:, building_id]
        problem_parameters["c_Hsto_init"].value = parameters["c_Hsto_init"][
            t, building_id
        ]

        # Cooling (Energy->cooling) Storage
        problem_parameters["C_p_Csto"].value = parameters["C_p_Csto"][t, building_id]
        problem_parameters["eta_Csto"].value = self.zeta["eta_Csto"][t:, building_id]
        problem_parameters["c_Csto_init"].value = parameters["c_Csto_init"][
            t, building_id
        ]
        problem_parameters["c_Csto_end"].value = self.zeta["c_Csto_end"][building_id]

        ## Update Parameters
        for key, prob_val in problem_parameters.items():
            self.prob[t].param_dict[key].value = prob_val.value

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
        self.get_problem(t, parameters, building_id)  # Form problem using DPP

        actions = {}

        try:
            status = self.prob[t].solve(
                verbose=debug,  # max_iters=100_000
            )  # Returns the optimal value.
            assert float("-inf") < status < float("inf"), "Solver failed"

        except:  # try another solver
            status = self.prob[t].solve(
                solver="SCS", verbose=debug  # max_iters=1_000_000
            )  # Returns the optimal value.
            self.scs_cnt[building_id] += 1

        if float("-inf") < status < float("inf"):
            for var in self.prob[t].variables():
                if dispatch:
                    offset = np.zeros(len(var.value))
                    if "action" in str(var.name()):
                        offset = self.rbc_actions[var.name()][
                            building_id, 24 - t % 24 :
                        ]
                    actions[var.name()] = np.array(var.value) + offset
                else:
                    offset = 0
                    if "action" in str(var.name()):
                        offset = self.rbc_actions[var.name()][building_id, t % 24]
                    actions[var.name()] = np.array(var.value)[0] + offset
        else:
            self.fail_cnt[building_id] += 1
            LOG(f"\nDefault solution at t = {t} for building {building_id}")

            for var in self.prob[t].variables():
                if dispatch:
                    offset = np.zeros(len(var.value))
                    if "action" in str(var.name()):
                        offset = self.rbc_actions[var.name()][
                            building_id, 24 - t % 24 :
                        ]
                    actions[var.name()] = offset
                else:
                    offset = 0
                    if "action" in str(var.name()):
                        offset = self.rbc_actions[var.name()][building_id, t % 24]
                    actions[var.name()] = offset

        if self.num_actions[building_id].shape[0] == 2:
            return (
                [
                    actions["action_H"],
                    actions["action_bat"],
                ],
                actions,  # debug
                actions["E_grid"] + actions["E_grid_sell"],
            )
        return (
            [
                actions["action_C"],
                actions["action_H"],
                actions["action_bat"],
            ],
            actions,  # debug
            actions["E_grid"] + actions["E_grid_sell"],
        )

    def get_zeta(self):
        """Returns set of differentiable parameters, zeta"""
        return self.zeta

    def set_zeta(
        self,
        zeta: tuple,
        building_id: int,
    ):
        """Sets values for zeta"""
        # get Zeta
        (p_ele, eta_bat, eta_Hsto, eta_Csto, eta_ehH, c_bat_end) = zeta
        # p_ele = zeta[0]

        # dimensions: 24
        self.zeta["p_ele"][:, building_id] = p_ele
        # self.zeta["eta_bat"][:, building_id] = eta_bat
        # self.zeta["eta_Hsto"][:, building_id] = eta_Hsto
        # self.zeta["eta_Csto"][:, building_id] = eta_Csto

        # # dimensions: 1
        # self.zeta["eta_ehH"][building_id] = eta_ehH
        # self.zeta["c_bat_end"][building_id] = c_bat_end

    def target_update(self, zeta_local: dict, building_id: int):
        """Update rule for Target Actor: zeta_target <-- rho * zeta_target + (1 - rho) * zeta_local"""
        # dimensions: 24
        self.zeta["p_ele"][:, building_id] = (
            self.rho * self.zeta["p_ele"][:, building_id]
            + (1 - self.rho) * zeta_local["p_ele"][:, building_id]
        )
        self.zeta["eta_bat"][:, building_id] = (
            self.rho * self.zeta["eta_bat"][:, building_id]
            + (1 - self.rho) * zeta_local["eta_bat"][:, building_id]
        )
        self.zeta["eta_Hsto"][:, building_id] = (
            self.rho * self.zeta["eta_Hsto"][:, building_id]
            + (1 - self.rho) * zeta_local["eta_Hsto"][:, building_id]
        )
        self.zeta["eta_Csto"][:, building_id] = (
            self.rho * self.zeta["eta_Csto"][:, building_id]
            + (1 - self.rho) * zeta_local["eta_Csto"][:, building_id]
        )

        # dimensions: 1
        self.zeta["eta_ehH"][building_id] = (
            self.rho * self.zeta["eta_ehH"][building_id]
            + (1 - self.rho) * zeta_local["eta_ehH"][building_id]
        )
        self.zeta["c_bat_end"][building_id] = (
            self.rho * self.zeta["c_bat_end"][building_id]
            + (1 - self.rho) * zeta_local["c_bat_end"][building_id]
        )
        self.zeta["eta_Csto"][:, building_id] = (
            self.rho * self.zeta["eta_Csto"][:, building_id]
            + (1 - self.rho) * zeta_local["eta_Csto"][:, building_id]
        )

    def convert_to_torch_tensor(self, params: dict) -> dict:
        """Converts cp.param to dict[torch.tensor]"""
        params_dict = {}
        for key, value in params.items():
            params_dict[key] = torch.tensor(
                np.array(value.value, dtype=np.float), requires_grad=True
            )
            params_dict[key].grad = None

        return params_dict

    def gradient_actions(
        self, t: int, parameters: dict, critic: Critic, building_id: int
    ):
        """Computes dQ/da, where a is the set of actions for building `building_id` at timestep `t`"""
        # set all params except for actions as constants (zeroes)
        # for k, v in parameters.items():
        #     if "action" in k:
        #         parameters[k] += np.random.normal(0, 0.1)

        # set zeta's to constant
        # zeta = self.initialize_zeta(*[0] * len(self.zeta))

        # fetch params in loss calculation
        E_grid_prevhour = parameters["E_grid_prevhour"][t, building_id]
        E_grid_pkhist = (
            max(0, parameters["E_grid_prevhour"][t, building_id])
            if t == 0
            else np.max([0, *parameters["E_grid"][: (t + 1), building_id]])
        )

        # problem formulation using Critic optimizaiton
        prob = critic.get_problem(
            t, parameters, self.zeta, building_id, return_prob=True
        )  # mu(s_t, a_t, zeta)

        # ––––––––– TEMP –––––––––
        # self.get_problem(t, parameters, building_id)  # mu(s_t, a_t, zeta)
        # assert self.prob[t].is_dpp(), "Problem must be DPP to compute dA/dzeta"
        # prob = self.prob[t]
        # ––––––––– TEMP –––––––––

        (zeta_plus_params, variables_actor,) = (
            prob.param_dict,
            prob.var_dict,
        )

        zeta_plus_params_tensor_dict = self.convert_to_torch_tensor(zeta_plus_params)

        # critic forward pass to get E_grid. And then use E_grid to get Q. Then get dQ/da
        mu = CvxpyLayer(
            prob,
            parameters=list(zeta_plus_params.values()),
            variables=list(variables_actor.values()),
        )

        self.debug = [mu, zeta_plus_params_tensor_dict]  # TEMP
        try:
            E_grid, *_ = mu(*zeta_plus_params_tensor_dict.values())
        except:
            LOG(f"dQ/da Solver error! Building: {building_id}, Timestep: {t}")

            E_grid, *_ = mu(
                *zeta_plus_params_tensor_dict.values(),
                solver_args={
                    # "verbose": True,
                    "max_iters": 10_000_000,
                    "solve_method": "SCS",
                    "eps": 5e-2,
                },
            )

        # Reward Warping function, Critic forward pass - Step 2
        r, p, e = critic.get_alphas()
        alpha_ramp = torch.from_numpy(r).float()
        alpha_peak1 = torch.from_numpy(p).float()
        # alpha_elec = torch.from_numpy(e).float()

        ramping_cost = torch.abs(E_grid[0] - E_grid_prevhour)
        if len(E_grid) > 1:  # not at eod
            ramping_cost += torch.sum(
                torch.abs(E_grid[1:] - E_grid[:-1])
            )  # E_grid_t+1 - E_grid_t

        peak_net_electricity_cost = torch.max(
            torch.tensor(E_grid.max()),
            torch.tensor(E_grid_pkhist),
        )

        reward_warping_loss = (
            -alpha_ramp[building_id] * ramping_cost
            - alpha_peak1[building_id] * peak_net_electricity_cost
            # - torch.sum(alpha_elec[building_id][t:] * E_grid)
        )
        # make sure that the reward is negative
        assert (
            reward_warping_loss.item() <= 1
        ), f"Loss must be negative, got: {reward_warping_loss.item()}"
        # add virtual electricity cost

        # Gradient w.r.t parameters (math: \zeta) - Step 3
        reward_warping_loss.backward()

        # dimensions: 24 - Pad zeta
        p_ele_grad = np.pad(
            zeta_plus_params_tensor_dict["p_ele"].grad.numpy(),
            (t, 0),
            constant_values=np.nan,
        )
        assert len(p_ele_grad) == 24, f"Invalid dimension. found {len(p_ele_grad)}"

        eta_bat_grad = np.pad(
            zeta_plus_params_tensor_dict["eta_bat"].grad.numpy(),
            (t, 0),
            constant_values=np.nan,
        )
        assert len(eta_bat_grad) == 24, f"Invalid dimension. found {len(eta_bat_grad)}"

        eta_Hsto_grad = np.pad(
            zeta_plus_params_tensor_dict["eta_Hsto"].grad.numpy(),
            (t, 0),
            constant_values=np.nan,
        )
        assert (
            len(eta_Hsto_grad) == 24
        ), f"Invalid dimension. found {len(eta_Hsto_grad)}"

        eta_Csto_grad = np.pad(
            zeta_plus_params_tensor_dict["eta_Csto"].grad.numpy(),
            (t, 0),
            constant_values=np.nan,
        )
        assert (
            len(eta_Csto_grad) == 24
        ), f"Invalid dimension. found {len(eta_Csto_grad)}"

        # dimensions: 1
        eta_ehH_grad = zeta_plus_params_tensor_dict["eta_ehH"].grad.item()
        c_bat_end_grad = zeta_plus_params_tensor_dict["c_bat_end"].grad.item()

        dq_da = {}
        dq_da["p_ele_grad"] = p_ele_grad
        dq_da["eta_bat_grad"] = eta_bat_grad
        dq_da["eta_Hsto_grad"] = eta_Hsto_grad
        dq_da["eta_Csto_grad"] = eta_Csto_grad
        dq_da["eta_ehH_grad"] = eta_ehH_grad
        dq_da["c_bat_end_grad"] = c_bat_end_grad

        return reward_warping_loss.item(), dq_da

    def gradient_zeta(self, t: int, parameters: dict, building_id: int):
        """
        Computes dA/dzeta, i.e., overall backward pass of actor optimization (Actor forward pass).
        Here, A represents the set of actions and zeta is the set of differentiable parameters.
        """
        da_dzeta = defaultdict()

        # set all params except for zetas as constants
        # zeta_params = parameters
        # for k, v in zeta_params.items():
        #     if k not in self.zeta.keys():
        #         zeta_params[k] = np.full(v.shape, fill_value=0)

        self.get_problem(t, parameters, building_id)

        assert self.prob[t].is_dpp(), "Problem must be DPP to compute dA/dzeta"

        # actor optimization, i.e., forward pass.
        self.grad_debug = [self.prob[t], parameters]

        try:
            status = self.prob[t].solve(requires_grad=True)
            if status == float("inf") or status == float("-inf"):
                raise Error("infeasible solution found. Trying SCS...")
        except:  # try another solver
            status = self.prob[t].solve(
                requires_grad=True, solver="SCS", max_iters=1_000_000
            )  # Returns the optimal value.

        assert (
            float("-inf") < status < float("inf")
        ), "E2E dA/dzeta returned unbounded solution."

        self.prob[t].backward()  # compute gradient

        for key in self.zeta.keys():
            grad = self.prob[t].param_dict[key].gradient
            if len(grad.shape) == 0:
                da_dzeta[key + "_grad"] = grad
            else:
                da_dzeta[key + "_grad"] = np.pad(grad, (t, 0), constant_values=np.nan)

        return da_dzeta

    def E2E_grad(
        self, t: int, parameters: dict, critic: Critic, building_id: int
    ) -> dict:
        """
        Computes chain rule for: dQ/dzeta = dQ/da * da/dzeta
        • dQ/da -> set all params except for actions as constants.
        • da/dzeta -> actor optimization (forward pass).
        """
        loss, dq_da = self.gradient_actions(
            t, deepcopy(parameters), critic, building_id
        )
        return [loss, *dq_da.values()]

        da_dzeta = self.gradient_zeta(t, deepcopy(parameters), building_id)

        e2e = {}
        for k in self.zeta.keys():
            k += "_grad"
            try:
                e2e[k] = dq_da[k] * da_dzeta[k]
            except KeyError as k:
                pass
            except Exception as e:
                LOG(f"Timestep: {t}\nError message: {e}\nKey: {k}")

        return e2e.values()

    def backward(
        self,
        t: int,
        critic: Critic,  # Critic local-1
        batch_parameters: list,
        building_id: int,
    ):
        """
        Computes the gradient first for optimization given parameters `params`.
        Updates actor parameters accordingly.

        This function calculates math: \del_\zeta Q(s, \mu(s, \zeta), w) - see section 1.3.1
        Step 1. Solve Actor optimization w/ actions this time as parameters, whose values were obtained in Actor forward pass.
        Step 2. Use reward warping function w/ E^grid given from (1) and perform forward pass.
        Step 3. Take backward pass of (2) with parameters \zeta.
        """

        parameter_gradients = defaultdict(list)
        costs = []

        for day_param in batch_parameters:
            daily_cost = 0.0
            for r in range(24):
                LOG(f"E2E\tBuilding: {building_id}, r: {str(r).zfill(2)}")
                (
                    loss,
                    p_ele_grad,
                    eta_bat_grad,
                    eta_Hsto_grad,
                    eta_Csto_grad,
                    eta_ehH_grad,
                    c_bat_end_grad,
                ) = self.E2E_grad(r, day_param, critic, building_id)

                # store gradients
                parameter_gradients["p_ele_grad"].append(p_ele_grad)
                parameter_gradients["eta_bat_grad"].append(eta_bat_grad)
                parameter_gradients["eta_Hsto_grad"].append(eta_Hsto_grad)
                parameter_gradients["eta_Csto_grad"].append(eta_Csto_grad)
                parameter_gradients["eta_ehH_grad"].append(eta_ehH_grad)
                parameter_gradients["c_bat_end_grad"].append(c_bat_end_grad)

                daily_cost += loss
            costs.append(daily_cost)

        # –––––––––– Compute Average Gradient ––––––––––

        # dimension : 24
        parameter_gradients["p_ele_grad"] = np.nanmean(
            parameter_gradients["p_ele_grad"], axis=0
        )

        # dimension : 24
        parameter_gradients["eta_bat_grad"] = np.nanmean(
            parameter_gradients["eta_bat_grad"], axis=0
        )

        # dimension : 24
        parameter_gradients["eta_Hsto_grad"] = np.nanmean(
            parameter_gradients["eta_Hsto_grad"], axis=0
        )

        # dimension : 24
        parameter_gradients["eta_Csto_grad"] = np.nanmean(
            parameter_gradients["eta_Csto_grad"], axis=0
        )

        # dimension : 1
        parameter_gradients["eta_ehH_grad"] = np.array(
            parameter_gradients["eta_ehH_grad"]
        ).mean()

        # dimension : 1
        parameter_gradients["c_bat_end_grad"] = np.array(
            parameter_gradients["c_bat_end_grad"]
        ).mean()

        ### –––––––––––––––––––––––––––––– Log gradients ––––––––––––––––––––––––––––––
        for k, v in parameter_gradients.items():
            self._grads[building_id][k].append(v)
        ### –––––––––––––––––––––––––––––– Log model costs ––––––––––––––––––––––––––––––
        self._losses[building_id].append(np.mean(costs))

        ### Update Parameter using Adam
        NUM_HOURS = len(batch_parameters) * 24

        p_ele = self.optim[building_id]["p_ele"].update(
            (t - self.adam_offset) // NUM_HOURS,
            self.zeta["p_ele"][:, building_id],
            parameter_gradients["p_ele_grad"],
        )
        eta_bat = self.optim[building_id]["eta_bat"].update(
            (t - self.adam_offset) // NUM_HOURS,
            self.zeta["eta_bat"][:, building_id],
            parameter_gradients["eta_bat_grad"],
        )
        eta_Hsto = self.optim[building_id]["eta_Hsto"].update(
            (t - self.adam_offset) // NUM_HOURS,
            self.zeta["eta_Hsto"][:, building_id],
            parameter_gradients["eta_Hsto_grad"],
        )
        eta_Csto = self.optim[building_id]["eta_Csto"].update(
            (t - self.adam_offset) // NUM_HOURS,
            self.zeta["eta_Csto"][:, building_id],
            parameter_gradients["eta_Csto_grad"],
        )
        eta_ehH = self.optim[building_id]["eta_ehH"].update(
            (t - self.adam_offset) // NUM_HOURS,
            self.zeta["eta_ehH"][building_id],
            parameter_gradients["eta_ehH_grad"],
        )
        c_bat_end = self.optim[building_id]["c_bat_end"].update(
            ((t - self.adam_offset) // NUM_HOURS),
            self.zeta["c_bat_end"][building_id],
            parameter_gradients["c_bat_end_grad"],
        )

        ## Update Zeta
        self.set_zeta(
            (
                p_ele,
                eta_bat,
                eta_Hsto,
                eta_Csto,
                eta_ehH,
                c_bat_end,
            ),
            building_id,
        )
