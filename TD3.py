import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Predictor():
  """ @Zhiyao + @Mingyu - estimates parameters, loads data supplied to `Actor` """
  def __init__(...):
    pass
  #TODO: See https://github.com/QasimWani/ROLEVT/blob/main/colab_implicit_agent.ipynb for functionality on loader.
  def predict_data(self, state, ...):
    pass
  ### @Qasim -- original full-agent
  def full_agent_data(self, env):
    pass
  
  
  
  
class Actor:
    """ Define Differential Optimization framework for CL. """
    def __init__(self, t: int, parameters: dict, building_id: int, num_actions: int):
            """
            @Param:
            - `parameters` : data (dict) from r <= t <= T following `get_current_data` format.
            - `T` : 24 hours (constant)
            - `t` : hour to solve optimization for.
            - `building_id`: building index number (0-based)
            - `num_actions`: Number of actions for building
                NOTE: right now, this is an integer, but will be checked programmatically.
            Solves per building as specified by `building_id`. Note: 0 based.
            """
            T = 24
            window = T - t
            self.constraints = []
            self.costs = []
            self.t = t
            self.num_actions = num_actions

            # -- define action space -- #
            bounds_high, bounds_low = np.vstack([actions_spaces[building_id].high, actions_spaces[building_id].low])
            # parse to dictionary --- temp... need to check w/ state-action-dictionary.json !!! @Zhaiyao !!!
            if len(bounds_high) == 2: #bug
                    bounds_high = {'action_C': bounds_high[0], 'action_H': None, 'action_bat': bounds_high[1]}
                    bounds_low = {'action_C': bounds_low[0], 'action_H': None, 'action_bat': bounds_low[1]}
            else:
                    bounds_high = {'action_C': bounds_high[0], 'action_H': bounds_high[1],
                                    'action_bat': bounds_high[2]}
                    bounds_low = {'action_C': bounds_low[0], 'action_H': bounds_low[1], 'action_bat': bounds_low[2]}

            # -- define action space -- #

            # define parameters and variables

            ### --- Parameters ---
            p_ele = cp.Parameter(name='p_ele', shape=(window), value=parameters['p_ele'][t:, building_id])
            E_grid_prevhour = cp.Parameter(name='E_grid_prevhour',
                                            value=0)

            E_grid_pkhist = cp.Parameter(name='E_grid_pkhist',
                                            value=0)

            # max-min normalization of ramping_cost to downplay E_grid_sell weight.
            ramping_cost_coeff = cp.Parameter(name='ramping_cost_coeff',
                                                value=parameters['ramping_cost_coeff'][t, building_id])

            # Loads
            E_ns = cp.Parameter(name='E_ns', shape=(window), value=parameters['E_ns'][t:, building_id])
            H_bd = cp.Parameter(name='H_bd', shape=(window), value=parameters['H_bd'][t:, building_id])
            C_bd = cp.Parameter(name='C_bd', shape=(window), value=parameters['C_bd'][t:, building_id])

            # PV generations
            E_pv = cp.Parameter(name='E_pv', shape=(window), value=parameters['E_pv'][t:, building_id])

            # Heat Pump
            COP_C = cp.Parameter(name='COP_C', shape=(window), value=parameters['COP_C'][t:, building_id])
            E_hpC_max = cp.Parameter(name='E_hpC_max', value=parameters['E_hpC_max'][t, building_id])

            # Electric Heater
            eta_ehH = cp.Parameter(name='eta_ehH', value=parameters['eta_ehH'][t, building_id])
            E_ehH_max = cp.Parameter(name='E_ehH_max', value=parameters['E_ehH_max'][t, building_id])

            # Battery
            C_f_bat = cp.Parameter(name='C_f_bat', value=parameters['C_f_bat'][t, building_id])
            C_p_bat = parameters['C_p_bat'][
                    t, building_id]  # cp.Parameter(name='C_p_bat', value=parameters['C_p_bat'][t, building_id])
            eta_bat = cp.Parameter(name='eta_bat', value=parameters['eta_bat'][t, building_id])
            soc_bat_init = cp.Parameter(name='soc_bat_init', value=parameters['c_bat_init'][building_id])
            soc_bat_norm_end = cp.Parameter(name='soc_bat_norm_end', value=parameters['c_bat_end'][t,building_id])

            # Heat (Energy->dhw) Storage
            C_f_Hsto = cp.Parameter(name='C_f_Hsto', value=parameters['C_f_Hsto'][t, building_id])  # make constant.
            C_p_Hsto = cp.Parameter(name='C_p_Hsto', value=parameters['C_p_Hsto'][t, building_id])
            eta_Hsto = cp.Parameter(name='eta_Hsto', value=parameters['eta_Hsto'][t, building_id])
            soc_Hsto_init = cp.Parameter(name='soc_Hsto_init', value=parameters['c_Hsto_init'][building_id])

            # Cooling (Energy->cooling) Storage
            C_f_Csto = cp.Parameter(name='C_f_Csto', value=parameters['C_f_Csto'][t, building_id])
            C_p_Csto = cp.Parameter(name='C_p_Csto', value=parameters['C_p_Csto'][t, building_id])
            eta_Csto = cp.Parameter(name='eta_Csto', value=parameters['eta_Csto'][t, building_id])
            soc_Csto_init = cp.Parameter(name='soc_Csto_init', value=parameters['c_Csto_init'][building_id])

            ### --- Variables ---

            # relaxation variables - prevents numerical failures when solving optimization
            E_bal_relax = cp.Variable(name='E_bal_relax', shape=(window))  # electricity balance relaxation
            H_bal_relax = cp.Variable(name='H_bal_relax', shape=(window))  # heating balance relaxation
            C_bal_relax = cp.Variable(name='C_bal_relax', shape=(window))  # cooling balance relaxation

            E_grid = cp.Variable(name='E_grid', shape=(window))  # net electricity grid
            E_grid_sell = cp.Variable(name='E_grid_sell', shape=(window))  # net electricity grid

            E_hpC = cp.Variable(name='E_hpC', shape=(window))  # heat pump
            E_ehH = cp.Variable(name='E_ehH', shape=(window))  # electric heater

            SOC_bat = cp.Variable(name='SOC_bat', shape=(window))  # electric battery
            SOC_Brelax = cp.Variable(name='SOC_Brelax', shape=(
                    window))  # electrical battery relaxation (prevents numerical infeasibilities)
            action_bat = cp.Variable(name='action_bat', shape=(window))  # electric battery

            SOC_H = cp.Variable(name='SOC_H', shape=(window))  # heat storage
            SOC_Hrelax = cp.Variable(name='SOC_Hrelax',
                                        shape=(window))  # heat storage relaxation (prevents numerical infeasibilities)
            action_H = cp.Variable(name='action_H', shape=(window))  # heat storage

            SOC_C = cp.Variable(name='SOC_C', shape=(window))  # cooling storage
            SOC_Crelax = cp.Variable(name='SOC_Crelax', shape=(
                    window))  # cooling storage relaxation (prevents numerical infeasibilities)
            action_C = cp.Variable(name='action_C', shape=(window))  # cooling storage

            ### objective function
            ramping_cost = cp.abs(E_grid[0] - E_grid_prevhour) + cp.sum(
                    cp.abs(E_grid[1:] - E_grid[:-1]))  # E_grid_t+1 - E_grid_t
            peak_net_electricity_cost = cp.max(
                    cp.atoms.affine.hstack.hstack([*E_grid, E_grid_pkhist]))  # max(E_grid, E_gridpkhist)
            electricity_cost = cp.sum(p_ele * E_grid)
            selling_cost = -1e2 * cp.sum(E_grid_sell)  # not as severe as violating constraints

            ### relaxation costs - L1 norm
            # balance eq.
            E_bal_relax_cost = cp.sum(cp.abs(E_bal_relax))
            H_bal_relax_cost = cp.sum(cp.abs(H_bal_relax))
            C_bal_relax_cost = cp.sum(cp.abs(C_bal_relax))
            # soc eq.
            SOC_Brelax_cost = cp.sum(cp.abs(SOC_Brelax))
            SOC_Crelax_cost = cp.sum(cp.abs(SOC_Crelax))
            SOC_Hrelax_cost = cp.sum(cp.abs(SOC_Hrelax))

            self.costs.append(ramping_cost_coeff.value * ramping_cost +
                                peak_net_electricity_cost + 0*electricity_cost + selling_cost +
                                E_bal_relax_cost * 1e4 + H_bal_relax_cost * 1e4 + C_bal_relax_cost * 1e4
                                + SOC_Brelax_cost * 1e4 + SOC_Crelax_cost * 1e4 + SOC_Hrelax_cost * 1e4)

            ### constraints
            self.constraints.append(E_grid >= 0)
            self.constraints.append(E_grid_sell <= 0)

            # energy balance constraints
            self.constraints.append(
                    E_pv + E_grid + E_grid_sell + E_bal_relax == E_ns + E_hpC + E_ehH + action_bat * C_p_bat)  # electricity balance
            self.constraints.append(E_ehH * eta_ehH + H_bal_relax == action_H * C_p_Hsto + H_bd)  # heat balance

            # !!!!! Problem Child !!!!!
            self.constraints.append(E_hpC * COP_C + C_bal_relax == action_C * C_p_Csto + C_bd)  # cooling balance
            # !!!!! Problem Child !!!!!

            # heat pump constraints
            self.constraints.append(E_hpC <= E_hpC_max)  # maximum cooling
            self.constraints.append(E_hpC >= 0)  # constraint minimum cooling to positive
            # electric heater constraints
            self.constraints.append(E_ehH >= 0)  # constraint to PD
            self.constraints.append(E_ehH <= E_ehH_max)  # maximum limit

            # electric battery constraints

            self.constraints.append(
                    SOC_bat[0] == (1 - C_f_bat) * soc_bat_init + action_bat[0] * eta_bat + SOC_Crelax[
                            0])  # initial SOC
            # soc updates
            for i in range(1, window):  # 1 = t + 1
                    self.constraints.append(
                            SOC_bat[i] == (1 - C_f_bat) * SOC_bat[i - 1] + action_bat[i] * eta_bat + SOC_Crelax[i])
            self.constraints.append(SOC_bat[-1] == soc_bat_norm_end)  # soc terminal condition
            self.constraints.append(SOC_bat >= 0)  # battery SOC bounds
            self.constraints.append(SOC_bat <= 1)  # battery SOC bounds

            # Heat Storage constraints
            self.constraints.append(
                    SOC_H[0] == (1 - C_f_Hsto) * soc_Hsto_init + action_H[0] * eta_Hsto + SOC_Hrelax[
                            0])  # initial SOC
            # soc updates
            for i in range(1, window):
                    self.constraints.append(
                            SOC_H[i] == (1 - C_f_Hsto) * SOC_H[i - 1] + action_H[i] * eta_Hsto + SOC_Hrelax[i])
            self.constraints.append(SOC_H >= 0)  # battery SOC bounds
            self.constraints.append(SOC_H <= 1)  # battery SOC bounds

            # Cooling Storage constraints
            self.constraints.append(
                    SOC_C[0] == (1 - C_f_Csto) * soc_Csto_init + action_C[0] * eta_Csto + SOC_Crelax[
                            0])  # initial SOC
            # soc updates
            for i in range(1, window):
                    self.constraints.append(
                            SOC_C[i] == (1 - C_f_Csto) * SOC_C[i - 1] + action_C[i] * eta_Csto + SOC_Crelax[i])
            self.constraints.append(SOC_C >= 0)  # battery SOC bounds
            self.constraints.append(SOC_C <= 1)  # battery SOC bounds

            #### action constraints (limit to action-space)
            # format: AS[building_id][0/1 (high/low)][heat, cool, battery]

            heat_high, cool_high, battery_high = bounds_high
            heat_low, cool_low, battery_low = bounds_low

            assert len(bounds_high) == 3, 'Invalid number of bounds for actions - see dict defined in `Optim`'

            for high, low in zip(bounds_high.items(), bounds_low.items()):
                    key, h, l = [*high, low[1]]
                    if not (h and l):  # throw DeMorgan's in!!!
                            continue

                    # heating action
                    if key == 'action_C':
                            self.constraints.append(action_C <= h)
                            self.constraints.append(action_C >= l)
                    # cooling action
                    elif key == 'action_H':
                            self.constraints.append(action_H <= h)
                            self.constraints.append(action_H >= l)
                    # Battery action
                    elif key == 'action_bat':
                            self.constraints.append(action_bat <= h)
                            self.constraints.append(action_bat >= l)

    def get_problem(self):
            """ Returns raw problem """
            # Form objective.
            obj = cp.Minimize(*self.costs)
            # Form and solve problem.
            prob = cp.Problem(obj, self.constraints)

            return prob

    def get_constraints(self):
            """ Returns constraints for problem """
            return self.constraints

    def forward(self, debug=False, dispatch=False):
            prob = self.get_problem()  # Form and solve problem
            actions = {}
            try:
                    status = prob.solve(verbose=debug)  # Returns the optimal value.
            except:
                    return [0,0,0], 0 if dispatch else None, actions
            if float('-inf') < status < float('inf'):
                    pass
            else:
                    return "Unbounded Solution"

            for var in prob.variables():
                    if dispatch:
                            actions[var.name()] = np.array(
                                    var.value)  # no need to clip... automatically restricts range
                    else:
                            actions[var.name()] = var.value[0]  # no need to clip... automatically restricts range

            # Temporary... needs fixing!
            ## compute dispatch cost
            params = {x.name(): x.value for x in prob.parameters()}

            if dispatch:
                    ramping_cost = np.sum(np.abs(actions['E_grid'][1:] + actions['E_grid_sell'][1:] -
                                                    actions['E_grid'][:-1] - actions['E_grid_sell'][:-1]))
                    net_peak_electricity_cost = np.max(actions['E_grid'])
                    virtual_electricity_cost = np.sum(params['p_ele'] * actions['E_grid'])
                    dispatch_cost = virtual_electricity_cost  # ramping_cost + net_peak_electricity_cost + virtual_electricity_cost

            if self.num_actions == 2:
                    return [actions['action_H'], actions['action_bat']], dispatch_cost if dispatch else None
            return [actions['action_C'], actions['action_H'],
                    actions['action_bat']], dispatch_cost if dispatch else None, actions


      def backward(self, problem_optim1, zeta, alphas, variables_actor): #actor-update
        """ Computes the gradient for optimization given parameters `params` """ 
        ## TODO: implement this... see elastic-net (https://github.com/cvxgrp/cvxpylayers/blob/master/examples/torch/tutorial.ipynb)
        
        ## this function calculates math: \nabla_\zeta Q(s, \mu(s, \zeta), w)
        ## Step 1. Solve Actor optimization w/ actions this time as parameters, whose values were obtained in Actor forward pass.
        ## Step 2. Use reward warping function w/ E^grid given from (1) and perform forward pass.
        ## Step 3. Take backward pass of (2) with parameters \zeta.
        
        # Actor forward pass  - Step 1
        def convert_to_torch_tensor(params:list):
            var_arr = []
            for var in params:
                var_arr += [torch.tensor(var, requires_grad=True)]
            return var_arr
                
                
        fit_actor = CvxpyLayer(problem_optim1, parameters=zeta, variables=variables_actor)
        
        # zeta_tch = torch.tensor([zeta], requires_grads=True)
                
        # zeta_tech.grad = None
        zeta = convert_to_torch_tensor(zeta)
        E_grid, *_ = fit_actor(*zeta)
        
        # Q function forward pass - Step 2
        alpha_ramp, alpha_peak1, alpha_peak2 = alphas ### parameters from Critic update
        
        ramping_cost = torch.abs(E_grid[0] - E_grid_prevhour) + torch.sum(torch.abs(E_grid[1:] - E_grid[:-1])) # E_grid_t+1 - E_grid_t
        peak_net_electricity_cost = torch.max(E_grid, E_grid_peakhist) #max(E_grid, E_gridpkhist)
        
        reward_warping_loss = alpha_ramp * ramping_cost + alpha_peak1 * peak_net_electricity_cost + alpha_peak2 * torch.square(peak_net_electricity_cost)
        
        # Gradient w.r.t parameters (math: \zeta) - Step 3
        reward_warping_loss.backward()
        
        ### @jinming --- mean taken across all timesteps and updated for each timestep, i.e, we update each timestep w/ same gradient
        for i, param in enumerate(zeta):
            zeta[i] = param + np.mean(param.grad, axis=1)
        
        return zeta #updated actor params
        

class Critic:
  		
	def __init__(self, t:int, parameters:dict, building_id:int, num_actions:int):
      
      
       """ 
            @Param:
            - `parameters` : data (dict) from r <= t <= T following `get_current_data` format.
            - `T` : 24 hours (constant)
            - `t` : hour to solve optimization for.
            - `building_id`: building index number (0-based)
            - `num_actions`: Number of actions for building 
                NOTE: right now, this is an integer, but will be checked programmatically.
            Solves per building as specified by `building_id`. Note: 0 based.
            """
      
      
     	T = 24
     	window = T - t
        self.constraints = []
        self.costs = []
        self.t = t
        self.num_actions = num_actions
        
    	
         ### --- Parameters share with the actor
        p_ele = cp.Parameter(name='p_ele', shape=(window), value=parameters['p_ele'][t:, building_id])
        E_grid_prevhour = cp.Parameter(name='E_grid_prevhour', 
                                       value=parameters['E_grid_past'][min(t-1, 0), building_id])
        
        E_grid_pkhist = cp.Parameter(name='E_grid_pkhist', 
                                    value=parameters['E_grid_past'][:(t+1), building_id].max())
        
        # Loads
        E_ns = cp.Parameter(name='E_ns', shape=(window), value=parameters['E_ns'][t:, building_id])
        H_bd = cp.Parameter(name='H_bd', shape=(window), value=parameters['H_bd'][t:, building_id])
        C_bd = cp.Parameter(name='C_bd', shape=(window), value=parameters['C_bd'][t:, building_id])
        
        # PV generations
        E_pv = cp.Parameter(name='E_pv', shape=(window), value=parameters['E_pv'][t:, building_id])

        # Heat Pump
        COP_C = cp.Parameter(name='COP_C', shape=(window), value=parameters['COP_C'][t:, building_id])
        E_hpC_max = cp.Parameter(name='E_hpC_max', value=parameters['E_hpC_max'][t, building_id])

        # Electric Heater
        eta_ehH = cp.Parameter(name='eta_ehH', value=parameters['eta_ehH'][t, building_id])
        E_ehH_max = cp.Parameter(name='E_ehH_max', value=parameters['E_ehH_max'][t, building_id])

        # Battery
        C_f_bat = cp.Parameter(name='C_f_bat', value=parameters['C_f_bat'][t, building_id])
        C_p_bat = parameters['C_p_bat'][t, building_id] #cp.Parameter(name='C_p_bat', value=parameters['C_p_bat'][t, building_id])
        eta_bat = cp.Parameter(name='eta_bat', value=parameters['eta_bat'][t, building_id])
        soc_bat_init = cp.Parameter(name='soc_bat_init', value=parameters['c_bat_end'][0, building_id])
        soc_bat_norm_end = cp.Parameter(name='soc_bat_norm_end', value=parameters['c_bat_end'][1, building_id])
        
        # Heat (Energy->dhw) Storage
        C_f_Hsto = cp.Parameter(name='C_f_Hsto', value=parameters['C_f_Hsto'][t, building_id]) #make constant.
        C_p_Hsto = cp.Parameter(name='C_p_Hsto', value=parameters['C_p_Hsto'][t, building_id])
        eta_Hsto = cp.Parameter(name='eta_Hsto', value=parameters['eta_Hsto'][t, building_id])
        soc_Hsto_init = cp.Parameter(name='soc_Hsto_init', value=parameters['c_Hsto_end'][0, building_id])
        soc_Hsto_norm_end = cp.Parameter(name='soc_Hsto_norm_end', value=parameters['c_Hsto_end'][1, building_id])

        # Cooling (Energy->cooling) Storage
        C_f_Csto = cp.Parameter(name='C_f_Csto', value=parameters['C_f_Csto'][t, building_id])
        C_p_Csto = cp.Parameter(name='C_p_Csto', value=parameters['C_p_Csto'][t, building_id])
        eta_Csto = cp.Parameter(name='eta_Csto', value=parameters['eta_Csto'][t, building_id])
        soc_Csto_init = cp.Parameter(name='soc_Csto_init', value=parameters['c_Csto_end'][0, building_id])
        soc_Csto_norm_end = cp.Parameter(name='soc_Csto_norm_end', value=parameters['c_Csto_end'][1, building_id])

        ### --- Variables ---
        
        #relaxation variables - prevents numerical failures when solving optimization
        E_bal_relax = cp.Variable(name='E_bal_relax', shape=(window)) #electricity balance relaxation
        H_bal_relax = cp.Variable(name='H_bal_relax', shape=(window)) #heating balance relaxation
        C_bal_relax = cp.Variable(name='C_bal_relax', shape=(window)) #cooling balance relaxation
        
        E_grid = cp.Variable(name='E_grid', shape=(window)) #net electricity grid
        E_grid_sell = cp.Variable(name='E_grid_sell', shape=(window)) #net electricity grid
        
        E_hpC = cp.Variable(name='E_hpC', shape=(window)) #heat pump
        E_ehH = cp.Variable(name='E_ehH', shape=(window)) #electric heater
        
        SOC_bat = cp.Variable(name='SOC_bat', shape=(window)) #electric battery
        SOC_Brelax = cp.Variable(name='SOH_Brelax', shape=(window)) #electrical battery relaxation (prevents numerical infeasibilities)
        action_bat = cp.Parameter(name='action_bat', value=parameters['action_bat'][:, building_id], shape=(window)) #electric battery
    
        SOC_H = cp.Variable(name='SOC_H', shape=(window)) #heat storage
        SOC_Hrelax = cp.Variable(name='SOH_Crelax', shape=(window)) #heat storage relaxation (prevents numerical infeasibilities)
        action_H = cp.Parameter(name='action_H', value=parameters['action_H'][:, building_id], shape=(window)) #heat storage

        SOC_C = cp.Variable(name='SOC_C', shape=(window)) #cooling storage
        SOC_Crelax = cp.Variable(name='SOC_Crelax', shape=(window)) #cooling storage relaxation (prevents numerical infeasibilities)
        action_C = cp.Parameter(name='action_C', value=parameters['action_C'][:, building_id], shape=(window)) #cooling storage
        
        
        ### objective function
        
        ### NEEDS CHANGING ###
        
        ramping_cost = cp.abs(E_grid[0] - E_grid_prevhour) + cp.sum(cp.abs(E_grid[1:] - E_grid[:-1])) # E_grid_t+1 - E_grid_t
        peak_net_electricity_cost = cp.max(cp.atoms.affine.hstack.hstack([*E_grid, E_grid_pkhist])) #max(E_grid, E_gridpkhist)
        
        #https://docs.google.com/document/d/1QbqCQtzfkzuhwEJeHY1-pQ28disM13rKFGTsf8dY8No/edit?disco=AAAAMzPtZMU
        reward_func = alpha_ramp.value * ramping_cost + alpha_peak1.value * peak_net_electricity_cost + alpha_peak2.value * cp.square(peak_net_electricity_cost)
		        
        ### relaxation costs - L1 norm
        # balance eq.
        E_bal_relax_cost = cp.sum(cp.abs( E_bal_relax ))
        H_bal_relax_cost = cp.sum(cp.abs( H_bal_relax ))
        C_bal_relax_cost = cp.sum(cp.abs( C_bal_relax ))
        # soc eq.
        SOC_Brelax_cost = cp.sum(cp.abs( SOC_Brelax ))
        SOC_Crelax_cost = cp.sum(cp.abs( SOC_Crelax ))
        SOC_Hrelax_cost = cp.sum(cp.abs( SOC_Hrelax ))
        
        ### @jinming --- do we need relaxation costs? or is `reward_func` guaranteed to produce feasable solutions?
        self.costs.append(reward_func + E_bal_relax_cost*1e4 + H_bal_relax_cost*1e4 + C_bal_relax_cost*1e4 + SOC_Brelax_cost*1e4 + SOC_Crelax_cost*1e4 + SOC_Hrelax_cost*1e4)
        
        ### NEEDS CHANGING ###
        
        ### constraints
        self.constraints.append( E_grid >= 0 )
        self.constraints.append( E_grid_sell <= 0 )
        
        #energy balance constraints
        self.constraints.append( E_pv + E_grid + E_grid_sell + E_bal_relax == E_ns + E_hpC + E_ehH + action_bat * C_p_bat) #electricity balance
        self.constraints.append( E_ehH * eta_ehH + H_bal_relax == action_H * C_p_Hsto + H_bd ) #heat balance
        
        #!!!!! Problem Child !!!!!
        self.constraints.append( E_hpC * COP_C + C_bal_relax == action_C * C_p_Csto + C_bd ) #cooling balance
        #!!!!! Problem Child !!!!!
        
        
        #heat pump constraints
        self.constraints.append( E_hpC <= E_hpC_max ) #maximum cooling
        self.constraints.append( E_hpC >= 0 ) #constraint minimum cooling to positive
        #electric heater constraints
        self.constraints.append( E_ehH >= 0 ) #constraint to PD
        self.constraints.append( E_ehH <= E_ehH_max ) #maximum limit
        
        #electric battery constraints
        
        self.constraints.append( SOC_bat[0] == (1 - C_f_bat)*soc_bat_init + action_bat[0]*eta_bat + SOC_Crelax[0] ) #initial SOC
        #soc updates
        for i in range(1, window): #1 = t + 1
            self.constraints.append( SOC_bat[i] == (1 - C_f_bat)*SOC_bat[i - 1] + action_bat[i]*eta_bat + SOC_Crelax[i])
        self.constraints.append( SOC_bat[-1] == soc_bat_norm_end ) #soc terminal condition
        self.constraints.append(SOC_bat >= 0) #battery SOC bounds
        self.constraints.append(SOC_bat <= 1) #battery SOC bounds

        #Heat Storage constraints
        self.constraints.append( SOC_H[0] == (1 - C_f_Hsto) * soc_Hsto_init + action_H[0]*eta_Hsto + SOC_Hrelax[0]) #initial SOC
        #soc updates
        for i in range(1, window):
            self.constraints.append( SOC_H[i] == (1 - C_f_Hsto)*SOC_H[i - 1] + action_H[i]*eta_Hsto + SOC_Hrelax[i])
        self.constraints.append(SOC_H >= 0) #battery SOC bounds
        self.constraints.append(SOC_H <= 1) #battery SOC bounds
        
        #Cooling Storage constraints
        self.constraints.append( SOC_C[0] == (1 - C_f_Csto) * soc_Csto_init + action_C[0]*eta_Csto + SOC_Crelax[0]) #initial SOC
        #soc updates
        for i in range(1, window):
            self.constraints.append( SOC_C[i] == (1 - C_f_Csto)*SOC_C[i - 1] + action_C[i]*eta_Csto + SOC_Crelax[i])
        self.constraints.append(SOC_C >= 0) #battery SOC bounds
        self.constraints.append(SOC_C <= 1) #battery SOC bounds
        
        


    def get_problem(self):
        """ Returns raw problem """
        # Form objective.
        obj = cp.Minimize(*self.costs)
        # Form and solve problem.
        prob = cp.Problem(obj, self.constraints)
        
        return prob
    
    def get_constraints(self):
        """ Returns constraints for problem """
        return self.constraints
    
    def forward(self, debug=False, dispatch=False):
        prob = self.get_problem() #Form and solve problem
        Q_reward_warping_output = prob.solve(verbose=debug)  # output of reward warping function
		
        if float('-inf') < Q_reward_warping_output < float('inf'):
            G_t_tPlusn = get_past_rewards(buffer, t) + [Q_reward_warping_output] ## takes past rewards nd.array of shape T - t - 1 {! need to define this !}
            G_t_lambda = (1 - lambda_) * np.sum([lambda_**(i - 1) * G_t_tPlusn[t : t + i] for i in range(len(G_t_tPlusn)) ]) + lambda_**(24 - t - 1) * G_t_tPlusn[t]
            return G_t_lambda, prob.variables()['E_grid']
            
        else:
            return "Unbounded Solution"
          
      def backward(self, data, t, building_id):
        """ Define least-squares optimization for generating values for alpha_ramp,peak1/2 """
        y_r, E_grid = obtain_target_Q(Q1, Q2) #note: Q1 and Q2 are functions defined below!
        ### variables
        alpha_ramp = cp.Variable(name='ramp', shape=(24 - t))
        alpha_peak1 = cp.Variable(name='peak1', shape=(24 - t))
        alpha_peak2 = cp.Variable(name='peak2', shape=(24 - t))
        
        ### parameters
        E_grid = cp.Parameter(name='E_grid', shape=(24 - t), value=E_grid[t:, building_id])
        E_grid_pk_hist = cp.Parameter(name='E_grid_pk_hist', shape=(24 - t), value=data['E_grid_pk_hist'][:, building_id])
        y_r = cp.Parameter(name='y_r', value=y_r[t, building_id]) ### defined in terms of `obtain_target_Q`
        
        #### cost
        
        ramping_cost = cp.abs(E_grid[0] - E_grid_prevhour) + cp.sum(cp.abs(E_grid[1:] - E_grid[:-1])) # E_grid_t+1 - E_grid_t
        peak_net_electricity_cost = cp.max(cp.atoms.affine.hstack.hstack([*E_grid, E_grid_pkhist])) #max(E_grid, E_gridpkhist)
        
        #https://docs.google.com/document/d/1QbqCQtzfkzuhwEJeHY1-pQ28disM13rKFGTsf8dY8No/edit?disco=AAAAMzPtZMU
        self.cost = [cp.sum(cp.square(alpha_ramp * ramping_cost + alpha_peak1 * peak_net_electricity_cost + alpha_peak2 * cp.square(peak_net_electricity_cost) - y_r)) ]
      	
        #### constraints
        self.constraints = []
        
        #alpha-peak
        self.constraints.append( alpha_ramp <= 2 )
        self.constraints.append( alpha_ramp >= 0.1 )

        #alpha-peak
        self.constraints.append( alpha_peak1 <= 2 )
        self.constraints.append( alpha_peak1 >= 0.1 )

        #alpha-peak
        self.constraints.append( alpha_peak2 <= 2 )
        self.constraints.append( alpha_peak2 >= 0.1 )

        
        # Form objective.
        obj = cp.Minimize(*self.costs)
        # Form and solve problem.
        prob = cp.Problem(obj, self.constraints)
        
        optim_solution = prob.solve()
        
        assert float('-inf') < optim_solution < float('inf'), 'Unbounded solution/primal infeasable'
        
        solution = {}
        for var in prob.variables():
          solution[var.name()] = var.value
          
        return solution ### returns optimal values for 3 alphas. LOCAL UPDATE
        ## after this, we call Actor.backward() to get actor update and pass in data from critic update (alphas).  
        
        
    def target_update(self, data_target, data_local):
        _alpha_ramp, _alpha_peak1, _alpha_peak2 = zip(*backward(data_local, t, building_id).values()) ## local update after Least square optim.
        ### main target update
        alpha_ramp, alpha_peak1, alpha_peak2 = get_current_target_alphas(data_target)
        alpha_ramp, alpha_peak1, alpha_peak2 = rho * np.array([_alpha_ramp, _alpha_peak1, _alpha_peak2]) + (1 - rho) * np.array([alpha_ramp, alpha_peak1, alpha_peak2])
      
        return alpha_ramp, alpha_peak1, alpha_peak2 ### updated target Qx (x = 1/2 depending on param. need to call this function twice)
      

def Q1(self, state, action):
    ### input data for local critic
    return Critic(data_target_1).forward() ### G_t_lambda_target_1, E_grid_target_1

def Q2(self, state, action):
    ### input data for target critic
    return Critic(data_target_2).forward() ### G_t_lambda_target_2, E_grid_target_2

  
def obtain_target_Q(Q1, Q2):
  Q1, E_grid1 = Q1
  Q2, E_grid2 = Q2
  E_grid = [E_grid1, E_grid2]
  return min(Q1, Q2), E_grid[np.argmin(Q1, Q2)] ### y_t, min(Q1, Q2) of E_grid



### IMPLEMENT THIS

# class TD3(object):
#     def __init__(
#         self,
#         state_dim,
#         action_dim,
#         max_action,
#         discount=0.99,
#         tau=0.005,
#         policy_noise=0.2,
#         noise_clip=0.5,
#         policy_freq=2
#     ):

#         self.actor = Actor(state_dim, action_dim, max_action).to(device)
#         self.actor_target = copy.deepcopy(self.actor)
#         self.actor_optimizer = #actor-update (backward pass) torch.optim.Adam(self.actor.parameters(), lr=3e-4)

#         self.critic = Critic(state_dim, action_dim).to(device)
#         self.critic_target = copy.deepcopy(self.critic)
#         self.critic_optimizer = # (see 1.3.2) torch.optim.Adam(self.critic.parameters(), lr=3e-4)

#         self.max_action = max_action
#         self.discount = discount
#         self.tau = tau
#         self.policy_noise = policy_noise
#         self.noise_clip = noise_clip
#         self.policy_freq = policy_freq

#         self.total_it = 0


#     def select_action(self, state):
#         state = torch.FloatTensor(state.reshape(1, -1)).to(device)
#         return self.actor(state).cpu().data.numpy().flatten()


#     def train(self, replay_buffer, batch_size=256):
#         self.total_it += 1

#         # Sample replay buffer 
#         state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

#         with torch.no_grad():
#             # Select action according to policy and add clipped noise
#             noise = (
#                 torch.randn_like(action) * self.policy_noise
#             ).clamp(-self.noise_clip, self.noise_clip)
            
#             next_action = (
#                 self.actor_target(next_state) + noise
#             ).clamp(-self.max_action, self.max_action)

#             # Compute the target Q value
#             target_Q1, target_Q2 = self.critic_target(next_state, next_action)
#             target_Q = torch.min(target_Q1, target_Q2) # y_t  -> flowchart
#             target_Q = reward + not_done * self.discount * target_Q # line 13 from TD3 spinning up   <x?>

#         ### Q: does critic have 2 optimizationss? one for forward, to get E^grid, and another for backward, to get params?
            
        
#         # Get current Q estimates
#         current_Q1, current_Q2 = self.critic(state, action)

#         # Compute critic loss
#         critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

#         # Optimize the critic
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward() ### change this for least square.
#         self.critic_optimizer.step()

#         # Delayed policy updates -- end of meta-episode
#         if self.total_it % self.policy_freq == 0:
            
            
#             # Compute actor losse
#             actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
#             # Optimize the actor 
#             self.actor_optimizer.zero_grad()
#             actor_loss.backward()
#             self.actor_optimizer.step()

#             # Update the frozen target models
#             for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
#                 target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

#             for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
#                 target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


#     def save(self, filename):
#         torch.save(self.critic.state_dict(), filename + "_critic")
#         torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
#         torch.save(self.actor.state_dict(), filename + "_actor")
#         torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


#     def load(self, filename):
#         self.critic.load_state_dict(torch.load(filename + "_critic"))
#         self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
#         self.critic_target = copy.deepcopy(self.critic)

#         self.actor.load_state_dict(torch.load(filename + "_actor"))
#         self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
#         self.actor_target = copy.deepcopy(self.actor)
        

        
        ############ Main.py the running procesure from the city learn document
        
        
        
    #Initialization for 2 Q network parameters for alphas

    alpha_ramp_Q1 = 1
    alpha_peak1_Q1 = 1
    alpha_peak2_Q1 = 1

    alpha_ramp_Q2 = 1
    alpha_peak1_Q2 = 1
    alpha_peak2_Q2 = 1

    alpha_ramp_Q1target = 1
    alpha_peak1_Q1target = 1
    alpha_peak2_Q1target = 1

    alpha_ramp_Q2target = 1
    alpha_peak1_Q2target = 1
    alpha_peak2_Q2target = 1

    #Initialisation of actor (zeta) and critic (w) parameters after the end of RBC period

    zeta = #From RBC 14 days
    zeta_target = zeta_zctor

    w_Q1 = (zeta_actor, alpha_ramp_Q1, alpha_peak1_Q1,  alpha_peak2_Q1)
    w_Q2 = (zeta_actor, alpha_ramp_Q2, alpha_peak1_Q2,  alpha_peak2_Q2)

    w_Q1_target = (zeta_actor_target, alpha_ramp_Q1target, alpha_peak1_Q1target,  alpha_peak2_Q1target)
    w_Q2_target = (zeta_actor_target, alpha_ramp_Q2target, alpha_peak1_Q2target,  alpha_peak2_Q2target)