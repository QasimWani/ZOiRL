import cvxpy as cp
import numpy as np
from utils import *

class Optim():
    def __init__(self, T):
        #define parameters and variables
        self.constraints = []

        #define timesteps
        # T = 24#*365*4
        window = 24

        #define misc. params 
        CI = cp.Parameter(name='CI', shape=(T), value=get_data(T)) #carbon intensity

        #heating
        H_sto = cp.Variable(name='H^STO', shape=(T)) 
        H_bd = cp.Parameter(name='H^bd', shape=(T), value=get_data(T))
        H_eh = cp.Parameter(name='H^eh', shape=(T), value=get_data(T))
        COP_H = cp.Parameter(name='COP^H', shape=(T), value=get_data(T))
        H_hp = cp.Parameter(name='H^hp', shape=(T), value=get_data(T))
        self.constraints.append(H_hp == H_sto - H_bd + H_eh) #H_hp = H_sto - H_bd + H_eh

        #cooling
        C_sto = cp.Variable(name='C^STO', shape=(T))
        C_bd = cp.Parameter(name='C^bd', shape=(T), value=get_data(T))
        C_dev = cp.Parameter(name='C^dev', shape=(T), value=get_data(T))
        COP_C = cp.Parameter(name='COP^C', shape=(T), value=get_data(T))
        C_hp = cp.Parameter(name='C^hp', shape=(T), value=get_data(T))
        self.constraints.append(C_hp == C_bd + C_dev - C_sto) #C_hp = C_bd + C_dev - C_sto

        #electricity
        E_bat = cp.Variable(name='E^bat', shape=(T))
        E_ns = cp.Parameter(name='E^NS', shape=(T), value=get_data(T))
        E_bd_C =  C_hp / COP_C
        E_bd_dhw = H_hp / COP_H
        E_pv = cp.Parameter(name='E^pv', shape=(T), value=get_data(T))
        E_grid = cp.Parameter(name='E^grid', shape=(T), value=get_data(T))
        self.constraints.append(E_ns + E_bd_C + E_bd_dhw + E_bat == E_pv + E_grid )
        
        self.variables = 

    def solve(self):
        self._problem.solve()
        return self._target

problem = MyProblem(4, ...)
for param_value in param_values:
    problem.param.value = param_value
    answer = problem.solve()