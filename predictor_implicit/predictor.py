from citylearn import  CityLearn
from pathlib import Path
from agents.rbc import RBC
import sys
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import torch

# transform state from array to dictionary for convenience
def to_dic(state):
    state = state[0]    # bdg 1
    s = {
    "month": state[0],
    "day": state[1],
    "hour": state[2],
    "t_out": state[3],
    "t_out_pred_6h": state[4],
    "t_out_pred_12h": state[5],
    "t_out_pred_24h": state[6],
    "rh_out": state[7],
    "rh_out_pred_6h": state[8],
    "rh_out_pred_12h": state[9],
    "rh_out_pred_24h": state[10],
    "diffuse_solar_rad": state[11],
    "diffuse_solar_rad_pred_6h": state[12],
    "diffuse_solar_rad_pred_12h": state[13],
    "diffuse_solar_rad_pred_24h": state[14],
    "direct_solar_rad": state[15],
    "direct_solar_rad_pred_6h": state[16],
    "direct_solar_rad_pred_12h": state[17],
    "direct_solar_rad_pred_24h": state[18],
    "t_in": state[19],
    "rh_in": state[20],
    "non_shiftable_load": state[21],
    "solar_gen": state[22],
    "cooling_storage_soc": state[23],
    "dhw_storage_soc": state[24],
    "electrical_storage_soc": state[25],
    "net_electricity_consumption": state[26],
    "carbon_intensity": state[27]}

    return s

# function for plotting bias between actual solar gen and predicted
def plot_bias(bias):
    fig = plt.figure(figsize=[6.4 * 2, 6.4])
    data = bias
    axis = np.arange(24)
    plt.plot(axis, data[0, :], alpha=0.7, marker="o", label="predicted")
    plt.plot(axis, data[1, :], alpha=0.7, marker="^", label="actual")
    # plt.plot_date(axis, data[0, :], label='predicted')
    # plt.scatter(24, data[0, -24:], label='predicted')

    # fig.plot(axis, r_rules, "r", alpha=0.3, label="Q value_proposed")
    plt.legend()
    plt.xlabel("timestep")
    plt.ylabel("solar_gen[kWh]")
    # plt.ylim(ylim)
    plt.title('Prediction Accuracy with Buffer='+str(int(BUFFER_SIZE/24))+"days")
    plt.minorticks_on()
    # plt.legend()
    plt.grid()
    plt.show()

# update
def calculate_avg(solar_gen):
    temporal = np.zeros(24)
    # sys.exit()
    size = np.shape(solar_gen)[0]

    for i in range(size):
        temporal[i % 24] += solar_gen[i, :].flatten()

    return temporal/(size/24)

def fit_model(solar, solar_gen, solar_avg):
    x_input = []
    y_output = []
    for i in range(2, np.shape(solar_gen)[0]-1):
        x_append1 = [solar_gen[i-1, 0]-solar_avg[(i-1) % 24], solar_gen[i, 0]-solar_avg[i % 24]]  # past 2h solar gen
        x_append2 = [solar[i, 0], solar[i, 1]]  # next hour solar prediction
        y = solar_gen[i+1, 0]-solar_avg[(i+1) % 24]
        # print("fit: ", i, [x_append1, x_append2], solar_gen[i+1, 0])
        # print(x_append1, x_append2, i, (i+1) % 24, solar_gen[i+1, 0]-solar_avg[i % 24])
        x_input.append(x_append1 + x_append2)
        y_output.append([y])
    return x_input, y_output


def gather_input():
    global x_solar, x_solar_6h, x_solar_12h, x_solar_24h, x_solar_gen
    input_array = np.zeros([24, 2])
    input_array[0, :] = x_solar[0, :]
    input_array[1:7, :] = x_solar_6h[-6:, :]
    input_array[7:13, :] = x_solar_12h[-6:, :]
    input_array[13:, :] = x_solar_24h[-12:-1, :]
    # print(input_array)
    return input_array


def predict():  # prediction need to be executed only at the last hour of the day
    x_input = gather_input()
    # print(x_input)
    x_gen = np.zeros([26], dtype=float)
    day_pred = np.zeros([24])
    x_gen[0:2] = x_solar_gen[-2:].flatten() - x_solar_avg[22: 24] # take prediction at the end of each day
    # print(x_gen)
    for i in range(24):
        x_pred = [[x_gen[i], x_gen[i+1], x_input[i, 0], x_input[i, 1]]]
        y_pred = regr.predict(x_pred)
        avg = x_solar_avg[i]
        day_pred[i] = max(0, y_pred.item()+avg)
        # print("predict: ", i, x_pred, y_pred, avg)
        x_gen[i+2] = max(0, y_pred.item() - x_solar_avg[i])
        # print("time, x_pred, y_pred: ",i, x_pred, x_gen)
    return day_pred


def record(s_dic):
    global x_solar, x_solar_6h, x_solar_12h, x_solar_24h, x_solar_gen
    x_solar[0, :] = [s_dic["diffuse_solar_rad"], s_dic["direct_solar_rad"]]
    x_solar_6h[0, :] = [s_dic["diffuse_solar_rad_pred_6h"], s_dic["direct_solar_rad_pred_6h"]]
    x_solar_12h[0, :] = [s_dic["diffuse_solar_rad_pred_12h"], s_dic["direct_solar_rad_pred_12h"]]
    x_solar_24h[0, :] = [s_dic["diffuse_solar_rad_pred_24h"], s_dic["direct_solar_rad_pred_24h"]]
    x_solar_gen[0, :] = [s_dic["solar_gen"]]

    x_solar = np.roll(x_solar, -1, 0)  # using roll function is more convenient for indexing
    x_solar_6h = np.roll(x_solar_6h, -1, 0)  # roll index=0 to index=-1
    x_solar_12h = np.roll(x_solar_12h, -1, 0)
    x_solar_24h = np.roll(x_solar_24h, -1, 0)
    x_solar_gen = np.roll(x_solar_gen, -1, 0)

# initialize env

climate_zone = 5
TOTAL_TIME_STEP = 8760  # 8760
params = {'data_path': Path(
    "D:/Reinforcement Learning/CityLearn-master/CityLearn-master/data/Climate_Zone_" + str(climate_zone)),
          'building_attributes': 'building_attributes.json',
          'weather_file': 'weather_data.csv',
          'solar_profile': 'solar_generation_1kW.csv',
          'carbon_intensity': 'carbon_intensity.csv',
          'building_ids': ["Building_" + str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]],
          'buildings_states_actions': 'buildings_state_action_space.json',
          'simulation_period': (0, TOTAL_TIME_STEP - 1),  # 8760
          'cost_function': ['ramping', '1-load_factor', 'average_daily_peak', 'peak_demand',
                            'net_electricity_consumption', 'carbon_emissions'],
          'central_agent': False,
          'save_memory': False}

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
env = CityLearn(**params)
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()

# define buffer and hyperparams (named with CAPITAL LETTERS)

BUFFER_SIZE = 24*14
GAMMA = 0.2
# initialize buffer
x_solar = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_solar_6h = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_solar_12h = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_solar_24h = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_solar_gen = np.zeros([BUFFER_SIZE, 1], dtype=float)
x_solar_avg = np.zeros([24], dtype=float)
bias = np.zeros([2, 24], dtype=float)   # col0: prediction; col1: actual value

# start
state = env.reset()
# transfer state values to buffer
s_dic = to_dic(state)
hour = s_dic["hour"]

record(s_dic)
# x_solar, x_solar_gen = append_avg(x_solar, x_solar_gen, state)

agents = RBC(actions_spaces)

done = False
# test for RBC
action = agents.select_action(state)
# print(action)
# the following loop collects initial data for first fitting
while not done:
    next_state, reward, done, _ = env.step(action)
    timestep = env.time_step # the next time step
    s_dic = to_dic(next_state)
    hour = s_dic["hour"]

    record(s_dic)

    # x_solar, x_solar_gen = append_avg(x_solar, x_solar_gen, next_state)
    action_next = agents.select_action(next_state)

    # print("time: ", timestep)
    # print(action_next)
    state = next_state
    action = action_next
    if timestep == BUFFER_SIZE-1:
        print(x_solar_gen[0])
        '''
        reshape x_solar_gen to calculate mean value
        '''
        x_solar_avg = calculate_avg(x_solar_gen)
        break
'''
init data collection finished
to-do:
    *1) fit regression model by (z_i, z_{i-1}, diffuse_solar, direct_power)
    2) make multi-hour-ahead (starting test from 1) predictions
    3) meanwhile collect current_day data
    4) update solar_avg, refit regression model
    5) plot data
    *Q: how to realize multiple linear regression?
'''

# define regression model and fit the model
regr = linear_model.LinearRegression()
x, y = fit_model(x_solar, x_solar_gen, x_solar_avg)

regr.fit(x, y)
#  x_in shape: [z_{t-1}, z_t, dif_{t+1}, dir_{t+1}]
timestep = env.time_step
print(timestep % 24)
'''
# make the first prediction
x_pred = [[x_solar_gen[timestep-1, 0]-x_solar_avg[(timestep-1) % 24],
          x_solar_gen[timestep, 0]-x_solar_avg[timestep % 24],
          x_solar[timestep, 0], x_solar[timestep, 1]]]
y_pred = regr.predict(x_pred) + x_solar_avg[(timestep+1) % 24]
y_pred = max(0, y_pred.item())
'''


day_pred = predict()

count = 0   # for bias record
# print(x_solar_gen.transpose()[:, -24:])

# this loop is with solar prediction at the end of each timestep before executing actions
while not done:
    next_state, reward, done, _ = env.step(action)
    timestep = env.time_step  # the next time step

    # append to buffer
    s_dic = to_dic(next_state)
    record(s_dic)
    '''
    # make next_hour prediction
    x_pred = [[x_solar_gen[-2, 0] - x_solar_avg[(timestep - 1) % 24],
               x_solar_gen[-1, 0] - x_solar_avg[timestep % 24],
               x_solar[-1, 0], x_solar[-1, 1]]]
    y_pred = regr.predict(x_pred) + x_solar_avg[(timestep + 1) % 24]
    y_pred = max(0, y_pred.item())
    '''
    action_next = agents.select_action(next_state)
    # print(x_solar_gen.transpose()[:, -24:])
    # print("time: ", timestep)
    # print(action_next)
    state = next_state
    action = action_next
    count += 1

    if (timestep+1) % 24 == 0:  # end of the day
        """
        * update mean value
        refit regression model
        """
        # ----------show deviation of prediction------------------------
        bias[0] = day_pred
        bias[1] = x_solar_gen[-24:, :].flatten()
        plot_bias(bias)
        # ----------update moving avg-----------------------------------
        new_avg = calculate_avg(x_solar_gen[-24:, :])
        x_solar_avg = x_solar_avg * (1-GAMMA) + new_avg * GAMMA
        # print(x_solar_avg)
        # ----------refit the model-------------------------------------
        x, y = fit_model(x_solar, x_solar_gen, x_solar_avg)
        regr.fit(x, y)
        # ----------make new prediction---------------------------------
        day_pred = predict()


    if timestep == BUFFER_SIZE+24*7:
        sys.exit()


# cost = env.cost()
# print(cost)