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
def plot_bias(bias, type):  # type = ["solar", "elec"]
    fig = plt.figure(figsize=[6.4 * 2, 6.4])
    data = bias
    axis = np.arange(23)
    plt.plot(axis, data[0, :], alpha=0.7, marker="o", label="predicted")
    plt.plot(axis, data[1, :], alpha=0.7, marker="^", label="actual")
    # plt.plot_date(axis, data[0, :], label='predicted')
    # plt.scatter(24, data[0, -24:], label='predicted')

    # fig.plot(axis, r_rules, "r", alpha=0.3, label="Q value_proposed")
    plt.legend()
    plt.xlabel("timestep")
    if type == "solar":
        plt.ylabel("solar_gen[kWh]")
    elif type == "elec":
        plt.ylabel("Non-shiftable elec demand[kWh]")
    # plt.ylim(ylim)
    plt.title('Prediction Accuracy with Buffer='+str(int(BUFFER_SIZE/24))+"days, type="+str(type))
    plt.minorticks_on()
    # plt.legend()
    plt.grid()
    plt.show()

# update
def calculate_avg(solar_gen):
    temporal = np.zeros(24)
    # sys.exit()
    size = np.shape(solar_gen)[0]
    assert size % 24 == 0, "size for avg calculation is wrong"
    for i in range(size):
        temporal[i % 24] += solar_gen[i, :].flatten()

    return temporal/(size/24)

def calculate_avg_elec(elec_dem, daytype):
    weekday = np.zeros(24)
    weekend = np.zeros(24)
    count_day = 0
    count_end = 0
    size = np.shape(elec_dem)[0]
    assert size % 24 == 0, "size for avg calculation is wrong"
    for i in range(size):
        if daytype[i] in [6, 7, 8]:
            weekend[i % 24] += elec_dem[i, :].flatten()
            count_end += 1
        else:
            weekday[i % 24] += elec_dem[i, :].flatten()
            count_day += 1
    print("weekdays: ", count_day, "weekends: ", count_end)

    return weekday/(count_day/24), weekend/(count_end/24)

def fit_model(solar, solar_gen, solar_avg):
    x_solar = []
    y_solar = []

    for i in range(2, np.shape(solar_gen)[0]-1):
        x_append1 = [solar_gen[i-2, 0]-solar_avg[(i-2) % 24], solar_gen[(i-1), 0]-solar_avg[(i-1) % 24]]
        # past 2h solar gen, e.g. h=22, 23 and params in h=0 to fit solar in h=0
        x_append2 = [solar[i, 0], solar[i, 1]]  # next hour solar prediction
        y = solar_gen[i, 0]-solar_avg[i % 24]
        # print("fit: ", i, [x_append1, x_append2], solar_gen[i+1, 0])
        # print(x_append1, x_append2, i, (i+1) % 24, solar_gen[i+1, 0]-solar_avg[i % 24])
        x_solar.append(x_append1 + x_append2)
        y_solar.append([y])

    return x_solar, y_solar


def fit_model_elec(elec, elec_dem, avg_wkday, avg_wkend, daytype):
    x_elec = []
    y_elec = []

    for i in range(2, np.shape(elec_dem)[0] - 1):
        if daytype[i-2] in [6, 7, 8]:
            x_temp1 = elec_dem[i - 2, 0] - avg_wkend[(i - 2) % 24]
        else:
            x_temp1 = elec_dem[i - 2, 0] - avg_wkday[(i - 2) % 24]
        if daytype[i - 1] in [6, 7, 8]:
            x_temp2 = elec_dem[i - 1, 0] - avg_wkend[(i - 1) % 24]
        else:
            x_temp2 = elec_dem[i - 1, 0] - avg_wkday[(i - 1) % 24]

        x_append1 = [x_temp1, x_temp2]
        # past 2h solar gen
        x_append2 = [elec[i, 0], elec[i, 1]]  # next hour solar prediction
        if daytype[i] in [6, 7, 8]:
            y = elec_dem[i, 0] - avg_wkend[i % 24]
        else:
            y = elec_dem[i, 0] - avg_wkday[i % 24]
        # print("fit: ", i, [x_append1, x_append2], solar_gen[i+1, 0])
        # print(x_append1, x_append2, i, (i+1) % 24, solar_gen[i+1, 0]-solar_avg[i % 24])
        x_elec.append(x_append1 + x_append2)
        y_elec.append([y])
    return x_elec, y_elec

def gather_input():
    global x_solar, x_solar_6h, x_solar_12h, x_solar_24h, x_solar_gen
    global x_elec_wkday, x_elec_6h_wkday, x_elec_12h_wkday, x_elec_24h_wkday, x_elec_dem_wkday
    global x_elec_wkend, x_elec_6h_wkend, x_elec_12h_wkend, x_elec_24h_wkend, x_elec_dem_wkend
    global x_elec_tin
    input_solar = np.zeros([23, 2])
    # input_solar[0, :] = x_solar[0, :]
    input_solar[0:6, :] = x_solar_6h[-6:, :]
    input_solar[6:12, :] = x_solar_12h[-6:, :]
    input_solar[12:, :] = x_solar_24h[-12:-1, :]

    input_elec = np.zeros([23, 2])
    # input_elec[0, :] = x_solar[0, :]
    input_elec[0:6, :] = x_solar_6h[-6:, :]
    input_elec[6:12, :] = x_solar_12h[-6:, :]
    input_elec[12:, :] = x_solar_24h[-12:-1, :]
    # print(input_array)
    return input_solar, input_elec


def predict(day):  # prediction need to be executed only at the first hour of the day
    input_solar, input_elec = gather_input()
    solar_gen = np.zeros([25], dtype=float)
    elec_dem = np.zeros([25], dtype=float)
    day_pred_solar = np.zeros([23])
    day_pred_elec = np.zeros([23])
    solar_gen[0] = x_solar_gen[-2, 0] - x_solar_avg[23]  # take prediction at the beginning of each day
    solar_gen[0] = x_solar_gen[-1, 0] - x_solar_avg[0]
    if day[-2] in [6, 7, 8]:
        elec_dem[0:2] = x_elec_dem[-2, 0] - x_elec_avg_wkend[23]
    else:
        elec_dem[0:2] = x_elec_dem[-2, 0] - x_elec_avg_wkday[23]
    if day[-1] in [6, 7, 8]:
        elec_dem[0:2] = x_elec_dem[-1, 0] - x_elec_avg_wkend[0]
    else:
        elec_dem[0:2] = x_elec_dem[1, 0] - x_elec_avg_wkday[0]

    for i in range(len(input_solar)):
        x_pred = [[solar_gen[i], solar_gen[i+1], input_solar[i, 0], input_solar[i, 1]]]
        y_pred = regr_solar.predict(x_pred)
        avg = x_solar_avg[i+1]
        day_pred_solar[i] = max(0, y_pred.item() + avg)
        solar_gen[i+2] = max(0, y_pred.item() - avg)

    for i in range(len(input_elec)):
        x_pred = [[elec_dem[i], elec_dem[i+1], input_elec[i, 0], input_elec[i, 1]]]
        y_pred = regr_elec.predict(x_pred)
        if day[i] in [6, 7, 8]:
            avg = x_elec_avg_wkend[i+1]
        else:
            avg = x_elec_avg_wkday[i+1]
        day_pred_elec[i] = max(0, y_pred.item() + avg)
        # print("predict: ", i, x_pred, y_pred, avg)
        elec_dem[i+2] = max(0, y_pred.item() - avg)
        # print("time, x_pred, y_pred: ",i, x_pred, solar_gen)

    return day_pred_solar, day_pred_elec


def record(s_dic):
    global x_solar, x_solar_6h, x_solar_12h, x_solar_24h, x_solar_gen, x_day
    global x_elec, x_elec_6h, x_elec_12h, x_elec_24h, x_elec_dem, x_elec_tin

    daytype = s_dic["day"]
    x_day[0, :] = [daytype]
    x_day = np.roll(x_day, -1, 0)

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

    x_elec[0, :] = [s_dic["t_out"], s_dic["rh_out"]]
    x_elec_6h[0, :] = [s_dic["t_out_pred_6h"], s_dic["rh_out_pred_6h"]]
    x_elec_12h[0, :] = [s_dic["t_out_pred_12h"], s_dic["rh_out_pred_12h"]]
    x_elec_24h[0, :] = [s_dic["t_out_pred_24h"], s_dic["rh_out_pred_24h"]]
    x_elec_dem[0, :] = [s_dic["non_shiftable_load"]]
    x_elec = np.roll(x_elec, -1, 0)  # using roll function is more convenient for indexing
    x_elec_6h = np.roll(x_elec_6h, -1, 0)  # roll index=0 to index=-1
    x_elec_12h = np.roll(x_elec_12h, -1, 0)
    x_elec_24h = np.roll(x_elec_24h, -1, 0)
    x_elec_dem = np.roll(x_elec_dem, -1, 0)

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

BUFFER_SIZE = 24*7

GAMMA = 0.15
# initialize buffer
x_solar = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_solar_6h = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_solar_12h = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_solar_24h = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_solar_gen = np.zeros([BUFFER_SIZE, 1], dtype=float)
x_solar_avg = np.zeros([24], dtype=float)

x_day = np.zeros([BUFFER_SIZE, 1], dtype=float)
x_elec = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_elec_6h = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_elec_12h = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_elec_24h = np.zeros([BUFFER_SIZE, 2], dtype=float)
x_elec_dem = np.zeros([BUFFER_SIZE, 1], dtype=float)

x_elec_avg_wkday = np.zeros([24], dtype=float)
x_elec_avg_wkend = np.zeros([24], dtype=float)

x_elec_tin = np.zeros([BUFFER_SIZE, 1], dtype=float)

bias_solar = np.zeros([2, 23], dtype=float)   # col0: prediction; col1: actual value
bias_elec = np.zeros([2, 23], dtype=float)   # col0: prediction; col1: actual value

# start
state = env.reset()
# transfer state values to buffer
s_dic = to_dic(state)
hour = s_dic["hour"]
day = s_dic["day"]
print("day, hour: ", day, hour)

record(s_dic)
# x_solar, x_solar_gen = append_avg(x_solar, x_solar_gen, state)

agents = RBC(actions_spaces)

done = False
# test for RBC
action = agents.select_action(state) # action for hour=0
# print(action)
# the following loop collects initial data for first fitting
while not done:
    next_state, reward, done, _ = env.step(action)
    timestep = env.time_step # the next time step
    s_dic = to_dic(next_state)
    hour = s_dic["hour"]
    day = s_dic["day"]
    print("day, hour: ", day, hour)
    record(s_dic)

    # x_solar, x_solar_gen = append_avg(x_solar, x_solar_gen, next_state)
    action_next = agents.select_action(next_state)

    # print("time: ", timestep)
    # print(action_next)
    state = next_state
    action = action_next
    if timestep == BUFFER_SIZE-1:
        x_solar_avg = calculate_avg(x_solar_gen)
        x_elec_avg_wkday, x_elec_avg_wkend = calculate_avg_elec(x_elec_dem, x_day)
    if timestep == BUFFER_SIZE:
        break

# define regression model and fit the model
regr_solar = linear_model.LinearRegression()
regr_elec = linear_model.LinearRegression()

'''
# make the first predictionm, hour-ahead
x_pred_solar = [[x_solar_gen[timestep - 2, 0] - x_solar_avg[(timestep - 2) % 24],
                 x_solar_gen[timestep - 1, 0] - x_solar_avg[(timestep - 1) % 24],
                 x_solar[timestep, 0], x_solar[timestep, 1]]]
y_pred_solar = regr_solar.predict(x_pred_solar) + x_solar_avg[(timestep) % 24]
y_pred_solar = max(0, y_pred_solar.item())

x_pred_elec = [[x_solar_gen[timestep - 1, 0] - x_solar_avg[(timestep - 1) % 24],
                 x_solar_gen[timestep, 0] - x_solar_avg[timestep % 24],
                 x_solar[timestep, 0], x_solar[timestep, 1]]]
y_pred_elec = regr_elec.predict(x_pred_elec)  # x_solar_avg[(timestep + 1) % 24]
y_pred_elec = max(0, y_pred_solar.item())
'''
x_s, y_s = fit_model(x_solar, x_solar_gen, x_solar_avg)  # fit solar
x_e, y_e = fit_model_elec(x_elec, x_elec_dem, x_elec_avg_wkday, x_elec_avg_wkend, x_day)    # fit elec
regr_solar.fit(x_s, y_s)
regr_elec.fit(x_e, y_e)

solar_pred, elec_pred = predict(x_day)
#  x_in shape: [z_{t-2}, z_t{t-1}, dif_{t}, dir_{t}]

count = 0   # for bias record
# print(x_solar_gen.transpose()[:, -24:])

# this loop is with solar prediction at the end of each timestep before executing actions
while not done:
    next_state, reward, done, _ = env.step(action)
    timestep = env.time_step  # the next time step
    # append to buffer
    s_dic = to_dic(next_state)
    hour = s_dic["hour"]
    day = s_dic["day"]
    print("day, hour: ", day, hour)
    record(s_dic)

    action_next = agents.select_action(next_state)
    # print(x_solar_gen.transpose()[:, -24:])
    # print("time: ", timestep)
    # print(action_next)
    state = next_state
    action = action_next
    count += 1

    if hour == 1:  # beginning of the day

        # ----------show deviation of prediction------------------------
        if timestep > BUFFER_SIZE+24*25:
            bias_solar[0] = solar_pred
            bias_solar[1] = x_solar_gen[-24:-1, :].flatten()
            bias_elec[0] = elec_pred
            bias_elec[1] = x_elec_dem[-24:-1, :].flatten()
            plot_bias(bias_solar, type="solar")
            plot_bias(bias_elec, type="elec")
        # ----------update moving avg-----------------------------------
        new_solar = x_solar_gen[-25:-1, 0]
        x_solar_avg = x_solar_avg * (1-GAMMA) + new_solar * GAMMA
        new_elec = x_elec_dem[-25:-1, 0]
        new_day = x_day[-1, 0]
        if new_day in [6, 7, 8]:
            x_elec_avg_wkend = x_elec_avg_wkend * (1-GAMMA) + new_elec * GAMMA
        else:
            x_elec_avg_wkday = x_elec_avg_wkday * (1-GAMMA) + new_elec * GAMMA
        # ----------refit the model-------------------------------------
        x_s, y_s = fit_model(x_solar, x_solar_gen, x_solar_avg)  # fit solar
        x_e, y_e = fit_model_elec(x_elec, x_elec_dem, x_elec_avg_wkday, x_elec_avg_wkend, x_day)  # fit elec
        regr_solar.fit(x_s, y_s)
        regr_elec.fit(x_e, y_e)
        # ----------make new prediction---------------------------------
        solar_pred, elec_pred = predict(x_day)


    if timestep == BUFFER_SIZE+24*30:
        sys.exit()


# cost = env.cost()
# print(cost)