from citylearn import  CityLearn
from pathlib import Path
from agents.rbc import RBC
import sys
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import torch

# transform state from array to dictionary for convenience
def to_dic(state_bdg):
    s_dic = {}
    count = 0
    for uid in building_ids:
        state = state_bdg[count]
        count += 1
        if len(state) == 28:
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
        else:
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
            "electrical_storage_soc": state[24],
            "net_electricity_consumption": state[25],
            "carbon_intensity": state[26]}
        s_dic[uid] = s
    return s_dic

# function for plotting bias between actual solar gen and predicted
def plot_bias(bias, daytype, type):  # type = ["solar", "elec"]
    fig = plt.figure(figsize=[6.4*2.5, 6.4*2.5])
    data = bias
    subplots = [331, 332, 333, 334, 335, 336, 337, 338, 339]
    for i in range(len(subplots)):
        plt.subplot(subplots[i])
        axis = np.arange(1, 23+1)
        plt.plot(axis, data[i, 0, :], alpha=0.7, marker="o", label="predicted")
        plt.plot(axis, data[i, 1, :], alpha=0.7, marker="^", label="actual")
        # fig.plot(axis, r_rules, "r", alpha=0.3, label="Q value_proposed")
        # plt.legend()
        plt.xlabel("timestep")
        plt.ylim([0, 70])
        plt.minorticks_on()
        plt.grid()
        plt.title("day "+str(daytype[i]))
        if type == "solar":
            plt.ylabel("solar_gen[kWh]")
        elif type == "elec":
            plt.ylabel("NS elec demand[kWh]")
    plt.suptitle("climate zone 1")
    # plt.ylim(ylim)
    plt.legend()
    plt.show()

# update
def calculate_avg(solar_gen):
    avg = {uid: [] for uid in building_ids}
    for uid in building_ids:
        temporal = np.zeros(24)
        # sys.exit()
        size = np.shape(solar_gen[uid])[0]
        assert size % 24 == 0, "size for avg calculation is wrong"
        for i in range(size):
            temporal[i % 24] += solar_gen[uid][i, :].flatten()
        avg[uid] = temporal / (size / 24)
    return avg

def calculate_avg_elec(elec_dem, daytype):
    weekday = {uid: np.zeros(24) for uid in building_ids}
    weekend = {uid: np.zeros(24) for uid in building_ids}
    weekend1 = {uid: np.zeros(24) for uid in building_ids}
    count_day = {uid: 0 for uid in building_ids}
    count_end1 = {uid: 0 for uid in building_ids}
    count_end = {uid: 0 for uid in building_ids}
    for uid in building_ids:
        size = np.shape(elec_dem[uid])[0]
        assert size % 24 == 0, "size for avg calculation is wrong"
        for i in range(size):
            if daytype[uid][i] in [7]:
                weekend[uid][i % 24] += elec_dem[uid][i, :].flatten()
                count_end[uid] += 1
            elif daytype[uid][i] in [1, 8]:
                weekend1[uid][i % 24] += elec_dem[uid][i, :].flatten()
                count_end1[uid] += 1
            else:
                weekday[uid][i % 24] += elec_dem[uid][i, :].flatten()
                count_day[uid] += 1
        weekday[uid] = weekday[uid]/(count_day[uid]/24)
        weekend[uid] = weekend[uid]/(count_end[uid]/24)
        weekend1[uid] = weekend1[uid]/(count_end1[uid]/24)

    print("weekdays: ", count_day, "Mons: ", count_end, "Tues: ", count_end1)
    print(weekday)

    return weekday, weekend, weekend1

def fit_model(solar, solar_gen, solar_avg):
    x_solar = {uid: [] for uid in building_ids}
    y_solar = {uid: [] for uid in building_ids}
    for uid in building_ids:
        for i in range(2, np.shape(solar_gen[uid])[0]-1):
            x_append1 = [solar_gen[uid][i-2, 0]-solar_avg[uid][(i-2) % 24], solar_gen[uid][(i-1), 0]-solar_avg[uid][(i-1) % 24]]
            # past 2h solar gen, e.g. h=22, 23 and params in h=0 to fit solar in h=0
            x_append2 = [solar[uid][i, 0], solar[uid][i, 1]]  # next hour solar prediction
            y = solar_gen[uid][i, 0]-solar_avg[uid][i % 24]
            # print("fit: ", i, [x_append1, x_append2], solar_gen[i+1, 0])
            # print(x_append1, x_append2, i, (i+1) % 24, solar_gen[i+1, 0]-solar_avg[i % 24])
            x_solar[uid].append(x_append1 + x_append2)
            y_solar[uid].append([y])

    return x_solar, y_solar


def fit_model_elec(elec, elec_dem, avg_wkday, avg_wkend, avg_wkend2, daytype):
    x_elec = {uid: [] for uid in building_ids}
    y_elec = {uid: [] for uid in building_ids}
    for uid in building_ids:
        for i in range(2, np.shape(elec_dem[uid])[0] - 1):
            if daytype[uid][i-2] in [7]:
                x_temp1 = elec_dem[uid][i - 2, 0] - avg_wkend[uid][(i - 2) % 24]
            elif daytype[uid][i-2] in [1, 8]:
                x_temp1 = elec_dem[uid][i - 2, 0] - avg_wkend2[uid][(i - 2) % 24]
            else:
                x_temp1 = elec_dem[uid][i - 2, 0] - avg_wkday[uid][(i - 2) % 24]
            if daytype[uid][i - 1] in [7]:
                x_temp2 = elec_dem[uid][i - 1, 0] - avg_wkend[uid][(i - 1) % 24]
            elif daytype[uid][i-1] in [1, 8]:
                x_temp2 = elec_dem[uid][i - 1, 0] - avg_wkend2[uid][(i - 1) % 24]
            else:
                x_temp2 = elec_dem[uid][i - 1, 0] - avg_wkday[uid][(i - 1) % 24]

            x_append1 = [x_temp1, x_temp2]
            # past 2h solar gen
            x_append2 = [elec[uid][i, 0]]  # next hour solar prediction
            # x_append2 = [elec[i, 0]]    # ignore humidity
            if daytype[uid][i] in [7]:
                y = elec_dem[uid][i, 0] - avg_wkend[uid][i % 24]
                # print("\twkend\t",i % 24, avg_wkend[i % 24])
            elif daytype[uid][i] in [1, 8]:
                y = elec_dem[uid][i, 0] - avg_wkend2[uid][i % 24]
            else:
                y = elec_dem[uid][i, 0] - avg_wkday[uid][i % 24]
                # print("\twkday\t",i % 24, avg_wkend[i % 24])
            # print("fit: ", i, [x_append1, x_append2], solar_gen[i+1, 0])
            # print(x_append1, x_append2, i, (i+1) % 24, solar_gen[i+1, 0]-solar_avg[i % 24])
            x_elec[uid].append(x_append1 + x_append2)
            # print("x_pred for elec: ", x_append1 + x_append2)
            y_elec[uid].append([y])
            # print(daytype[i], x_append1 + x_append2, y)
    return x_elec, y_elec

def gather_input():
    global x_solar, x_solar_6h, x_solar_12h, x_solar_24h, x_solar_gen
    global x_elec_wkday, x_elec_6h, x_elec_12h, x_elec_24h, x_elec_dem, x_elec_tin
    input_solar = {uid: np.zeros([23, 2]) for uid in building_ids}
    input_elec = {uid: np.zeros([23, 2]) for uid in building_ids}
    for uid in building_ids:
        # input_solar[0, :] = x_solar[0, :]
        input_solar[uid][0:6, :] = x_solar_6h[uid][-6:, :]
        input_solar[uid][6:12, :] = x_solar_12h[uid][-6:, :]
        input_solar[uid][12:, :] = x_solar_24h[uid][-12:-1, :]

        # input_elec[0, :] = x_elec[0, :]
        input_elec[uid][0:6, :] = x_elec_6h[uid][-6:, :]
        input_elec[uid][6:12, :] = x_elec_12h[uid][-6:, :]
        input_elec[uid][12:, :] = x_elec_24h[uid][-12:-1, :]
    # print(input_array)
    return input_solar, input_elec


def predict(day):  # prediction need to be executed only at the first hour of the day
    input_solar, input_elec = gather_input()
    solar_gen = {uid: np.zeros([25], dtype=float) for uid in building_ids}
    elec_dem = {uid: np.zeros([25], dtype=float) for uid in building_ids}
    day_pred_solar = {uid: np.zeros([23]) for uid in building_ids}
    day_pred_elec = {uid: np.zeros([23]) for uid in building_ids}
    for uid in building_ids:
        solar_gen[uid][0] = x_solar_gen[uid][-2, 0] - x_solar_avg[uid][23]  # take prediction at the beginning of each day
        solar_gen[uid][1] = x_solar_gen[uid][-1, 0] - x_solar_avg[uid][0]
        if day[uid][-2] in [7]:
            elec_dem[uid][0] = x_elec_dem[uid][-2, 0] - x_elec_avg_wkend[uid][23]
        elif day[uid][-2] in [1, 8]:
            elec_dem[uid][0] = x_elec_dem[uid][-2, 0] - x_elec_avg_wkend1[uid][23]
        else:
            elec_dem[uid][0] = x_elec_dem[uid][-2, 0] - x_elec_avg_wkday[uid][23]
        if day[uid][-1] in [7]:
            elec_dem[uid][1] = x_elec_dem[uid][-1, 0] - x_elec_avg_wkend[uid][0]
        elif day[uid][-1] in [1, 8]:
            elec_dem[uid][1] = x_elec_dem[uid][-1, 0] - x_elec_avg_wkend1[uid][0]
        else:
            elec_dem[uid][1] = x_elec_dem[uid][-1, 0] - x_elec_avg_wkday[uid][0]

        for i in range(len(input_solar[uid])):
            x_pred = [[solar_gen[uid][i], solar_gen[uid][i+1], input_solar[uid][i, 0], input_solar[uid][i, 1]]]
            y_pred = regr_solar[uid].predict(x_pred)
            # print("x_pred for solar: ", x_pred)
            avg = x_solar_avg[uid][i+1]
            day_pred_solar[uid][i] = max(0, y_pred.item() + avg)
            solar_gen[uid][i+2] = y_pred.item()

        for i in range(len(input_elec[uid])):
            x_pred = [[elec_dem[uid][i], elec_dem[uid][i+1], input_elec[uid][i, 0]]]   # ignore humidity
            # print("x_pred for elec: ", x_pred)
            y_pred = regr_elec[uid].predict(x_pred)
            if day[uid][-1] in [7]:
                avg = x_elec_avg_wkend[uid][i+1]
                print("predict: ", day[uid][i], x_pred, y_pred, avg)
            elif day[uid][-1] in [1, 8]:
                avg = x_elec_avg_wkend1[uid][i+1]
            else:
                avg = x_elec_avg_wkday[uid][i+1]
            day_pred_elec[uid][i] = max(0, y_pred.item() + avg)
            # print("predict: ", day[i], x_pred, y_pred, avg)
            elec_dem[uid][i+2] = y_pred.item()
            # print("time, x_pred, y_pred: ",i, x_pred, solar_gen)
    return day_pred_solar, day_pred_elec


def record(s_dic):
    global x_solar, x_solar_6h, x_solar_12h, x_solar_24h, x_solar_gen, x_day
    global x_elec, x_elec_6h, x_elec_12h, x_elec_24h, x_elec_dem, x_elec_tin
    for uid in building_ids:
        daytype = s_dic[uid]["day"]
        x_day[uid][0, :] = [daytype]
        x_day[uid] = np.roll(x_day[uid], -1, 0)

        x_solar[uid][0, :] = [s_dic[uid]["diffuse_solar_rad"], s_dic[uid]["direct_solar_rad"]]
        x_solar_6h[uid][0, :] = [s_dic[uid]["diffuse_solar_rad_pred_6h"], s_dic[uid]["direct_solar_rad_pred_6h"]]
        x_solar_12h[uid][0, :] = [s_dic[uid]["diffuse_solar_rad_pred_12h"], s_dic[uid]["direct_solar_rad_pred_12h"]]
        x_solar_24h[uid][0, :] = [s_dic[uid]["diffuse_solar_rad_pred_24h"], s_dic[uid]["direct_solar_rad_pred_24h"]]
        x_solar_gen[uid][0, :] = [s_dic[uid]["solar_gen"]]
        x_solar[uid] = np.roll(x_solar[uid], -1, 0)  # using roll function is more convenient for indexing
        x_solar_6h[uid] = np.roll(x_solar_6h[uid], -1, 0)  # roll index=0 to index=-1
        x_solar_12h[uid] = np.roll(x_solar_12h[uid], -1, 0)
        x_solar_24h[uid] = np.roll(x_solar_24h[uid], -1, 0)
        x_solar_gen[uid] = np.roll(x_solar_gen[uid], -1, 0)

        x_elec[uid][0, :] = ([s_dic[uid]["t_out"], s_dic[uid]["rh_out"]])
        x_elec_6h[uid][0, :] = ([s_dic[uid]["t_out_pred_6h"], s_dic[uid]["rh_out_pred_6h"]])
        x_elec_12h[uid][0, :] = ([s_dic[uid]["t_out_pred_12h"], s_dic[uid]["rh_out_pred_12h"]])
        x_elec_24h[uid][0, :] = ([s_dic[uid]["t_out_pred_24h"], s_dic[uid]["rh_out_pred_24h"]])
        x_elec_dem[uid][0, :] = ([s_dic[uid]["non_shiftable_load"]])
        x_elec[uid] = np.roll(x_elec[uid], -1, 0)  # using roll function is more convenient for indexing
        x_elec_6h[uid] = np.roll(x_elec_6h[uid], -1, 0)  # roll index=0 to index=-1
        x_elec_12h[uid] = np.roll(x_elec_12h[uid], -1, 0)
        x_elec_24h[uid] = np.roll(x_elec_24h[uid], -1, 0)
        x_elec_dem[uid] = np.roll(x_elec_dem[uid], -1, 0)

# initialize env

climate_zone = 1
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

building_ids = ["Building_" + str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]]

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()

# define buffer and hyperparams (named with CAPITAL LETTERS)

BUFFER_SIZE = 24*7

GAMMA = 0.15
# initialize buffer
x_solar = {uid: np.zeros([BUFFER_SIZE, 2], dtype=float) for uid in building_ids}
x_solar_6h = {uid: np.zeros([BUFFER_SIZE, 2], dtype=float) for uid in building_ids}
x_solar_12h = {uid: np.zeros([BUFFER_SIZE, 2], dtype=float) for uid in building_ids}
x_solar_24h = {uid: np.zeros([BUFFER_SIZE, 2], dtype=float) for uid in building_ids}
x_solar_gen = {uid: np.zeros([BUFFER_SIZE, 1], dtype=float) for uid in building_ids}
x_solar_avg = {uid: np.zeros([24], dtype=float) for uid in building_ids}
# x_solar = np.zeros([BUFFER_SIZE, 2], dtype=float)
# x_solar_6h = np.zeros([BUFFER_SIZE, 2], dtype=float)
# x_solar_12h = np.zeros([BUFFER_SIZE, 2], dtype=float)
# x_solar_24h = np.zeros([BUFFER_SIZE, 2], dtype=float)
# x_solar_gen = np.zeros([BUFFER_SIZE, 1], dtype=float)
# x_solar_avg = np.zeros([24], dtype=float)

x_day = {uid: np.zeros([BUFFER_SIZE, 1], dtype=float) for uid in building_ids}
x_elec = {uid: np.zeros([BUFFER_SIZE, 2], dtype=float) for uid in building_ids}
x_elec_6h = {uid: np.zeros([BUFFER_SIZE, 2], dtype=float) for uid in building_ids}
x_elec_12h = {uid: np.zeros([BUFFER_SIZE, 2], dtype=float) for uid in building_ids}
x_elec_24h = {uid: np.zeros([BUFFER_SIZE, 2], dtype=float) for uid in building_ids}
x_elec_dem = {uid: np.zeros([BUFFER_SIZE, 1], dtype=float) for uid in building_ids}
x_elec_avg_wkday = {uid: np.zeros([24], dtype=float) for uid in building_ids}
x_elec_avg_wkend = {uid: np.zeros([24], dtype=float) for uid in building_ids}
x_elec_avg_wkend1 = {uid: np.zeros([24], dtype=float) for uid in building_ids}
# x_day = np.zeros([BUFFER_SIZE, 1], dtype=float)
# x_elec = np.zeros([BUFFER_SIZE, 2], dtype=float)
# x_elec_6h = np.zeros([BUFFER_SIZE, 2], dtype=float)
# x_elec_12h = np.zeros([BUFFER_SIZE, 2], dtype=float)
# x_elec_24h = np.zeros([BUFFER_SIZE, 2], dtype=float)
# x_elec_dem = np.zeros([BUFFER_SIZE, 1], dtype=float)
#
# x_elec_avg_wkday = np.zeros([24], dtype=float)
# x_elec_avg_wkend = np.zeros([24], dtype=float)
# x_elec_avg_wkend1 = np.zeros([24], dtype=float)

bias_solar = {uid: np.zeros([2, 23], dtype=float) for uid in building_ids}   # col0: prediction; col1: actual value
bias_elec = {uid: np.zeros([2, 23], dtype=float) for uid in building_ids}  # col0: prediction; col1: actual value

# start
state = env.reset()
# transfer state values to buffer
s_dic = to_dic(state)
hour = s_dic["Building_1"]["hour"]
day = s_dic["Building_1"]["day"]
print("day, hour: ", day, hour)

record(s_dic)
# x_solar, x_solar_gen = append_avg(x_solar, x_solar_gen, state)
agents = RBC(actions_spaces)

done = False
# test for RBC
action = agents.select_action(state) # action for hour=0

# the following loop collects initial data for fitting
while not done:
    next_state, reward, done, _ = env.step(action)
    timestep = env.time_step  # the next time step
    s_dic = to_dic(next_state)
    hour = s_dic["Building_1"]["hour"]
    day = s_dic["Building_1"]["day"]
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
        x_elec_avg_wkday, x_elec_avg_wkend, x_elec_avg_wkend1 = calculate_avg_elec(x_elec_dem, x_day)
        print(x_solar_avg)
    if timestep == BUFFER_SIZE:
        break

# define regression model and fit the model
regr_solar = {uid: linear_model.LinearRegression() for uid in building_ids}
regr_elec = {uid: linear_model.LinearRegression() for uid in building_ids}
# regr_elec_1 = {uid: linear_model.LinearRegression() for uid in building_ids}
# regr_elec_2 = {uid: linear_model.LinearRegression() for uid in building_ids}

x_s, y_s = fit_model(x_solar, x_solar_gen, x_solar_avg)  # fit solar
x_e, y_e = fit_model_elec(x_elec, x_elec_dem, x_elec_avg_wkday, x_elec_avg_wkend, x_elec_avg_wkend1, x_day)    # fit elec
for uid in building_ids:
    regr_solar[uid].fit(x_s[uid], y_s[uid])
    regr_elec[uid].fit(x_e[uid], y_e[uid])

solar_pred, elec_pred = predict(x_day)
#  x_in shape: [z_{t-2}, z_t{t-1}, dif_{t}, dir_{t}]

count = 0   # for bias record
bias_s_total = {uid: np.zeros([10, 2, 23]) for uid in building_ids}
bias_e_total = {uid: np.zeros([10, 2, 23]) for uid in building_ids}
daytype = {uid: np.zeros([11]) for uid in building_ids}
# print(x_solar_gen.transpose()[:, -24:])

# this loop is with solar prediction at the end of each timestep before executing actions
while not done:
    next_state, reward, done, _ = env.step(action)
    timestep = env.time_step  # the next time step
    # append to buffer
    s_dic = to_dic(next_state)
    hour = s_dic["Building_1"]["hour"]
    day = s_dic["Building_1"]["day"]
    print("day, hour: ", day, hour)
    record(s_dic)

    action_next = agents.select_action(next_state)
    # print(x_solar_gen.transpose()[:, -24:])
    # print("time: ", timestep)
    # print(action_next)
    state = next_state
    action = action_next

    if hour == 1:  # beginning of the day

        # ----------show deviation of prediction------------------------
        if timestep > BUFFER_SIZE+24*30*3:
            for uid in building_ids:
                bias_solar[uid][0] = solar_pred[uid]
                bias_solar[uid][1] = x_solar_gen[uid][-25:-2, :].flatten()
                bias_elec[uid][0] = elec_pred[uid]
                bias_elec[uid][1] = x_elec_dem[uid][-25:-2, :].flatten()

                # plot_bias(bias_solar, type="solar")
                bias_s_total[uid][count, :, :] = bias_solar[uid]
                bias_e_total[uid][count, :, :] = bias_elec[uid]
            count += 1
            # plot_bias(bias_elec, type="elec")
        # ----------update moving avg-----------------------------------
        for uid in building_ids:
            new_solar = x_solar_gen[uid][-26:-2, 0]
            x_solar_avg[uid] = x_solar_avg[uid] * (1-GAMMA) + new_solar * GAMMA
            new_elec = x_elec_dem[uid][-26:-2, 0]
            last_day = x_day[uid][-2, 0]
            daytype[uid][count - 1] = last_day
            if last_day in [7]:
                x_elec_avg_wkend[uid] = x_elec_avg_wkend[uid] * (1-GAMMA) + new_elec * GAMMA
            elif last_day in [1, 8]:
                x_elec_avg_wkend1[uid] = x_elec_avg_wkend1[uid] * (1 - GAMMA) + new_elec * GAMMA
            else:
                x_elec_avg_wkday[uid] = x_elec_avg_wkday[uid] * (1-GAMMA) + new_elec * GAMMA
        # ----------refit the model-------------------------------------
        x_s, y_s = fit_model(x_solar, x_solar_gen, x_solar_avg)  # fit solar
        x_e, y_e = fit_model_elec(x_elec, x_elec_dem, x_elec_avg_wkday, x_elec_avg_wkend, x_elec_avg_wkend1, x_day)  # fit elec
        for uid in building_ids:
            regr_solar[uid].fit(x_s[uid], y_s[uid])
            regr_elec[uid].fit(x_e[uid], y_e[uid])
        # ----------make new prediction---------------------------------
        solar_pred, elec_pred = predict(x_day)
        print(elec_pred)

          # day of estimation

        if count == 10:
            for uid in building_ids:
                # plot_bias(np.array(bias_e_total[uid]), daytype[uid][0: 10], type="elec")
                plot_bias(np.array(bias_s_total[uid]), daytype[uid][0: 10], type="solar")
            sys.exit()
            count = 0



# cost = env.cost()
# print(cost)