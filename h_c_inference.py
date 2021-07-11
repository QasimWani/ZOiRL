from citylearn import  CityLearn
from pathlib import Path
from agents.rbc import RBC
import numpy as np
from utils import ReplayBuffer
import sys
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, QuantileRegressor

true_val_h = [10.68, 49.35, 1e-05, 1e-05, 60.12, 105.12, 85.44, 111.96, 102.24]
true_val_b = [140, 80, 50, 75, 50, 30, 40, 30, 35]
true_val_c = [618.12, 227.37, 414.68, 383.565, 244.685, 96.87, 127.82, 165.45, 175.23]
CF_C = 0.006
CF_H = 0.008
CF_B = 0
building_ids = range(9)

"""
problem w code: no cooling loads 
"""


class Predictor:
    def __init__(self):
        self.building_ids = range(9)
        self.state_buffer = ReplayBuffer(buffer_size=365, batch_size=32)
        self.action_buffer = ReplayBuffer(buffer_size=365, batch_size=32)
        self.load_buffer = ReplayBuffer(buffer_size=365, batch_size=32)
        self.regr = LinearRegression(fit_intercept=False, positive=True)
        self.avg_h_load = {uid: np.zeros(24) for uid in building_ids}
        self.avg_c_load = {uid: np.zeros(24) for uid in building_ids}
        self.timestep = 0

        pass

    def record_dic(self, current_state: list, current_action: list):
        """
        call record_dic at the beginning of each step
        """
        now_state = self.state_to_dic(current_state)
        state = self.state_buffer.get_recent()
        parse_state = self.parse_data(state, now_state)
        self.state_buffer.add(parse_state)
        current_action = self.action_to_dic(current_action)
        action = self.action_buffer.get_recent()
        parse_action = self.parse_data(action, current_action)
        self.action_buffer.add(parse_action)

    def state_to_dic(self, state_list: list):
        state_bdg = {}
        for uid in building_ids:
            state = state_list[uid]
            s = {
                "month": state[0],
                "day": state[1],
                "hour": state[2],
                "daylight_savings_status": state[3],
                "t_out": state[4],
                "t_out_pred_6h": state[5],
                "t_out_pred_12h": state[6],
                "t_out_pred_24h": state[7],
                "rh_out": state[8],
                "rh_out_pred_6h": state[9],
                "rh_out_pred_12h": state[10],
                "rh_out_pred_24h": state[11],
                "diffuse_solar_rad": state[12],
                "diffuse_solar_rad_pred_6h": state[13],
                "diffuse_solar_rad_pred_12h": state[14],
                "diffuse_solar_rad_pred_24h": state[15],
                "direct_solar_rad": state[16],
                "direct_solar_rad_pred_6h": state[17],
                "direct_solar_rad_pred_12h": state[18],
                "direct_solar_rad_pred_24h": state[19],
                "t_in": state[20],
                "avg_unmet_setpoint": state[21],
                "rh_in": state[22],
                "non_shiftable_load": state[23],
                "solar_gen": state[24],
                "cooling_storage_soc": state[25],
                "dhw_storage_soc": state[26],
                "electrical_storage_soc": state[27],
                "net_electricity_consumption": state[28],
                "carbon_intensity": state[29]}
            state_bdg[uid] = s

        s_dic = {}
        daytype = [state_bdg[i]["day"] for i in self.building_ids]
        hour = [state_bdg[i]["hour"] for i in self.building_ids]
        t_out = [state_bdg[i]["t_out"] for i in self.building_ids]
        rh_out = [state_bdg[i]["rh_out"] for i in self.building_ids]
        t_in = [state_bdg[i]["t_in"] for i in self.building_ids]
        rh_in = [state_bdg[i]["rh_in"] for i in self.building_ids]
        elec_dem = [state_bdg[i]["non_shiftable_load"] for i in self.building_ids]
        solar_gen = [state_bdg[i]["solar_gen"] for i in self.building_ids]
        soc_c = [state_bdg[i]["cooling_storage_soc"] for i in self.building_ids]
        soc_h = [state_bdg[i]["dhw_storage_soc"] for i in self.building_ids]
        soc_b = [state_bdg[i]["electrical_storage_soc"] for i in self.building_ids]
        elec_cons = [state_bdg[i]["net_electricity_consumption"] for i in self.building_ids]

        s_dic["daytype"] = daytype
        s_dic["hour"] = hour
        s_dic["t_out"] = t_out
        s_dic["rh_out"] = rh_out
        s_dic["t_in"] = t_in
        s_dic["rh_in"] = rh_in
        s_dic["elec_dem"] = elec_dem
        s_dic["solar_gen"] = solar_gen
        s_dic["soc_c"] = soc_c
        s_dic["soc_h"] = soc_h
        s_dic["soc_b"] = soc_b
        s_dic["elec_cons"] = elec_cons

        return s_dic

    def action_to_dic(self, action):
        s_dic = {}
        a_c = [action[i][0] for i in self.building_ids]
        a_h = [action[i][1] for i in self.building_ids]
        a_b = [action[i][2] for i in self.building_ids]

        s_dic["a_c"] = a_c
        s_dic["a_h"] = a_h
        s_dic["a_b"] = a_b

        return s_dic

    def cop_cal(self, temp):
        eta_tech = 0.22
        target_c = 8
        if temp == target_c:
            cop_c = 20
        else:
            cop_c = eta_tech * (target_c + 273.15) / (temp - target_c)
        if cop_c <= 0 or cop_c > 20:
            cop_c = 20
        return cop_c

    def parse_data(self, data: dict, current_data: dict) -> list:
        """Parses `current_data` for optimization and loads into `data`"""
        for key, value in current_data.items():
            if key not in data:
                data[key] = []
            data[key].append(value)
        return data

    def infer_load(self):
        """
        Note: h&c should be inferred simultaneously
        inferring all-day h&c loads according to three methods accordingly:
        1. direct calculation and power balance equation (if either is clipped)
        2. two-point regression estimation (if nearby (t-1 or t+1) loads are calculated directly)
        3. main method regression estimation (at least two different COPs among consecutive three hours)
        **assuming conduct inference at the beginning hour of the day(aft recording in buffer, bef executing actions)
        **so that when we obtain from ReplayBuffer.get_recent(), we get day-long data.
        :return: daily h&c load inference
        """
        est_c_load = {uid: np.zeros(24) for uid in self.building_ids}
        est_h_load = {uid: np.zeros(24) for uid in self.building_ids}
        c_hasest = {uid: np.zeros(24) for uid in self.building_ids} # -1:clipped, 0:non-est, 1:regression, 2: moving avg
        h_hasest = {uid: np.zeros(24) for uid in self.building_ids}
        # hasest indicates whether every hour of the day has estimation.
        # only when all 0 become 1 in has_est, the function runs over.
        effi_h = 0.9

        for uid in self.building_ids:
            # starting from t=0, need a loop to cycle time
            # say at hour=t, check if the action of c/h is clipped
            # if so, directly calculate h/c load and continue this loop
            repeat_times = 0
            time = 0
            jump_out = False
            while jump_out is not True:
                if c_hasest[uid][time] in [0, 2]:
                    now_state = self.state_buffer.get(-2)
                    now_c_soc = now_state["soc_c"][time][uid]
                    now_h_soc = now_state["soc_h"][time][uid]
                    now_b_soc = now_state["soc_b"][time][uid]
                    now_t_out = now_state["t_out"][time][uid]
                    now_solar = now_state["solar_gen"][time][uid]
                    now_elec_dem = now_state["elec_dem"][time][uid]
                    cop_c = self.cop_cal(now_t_out)  # cop at t
                    now_action = self.action_buffer.get(-2)
                    now_action_c = now_action["a_c"][time][uid]
                    now_action_h = now_action["a_h"][time][uid]
                    now_action_b = now_action["a_b"][time][uid]
                    prev_state = now_state if time != 0 else self.state_buffer.get(-3)
                    prev_t_out = prev_state["t_out"][time - 1][uid]  # when time=0, time-1=-1
                    if time != 23:
                        next_state = now_state
                        next_c_soc = next_state["soc_c"][time+1][uid]
                        next_h_soc = next_state["soc_h"][time+1][uid]
                        next_b_soc = next_state["soc_b"][time+1][uid]
                        next_t_out = next_state["t_out"][time+1][uid]
                        next_elec_con = next_state["elec_cons"][time+1][uid]
                        y = now_solar + next_elec_con - now_elec_dem - \
                            (true_val_c[uid]/cop_c) * (next_c_soc - (1-CF_C) * now_c_soc)\
                            - (true_val_h[uid]/effi_h) * (next_h_soc - (1-CF_H) * now_h_soc)\
                            - (next_b_soc - (1-CF_B) * now_b_soc) * true_val_b[uid] / 0.9
                    else:
                        next_state = self.state_buffer.get_recent()
                        next_c_soc = next_state["soc_c"][0][uid]
                        next_h_soc = next_state["soc_h"][0][uid]
                        next_b_soc = next_state["soc_b"][0][uid]
                        next_t_out = next_state["t_out"][0][uid]
                        next_elec_con = next_state["elec_cons"][0][uid]
                        y = now_solar + next_elec_con - now_elec_dem - (true_val_c[uid] / cop_c) * (next_c_soc - (1 - CF_C) * now_c_soc) * 1.1 \
                            - (true_val_h[uid] / effi_h) * (next_h_soc - (1 - CF_H) * now_h_soc) - (next_b_soc - (1 - CF_B) * now_b_soc) \
                            * true_val_b[uid] / 0.9

                    a_clip_c = next_c_soc - (1-CF_C) * now_c_soc
                    a_clip_h = next_h_soc - (1-CF_H) * now_h_soc

                    if repeat_times == 0:   # can we calculate direct when now_action > 0?
                        if uid in [2, 3]:
                            c_load = max(0, y * cop_c)
                            h_load = 0
                            est_h_load[uid][time] = h_load
                            est_c_load[uid][time] = c_load
                            c_hasest[uid][time], h_hasest[uid][time] = -1, -1
                        else:
                            if abs(a_clip_c - now_action_c) > 0.001 and now_action_c < 0: # cooling get clipped
                                c_load = abs(a_clip_c * true_val_c[uid])
                                if abs(a_clip_h - now_action_h) > 0.001 and now_action_h < 0:  # heating get clipped
                                    h_load = a_clip_h * true_val_h[uid]
                                else:   # heating not clipped
                                    h_load = (y - c_load / cop_c) * effi_h
                                est_h_load[uid][time] = h_load
                                est_c_load[uid][time] = c_load
                                c_hasest[uid][time], h_hasest[uid][time] = -1, -1
                            elif abs(a_clip_h > now_action_h) > 0.01 and a_clip_h < 0:  # h clipped but c not clipped
                                h_load = abs(a_clip_h * true_val_h[uid])
                                c_load = (y - h_load / effi_h) * cop_c
                                c_hasest[uid][time], h_hasest[uid][time] = -1, -1
                                est_h_load[uid][time] = h_load
                                est_c_load[uid][time] = c_load
                    else:
                        prev_t_cop = self.cop_cal(prev_t_out)
                        now_t_cop = self.cop_cal(now_t_out)
                        next_t_cop = self.cop_cal(next_t_out)
                        if prev_t_cop != now_t_cop or prev_t_cop != next_t_cop or now_t_cop != next_t_cop:
                            reg_x = []
                            reg_y = []
                            reg_x.append([1/prev_t_cop, 1/0.9])
                            reg_y.append([y])
                            reg_x.append([1/now_t_cop, 1/0.9])
                            reg_y.append([y])
                            reg_x.append([1/next_t_cop, 1/0.9])
                            reg_y.append([y])
                            c_hasest[uid][time], h_hasest[uid][time] = 1, 1
                            if c_hasest[uid][max(time - 1, 0)] == -1 or c_hasest[uid][min(time + 1, 23)] == -1:
                                # t-1 or t+1 has clipped est (both h and c since they couple)
                                if c_hasest[uid][max(time - 1, 0)] == -1:
                                    reg_x.append([1, 0])
                                    reg_y.append([est_c_load[uid][max(time - 1, 0)]])
                                    reg_x.append([0, 1])
                                    reg_y.append([est_h_load[uid][max(time - 1, 0)]])
                                if c_hasest[uid][min(time + 1, 23)] == -1:
                                    reg_x.append([1, 0])
                                    reg_y.append([est_c_load[uid][min(time + 1, 23)]])
                                    reg_x.append([0, 1])
                                    reg_y.append([est_h_load[uid][min(time + 1, 23)]])
                            self.regr.fit(reg_x, reg_y)
                            [[c_load, h_load]] = self.regr.coef_
                            ## get results of slope in regr model
                        else:   # COP remaining the same (zero)
                            h_load = self.avg_h_load[uid][time]
                            c_load = self.avg_c_load[uid][time]
                            c_hasest[uid][time], h_hasest[uid][time] = 2, 2
                    # save load est to buffer
                        est_h_load[uid][time] = np.round(h_load, 2)
                        est_c_load[uid][time] = np.round(c_load, 2)
                    if c_hasest[uid][time] not in [0, 2]:   # meaning that avg can be updated
                        if self.timestep >= 1:
                            self.avg_h_load[uid][time] = self.avg_h_load[uid][time] * 0.8 + h_load * 0.2
                            self.avg_c_load[uid][time] = self.avg_c_load[uid][time] * 0.8 + c_load * 0.2
                        else:
                            self.avg_h_load[uid][time] = h_load
                            self.avg_c_load[uid][time] = c_load

                repeat_times += 1 if time == 23 else 0
                time = (time+1) % 24
                jump_out = True
                for i in range(24):
                    if c_hasest[uid][i] == 0 or h_hasest[uid][i] == 0:
                        jump_out = False
                if jump_out is True:
                    self.timestep += 1
                    break
        return est_h_load, est_c_load

            # jumping out criteria: every hour has loads est


for algorithm in ['RBC']:
    for climate in [5]:
        climate_zone = climate
        TOTAL_TIME_STEP = 8760  # 8760
        CF_C = 0.006
        CF_H = 0.008
        CF_B = 0
        pred = Predictor()
        params = {'data_path': Path("D:/Reinforcement Learning/CityLearn-master/CityLearn-master/data/Climate_Zone_" + str(climate_zone)),
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

        agents = RBC(actions_spaces)
        print("action space: ", actions_spaces)
        state = env.reset()  # hour 0
        done = False
        # test for RBC
        action = agents.select_action(state)    # action for hour 0
        pred.record_dic(state, action)
        # print(action)
        while not done:
            # prev_hour_est_b = False if prev_hour_est_b is None else prev_hour_est_b
            next_state, reward, done, _ = env.step(action)  # execution of hour 0
            action_next = agents.select_action(next_state)
            state = next_state
            action = action_next
            pred.record_dic(state, action)
            if env.time_step % 24 == 0 and env.time_step >= 24*7-1:
                est_h, est_c = pred.infer_load()
                if env.time_step >= 24*30 + 24*7:
                    print("day: ", env.time_step)
                    print("estimation of cooling: ", est_c)
                    print("estimation of heating: ", est_h)


        # plot_cap_b(cap_b_all, climate, type="elec")
        # plot_cap_h(cap_c_all, climate, type="cooling")
        # plot_cap_h(cap_h_all, climate, type="heat")