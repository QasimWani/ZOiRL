from utils import ReplayBuffer

from citylearn import CityLearn

import numpy as np
import pandas as pd

from scipy import stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


class DataLoader:
    def __init__(self, action_space: list) -> None:
        self.action_space = action_space

    def upload_data(self, replay_buffer: ReplayBuffer) -> None:
        """Upload to memory"""
        raise NotImplementedError

    def load_data(self):
        """Optional: not directly called. Should be called within `upload_data` if used."""
        raise NotImplementedError

    def parse_data(self, data: dict, current_data: dict) -> list:
        """Parses `current_data` for optimization and loads into `data`"""
        TOTAL_PARAMS = 23
        assert (
            len(current_data)
            == TOTAL_PARAMS  # actions + rewards + E_grid_collect. Section 1.3.1
        ), f"Invalid number of parameters, found: {len(current_data)}, expected: {TOTAL_PARAMS}. Can't run Oracle agent optimization."

        for key, value in current_data.items():
            if key not in data:
                data[key] = []
            data[key].append(value)

        return data

    def convert_to_numpy(self, params: dict):
        """Converts dic[key] to nd.array"""
        for key in params:
            if key == "c_bat_init" or key == "c_Csto_init" or key == "c_Hsto_init":
                params[key] = np.array(params[key][0])
            else:
                params[key] = np.array(params[key])

    def get_dimensions(self, data: dict):
        """Prints shape of each param"""
        for key in data.keys():
            print(data[key].shape)

    def get_building(self, data: dict, building_id: int):
        """Loads data (dict) from a particular building. 1-based indexing for building"""
        assert building_id > 0, "building_id is 1-based indexing."
        building_data = {}
        for key in data.keys():
            building_data[key] = np.array(data[key])[:, building_id - 1]
        return building_data

    def create_random_data(self, data: dict):
        """Synthetic data (Gaussian) generation"""
        for key in data:
            data[key] = np.clip(np.random.random(size=data[key].shape), 0, 1)
        return data


class Predictor(DataLoader):
    def __init__(self, action_space: list) -> None:
        super().__init__(action_space)
        self.building_ids = range(len(self.action_space))  # number of buildings
        self.state_buffer = ReplayBuffer(buffer_size=365, batch_size=32)
        self.action_buffer = ReplayBuffer(buffer_size=365, batch_size=32)

        # define constants
        self.true_val_h = [
            10.68,
            49.35,
            1e-05,
            1e-05,
            60.12,
            105.12,
            85.44,
            111.96,
            102.24,
        ]
        self.true_val_b = [140, 80, 50, 75, 50, 30, 40, 30, 35]
        self.true_val_c = [
            618.12,
            227.37,
            414.68,
            383.565,
            244.685,
            96.87,
            127.82,
            165.45,
            175.23,
        ]
        self.CF_C = 0.006
        self.CF_H = 0.008
        self.CF_B = 0

        # define regression model
        self.regr = LinearRegression(
            fit_intercept=False
        )  # , positive=True) # version error
        self.avg_h_load = {uid: np.zeros(24) for uid in self.building_ids}
        self.avg_c_load = {uid: np.ones(24) for uid in self.building_ids}
        self.timestep = 0

    def upload_data(
        self, replay_buffer: ReplayBuffer, state, action, reward, next_state, done
    ):
        """Main function to append eod data to main replaybuffer passed into optimization model"""
        self.record_dic(state, action, reward, next_state)

        if self.timestep % 24 == 23:  # new day - add to buffer
            data = self.parse_data(
                replay_buffer.get_recent(), self.get_day_data(replay_buffer)
            )
            replay_buffer.add(data)
            replay_buffer.total_it += 24

    def parse_data(self, data: dict, current_data: dict):  # override in child class
        """Parses `current_data` for optimization and loads into `data`. Everything is of shape 24, 9"""
        TOTAL_PARAMS = 23
        assert (
            len(current_data)
            == TOTAL_PARAMS  # actions + rewards + E_grid_collect. Section 1.3.1
        ), f"Invalid number of parameters, found: {len(current_data)}, expected: {TOTAL_PARAMS}. Can't run Oracle agent optimization."

        for key, value in current_data.items():
            if key not in data:
                data[key] = []
            if len(value.shape) == 1:
                value = np.repeat(value, 24).reshape(24, len(self.building_ids))
            data[key].append(value)

        return data

    def get_day_data(self, replay_buffer: ReplayBuffer):
        """Helper method for uploading data to memory"""
        # get current day's buffer
        data = replay_buffer[-1] if len(replay_buffer) > 0 else None

        # get heating and cooling estimate
        heating_estimate, cooling_estimate = self.infer_load()

        E_ns = np.array(
            self.state_buffer.get(-1)["elec_cons"]
        )  # net electricity consumption -2 to -1--done
        H_bd = np.array([heating_estimate[key] for key in self.building_ids]).T
        C_bd = np.array([cooling_estimate[key] for key in self.building_ids]).T

        H_max = None if "H_max" not in data else data["H_max"]  # load previous H_max
        if H_max is None:
            H_max = np.max(H_bd, axis=0)
        else:
            H_max = np.max([H_max, H_bd.max(axis=0)], axis=0)  # global max

        C_max = None if "C_max" not in data else data["C_max"]  # load previous C_max
        if C_max is None:
            C_max = np.max(C_bd, axis=0)
        else:
            H_max = np.max([C_max, C_bd.max(axis=0)], axis=0)  # global max

        E_pv = np.array(self.state_buffer.get(-1)["solar_gen"])  # solar energy -2 to -1--done

        temp = np.array(self.state_buffer.get(-1)["t_out"])  # 24, 9 intermediate value
        COP_C = np.zeros((24, len(self.building_ids)))
        for hour in range(24):
            for bid in self.building_ids:
                COP_C[hour, bid] = self.cop_cal(temp[hour, bid])

        E_hpC_max = np.max(C_bd / COP_C, axis=0)
        E_ehH_max = H_max / 0.9
        C_p_bat = np.full((24, len(self.building_ids)), fill_value=60)

        c_bat_init = (
            np.array(self.state_buffer.get(-1)["soc_b"]) / C_p_bat
        )  # -2 to -1 (confirm)--done
        c_bat_init[c_bat_init == np.inf] = 0

        C_p_Hsto = 3 * H_max

        c_Hsto_init = (
            np.array(self.state_buffer.get(-1)["soc_h"]) / C_p_Hsto
        )  # -2 to -1 (confirm)--done
        c_Hsto_init[c_Hsto_init == np.inf] = 0
        C_p_Csto = 2 * C_max

        c_Csto_init = (
            np.array(self.state_buffer.get(-1)["soc_c"]) / C_p_Csto
        )  # -2 to -1 (confirm)--done
        c_Csto_init[c_Csto_init == np.inf] = 0

        # add E-grid (part of E-grid_collect)
        observation_data["E_grid"] = np.array(self.state_buffer.get(-1)["elec_cons"])
        observation_data["E_grid_prevhour"] = np.zeros((24, len(self.building_ids)))
        for hour in range(24):
            for bid in self.building_ids:
                observation_data["E_grid_prevhour"][hour, bid] = (
                    np.array(self.state_buffer.get(-2)["elec_cons"])[-1, bid]
                    if hour == 0
                    else observation_data["E_grid"][hour - 1, bid]
                )

        observation_data["E_ns"] = E_ns
        observation_data["H_bd"] = H_bd
        observation_data["C_bd"] = C_bd
        observation_data["H_max"] = H_max
        observation_data["C_max"] = C_max

        observation_data["E_pv"] = E_pv

        observation_data["E_hpC_max"] = E_hpC_max
        observation_data["E_ehH_max"] = E_ehH_max
        observation_data["COP_C"] = COP_C

        observation_data["C_p_bat"] = C_p_bat
        observation_data["c_bat_init"] = c_bat_init

        observation_data["C_p_Hsto"] = C_p_Hsto
        observation_data["c_Hsto_init"] = c_Hsto_init

        observation_data["C_p_Csto"] = C_p_Csto
        observation_data["c_Csto_init"] = c_Csto_init

        observation_data["action_H"] = self.action_buffer.get(-1)["action_H"]
        observation_data["action_C"] = self.action_buffer.get(-1)["action_C"]
        observation_data["action_bat"] = self.action_buffer.get(-1)["action_bat"]

        # add reward \in R^9 (scalar value for each building)
        observation_data["reward"] = self.action_buffer.get(-1)["reward"]

        return observation_data

    def record_dic(
        self, current_state: list, current_action: list, current_reward: list, next_state: list
    ):
        """
        call record_dic at the beginning of each step
        """
        state = self.state_buffer.get_recent()
        now_state = self.state_to_dic(current_state, next_state)
        parse_state = self.parse_data(state, now_state)
        self.state_buffer.add(parse_state)

        current_action = self.action_reward_to_dic(current_action, current_reward)
        action = self.action_buffer.get_recent()
        parse_action = self.parse_data(action, current_action)
        self.action_buffer.add(parse_action)



    def state_to_dic(self, state_list: list, next_state_list: list):
        state_bdg = {}
        next_state_bdg = {}
        for uid in self.building_ids:
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
                "carbon_intensity": state[29],
            }
            state_bdg[uid] = s
        for uid in self.building_ids:
            state = next_state_list[uid]
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
                "carbon_intensity": state[29],
            }
            next_state_bdg[uid] = s

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
        next_daytype = [next_state_bdg[i]["day"] for i in self.building_ids]
        next_hour = [next_state_bdg[i]["hour"] for i in self.building_ids]
        next_t_out = [next_state_bdg[i]["t_out"] for i in self.building_ids]
        next_rh_out = [next_state_bdg[i]["rh_out"] for i in self.building_ids]
        next_t_in = [next_state_bdg[i]["t_in"] for i in self.building_ids]
        next_rh_in = [next_state_bdg[i]["rh_in"] for i in self.building_ids]
        next_elec_dem = [next_state_bdg[i]["non_shiftable_load"] for i in self.building_ids]
        next_solar_gen = [next_state_bdg[i]["solar_gen"] for i in self.building_ids]
        next_soc_c = [next_state_bdg[i]["cooling_storage_soc"] for i in self.building_ids]
        next_soc_h = [next_state_bdg[i]["dhw_storage_soc"] for i in self.building_ids]
        next_soc_b = [next_state_bdg[i]["electrical_storage_soc"] for i in self.building_ids]
        next_elec_cons = [
            next_state_bdg[i]["net_electricity_consumption"] for i in self.building_ids
        ]

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

        s_dic["next_daytype"] = next_daytype
        s_dic["next_hour"] = next_hour
        s_dic["next_t_out"] = next_t_out
        s_dic["next_rh_out"] = next_rh_out
        s_dic["next_t_in"] = next_t_in
        s_dic["next_rh_in"] = next_rh_in
        s_dic["next_elec_dem"] = next_elec_dem
        s_dic["next_solar_gen"] = next_solar_gen
        s_dic["next_soc_c"] = next_soc_c
        s_dic["next_soc_h"] = next_soc_h
        s_dic["next_soc_b"] = next_soc_b
        s_dic["next_elec_cons"] = next_elec_cons

        return s_dic

    def action_reward_to_dic(self, action, reward):
        s_dic = {}
        a_c = [action[i][0] for i in self.building_ids]
        a_h = [action[i][1] for i in self.building_ids]
        a_b = [action[i][2] for i in self.building_ids]

        s_dic["action_C"] = a_c
        s_dic["action_H"] = a_h
        s_dic["action_bat"] = a_b
        s_dic["reward"] = reward

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
        c_hasest = {
            uid: np.zeros(24) for uid in self.building_ids
        }  # -1:clipped, 0:non-est, 1:regression, 2: moving avg
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
                    now_state = self.state_buffer.get(-2)  # this is previous, not now!
                    now_c_soc = now_state["soc_c"][time][uid]
                    now_h_soc = now_state["soc_h"][time][uid]
                    now_b_soc = now_state["soc_b"][time][uid]
                    now_t_out = now_state["t_out"][time][uid]
                    now_solar = now_state["solar_gen"][time][uid]
                    now_elec_dem = now_state["elec_dem"][time][uid]
                    cop_c = self.cop_cal(now_t_out)  # cop at t
                    now_action = self.action_buffer.get(
                        -2
                    )  # this is previous, not now!
                    now_action_c = now_action["action_C"][time][uid]
                    now_action_h = now_action["action_H"][time][uid]
                    now_action_b = now_action["action_bat"][time][uid]
                    prev_state = (
                        now_state if time != 0 else self.state_buffer.get(-3)
                    )  # this is 2 days ago, not previous
                    prev_t_out = prev_state["t_out"][time - 1][
                        uid
                    ]  # when time=0, time-1=-1
                    if time != 23:
                        next_state = now_state
                        next_c_soc = next_state["soc_c"][time + 1][uid]
                        next_h_soc = next_state["soc_h"][time + 1][uid]
                        next_b_soc = next_state["soc_b"][time + 1][uid]
                        next_t_out = next_state["t_out"][time + 1][uid]
                        next_elec_con = next_state["elec_cons"][time + 1][uid]
                        y = (
                            now_solar
                            + next_elec_con
                            - now_elec_dem
                            - (true_val_c[uid] / cop_c)
                            * (next_c_soc - (1 - self.CF_C) * now_c_soc)
                            * 0.9
                            - (true_val_h[uid] / effi_h)
                            * (next_h_soc - (1 - self.CF_H) * now_h_soc)
                            - (next_b_soc - (1 - self.CF_B) * now_b_soc)
                            * true_val_b[uid]
                            / 0.9
                        )
                    else:
                        next_state = self.state_buffer.get_recent()
                        next_c_soc = next_state["soc_c"][0][uid]
                        next_h_soc = next_state["soc_h"][0][uid]
                        next_b_soc = next_state["soc_b"][0][uid]
                        next_t_out = next_state["t_out"][0][uid]
                        next_elec_con = next_state["elec_cons"][0][uid]
                        y = (
                            now_solar
                            + next_elec_con
                            - now_elec_dem
                            - (true_val_c[uid] / cop_c)
                            * (next_c_soc - (1 - self.CF_C) * now_c_soc)
                            * 0.9
                            - (true_val_h[uid] / effi_h)
                            * (next_h_soc - (1 - self.CF_H) * now_h_soc)
                            - (next_b_soc - (1 - self.CF_B) * now_b_soc)
                            * true_val_b[uid]
                            / 0.9
                        )

                    a_clip_c = next_c_soc - (1 - self.CF_C) * now_c_soc
                    a_clip_h = next_h_soc - (1 - self.CF_H) * now_h_soc

                    if (
                        repeat_times == 0
                    ):  # can we calculate direct when now_action > 0?
                        if uid in [2, 3]:
                            c_load = max(0, y * cop_c)
                            h_load = 0
                            est_h_load[uid][time] = h_load
                            est_c_load[uid][time] = c_load
                            c_hasest[uid][time], h_hasest[uid][time] = -1, -1
                        else:
                            if (
                                abs(a_clip_c - now_action_c) > 0.001
                                and now_action_c < 0
                            ):  # cooling get clipped
                                c_load = abs(a_clip_c * true_val_c[uid])
                                if (
                                    abs(a_clip_h - now_action_h) > 0.001
                                    and now_action_h < 0
                                ):  # heating get clipped
                                    h_load = a_clip_h * true_val_h[uid]
                                else:  # heating not clipped
                                    h_load = (y - c_load / cop_c) * effi_h
                                est_h_load[uid][time] = h_load
                                est_c_load[uid][time] = c_load
                                c_hasest[uid][time], h_hasest[uid][time] = -1, -1
                            elif (
                                abs(a_clip_h > now_action_h) > 0.01 and a_clip_h < 0
                            ):  # h clipped but c not clipped
                                h_load = abs(a_clip_h * true_val_h[uid])
                                c_load = (y - h_load / effi_h) * cop_c
                                c_hasest[uid][time], h_hasest[uid][time] = -1, -1
                                est_h_load[uid][time] = h_load
                                est_c_load[uid][time] = c_load
                    else:
                        prev_t_cop = self.cop_cal(prev_t_out)
                        now_t_cop = self.cop_cal(now_t_out)
                        next_t_cop = self.cop_cal(next_t_out)
                        if (
                            prev_t_cop != now_t_cop
                            or prev_t_cop != next_t_cop
                            or now_t_cop != next_t_cop
                        ):
                            reg_x = []
                            reg_y = []
                            reg_x.append([1 / prev_t_cop, 1 / 0.9])
                            reg_y.append([y])
                            reg_x.append([1 / now_t_cop, 1 / 0.9])
                            reg_y.append([y])
                            reg_x.append([1 / next_t_cop, 1 / 0.9])
                            reg_y.append([y])
                            c_hasest[uid][time], h_hasest[uid][time] = 1, 1
                            if (
                                c_hasest[uid][max(time - 1, 0)] == -1
                                or c_hasest[uid][min(time + 1, 23)] == -1
                            ):
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
                            c_load = max(0, (h_load * 0.8 - 5) * 0.6 * cop_c)
                            # c_load = max(c_load, self.avg_c_load[uid][time])
                            ## get results of slope in regr model
                        else:  # COP remaining the same (zero)
                            h_load = self.avg_h_load[uid][time]
                            c_load = self.avg_c_load[uid][time]
                            c_hasest[uid][time], h_hasest[uid][time] = 2, 2
                        # save load est to buffer
                        est_h_load[uid][time] = np.round(h_load, 2)
                        est_c_load[uid][time] = np.round(c_load, 2)
                    if c_hasest[uid][time] not in [
                        0,
                        2,
                    ]:  # meaning that avg can be updated
                        if self.timestep >= 1:
                            self.avg_h_load[uid][time] = (
                                self.avg_h_load[uid][time] * 0.8 + h_load * 0.2
                            )
                            self.avg_c_load[uid][time] = (
                                self.avg_c_load[uid][time] * 0.8 + c_load * 0.2
                            )
                        else:
                            self.avg_h_load[uid][time] = h_load
                            self.avg_c_load[uid][time] = c_load

                repeat_times += 1 if time == 23 else 0
                time = (time + 1) % 24
                jump_out = True
                for i in range(24):
                    if c_hasest[uid][i] == 0 or h_hasest[uid][i] == 0:
                        jump_out = False
                if jump_out is True:
                    self.timestep += 1
                    break
        return est_h_load, est_c_load


# class DataLoader:
#     """Main Class for loading and uplaoding data to buffer."""

#     def __init__(
#         self,
#         is_oracle: bool,
#         action_space: list,
#         env: CityLearn = None,
#     ) -> None:

#         self.env = env
#         if is_oracle:
#             self.model = Oracle(env, action_space)
#         else:
#             self.model = Predictor(...)

#         self.data = {} if is_oracle else None

#     def upload_data(
#         self,
#         replay_buffer: ReplayBuffer,
#         E_grid: list,
#         action: list,
#         reward: list,
#         env: CityLearn = None,
#         t_idx: int = -1,  # timestep (hour) of the simulation [0 - (4years-1)]
#     ):
#         """Upload to memory"""

#         self.model.upload_data(replay_buffer, E_grid, action, reward, env, t_idx)

#     def load_data(self):
#         """Sample from Memory. NOTE: Optional"""
#         self.model.load_data()


# class Predictor:
#     """@Zhiyao + @Mingyu - estimates parameters, loads data supplied to `Actor`"""

#     def __init__(self):
#         raise NotImplementedError("Predictor class not implemented")

#     def upload_data(self, replay_buffer: ReplayBuffer):
#         raise NotImplementedError("Functionality not implemented")

#     def load_data(self, state, t):
#         raise NotImplementedError("Functionality not implemented")

#     # TODO: implement other methods here. Make sure `DataLoader.upload_data()` and `DataLoader.load_data()` are processed correctly
#     def parse_data(self, data: dict, current_data: dict) -> list:
#         """Parses `current_data` for optimization and loads into `data`"""
#         assert (
#             len(current_data) == 30  # includes actions + rewards + E_grid_collect
#         ), "Invalid number of parameters. Can't run basic (root) agent optimization"

#         for key, value in current_data.items():
#             if key not in data:
#                 data[key] = []  # [] x, 9 1, 9 -> x + 1, 9
#             data[key].append(value)

#         return data

#     def convert_to_numpy(self, params: dict):
#         """Converts dic[key] to nd.array"""
#         for key in params:
#             if key == "c_bat_init" or key == "c_Csto_init" or key == "c_Hsto_init":
#                 params[key] = np.array(params[key][0])
#             else:
#                 params[key] = np.array(params[key])


class Oracle:
    """Agent with access to true environment data."""

    def __init__(self, env: CityLearn, action_space: list) -> None:
        self.action_space = action_space
        self.weather_data = self.get_weather_data(env)

    def upload_data(
        self,
        replay_buffer: ReplayBuffer,
        E_grid: list,
        actions: list,
        rewards: list,
        env: CityLearn = None,
        t_idx: int = -1,
    ):
        """Returns state information to be later added to replay buffer"""
        assert (
            env is not None and t_idx >= 0
        ), "Invalid argument passed. Missing env object and/or invalid time index passed"

        ## load current data and pass it as an argument to parse_data where data needs to be a dictionary.
        if t_idx == 2:
            E_grid_memory = [0] * len(self.action_space)
        else:
            # last hour or eod E_grid values.
            E_grid_memory = replay_buffer.get(-1)["E_grid"][-1]

        data = self.parse_data(
            replay_buffer.get_recent(),
            self.get_current_data_oracle(
                env, t_idx, E_grid, E_grid_memory, actions, rewards
            ),
        )
        replay_buffer.add(data)

    def load_data(self, state, t):
        raise NotImplementedError("Functionality not implemented")

    def get_weather_data(self, env):
        """load weather data for calculation of COP"""

        with open(env.data_path / env.weather_file) as csv_file:
            weather_data = pd.read_csv(csv_file)
        weather_data = weather_data["Outdoor Drybulb Temperature [C]"]
        return weather_data

    def parse_data(self, data: dict, current_data: dict) -> list:
        """Parses `current_data` for optimization and loads into `data`"""
        assert (
            len(current_data) == 23  # actions + rewards + E_grid_collect. Section 1.3.1
        ), f"Invalid number of parameters, found: {len(current_data)}, expected: 23. Can't run Oracle agent optimization."

        for key, value in current_data.items():
            if key not in data:
                data[key] = []
            data[key].append(value)

        return data

    def convert_to_numpy(self, params: dict):
        """Converts dic[key] to nd.array"""
        for key in params:
            if key == "c_bat_init" or key == "c_Csto_init" or key == "c_Hsto_init":
                params[key] = np.array(params[key][0])
            else:
                params[key] = np.array(params[key])

    def get_dimensions(self, data: dict):
        """Prints shape of each param"""
        for key in data.keys():
            print(data[key].shape)

    def get_building(self, data: dict, building_id: int):
        """Loads data (dict) from a particular building. 1-based indexing for building"""
        assert building_id > 0, "building_id is 1-based indexing."
        building_data = {}
        for key in data.keys():
            building_data[key] = np.array(data[key])[:, building_id - 1]
        return building_data

    def create_random_data(self, data: dict):
        """Synthetic data (Gaussian) generation"""
        for key in data:
            data[key] = np.clip(np.random.random(size=data[key].shape), 0, 1)
        return data

    def get_current_data_oracle(
        self,
        env: CityLearn,
        t: int,  # t goes from 0 - end of simulation (not 24 hour counter!)
        E_grid: list,
        E_grid_memory: list,
        actions: list = None,
        rewards: list = None,
    ):
        """
        Returns data (dict) for each building from `env` for `t` timestep
        @Params:
        1. env: CityLearn environment.
        2. t: current hour from TD3.py (agents.total_it). Goes from (0 - 4 * 365 * 24]
        3. E_grid: Current net electricity consumption, obtained from environment.
        4. E_grid_memory: Previous hours' net electricity consumption. Obtained from memory (replaybuffer).
        5. actions: set of actions for current hour. Appends to replaybuffer.
        6. rewards: per building reward. Appends to replaybuffer.
        """
        ### FB - Full batch. Trim output X[:time-step]
        ### CT - current timestep only. X = full_data[time-step], no access to full_data
        ### DP - dynamic update. time-step k = [... k], time-step k+n = [... k + n].
        ### P - constant value across all time steps. changes per building only.

        _num_buildings = len(self.action_space)  # total number of buildings in env.
        observation_data = {}

        # Loads
        E_ns = [
            env.buildings["Building_" + str(i)].sim_results["non_shiftable_load"][t]
            for i in range(1, _num_buildings + 1)
        ]  # CT
        H_bd = [
            env.buildings["Building_" + str(i)].sim_results["dhw_demand"][t]
            for i in range(1, _num_buildings + 1)
        ]  # DP
        C_bd = [
            env.buildings["Building_" + str(i)].sim_results["cooling_demand"][t]
            for i in range(1, _num_buildings + 1)
        ]  # DP
        H_max = np.max(
            [
                env.buildings["Building_" + str(i)].sim_results["dhw_demand"]
                for i in range(1, _num_buildings + 1)
            ],
            axis=1,
        )  # DP
        C_max = np.max(
            [
                env.buildings["Building_" + str(i)].sim_results["cooling_demand"]
                for i in range(1, _num_buildings + 1)
            ],
            axis=1,
        )  # DP

        # PV generations
        E_pv = [
            env.buildings["Building_" + str(i)].sim_results["solar_gen"][t]
            for i in range(1, _num_buildings + 1)
        ]  # CT

        # Heat Pump
        eta_hp = [0.22] * _num_buildings  # P
        t_C_hp = [
            8
        ] * _num_buildings  # P target cooling temperature (universal constant)

        COP_C = [None for i in range(_num_buildings)]  # DP

        E_hpC_max = [None] * _num_buildings
        for i in range(1, _num_buildings + 1):
            COP_C_t = (
                eta_hp[i - 1]
                * float(t_C_hp[i - 1] + 273.15)
                / (self.weather_data - t_C_hp[i - 1])
            )
            COP_C_t[COP_C_t < 0] = 20.0
            COP_C_t[COP_C_t > 20] = 20.0
            COP_C_t = COP_C_t.to_numpy()
            COP_C[i - 1] = COP_C_t[t]
            E_hpC_max[i - 1] = np.max(
                env.buildings["Building_" + str(i)].sim_results["cooling_demand"]
                / COP_C_t
            )

        # Electric Heater
        # replaced capacity (not avaiable in electric heater) w/ nominal_power
        E_ehH_max = [H_max[i] / 0.9 for i in range(_num_buildings)]  # P

        # Battery
        C_f_bat = [0.00001 for i in range(_num_buildings)]  # P
        C_p_bat = [60] * _num_buildings  # P (range: [20, 200])
        # current hour soc. normalized
        c_bat_init = [
            None
        ] * _num_buildings  # can't get future data since action dependent
        for i in range(1, _num_buildings + 1):
            building = env.buildings["Building_" + str(i)].electrical_storage
            try:
                c_bat_init[i - 1] = building.soc[t - 1] / building.capacity
            except:
                c_bat_init[i - 1] = 0
        # Heat (Energy->dhw) Storage
        C_f_Hsto = [0.008] * _num_buildings  # P
        C_p_Hsto = [3 * H_max[i] for i in range(_num_buildings)]  # P
        # current hour soc. normalized
        c_Hsto_init = [
            None
        ] * _num_buildings  # can't get future data since action dependent
        for i in range(1, _num_buildings + 1):
            building = env.buildings["Building_" + str(i)].dhw_storage
            try:
                c_Hsto_init[i - 1] = building.soc[t - 1] / building.capacity
            except:
                c_Hsto_init[i - 1] = 0
            # Cooling (Energy->cooling) Storage
        C_f_Csto = [0.006] * _num_buildings  # P
        C_p_Csto = [2 * C_max[i] for i in range(_num_buildings)]  # P

        # current hour soc. normalized
        c_Csto_init = [
            None
        ] * _num_buildings  # can't get future data since action dependent
        for i in range(1, _num_buildings + 1):
            building = env.buildings["Building_" + str(i)].cooling_storage
            try:
                c_Csto_init[i - 1] = building.soc[t - 1] / building.capacity
            except:
                c_Csto_init[i - 1] = 0

        # add actions - size 9 for each action
        action_H, action_C, action_bat = (
            [None] * 3 if actions is None else zip(*actions)
        )

        # fill data
        # add E-grid (part of E-grid_collect)
        observation_data["E_grid"] = (
            E_grid if E_grid is not None else [0] * _num_buildings
        )
        observation_data["E_grid_prevhour"] = E_grid_memory

        observation_data["E_ns"] = E_ns
        observation_data["H_bd"] = H_bd
        observation_data["C_bd"] = C_bd
        observation_data["H_max"] = H_max
        observation_data["C_max"] = C_max

        observation_data["E_pv"] = E_pv

        observation_data["E_hpC_max"] = E_hpC_max
        observation_data["E_ehH_max"] = E_ehH_max
        # observation_data["eta_hp"] = eta_hp # NOT NEEEDED!
        # observation_data["t_C_hp"] = t_C_hp # NOT NEEDED!
        observation_data["COP_C"] = COP_C

        # observation_data["C_f_bat"] = C_f_bat
        observation_data["C_p_bat"] = C_p_bat
        observation_data["c_bat_init"] = c_bat_init

        # observation_data["C_f_Hsto"] = C_f_Hsto
        observation_data["C_p_Hsto"] = C_p_Hsto
        observation_data["c_Hsto_init"] = c_Hsto_init

        # observation_data["C_f_Csto"] = C_f_Csto
        observation_data["C_p_Csto"] = C_p_Csto
        observation_data["c_Csto_init"] = c_Csto_init

        observation_data["action_H"] = action_H
        observation_data["action_C"] = action_C
        observation_data["action_bat"] = action_bat

        # add reward \in R^9 (scalar value for each building)
        observation_data["reward"] = rewards

        return observation_data

    def estimate_data(
        self,
        surrogate_env: CityLearn,
        data: dict,
        t_start: int,
        init_updates: dict,
        replay_buffer: ReplayBuffer,
        t_end: int = 24,
    ):
        """Returns data for hours `t_start` - 24 using `surrogate_env` running RBC `agent`"""
        for i in range(t_start % 24, t_end):
            data = self.parse_data(
                data,
                self.get_current_data_oracle(
                    surrogate_env,
                    t_start + i,
                    E_grid=None,
                    E_grid_memory=np.array(
                        [0] * len(self.action_space)
                    ),  # replay_buffer.get(-2)["E_grid"][(t_start + i) % 24],  # -2 : previous day E_grid values
                ),
            )

        return (
            self.init_values(data, init_updates)[0] if t_start % 24 == 0 else data
        )  # only load previous values at start of day

    def init_values(self, data: dict, update_values: dict = None):
        """Loads eod values for SOC and E_grid_past before(after) wiping data cache"""
        if update_values:
            # assign previous day's end socs.
            data["c_bat_init"][0] = update_values["c_bat_init"]
            data["c_Hsto_init"][0] = update_values["c_Hsto_init"]
            data["c_Csto_init"][0] = update_values["c_Csto_init"]

            # assign previous day's end E_grid.
            # data["E_grid_true"][0] = update_values["E_grid_true"]
        else:
            update_values = {
                "c_bat_init": data["c_bat_init"][-1],
                "c_Hsto_init": data["c_Hsto_init"][-1],
                "c_Csto_init": data["c_Csto_init"][-1],
                # "E_grid_true": data["E_grid_true"][-1],
            }

        return data, update_values
