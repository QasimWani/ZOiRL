from collections import defaultdict
from copy import Error
from utils import ReplayBuffer

from citylearn import CityLearn, building_loader

import numpy as np
import pandas as pd
import sys

from scipy import stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


class DataLoader:
    """Base Class"""

    def __init__(self, action_space: list) -> None:
        self.action_space = action_space

    def upload_data(self) -> None:
        """Upload to memory"""
        raise NotImplementedError

    def load_data(self):
        """Optional: not directly called. Should be called within `upload_data` if used."""
        raise NotImplementedError

    def parse_data(self, data: dict, current_data: dict):
        """Parses `current_data` for optimization and loads into `data`"""
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
            print(key, data[key].shape)

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


# TODO: @Zhiyao - add in parameter/initialization for capacity as discussed.
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
        self.daystep = 0  # this is for h/c loads estimation--not real daystep!
        self.h_peak = {uid: np.zeros(24, dtype=int) for uid in self.building_ids}
        self.c_peak = {uid: np.zeros(24, dtype=int) for uid in self.building_ids}

        self.gamma = 0.2
        self.avg_type = [
            "solar_gen",
            "elec_weekday",
            "elec_weekend1",
            "elec_weekend2",
        ]  # weekend1: Sat, weekend2: Sun and holidays
        self.solar_avg = {i: np.zeros(24) for i in self.building_ids}
        self.elec_weekday_avg = {i: np.zeros(24) for i in self.building_ids}
        self.elec_weekend1_avg = {i: np.zeros(24) for i in self.building_ids}
        self.elec_weekend2_avg = {i: np.zeros(24) for i in self.building_ids}
        self.regr_solar = {uid: LinearRegression() for uid in self.building_ids}
        self.regr_elec = {uid: LinearRegression() for uid in self.building_ids}

    # TODO: @Zhiyao + @Qasim - this function has not tested. Depends on internal predictor to work.
    def estimate_data(
        self, replay_buffer: ReplayBuffer, timestep: int, is_adaptive: bool
    ):
        """Estimates data to be passed into Optimization model for 24hours into future."""
        if is_adaptive:
            # if hour start of day, `get_recent()` will automatically return an empty dictionary
            data = self.full_parse_data(
                self.get_day_data(replay_buffer, timestep), 24 - timestep % 24
            )
            replay_buffer.add(data)
        else:
            data = self.full_parse_data(self.get_day_data(replay_buffer, timestep))
            replay_buffer.add(data, full_day=True)
        return data

    def full_parse_data(self, current_data: dict, window: int = 24):
        """Parses `current_data` for optimization and loads into `data`. Everything is of shape 24, 9"""
        TOTAL_PARAMS = 20
        assert (
            len(current_data)
            == TOTAL_PARAMS  # actions + rewards + E_grid_collect. Section 1.3.1
        ), (
            f"Invalid number of parameters, found: {len(current_data)}, expected: {TOTAL_PARAMS}. "
            f"Can't run Predictor agent optimization.\n"
            f"@Zhiyao, these parameters come from `get_day_data`. "
            f"Count the number of keys returned in that function and make sure its equal to `current_data` parameter. "
            f"Otherwise, there's a mismatch that the actor wont be able to run. 23 is set on previous version. "
            f"Change this is you believe you'd need more parameters."
        )
        data = {}
        for key, value in current_data.items():
            value = np.array(value)
            if len(value.shape) == 1:
                value = np.repeat(value, window).reshape(window, len(self.building_ids))
            if np.shape(value) == (24, len(self.building_ids)):
                data[key] = value
            else:
                # makes sure horizontal dimensions are 24
                data[key] = np.pad(value, ((24 - window, 0), (0, 0)))

        return data

    # TODO: @Zhiyao - needs fixing -- done
    def calculate_avg(self):
        """calculate hourly avg value of the day"""
        buffer = self.state_buffer
        daytype = {i: [] for i in self.building_ids}
        elec_dem = {i: [] for i in self.building_ids}
        solar_gen = {i: [] for i in self.building_ids}
        elec_weekday = {i: np.zeros([24]) for i in self.building_ids}
        elec_weekend1 = {i: np.zeros([24]) for i in self.building_ids}
        elec_weekend2 = {i: np.zeros([24]) for i in self.building_ids}
        solar_alldays = {i: np.zeros([24]) for i in self.building_ids}
        weekend1, weekend2, weekday = 0, 0, 0

        for i in range(-14, -1):
            for uid in self.building_ids:
                daytype[uid].append(np.array(buffer.get(i)["daytype"])[0, uid])
                elec_dem[uid].append(np.array(buffer.get(i)["elec_dem"])[:, uid])
                solar_gen[uid].append(np.array(buffer.get(i)["solar_gen"])[:, uid])

        for i in range(len(elec_dem[0])):
            for uid in self.building_ids:
                solar_alldays[uid] += solar_gen[uid][i]

            if daytype[0][i] in [7]:
                weekend1 += 1
                for uid in self.building_ids:
                    elec_weekend1[uid] += elec_dem[uid][i]
            elif daytype[0][i] in [1, 8]:
                weekend2 += 1
                for uid in self.building_ids:
                    elec_weekend2[uid] += elec_dem[uid][i]
            else:
                weekday += 1
                for uid in self.building_ids:
                    elec_weekday[uid] += elec_dem[uid][i]

        for uid in self.building_ids:
            self.solar_avg[uid] = solar_alldays[uid] / 13
            self.elec_weekday_avg[uid] = elec_weekday[uid] / weekday
            self.elec_weekend1_avg[uid] = elec_weekend1[uid] / weekend1
            self.elec_weekend2_avg[uid] = elec_weekend1[uid] / weekend2

    # TODO: @Zhiyao - make sure in the case of adaptive this returns (and is sent to actor.py --> see TD3.py (select_action)) data of dimensions (window, 9)
    # >>> Now, in the case of adaptive, say we're on hour 10. So, we only need to make predictions from hour 10 - 24 (1-based indexing).
    # >>> You should return of dimensions (24 - 10, 9) and NOT 24, 9. This is because we've already observed loads in the first 9 hours and nothing can be changed for that.
    # >>> You should make use of observed state & action information in `infer_load` functions. That should make use of observed information in the current day.
    # NOTE: I think this should work fine, but you may need to change `state_buffer.get(-1)`. If this confuses you, and EVERYTHING else is working properly, ping me ASAP
    # >>> and I can fix it accordingly.

    def get_day_data(self, replay_buffer: ReplayBuffer, timestep: int):
        """Helper method for uploading data to memory. This is for estimation only!!!"""
        T = 24
        window = T - timestep % 24

        observation_data = {}

        # get previous day's buffer
        data = replay_buffer.get(-1) if len(replay_buffer) > 0 else None

        # NOTE: @Zhiyao - dimensions should be of `window, num_buildings`
        # get heating, cooling, electricity, and solar estimate
        (
            heating_estimate,
            cooling_estimate,
            solar_estimate,
            electricity_estimate,
            future_temp,
        ) = self.infer_load(timestep)

        E_ns = np.array([electricity_estimate[key] for key in self.building_ids]).T
        E_pv = np.array([solar_estimate[key] for key in self.building_ids]).T

        H_bd = np.array([heating_estimate[key] for key in self.building_ids]).T
        C_bd = np.array([cooling_estimate[key] for key in self.building_ids]).T

        H_max = (
            None if data is None else data["H_max"].max(axis=0)
        )  # load previous H_max
        if H_max is None:
            H_max = np.max(H_bd, axis=0)
        else:
            H_max = np.max([H_max, H_bd.max(axis=0)], axis=0)  # global max

        C_max = (
            None if data is None else data["C_max"].max(axis=0)
        )  # load previous C_max
        if C_max is None:
            C_max = np.max(C_bd, axis=0)
        else:
            H_max = np.max([C_max, C_bd.max(axis=0)], axis=0)  # global max

        temp = np.array([future_temp[uid].flatten() for uid in self.building_ids]).T
        COP_C = np.zeros((window, len(self.building_ids)))
        for hour in range(window):
            for bid in self.building_ids:
                COP_C[hour, bid] = self.cop_cal(temp[hour, bid])

        E_hpC_max = np.max(C_bd / COP_C, axis=0)
        E_ehH_max = H_max / 0.9
        C_p_bat = np.full((len(self.building_ids)), fill_value=60)

        c_bat_init = np.array(
            self.state_buffer.get(-1)["soc_b"][-1]  # current condition
        )  # -2 to -1 (confirm)--done
        c_bat_init[c_bat_init == np.inf] = 0

        C_p_Hsto = 3 * H_max

        c_Hsto_init = np.array(
            self.state_buffer.get(-1)["soc_h"][-1]
        )  # -2 to -1 (confirm)--done
        c_Hsto_init[c_Hsto_init == np.inf] = 0
        C_p_Csto = 2 * C_max

        c_Csto_init = np.array(
            self.state_buffer.get(-1)["soc_c"][-1]
        )  # -2 to -1 (confirm)--done
        c_Csto_init[c_Csto_init == np.inf] = 0

        # add E-grid - default day-ahead
        egc = np.array(self.state_buffer.get(-1)["elec_cons"])
        observation_data["E_grid"] = np.pad(egc, ((0, T - egc.shape[0]), (0, 0)))

        observation_data["E_grid_prevhour"] = np.zeros((T, len(self.building_ids)))
        observation_data["E_grid_prevhour"][0] = np.array(
            self.state_buffer.get(-2)["elec_cons"]
        )[-1]
        for hour in range(1, timestep % 24):
            observation_data["E_grid_prevhour"][hour] = observation_data["E_grid"][
                hour - 1
            ]

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
        # observation_data["reward"] = self.action_buffer.get(-1)["reward"]

        return observation_data

    def upload_data(self, state, action):
        """Uploads state and action_reward replay buffer"""
        raise Error("This function is not called, and should not be called anywhere")

    # TODO: @Zhiyao - needs implementation. See comment below -- done
    def upload_state(self, state_list: list):
        # print(
        #     "@Zhiyao, you'd need to implement `record_dic` functionality in here where you're only adding to `state_buffer`.\n"
        #     "From `record_dic`, extract state information."
        # )
        # raise NotImplementedError

        state = self.state_buffer.get_recent()
        state_bdg = self.state_to_dic(state_list)
        parse_state = self.parse_data(state, state_bdg)
        self.state_buffer.add(parse_state)

    # TODO: @Zhiyao - needs implementation. See comment below -- done
    def upload_action(self, action_list: list):
        # print(
        #     "@Zhiyao, you'd need to implement `record_dic` functionality in here where you're only adding to `action_buffer`\n"
        #     "From `record_dic`, extract action information."
        # )
        # raise NotImplementedError
        action_bdg = self.action_reward_to_dic(action_list)
        action = self.action_buffer.get_recent()
        parse_action = self.parse_data(action, action_bdg)
        self.action_buffer.add(parse_action)

    # TODO: @Zhiyao - This needs significant modification. only make use of `next_state` which will come from ** main.py **
    # >>> this should include state information required for both heating, cooling, solar, electricity, and if necessary capcity estimation.
    # >>> make sure to use only 2 buffers, one for state and one for action. You can make use of state buffer for various purposes - heating, cooling, etc. estimation.
    def state_to_dic(self, state_list: list):
        state_bdg = {}
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

        s_dic = {}
        # heating/cooling generation
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
        elec_cons = [
            state_bdg[i]["net_electricity_consumption"] for i in self.building_ids
        ]
        # solar/pv generation
        diffuse_solar_rad = [
            state_bdg[i]["diffuse_solar_rad"] for i in self.building_ids
        ]
        direct_solar_rad = [state_bdg[i]["direct_solar_rad"] for i in self.building_ids]
        diffuse_6h = [
            state_bdg[i]["diffuse_solar_rad_pred_6h"] for i in self.building_ids
        ]
        direct_6h = [
            state_bdg[i]["direct_solar_rad_pred_6h"] for i in self.building_ids
        ]
        diffuse_12h = [
            state_bdg[i]["diffuse_solar_rad_pred_12h"] for i in self.building_ids
        ]
        direct_12h = [
            state_bdg[i]["direct_solar_rad_pred_12h"] for i in self.building_ids
        ]
        diffuse_24h = [
            state_bdg[i]["diffuse_solar_rad_pred_24h"] for i in self.building_ids
        ]
        direct_24h = [
            state_bdg[i]["direct_solar_rad_pred_24h"] for i in self.building_ids
        ]
        t_out_6h = [state_bdg[i]["t_out_pred_6h"] for i in self.building_ids]
        t_out_12h = [state_bdg[i]["t_out_pred_12h"] for i in self.building_ids]
        t_out_24h = [state_bdg[i]["t_out_pred_24h"] for i in self.building_ids]

        # heating/cooling generation
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
        # solar/pv generation
        s_dic["diffuse_solar_rad"] = diffuse_solar_rad
        s_dic["direct_solar_rad"] = direct_solar_rad
        s_dic["diffuse_6h"] = diffuse_6h
        s_dic["direct_6h"] = direct_6h
        s_dic["diffuse_12h"] = diffuse_12h
        s_dic["direct_12h"] = direct_12h
        s_dic["diffuse_24h"] = diffuse_24h
        s_dic["direct_24h"] = direct_24h
        s_dic["solar_gen"] = solar_gen
        s_dic["t_out"] = t_out
        s_dic["t_out_6h"] = t_out_6h
        s_dic["t_out_12h"] = t_out_12h
        s_dic["t_out_24h"] = t_out_24h

        return s_dic

    # TODO: @Zhiyao - no need to store reward. No RL happening. Plz remove functionality. -- done
    def action_reward_to_dic(self, action):
        a_dic = {}
        a_c = [action[i][0] for i in self.building_ids]
        a_h = [action[i][1] for i in self.building_ids]
        a_b = [action[i][2] for i in self.building_ids]

        a_dic["action_C"] = a_c
        a_dic["action_H"] = a_h
        a_dic["action_bat"] = a_b

        return a_dic

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

    def gather_input(self, timestep):
        # assert daystep % 24 == 0, "only gather input at the beginning of the day"
        """buffer(-1) is the current day record, so buffer(-2) is for yesterday"""
        T = 24
        window = T - timestep % 24
        buffer = self.state_buffer.get(-2)

        input_solar_full = {uid: np.zeros([24, 2]) for uid in self.building_ids}
        input_elec_full = {uid: np.zeros([24, 1]) for uid in self.building_ids}
        input_solar = {uid: np.zeros([window, 2]) for uid in self.building_ids}
        input_elec = {uid: np.zeros([window, 1]) for uid in self.building_ids}

        for uid in self.building_ids:
            x_diffuse_6h = np.array(buffer["diffuse_6h"])[-6:, uid]
            x_diffuse_12h = np.array(buffer["diffuse_12h"])[-6:, uid]
            x_diffuse_24h = np.array(buffer["diffuse_24h"])[-12:, uid]
            x_direct_6h = np.array(buffer["direct_6h"])[-6:, uid]
            x_direct_12h = np.array(buffer["direct_12h"])[-6:, uid]
            x_direct_24h = np.array(buffer["direct_24h"])[-12:, uid]

            input_solar_full[uid][0:6, 0] = x_diffuse_6h
            input_solar_full[uid][0:6, 1] = x_direct_6h
            input_solar_full[uid][6:12, 0] = x_diffuse_12h
            input_solar_full[uid][6:12, 1] = x_direct_12h
            input_solar_full[uid][12:, 0] = x_diffuse_24h
            input_solar_full[uid][12:, 1] = x_direct_24h

            x_elec_6h = np.array(buffer["t_out_6h"])[-6:, uid]
            x_elec_12h = np.array(buffer["t_out_12h"])[-6:, uid]
            x_elec_24h = np.array(buffer["t_out_24h"])[-12:, uid]

            input_elec_full[uid][0:6, 0] = x_elec_6h
            input_elec_full[uid][6:12, 0] = x_elec_12h
            input_elec_full[uid][12:, 0] = x_elec_24h

            input_solar[uid][:, :] = input_solar_full[uid][timestep % 24 :, :]
            input_elec[uid][:, :] = input_elec_full[uid][timestep % 24 :, :]

        return input_solar, input_elec

    def infer_solar_electricity_load(self, timestep: int):
        # assert (
        #     daystep % 24 == 0
        # ), "only make day-ahead prediction at the first hour of the day"
        """changed for adaptive dispatch--make inference every hour"""
        if timestep % 24 == 0:
            self.calculate_avg()  # make sure get_recent() returns in 24*9 shape

        T = 24
        window = T - timestep % 24

        pred_buffer = self.state_buffer
        # ---------------fitting regression model---------------
        x_solar, y_solar, x_elec, y_elec = self.reshape_array(pred_buffer)
        for uid in self.building_ids:
            self.regr_solar[uid].fit(x_solar[uid], y_solar[uid])
            self.regr_elec[uid].fit(x_elec[uid], y_elec[uid])
        # ------------------start prediction-------------------
        input_solar, input_elec = self.gather_input(timestep)
        solar_gen = {uid: np.zeros([window + 2]) for uid in self.building_ids}
        elec_dem = {uid: np.zeros([window + 2]) for uid in self.building_ids}

        daytype = {uid: 0 for uid in self.building_ids}
        day_pred_solar = {uid: np.zeros([window]) for uid in self.building_ids}
        day_pred_elec = {uid: np.zeros([window]) for uid in self.building_ids}

        for uid in self.building_ids:
            if timestep % 24 in [0]:
                solar_gen[uid][0] = (
                    pred_buffer.get(-2)["solar_gen"][-1][uid] - self.solar_avg[uid][23]
                )
                solar_gen[uid][1] = (
                    pred_buffer.get(-1)["solar_gen"][timestep % 24][uid]
                    - self.solar_avg[uid][0]
                )
            else:
                solar_gen[uid][0] = (
                    pred_buffer.get(-1)["solar_gen"][(timestep - 1) % 24][uid]
                    - self.solar_avg[uid][(timestep - 1) % 24]
                )
                solar_gen[uid][1] = (
                    pred_buffer.get(-1)["solar_gen"][timestep % 24][uid]
                    - self.solar_avg[uid][timestep % 24]
                )
            daytype[uid] = pred_buffer.get(-1)["daytype"][0][uid]

            if timestep % 24 in [0]:
                if daytype[uid] in [7]:
                    elec_dem[uid][0] = (
                        pred_buffer.get(-2)["elec_dem"][-1][uid]
                        - self.elec_weekend1_avg[uid][23]
                    )
                    elec_dem[uid][1] = (
                        pred_buffer.get(-1)["elec_dem"][0][uid]
                        - self.elec_weekend1_avg[uid][0]
                    )
                elif daytype[uid] in [1, 8]:
                    elec_dem[uid][0] = (
                        pred_buffer.get(-2)["elec_dem"][-1][uid]
                        - self.elec_weekend2_avg[uid][23]
                    )
                    elec_dem[uid][1] = (
                        pred_buffer.get(-1)["elec_dem"][0][uid]
                        - self.elec_weekend2_avg[uid][0]
                    )
                else:
                    elec_dem[uid][0] = (
                        pred_buffer.get(-2)["elec_dem"][-1][uid]
                        - self.elec_weekday_avg[uid][23]
                    )
                    elec_dem[uid][1] = (
                        pred_buffer.get(-1)["elec_dem"][0][uid]
                        - self.elec_weekday_avg[uid][0]
                    )
            else:
                if daytype[uid] in [7]:
                    elec_dem[uid][0] = (
                        pred_buffer.get(-1)["elec_dem"][(timestep - 1) % 24][uid]
                        - self.elec_weekend1_avg[uid][(timestep - 1) % 24]
                    )
                    elec_dem[uid][1] = (
                        pred_buffer.get(-1)["elec_dem"][timestep % 24][uid]
                        - self.elec_weekend1_avg[uid][timestep % 24]
                    )
                elif daytype[uid] in [1, 8]:
                    elec_dem[uid][0] = (
                        pred_buffer.get(-1)["elec_dem"][(timestep - 1) % 24][uid]
                        - self.elec_weekend2_avg[uid][(timestep - 1) % 24]
                    )
                    elec_dem[uid][1] = (
                        pred_buffer.get(-1)["elec_dem"][timestep % 24][uid]
                        - self.elec_weekend2_avg[uid][timestep % 24]
                    )
                else:
                    elec_dem[uid][0] = (
                        pred_buffer.get(-1)["elec_dem"][(timestep - 1) % 24][uid]
                        - self.elec_weekday_avg[uid][(timestep - 1) % 24]
                    )
                    elec_dem[uid][1] = (
                        pred_buffer.get(-1)["elec_dem"][timestep % 24][uid]
                        - self.elec_weekday_avg[uid][timestep % 24]
                    )

            # # daytype[uid] = pred_buffer.get_recent()["day"][0][uid]
            # if daytype[uid] in [7]:
            #     elec_dem[uid][1] = (
            #         pred_buffer.get(-1)["non_shiftable_load"][0][uid]
            #         - self.elec_weekend1_avg[uid][0]
            #     )
            # elif daytype[uid] in [1, 8]:
            #     elec_dem[uid][1] = (
            #         pred_buffer.get(-1)["non_shiftable_load"][0][uid]
            #         - self.elec_weekend2_avg[uid][0]
            #     )
            # else:
            #     elec_dem[uid][1] = (
            #         pred_buffer.get(-1)["non_shiftable_load"][0][uid]
            #         - self.elec_weekday_avg[uid][0]
            #     )

            for i in range(np.shape(input_solar[uid])[0]):
                x_pred = [
                    [
                        solar_gen[uid][i],
                        solar_gen[uid][i + 1],
                        input_solar[uid][i, 0],
                        input_solar[uid][i, 1],
                    ]
                ]
                y_pred = self.regr_solar[uid].predict(x_pred)
                avg = self.solar_avg[uid][(i + 1 + timestep) % 24]
                day_pred_solar[uid][i] = max(0, y_pred.item() + avg)
                solar_gen[uid][i + 2] = y_pred.item()
            day_pred_solar[uid] = np.append(
                np.array([pred_buffer.get(-1)["solar_gen"][timestep % 24][uid]]),
                day_pred_solar[uid][0 : window - 1],
            )

            for i in range(len(input_elec[uid])):
                x_pred = [
                    [elec_dem[uid][i], elec_dem[uid][i + 1], input_elec[uid][i, 0]]
                ]  # ignore humidity
                y_pred = self.regr_elec[uid].predict(x_pred)
                if daytype[uid] in [7]:
                    avg = self.elec_weekend1_avg[uid][(i + 1 + timestep) % 24]
                elif daytype[uid] in [1, 8]:
                    avg = self.elec_weekend2_avg[uid][(i + 1 + timestep) % 24]
                else:
                    avg = self.elec_weekday_avg[uid][(i + 1 + timestep) % 24]
                day_pred_elec[uid][i] = max(0, y_pred.item() + avg)
                elec_dem[uid][i + 2] = y_pred.item()
            day_pred_elec[uid] = np.append(
                np.array([pred_buffer.get(-1)["elec_dem"][timestep % 24][uid]]),
                day_pred_elec[uid][: window - 1],
            )

        return day_pred_solar, day_pred_elec, input_elec

    # reshape input for fitting the model
    def reshape_array(self, pred_buffer: ReplayBuffer):
        """only reshape array at the beginning of the day"""
        x_solar = {i: [] for i in self.building_ids}
        y_solar = {i: [] for i in self.building_ids}
        x_elec = {i: [] for i in self.building_ids}
        y_elec = {i: [] for i in self.building_ids}
        solar_gen = []
        elec_dem = []
        x_diffuse = []
        x_direct = []
        x_tout = []
        daytype = []

        for i in range(-14, 0):
            solar_gen = pred_buffer.get(i)["solar_gen"]
            # expect solar_gen to be [24*7, 9] vertically sequential
            elec_dem = pred_buffer.get(i)["elec_dem"]
            x_diffuse = pred_buffer.get(i)["diffuse_solar_rad"]
            x_direct = pred_buffer.get(i)["direct_solar_rad"]
            x_tout = pred_buffer.get(i)["t_out"]
            daytype = pred_buffer.get(i)["daytype"]

            for uid in self.building_ids:
                for i in range(2, np.shape(solar_gen)[0] - 1):
                    x_append1 = [
                        solar_gen[i - 2][uid] - self.solar_avg[uid][(i - 2) % 24],
                        solar_gen[i - 1][uid] - self.solar_avg[uid][(i - 1) % 24],
                    ]
                    x_append2 = [x_diffuse[i][uid], x_direct[i][uid]]
                    y = [solar_gen[i][uid] - self.solar_avg[uid][i % 24]]
                    x_solar[uid].append(x_append1 + x_append2)
                    y_solar[uid].append(y)

                for i in range(2, np.shape(elec_dem)[0] - 1):
                    if daytype[i - 2][uid] in [7]:
                        x_temp1 = (
                            elec_dem[i - 2][uid]
                            - self.elec_weekend1_avg[uid][(i - 2) % 24]
                        )
                    elif daytype[i - 2][uid] in [1, 8]:
                        x_temp1 = (
                            elec_dem[i - 2][uid]
                            - self.elec_weekend2_avg[uid][(i - 2) % 24]
                        )
                    else:
                        x_temp1 = (
                            elec_dem[i - 2][uid]
                            - self.elec_weekday_avg[uid][(i - 2) % 24]
                        )
                    if daytype[i - 1][uid] in [7]:
                        x_temp2 = (
                            elec_dem[i - 1][uid]
                            - self.elec_weekend1_avg[uid][(i - 1) % 24]
                        )
                    elif daytype[i - 1][uid] in [1, 8]:
                        x_temp2 = (
                            elec_dem[i - 1][uid]
                            - self.elec_weekend2_avg[uid][(i - 1) % 24]
                        )
                    else:
                        x_temp2 = (
                            elec_dem[i - 1][uid]
                            - self.elec_weekday_avg[uid][(i - 1) % 24]
                        )

                    x_append1 = [x_temp1, x_temp2]
                    x_append2 = [x_tout[i][uid]]

                    if daytype[i][uid] in [7]:
                        y = elec_dem[i][uid] - self.elec_weekend1_avg[uid][i % 24]
                    elif daytype[i][uid] in [1, 8]:
                        y = elec_dem[i][uid] - self.elec_weekend2_avg[uid][i % 24]
                    else:
                        y = elec_dem[i][uid] - self.elec_weekday_avg[uid][i % 24]

                    x_elec[uid].append(x_append1 + x_append2)
                    # print("x_pred for elec: ", x_append1 + x_append2)
                    y_elec[uid].append([y])

        return x_solar, y_solar, x_elec, y_elec

    # TODO: @Zhiyao - need to add in capacity estimation
    def infer_load(self, timestep: int):
        """Returns heating, cooling, solar and electricity loads"""
        return [
            *self.infer_heating_cooling_estimate(timestep),
            *self.infer_solar_electricity_load(timestep),
        ]

    # TODO: @Zhiyao - state_buffer is appended before calling action. So at start of day, we have already stored state data for hour 0.
    # >>> We infer at start of day, not end of day (in case of day-ahead). In case of adaptive, the dimensions need to be 24 - t, where t [0, 23].
    def infer_heating_cooling_estimate(self, timestep):
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
        T = 24
        window = T - timestep % 24

        est_c_load = {uid: np.zeros(24) for uid in self.building_ids}
        est_h_load = {uid: np.zeros(24) for uid in self.building_ids}
        adaptive_c_load = {uid: np.zeros(window) for uid in self.building_ids}
        adaptive_h_load = {uid: np.zeros(window) for uid in self.building_ids}

        c_hasest = {
            uid: np.zeros(24, dtype=np.int16) for uid in self.building_ids
        }  # -1:clipped, 0:non-est, 1:regression, 2: moving avg
        h_hasest = {uid: np.zeros(24, dtype=np.int16) for uid in self.building_ids}
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
            while not jump_out:
                now_state = self.state_buffer.get(-2)
                now_c_soc = now_state["soc_c"][time][uid]
                now_h_soc = now_state["soc_h"][time][uid]
                now_b_soc = now_state["soc_b"][time][uid]
                now_t_out = now_state["t_out"][time][uid]
                now_solar = now_state["solar_gen"][time][uid]
                now_elec_dem = now_state["elec_dem"][time][uid]
                cop_c = self.cop_cal(now_t_out)  # cop at t
                now_action = self.action_buffer.get(-2)
                now_action_c = now_action["action_C"][time][uid]
                now_action_h = now_action["action_H"][time][uid]
                now_action_b = now_action["action_bat"][time][uid]

                if time != 0:
                    prev_state = now_state
                    prev_t_out = prev_state["t_out"][time - 1][
                        uid
                    ]  # when time=0, time-1=-1
                else:
                    prev_state = self.state_buffer.get(-3)
                    prev_t_out = prev_state["t_out"][-1][uid]

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
                        - (self.true_val_c[uid] / cop_c)
                        * (next_c_soc - (1 - self.CF_C) * now_c_soc)
                        * 0.9
                        - (self.true_val_h[uid] / effi_h)
                        * (next_h_soc - (1 - self.CF_H) * now_h_soc)
                        - (next_b_soc - (1 - self.CF_B) * now_b_soc)
                        * self.true_val_b[uid]
                        / 0.9
                    )
                else:
                    next_state = self.state_buffer.get(-2)
                    next_c_soc = next_state["soc_c"][0][uid]
                    next_h_soc = next_state["soc_h"][0][uid]
                    next_b_soc = next_state["soc_b"][0][uid]
                    next_t_out = next_state["t_out"][0][uid]
                    next_elec_con = next_state["elec_cons"][0][uid]
                    y = (
                        now_solar
                        + next_elec_con
                        - now_elec_dem
                        - (self.true_val_c[uid] / cop_c)
                        * (next_c_soc - (1 - self.CF_C) * now_c_soc)
                        * 0.9
                        - (self.true_val_h[uid] / effi_h)
                        * (next_h_soc - (1 - self.CF_H) * now_h_soc)
                        - (next_b_soc - (1 - self.CF_B) * now_b_soc)
                        * self.true_val_b[uid]
                        / 0.9
                    )

                a_clip_c = next_c_soc - (1 - self.CF_C) * now_c_soc
                a_clip_h = next_h_soc - (1 - self.CF_H) * now_h_soc

                # if (
                #     repeat_times == 0
                # ):  # can we calculate direct when now_action > 0?
                #     if uid in [2, 3]:
                #         c_load = max(0, y * cop_c)
                #         h_load = 0
                #         est_h_load[uid][time] = h_load
                #         est_c_load[uid][time] = c_load
                #         c_hasest[uid][time], h_hasest[uid][time] = -1, -1
                #     else:
                #         if (
                #             abs(a_clip_c - now_action_c) > 0.001
                #             and now_action_c < 0
                #         ):  # cooling get clipped
                #             c_load = abs(a_clip_c * self.true_val_c[uid])
                #             if (
                #                 abs(a_clip_h - now_action_h) > 0.001
                #                 and now_action_h < 0
                #             ):  # heating get clipped
                #                 h_load = a_clip_h * self.true_val_h[uid]
                #             else:  # heating not clipped
                #                 h_load = (y - c_load / cop_c) * effi_h
                #             est_h_load[uid][time] = h_load
                #             est_c_load[uid][time] = c_load
                #             c_hasest[uid][time], h_hasest[uid][time] = -1, -1
                #         elif (
                #             abs(a_clip_h > now_action_h) > 0.01 and a_clip_h < 0
                #         ):  # h clipped but c not clipped
                #             h_load = abs(a_clip_h * self.true_val_h[uid])
                #             c_load = (y - h_load / effi_h) * cop_c
                #             c_hasest[uid][time], h_hasest[uid][time] = -1, -1
                #             est_h_load[uid][time] = h_load
                #             est_c_load[uid][time] = c_load
                if uid in [2, 3]:
                    c_load = max(0, y * cop_c)
                    h_load = 0
                    est_h_load[uid][time] = h_load
                    est_c_load[uid][time] = c_load
                    c_hasest[uid][time], h_hasest[uid][time] = -1, -1
                else:
                    prev_t_cop = self.cop_cal(prev_t_out)
                    now_t_cop = self.cop_cal(now_t_out)
                    next_t_cop = self.cop_cal(next_t_out)
                    # if (
                    #     prev_t_cop != now_t_cop
                    #     or prev_t_cop != next_t_cop
                    #     or now_t_cop != next_t_cop
                    # ):
                    ## get results of slope in regr model
                    c_load = h_load = max(1, y - (1 / now_t_cop + 1 / 0.9))
                    c_hasest[uid][time], h_hasest[uid][time] = 1, 1
                    # else:  # COP remaining the same (zero)
                    #     h_load = self.avg_h_load[uid][time]
                    #     c_load = self.avg_c_load[uid][time]
                    #     c_hasest[uid][time], h_hasest[uid][time] = 2, 2
                    # save load est to buffer
                    est_h_load[uid][time] = np.round(h_load, 2)
                    est_c_load[uid][time] = np.round(c_load, 2)
                # if c_hasest[uid][time] not in [
                #     0,
                #     2,
                # ]:  # meaning that avg can be updated
                if self.daystep >= 1:
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
                    self.daystep += 1
                    break
        for uid in self.building_ids:
            est_h_load[uid] *= 1.5
            est_c_load[uid] *= 1.5
            adaptive_h_load[uid] = est_h_load[uid][T - window :]
            adaptive_c_load[uid] = est_c_load[uid][T - window :]
        return adaptive_h_load, adaptive_c_load


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
#         t_idx: int = -1,  # daystep (hour) of the simulation [0 - (4years-1)]
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
            len(current_data) == 20  # actions E_grid_collect. Section 1.3.1
        ), f"Invalid number of parameters, found: {len(current_data)}, expected: 20. Can't run Oracle agent optimization."

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

    # TODO: INTEGRATE testing for ablation
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
        Returns data (dict) for each building from `env` for `t` daystep
        @Params:
        1. env: CityLearn environment.
        2. t: current hour from TD3.py (agents.total_it). Goes from (0 - 4 * 365 * 24]
        3. E_grid: Current net electricity consumption, obtained from environment.
        4. E_grid_memory: Previous hours' net electricity consumption. Obtained from memory (replaybuffer).
        5. actions: set of actions for current hour. Appends to replaybuffer.
        6. rewards: per building reward. Appends to replaybuffer.
        """
        ### FB - Full batch. Trim output X[:time-step]
        ### CT - current daystep only. X = full_data[time-step], no access to full_data
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
