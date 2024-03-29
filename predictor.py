from utils import ReplayBuffer, DataLoader
from collections import defaultdict
from citylearn import CityLearn
from copy import deepcopy

import numpy as np
import pandas as pd
import sys

from scipy import stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# TODO: @Zhiyao - add in parameter/initialization for capacity as discussed.
class Predictor(DataLoader):
    """
    we have following functions:

    def __init__(self, action_space: list) -> None: see lines for explanations of vars

    def estimate_data(
        self, replay_buffer: ReplayBuffer, timestep: int, is_adaptive: bool = False
    ): return prediction data to TD3 agent (both day ahead and adaptive)

    def full_parse_data(
        self, previous_data: dict, current_data: dict, window: int = 24
    ): merge current hour data in current day data in the buffer

    def calculate_avg(self): calculate last 14-day solar gen/elec loads for updating predictor
    def get_day_data(self, replay_buffer: ReplayBuffer, timestep: int): helper method for uploading data to memory for estimation
    def upload_data(self, state, action): NOT IMPLEMENTED
    def upload_state(self, state_list: list): upload state to state buffer
    def upload_action(self, action_list: list): upload action to action buffer
    def state_to_dic(self, state_list: list): convert type from list to dict with state names
    def cop_cal(self, temp): calculate COP
    def gather_input(self, timestep): gather predictions of today's params for elec/solar prediction
    def infer_solar_electricity_load(self, timestep: int): make predictions for solar/elec
    def reshape_array(self, pred_buffer: ReplayBuffer): update regression model for prediction w/ latest 14-day data to fit
    def infer_load(self, timestep: int): Returns heating, cooling, solar and electricity loads
    def infer_heating_cooling_estimate(self, timestep): infer previous day h/c loads and calculate moving average
        --use average with peak scaling as day-ahead predictions
    def select_action(self, timestep: int): interface w/ TD3.py to select actions in online exploration period
    def get_params(self, timestep: int) -> dict: return estimated params from online exploration
    def quantile_reg(self): quantile regression for h/c capacity estimation--convert to DataFrame
    def quantile_reg_H(self, uid, quantiles, H_dataframe, climate_zone): quantile regression for H capacity
    def quantile_reg_C(self, uid, quantiles, C_dataframe, climate_zone): quantile regression for C capacity
    def estimate_h(self): choose actions for estimating h capacity
    def estimate_c(self): choose actions for estimating c capacity
    def estimate_bat(self): choose actions for estimating bat capacity

    THIS IS HOW THE WHOLE CLASS RUNS:

    when select_action in TD3.py is called
        store state in state buffer
        run select_action function (around line 1100) in predictor.py
        store action from select_action

    select_action runs as follows:
        all the first, run estimate_bat (i manually set three whole windows (from 22h to 6h+1) as time limit to run this,
                                        otherwise this loop will be forced over and continue to h/c capacity estimation)
        if we have reliable values of bat capacity (1 datapoint) and nominal power (3 datapoints, finally choose the maximal one),
            then we jump out from E estimation and enter H/C estimation
        during H/C estimation, i set each whole window for H/C est alternately everyday,
            i.e. if current window (22 to 6(+1)) is for H est, then the next (22 to 6(+1)) will be C est.
        run estimate_h or estimate_c alternately in each window
        if current timestep is RBC_THRESHOLD-1 (last step for online exploration), run quantile_reg()

    in quantile_reg, we have following process:
        convert ndarrays to pd.DataFrame since it is required by statsmodel.QuantileRegressor
        run quantile_reg_H and quantile_reg_C for each building and choose a certain quantile (currently we use two-point quantile avg)

    when we initialize the digital twin, we may call get_params for capacity and nominal power data

    """

    def __init__(self, building_info: dict, action_space: list) -> None:
        super().__init__(action_space)
        self.building_ids = range(len(self.action_space))  # number of buildings
        # initialize two buffers
        self.state_buffer = ReplayBuffer(buffer_size=365, batch_size=32)
        self.action_buffer = ReplayBuffer(buffer_size=365, batch_size=32)

        # define constants
        self.CF_C = 0.006
        self.CF_H = 0.008
        self.CF_B = 0
        self.rbc_threshold = 336
        self.annual_c_demand = {uid: 0 for uid in self.building_ids}
        self.annual_h_demand = {uid: 0 for uid in self.building_ids}
        for uid in self.building_ids:
            self.annual_h_demand[uid] = building_info[f"Building_{str(uid+1)}"][
                "Annual_DHW_demand (kWh)"
            ]
            self.annual_c_demand[uid] = building_info[f"Building_{str(uid+1)}"][
                "Annual_cooling_demand (kWh)"
            ]

        self.a_c_high = [action_space[uid].high[0] for uid in self.building_ids]
        self.a_h_high = [action_space[uid].high[1] for uid in self.building_ids]

        # define regression model
        self.regr = LinearRegression(fit_intercept=False)  # , positive=True)
        # average and peak values for load prediction
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
        self.c_weekday_avg = {i: np.zeros(24) for i in self.building_ids}
        self.c_weekend1_avg = {i: np.zeros(24) for i in self.building_ids}
        self.c_weekend2_avg = {i: np.zeros(24) for i in self.building_ids}
        self.h_weekday_avg = {i: np.zeros(24) for i in self.building_ids}
        self.h_weekend1_avg = {i: np.zeros(24) for i in self.building_ids}
        self.h_weekend2_avg = {i: np.zeros(24) for i in self.building_ids}
        self.regr_solar = {uid: LinearRegression() for uid in self.building_ids}
        self.regr_elec = {uid: LinearRegression() for uid in self.building_ids}
        # ----------below vars are for capacity estimation-------------
        self.timestep = 0
        # -----------thresholds-------------
        self.tau_c = {uid: 0.2 for uid in self.building_ids}
        self.tau_h = {uid: 0.2 for uid in self.building_ids}
        self.tau_b = {uid: 0.6 for uid in self.building_ids}
        self.tau_cplus = {uid: 0.1 for uid in self.building_ids}
        self.tau_hplus = {uid: 0.1 for uid in self.building_ids}
        self.action_c = {uid: 0.1 for uid in self.building_ids}
        self.action_h = {uid: 0.1 for uid in self.building_ids}
        # ------------indications of estimation procedure----------
        self.prev_hour_est_b = {uid: False for uid in self.building_ids}
        self.prev_hour_est_c = {uid: False for uid in self.building_ids}
        self.prev_hour_est_h = {uid: False for uid in self.building_ids}
        self.has_heating = {uid: True for uid in self.building_ids}
        self.a_clip = {uid: None for uid in self.building_ids}
        self.avail_ratio_est_c = {uid: False for uid in self.building_ids}
        self.avail_ratio_est_h = {uid: False for uid in self.building_ids}
        self.avail_nominal = {uid: False for uid in self.building_ids}
        self.prev_hour_nom = {uid: False for uid in self.building_ids}
        # ------------number of est points---------------
        self.num_elec_points = {uid: 0 for uid in self.building_ids}
        self.num_h_points = {uid: 0 for uid in self.building_ids}
        self.num_c_points = {uid: 0 for uid in self.building_ids}
        self.ratio_c_est = {uid: [] for uid in self.building_ids}
        self.C_bd_est = {uid: [] for uid in self.building_ids}
        self.ratio_h_est = {uid: [] for uid in self.building_ids}
        self.H_bd_est = {uid: [] for uid in self.building_ids}
        # ------------estimated values: for return---------
        self.cap_c_est = {uid: [] for uid in self.building_ids}
        self.cap_h_est = {uid: [] for uid in self.building_ids}
        self.cap_b_est = {uid: [] for uid in self.building_ids}
        self.effi_b = {uid: 0 for uid in self.building_ids}
        self.effi_c = {uid: 0 for uid in self.building_ids}
        self.effi_h = {uid: 0 for uid in self.building_ids}
        self.nominal_b = {uid: [] for uid in self.building_ids}
        self.ratio_c = {uid: 0 for uid in self.building_ids}
        self.ratio_h = {uid: 0 for uid in self.building_ids}
        # ---------------results of est------------------
        self.nom_p_est = {uid: 0 for uid in self.building_ids}
        self.capacity_b = {uid: 0 for uid in self.building_ids}
        self.H_qr_est = {uid: 0 for uid in self.building_ids}
        self.C_qr_est = {uid: 0 for uid in self.building_ids}

        self.E_day = True
        self.C_day = self.H_day = False

    # TODO: @Zhiyao + @Qasim - this function has not tested. Depends on internal predictor to work.
    def estimate_data(
        self, replay_buffer: ReplayBuffer, timestep: int, is_adaptive: bool = False
    ):
        """Estimates data to be passed into Optimization model for 24hours into future."""
        if is_adaptive:
            # if hour start of day, `get_recent()` will automatically return an empty dictionary
            data = self.full_parse_data(
                deepcopy(replay_buffer.get_recent()),
                self.get_day_data(replay_buffer, timestep),
                24 - timestep % 24,
            )

            replay_buffer.add(data)
        else:
            data = self.full_parse_data(
                deepcopy(replay_buffer.get_recent()),
                self.get_day_data(replay_buffer, timestep),
            )
            replay_buffer.add(data, full_day=True)

        return data

    def full_parse_data(
        self, previous_data: dict, current_data: dict, window: int = 24
    ):
        """Parses `current_data` for optimization and loads into `data`. Everything is of shape 24, 9"""
        TOTAL_PARAMS = 21
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
                value = np.repeat(value.reshape(1, -1), window, axis=0)

            if np.shape(value) == (24, len(self.building_ids)):
                data[key] = value
            else:
                # makes sure horizontal dimensions are 24
                if previous_data and "action" not in key:
                    data[key] = np.concatenate(
                        (previous_data[key][: 24 - window], value), axis=0
                    )
                else:
                    data[key] = np.pad(value, ((window, 0), (0, 0)))

            assert data[key].shape == (
                24,
                9,
            ), f"Incorrect dimensions for {key}: {data[key].shape}"

        return data

    # TODO: @Zhiyao - needs fixing -- done
    def calculate_avg(self):
        """calculate hourly avg value of the day"""
        # print("calculating avg")
        buffer = self.state_buffer
        daytype = {i: 0 for i in self.building_ids}
        elec_dem = {i: [] for i in self.building_ids}
        solar_gen = {i: [] for i in self.building_ids}
        range_index = -13
        for day in range(range_index, 0):
            for uid in self.building_ids:
                elec_dem[uid] = np.array(buffer.get(day - 1)["elec_dem"])[:, uid]
                solar_gen[uid] = np.array(buffer.get(day - 1)["solar_gen"])[:, uid]
                daytype[uid] = np.array(buffer.get(day - 1)["daytype"])[0, uid]

                if self.solar_avg[uid].all() != 0:
                    self.solar_avg[uid] = (
                        self.solar_avg[uid] * 0.5 + solar_gen[uid] * 0.5
                    )
                else:
                    self.solar_avg[uid] = solar_gen[uid]

                if daytype[uid] in [7]:
                    if self.elec_weekend1_avg[uid].all() != 0:
                        self.elec_weekend1_avg[uid] = (
                            self.elec_weekend1_avg[uid] * 0.5 + elec_dem[uid] * 0.5
                        )
                    else:
                        self.elec_weekend1_avg[uid] = elec_dem[uid]
                elif daytype[uid] in [1, 8]:
                    if self.elec_weekend2_avg[uid].all() != 0:
                        self.elec_weekend2_avg[uid] = (
                            self.elec_weekend2_avg[uid] * 0.5 + elec_dem[uid] * 0.5
                        )
                    else:
                        self.elec_weekend2_avg[uid] = elec_dem[uid]
                else:
                    if self.elec_weekday_avg[uid].all() != 0:
                        self.elec_weekday_avg[uid] = (
                            self.elec_weekday_avg[uid] * 0.5 + elec_dem[uid] * 0.5
                        )
                    else:
                        self.elec_weekday_avg[uid] = elec_dem[uid]

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
        self.timestep = timestep
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

        # get capacity and nominal power parameters
        additional_parameters = self.get_params(timestep)

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
            C_max = np.max([C_max, C_bd.max(axis=0)], axis=0)  # global max

        temp = np.array([future_temp[uid].flatten() for uid in self.building_ids]).T
        COP_C = np.zeros((window, len(self.building_ids)))
        for hour in range(window):
            for bid in self.building_ids:
                COP_C[hour, bid] = self.cop_cal(temp[hour, bid])

        C_p_Csto = additional_parameters["C_p_Csto"]
        C_p_Hsto = additional_parameters["C_p_Hsto"]
        C_p_bat = additional_parameters["C_p_bat"]

        E_hpC_max = np.array(C_p_Csto) * np.array(self.a_c_high) / 2
        E_ehH_max = np.array(C_p_Hsto) * np.array(self.a_h_high) / 0.9

        c_bat_init = np.array(self.state_buffer.get(-1)["soc_b"])[-1]
        c_bat_init[c_bat_init == np.inf] = 0

        c_Hsto_init = np.array(self.state_buffer.get(-1)["soc_h"])[-1]
        c_Hsto_init[c_Hsto_init == np.inf] = 0

        # nominal power
        E_bat_max = additional_parameters["E_bat_max"]

        c_Csto_init = np.array(self.state_buffer.get(-1)["soc_c"])[-1]
        c_Csto_init[c_Csto_init == np.inf] = 0

        # add E-grid - default day-ahead
        egc = np.array(self.state_buffer.get(-1)["elec_cons"])
        observation_data["E_grid"] = np.pad(egc, ((0, T - egc.shape[0]), (0, 0)))

        observation_data["E_grid_prevhour"] = np.zeros((T, len(self.building_ids)))
        # observation_data["E_grid_prevhour"][0] = np.array(
        #     self.state_buffer.get(-2)["elec_cons"]
        # )[-1]
        observation_data["E_grid_prevhour"][0] = egc[0]
        for hour in range(1, timestep % 24 + 1):
            observation_data["E_grid_prevhour"][hour] = observation_data["E_grid"][hour]

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

        observation_data["E_bat_max"] = E_bat_max

        observation_data["action_H"] = self.action_buffer.get(-1)["action_H"]
        observation_data["action_C"] = self.action_buffer.get(-1)["action_C"]
        observation_data["action_bat"] = self.action_buffer.get(-1)["action_bat"]

        # add reward \in R^9 (scalar value for each building)
        # observation_data["reward"] = self.action_buffer.get(-1)["reward"]
        # print("\nE_ns_B4 from get_day_data:\n", np.array(observation_data["E_ns"])[:, 3].flatten(),
        #       np.size(np.array(observation_data["E_ns"])[:, 3].flatten()))
        return observation_data

    def upload_data(self, state, action):
        """Uploads state and action_reward replay buffer"""
        raise NotImplementedError(
            "This function is not called, and should not be called anywhere"
        )

    # TODO: @Zhiyao - needs implementation. See comment below -- done
    def upload_state(self, state_list: list):
        """upload state to state buffer"""
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
        """upload action to action buffer"""
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
        """convert type from list to dict with state names"""
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
        """convert type from list to dict with action names"""
        a_dic = {}
        a_c = [action[i][0] for i in self.building_ids]
        a_h = [action[i][1] for i in self.building_ids]
        a_b = [action[i][2] for i in self.building_ids]

        a_dic["action_C"] = a_c
        a_dic["action_H"] = a_h
        a_dic["action_bat"] = a_b

        return a_dic

    def cop_cal(self, temp):
        """calculate COP"""
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
        """gather predictions of today's params for elec/solar prediction"""
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
        """changed for adaptive dispatch--make inference every hour"""
        """make predictions for solar/elec"""
        if timestep % 24 in [0]:
            self.calculate_avg()  # make sure get_recent() returns in 24*9 shape

        T = 24
        window = T - timestep % 24

        pred_buffer = self.state_buffer
        # ---------------fitting regression model---------------
        x_solar, y_solar, x_elec, y_elec = self.reshape_array(pred_buffer)
        for uid in self.building_ids:
            self.regr_solar[uid].fit(x_solar[uid], y_solar[uid])
            # self.regr_elec[uid].fit(x_elec[uid], y_elec[uid])
        # ------------------start prediction-------------------
        input_solar, input_elec = self.gather_input(timestep)  # input shape: (window)
        solar_gen = {uid: np.zeros([T]) for uid in self.building_ids}
        elec_dem = {uid: np.zeros([T]) for uid in self.building_ids}

        daytype = {uid: 0 for uid in self.building_ids}
        day_pred_solar = {uid: np.zeros([window]) for uid in self.building_ids}
        day_pred_elec = {uid: np.zeros([window]) for uid in self.building_ids}

        for uid in self.building_ids:
            daytype[uid] = pred_buffer.get(-1)["daytype"][0][0]
            """now you are at hour 7, so far you have obsverd actual loads e1,...,e7. 
            and you want to predict for load for hours 8 to 24. the average load profile is a1,...,,a24. 
            so you first calculate the offset d= (e1+...+d7)/7 - (a1+...+a7)/7. 
            if d >0, it means the current day may have higher loads than average. 
            then for the prediction, yoou just predict a8+d,...,a_24+d"""
            if daytype[uid] in [7]:
                elec_dem = self.elec_weekend1_avg[uid]
            elif daytype[uid] in [1, 8]:
                elec_dem = self.elec_weekend2_avg[uid]
            else:
                elec_dem = self.elec_weekday_avg[uid]
            solar_gen = self.solar_avg[uid]

            if timestep % 24 == 0:
                day_pred_elec[uid] = elec_dem
                day_pred_solar[uid] = solar_gen
            else:
                today_e_load = np.array(pred_buffer.get(-1)["elec_dem"])
                today_s_load = np.array(pred_buffer.get(-1)["solar_gen"])
                day_hour = timestep % 24
                if timestep % 24 in [1, 2, 3, 4, 5]:
                    offset_e1 = np.sum(today_e_load[-day_hour - 1 :, uid]) / (
                        day_hour + 1
                    )
                    offset_e2 = np.sum(elec_dem[: day_hour + 1]) / (day_hour + 1)
                    offset_e = offset_e1 - offset_e2
                    day_pred_elec[uid] = elec_dem[day_hour:] + offset_e

                    # offset_s1 = np.sum(today_s_load[-day_hour-1:, uid]) / (day_hour+1)
                    # offset_s2 = np.sum(solar_gen[: day_hour+1]) / (day_hour+1)
                    # offset_s = offset_s1 - offset_s2
                    # day_pred_solar[uid] = solar_gen[day_hour:] + offset_s
                else:
                    offset_e1 = np.sum(today_e_load[-5:, uid]) / 5
                    offset_e2 = np.sum(elec_dem[day_hour - 4 : day_hour + 1]) / 5
                    offset_e = offset_e1 - offset_e2
                    day_pred_elec[uid] = elec_dem[day_hour:] + offset_e

                    # offset_s1 = np.sum(today_s_load[-5:, uid]) / 5
                    # offset_s2 = np.sum(solar_gen[day_hour - 4: day_hour+1]) / 5
                    # offset_s = offset_s1 - offset_s2
                    # day_pred_solar[uid] = solar_gen[day_hour:] + offset_s

                for t in range(len(day_pred_elec[uid])):
                    day_pred_elec[uid][t] = max(0, day_pred_elec[uid][t])
                    # day_pred_solar[uid][t] = max(0, day_pred_solar[uid][t])

            for i in range(np.shape(input_solar[uid])[0]):
                x_pred = [
                    [
                        input_solar[uid][i, 0],
                        input_solar[uid][i, 1],
                    ]
                ]
                y_pred = self.regr_solar[uid].predict(x_pred)
                avg = self.solar_avg[uid][(i + timestep) % 24]
                day_pred_solar[uid][i] = max(0, y_pred.item() + avg)

            day_pred_solar[uid][0] = pred_buffer.get(-1)["solar_gen"][-1][uid]
            day_pred_elec[uid][0] = pred_buffer.get(-1)["elec_dem"][-1][uid]

        return day_pred_solar, day_pred_elec, input_elec

    # reshape input for fitting the model
    def reshape_array(self, pred_buffer: ReplayBuffer):
        """only reshape array at the beginning of the day"""
        """update regression model for prediction w/ latest 14-day data to fit"""
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

        for i in range(-14, -1):
            solar_gen = pred_buffer.get(i)["solar_gen"]
            # expect solar_gen to be [24*7, 9] vertically sequential
            elec_dem = pred_buffer.get(i)["elec_dem"]
            x_diffuse = pred_buffer.get(i)["diffuse_solar_rad"]
            x_direct = pred_buffer.get(i)["direct_solar_rad"]
            x_tout = pred_buffer.get(i)["t_out"]
            daytype = pred_buffer.get(i)["daytype"]

            for uid in self.building_ids:
                for i in range(2, np.shape(solar_gen)[0] - 1):

                    x_append2 = [x_diffuse[i][uid], x_direct[i][uid]]
                    y = [solar_gen[i][uid] - self.solar_avg[uid][i % 24]]
                    x_solar[uid].append(x_append2)
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
        """infer previous day h/c loads and calculate moving average--use average with peak scaling as day-ahead predictions"""

        T = 24
        window = T - timestep % 24

        # hasest indicates whether every hour of the day has estimation.
        # only when all 0 become 1 in has_est, the function runs over.
        effi_h = 0.9

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
        daytype = self.state_buffer.get(-1)["daytype"][0][0]
        prev_daytype = self.state_buffer.get(-2)["daytype"][0][0]

        for uid in self.building_ids:
            # starting from t=0, need a loop to cycle time
            # say at hour=t, check if the action of c/h is clipped
            # if so, directly calculate h/c load and continue this loop
            if timestep % 24 == 0:
                for time in range(24):
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
                            - (self.C_qr_est[uid] / cop_c)
                            * (next_c_soc - (1 - self.CF_C) * now_c_soc)
                            * 0.9
                            - (self.H_qr_est[uid] / effi_h)
                            * (next_h_soc - (1 - self.CF_H) * now_h_soc)
                            - (next_b_soc - (1 - self.CF_B) * now_b_soc)
                            * self.capacity_b[uid]
                            / 0.9
                        )
                    else:
                        next_state = self.state_buffer.get(-1)
                        next_c_soc = next_state["soc_c"][0][uid]
                        next_h_soc = next_state["soc_h"][0][uid]
                        next_b_soc = next_state["soc_b"][0][uid]
                        next_t_out = next_state["t_out"][0][uid]
                        next_elec_con = next_state["elec_cons"][0][uid]
                        y = (
                            now_solar
                            + next_elec_con
                            - now_elec_dem
                            - (self.C_qr_est[uid] / cop_c)
                            * (next_c_soc - (1 - self.CF_C) * now_c_soc)
                            * 0.9
                            - (self.H_qr_est[uid] / effi_h)
                            * (next_h_soc - (1 - self.CF_H) * now_h_soc)
                            - (next_b_soc - (1 - self.CF_B) * now_b_soc)
                            * self.capacity_b[uid]
                            / 0.9
                        )

                    a_clip_c = next_c_soc - (1 - self.CF_C) * now_c_soc
                    a_clip_h = next_h_soc - (1 - self.CF_H) * now_h_soc

                    if self.has_heating[uid] is False:
                        c_load = max(0, y)
                        h_load = 0
                        est_h_load[uid][time] = h_load
                        est_c_load[uid][time] = c_load
                        c_hasest[uid][time], h_hasest[uid][time] = -1, -1
                    else:
                        ## get results of slope in regr model
                        # c_load = max(0, y * cop_c)
                        # h_load = max(0, y * effi_h)
                        c_load = h_load = max(0, y - (1 / cop_c + 1 / effi_h))
                        c_hasest[uid][time], h_hasest[uid][time] = 1, 1

                        # save load est to buffer
                        est_h_load[uid][time] = np.round(h_load, 2)
                        est_c_load[uid][time] = np.round(c_load, 2)

                    # ----------record average-------------
                    if time == 23:
                        if prev_daytype in [7]:
                            self.c_weekend1_avg[uid] = (
                                self.c_weekend1_avg[uid] * 0.5 + est_c_load[uid] * 0.5
                                if self.c_weekend1_avg[uid].all() != 0
                                else est_c_load[uid]
                            )
                            self.h_weekend1_avg[uid] = (
                                self.h_weekend1_avg[uid] * 0.5 + est_h_load[uid] * 0.5
                                if self.h_weekend1_avg[uid].all() != 0
                                else est_h_load[uid]
                            )
                        elif prev_daytype in [1, 8]:
                            self.c_weekend2_avg[uid] = (
                                self.c_weekend2_avg[uid] * 0.5 + est_c_load[uid] * 0.5
                                if self.c_weekend2_avg[uid].all() != 0
                                else est_c_load[uid]
                            )
                            self.h_weekend2_avg[uid] = (
                                self.h_weekend2_avg[uid] * 0.5 + est_h_load[uid] * 0.5
                                if self.h_weekend2_avg[uid].all() != 0
                                else est_h_load[uid]
                            )
                        else:
                            self.c_weekday_avg[uid] = (
                                self.c_weekday_avg[uid] * 0.5 + est_c_load[uid] * 0.5
                                if self.c_weekday_avg[uid].all() != 0
                                else est_c_load[uid]
                            )
                            self.h_weekday_avg[uid] = (
                                self.h_weekday_avg[uid] * 0.5 + est_h_load[uid] * 0.5
                                if self.h_weekday_avg[uid].all() != 0
                                else est_h_load[uid]
                            )
            # ------------use average-------------
            if daytype in [7]:
                est_h_load[uid] = self.h_weekend1_avg[uid]
                est_c_load[uid] = self.c_weekend1_avg[uid]
            elif daytype in [1, 8]:
                est_h_load[uid] = self.h_weekend2_avg[uid]
                est_c_load[uid] = self.c_weekend2_avg[uid]
            else:
                est_h_load[uid] = self.h_weekday_avg[uid]
                est_c_load[uid] = self.c_weekday_avg[uid]
            # ------------scaling maximal three hours--------------
            index_sort_h = np.argsort(est_h_load[uid])
            index_sort_c = np.argsort(est_c_load[uid])

            for ind in range(-3, 0):
                index_h = index_sort_h[ind]
                index_c = index_sort_c[ind]
                est_h_load[uid][index_h] = est_h_load[uid][index_h] * 1
                est_c_load[uid][index_c] = est_c_load[uid][index_c] * 1

            adaptive_h_load[uid] = est_h_load[uid][T - window :]
            adaptive_c_load[uid] = est_c_load[uid][T - window :]

        """
        time = timestep % 24
        if timestep == 24 * 14:
            self.calculate_avg_h_c()
        else:
            for uid in self.building_ids:
                # starting from t=0, need a loop to cycle time
                # say at hour=t, check if the action of c/h is clipped
                # if so, directly calculate h/c load and continue this loop
                # ------------compare w/ true action--------
                if time in [0]:
                    prev_state = self.state_buffer.get(-2)
                    now_state = self.state_buffer.get(-2)
                    next_state = self.state_buffer.get(-1)
                    now_action = self.action_buffer.get(-2)
                elif time in [1]:
                    prev_state = self.state_buffer.get(-2)
                    now_state = self.state_buffer.get(-1)
                    next_state = self.state_buffer.get(-1)
                    now_action = self.action_buffer.get(-1)
                else:
                    prev_state = self.state_buffer.get(-1)
                    now_state = self.state_buffer.get(-1)
                    next_state = self.state_buffer.get(-1)
                    now_action = self.action_buffer.get(-1)

                now_c_soc = now_state["soc_c"][time-1][uid]
                now_h_soc = now_state["soc_h"][time-1][uid]
                now_b_soc = now_state["soc_b"][time-1][uid]
                now_t_out = now_state["t_out"][time-1][uid]
                now_solar = now_state["solar_gen"][time-1][uid]
                now_elec_dem = now_state["elec_dem"][time-1][uid]
                daytype = now_state["daytype"][time-1][uid]
                cop_c = self.cop_cal(now_t_out)  # cop at t
                now_action_c = now_action["action_C"][time-1][uid]
                now_action_h = now_action["action_H"][time-1][uid]
                now_action_b = now_action["action_bat"][time-1][uid]
                prev_t_out = prev_state["t_out"][time-2][uid]
                next_c_soc = next_state["soc_c"][time][uid]
                next_h_soc = next_state["soc_h"][time][uid]
                next_b_soc = next_state["soc_b"][time][uid]
                next_elec_con = next_state["elec_cons"][time][uid]
                y = (
                        now_solar
                        + next_elec_con
                        - now_elec_dem
                        - (self.C_qr_est[uid] / cop_c)
                        * (next_c_soc - (1 - self.CF_C) * now_c_soc)
                        - (self.H_qr_est[uid] / effi_h)
                        * (next_h_soc - (1 - self.CF_H) * now_h_soc)
                        - (next_b_soc - (1 - self.CF_B) * now_b_soc)
                        * self.capacity_b[uid]
                        / 0.9
                )
                a_clip_c = next_c_soc - (1 - 0) * now_c_soc
                a_clip_h = next_h_soc - (1 - 0) * now_h_soc

                clip_action = False

                if a_clip_h > now_action_h and now_action_h < 0:  # get clipped
                    h_load = -(next_h_soc - (1 - self.CF_H) * now_h_soc) * self.H_qr_est[uid]
                    c_load = (now_solar + next_elec_con - now_elec_dem
                        - (next_b_soc - (1 - self.CF_B) * now_b_soc)
                        * self.capacity_b[uid] * 0.95) * cop_c - (next_c_soc - (1 - self.CF_C) * now_c_soc) * self.C_qr_est[uid]
                    clip_action = True
                    h_load = max(0, h_load)
                    c_load = max(0, c_load)

                if a_clip_c > now_action_c and now_action_c < 0:
                    c_load_new = -(next_c_soc - (1 - self.CF_C) * now_c_soc) * self.C_qr_est[uid]
                    h_load_new = (now_solar + next_elec_con - now_elec_dem
                        - (next_b_soc - (1 - self.CF_B) * now_b_soc)
                        * self.capacity_b[uid] * 0.95) * effi_h - (next_h_soc - (1 - self.CF_H) * now_h_soc) * self.H_qr_est[uid]

                    if clip_action is True:
                        c_load = c_load * 0.5 + c_load_new * 0.5
                        h_load = h_load * 0.5 + h_load_new * 0.5
                    else:
                        c_load = c_load_new
                        h_load = h_load_new
                    clip_action = True
                    h_load = max(0, h_load)
                    c_load = max(0, c_load)

                if clip_action is False:    #neither is clipped
                    h_load = -min(0, a_clip_h) * self.H_qr_est[uid]
                    c_load = (now_solar + next_elec_con - now_elec_dem
                        - (next_b_soc - (1 - self.CF_B) * now_b_soc)
                        * self.capacity_b[uid] * 0.95) * cop_c - a_clip_c * self.C_qr_est[uid]
                    if daytype in [7]:
                        avg_cooling = self.c_weekend1_avg[uid][time-1]
                    elif daytype in [1, 8]:
                        avg_cooling = self.c_weekend2_avg[uid][time-1]
                    else:
                        avg_cooling = self.c_weekday_avg[uid][time-1]
                    c_load = min(avg_cooling, c_load)

                ratio = 0.8 if clip_action is True else 0.05
                if daytype in [7]:
                    self.h_weekend1_avg[uid][time-1] = max(0, self.h_weekend1_avg[uid][time-1] * (1-ratio) + h_load * ratio)
                    self.c_weekend1_avg[uid][time-1] = max(0, self.c_weekend1_avg[uid][time-1] * (1-ratio) + c_load * ratio)
                elif daytype in [1, 8]:
                    self.h_weekend2_avg[uid][time-1] = max(0, self.h_weekend2_avg[uid][time-1] * (1-ratio) + h_load * ratio)
                    self.c_weekend2_avg[uid][time-1] = max(0, self.c_weekend2_avg[uid][time-1] * (1-ratio) + c_load * ratio)
                else:
                    self.h_weekday_avg[uid][time-1] = max(0, self.h_weekday_avg[uid][time-1] * (1-ratio) + h_load * ratio)
                    self.c_weekday_avg[uid][time-1] = max(0, self.c_weekday_avg[uid][time-1] * (1-ratio) + c_load * ratio)

                # ------------use average-------------
                if daytype in [7]:
                    est_h_load[uid] = self.h_weekend1_avg[uid]
                    est_c_load[uid] = self.c_weekend1_avg[uid]
                elif daytype in [1, 8]:
                    est_h_load[uid] = self.h_weekend2_avg[uid]
                    est_c_load[uid] = self.c_weekend2_avg[uid]
                else:
                    est_h_load[uid] = self.h_weekday_avg[uid]
                    est_c_load[uid] = self.c_weekday_avg[uid]
                # ------------scaling maximal n hours--------------
                index_sort_h = np.argsort(est_h_load[uid])
                index_sort_c = np.argsort(est_c_load[uid])

                for ind in range(-3, 0):
                    index_h = index_sort_h[ind]
                    index_c = index_sort_c[ind]
                    est_h_load[uid][index_h] = est_h_load[uid][index_h] * 1
                    est_c_load[uid][index_c] = est_c_load[uid][index_c] * 1

                adaptive_h_load[uid] = est_h_load[uid][T - window :]
                adaptive_c_load[uid] = est_c_load[uid][T - window :]
        """

        return adaptive_h_load, adaptive_c_load

    def calculate_avg_h_c(self):
        for uid in self.building_ids:
            self.h_weekday_avg[uid][:] = self.annual_h_demand[uid] / (365 * 24)
            self.h_weekend1_avg[uid][:] = self.annual_h_demand[uid] / (365 * 24)
            self.h_weekend2_avg[uid][:] = self.annual_h_demand[uid] / (365 * 24)
            self.c_weekday_avg[uid][:] = self.annual_c_demand[uid] / (365 * 24)
            self.c_weekend1_avg[uid][:] = self.annual_c_demand[uid] / (365 * 24)
            self.c_weekend2_avg[uid][:] = self.annual_c_demand[uid] / (365 * 24)

    def select_action(self, timestep: int):
        """interface w/ TD3.py to select actions in online exploration period"""
        RBC_THRESHOLD = 336
        self.timestep = timestep
        sign = False
        # if self.E_day is False and self.timestep % 22 == 0:
        #     self.C_day = not self.C_day
        #     self.H_day = not self.H_day

        if self.timestep == RBC_THRESHOLD - 1:
            for uid in self.building_ids:
                if self.ratio_h_est[uid] and self.H_bd_est[uid]:
                    pass
                else:
                    self.ratio_h_est[uid].append([0])
                    self.H_bd_est[uid].append([0])
            self.quantile_reg()

        if self.E_day is False and self.timestep % 24 == 22:
            self.H_day = not self.H_day
            self.C_day = not self.C_day

        if self.E_day is True:
            action, cap_bat, effi, nominal_p, add_points = self.estimate_bat()
            for uid in self.building_ids:
                if add_points[uid] is True:
                    self.num_elec_points[uid] += 1
                    self.effi_b[uid] = effi[uid] if effi[uid] != 0 else self.effi_b[uid]
                    self.cap_b_est[uid].append(cap_bat[uid] * self.effi_b[uid])
                if nominal_p[uid] > 0:
                    self.nominal_b[uid].append(nominal_p[uid])
                self.avail_nominal[uid] = (
                    True if self.num_elec_points[uid] >= 1 else False
                )
                if len(self.nominal_b[uid]) < 3 and sign is True:
                    sign = False
                elif len(self.nominal_b[uid]) >= 3:
                    sign = True
                sign = True if timestep >= 48 else sign
            if sign is True:
                self.E_day = False
                self.H_day, self.C_day = False, True
                for uid in self.building_ids:
                    """below two are parameters to be configured in predictor"""
                    self.nom_p_est[uid] = (
                        max(self.nominal_b[uid])
                        if self.nominal_b[uid]
                        else self.cap_b_est[uid][0] * 0.75
                    )
                    self.capacity_b[uid] = self.cap_b_est[uid][0]
                    # print(uid, "Bat: ", self.capacity_b[uid], ", Nominal P: ", self.nom_p_est[uid])
                # print("real nominal power: 75 40 20 30 25 10 15 10 20")

        elif self.C_day is True:
            action, cap_c, add_points, ratio_c, C_bd = self.estimate_c()
            for uid in self.building_ids:
                if add_points[uid] == 1:
                    if cap_c[uid] > 0:
                        self.num_c_points[uid] += add_points[uid]
                        self.cap_c_est[uid].append(cap_c[uid])
                        # ratio_c_est[uid].append(ratio_c[uid])
                        self.ratio_c_est[uid].append(ratio_c[uid])  # Two point avg
                        self.C_bd_est[uid].append(C_bd[uid])
                sign = self.num_c_points[uid] >= 7
            if sign is True:
                pass
                # self.H_day, self.C_day = True, False
                # for uid in self.building_ids:
                #     print(uid, "C:", self.cap_c_est[uid])

        elif self.H_day is True:
            action, cap_h, add_points, ratio_h, H_bd = self.estimate_h()
            sign = False
            for uid in self.building_ids:
                if add_points[uid] == 1:
                    if cap_h[uid] > 0:
                        self.num_h_points[uid] += add_points[uid]
                        self.cap_h_est[uid].append(cap_h[uid])
                        # ratio_h_est[uid].append(ratio_h[uid])
                        self.ratio_h_est[uid].append(ratio_h[uid])  # Two point avg
                        self.H_bd_est[uid].append(H_bd[uid])
                        # self.H_day, self.C_day = False, True
                sign = True if self.num_h_points[uid] >= 9 else False
            if sign is True:
                pass
                # self.H_day, self.C_day = False, False
                # for uid in self.building_ids:
                #     print(uid, "H:", self.cap_h_est[uid])
        else:
            raise TypeError("No energy type is configured")

        return action

    def get_params(self, timestep: int) -> dict:
        """return estimated params from online exploration"""
        assert timestep >= self.rbc_threshold, ValueError(
            "online exploration is still running"
        )

        param_dict = {}
        param_dict["C_p_Csto"] = np.array(list(self.C_qr_est.values()))
        param_dict["C_p_Hsto"] = np.array(list(self.H_qr_est.values()))
        param_dict["C_p_bat"] = np.array(list(self.capacity_b.values()))
        param_dict["E_bat_max"] = np.array(list(self.nom_p_est.values()))

        return param_dict

    def quantile_reg(self):
        """quantile regression for h/c capacity estimation--convert to DataFrame"""
        df_cap_h = pd.DataFrame(
            {key: pd.Series(value) for key, value in self.cap_h_est.items()}
        )  # conversion of dictionary to dataframe with different length
        df_cap_h.columns = [
            "Building_" + str(uid) + "_cap_h" for uid in self.building_ids
        ]
        # print("cap_h dataframe:\n",df_cap_h)

        df_H_bd = pd.DataFrame(
            {key: pd.Series(value) for key, value in self.H_bd_est.items()}
        )  # conversion of dictionary to dataframe with different length
        df_H_bd.columns = [
            "Building_" + str(uid) + "_H_bd" for uid in self.building_ids
        ]
        # print("H_bd dataframe:\n",df_H_bd)
        df_ratio_h = pd.DataFrame(
            {key: pd.Series(value) for key, value in self.ratio_h_est.items()}
        )  # conversion of dictionary to dataframe with different length
        df_ratio_h.columns = [
            "Building_" + str(uid) + "_ratio_h" for uid in self.building_ids
        ]
        # print("ratio H dataframe:\n",df_ratio_h)

        H_dataframe = pd.concat(
            [df_cap_h, df_H_bd, df_ratio_h], axis=1
        )  # Concatenation of H relevant data
        # print(H_dataframe)

        ################################  Conversion of dicionary to dataframe //// Cooling
        # my_array = np.array([cap_h_est])
        df_cap_c = pd.DataFrame(
            {key: pd.Series(value) for key, value in self.cap_c_est.items()}
        )  # conversion of dictionary to dataframe with different length
        df_cap_c.columns = [
            "Building_" + str(uid) + "_cap_c" for uid in self.building_ids
        ]
        # print("cap_c dataframe:\n",df_cap_c)

        df_C_bd = pd.DataFrame(
            {key: pd.Series(value) for key, value in self.C_bd_est.items()}
        )  # conversion of dictionary to dataframe with different length
        df_C_bd.columns = [
            "Building_" + str(uid) + "_C_bd" for uid in self.building_ids
        ]
        # print("C_bd dataframe:\n",df_H_bd)
        df_ratio_c = pd.DataFrame(
            {key: pd.Series(value) for key, value in self.ratio_c_est.items()}
        )  # conversion of dictionary to dataframe with different length
        df_ratio_c.columns = [
            "Building_" + str(uid) + "_ratio_c" for uid in self.building_ids
        ]
        # print("ratio C dataframe:\n",df_ratio_h)

        C_dataframe = pd.concat(
            [df_cap_c, df_C_bd, df_ratio_c], axis=1
        )  # Concatenation of H relevant data
        # print(C_dataframe)

        ################################ dataframe to array (both Heat and Cooling) to use the index

        ## Quantile regession %
        quantiles = [0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95]

        for uid in self.building_ids:  # j is building id
            self.quantile_reg_H(uid, quantiles, H_dataframe, 5)
            self.quantile_reg_C(uid, quantiles, C_dataframe, 5)

    def quantile_reg_H(self, uid, quantiles, H_dataframe, climate_zone):
        """quantile regression for H capacity"""
        ############### Qunatile Regression
        if self.has_heating[uid] is False:
            self.H_qr_est[uid] = 1e-5
        else:
            mod = smf.quantreg(
                " Building_"
                + str(uid)
                + "_H_bd ~ Building_"
                + str(uid)
                + "_ratio_h -1",
                H_dataframe,
            )  # -1 means no intercept
            models = []
            for qr in quantiles:
                res = mod.fit(q=qr)
                models.append(
                    [qr, res.params["Building_" + str(uid) + "_ratio_h"]]
                    + res.conf_int().loc["Building_" + str(uid) + "_ratio_h"].tolist()
                )

            models = pd.DataFrame(
                models,
                columns=[
                    "quantile reg %",
                    "coefficient",
                    "lower bound(coeff)",
                    "upper bound(coeff)",
                ],
            )

            ord_least_sq = smf.ols(
                " Building_" + str(uid) + "_H_bd~ Building_" + str(uid) + "_ratio_h -1",
                H_dataframe,
            ).fit()  # ordinary least square
            ord_least_sq_ci = (
                ord_least_sq.conf_int()
                .loc["Building_" + str(uid) + "_ratio_h"]
                .tolist()
            )
            ord_least_sq = dict(
                y_interc=0,  # y-intercept
                coeff=ord_least_sq.params["Building_" + str(uid) + "_ratio_h"],  # slope
                lower_bound=ord_least_sq_ci[0],
                upper_bound=ord_least_sq_ci[1],
            )

            # for uid in building_ids
            ##### Quantile regression % assignment to each building for better capacity estimation based on observation
            self.H_qr_est[uid] = (
                models["coefficient"][3] + models["coefficient"][4]
            ) / 2

    def quantile_reg_C(self, uid, quantiles, C_dataframe, climate_zone):
        """quantile regression for C capacity"""
        mod = smf.quantreg(
            " Building_" + str(uid) + "_C_bd ~ Building_" + str(uid) + "_ratio_c -1",
            C_dataframe,
        )  # -1 means no intercept = intercept at 0
        # quantiles = [0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95]
        models = []
        for qr in quantiles:
            res = mod.fit(q=qr)
            models.append(
                [qr, res.params["Building_" + str(uid) + "_ratio_c"]]
                + res.conf_int().loc["Building_" + str(uid) + "_ratio_c"].tolist()
            )
        # models = [self.fit_model_c(mod, qr, uid) for qr in quantiles]
        models = pd.DataFrame(
            models,
            columns=[
                "quantile reg %",
                "coefficient",
                "lower bound (coeff)",
                "upper bound(coeff)",
            ],
        )

        ord_least_sq = smf.ols(
            " Building_" + str(uid) + "_C_bd~ Building_" + str(uid) + "_ratio_c -1",
            C_dataframe,
        ).fit()  # ordinary least square
        ord_least_sq_ci = (
            ord_least_sq.conf_int().loc["Building_" + str(uid) + "_ratio_c"].tolist()
        )
        ord_least_sq = dict(
            y_interc=0,  # y-intercept
            coeff=ord_least_sq.params["Building_" + str(uid) + "_ratio_c"],  # slope
            lower_bound=ord_least_sq_ci[0],
            upper_bound=ord_least_sq_ci[1],
        )

        ##### Quantile regression % assignment to each building for better capacity estimation based on observation
        # if uid in self.building_ids:
        self.C_qr_est[uid] = (models["coefficient"][1] + models["coefficient"][2]) / 2

    def estimate_h(self):
        """choose actions for estimating h capacity"""
        action_gen = []
        # e_wh = {uid: 0 for uid in building_ids}
        # a_b = {uid: 0 for uid in building_ids} if a_b is None else a_b
        # a_h = {uid: 0 for uid in building_ids} if a_h is None else a_h
        a_clip = {uid: 0 for uid in self.building_ids}
        add_points = {uid: 0 for uid in self.building_ids}
        cap_h = {uid: 0 for uid in self.building_ids}
        effi = {uid: 0 for uid in self.building_ids}
        a_b = {uid: 0 for uid in self.building_ids}
        a_h = {uid: 0 for uid in self.building_ids}
        a_c = {uid: 0 for uid in self.building_ids}
        ratio_h = {uid: 0 for uid in self.building_ids}
        H_bd = {uid: 0 for uid in self.building_ids}

        effi_h = 0.9

        CF_C = 0.006
        CF_H = 0.008
        CF_B = 0

        state = self.state_buffer
        action = self.action_buffer

        if self.timestep % 24 in [0]:
            prev_soc_c = state.get(-2)["soc_c"][-1]  # shape(9)
            prev_soc_h = state.get(-2)["soc_h"][-1]
            prev_soc_b = state.get(-2)["soc_b"][-1]
            prev_solar_gen = state.get(-2)["solar_gen"][-1]
            prev_elec_dem = state.get(-2)["elec_dem"][-1]
            prev_temp = state.get(-2)["t_out"][-1]
            # prev_action = action.get(-1)["action_bat"][-1]
            # now_soc_c = state.get(-1)["soc_c"][-1]
            # now_soc_h = state.get(-1)["soc_h"][-1]
            # now_soc_b = state.get(-1)["soc_b"][-1]
            # now_elec_con = state.get(-1)["elec_cons"][-1]
        else:  # -2 index means timestep % 24 - 1
            prev_soc_c = state.get(-1)["soc_c"][-2]
            prev_soc_h = state.get(-1)["soc_h"][-2]
            prev_soc_b = state.get(-1)["soc_b"][-2]
            prev_solar_gen = state.get(-1)["solar_gen"][-2]
            prev_elec_dem = state.get(-1)["elec_dem"][-2]
            prev_temp = state.get(-1)["t_out"][-2]

        if self.timestep % 24 in [0]:
            prev_2_soc_h = state.get(-2)["soc_h"][-2]
        elif self.timestep % 24 in [1]:
            prev_2_soc_h = state.get(-2)["soc_h"][-1]
        else:
            prev_2_soc_h = state.get(-1)["soc_h"][-3]

        prev_action = action.get(-1)["action_H"][-1]
        now_soc_c = state.get(-1)["soc_c"][-1]
        now_soc_h = state.get(-1)["soc_h"][-1]
        now_soc_b = state.get(-1)["soc_b"][-1]
        now_elec_con = state.get(-1)["elec_cons"][-1]

        prev_cop = self.cop_cal(prev_temp[0])

        """ params over one step: e_wh, a_h """
        for uid in self.building_ids:
            if prev_action[uid] > 0 and now_soc_h[uid] == 0:
                self.has_heating[uid] = False
            if self.prev_hour_est_h[uid] is True or self.avail_ratio_est_h[uid] is True:
                # --------------update tau_plus if not satisfied-----------
                # ##########
                # if avail_ratio_est_h[uid] is True and soc_h[uid][-1] == 0:
                #     tau_h[uid] = min(tau_h[uid] + 0.1, 0.8)
                #     action_h[uid] = min(action_h[uid] + 0.05, 0.5)
                #     tau_hplus[uid] = max(0, min(0.8 - tau_h[uid],
                #                                 max(tau_hplus[uid], -(soc_h[uid][-1] - (1 - CF_H) * soc_h[uid][-2]) / 1.1)))
                # #########
                if self.avail_ratio_est_h[uid] is True and now_soc_c[uid] < 0.01:
                    self.tau_c[uid] = min(self.tau_c[uid] + 0.1, 0.8)
                    self.action_c[uid] = min(self.action_c[uid] + 0.05, 0.5)
                    self.tau_cplus[uid] = max(
                        0,
                        min(
                            0.8 - self.tau_c[uid],
                            max(
                                self.tau_cplus[uid],
                                -(now_soc_c[uid] - (1 - CF_C) * prev_soc_c[uid]),
                            ),
                        ),
                    )
                    self.avail_ratio_est_h[uid], self.prev_hour_est_h[uid] = (
                        False,
                        False,
                    )
                    # two_points_avg[uid] = False

                if (
                    self.avail_ratio_est_h[uid] is True
                    and self.has_heating[uid] is True
                    and now_soc_h[uid] < 0.01
                ):
                    self.tau_h[uid] = min(self.tau_h[uid] + 0.1, 0.8)
                    self.action_h[uid] = min(self.action_h[uid] + 0.05, 0.5)
                    # tau_hplus = max(0, min(0.8-tau_h, max(tau_hplus, -(soc_h[uid][-1] - (1-CF_H)*soc_h[uid][-2])/1.1)))
                    self.avail_ratio_est_h[uid], self.prev_hour_est_h[uid] = (
                        False,
                        False,
                    )
                    # two_points_avg[uid] = False
                if self.timestep % 24 == 22:
                    self.avail_ratio_est_h[uid], self.prev_hour_est_h[uid] = (
                        False,
                        False,
                    )
                # --------------estimate e_wh if avail---------------------
            if (
                self.prev_hour_est_h[uid] is True
                and self.avail_ratio_est_h[uid] is True
            ):  # calculate capacity here
                e_wh = (
                    prev_solar_gen[uid] + now_elec_con[uid] - prev_elec_dem[uid]
                ) - (
                    (now_soc_b[uid] - (1 - CF_B) * prev_soc_b[uid])
                    * self.capacity_b[uid]
                    / self.effi_b[uid]
                )

                # if two_points_est[uid] is True:
                self.avail_ratio_est_h[uid], self.prev_hour_est_h[uid] = False, False
                ratio_h[uid] = -(prev_soc_h[uid] - (1 - CF_H) * prev_2_soc_h[uid])
                # ratio_h2[uid] = -(soc_h[uid][-4] - (1 - CF_H) * soc_h[uid][-5])
                # two_point_est_h[uid] = a_h_temp[uid][-1] + (ratio_h[uid] + ratio_h2[uid]) / 2
                cap_h[uid] = e_wh * effi_h / (prev_action[uid] + ratio_h[uid])
                ratio_h[uid] = prev_action[uid] + ratio_h[uid]
                # H_bd[uid] = e_wh * effi_h
                # H_bd[uid] = e_wh * effi_h - (soc_h[uid][-1] - (1 - CF_H) * soc_h[uid][-2])
                H_bd[uid] = e_wh * effi_h
                #######

                add_points[uid] += 1

                """true_val_h = [10.68, 49.35, 1e-05, 1e-05, 60.12, 105.12, 85.44, 111.96, 102.24]"""

            """
            1) execute action to est ratio if soc > threshold  
            2) observe soc and calculate ratio if satisfying requirements, and execute action for e_wh calculation
            3) observe params and calculate e_wh, and execute action to est ratio if soc > threshold
            4) observe soc and est capacity. restart the procedure
            5) if # of est points is enough, end up est.
            """
            a_b[uid] = 0.03
            if now_soc_h[uid] < self.tau_h[uid]:
                a_h[uid] = self.action_h[uid]
            elif now_soc_h[uid] > 0.9:
                # elif soc_c[uid][-1] > 0.9:
                a_h[uid] = -0.2
            else:
                a_h[uid] = 0.04

            a_c[uid] = (
                self.action_c[uid]
                if now_soc_c[uid] < self.tau_c[uid] + self.tau_cplus[uid]
                else 0.1
            )

            if (
                now_soc_h[uid] >= self.tau_h[uid]
                and self.avail_ratio_est_h[uid] is False
            ):
                if now_soc_c[uid] >= self.tau_c[uid] + self.tau_cplus[uid]:
                    a_c[uid], a_h[uid] = -1, -1
                    a_b[uid] = 0.03  ## added
                    # action_now = [a_h, a_h, a_b[uid]]
                    self.avail_ratio_est_h[uid] = True

                # action.append(action_now)   # exit
            elif self.avail_ratio_est_h[uid] is True:
                if now_soc_c[uid] < self.tau_c[uid]:
                    self.tau_cplus[uid] = max(
                        0,
                        min(
                            0.8 - self.tau_c[uid],
                            max(
                                self.tau_cplus[uid] + 0.1,
                                -(now_soc_c[uid] - (1 - CF_C) * prev_soc_c[uid]),
                            ),
                        ),
                    )
                # if soc_h[uid][-1] < tau_h[uid]:
                #     tau_hplus[uid] = max(0, min(0.8 - tau_h[uid],
                #                                 max(tau_hplus[uid] + 0.1,
                #                                     -(soc_h[uid][-1] - (1 - CF_H) * soc_h[uid][-2]) / 1.1)))
                a_b[uid], a_c[uid] = 0.03, -1
                a_clip[uid] = -(
                    now_soc_h[uid] - (1 - CF_H) * prev_soc_h[uid]
                )  # calculate a_t here
                if now_soc_h[uid] >= 0.97:
                    a_h[uid] = -1
                    action_now = [a_c[uid], a_h[uid], a_b[uid]]
                    action_gen.append(action_now)
                    continue
                elif a_clip[uid] > 0.01:
                    a_h[uid] = min(a_clip[uid], 1 - now_soc_h[uid])
                else:
                    a_h[uid] = min(0.03, 1 - now_soc_h[uid])
                    # a_h[uid] = min(0.03, 1 - soc_c[uid][-1])
                a_c[uid] = -1
                self.prev_hour_est_h[uid] = True
            action_now = [a_c[uid], a_h[uid], a_b[uid]]
            action_gen.append(action_now)  # exit
        # print(action)
        # sys.exit()

        return action_gen, cap_h, add_points, ratio_h, H_bd

    def estimate_c(self):
        """choose actions for estimating c capacity"""
        action_gen = []
        # e_hpc = {uid: 0 for uid in building_ids}
        # a_b = {uid: 0 for uid in building_ids} if a_b is None else a_b
        # a_c = {uid: 0 for uid in building_ids} if a_c is None else a_c
        a_clip = {uid: 0 for uid in self.building_ids}
        add_points = {uid: 0 for uid in self.building_ids}
        cap_c = {uid: 0 for uid in self.building_ids}
        # effi = {uid: 0 for uid in self.building_ids}
        a_b = {uid: 0 for uid in self.building_ids}
        a_c = {uid: 0 for uid in self.building_ids}
        ratio_c = {uid: 0 for uid in self.building_ids}
        C_bd = {uid: 0 for uid in self.building_ids}
        CF_C = 0.006
        CF_H = 0.008
        CF_B = 0

        state = self.state_buffer
        action = self.action_buffer

        if self.timestep % 24 in [0]:
            prev_soc_c = state.get(-2)["soc_c"][-1]  # shape(9)
            prev_soc_h = state.get(-2)["soc_h"][-1]
            prev_soc_b = state.get(-2)["soc_b"][-1]
            prev_solar_gen = state.get(-2)["solar_gen"][-1]
            prev_elec_dem = state.get(-2)["elec_dem"][-1]
            prev_temp = state.get(-2)["t_out"][-1]
            # prev_action = action.get(-1)["action_bat"][-1]
            # now_soc_c = state.get(-1)["soc_c"][-1]
            # now_soc_h = state.get(-1)["soc_h"][-1]
            # now_soc_b = state.get(-1)["soc_b"][-1]
            # now_elec_con = state.get(-1)["elec_cons"][-1]
        else:  # -2 index means timestep % 24 - 1
            prev_soc_c = state.get(-1)["soc_c"][-2]
            prev_soc_h = state.get(-1)["soc_h"][-2]
            prev_soc_b = state.get(-1)["soc_b"][-2]
            prev_solar_gen = state.get(-1)["solar_gen"][-2]
            prev_elec_dem = state.get(-1)["elec_dem"][-2]
            prev_temp = state.get(-1)["t_out"][-2]

        if self.timestep % 24 in [0]:
            prev_2_soc_c = state.get(-2)["soc_c"][-2]
        elif self.timestep % 24 in [1]:
            prev_2_soc_c = state.get(-2)["soc_c"][-1]
        else:
            prev_2_soc_c = state.get(-1)["soc_c"][-3]

        prev_action = action.get(-1)["action_C"][-1]
        prev_action_h = action.get(-1)["action_H"][-1]
        now_soc_c = state.get(-1)["soc_c"][-1]
        now_soc_h = state.get(-1)["soc_h"][-1]
        now_soc_b = state.get(-1)["soc_b"][-1]
        now_elec_con = state.get(-1)["elec_cons"][-1]

        prev_cop = self.cop_cal(prev_temp[0])
        """ params over one step: e_hpc, a_c """
        for uid in self.building_ids:
            if prev_action_h[uid] > 0 and now_soc_h[uid] == 0:
                self.has_heating[uid] = False
            if self.prev_hour_est_c[uid] is True or self.avail_ratio_est_c[uid] is True:
                # --------------update tau_plus if not satisfied-----------
                if self.avail_ratio_est_c[uid] is True and now_soc_c[uid] < 0.01:
                    self.tau_c[uid] = min(self.tau_c[uid] + 0.1, 0.8)
                    self.action_c[uid] = min(self.action_c[uid] + 0.05, 0.5)
                    # tau_cplus = max(0, min(0.8-tau_c, max(tau_cplus, -(soc_c[uid][-1] - (1-CF_C)*soc_c[uid][-2])/1.1)))
                    self.avail_ratio_est_c[uid], self.prev_hour_est_c[uid] = (
                        False,
                        False,
                    )
                    # two_points_avg[uid] = False

                if (
                    self.avail_ratio_est_c[uid] is True
                    and self.has_heating[uid] is True
                    and now_soc_h[uid] < 0.01
                ):
                    self.tau_h[uid] = min(self.tau_h[uid] + 0.1, 0.8)
                    self.action_h[uid] = min(self.action_h[uid] + 0.05, 0.5)
                    self.tau_hplus[uid] = max(
                        0,
                        min(
                            0.8 - self.tau_h[uid],
                            max(
                                self.tau_hplus[uid],
                                -(now_soc_h[uid] - (1 - CF_H) * prev_soc_h[uid]),
                            ),
                        ),
                    )
                    self.avail_ratio_est_c[uid], self.prev_hour_est_c[uid] = (
                        False,
                        False,
                    )
                    # two_points_avg[uid] = False
                if self.timestep % 24 == 22:
                    self.avail_ratio_est_c[uid], self.prev_hour_est_c[uid] = (
                        False,
                        False,
                    )

                # --------------estimate e_hpc if avail---------------------
            if (
                self.prev_hour_est_c[uid] is True
                and self.avail_ratio_est_c[uid] is True
            ):  # calculate capacity here
                e_hpc = (
                    prev_solar_gen[uid]
                    + now_elec_con[uid]
                    - prev_elec_dem[uid]
                    - (now_soc_b[uid] - (1 - CF_B) * prev_soc_b[uid])
                    * self.capacity_b[uid]
                    / self.effi_b[uid]
                )
                # if two_points_est[uid] is True:
                self.avail_ratio_est_c[uid], self.prev_hour_est_c[uid] = False, False
                ratio_c[uid] = -(prev_soc_c[uid] - (1 - CF_C) * prev_2_soc_c[uid])
                # self.ratio_c2[uid] = -(soc_c[uid][-4] - (1 - CF_C) * soc_c[uid][-5])
                # two_point_est_c[uid] =  a_c_temp[uid][-1] + (ratio_c[uid]+ratio_c2[uid]/2)

                ratio_c[uid] = ratio_c[uid] + prev_action[uid]
                cap_c[uid] = e_hpc * prev_cop / (ratio_c[uid])

                C_bd[uid] = e_hpc * prev_cop

                add_points[uid] += 1
                if C_bd[uid] < 0:
                    pass

            """
            1) execute action to est ratio if soc > threshold  
            2) observe soc and calculate ratio if satisfying requirements, and execute action for e_hpc calculation
            3) observe params and calculate e_hpc, and execute action to est ratio if soc > threshold
            4) observe soc and est capacity. restart the procedure
            5) if # of est points is enough, end up est.
            """
            a_b[uid] = 0.03
            if now_soc_c[uid] < self.tau_c[uid]:
                a_c[uid] = self.action_c[uid]
            elif now_soc_c[uid] > 0.9:
                a_c[uid] = -0.2
            else:
                a_c[uid] = 0.04

            if self.has_heating[uid] is True:
                a_h = (
                    self.action_h[uid]
                    if now_soc_h[uid] < self.tau_h[uid] + self.tau_hplus[uid]
                    else 0.1
                )
            else:
                a_h = 0

            if (
                now_soc_c[uid] >= self.tau_c[uid]
                and self.avail_ratio_est_c[uid] is False
            ):
                if self.has_heating[uid] is True and now_soc_h[uid] >= (
                    self.tau_h[uid] + self.tau_hplus[uid]
                ):
                    a_c[uid], a_h = -1, -1
                    # action_now = [a_c, a_h, a_b[uid]]
                    self.avail_ratio_est_c[uid] = True

                if self.has_heating[uid] is False:
                    a_c[uid] = -1
                    # action_now = [a_c, a_b[uid]]
                    self.avail_ratio_est_c[uid] = True
                # action.append(action_now)   # exit
            elif self.avail_ratio_est_c[uid] is True:
                if self.has_heating[uid] is True and now_soc_h[uid] < self.tau_h[uid]:
                    self.tau_hplus[uid] = max(
                        0,
                        min(
                            0.8 - self.tau_h[uid],
                            max(
                                self.tau_hplus[uid] + 0.1,
                                -(now_soc_h[uid] - (1 - CF_H) * prev_soc_h[uid]) / 1.1,
                            ),
                        ),
                    )
                a_b[uid], a_h = 0.03, -1

                a_clip[uid] = (
                    -(now_soc_c[uid] - (1 - CF_C) * prev_soc_c[uid]) / 1.1
                )  # calculate a_t here
                if now_soc_c[uid] >= 0.97:
                    a_c[uid] = -1
                    action_now = [a_c[uid], a_h, a_b[uid]]
                    action_gen.append(action_now)
                    continue
                elif a_clip[uid] > 0.03:
                    a_c[uid] = min(a_clip[uid], 1 - now_soc_c[uid])
                else:
                    a_c[uid] = min(0.03, 1 - now_soc_c[uid])
                self.prev_hour_est_c[uid] = True
            action_now = [a_c[uid], a_h, a_b[uid]]
            action_gen.append(action_now)  # exit

        return action_gen, cap_c, add_points, ratio_c, C_bd

    def estimate_bat(self):
        """choose actions for estimating bat capacity"""
        add_points = {uid: False for uid in self.building_ids}
        cap_bat = {uid: 0 for uid in self.building_ids}
        action_gen = []
        a_clip = {uid: 0 for uid in self.building_ids}
        e_bat = {uid: 0 for uid in self.building_ids}
        effi = {uid: 0 for uid in self.building_ids}
        nominal_p = {uid: 0 for uid in self.building_ids}
        CF_B = 0

        state = self.state_buffer
        action = self.action_buffer

        if self.timestep % 24 in [0] and self.timestep not in [0]:
            prev_soc_c = state.get(-2)["soc_c"][-1]  # shape(9)
            prev_soc_h = state.get(-2)["soc_h"][-1]
            prev_soc_b = state.get(-2)["soc_b"][-1]
            prev_solar_gen = state.get(-2)["solar_gen"][-1]
            prev_elec_dem = state.get(-2)["elec_dem"][-1]
            # prev_action = action.get(-1)["action_bat"][-1]
            # now_soc_c = state.get(-1)["soc_c"][-1]
            # now_soc_h = state.get(-1)["soc_h"][-1]
            # now_soc_b = state.get(-1)["soc_b"][-1]
            # now_elec_con = state.get(-1)["elec_cons"][-1]
        elif self.timestep not in [0]:  # -2 index means timestep % 24 - 1
            prev_soc_c = state.get(-1)["soc_c"][-2]
            prev_soc_h = state.get(-1)["soc_h"][-2]
            prev_soc_b = state.get(-1)["soc_b"][-2]
            prev_solar_gen = state.get(-1)["solar_gen"][-2]
            prev_elec_dem = state.get(-1)["elec_dem"][-2]
        prev_action = action.get(-1)["action_bat"][-1]
        prev_action_h = action.get(-1)["action_H"][-1]
        now_soc_c = state.get(-1)["soc_c"][-1]
        now_soc_h = state.get(-1)["soc_h"][-1]
        now_soc_b = state.get(-1)["soc_b"][-1]
        now_elec_con = state.get(-1)["elec_cons"][-1]

        for uid in self.building_ids:
            if prev_action_h[uid] > 0 and now_soc_h[uid] == 0:
                self.has_heating[uid] = False
            if self.prev_hour_est_b[uid] is True or self.prev_hour_nom[uid] is True:
                if now_soc_c[uid] < 0.01:
                    self.tau_c[uid] = min(self.tau_c[uid] + 0.1, 0.7)
                    self.action_c[uid] = min(self.action_c[uid] + 0.05, 0.5)
                    self.prev_hour_est_b[uid] = False
                    self.prev_hour_nom[uid] = False

                if self.has_heating[uid] is True and now_soc_h[uid] < 0.01:
                    self.tau_h[uid] = min(self.tau_h[uid] + 0.1, 0.7)
                    self.action_h[uid] = min(self.action_h[uid] + 0.05, 0.5)
                    self.prev_hour_est_b[uid] = False
                    self.prev_hour_nom[uid] = False

                if self.prev_hour_nom[uid] is True and now_soc_b[uid] < 0.01:
                    self.tau_b[uid] = min(self.tau_b[uid] + 0.1, 0.8)
                    self.prev_hour_nom[uid] = False

                if self.timestep % 24 == 22:
                    self.prev_hour_est_b[uid] = False
                    self.prev_hour_nom[uid] = False

            if self.prev_hour_est_b[uid] is True:
                # soc_1 = soc_b[uid][-2], now replaced by prev_soc_b
                # soc_2 = soc_b[uid][-1], now replaced by now_soc_b

                a_clip[uid] = now_soc_b[uid] - (1 - CF_B) * prev_soc_b[uid]
                e_bat[uid] = (
                    prev_solar_gen[uid] + now_elec_con[uid] - prev_elec_dem[uid]
                )
                cap_bat[uid] = e_bat[uid] / a_clip[uid]
                effi[uid] = a_clip[uid] / prev_action[uid]
                add_points[uid] = True
                # prev_hour_est[uid] = False

            elif self.prev_hour_nom[uid] is True:
                # soc_1 = soc_b[uid][-2], now replaced by prev_soc_b
                # soc_2 = soc_b[uid][-1], now replaced by now_soc_b

                a_clip[uid] = now_soc_b[uid] - (1 - CF_B) * prev_soc_b[uid]
                nominal_p[uid] = max(self.cap_b_est[uid]) * (-a_clip[uid])

                """reality:     75   40   20   30   25   10   15   10   20"""
                """estimated:   89.3 40.5 21.6 34.1 26.9 11.9 15.8 11.9 23.8"""
            # else:
            #     print("t elec %s: " % uid, soc_b[uid][-2])

            a_c = self.action_c[uid] if now_soc_c[uid] < self.tau_c[uid] else 0.02
            a_b = -0.2 if now_soc_b[uid] > 0.8 else 0.02
            if self.has_heating[uid] is True:
                a_h = self.action_h[uid] if now_soc_h[uid] < self.tau_h[uid] else 0.04
            else:
                a_h = 0

            if self.avail_nominal[uid] is True:
                if now_soc_b[uid] >= self.tau_b[uid]:
                    action_now = [-1, -1, -1]
                    self.prev_hour_nom[uid] = True
                    self.prev_hour_est_b[uid] = False
                else:
                    action_now = [a_c, a_h, 0.4]
                    self.prev_hour_nom[uid] = False
                    self.prev_hour_est_b[uid] = False
                action_gen.append(action_now)
                continue

            if (
                now_soc_c[uid] > self.tau_c[uid]
                and now_soc_b[uid] < 0.8
                and self.has_heating[uid] is False
                or now_soc_c[uid] > self.tau_c[uid]
                and now_soc_b[uid] < 0.8
                and now_soc_h[uid] > self.tau_h[uid]
                and self.has_heating[uid] is True
            ):
                action_now = [-1, -1, 0.05]
                self.prev_hour_est_b[uid] = True
            else:
                action_now = [a_c, a_h, a_b]
                self.prev_hour_est_b[uid] = False
            # print("action %s:" % uid, action_now, prev_hour_est[uid])
            action_gen.append(action_now)

        return action_gen, cap_bat, effi, nominal_p, add_points
