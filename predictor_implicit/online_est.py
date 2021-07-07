from citylearn import  CityLearn
from pathlib import Path
from agents.rbc import RBC
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, QuantileRegressor


def record_dic(s_dic):
    global soc_c, soc_h, soc_b, solar_gen, ns_load
    for uid in building_ids:
        t_out[uid].append(s_dic[uid]["t_out"])
        soc_c[uid].append(s_dic[uid]["cooling_storage_soc"])
        soc_b[uid].append(s_dic[uid]["electrical_storage_soc"])
        solar_gen[uid].append(s_dic[uid]["solar_gen"])
        ns_load[uid].append(s_dic[uid]["non_shiftable_load"])
        net_elec[uid].append(s_dic[uid]["net_electricity_consumption"])
        if uid not in building_no_heat:
            soc_h[uid].append(s_dic[uid]["dhw_storage_soc"])

def plot_cap_h(cap_all, climate, type="heat"):
    subplots = [331, 332, 333, 334, 335, 336, 337, 338, 339]
    true_val_h = [10.68, 49.35, 1e-05, 1e-05, 60.12, 105.12, 85.44, 111.96, 102.24]
    true_val_b = [140, 80, 50, 75, 50, 30, 40, 30, 35]
    true_val_c = [618.12, 227.37, 414.68, 383.565, 244.685, 96.87, 127.82, 165.45, 175.23]
    length = len(subplots)
    fig = plt.figure(figsize=[6.4 * 2.5, 6.4 * 2.5])
    for i in range(length):
        fig = plt.subplot(subplots[i])
        A = np.array(ratio_c[i, :])
        a = np.array(ratio_c[i, :]).reshape(-1, 1)  # reshaping to 50 x 1 to let them fit into .fit()
        b = np.array(C_bd[i, :]).reshape(-1, 1)
        y_normal = np.array(C_bd[i, :]).reshape(-1) ##??????????????
        ''' with numpy, Linear regression
        # A = np.reshape(a, (50, 1))
        # # A = np.vstack([a, np.ones(len(a))]).T
        # m, c = np.linalg.lstsq(A, b, rcond=None)[0]
        # print("m:", m, "c:", c)
        # _ = plt.plot(a, m * a + c, 'r', label='Fitted line')
        # _ = plt.legend()
        '''
        ########## with the sklearn, Lienar regression
        regr = linear_model.LinearRegression()
        regr.fit(a, b)
        print("y-intercept: ", regr.intercept_)
        print("coefficient: ", regr.coef_)
        linear_reg = np.array(regr.coef_ * a) # y-intercept = 0
        # print(a)
        # print(linear_reg)
        # _ = plt.plot(a, linear_reg, 'r', label='Fitted line')
        # _ = plt.legend()
        ########## with the sklearn, Qunatile regression
        rng = np.random.RandomState(42)  # random seed
        # print("linear regression:",linear_reg)
        # print("lr length",len(linear_reg))
        # print("rng",rng.normal(loc=0, scale=0.5 + 0.5*A, size=A.shape[0]))
        # print("rng length",len(rng.normal(loc=0, scale=0.5 + 0.5*A, size=A.shape[0])))
        # p = rng.normal(loc=0, scale=0.5 + 0.5 * a)
        # print("p:", p)
        # y_normal = np.array(linear_reg + rng.normal(loc=0, scale=0.5 + 0.5*a)).reshape(-1) ## why?
        # print("y normal:",y_normal)
        quantiles = [0.05, 0.4, 0.5, 0.6, 0.7, 0.95]
        predictions = {}
        out_bounds_predictions = np.zeros_like(linear_reg, dtype=np.bool_)
        ######## no error above
        for quantile in quantiles:
            qr = QuantileRegressor(quantile=quantile, alpha=0, fit_intercept=False)
            y_pred = qr.fit(a, y_normal).predict(a)
            param_pred = qr.get_params()
            predictions[quantile] = y_pred
            print("y_pred:", y_pred)
            print("param", param_pred)
            #

        # plt.scatter(a, linear_reg, color="black", label="Linear Regression")
        for quantile, y_pred in predictions.items():
            plt.scatter(a, y_pred , label=f"Quantile: {quantile}")
        print(out_bounds_predictions)
        # plt.scatter(
        #     a[out_bounds_predictions[i, :]],
        #     y_normal[out_bounds_predictions[i, :]],
        #     color="red",
        #     marker="+",
        #     alpha=0.5,
        #     label="Outside interval",
        # )
        # plt.scatter(
        #     a[~out_bounds_predictions[i, :]],
        #     y_normal[~out_bounds_predictions[i, :]],
        #     color="green",
        #     marker="v",
        #     alpha=0.5,
        #     label="Inside interval",
        # )
        plt.legend(fontsize=7)  # using a size in points
        ##########

        # plt.scatter(a, cap_h_all[i, :],label='Original data')
        plt.ylabel("Estimated Load")
        plt.xlabel('ratio (Load/Capacity)')
        if type == "heat":
            plt.title("Bdg " + str(i + 1) + ", True Cap: " + str(true_val_h[i]))
        elif type == "elec":
            plt.title("Bdg " + str(i + 1) + ", True Cap: " + str(true_val_b[i]))
        elif type == "cooling":
            plt.title("Bdg " + str(i + 1) + ", True Cap: " + str(true_val_c[i]))
        plt.grid()
    plt.suptitle("climate zone " + str(climate), fontsize=20)
    # fig.tight_layout()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.show()

def plot_cap_b(cap_b_all, climate, type="elec"):
    subplots = [331, 332, 333, 334, 335, 336, 337, 338, 339]
    true_val_h = [10.68, 49.35, 1e-05, 1e-05, 60.12, 105.12, 85.44, 111.96, 102.24]
    true_val_b = [140, 80, 50, 75, 50, 30, 40, 30, 35]
    true_val_c = [618.12, 227.37, 414.68, 383.565, 244.685, 96.87, 127.82, 165.45, 175.23]
    length = len(subplots)
    fig = plt.figure(figsize=[6.4*2.5, 6.4*2.5])
    for i in range(length):
        plt.subplot(subplots[i])
        plt.scatter(range(len(cap_b_all[i, :])), cap_b_all[i, :])
        fig.tight_layout()
        plt.ylabel("est. of capacity")
        if type == "heat":
            plt.title("Bdg "+str(i+1)+", True Cap: "+str(true_val_h[i]))
        elif type == "elec":
            plt.title("Bdg "+str(i+1)+", True Cap: "+str(true_val_b[i]))
        elif type == "cool":
            plt.title("Bdg "+str(i+1)+", True Cap: "+str(true_val_c[i]))
        plt.grid()
    plt.suptitle("climate zone "+str(climate))
    plt.show()

def to_dic(state_bdg):
    s_dic = {}
    count = 0
    for uid in building_ids:
        state = state_bdg[count]
        count += 1
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
        s_dic[uid] = s
    return s_dic

def cop_cal():
    t = t_out["Building_1"][-1]
    t_1 = t_out["Building_1"][-2]
    eta_tech = 0.22
    target_c = 8
    cop_c = {i: 0 for i in building_ids}
    cop_c_1 = {i: 0 for i in building_ids}
    for uid in building_ids:
        cop_c[uid] = eta_tech * (target_c + 273.15) / (t - target_c)
        if cop_c[uid] <= 0 or cop_c[uid] > 20:
            cop_c[uid] = 20
        cop_c_1[uid] = eta_tech * (target_c + 273.15) / (t_1 - target_c)
        if cop_c_1[uid] <= 0 or cop_c_1[uid] > 20:
            cop_c_1[uid] = 20
    return cop_c_1, cop_c

def estimate_h(a_c_temp: dict, a_h_temp: dict, a_b_temp: dict):
    global tau_c, tau_h, tau_cplus, tau_hplus, action_c, action_h, cap_b, a_c_buffer
    global soc_c, soc_h, soc_b, ns_load, solar_gen, num_c_points, prev_hour_est_h, avail_ratio_est_h

    action = []
    # e_wh = {uid: 0 for uid in building_ids}
    # a_b = {uid: 0 for uid in building_ids} if a_b is None else a_b
    # a_h = {uid: 0 for uid in building_ids} if a_h is None else a_h
    a_clip = {uid: 0 for uid in building_ids}
    add_points = {uid: 0 for uid in building_ids}
    cap_h = {uid: 0 for uid in building_ids}
    effi = {uid: 0 for uid in building_ids}
    a_b = {uid: 0 for uid in building_ids}
    a_h = {uid: 0 for uid in building_ids}
    a_c = {uid: 0 for uid in building_ids}
    effi_h = 1
    ''' params over one step: e_wh, a_h '''
    for uid in building_ids:
        if uid in building_no_heat:
            action_now = [action_c[uid], 0, 0.03]
            action.append(action_now)
            cap_h[uid] = 0
            continue
        if prev_hour_est_h[uid] is True or avail_ratio_est_h[uid] is True:
            # --------------update tau_plus if not satisfied-----------
            if avail_ratio_est_h[uid] is True and soc_c[uid][-1] == 0:
                tau_c[uid] = min(tau_c[uid] + 0.1, 0.8)
                action_c[uid] = min(action_c[uid] + 0.05, 0.5)
                tau_cplus[uid] = max(0, min(0.8-tau_c[uid], max(tau_cplus[uid], -(soc_c[uid][-1] - (1-CF_C)*soc_c[uid][-2])/1.1)))
                avail_ratio_est_h[uid], prev_hour_est_h[uid] = False, False
                # two_points_avg[uid] = False

            if avail_ratio_est_h[uid] is True and uid not in building_no_heat and soc_h[uid][-1] == 0:
                tau_h[uid] = min(tau_h[uid] + 0.1, 0.8)
                action_h[uid] = min(action_h[uid] + 0.05, 0.5)
                # tau_hplus = max(0, min(0.8-tau_h, max(tau_hplus, -(soc_h[uid][-1] - (1-CF_H)*soc_h[uid][-2])/1.1)))
                avail_ratio_est_h[uid], prev_hour_est_h[uid] = False, False
                # two_points_avg[uid] = False
            if s_dic[uid]["hour"] == 22:
                avail_ratio_est_h[uid], prev_hour_est_h[uid] = False, False
            #--------------estimate e_wh if avail---------------------
        if prev_hour_est_h[uid] is True and avail_ratio_est_h[uid] is True:     # calculate capacity here
            e_wh = (solar_gen[uid][-2] + net_elec[uid][-1] - ns_load[uid][-2] -
                    (soc_b[uid][-1] - (1 - CF_B) * soc_b[uid][-2]) * cap_b[uid] / effi_b[uid])
            # if two_points_est[uid] is True:
            avail_ratio_est_h[uid], prev_hour_est_h[uid] = False, False
            ratio = -(soc_h[uid][-2] - (1 - CF_H) * soc_h[uid][-3])
            cap_h[uid] = e_wh * effi_h / (a_h_temp[uid][-1] / 0.9 + ratio)
            effi[uid] = (soc_h[uid][-1] - (1 - CF_H) * soc_h[uid][-2]) / a_h_temp[uid][-1]
            print(uid, ":", cap_h[uid])
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
        if soc_h[uid][-1] < tau_h[uid]:
            a_h[uid] = action_h[uid]
        elif soc_c[uid][-1] > 0.9:
            a_h[uid] = -0.2
        else:
            a_h[uid] = 0.04

        a_c[uid] = action_c[uid] if soc_c[uid][-1] < tau_c[uid] + tau_cplus[uid] else 0.1

        if soc_h[uid][-1] >= tau_h[uid] and avail_ratio_est_h[uid] is False:
            if soc_c[uid][-1] >= tau_c[uid] + tau_cplus[uid]:
                a_c[uid], a_h[uid] = -1, -1
                # action_now = [a_h, a_h, a_b[uid]]
                avail_ratio_est_h[uid] = True

            # action.append(action_now)   # exit
        elif avail_ratio_est_h[uid] is True:
            if soc_h[uid][-1] < tau_h[uid]:
                tau_hplus[uid] = max(0, min(0.8 - tau_h[uid],
                                       max(tau_hplus[uid] + 0.1, -(soc_h[uid][-1] - (1 - CF_H) * soc_h[uid][-2]) / 1.1)))
            a_b[uid], a_c[uid] = 0.03, -1
            a_clip[uid] = -(soc_h[uid][-1] - (1 - CF_H) * soc_h[uid][-2])  # calculate a_t here
            if soc_h[uid][-1] >= 0.97:
                a_h[uid] = -1
                action_now = [a_c[uid], a_h[uid], a_b[uid]]
                a_c_temp[uid].append(a_c[uid])
                a_h_temp[uid].append(a_h[uid])
                a_b_temp[uid].append(a_b[uid])
                action.append(action_now)
                continue
            elif a_clip[uid] > 0.01:
                a_h[uid] = min(a_clip[uid], 1 - soc_h[uid][-1])
            else:
                a_h[uid] = min(0.03, 1 - soc_h[uid][-1])
            a_c[uid] = -1
            prev_hour_est_h[uid] = True
        action_now = [a_c[uid], a_h[uid], a_b[uid]]
        a_c_temp[uid].append(a_c[uid])
        a_h_temp[uid].append(a_h[uid])
        a_b_temp[uid].append(a_b[uid])
        action.append(action_now)  # exit
    # print(action)
    # sys.exit()

    return action, cap_h, effi, add_points


def estimate_c(a_c_temp: dict, a_h_temp: dict, a_b_temp: dict):
    global tau_c, tau_h, tau_cplus, tau_hplus, action_c, action_h, cap_b, a_c_buffer
    global soc_c, soc_h, soc_b, ns_load, solar_gen, num_c_points, prev_hour_est_c, avail_ratio_est_c

    action = []
    # e_hpc = {uid: 0 for uid in building_ids}
    # a_b = {uid: 0 for uid in building_ids} if a_b is None else a_b
    # a_c = {uid: 0 for uid in building_ids} if a_c is None else a_c
    a_clip = {uid: 0 for uid in building_ids}
    add_points = {uid: 0 for uid in building_ids}
    cap_c = {uid: 0 for uid in building_ids}
    effi = {uid: 0 for uid in building_ids}
    a_b = {uid: 0 for uid in building_ids}
    a_c = {uid: 0 for uid in building_ids}
    cop_c_1, cop_c = cop_cal()
    ''' params over one step: e_hpc, a_c '''
    for uid in building_ids:
        if prev_hour_est_c[uid] is True or avail_ratio_est_c[uid] is True:
            # --------------update tau_plus if not satisfied-----------
            if avail_ratio_est_c[uid] is True and soc_c[uid][-1] == 0:
                tau_c[uid] = min(tau_c[uid] + 0.1, 0.8)
                action_c[uid] = min(action_c[uid] + 0.05, 0.5)
                # tau_cplus = max(0, min(0.8-tau_c, max(tau_cplus, -(soc_c[uid][-1] - (1-CF_C)*soc_c[uid][-2])/1.1)))
                avail_ratio_est_c[uid], prev_hour_est_c[uid] = False, False
                # two_points_avg[uid] = False

            if avail_ratio_est_c[uid] is True and uid not in building_no_heat and soc_h[uid][-1] == 0:
                tau_h[uid] = min(tau_h[uid] + 0.1, 0.8)
                action_h[uid] = min(action_h[uid] + 0.05, 0.5)
                tau_hplus[uid] = max(0, min(0.8-tau_h[uid], max(tau_hplus[uid], -(soc_h[uid][-1] - (1-CF_H)*soc_h[uid][-2])/1.1)))
                avail_ratio_est_c[uid], prev_hour_est_c[uid] = False, False
                # two_points_avg[uid] = False
            if s_dic[uid]["hour"] == 22:
                avail_ratio_est_c[uid], prev_hour_est_c[uid] = False, False

            #--------------estimate e_hpc if avail---------------------
        if prev_hour_est_c[uid] is True and avail_ratio_est_c[uid] is True:     # calculate capacity here
            e_hpc = (solar_gen[uid][-2] + net_elec[uid][-1] - ns_load[uid][-2] -
                     (soc_b[uid][-1] - (1 - CF_B) * soc_b[uid][-2]) * cap_b[uid] / effi_b[uid])
            # if two_points_est[uid] is True:
            avail_ratio_est_c[uid], prev_hour_est_c[uid] = False, False
            ratio = -(soc_c[uid][-2] - (1 - CF_C) * soc_c[uid][-3])
            cap_c[uid] = e_hpc * cop_c_1[uid] / (a_c_temp[uid][-1] + ratio)
            effi[uid] = (soc_c[uid][-1] - (1 - CF_C) * soc_c[uid][-2]) / a_c_temp[uid][-1]
            print(uid, ":", cap_c[uid])
            add_points[uid] += 1

        """
        1) execute action to est ratio if soc > threshold  
        2) observe soc and calculate ratio if satisfying requirements, and execute action for e_hpc calculation
        3) observe params and calculate e_hpc, and execute action to est ratio if soc > threshold
        4) observe soc and est capacity. restart the procedure
        5) if # of est points is enough, end up est.
        """
        a_b[uid] = 0.03
        if soc_c[uid][-1] < tau_c[uid]:
            a_c[uid] = action_c[uid]
        elif soc_c[uid][-1] > 0.9:
            a_c[uid] = -0.2
        else:
            a_c[uid] = 0.04

        if uid not in building_no_heat:
            a_h = action_h[uid] if soc_h[uid][-1] < tau_h[uid] + tau_hplus[uid] else 0.1
        else:
            a_h = 0

        if soc_c[uid][-1] >= tau_c[uid] and avail_ratio_est_c[uid] is False:
            if uid not in building_no_heat and soc_h[uid][-1] >= (tau_h[uid] + tau_hplus[uid]):
                a_c[uid], a_h = -1, -1
                # action_now = [a_c, a_h, a_b[uid]]
                avail_ratio_est_c[uid] = True

            if uid in building_no_heat:
                a_c[uid] = -1
                # action_now = [a_c, a_b[uid]]
                avail_ratio_est_c[uid] = True
            # action.append(action_now)   # exit
        elif avail_ratio_est_c[uid] is True:
            if uid not in building_no_heat and soc_h[uid][-1] < tau_h[uid]:
                tau_hplus[uid] = max(0, min(0.8 - tau_h[uid],
                                       max(tau_hplus[uid] + 0.1, -(soc_h[uid][-1] - (1 - CF_H) * soc_h[uid][-2]) / 1.1)))
            a_b[uid], a_h = 0.03, -1

            a_clip[uid] = -(soc_c[uid][-1] - (1 - CF_C) * soc_c[uid][-2]) / 1.1  # calculate a_t here
            if soc_c[uid][-1] >= 0.97:
                a_c[uid] = -1
                action_now = [a_c[uid], a_h, a_b[uid]]
                a_c_temp[uid].append(a_c[uid])
                a_h_temp[uid].append(a_h)
                a_b_temp[uid].append(a_b[uid])
                action.append(action_now)
                continue
            elif a_clip[uid] > 0.01:
                a_c[uid] = min(a_clip[uid], 1 - soc_c[uid][-1])
            else:
                a_c[uid] = min(0.03, 1 - soc_c[uid][-1])
            prev_hour_est_c[uid] = True
        action_now = [a_c[uid], a_h, a_b[uid]]
        a_c_temp[uid].append(a_c[uid])
        a_h_temp[uid].append(a_h)
        a_b_temp[uid].append(a_b[uid])
        action.append(action_now)  # exit
    # print(action)
    # sys.exit()

    return action, cap_c, effi, add_points

def estimate_bat(s_dic, prev_hour_est, avail_nominal, prev_hour_nom):
    global tau_c, action_c, tau_h, action_h, soc_b, tau_b, solar_gen, ns_load, net_elec, cap_b_est, effi_b
    add_points = {uid : 0 for uid in building_ids}
    cap_bat = {uid : 0 for uid in building_ids}
    action = []
    a_clip = {uid : 0 for uid in building_ids}
    e_bat = {uid : 0 for uid in building_ids}
    effi = {uid : 0 for uid in building_ids}
    nominal_p = {uid : 0 for uid in building_ids}

    print("time: ", s_dic["Building_1"]["hour"])

    for uid in building_ids:
        if prev_hour_est[uid] is True or prev_hour_nom[uid] is True:
            if s_dic[uid]["cooling_storage_soc"] < 0.01:
                tau_c[uid] = min(tau_c[uid] + 0.1, 0.8)
                action_c[uid] = min(action_c[uid] + 0.05, 0.5)
                prev_hour_est[uid] = False
                prev_hour_nom[uid] = False

            if uid not in ["Building_3", "Building_4"] and s_dic[uid]["dhw_storage_soc"] < 0.01:
                tau_h[uid] = min(tau_h[uid] + 0.1, 0.8)
                action_h[uid] = min(action_h[uid] + 0.05, 0.5)
                prev_hour_est[uid] = False
                prev_hour_nom[uid] = False

            if prev_hour_nom[uid] is True and s_dic[uid]["electrical_storage_soc"] < 0.01:
                tau_b[uid] = min(tau_b[uid] + 0.1, 0.8)
                prev_hour_nom[uid] = False

            if s_dic[uid]["hour"] == 22:
                prev_hour_est[uid] = False
                prev_hour_nom[uid] = False

        if prev_hour_est[uid] is True:
            soc_1 = soc_b[uid][-2]
            soc_2 = soc_b[uid][-1]
            print("t+1 elec %s: " % uid, soc_b[uid][-1])
            a_clip[uid] = (soc_2 - (1 - CF_B) * soc_1)
            e_bat[uid] = solar_gen[uid][-2] + net_elec[uid][-1] - ns_load[uid][-2]
            cap_bat[uid] = (e_bat[uid] / a_clip[uid])
            effi[uid] = a_clip[uid] / 0.05
            add_points[uid] = 1
            # prev_hour_est[uid] = False

        elif prev_hour_nom[uid] is True:
            soc_1 = soc_b[uid][-2]
            soc_2 = soc_b[uid][-1]
            a_clip[uid] = -(soc_2 - (1 - CF_B) * soc_1)
            nominal_p[uid] = max(cap_b_est[uid]) * (a_clip[uid] / effi_b[uid])
            print("nominal power %s: " % uid, nominal_p[uid])

            """reality:     75   40   20   30   25   10   15   10   20"""
            """estimated:   89.3 40.5 21.6 34.1 26.9 11.9 15.8 11.9 23.8"""
        else:
            print("t elec %s: " % uid, soc_b[uid][-2])

        a_c = action_c[uid] if s_dic[uid]["cooling_storage_soc"] < tau_c[uid] else 0.02
        a_b = -0.3 if s_dic[uid]["electrical_storage_soc"] > 0.9 else 0.02
        if uid not in ["Building_3", "Building_4"]:
            a_h = action_h[uid] if s_dic[uid]["dhw_storage_soc"] < tau_h[uid] else 0.04
        else:
            a_h = 0

        if avail_nominal[uid] is True:
            if soc_b[uid][-1] > tau_b[uid]:
                action_now = [-1, -1, -1]
                prev_hour_nom[uid] = True
                prev_hour_est[uid] = False
            else:
                action_now = [a_c, a_h, 0.3]
                prev_hour_nom[uid] = False
                prev_hour_est[uid] = False
            action.append(action_now)
            continue

        if s_dic[uid]["cooling_storage_soc"] > tau_c[uid] and s_dic[uid]["electrical_storage_soc"] < 0.9 and uid in ["Building_3", "Building_4"]\
            or s_dic[uid]["cooling_storage_soc"] > tau_c[uid] and s_dic[uid]["electrical_storage_soc"] < 0.9 and s_dic[uid]["dhw_storage_soc"] > tau_h[uid] and uid not in ["Building_3", "Building_4"]:
            action_now = [-1, -1, 0.05] if uid not in ["Building_3", "Building_4"] else [-1, 0, 0.05]
            prev_hour_est[uid] = True
        else:
            action_now = [a_c, a_h, a_b]
            prev_hour_est[uid] = False
        print("action %s:" % uid, action_now, prev_hour_est[uid])
        action.append(action_now)

    return action, cap_bat, effi, nominal_p, prev_hour_est, prev_hour_nom, add_points


for algorithm in ['RBC']:
    for climate in [5]:
        climate_zone = climate
        TOTAL_TIME_STEP = 8760  # 8760
        CF_C = 0.006
        CF_H = 0.008
        CF_B = 0
        num_data = 5*9  # the number of data we want to collect

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
        building_ids = ["Building_" + str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
        a_space = {uid: a_space for uid, a_space in zip(building_ids, actions_spaces)}
        # Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
        building_info = env.get_building_information()

        building_no_heat = ["Building_3", "Building_4"]

        cap_h_all = np.zeros([9, num_data])
        cap_b_all = np.zeros([9, num_data])
        cap_c_all = np.zeros([9, num_data])
        ratio_c = np.zeros([9,num_data])
        ratio_h = np.zeros([9,num_data])
        C_bd = np.zeros([9,num_data])
        H_bd = np.zeros([9,num_data])

        #---------following part record values in states--------------------
        t_out = {uid: [] for uid in building_ids}
        soc_b = {uid: [] for uid in building_ids}
        soc_c = {uid: [] for uid in building_ids}
        soc_h = {uid: [] for uid in building_ids}
        solar_gen = {uid: [] for uid in building_ids}
        ns_load = {uid: [] for uid in building_ids}
        pv_gen = {uid: [] for uid in building_ids}
        net_elec = {uid: [] for uid in building_ids}
        # ---------following part record thresholds--------------------

        tau_c = {uid: 0.2 for uid in building_ids}
        tau_h = {uid: 0.2 for uid in building_ids}
        tau_b = {uid: 0.5 for uid in building_ids}
        tau_cplus = {uid: 0.1 for uid in building_ids}
        tau_hplus = {uid: 0.1 for uid in building_ids}
        action_c = {uid: 0.1 for uid in building_ids}
        action_h = {uid: 0.1 for uid in building_ids}

        # ---------following part record boolean vars for est--------------------
        prev_hour_est_b = {uid: False for uid in building_ids}
        prev_hour_est_c = {uid: False for uid in building_ids}
        prev_hour_est_h = {uid: False for uid in building_ids}
        a_clip = {uid: None for uid in building_ids}
        avail_ratio_est_c = {uid: False for uid in building_ids}
        avail_ratio_est_h = {uid: False for uid in building_ids}
        avail_nominal = {uid: False for uid in building_ids}
        prev_hour_nom = {uid: False for uid in building_ids}

        a_c_buffer = {uid: None for uid in building_ids}
        # ---------following part record # of reliable est points--------------------
        num_elec_points = {uid: 0 for uid in building_ids}
        num_h_points = {uid: 0 for uid in building_ids}
        num_c_points = {uid: 0 for uid in building_ids}
        # ------------------------temporal vars---------------------------------
        a_b_temp = {uid: [] for uid in building_ids}
        a_c_temp = {uid: [] for uid in building_ids}
        a_h_temp = {uid: [] for uid in building_ids}
        e_hpc_temp = {uid: [] for uid in building_ids}
        # -----------------------record est results-----------------------------
        cap_c_est = {uid: [] for uid in building_ids}
        cap_h_est = {uid: [] for uid in building_ids}
        cap_b_est = {uid: [] for uid in building_ids}
        effi_b = {uid: 0 for uid in building_ids}
        effi_c = {uid: 0 for uid in building_ids}
        effi_h = {uid: 0 for uid in building_ids}
        nominal_b = {uid: [] for uid in building_ids}

        ratio_est_c = []
        ratio_est_h = []
        C_bd_temp = []
        H_bd_temp = []

        agents = RBC(actions_spaces)
        print("action space: ", actions_spaces)
        state = env.reset()  # hour 0
        s_dic = to_dic(state)
        record_dic(s_dic)

        done = False
        # test for RBC
        action = agents.select_action(state)    # action for hour 0
        E_day = True
        H_day = False
        C_day = False
        # print(action)
        while not done:
            # prev_hour_est_b = False if prev_hour_est_b is None else prev_hour_est_b
            next_state, reward, done, _ = env.step(action)  # execution of hour 0
            # print(next_state)
            # next state is for hour 1
            s_dic = to_dic(next_state)
            record_dic(s_dic)
            # print("time: ", s_dic["hour"])
            # if s_dic["hour"] == 3:
            # print(s_dic["cooling_storage_soc"], s_dic["dhw_storage_soc"], s_dic["electrical_storage_soc"])

            action_next = agents.select_action(next_state)
            state = next_state
            action = action_next

            ###### Cooling Storage
            # if s_dic["hour"] == [8]:      # cooling storage est
            #     if len(cap_c_est) <= num_data:
            #         state, action, cap_c, avail, two_point_est_c, C_bd_est = estimate_c(s_dic, action, bdg=i)
            #         if 0 < cap_c < 1000 and avail is True:
            #             ratio_est_c.append(two_point_est_c)
            #             cap_c_est.append(cap_c)
            #             C_bd_temp.append(C_bd_est)
            #         if len(cap_c_est) == num_data:
            #             print(cap_c_est)
            #             print(np.mean(np.array(cap_c_est)), np.min(np.array(cap_c_est)))
            #             cap_c_all[i, :] = np.array(cap_c_est)
            #             ratio_c[i, :] = np.array(ratio_est_c)
            #             C_bd[i, :] = np.array(C_bd_temp)
            #             break

            ###### Battery

            if E_day is True and s_dic["Building_1"]["hour"] in [1, 2, 3, 4, 5, 6, 22, 23, 24]:      # battery cap est
                action, cap_b, effi, nominal, prev_hour_est_b, prev_hour_nom, add_point = \
                    estimate_bat(s_dic, prev_hour_est_b, avail_nominal, prev_hour_nom)
                sign = False
                for uid in building_ids:
                    if add_point[uid] == 1:
                        num_elec_points[uid] += add_point[uid]
                        effi_b[uid] = effi[uid] if effi[uid] != 0 else effi_b[uid]
                        cap_b_est[uid].append(cap_b[uid] * effi_b[uid])
                    if nominal[uid] > 0:
                        nominal_b[uid].append(nominal[uid])
                    avail_nominal[uid] = True if num_elec_points[uid] >= 3 else False
                    sign = True if len(nominal_b[uid]) >= 3 else False
                if sign is True:
                    E_day = False
                    H_day, C_day = False, True
                    for uid in building_ids:
                        cap_b[uid] = max(cap_b_est[uid])
                        print(uid, ":", cap_b[uid], "|", nominal_b[uid])
                    print("real nominal power: 75 40 20 30 25 10 15 10 20")

            if C_day is True and s_dic["Building_1"]["hour"] in [1, 2, 3, 4, 5, 6, 22, 23, 24]:
                action, cap_c, effi, add_points = \
                    estimate_c(a_c_temp, a_h_temp, a_b_temp)
                sign = False
                for uid in building_ids:
                    if add_points[uid] == 1:
                        if cap_c[uid] > 0:
                            num_c_points[uid] += add_points[uid]
                            effi_c[uid] = effi[uid]
                            cap_c_est[uid].append(cap_c[uid] * effi_c[uid])
                    sign = True if num_c_points[uid] >= 8 else False
                if sign is True:
                    H_day, C_day = True, False
                    for uid in building_ids:
                        print(uid, ":", cap_c_est[uid])

            if H_day is True and s_dic["Building_1"]["hour"] in [1, 2, 3, 4, 5, 6, 22, 23, 24]:
                action, cap_h, effi, add_points = \
                    estimate_h(a_c_temp, a_h_temp, a_b_temp)
                sign = False
                for uid in building_ids:
                    if add_points[uid] == 1:
                        if cap_h[uid] > 0:
                            num_h_points[uid] += add_points[uid]
                            effi_h[uid] = effi[uid]
                            cap_h_est[uid].append(cap_h[uid] * effi_h[uid])
                    sign = True if num_h_points[uid] >= 8 else False
                if sign is True:
                    H_day, C_day = False, False
                    for uid in building_ids:
                        print(uid, ":", cap_h_est[uid])
                    break
            ###### Heat Storage
            # if s_dic["hour"] in [1]:        # heating storage soc
            #     if i not in [2, 3]:
            #         if len(cap_h_est) <= num_data:
            #             state, action, cap_h, avail,two_point_est_h = estimate_hc(s_dic, action, bdg=i)
            #             if 0 < cap_h < 500 and avail is True:
            #                 ratio_est_h.append(two_point_est_h)
            #                 cap_h_est.append(cap_h)
            #             if len(cap_h_est) == num_data:
            #                 print(cap_h_est)
            #                 print(np.mean(np.array(cap_h_est)), np.min(np.array(cap_h_est)))
            #                 cap_h_all[i, :] = np.array(cap_h_est)
            #                 print("ratio est:",ratio_est_h)
            #                 ratio_h[i, :] = np.array(ratio_est_h)
            #                 break


        # plot_cap_b(cap_b_all, climate, type="elec")
        # plot_cap_h(cap_c_all, climate, type="cooling")
        # plot_cap_h(cap_h_all, climate, type="heat")