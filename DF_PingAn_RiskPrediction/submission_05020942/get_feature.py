import pandas as pd
import numpy as np
from utils import*



def extrac_aver_time(user,diff_div):


    ## 如果两次开车时间间隔大于1小时，判断为不同次开车
    #user["diff_div"] = (user["delta_time"] > 3600).astype(int)
    #user["diff_div"] = np.where(user["delta_time"].values.astype(int) > 3600, 1, 0)
    start_time_ind = np.argwhere(user[diff_div==1])


    try:
        start_time_ind = np.append(start_time_ind, start_time_ind[-1] + 1)

    except Exception as e:
        start_time_ind = np.append(start_time_ind, user.shape[0])

    #print(start_time)
    drive_times = []
    st = user["second_time"][0]
    fatigue_dirve = 0
    for end_ind in start_time_ind:
        end = user["second_time"][end_ind-1]

        delta_time = end - st
        if delta_time > 4*3600:
            fatigue_dirve += 1
        st = end
        drive_times.append(delta_time)
    drive_times = np.array(drive_times)/3600
    aver_delta = np.mean(drive_times)
    max_delta = np.max(drive_times)
    freq_fatigue_dirve = fatigue_dirve*1.0/(diff_div.sum() + 1)
    return aver_delta,max_delta,freq_fatigue_dirve

def extrac_speed_interval(user,diff_div):

    ## 如果两次开车时间间隔大于1小时，判断为不同次开车
    #user["diff_div"] = (user["delta_time"].values > 3600).astype(int)
    #user["diff_div"] = np.where(user["delta_time"].values.astype(int) > 3600, True, False)
    sum_times_div = diff_div.sum() + 1
    #start_time_ind = user[user['diff_div'].isin([1])].index.tolist()
    start_time_ind = np.argwhere(user[diff_div==1])
    try:
        start_time_ind = np.append(start_time_ind,start_time_ind[-1] + 1)
      
    except Exception as e:
        start_time_ind = np.append(start_time_ind, user.shape[0])
       

    #print(start_time)
    drive_times = []
    st = 0
    speed_class = np.zeros(3)
    for end_ind in start_time_ind:
        if (user["SPEED"][st:end_ind] > 60).sum() == 0 :
            speed_class[0] += 1.0
        elif (user["SPEED"][st:end_ind] < 120).sum() > 0:
            speed_class[1] += 1.0
        elif (user["SPEED"][st:end_ind] >= 120).sum() > 0:
            speed_class[2] += 1.0

    return speed_class/sum_times_div


def get_turn_r_l_a(user):
    user = pd.DataFrame(user)

    unknown_dire = user[user.DIRECTION.values < 0].index.tolist()
    count_unknown_times = 0
    for unk in unknown_dire:
        count_unknown_times += 1
        try:
            user.iloc[unk].DIRECTION = user.iloc[unk - 1].DIRECTION
        except Exception as e:
            print(e)
            ## 如果一开始行驶方向就不知道，那么我们删除
            user.drop(unk, axis=0, inplace=True)

    def judge_dire(x):
        if 90 > x[2] > 10:
            if x[1] > x[0] + 180:
                return "left"
            else:
                return "right"
        elif -90 < x[2] < -10:
            if x[1] < x[0] - 180:
                return "right"
            else:
                return "left"
        elif -10 < x[2] < 10:
            return "straight"
        else:
            return "turn_around"

    dire = ["left", "right", "straight", "turn_around", "unknown_fre"]
    user["DIRECTION_shift"] = user.DIRECTION.shift(1)
    user["change_angle"] = user.DIRECTION.values - user.DIRECTION_shift.values
    user["dire_chan_fre"] = user[["DIRECTION", "DIRECTION_shift", "change_angle"]].apply(judge_dire, axis="columns")
    k = user.groupby('dire_chan_fre', as_index=False)["dire_chan_fre"].agg({'change_times': 'count'})
    k.set_index("dire_chan_fre", inplace=True)
    k["change_times"] = k["change_times"].map(lambda x: x * 1.0 / user.shape[0])
    k.loc["unknown_fre", "change_times"] = count_unknown_times * 1.0 / user.shape[0]
    if k.shape[0] != 5:
        x = k.index.tolist()
        for i in dire:
            if i not in x:
                k.loc[i, "change_times"] = 0
    k.sort_index(inplace=True)
    return list(k.change_times)


def get_phone_state(user,num):
    p_s_f=np.zeros(5)
    p_s_f[0]=(user["CALLSTATE"] == 0).sum()
    p_s_f[1] = (user["CALLSTATE"] == 1).sum()
    p_s_f[2] = (user["CALLSTATE"] == 2).sum()
    p_s_f[3] = (user["CALLSTATE"] == 3).sum()
    p_s_f[4] = (user["CALLSTATE"] == 4).sum()

    p_s_f = p_s_f*1.0/num
    return p_s_f




def get_drive_quar(user,num):

    q = np.zeros(4)
    q[0] = (user["quarter"] == 1).sum()
    q[1] = (user["quarter"] == 2).sum()
    q[2] = (user["quarter"] == 3).sum()
    q[3] = (user["quarter"] == 4).sum()

    return q/num


def get_drive_night(user):

    user["drive_n"] = np.where((user["hour"].values > 7) & (user["hour"].values <19),1,0)
    in_day = user["drive_n"].values.mean()
    in_night = 1 - in_day
    return in_day, in_night


if __name__ == '__main__':
    pass



