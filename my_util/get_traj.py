"""this is a module for getting traj data"""
import glob
import pandas as pd
import numpy as np

def get_traj(lat, lon, animal):
    """select dir"""
    if animal == "bird":
        dir_ = "bird_data/1min_median_label"
    elif animal == "cel":
        dir_ = "worm_data/data_by_individual"

    ls_csv = sorted(glob.glob("../"+dir_+"/*.csv"))
    dir_traj = {}
    j_pre = 0
    if animal == "bird":
        for i, csv in enumerate(ls_csv):
            tmp = pd.read_csv(csv)
            tripno_df = tmp["tripno"]
            for j in range(max(tripno_df)):
                tmp_ = pd.concat([tmp.loc[tmp["tripno"] == j][lat], tmp.loc[tmp["tripno"] == j][lon]], axis=1)
                tmp_ = tmp_.reset_index(drop=True)
                dir_traj[j+j_pre] = tmp_
            j_pre = j
    elif animal == "cel":
        for i, csv in enumerate(ls_csv):
            tmp = pd.read_csv(csv)
            tmp_ = pd.concat([tmp[lat], tmp[lon]], axis=1)
            dir_traj[i] = tmp_

    return dir_traj

def get_len_traj(traj):
    """get length of traj"""
    ls_traj_length = []
    for x in traj:
        ls_traj_length.append(traj[x].shape[0])
    return ls_traj_length

def norm_fromdir_todf_traj(trajectory, time_dim):
    """reshape dir traj to df traj"""
    df_traj = pd.DataFrame(None)
    # time
    time_setting = False
    if time_setting:
        time = np.linspace(0, 255, 6000)
    for i in trajectory:
        # (R,G,B)=(X,Y,t)
        if time_dim:
            if not time_setting:
                time = np.linspace(0, 255, trajectory[i].shape[0])
            zero_ = pd.Series(time)
        # (R,G,B)=(X,Y,0) omosugi
        else:
            zero_ = pd.Series(np.zeros(trajectory[i].shape[0]))

        traj = trajectory[i]
        # norm 0~255
        #max_ = traj.iloc[:,:2].max().max()
        #min_ = traj.iloc[:,:2].min().min()
        #print(i,max_,min_)
        for j in range(2):
            #traj.iloc[:,j] = (traj.iloc[:,j]-min_)/(max_-min_)*255
            traj.iloc[:, j] = (traj.iloc[:, j]-traj.iloc[:, j].min())/(traj.iloc[:, j].max()-traj.iloc[:, j].min())*255
        #add
        df_traj = pd.concat([df_traj, traj, zero_], axis=1)
        #print(df_traj)
    return df_traj#.fillna(0)
