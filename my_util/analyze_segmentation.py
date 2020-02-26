"""calc (v, a, angle, ...) and plot"""
import os
import glob
import datetime
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### begin region ###

import logging

# create logger
logger = logging.getLogger('analyze_seg')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
sh = logging.StreamHandler()
fh = logging.FileHandler("./log/test.log")
sh.setLevel(logging.INFO)
fh.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s')

# add formatter to handler
sh.setFormatter(formatter)
fh.setFormatter(formatter)

# add handler to logger
logger.addHandler(sh)
logger.addHandler(fh)

### end region ###

def get_traj(lat, lon, animal):
    """select dir"""
    if animal == "bird":
        dir_ = "bird_data/1min_median_label"
    elif animal == "cel":
        dir_ = "worm_data/data_by_individual"

    ls_csv = sorted(glob.glob("../../"+dir_+"/*.csv"))
    df_traj = pd.DataFrame(None)

    if animal == "bird":

        def transform_int(timestamp):
            """transform str(yyyy/mm/dd hh:mm:ss) into int(0 ~ )"""
            timestamp_v = timestamp.values
            start_time = datetime.datetime.strptime(timestamp_v[0], "%Y-%m-%d %H:%M:%S")

            timestamp.iloc[0] = 0.0

            for i in range(1, timestamp.shape[0]):
                if isinstance(timestamp_v[i], str):
                    now_time = datetime.datetime.strptime(timestamp_v[i], "%Y-%m-%d %H:%M:%S")
                    dt = now_time - start_time
                    timestamp.iloc[i] = dt.total_seconds()
                else:
                    continue

            return timestamp

        for i, csv in enumerate(ls_csv):
            tmp = pd.read_csv(csv)

            for j in range(min(tmp["tripno"]), max(tmp["tripno"])):
                if j in set(tmp["tripno"]):
                    timestamp = transform_int(tmp.where(tmp["tripno"] == j)["timestamp"].dropna(how='all'))

                    df_traj = pd.concat([df_traj, \
                        tmp.where(tmp["tripno"] == j)[lat], \
                            tmp.where(tmp["tripno"] == j)[lon], \
                                timestamp, \
                                    tmp.where(tmp["tripno"] == j)["distance"], \
                                        tmp.where(tmp["tripno"] == j)["speed"], \
                                            tmp.where(tmp["tripno"] == j)["angle"], \
                                                tmp.where(tmp["tripno"] == j)["label"]], \
                                                    axis=1)
                    df_traj = df_traj.reset_index(drop=True)

    elif animal == "cel":

        for i, csv in enumerate(ls_csv):
            tmp = pd.read_csv(csv)

            df_traj = pd.concat([df_traj, \
                tmp[lat], tmp[lon], tmp["time(sec)"], \
                    tmp["odor_type"], tmp["turn"]], axis=1)

    return df_traj

def load_df(path, lat, lon, animal):
    if os.path.exists(path):
        # load
        df = pd.read_pickle(path)
    else:
        # make
        df = get_traj(lat, lon, animal)
        # save
        df.to_pickle(path)

    return df

def label_trans(label):
    label_trans_list = [0]
    for i, x in enumerate(label):
        if i == 0:
            pre_x = x

        elif i == len(label)-1:
            label_trans_list.append(i)

        else:
            if x != pre_x:
                pre_x = x
                label_trans_list.append(i)

    logger.debug(label_trans_list)

    return sorted(label_trans_list)

def re_label(label):
    ret = []
    dic = {}# {label:index}

    for i, x in enumerate(label):

        if i == 0:

            dic[x] = len(dic.keys())

        else:

            if x not in dic.keys():

                dic[x] = len(dic.keys())

    for i, x in enumerate(label):

        ret.append(dic[x])

    logger.debug(label)
    logger.debug(dic)
    logger.debug(ret)
    return ret

class Worm(object):
    
    def __init__(self, traj_id, label):
        self.pure_id = traj_id
        self.traj_id = traj_id * 5
        self.label = label
        #HACK:too long load
        self.df_worm = load_df("../worm_data/worm_df_all_data.pkl", "x", "y", "cel")
        self.label_trans_list = label_trans(self.label)
        self.re_label_list = re_label(self.label)

    def calc_data(self):

        df_dx = self.df_worm.iloc[:, self.traj_id].diff().dropna(how="all")
        df_dy = self.df_worm.iloc[:, self.traj_id + 1].diff().dropna(how="all")
        df_dx_delay = pd.concat([df_dx.iloc[1:], pd.Series(df_dx.iloc[-1])], axis=0)
        df_dy_delay = pd.concat([df_dy.iloc[1:], pd.Series(df_dy.iloc[-1])], axis=0)
        df_dx_delay = df_dx_delay.reset_index(drop=True)
        df_dy_delay = df_dy_delay.reset_index(drop=True)

        df_speed = np.sqrt(np.power(df_dx, 2) + np.power(df_dy, 2))
        df_speed_delay = np.sqrt(np.power(df_dx_delay, 2) + np.power(df_dy_delay, 2))
        df_angle = np.arccos((df_dx * df_dx_delay + df_dy * df_dy_delay) / (df_speed * df_speed_delay))
        df_acc = df_speed.diff().dropna(how="all")
        df_turn = self.df_worm.iloc[:,self.traj_id + 4].dropna(how="all")

        df_speed = df_speed.reset_index(drop=True)
        df_angle = df_angle.reset_index(drop=True)
        df_acc = df_acc.reset_index(drop=True)
        df_turn = df_turn.reset_index(drop=True)

        df_cat = pd.concat([df_speed, df_acc, df_angle, df_turn], axis=1)
        df_cat.columns = ["speed", "acc", "angle", "turn"]
        return df_cat


class Bird(object):
    
    def __init__(self, traj_id, label):
        self.pure_id = traj_id
        self.traj_id = traj_id * 7 + 4
        self.label = label
        self.df_bird = load_df("../bird_data/bird_df_all_data.pkl", "latitude", "longitude", "bird")
        self.label_trans_list = label_trans(self.label)
        self.re_label_list = re_label(self.label)

    def calc_data(self):
        df_speed = self.df_bird.iloc[:, self.traj_id].dropna(how="all")
        df_acc = self.df_bird.iloc[:, self.traj_id].diff().dropna(how="all")
        df_angle = self.df_bird.iloc[:, self.traj_id + 1].dropna(how="all")

        df_speed = df_speed.reset_index(drop=True)
        df_acc = df_acc.reset_index(drop=True)
        df_angle = df_angle.reset_index(drop=True)

        df_cat = pd.concat([df_speed, df_acc, df_angle], axis=1)
        df_cat.columns = ["speed", "acc", "angle"]
        return df_cat


def set_animal(traj_id, label, animal):
    if animal == "bird":
        return Bird(traj_id, label)

    elif animal == "cel":
        return Worm(traj_id, label)

    else:
        raise ValueError("no animal")

def Plot_Calc_Data_by_label(Animal, result_dir, epoch):

    n = int(np.ceil(np.sqrt(len(Animal.label_trans_list)))) #必要分割数n

    fig, ax = plt.subplots(nrows=n, ncols=n, figsize=(10, 10))

    df = Animal.calc_data()

    for i in range(len(Animal.label_trans_list)-1):

        s, e = Animal.label_trans_list[i], Animal.label_trans_list[i+1]

        df_tmp = df.iloc[s:e]

        logger.debug(df_tmp.shape)

        if df_tmp.empty:
            continue

        df_tmp.plot(ax=ax[i//n, i%n], legend=(i==0), alpha=0.75, ylim=[df.min().min(), df.max().max()])

        ax[i//n, i%n].set_title("Seg:{}, Len:{}".format(i, e-s))

    plt.savefig("result/{}/analyze_trip{:0=3}_epoch{:0=3}.png".format(result_dir, Animal.pure_id, epoch))
    plt.close()

# main used
def analyze(label, animal, result_dir, traj_id, epoch):
    Animal = set_animal(traj_id, label, animal)
    Plot_Calc_Data_by_label(Animal, result_dir, epoch)
