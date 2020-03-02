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

    logger.debug(dic)
    logger.debug(ret)
    return ret

class Worm(object):
    
    def __init__(self, traj_id, label):
        self.pure_id = traj_id
        self.label = label
        self.df_worm = load_df("../worm_data/worm_df_all_data.pkl", "x", "y", "cel")
        self.label_trans_list = label_trans(self.label)
        self.re_label_list = re_label(self.label)

    def calc_data(self):

        df_speed, df_acc, df_angle = self.load_df_speed_acc_angle()
        df_turn = self.load_df_turn()
        df_timestamp = self.load_df_timestamp().astype("int32")

        df_cat = pd.concat([df_timestamp, df_speed, df_acc, df_angle, df_turn], axis=1)
        df_cat.columns = ["timestamp", "speed", "acc", "angle", "turn"]

        df_cat = df_cat.dropna().reset_index(drop=True)
        return df_cat.drop("timestamp", axis=1), df_cat["timestamp"]

    def load_df_speed_acc_angle(self):
        df_dx = self.load_df_dx()
        df_dy = self.load_df_dy()

        df_dx_delay = pd.concat([df_dx.iloc[1:], pd.Series(df_dx.iloc[-1])], axis=0).reset_index(drop=True)
        df_dy_delay = pd.concat([df_dy.iloc[1:], pd.Series(df_dy.iloc[-1])], axis=0).reset_index(drop=True)

        df_speed = np.sqrt(np.power(df_dx, 2) + np.power(df_dy, 2))
        df_speed_delay = np.sqrt(np.power(df_dx_delay, 2) + np.power(df_dy_delay, 2))
        df_acc = df_speed.diff()
        df_angle = np.arccos((df_dx * df_dx_delay + df_dy * df_dy_delay) / (df_speed * df_speed_delay))
        df_angle = np.rad2deg(df_angle)

        return df_speed.reset_index(drop=True), df_acc.reset_index(drop=True), df_angle

    def load_df_dx(self):
        return self.df_worm["x"].iloc[:, self.pure_id].diff().reset_index(drop=True)
    def load_df_dy(self):
        return self.df_worm["y"].iloc[:, self.pure_id].diff().reset_index(drop=True)
    def load_df_turn(self):
        return self.df_worm["turn"].iloc[:, self.pure_id].reset_index(drop=True)
    def load_df_timestamp(self):
        return self.df_worm["time(sec)"].iloc[:, self.pure_id]

    def load_start_end_index(self, i):
        return self.label_trans_list[i], self.label_trans_list[i+1]
    def load_start_end_timestamp(self, i, s, e, df):
        if i == len(self.label_trans_list) - 2:
            return df.iloc[s], df.iloc[-1]
        else:
            return df.iloc[s], df.iloc[e]

class Bird(object):
    
    def __init__(self, traj_id, label):
        self.pure_id = traj_id
        self.label = label
        self.df_bird = load_df("../bird_data/bird_df_all_data.pkl", "latitude", "longitude", "bird")
        self.label_trans_list = label_trans(self.label)
        self.re_label_list = re_label(self.label)

    def calc_data(self):
        df_speed = self.load_df_speed()
        df_acc = self.load_df_acc()
        df_angle = self.load_df_angle()
        df_label = self.load_df_label()
        df_timestamp = self.load_df_timestamp()

        df_cat = pd.concat([df_timestamp, df_speed, df_acc, df_angle], axis=1)
        df_cat.columns = ["timestamp", "speed", "acc", "angle"]

        if not df_label.empty:

            df_cat = df_cat.assign(activity_label=df_label)

        df_cat = df_cat.dropna().reset_index(drop=True)
        return df_cat.drop("timestamp", axis=1), df_cat["timestamp"]

    def load_df_speed(self):
        return self.df_bird["speed"].iloc[:, self.pure_id].reset_index(drop=True)
    def load_df_acc(self):
        return self.df_bird["speed"].iloc[:, self.pure_id].diff().reset_index(drop=True)
    def load_df_angle(self):
        return self.df_bird["angle"].iloc[:, self.pure_id].reset_index(drop=True)
    def load_df_label(self):
        return self.df_bird["label"].iloc[:, self.pure_id].reset_index(drop=True)
    def load_df_timestamp(self):
        return pd.to_numeric(self.df_bird["timestamp"].iloc[:, self.pure_id], errors="ignore", downcast="integer")

    def load_start_end_index(self, i):
        return self.label_trans_list[i], self.label_trans_list[i+1]
    def load_start_end_timestamp(self, i, s, e, df):
        if i == len(self.label_trans_list) - 2:
            return df.iloc[s], df.iloc[-1]
        else:
            return df.iloc[s], df.iloc[e]

def set_animal(traj_id, label, animal):
    if animal == "bird":
        return Bird(traj_id, label)

    elif animal == "cel":
        return Worm(traj_id, label)

    else:
        raise ValueError("no animal")

def Plot_Calc_Data_by_label(Animal, result_dir, epoch):

    df, df_timestamp = Animal.calc_data()

    fig, ax = plt.subplots(nrows=df.shape[1], ncols=1, figsize=(10, 10), sharex=True)

    cmap = plt.get_cmap("tab20")

    for j, (column_name, item) in enumerate(df.iteritems()):

        def concat_item_and_raw_timestamp(item=item, df_timestamp=df_timestamp):
            df_tmp = item.copy()
            df_tmp.index = df_timestamp
            return df_tmp

        df_cat = concat_item_and_raw_timestamp()
        ax[j].plot(df_cat, color="gray", alpha=0.7)
        ax[j].scatter(x=df_timestamp, y=item, color="gray", s=3, alpha=0.7)

        for i in range(len(Animal.label_trans_list)-1):

            s, e = Animal.load_start_end_index(i)
            raw_s, raw_e = Animal.load_start_end_timestamp(i, s, e, df_timestamp)

            ax[j].axvspan(raw_s, raw_e, alpha=0.35, color=cmap(Animal.re_label_list[s]))

            ax[j].set_ylabel(column_name, fontsize=15)

    ax[df.shape[1]-1].set_xlabel("time (sec)", fontsize=15)

    plt.savefig("result/{}/analyze_trip{:0=3}_epoch{:0=3}.png".format(result_dir, Animal.pure_id, epoch))
    plt.close()

def Plot_Label_Raw_Data(Animal, lat, lon, traj_id, result_dir, epoch):
    """plot all label"""
    df, df_timestamp = Animal.calc_data()

    plt.figure(figsize=(10, 10))

    cmap = plt.get_cmap("tab20")

    for i in range(len(Animal.label_trans_list) - 1):
        s, e = Animal.load_start_end_index(i)
        raw_s, raw_e = Animal.load_start_end_timestamp(i, s, e, df_timestamp)

        logger.debug("%d, %d, %d, %d"%(s, e, raw_s, raw_e))

        plt.plot(lat.iloc[s:e + 1, 0], lon.iloc[s:e + 1, 0], c=cmap(Animal.re_label_list[s]), alpha=0.5)
        plt.scatter(lat.iloc[s:e + 1, 0], lon.iloc[s:e + 1, 0], c=cmap(Animal.re_label_list[s]), s=8, alpha=1)
        plt.text(lat.iloc[s, 0], lon.iloc[s, 0], "{}:{}~{}".format(Animal.re_label_list[s], raw_s, raw_e), \
            color=cmap(Animal.re_label_list[s]), size=18, alpha=0.75)

    plt.title("Segmentation Result", fontsize=15)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)

    plt.savefig("./result/{}/segment_trip{:0=3}_epoch{:0=3}.png".format(result_dir, traj_id, epoch))
    plt.close()

# main used
def analyze(label, animal, result_dir, traj_id, epoch):
    Animal = set_animal(traj_id, label, animal)
    Plot_Calc_Data_by_label(Animal, result_dir, epoch)

def plot_relabel(label, lat, lon, animal, result_dir, traj_id, epoch):
    Animal = set_animal(traj_id, label, animal)
    Plot_Label_Raw_Data(Animal, lat, lon, traj_id, result_dir, epoch)
