"""this is a module for getting traj data"""
import glob
import pandas as pd
import numpy as np
import datetime
import pickle

def get_traj(lat, lon, animal):
    """select dir"""
    if animal == "bird":
        dir_ = "bird_data/1min_median_label"
    elif animal == "cel":
        dir_ = "worm_data/data_by_individual"

    ls_csv = sorted(glob.glob("../"+dir_+"/*.csv"))
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
                                timestamp], axis=1)
                    df_traj = df_traj.reset_index(drop=True)

    elif animal == "cel":

        for i, csv in enumerate(ls_csv):
            tmp = pd.read_csv(csv)
            df_traj = pd.concat([df_traj, tmp[lat], tmp[lon], tmp["time(sec)"]], axis=1)

    return df_traj

def animal_path(animal):
    if animal == "bird":
        path = "../bird_data/bird_df_traj.pkl"

    elif animal == "cel":
        path = "../worm_data/worm_df_traj.pkl"

    return path

def save_traj(animal, df_traj):
    path = animal_path(animal)
    df_traj.to_pickle(path)


def load_traj(animal):
    path = animal_path(animal)
    return pd.read_pickle(path)

def norm_traj(df_traj):
    for j in range(df_traj.shape[1]):
        """Rescale time"""
        if (j + 1) % 3 == 0:
            scale = 1
        else:
            scale = 1

        df_traj.iloc[:, j] = \
            (df_traj.iloc[:, j] - df_traj.iloc[:, j].min()) / (df_traj.iloc[:, j].max() - df_traj.iloc[:, j].min()) * scale

    return df_traj
