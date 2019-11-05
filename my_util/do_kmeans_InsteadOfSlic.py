import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def plt_kmeans(pred, df):
    plt.figure(figsize=(5, 5))
    plt.style.use('dark_background')
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=pred, cmap="tab20", s=5)
    plt.title("Kmeans instead of SLIC color")
    #ax=plt.colorbar()
    plt.show()

def do_kmeans(k, traj):
    if k == 1:
        pass
    else:
        df_traj = pd.DataFrame(traj[:, 0, :])
        df_traj = df_traj.fillna(0)
        # idx = np.random.permutation(df_traj.shape[0])[:k]
        # init = df_traj.iloc[idx, :]
        pred = KMeans(n_clusters=k).fit_predict(df_traj.values)
        return pred

def do_kmeans_window(k, traj, window):
    if k == 1:
        pass
    else:
        df_traj = pd.DataFrame(traj[:, 0, :])
        df_traj = df_traj.fillna(0)
        # idx = np.random.permutation(df_traj.shape[0])[:k]
        # init = df_traj.iloc[idx, :]
        # pred = KMeans(n_clusters=k, init=init).fit_predict(df_traj.values)
        s = 0
        e = window
        # k < window
        k = window//2
        print("k of kmeans is ", k)
        for i in range(df_traj.shape[0]//window):
            if i == 0:
                pred = np.array(KMeans(n_clusters=k).fit_predict(df_traj.values[s:e]))
            else:
                tmp_pred = KMeans(n_clusters=k).fit_predict(df_traj.values[s:e])
                tmp_pred = np.array(tmp_pred)+num_label
                pred = np.append(pred, tmp_pred)
            num_label = len(set(pred))
            s += window
            e += window
            if e > df_traj.shape[0]:
                e = df_traj.shape[0]-1
        return pred

def handle_kmeans(k, traj, window, time_dim):
    """ SELECT
        1.vanilla kmeans
        2.timewindoww kmeans
    """
    if time_dim:
        ret_seg_map = do_kmeans(k=k, traj=traj)
    else:
        ret_seg_map = do_kmeans_window(k=k, traj=traj, window=window)

    return ret_seg_map
