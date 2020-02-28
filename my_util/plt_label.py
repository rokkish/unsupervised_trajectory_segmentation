"""this is plot func"""
import os
import glob

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
from PIL import Image

import pandas as pd
import numpy as np
from scipy.stats import entropy

from my_util import do_kmeans_InsteadOfSlic as kmios


#"""these function is used in main() """
def plot_label(label, lat, lon, result_dir, trip_no, epoch):
    """plot all label"""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(lat.iloc[:, 0], lon.iloc[:, 0], c=label / max(label), cmap="tab20", s=16)
    plt.title("Segmentation")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.plot(lat.iloc[:, 0], lon.iloc[:, 0], color="white", alpha=0.2)
    plt.scatter(lat.iloc[:, 0], lon.iloc[:, 0],
                c=np.linspace(0, lat.shape[0], lon.shape[0]), cmap="magma", s=16)
    plt.title("Time color")
    plt.colorbar()

    plt.savefig("./result/{}/segment_trip{:0=3}_epoch{:0=3}.png".format(result_dir, trip_no, epoch))
    #plt.savefig("./"+result_dir+"/"+"{:0=3}".format(trip_no)+"_all.png")
    plt.close()
    #plt.show()

    """plot each label"""
    df_label = pd.DataFrame(label)
    df_tmp = pd.concat([lat, lon, df_label], axis=1)
    df_tmp.columns = ["lat", "lon", "label"]

    n = len(pd.unique(label))
    n_split = n + 4

    plt.figure(figsize=(15, 10))
    plt.rcParams["font.size"] = 8
    for i in range(n):
        plt.subplot(4, int(n_split/4), i+1)

        plt.plot(lat.iloc[:, 0], lon.iloc[:, 0], color="gray", alpha=0.4)

        plt.scatter(df_tmp.loc[df_tmp["label"] == pd.unique(label)[i]]["lat"],
                    df_tmp.loc[df_tmp["label"] == pd.unique(label)[i]]["lon"], s=8)

        indexs_list = df_tmp.loc[df_tmp["label"] == pd.unique(label)[i]].index.values
        plt.text(0.5, 0.5, "Time from {} to{}".format(min(indexs_list), max(indexs_list)), alpha=0.5)

        plt.title(str(i))

    plt.savefig("./result/{}/each_segment_trip{:0=3}_epoch{:0=3}.png".format(result_dir, trip_no, epoch))
    #plt.savefig("./result/each_segment.png")
    #plt.savefig("./"+result_dir+"/"+"{:0=3}".format(trip_no)+"_each.png")
    plt.close()
    #plt.show()

#"""these function is used in run() """
def plot_entropy_at_each_pixel(output, batch_idx, args, file_name):
    """plot entropy"""
    for i in range(output.shape[1]):
        tmp = output[:, i]
        mat_i = torch.stack((tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp), dim=0)
        mat_i = torch.cat((mat_i, mat_i), dim=0)
        if i == 0:
            mat = mat_i
        else:
            mat = torch.cat((mat, mat_i), dim=0)

    mat = pd.DataFrame(mat.detach().cpu().numpy())
    mat = (mat - mat.min()) / (mat.max() - mat.min())
    mat = torch.from_numpy(mat.values)
    plt.figure(figsize=(10, 10))
    plt.imshow(mat)
    #plt.colorbar()
    plt.title("plt entropy at each pixel: " + str(batch_idx))
    plt.xlabel("t")
    plt.ylabel("class [0-31]*16")
    plt.yticks(np.arange(0, mat.shape[0], 16))
    plt.savefig("./result/{}/{}_{:0=3}.png".format(args.result_dir, file_name, batch_idx))
    #plt.show()
    plt.close()

def save_gif(args, number_traj, epoch, file_name="output"):

    files = sorted(glob.glob("result/" + args.result_dir + "/" + file_name + "_*.png"))
    images = list(map(lambda file: Image.open(file), files))
    images[0].save("result/{}/{}_traj{:0=3}_epoch{:0=3}.gif".format(args.result_dir, file_name, number_traj, epoch),\
        save_all=True, append_images=images[1:], duration=500, loop=1)

    def remove_glob(files):
        for path in files:
            os.remove(path)

    remove_glob(files)

def plot_entropy_at_each_batch(output, traj, len_traj, args, batch_idx):
    """ plot entropy each batch"""
    output_ = output.detach().cpu().numpy().copy()
    tmp = output_ - output_.min(1).reshape([len_traj, 1])
    tmp = tmp / tmp.sum(1).reshape([len_traj, 1])
    ent = []
    for t in range(len_traj):
        if np.isnan(traj[args.lat][t]):
            ent.append(traj[args.lat][t])
        else:
            #tmpt = np.sort(tmp[t,:])[::-1]
            #ent.append(tmpt[0]-tmpt[1])
            ent.append(entropy(tmp[t, :32]))

    #entropy all plot
    plt.figure(figsize=(8, 5))
    plt.scatter(traj[args.lat], traj[args.lon], c=ent)
    plt.title("entropy")
    plt.colorbar()
    plt.savefig("./result/"+args.result_dir+"/entropy_"+str(batch_idx)+".png")
    plt.close()
    # plt.show()
