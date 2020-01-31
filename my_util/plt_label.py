"""this is plot func"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from my_util import do_kmeans_InsteadOfSlic as kmios

#"""these function is used in main() """
def plot_freq(label, loss):
    """view freq label"""
    bins = len(pd.unique(label))
    fig = plt.figure(figsize=(15, 5))
    plt.style.use('dark_background')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(label, bins=bins)
    ax1.set_ylabel("freq")
    ax1.set_xlabel("no.label")
    ax1.set_title("freq of label")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(loss)
    ax2.set_ylabel("loss")
    ax2.set_xlabel("epochs")
    ax2.set_title("loss")
    #plt.close()
    plt.show()

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

    #TODO:mkdir
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
    for i in range(n):
        plt.subplot(4, int(n_split/4), i+1)
        plt.plot(lat.iloc[:, 0], lon.iloc[:, 0], color="gray", alpha=0.4)
        plt.scatter(df_tmp.loc[df_tmp["label"] == pd.unique(label)[i]]["lat"],
                    df_tmp.loc[df_tmp["label"] == pd.unique(label)[i]]["lon"], s=8)
        plt.title(str(i))

    plt.savefig("./result/{}/each_segment_trip{:0=3}_epoch{:0=3}.png".format(result_dir, trip_no, epoch))
    plt.savefig("./result/each_segment.png")
    #plt.savefig("./"+result_dir+"/"+"{:0=3}".format(trip_no)+"_each.png")
    plt.close()
    #plt.show()

def plot_ouput_entropy(model_output, trajectory_i, args, entropy):
    """plot model output entropy"""
    output = model_output.detach().cpu().numpy().copy()
    tmp = output - output.min(1).reshape([600, 1])
    tmp = tmp  / tmp.max(1).reshape([600, 1])
    ent = []
    for t in range(600):
        if np.isnan(trajectory_i[args.lat][t]):
            ent.append(trajectory_i[args.lat][t])
        else:
            tmpt = np.sort(tmp[t, :])[::-1]
            ent.append(tmpt[0]-tmpt[1])
            # ent.append(entropy(tmp[t,:32]))
        plt.scatter(trajectory_i[args.lat], trajectory_i[args.lon], c=ent)
        plt.title("sub_1st_2nd")
        plt.colorbar()
        plt.savefig("./"+args.result_dir+"/sub_firstsecond.png")
        #plt.close()
        plt.show()
    tmp = output - output.min(1).reshape([600, 1])
    tmp = tmp  / tmp.sum(1).reshape([600, 1])
    ent = []
    for t in range(600):
        if np.isnan(trajectory_i[args.lat][t]):
            ent.append(trajectory_i[args.lat][t])
        else:
            #tmpt = np.sort(tmp[t,:])[::-1]
            #ent.append(tmpt[0]-tmpt[1])
            ent.append(entropy(tmp[t, :32]))
        plt.figure(figsize=(8, 5))
        plt.scatter(trajectory_i[args.lat], trajectory_i[args.lon], c=ent)
        plt.title("entropy")
        plt.colorbar()
        plt.savefig("./"+args.result_dir+"/entropy.png")
        #plt.close()
        plt.show()

def plot_only_kmeans(traj_i, lat, lon, args):
    """plot segmentation results with kmeans"""
    ret_seg_map = kmios.do_kmeans(k=10, traj=traj_i)
    seg_map = ret_seg_map.flatten()
    plt.figure(figsize=(10, 10))
    plt.scatter(lat.iloc[:, 0], lon.iloc[:, 0], c=seg_map / max(seg_map), cmap="tab20", s=16)
    plt.title("Only Kmeans")
    #ax = plt.colorbar()
    
    plt.savefig("./result/only_k.png")

    #plt.savefig("./"+args.result_dir+"/only_k.png")
    plt.close()
    #plt.show()

#"""these function is used in run() """

def plot_initial_kmeans_segmentation_results(df_traj_i, args):
    """ plot initial kmeans Segmentation"""
    plt.figure(figsize=(10, 5))
    plt.scatter(df_traj_i[args.lat], df_traj_i[args.lon], cmap="tab20", s=16)
    plt.title("Kmeans")
    plt.savefig("./"+args.result_dir+"/k_.png")
    plt.close()
    #plt.show()

def plot_entropy_at_each_pixel(output, train_epoch, batch_idx):
    """ plot entropy"""
    if batch_idx == train_epoch-1:
        #print(output.shape)
        plt.figure(figsize=(10, 5))
        plt.imshow(output.detach().cpu())
        plt.colorbar()
        plt.title("plt entropy at each pixel")
        plt.close()
        #plt.show()

def plot_segmentresult_each_batch(im_target, df_traj_i, args, batch_idx, file_name):
    """ Plot segment traj colored by cluster"""
    if batch_idx%5 == 0 and batch_idx < 30:
        #print("batch %.1f"%(batch_idx), end="")
        plt.figure(figsize=(10, 5))
        plt.style.use('dark_background')
        plt.scatter(df_traj_i[args.lat], df_traj_i[args.lon], c=im_target, cmap="tab20", s=16)
        plt.title("Segmentation:"+str(batch_idx))
        #ax = plt.colorbar()
        plt.title("Plot" + str(file_name) + "refine")
        plt.savefig("./"+args.result_dir+"/"+str(file_name)+"_refine"+str(batch_idx)+".png")
        plt.close()
        #plt.show()
    else:
        pass

def plot_entropy_at_each_batch(output, df_traj_i, args, batch_idx, entropy):
    """ plot entropy each batch"""
    output_ = output.detach().cpu().numpy().copy()
    tmp = output_ - output_.min(1).reshape([600, 1])
    tmp = tmp / tmp.sum(1).reshape([600, 1])
    ent = []
    for t in range(600):
        if np.isnan(df_traj_i[args.lat][t]):
            ent.append(df_traj_i[args.lat][t])
        else:
            #tmpt = np.sort(tmp[t,:])[::-1]
            #ent.append(tmpt[0]-tmpt[1])
            ent.append(entropy(tmp[t, :32]))
    #entropy all plot
    plt.figure(figsize=(8, 5))
    plt.scatter(df_traj_i[args.lat], df_traj_i[args.lon], c=ent)
    plt.title("entropy")
    plt.colorbar()
    plt.savefig("./"+args.result_dir+"/entropy_"+str(batch_idx)+".png")
    plt.close()
    # plt.show()

def plot_num_of_same_label(ls_target_idx):
    """ plot num same label all epoch"""
    plt.figure(figsize=(3, 3))
    plt.style.use('dark_background')
    plt.plot(np.array(ls_target_idx))
    plt.title("num of same label")
    plt.xlabel("epoch")
    #plt.close()
    plt.show()
