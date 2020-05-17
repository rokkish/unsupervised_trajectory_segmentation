"""this is main program of segnet with kmeans(use one trajectory)"""
#!/usr/bin/env python
# coding: utf-8
# conda envs==py35

import argparse
import time
import os
import numpy as np
import pandas as pd
import torch

from my_util import get_traj, plt_label, set_hypara, sec_argmax, Args
from my_util import do_kmeans_InsteadOfSlic, utils, analyze_segmentation

import matplotlib.pyplot as plt

import get_logger
logger = get_logger.get_logger(name='train_model')

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def run(traj, len_traj, args, model, optimizer, number_traj, epoch):
    """
    Run algorithm at each trajectory
        traj        :   [np.array]      :   train data and reshape into tensor
        len_traj    :   [int]           :   len of traj without None
        args        :   [class]         :   hyper params
        model       :   [model]         :
        optimizer   :   [optimizer]     :
    """

    def reshape_train_data(traj):
        # reshape
        tensor = traj.transpose((2, 0, 1))
        # float
        tensor = tensor.astype(np.float32)
        # reshape
        tensor = tensor[np.newaxis, :, :, :]
        tensor = torch.from_numpy(tensor).to(device)
        return tensor

    def load_timestamp(len_traj, args, number_traj):
        dummy_label = [0]*len_traj
        Animal = analyze_segmentation.set_animal(traj_id=number_traj, label=dummy_label, animal=args.animal)
        #TODO:column nameがAnimalで異なる
        return Animal.load_df_timestamp().dropna().reset_index(drop=True)

    logger.debug("start segmentation:")

    # TODO:遅いので，事前に全軌跡のKmeansを算出，保存しといて，参照だけにする
    ret_seg_map = do_kmeans_InsteadOfSlic.handle_kmeans(\
        k=int(len_traj/4), traj=traj, window=args.window, time_dim=args.time_dim)

    seg_map = ret_seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0] for u_label in np.unique(seg_map)]

    logger.debug("end segmentation:")

    tensor = reshape_train_data(traj)
    timestamp = load_timestamp(len_traj, args, number_traj)

    '''train loop'''

    model.train()

    # loss every batch
    ret_loss = []

    for BATCH_IDX in range(args.train_epoch):

        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)
        output = output[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)

        if args.sec_argmax:
            target_idx = sec_argmax.get_idx_samelabel(target)
            target = sec_argmax.get_argsecmax(output, target, target_idx, BATCH_IDX)

        plt_label.plot_entropy_at_each_pixel(output, BATCH_IDX, args, "output")

        im_target = target.data.cpu().numpy()

        """refine"""
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        """backward"""
        target = torch.from_numpy(im_target)
        target = target.to(device)

        """get loss"""
        loss = utils.set_loss_functtion(output=output, target=target, args=args, timestamp=timestamp)
        loss.backward()
        optimizer.step()

        un_label = np.unique(im_target, )

        ret_loss.append(loss.item())

        logger.debug("Trajectory epoch: %d/%d, loss: %1.5f" % \
            (BATCH_IDX, args.train_epoch, loss.item()))

        if len(un_label) < args.MIN_LABEL_NUM:
            break
        
        #plt_label.plot_entropy_at_each_batch(output, traj, len_traj, args, BATCH_IDX)

    plt_label.save_gif(args, number_traj, epoch)

    return im_target, ret_loss

def train(args, traj, model, optimizer):

    def reshape_traj(i):
        traj_ = traj[:, i, :]
        traj_ = traj_[~np.isnan(traj_).any(axis=1)]
        traj_ = traj_[:, np.newaxis, :]
        return traj_

    def get_latlon(i):
        lat_i = pd.DataFrame(traj[:, i, 0]).dropna(how="all")
        lat_i = lat_i.reset_index(drop=True)
        lon_i = pd.DataFrame(traj[:, i, 1]).dropna(how="all")
        lon_i = lon_i.reset_index(drop=True)
        return lat_i, lon_i

    def plt_loss(loss_all, traj_id):
        plt.plot(loss_all)
        plt.ylabel("loss")
        plt.xlabel("batches")
        plt.title("loss")
        plt.savefig("./result/{}/loss_trip{:0=3}.png".format(args.result_dir, traj_id))

    def get_label_bykmeans(traj_):
        label_kmeans = do_kmeans_InsteadOfSlic.do_kmeans(args.k, traj_)
        return label_kmeans.flatten()

    def plt_kmeans(label_kmeans):
        analyze_segmentation.analyze(label_kmeans, args.animal, args.result_dir, i, 100*args.k)
        analyze_segmentation.plot_relabel(label_kmeans, lat_i, lon_i, args.animal, args.result_dir, i, 100*args.k)

    for i in range(args.START_ID, args.END_ID):

        loss_all = []

        del model, optimizer

        model = utils.set_network_model(network=args.network,\
            mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2, device=device)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        logger.debug("Trajectory Num: %d/%d" % (i - args.START_ID, args.END_ID - args.START_ID))

        traj_ = reshape_traj(i)

        if traj_.shape[0] < 10:
            # assert(ValueError("Too short data"))
            continue

        """get lat, lon for plot only kmeans"""
        lat_i, lon_i = get_latlon(i)
        if "kmeans" in args.d:
            label_kmeans = get_label_bykmeans(traj_)
            plt_kmeans(label_kmeans)

        for e in range(args.epoch_all):

            logger.info("Train epoch: %d/%d " % (e + 1, args.epoch_all))

            """ run """
            length_traj_ = lat_i.shape[0]
            if args.plot_mode:
                label = torch.load("{}/trip{:0=3}.pkl".format(args.label_dir, i))

            else:
                label, loss = run(traj_, length_traj_, args, model, optimizer, i, e)
                loss_all.extend(loss)

            """plot result seg"""
            analyze_segmentation.plot_relabel(label, lat_i, lon_i, \
                args.animal, args.result_dir, i, e)

            analyze_segmentation.analyze(label, args.animal, args.result_dir, i, e)

            if not args.plot_mode:
                break

        torch.save(label, "result/{}/trip{:0=3}.pkl".format(args.result_dir, i))

        plt_loss(loss_all, traj_id=i)

    torch.save(model.state_dict(), "./models/model.pkl")


def load_data(args):
    """preprocessed data"""
    if os.path.exists(get_traj.animal_path(args.animal)):
        logger.debug("Load df")
        df_trajectory = get_traj.load_traj(args.animal)
        df_trajectory = get_traj.norm_traj(df_trajectory)
    else:
        logger.debug("Make df, Save")
        df_trajectory = get_traj.get_traj(args.lat, args.lon, args.animal)
        get_traj.save_traj(args.animal, df_trajectory)

    """reshape data"""
    traj = df_trajectory.values
    traj = np.reshape(traj, (traj.shape[0], int(traj.shape[1]/3), 3))
    return traj

def main(args):
    """do trian(), run() and plot results for all trajectory"""

    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU_ID)

    traj = load_data(args)

    model = utils.set_network_model(network=args.network,\
        mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2, device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train(args, traj, model, optimizer)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="this is Segnet with Kmeans")

    # data
    PARSER.add_argument("--animal", choices=["cel", "bird"], default="bird")

    # train
    PARSER.add_argument("-e", "--epoch", type=int, default=2**5, help="BATCH: num of trainging with one trajectory")
    PARSER.add_argument("--epoch_all", type=int, default=2**3, help="EPOCH: num of training with all trajectories")

    # hypara of custom loss
    PARSER.add_argument("--alpha", type=float, default=10, help="to be bigger, enlarge d. To be smaller, ensmaller d")
    PARSER.add_argument("--lambda_p", type=float, default=0.01)
    PARSER.add_argument("--tau", type=float, default=100, help="to be smaller, enlarge w.")

    # on/off custom module
    PARSER.add_argument("--time", action="store_true")
    PARSER.add_argument("--myloss", action="store_false")
    PARSER.add_argument("--secmax", action="store_true")

    # set traj id
    PARSER.add_argument("--start", type=int, default=0, help="this is start_id")
    PARSER.add_argument("--end", type=int, default=5, help="this is end_id")

    # select network
    PARSER.add_argument("--net", type=str, default="segnet")

    # train high param
    PARSER.add_argument("--lr", type=float, default=0.05, help="learing rate")
    PARSER.add_argument("--momentum", type=float, default=0.9, help="learing rate")

    PARSER.add_argument("-d", default="", help="header of result dir name, result/d~")
    PARSER.add_argument("-k", type=int, default=10, help="set k of kmeans for compared method")
    PARSER.add_argument("--plot_mode", action="store_true")
    PARSER.add_argument("--label_dir", 
        default="result/paper_xmodify_bird_xyt_sec_arg_myloss_a0.1_p0.01_tau10000.0", 
            help="set dir name of load label")

    ARGS_TMP = PARSER.parse_args()
    ARGSE = Args()

    # HACK: this is too uneffectively.
    # translate from input arg to default arg
    ARGSE = set_hypara.set_hypara(ARGSE, ARGS_TMP)

    logger.info("Train Start")

    main(ARGSE)

    logger.info("Train End")
