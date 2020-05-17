"""
    Trainer model
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from my_util.visualization import plt_label, analyze_segmentation
from my_util.features import sec_argmax, kmeans
from my_util import utils, get_logger
logger = get_logger.get_logger(name='trainer')

class Trainer():
    def __init__(self, model, optimizer, device, args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.device = device
        self.args = args
    
    def reinit_model(self):
        del self.model, self.optimizer

        model = utils.set_network_model(network=self.args.network,\
            mod_dim1=self.args.mod_dim1, mod_dim2=self.args.mod_dim2, device=self.device)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        self.model = model
        self.optimizer = optimizer

    def fit(self, traj):
        """
            for traj_i in (start, end):
                for epoch in (max_epoch):
                    run()
        """
        args = self.args

        for i in range(args.START_ID, args.END_ID):

            loss_all = []

            logger.debug("Trajectory Num: %d/%d" % (i - args.START_ID, args.END_ID - args.START_ID))

            traj_i = self.sampling_traji(i, traj)
            lat_i, lon_i = self.get_latlon(traj_i)

            if traj_i.shape[0] < 10:
                logger.info("Too short data:{}".format(traj_i.shape))
                continue

            if i > args.START_ID:
                self.reinit_model()

            if "kmeans" in args.d:
                label_kmeans = self.get_label_bykmeans(traj_i)
                self.plt_kmeans(label_kmeans)

            for e in range(args.epoch_all):

                logger.info("Train epoch: %d/%d " % (e + 1, args.epoch_all))

                """ run """
                length_traj_i = lat_i.shape[0]
                if args.plot_mode:
                    label = torch.load("{}/trip{:0=3}.pkl".format(args.label_dir, i))

                else:
                    label, loss = self.run(traj_i, length_traj_i, i, e)
                    loss_all.extend(loss)

                """plot result seg"""
                analyze_segmentation.plot_relabel(label, lat_i, lon_i, \
                    args.animal, args.result_dir, i, e)

                analyze_segmentation.analyze(label, args.animal, args.result_dir, i, e)

                if not args.plot_mode:
                    break

            torch.save(label, "result/{}/trip{:0=3}.pkl".format(args.result_dir, i))

            self.plt_loss(loss_all, traj_id=i)

        torch.save(self.model.state_dict(), "./models/model.pkl")

    @staticmethod
    def sampling_traji(i, traj):
        """Sample traj_i
            Input : (MaxLength, sample, features)
            Return: (MinLength,      1, features)
        """
        traj_i = traj[:, i, :]
        traj_i = traj_i[~np.isnan(traj_i).any(axis=1)]
        traj_i = traj_i[:, np.newaxis, :]
        return traj_i

    @staticmethod
    def get_latlon(traj_i):
        """Return x, y
            Input : (MinLength, 1, features{x, y, t})
            Return: (MinLength, xory)
        """
        lat_i = pd.DataFrame(traj_i[:, 0, 0]).dropna(how="all")
        lat_i = lat_i.reset_index(drop=True)
        lon_i = pd.DataFrame(traj_i[:, 0, 1]).dropna(how="all")
        lon_i = lon_i.reset_index(drop=True)
        return lat_i, lon_i

    def plt_loss(self, loss_all, traj_id):
        plt.plot(loss_all)
        plt.ylabel("loss")
        plt.xlabel("batches")
        plt.title("loss")
        plt.savefig("./result/{}/loss_trip{:0=3}.png".format(self.args.result_dir, traj_id))

    def get_label_bykmeans(self, traj):
        label_kmeans = kmeans.do_kmeans(self.args.k, traj)
        return label_kmeans.flatten()

    def plt_kmeans(self, label_kmeans):
        analyze_segmentation.analyze(label_kmeans, self.args.animal, self.args.result_dir, i, 100*self.args.k)
        analyze_segmentation.plot_relabel(label_kmeans, lat_i, lon_i, self.args.animal, self.args.result_dir, i, 100*self.args.k)

    def run(self, traj, len_traj, number_traj, epoch):
        """
        Run algorithm at each trajectory
            traj        :   [np.array]      :   train data and reshape into tensor
            len_traj    :   [int]           :   len of traj without None
            number_traj :   [int]           :   number of trajectory
            epoch       :   [int]           :   epoch at a trajectory
        """
        args, model, optimizer = self.args, self.model, self.optimizer

        def reshape_train_data(traj):
            # reshape
            tensor = traj.transpose((2, 0, 1))
            # float
            tensor = tensor.astype(np.float32)
            # reshape
            tensor = tensor[np.newaxis, :, :, :]
            tensor = torch.from_numpy(tensor).to(self.device)
            return tensor

        def load_timestamp(len_traj, args, number_traj):
            dummy_label = [0]*len_traj
            Animal = analyze_segmentation.set_animal(traj_id=number_traj, label=dummy_label, animal=args.animal)
            #TODO:column nameがAnimalで異なる
            return Animal.load_df_timestamp().dropna().reset_index(drop=True)

        logger.debug("start segmentation:")

        # TODO:遅いので，事前に全軌跡のKmeansを算出，保存しといて，参照だけにする
        ret_seg_map = kmeans.handle_kmeans(\
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
            target = target.to(self.device)

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

        plt_label.save_gif(args, number_traj, epoch)

        return im_target, ret_loss
