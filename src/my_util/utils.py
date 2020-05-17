""" collect many functions """
import sys
import os
import numpy as np
import torch.nn as nn
from my_util.models.segnet_model import MyNet
from my_util.models import my_lossfn
from my_util.features import get_traj
from my_util import get_logger
logger = get_logger.get_logger(name='utils')

def set_network_model(network, mod_dim1, mod_dim2, device):
    """ manage (segnet, unet, other network) """
    if network == "segnet":
        model = MyNet(inp_dim=3, mod_dim1=mod_dim1, mod_dim2=mod_dim2).to(device)
    else:
        print("Undefined network:" + network)
        sys.exit()
    return model

def set_loss_functtion(output, target, args, timestamp):
    """ manage (loss + penalty, crossentropy loss) """
    if not args.crossonly:
        loss = my_lossfn.my_penalty(output, target, \
            args.alpha, args.lambda_p, args.Tau, timestamp)
    else:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
    return loss

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
