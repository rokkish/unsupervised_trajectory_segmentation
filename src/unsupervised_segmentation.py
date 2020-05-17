"""this is main program of segnet with kmeans(use one trajectory)"""
#!/usr/bin/env python
# coding: utf-8
# conda envs==py35

import os
import numpy as np
import torch

from my_util.parameters import set_hypara
from my_util.parameters.my_args import args
from my_util import utils, get_logger, Trainer

logger = get_logger.get_logger(name='train_model')

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def main(args):
    """do trian(), run() and plot results for all trajectory"""

    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU_ID)

    traj = utils.load_data(args)

    model = utils.set_network_model(network=args.network,\
        mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2, device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    trainer = Trainer(model, optimizer, device, args)
    trainer.fit(traj)

if __name__ == '__main__':

    args = set_hypara.main(args)

    logger.info("Train Start")

    main(args)

    logger.info("Train End")
