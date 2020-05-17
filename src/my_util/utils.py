""" collect many functions """
import sys
import torch.nn as nn
from my_util.models.segnet_model import MyNet
from my_util import my_lossfn

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
