""" this is module for sec argmax """
import torch

def get_idx_samelabel(target):
    """ Vanish Reusing Same Label """
    count = 0
    now_label = target[0]
    target_idx = []
    queue_label = []
    for i, t in enumerate(target):
        if t != now_label:
            queue_label.extend([t])
            now_label = t
            count += 1
        if count >= 2:
            if t in queue_label[:-1]:
                target_idx.extend([i])
    return target_idx

def get_argsecmax(output, target, target_idx, batch_idx):
    """ get second max with argmax"""
    #print("Batch id:", batch_idx, ", num of deleted target:", output[target_idx].data.cpu().numpy().shape[0], "\r", end="")
    if len(target_idx) != 0:
        target[target_idx] = torch.argsort(output[target_idx], 1)[:, -2]
        # target[target_idx] = torch.argmax(output[target_idx],1)
    return target
