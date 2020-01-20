""" loss """
import torch
import torch.nn as nn
import numpy as np

def my_penalty(outputs, labels, alpha, lambda_p, Tau):
    # outputs =(time, channel)
    # x =(batch, channel, time)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    x = outputs[np.newaxis, :, :]
    x = x.transpose(1, 2)
    T = x.shape[2]
    weight = torch.zeros([T, T])
    weight = weight.to(device)
    for t in range(T):
        lag = torch.arange(T) - t
        # lag = lag / (2*T+1.0)
        lag = lag.to(device)
        lag = torch.exp(torch.sqrt(lag.float()**2) / Tau) - 1.0
        weight[t, :] = lag

    # 全時刻間のユークリッド距離行列の逆指数(距離が近いと大きくなる)
    x = x.transpose(1, 2)
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(2, 1)
    dist = torch.exp(- pairwise_distance / alpha)


    #print(pairwise_distance.shape)
    """
    plt.imshow(pairwise_distance.detach().cpu()[0,...])
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.imshow(dist.detach().cpu()[0,...])
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.imshow(weight.detach().cpu())
    plt.colorbar()
    plt.show()
    """


    """ペナルティ項"""
    dist = dist.to(device)
    penalty = int(torch.sum(weight * dist[0, :, :]).item())
    #print(penalty)

    """CrossEntropyloss"""
    ce_fn = nn.CrossEntropyLoss(ignore_index=0)
    ce_loss = ce_fn(outputs, labels)
    #print("penalty:",lambda_p*penalty)

    return ce_loss + lambda_p * penalty
