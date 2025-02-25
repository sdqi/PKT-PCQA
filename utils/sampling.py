import torch
from utils.common import get_dists

def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device) #建立B行M列的矩阵，用于存放M个采样点的索引
    dists = torch.ones(B, N).to(device) * 1e5 #设置一个很大的距离值，大小为B行N列，作用？？？
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device) #生成[0,N)范围内的随机数，共B个
    batchlists = torch.arange(0, B, dtype=torch.long).to(device) #生成[0,B)范围内的整数序列，维度为B
    for i in range(M):
        centroids[:, i] = inds#用于存放Batch中每个点云集的起始点索引
        cur_point = xyz[batchlists, inds, :] # (B, 3)，保存的是batch中每个集合的初始点坐标
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)#unsqueeze拓展维度，1为列拓展使得curpoint与xyz维度一致，squeeze降低维度
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    # print(centroids)
    return centroids