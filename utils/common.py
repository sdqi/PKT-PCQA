import numpy as np
import torch

def setup_seed(seed):#为保证结果可复现，设置模型初始化的随机数种子
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return: (B,M,N)
    '''
    B, M, C = points1.shape 
    _, N, _ = points2.shape

    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N) #torch.pow求幂运算，一个中心点对应多个周围的点
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1)) #dists的shape = (B,M,N)每个中心点与其他点的距离
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    #torch.where 第一个是判断条件，第二个是符合条件的设置值，第三个是不满足条件的设置值。
    return torch.sqrt(dists).float()

def pre_get_dists(points1_np, points2_np):
    '''
    Calculate dists between two group points
    :param points1_np: shape=(M, C)
    :param points2_np: shape=(N, C)
    :return: (M,N)
    '''
    M, C = points1_np.shape 
    N, _ = points2_np.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points1 = torch.from_numpy(points1_np).to(device)
    points2 = torch.from_numpy(points2_np).to(device)

    a = torch.sum(torch.pow(points1, 2), dim=-1).view(M, 1).cpu()
    b = torch.sum(torch.pow(points2, 2), dim=-1).view(1, N).cpu()
    dists = a + b #torch.pow求幂运算，一个中心点对应多个周围的点
    del a,b
    
    # dists -= 2 * torch.matmul(points1.cpu(), points2.permute(1, 0).cpu()) #dists的shape = (B,M,N)每个中心点与其他点的距离
    dists -= 2 * torch.matmul(points1.cpu().to(torch.float32), points2.permute(1, 0).cpu().to(torch.float32))
    # dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    #torch.where 第一个是判断条件，第二个是符合条件的设置值，第三个是不满足条件的设置值。
    return torch.sqrt(dists).float().cpu()

def gather_points(points, inds):
    '''
    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])#开始位置包含
    repeat_shape = list(inds.shape)     
    repeat_shape[0] = 1
    # repeat 指定在特定维度下重复的次数
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]