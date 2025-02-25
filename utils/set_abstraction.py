import torch
import torch.nn as nn
from utils.sampling import fps
from utils.grouping import ball_query
from utils.grouping import knn_cluster ##LQ add it to compare the results of ball_query and knn_query
from utils.common import gather_points
# import ipdb
def init_sample_and_group(xyz, points, use_xyz=True,use_key_points=False):
    '''
        xyz, shape=(B, N, 3),坐标信息，前M个点为关键点
        points, shape=(B, N, 3),颜色信息
        new_xyz, shape=(B, M, 3)
        new_points, shape=(B, M, K, C+3)
    '''
    M = 1024 #簇
    K = 16 #簇中点个数
    B, N, C = points.shape
    local_xyz = xyz[:,M:,:]
    local_rgb = points[:,M:,:]
    new_xyz = xyz[:,:M,:] #以关键点的坐标作为簇的坐标
    # print(local_xyz.shape)
    # ipdb.set_trace()
    grouped_points_xyz = local_xyz.reshape(B,M,K,3)
    grouped_points = local_rgb.reshape(B,M,K,C)
    #是否使用关键点信息
    if use_key_points:
        key_points_xyz = xyz[:,:M,:]
        key_points_rgb  = points[:,:M,:]

    if use_xyz:
        new_points = torch.cat((grouped_points_xyz.float(), grouped_points.float()), dim=-1)
    else:
        new_points = grouped_points
    return new_xyz, new_points

# def init_sample_and_group(xyz, points, use_xyz=True,use_key_points=False):
#     '''
#         xyz, shape=(B, N, 3),坐标信息，前M个点为关键点
#         points, shape=(B, N, 3),颜色信息
#         new_xyz, shape=(B, M, 3)
#         new_points, shape=(B, M, K, C+3)
#     '''
#     M = 1024 #簇
#     K = 16 #簇中点个数
#     B, N, C = points.shape
#     local_xyz = xyz[:,:M,:]
#     local_rgb = points[:,:M,:]
#     new_xyz = xyz[:,:M,:] #以关键点的坐标作为簇的坐标
#     # print(local_xyz.shape)
#     # ipdb.set_trace()
#     grouped_points_xyz = local_xyz.reshape(B,M,K,3)
#     grouped_points = local_rgb.reshape(B,M,K,C)
#     #是否使用关键点信息
#     if use_key_points:
#         key_points_xyz = xyz[:,:M,:]
#         key_points_rgb  = points[:,:M,:]

#     if use_xyz:
#         new_points = torch.cat((grouped_points_xyz.float(), grouped_points.float()), dim=-1)
#     else:
#         new_points = grouped_points
#     return new_xyz, new_points

def sample_and_group(xyz, points, M, radius, K, use_xyz=True):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
            group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
    '''
    new_xyz = gather_points(xyz, fps(xyz, M))
    grouped_inds = ball_query(xyz, new_xyz, radius, K)
    #grouped_inds = knn_cluster(xyz, new_xyz, K)##LQ add it to compare the results of ball_query and knn_query
    grouped_xyz = gather_points(xyz, grouped_inds)
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz

def sample_and_group_all(xyz, points, use_xyz=True):
    '''

    :param xyz: shape=(B, M, 3)
    :param points: shape=(B, M, C)
    :param use_xyz:
    :return: new_xyz, shape=(B, 1, 3); new_points, shape=(B, 1, M, C+3);
             group_inds, shape=(B, 1, M); grouped_xyz, shape=(B, 1, M, 3)
    '''
    B, M, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C)
    grouped_inds = torch.arange(0, M).long().view(1, 1, M).repeat(B, 1, 1)
    grouped_xyz = xyz.view(B, 1, M, C)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz.float(), points.float()], dim=2)
        else:
            new_points = points
        new_points = torch.unsqueeze(new_points, dim=1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz

class PointNet_Pre_SA_Module(nn.Module):
    def __init__(self, in_channels, mlp, bn=True, pooling='max', use_xyz=True, use_key_points=False):
        super(PointNet_Pre_SA_Module, self).__init__()
        self.in_channels = in_channels
        self.mlp = mlp #每个全连接层模块的输出个数 
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.use_key_points = use_key_points
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            # class torch.nn.Conv2d(in_channels,out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))# w*x+b
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels

    def forward(self, xyz, points):
        if self.use_key_points:
            new_xyz, new_points = init_sample_and_group(xyz, points, self.use_key_points)
        else:
            new_xyz, new_points = init_sample_and_group(xyz, points)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())
        if self.pooling == 'avg':
            new_points = torch.mean(new_points, dim=2)
        else:
            new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points

class PointNet_SA_Module(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, group_all, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module, self).__init__()
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp #每个全连接层模块的输出个数，mlp=[64, 64, 128]
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            # class torch.nn.Conv2d(in_channels,out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))# w*x+b
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels
    
    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz=xyz,
                                                                              points=points,
                                                                              M=self.M,
                                                                              radius=self.radius,
                                                                              K=self.K,
                                                                              use_xyz=self.use_xyz)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())
        if self.pooling == 'avg':
            new_points = torch.mean(new_points, dim=2)
        else:
            new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points

class PointNet_SA_Module_MSG(nn.Module):
    def __init__(self, M, radiuses, Ks, in_channels, mlps, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module_MSG, self).__init__()
        self.M = M
        self.radiuses = radiuses
        self.Ks = Ks
        self.in_channels = in_channels
        self.mlps = mlps
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbones = nn.ModuleList()
        for j in range(len(mlps)):
            mlp = mlps[j]
            backbone = nn.Sequential()
            in_channels = self.in_channels
            for i, out_channels in enumerate(mlp):
                backbone.add_module('Conv{}_{}'.format(j, i),
                                         nn.Conv2d(in_channels, out_channels, 1,
                                                   stride=1, padding=0, bias=False))
                if bn:
                    backbone.add_module('Bn{}_{}'.format(j, i),
                                             nn.BatchNorm2d(out_channels))
                backbone.add_module('Relu{}_{}'.format(j, i), nn.ReLU())
                in_channels = out_channels
            self.backbones.append(backbone)

    def forward(self, xyz, points):
        new_xyz = gather_points(xyz, fps(xyz, self.M))
        new_points_all = []
        for i in range(len(self.mlps)):
            radius = self.radiuses[i]
            K = self.Ks[i]
            grouped_inds = ball_query(xyz, new_xyz, radius, K)
            #grouped_inds = knn_cluster(xyz, new_xyz, K)##LQ add it to compare the results of ball_query and knn_query
            grouped_xyz = gather_points(xyz, grouped_inds)
            grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
            if points is not None:
                grouped_points = gather_points(points, grouped_inds)
                if self.use_xyz:
                    new_points = torch.cat(
                        (grouped_xyz.float(), grouped_points.float()),
                        dim=-1)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz
            new_points = self.backbones[i](new_points.permute(0, 3, 2, 1).contiguous())
            if self.pooling == 'avg':
                new_points = torch.mean(new_points, dim=2)
            else:
                new_points = torch.max(new_points, dim=2)[0]
            new_points = new_points.permute(0, 2, 1).contiguous()
            new_points_all.append(new_points)
        return new_xyz, torch.cat(new_points_all, dim=-1)

