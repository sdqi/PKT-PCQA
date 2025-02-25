import torch
import torch.nn as nn
from utils.set_abstraction import init_sample_and_group, sample_and_group_all, sample_and_group
# import ipdb
#*****************************************子模块*************************************#
'''
获取通道的权重：cSE
1、将feature map通过global average pooling方法从[C, H, W]变为[C, 1, 1]
2、然后使用两个1×1×1卷积进行信息的处理，最终得到C维的向量
3、然后使用sigmoid函数进行归一化，得到对应的mask
4、最后通过channel-wise相乘，得到经过信息校准过的feature map
'''
# 代码1，缺少ReLU                                                                                                                                                                                                                                      
# class cSE(nn.Module):
#     def __init__(self, in_channels):
#         super(cSE, self).__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         # //为取整除 - 向下取接近商的整数，eg:9//2=4
#         self.Conv_Squeeze = nn.Conv2d(in_channels,
#                                       in_channels // 2, 
#                                       kernel_size=1,
#                                       bias=False)
#         self.Conv_Excitation = nn.Conv2d(in_channels // 2,
#                                          in_channels,
#                                          kernel_size=1,
#                                          bias=False)
#         self.norm = nn.Sigmoid()

#     def forward(self, U):
#         z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
#         z = self.Conv_Squeeze(z)  # shape: [bs, c/2, 1, 1]
#         z = self.Conv_Excitation(z)  # shape: [bs, c, 1, 1]
#         z = self.norm(z)
#         return U * z.expand_as(U)
class cSE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(cSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
'''
获取空间的权重：sSE
1、直接对feature map使用1×1×1卷积, 从[C, H, W]变为[1, H, W]的features
2、然后使用sigmoid进行激活得到spatial attention map
3、然后直接施加到原始feature map中，完成空间的信息校准
'''
class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U) # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q # 广播机制

'''
通道和空间的权重都获取：scSE
cSE+sSE
'''
class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

#**************************************初始特征提取模块*********************************#
'''
融合了自注意力机制模块的初始特征提取层
attention_model = {'cSE', 'sSE','scSE'}
'''
class PointNet_Pre_SA_AT_Module(nn.Module):
    def __init__(self, in_channels, mlp, attention_model, bn=True, pooling='max', use_xyz=True, use_key_points=False):
        super(PointNet_Pre_SA_AT_Module, self).__init__()
        self.in_channels = in_channels
        self.mlp = mlp #每个全连接层模块的输出个数 
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.use_key_points = use_key_points
        self.backbone = nn.Sequential()
        self.attention_model = attention_model
        # self.c_se = cSE(in_channels)
        for i, out_channels in enumerate(mlp):
            # class torch.nn.Conv2d(in_channels,out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))# w*x+b
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            #####################################
            # self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            #####################################
            # if i == 2: #说明是最后一层
            #     self.backbone.add_module('sSE{}'.format(i), sSE(out_channels))

            #     if self.attention_model == 'cSE':
            #         self.backbone.add_module('cSE{}'.format(i), cSE(out_channels))
            #     elif self.attention_model == 'sSE':
            #         self.backbone.add_module('sSE{}'.format(i), sSE(out_channels))
            #     elif self.attention_model == 'scSE':
            #         self.backbone.add_module('scSE{}'.format(i), scSE(out_channels))

            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels

    def forward(self, xyz, points):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
#**************************************其他特征提取模块*********************************#
'''
融合了自注意力机制模块的其他特征提取层
attention_model = {'cSE', 'sSE'}
'''        
class PointNet_SA_AT_Module(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, attention_model, group_all, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_AT_Module, self).__init__()
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
        self.attention_model = attention_model
        for i, out_channels in enumerate(mlp):
            # class torch.nn.Conv2d(in_channels,out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))# w*x+b
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            #####################################
            # self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            #####################################
            # if (i == 1): #说明是倒数一层
            #     if self.attention_model == 'cSE':
            #         self.backbone.add_module('cSE{}'.format(i), cSE(out_channels))
            #     elif self.attention_model == 'sSE':
            #         self.backbone.add_module('sSE{}'.format(i), sSE(out_channels))
            #     elif self.attention_model == 'scSE':
            #         self.backbone.add_module('scSE{}'.format(i), scSE(out_channels))

            if (i == 2): #说明是最后一层
                if self.attention_model == 'cSE':
                    self.backbone.add_module('cSE{}'.format(i), cSE(out_channels))
                elif self.attention_model == 'sSE':
                    self.backbone.add_module('sSE{}'.format(i), sSE(out_channels))
                elif self.attention_model == 'scSE':
                    self.backbone.add_module('scSE{}'.format(i), scSE(out_channels))

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

######################################################################################################################
class PointNet_SA_AT_Module_tmp(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, attention_model, group_all, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_AT_Module_tmp, self).__init__()
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
        self.attention_model = attention_model
        for i, out_channels in enumerate(mlp):
            # class torch.nn.Conv2d(in_channels,out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))# w*x+b
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            #####################################
            # self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            #####################################
            # if (i == 1): #说明是倒数一层
            #     if self.attention_model == 'cSE':
            #         self.backbone.add_module('cSE{}'.format(i), cSE(out_channels))
            #     elif self.attention_model == 'sSE':
            #         self.backbone.add_module('sSE{}'.format(i), sSE(out_channels))
            #     elif self.attention_model == 'scSE':
            #         self.backbone.add_module('scSE{}'.format(i), scSE(out_channels))

            if (i == 2): #说明是最后一层
                if self.attention_model == 'cSE':
                    self.backbone.add_module('cSE{}'.format(i), cSE(out_channels))
                elif self.attention_model == 'sSE':
                    self.backbone.add_module('sSE{}'.format(i), sSE(out_channels))
                elif self.attention_model == 'scSE':
                    self.backbone.add_module('scSE{}'.format(i), scSE(out_channels))

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