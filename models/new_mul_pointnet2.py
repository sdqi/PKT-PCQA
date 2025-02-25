import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.set_abstraction import PointNet_SA_Module, PointNet_SA_Module_MSG, PointNet_Pre_SA_Module
# 损失函数
# from scipy.stats import pearsonr
from audtorch.metrics.functional import pearsonr
import math
import ipdb

#不融合关键点特征
class mul_pointnet2(nn.Module):
    def __init__(self, in_channels, nclasses):#初始in_channels为6
        super(mul_pointnet2, self).__init__()
        # 提取关键点周围区域的特征
        self.pt_sa1 = PointNet_Pre_SA_Module(in_channels=in_channels, mlp=[64, 128, 256])
        # self.pt_sa2 = PointNet_SA_Module(M=128, radius=0.4, K=64, in_channels=131, mlp=[128, 256], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.cls = nn.Linear(256, nclasses)

    def forward(self, xyz, points):
        batchsize = xyz.shape[0]
        new_xyz, new_points = self.pt_sa1(xyz, points)
        # new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
        net = new_points.view(batchsize, -1)
        # ipdb.set_trace()
        net = self.dropout1(F.relu(self.bn1(self.fc1(net))))
        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))
        net = self.cls(net)
        return net
        
#融合关键点特征
# class mul_pointnet2_plus_key(nn.Module):
#     def __init__(self, in_channels):#初始in_channels为6
#         super(mul_pointnet2_plus_key, self).__init__()
#         # 提取关键点的特征
#         # 提取关键点周围区域的特征
        

#     def forward(self, xyz, points):



class cls_loss(nn.Module):
    def __init__(self):
        super(cls_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss() # softmax和损失函数的融合
    def forward(self, pred, label):
        '''
        :param pred: shape=(B, nclass)
        :param lable: shape=(B, )
        :return: loss
        '''
        loss = self.loss(pred, label)
        return loss