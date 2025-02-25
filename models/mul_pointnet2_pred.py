import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.set_abstraction import PointNet_SA_Module, PointNet_SA_Module_MSG, PointNet_Pre_SA_Module
# 损失函数
# from scipy.stats import pearsonr
from audtorch.metrics.functional import pearsonr
import math

#不融合关键点特征
class mul_pointnet2_pred(nn.Module):
    def __init__(self, in_channels):#初始in_channels为6
        super(mul_pointnet2_pred, self).__init__()
        # 提取关键点周围区域的特征
        self.pt_sa1 = PointNet_Pre_SA_Module(in_channels=in_channels, mlp=[64, 64, 128])
        self.pt_sa2 = PointNet_SA_Module(M=128, radius=0.33, K=64, in_channels=131, mlp=[128, 128, 256], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.pred = nn.Linear(256, 1)

    def forward(self, xyz, points):
        batchsize = xyz.shape[0]
        new_xyz, new_points = self.pt_sa1(xyz, points)
        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
        net = new_points.view(batchsize, -1)
        net = self.dropout1(F.relu(self.bn1(self.fc1(net))))
        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))
        net = self.pred(net)
        return net
        
#融合关键点特征
# class mul_pointnet2_plus_key(nn.Module):
#     def __init__(self, in_channels):#初始in_channels为6
#         super(mul_pointnet2_plus_key, self).__init__()
#         # 提取关键点的特征
#         # 提取关键点周围区域的特征
        

#     def forward(self, xyz, points):




class pearson_loss(nn.Module):
    def __init__(self):
        super(pearson_loss, self).__init__()
        
    def forward(self, pred, label):
        pcorrelation = pearsonr(pred, label)
        loss = (1 - pcorrelation)**2
        return loss 