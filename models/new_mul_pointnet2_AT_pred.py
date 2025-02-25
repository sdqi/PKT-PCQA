import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.set_abstraction import PointNet_SA_Module, PointNet_SA_Module_MSG, PointNet_Pre_SA_Module
from utils.self_attention import PointNet_Pre_SA_AT_Module, PointNet_SA_AT_Module
# 损失函数
# from scipy.stats import pearsonr
from audtorch.metrics.functional import pearsonr
import math

#不融合关键点特征
class mul_pointnet2_AT_pred(nn.Module):
    def __init__(self, in_channels, attention_type):#初始in_channels为6
        super(mul_pointnet2_AT_pred, self).__init__()
        # 提取关键点周围区域的特征
        self.pt_sa1 = PointNet_Pre_SA_AT_Module(in_channels=in_channels, mlp=[64, 128, 256], attention_model = attention_type)
        # self.pt_sa2 = PointNet_SA_AT_Module(M=128, radius=0.4, K=64, in_channels=131, mlp=[128, 128, 256], group_all=False, attention_model = attention_type)
        self.pt_sa3 = PointNet_SA_AT_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True, attention_model = attention_type)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.pred = nn.Linear(256, 1)

    def forward(self, xyz, points):
        batchsize = xyz.shape[0]
        # print(points)
        new_xyz, new_points = self.pt_sa1(xyz, points)
        # print('1: ',new_points)
        # new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        # print('2: ',new_points)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
        # print('3: ',new_points)
        net = new_points.view(batchsize, -1)
        #F.relu
        net = self.dropout1(F.relu(self.bn1(self.fc1(net))))
        # print('4: ',net)
        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))
        # print('5: ',net)
        net = self.pred(net)
        # print('6: ',net)
        return net



class pearson_loss(nn.Module):
    def __init__(self):
        super(pearson_loss, self).__init__()
        
    def forward(self, pred, label):
        pcorrelation = pearsonr(pred, label)
        loss = (1 - pcorrelation)**2
        return loss 


