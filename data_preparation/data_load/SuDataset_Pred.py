from torch.utils.data import Dataset
import numpy as np
import os
from plyfile import PlyData
# 与数据增强相关的库
from data_preparation.provider import pc_normalize, rotate_point_cloud_with_normal, rotate_perturbation_point_cloud_with_normal, \
    random_scale_point_cloud, shift_point_cloud, jitter_point_cloud, shuffle_points, random_point_dropout

class SuDataset_Pred(Dataset):

    def __init__(self, data_root, data_aux_root, split, npoints, augment=False, dp=False, normalize=True):
        assert(split == 'train' or split == 'test')
        self.npoints = npoints
        self.augment = augment
        self.dp = dp
        self.normalize = normalize
        self.data_root = data_root

        # train_list_path = os.path.join(data_aux_root, 'pred_train_pen_stone_pumpkin_glasses.txt')
        train_list_path = os.path.join(data_aux_root, 'pred_train_1.txt')
        train_files_list = self.read_list_file(train_list_path)
        # test_list_path = os.path.join(data_aux_root, 'pred_test_pen_stone_pumpkin_glasses.txt')
        test_list_path = os.path.join(data_aux_root, 'pred_test_1.txt')        
        test_files_list = self.read_list_file(test_list_path)
        self.files_list = train_files_list if split == 'train' else test_files_list
        self.caches = {}

    # 读取train.txt和test.txt文件的内容，获取数据集的所有文件名
    def read_list_file(self, file_path):
        base = self.data_root #存储数据的根目录
        files_list = []
        with open(file_path, 'r') as f: # 
            for line in f.readlines():
                label = line.strip().split(',')[1] # score 1,score 2,..,score n
                file_name = line.strip().split(',')[0]
                cur = os.path.join(base, '{}.ply'.format(file_name))
                files_list.append([cur, float(label)]) # [文件名，类型编号]
        return files_list  

    def augment_pc(self, pc_normal):
        rotated_pc_normal = rotate_point_cloud_with_normal(pc_normal)
        rotated_pc_normal = rotate_perturbation_point_cloud_with_normal(rotated_pc_normal)
        jittered_pc = random_scale_point_cloud(rotated_pc_normal[:, :3])
        jittered_pc = shift_point_cloud(jittered_pc)
        jittered_pc = jitter_point_cloud(jittered_pc)
        rotated_pc_normal[:, :3] = jittered_pc
        return rotated_pc_normal

    def __getitem__(self, index):
        if index in self.caches:
            return self.caches[index]
        file, label = self.files_list[index]
        # print('file: ',file,'label: ',label)
        #**************************读取ply文件***************************
        plydata = PlyData.read(file)
        
        xlist = plydata['vertex']['x'].reshape(-1,1)#将1维数组转换为n维数组
        ylist = plydata['vertex']['y'].reshape(-1,1)
        zlist = plydata['vertex']['z'].reshape(-1,1)
        red = plydata['vertex']['red'].reshape(-1,1)
        green = plydata['vertex']['green'].reshape(-1,1)
        blue = plydata['vertex']['blue'].reshape(-1,1)

        # 组合为n*6维数据
        xyz_points = np.concatenate((xlist, ylist, zlist, red, green, blue), axis = 1)
        #******************************************************************
        
        if self.npoints > 0:# 是否使用数据的全部点
            inds = np.random.randint(0, len(xyz_points), size=(self.npoints, )) # ？？
        else:
            inds = np.arange(len(xyz_points))
            np.random.shuffle(inds) #打乱顺序

        # xyz_points = xyz_points[inds, :]
        # xyz_points = xyz_points[:, :]
        if self.normalize: # 默认会调用
            xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])
        if self.augment: 
            xyz_points = self.augment_pc(xyz_points)
        if self.dp:
            xyz_points = random_point_dropout(xyz_points)#随机丢点，丢掉的点的原位置默认为第一个点
        self.caches[index] = [xyz_points, label]
        return xyz_points, label

    def __len__(self):
        return len(self.files_list)