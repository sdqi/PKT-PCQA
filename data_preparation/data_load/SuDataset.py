# 针对SU Dataset的数据装载
import os
import numpy as np
from torch.utils.data import Dataset
# 与数据增强相关的库
from data_preparation.provider import pc_normalize, rotate_point_cloud_with_normal, rotate_perturbation_point_cloud_with_normal, \
    random_scale_point_cloud, shift_point_cloud, jitter_point_cloud, shuffle_points, random_point_dropout
from plyfile import PlyData
# import ipdb 
class SuDataset(Dataset):

    def __init__(self, data_root, data_aux_root, split, npoints, augment=False, dp=False, normalize=True):
        '''
        data_root: 存储ply文件的地方
        data_aux_root: 存储txt文件的地方(train,test的文件名)
         '''
        assert(split == 'train' or split == 'test')
        self.npoints = npoints
        self.augment = augment
        self.dp = dp
        self.normalize = normalize
        self.data_root = data_root
# F:\LQ\PointNet2\pointnet2_self_attention\datasets\sj_dataset\cls_txt\sj_name2mos.txt
        cls2name, name2cls = self.decode_classes(os.path.join(data_aux_root, 'cls_name.txt'))
        # train_list_path = os.path.join(data_aux_root, 'cls_train_flowerpot_stone_pumpkin_glasses.txt')
        train_list_path = os.path.join(data_aux_root, 'train_1.txt')
        train_files_list = self.read_list_file(train_list_path, name2cls)
        # test_list_path = os.path.join(data_aux_root, 'cls_test_flowerpot_stone_pumpkin_glasses.txt')
        test_list_path = os.path.join(data_aux_root, 'test_1.txt')
        test_files_list = self.read_list_file(test_list_path, name2cls)
        self.files_list = train_files_list if split == 'train' else test_files_list
        self.caches = {}

        # cls2name, name2cls = self.decode_classes(os.path.join(data_aux_root, 'su_shape_names.txt'))
        # train_list_path = os.path.join(data_aux_root, 'su_train_7.txt')
        # train_files_list = self.read_list_file(train_list_path, name2cls)
        # test_list_path = os.path.join(data_aux_root, 'su_test_7.txt')
        # test_files_list = self.read_list_file(test_list_path, name2cls)
        # self.files_list = train_files_list if split == 'train' else test_files_list
        # self.caches = {}

    # 数字与文件名的对应
    def decode_classes(self, file_path):
        cls2name, name2cls = {}, {}
        with open(file_path) as f:
            for i, name in enumerate(f.readlines()):
                cls2name[i] = name.strip()
                name2cls[name.strip()] = i
        return cls2name, name2cls

    # 读取train.txt和test.txt文件的内容，获取数据集的所有文件名
    def read_list_file(self, file_path, name2cls):
        base = self.data_root #存储数据的根目录
        files_list = []
        with open(file_path) as f: # 
            for line in f.readlines():
                label = line.strip().split(',')[1] # good, fair, bad
                file_name = line.strip().split(',')[0]
                cur = os.path.join(base, '{}.ply'.format(file_name))
                files_list.append([cur, name2cls[label]]) # [文件名，类型编号]
        return files_list

    # 数据增强处理，默认不使用；__getitem__里调用
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
        # xyz_points = np.load(file) #读取数据的所有点
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
        
        # if self.npoints > 0:# 是否使用数据的全部点
        #     inds = np.random.randint(0, len(xyz_points), size=(self.npoints, )) # ？？
        # else:
        #     inds = np.arange(len(xyz_points))
        #     np.random.shuffle(inds) #打乱顺序

        # xyz_points = xyz_points[inds, :]
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

# class SuDataset(Dataset):

#     def __init__(self, data_root, data_aux_root, split, npoints, augment=False, dp=False, normalize=True):
#         assert(split == 'train' or split == 'test')
#         self.npoints = npoints
#         self.augment = augment
#         self.dp = dp
#         self.normalize = normalize
#         self.data_root = data_root

#         cls2name, name2cls = self.decode_classes(os.path.join(data_aux_root, 'su_dataset_shape_names.txt'))
#         train_list_path = os.path.join(data_aux_root, 'su_dataset_train.txt')
#         train_files_list = self.read_list_file(train_list_path, name2cls)
#         test_list_path = os.path.join(data_aux_root, 'su_dataset_test.txt')
#         test_files_list = self.read_list_file(test_list_path, name2cls)
#         self.files_list = train_files_list if split == 'train' else test_files_list
#         self.caches = {}

#     # 数字与文件名的对应
#     def decode_classes(self, file_path):
#         cls2name, name2cls = {}, {}
#         with open(file_path, 'r') as f:
#             for i, name in enumerate(f.readlines()):
#                 cls2name[i] = name.strip()
#                 name2cls[name.strip()] = i
#         return cls2name, name2cls

#     # 读取train.txt和test.txt文件的内容，获取数据集的所有文件名
#     def read_list_file(self, file_path, name2cls):
#         base = self.data_root #存储数据的根目录
#         files_list = []
#         with open(file_path, 'r') as f: # 
#             for line in f.readlines():
#                 label = line.strip().split(',')[1] # good, fair, bad
#                 file_name = line.strip().split(',')[0]
#                 cur = os.path.join(base, '{}.npy'.format(file_name))
#                 files_list.append([cur, name2cls[label]]) # [文件名，类型编号]
#         return files_list

#     # 数据增强处理，默认不使用；__getitem__里调用
#     def augment_pc(self, pc_normal):
#         rotated_pc_normal = rotate_point_cloud_with_normal(pc_normal)
#         rotated_pc_normal = rotate_perturbation_point_cloud_with_normal(rotated_pc_normal)
#         jittered_pc = random_scale_point_cloud(rotated_pc_normal[:, :3])
#         jittered_pc = shift_point_cloud(jittered_pc)
#         jittered_pc = jitter_point_cloud(jittered_pc)
#         rotated_pc_normal[:, :3] = jittered_pc
#         return rotated_pc_normal

#     def __getitem__(self, index):
#         if index in self.caches:
#             return self.caches[index]
#         file, label = self.files_list[index]
#         # ipdb.set_trace()
#         xyz_points = np.load(file) #读取数据的所有点
#         # ipdb.set_trace()
#         if self.npoints > 0:# 是否使用数据的全部点
#             inds = np.random.randint(0, len(xyz_points), size=(self.npoints, )) # ？？
#         else:
#             inds = np.arange(len(xyz_points))
#             np.random.shuffle(inds) #打乱顺序

#         xyz_points = xyz_points[inds, :].astype(float)
#         if self.normalize: # 默认会调用
#             xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])
#         if self.augment: 
#             xyz_points = self.augment_pc(xyz_points)
#         if self.dp:
#             xyz_points = random_point_dropout(xyz_points)#随机丢点，丢掉的点的原位置默认为第一个点
#         self.caches[index] = [xyz_points, label]
#         return xyz_points, label

#     def __len__(self):
#         return len(self.files_list)