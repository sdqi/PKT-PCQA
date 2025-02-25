import torch
from utils.common import get_dists
import numpy as np  #LQ add it to compare ball_query and knn
from sklearn.neighbors import NearestNeighbors #LQ add it to compare ball_query and knn

def ball_query(xyz, new_xyz, radius, K):
    '''
    :param xyz: shape=(B, N, 3)，所有点
    :param new_xyz: shape=(B, M, 3),中心点
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K),返回的是前K个点到聚类中心的索引
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]#中心点个数
    # grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    grouped_inds[dists > radius] = N # 用mask来获得索引 
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K] # 此处是随机的K个点的索引，保留半径以内的点
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K) # 获得最近点的距离，并且复制K个，shape = (B,M,K)
    #处理在半径r内点的个数不足K个的情况
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N] #grouped_inds的shape = (B,M,K)
    # print(grouped_inds.shape)
    return grouped_inds


###LYY编写下述代码实现KNN聚类
def knn_cluster(xyz, new_xyz, K):
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]#中心点个数

    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    grouped_inds = torch.argsort(dists)[:,:,:K]
    # print(new_xyz.shape, xyz.shape, grouped_inds.shape)
    # knn_model=NearestNeighbors(n_neighbors=k,algorithm='kd_tree')
    # knn_model.fit(xyz, new_xyz)
    # predicted_labels = knn_model.predict(xyz)

    # grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    return grouped_inds


###LQ编写下述代码实现KNN聚类
def knn_cluster_lq(xyz, new_xyz, k):
    #xyz相当于一个原来的point cloud,
    #new_xyz相当于要聚类的中心点

    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]#中心点个数
    #创建一个NearestNeighbors对象
    xyz_np=xyz.cpu().numpy()
    xyz_np=xyz_np.reshape(-1,3)
    new_xyz_np=new_xyz.cpu().numpy()
    knn_model=NearestNeighbors(n_neighbors=k,algorithm='kd_tree')#we can choose the auto, kd_tree,ball_tree,brute
    #对点云数据进行训练，建立knn模型
    #knn_model.fit(xyz.cpu(), new_xyz.cpu())
    grouped_inds=[]
    for center in new_xyz_np:
        knn_model.fit(xyz_np)
        #查找中心点center的最近邻
        center_np=np.array(center)
        distances, indices = knn_model.kneighbors(center_np.reshape(1,-1))

        if indices[0].dtype!=np.int:
            indices=indices.astype(np.int)
        nearest_neighbors = xyz[torch.tensor(indices[0]).to(device)]

       #将最近邻加入到聚类结果中
        grouped_inds.append(nearest_neighbors)
        #grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
        grouped_inds=torch.stack(grouped_inds,dim=0).to(device)
        
        # #使用DBSCAN算法对最近邻进行聚类
        # dbscan_model = DBSCAN(eps=1.0, min_samples=k)
        # dbscan_model.fit(nearest_neighbors)

        # #将聚类结果加入到结果例表中
        # grouped_inds.append(dbscan_model.labels_)

    return grouped_inds





