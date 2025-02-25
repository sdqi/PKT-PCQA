import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors
import pandas as pd
# from data_preparation.provider import pc_normalize
from tqdm import tqdm

# 文件路径
key_root_path = r"F:\deep_learning\PKT_PCQA\kaiyuan\data\key_points\test_key1024"  # 存储关键点的根路径
points_root_path = r"F:\deep_learning\PKT_PCQA\kaiyuan\data\datasets\test"  # 存储所有点文件的根路径
out_root_path = r"F:\deep_learning\PKT_PCQA\kaiyuan\data\key_points_local16\test_key1024_local16"  # 输出文件的根路径
excel_path = r'F:\deep_learning\PKT_PCQA\kaiyuan\data\excel\test_content.xlsx'  # Excel 文件路径

# 读取 Excel 文件中的 'content' 列
df = pd.read_excel(excel_path)
point_cloud_names = df['content'].tolist() 

if not os.path.exists(out_root_path):
    os.makedirs(out_root_path)
    
def pc_normalize(pc):
    mean = np.mean(pc, axis=0)
    pc -= mean
    m = np.max(np.sqrt(np.sum(np.power(pc, 2), axis=1)))
    pc /= m
    return pc

# 定义获取有效索引的函数
def get_valid_indices(grouped_inds, dists, radius, K, invalid_value):
    grouped_inds[dists > radius] = invalid_value
    valid_inds = grouped_inds[grouped_inds != invalid_value]
    if len(valid_inds) > 0:
        if len(valid_inds) < K:
            repeat_count = K // len(valid_inds) + 1
            return valid_inds.repeat(repeat_count)[:K]
        else:
            return torch.sort(valid_inds, dim=-1)[0][:K]
    return None

# 遍历每个点云文件名进行处理
for point_cloud_name in tqdm(point_cloud_names):
    # 构造文件路径
    key_points_path = os.path.join(key_root_path, f"{point_cloud_name}.ply")
    points_path = os.path.join(points_root_path, f"{point_cloud_name}.ply")

    # 检查文件是否存在
    if not os.path.exists(key_points_path) or not os.path.exists(points_path):
        print(f"Warning: Missing file for {point_cloud_name}, skipping.")
        continue

    # 读取 .ply 文件
    key_points = PlyData.read(key_points_path)
    org_points = PlyData.read(points_path)

    # 提取数据
    key_xlist = key_points['vertex']['x'].reshape(-1, 1)
    key_ylist = key_points['vertex']['y'].reshape(-1, 1)
    key_zlist = key_points['vertex']['z'].reshape(-1, 1)
    key_red = key_points['vertex']['red'].reshape(-1, 1)
    key_green = key_points['vertex']['green'].reshape(-1, 1)
    key_blue = key_points['vertex']['blue'].reshape(-1, 1)
    key_xyzrgb_points = np.concatenate((key_xlist, key_ylist, key_zlist, key_red, key_green, key_blue), axis=1)

    org_xlist = org_points['vertex']['x'].reshape(-1, 1)
    org_ylist = org_points['vertex']['y'].reshape(-1, 1)
    org_zlist = org_points['vertex']['z'].reshape(-1, 1)
    org_red = org_points['vertex']['red'].reshape(-1, 1)
    org_green = org_points['vertex']['green'].reshape(-1, 1)
    org_blue = org_points['vertex']['blue'].reshape(-1, 1)
    org_xyzrgb_points = np.concatenate((org_xlist, org_ylist, org_zlist, org_red, org_green, org_blue), axis=1)

    # 归一化处理
    key_xyzrgb_points[:, :3] = pc_normalize(key_xyzrgb_points[:, :3])
    org_xyzrgb_points[:, :3] = pc_normalize(org_xyzrgb_points[:, :3])

    # 使用 KNN 算法查找邻居
    N, C = org_xyzrgb_points.shape
    M, D = key_xyzrgb_points.shape
    K = 16
    knn = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(org_xyzrgb_points[:, :3])
    distances, indices = knn.kneighbors(key_xyzrgb_points[:, :3])

    grouped_points = org_xyzrgb_points[indices.reshape(-1), :]
    all_points = np.concatenate((key_xyzrgb_points, grouped_points), axis=0)

    # 存储为 ply 格式
    vertices = np.array(
        [tuple(row) for row in all_points],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    ply_out = PlyData([PlyElement.describe(vertices, 'vertex')], text=True)
    ply_out.write(os.path.join(out_root_path, f"{point_cloud_name}.ply"))
