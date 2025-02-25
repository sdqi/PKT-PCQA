import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取Excel文件
file_path = r'F:\deep_learning\PKT_PCQA\kaiyuan\data\excel\test_content.xlsx'
txt_path = r'F:\deep_learning\PKT_PCQA\kaiyuan\data\txt\test'
df = pd.read_excel(file_path)
mos_max = 100
mos_min = 0

if not os.path.exists(txt_path):
    os.makedirs(txt_path)

mos_data = df['MOS'].values.reshape(-1, 1)

# 对'MOS'数据进行最大最小值归一化
mos_normalized = (mos_data - mos_min) / (mos_max - mos_min)

df['mos_normalized'] = mos_normalized

labels = []

for score in mos_normalized:
    if score <= 0.25:
        labels.append('bad')
    elif score <= 0.75:
        labels.append('fair')
    else:
        labels.append('good')

df['label'] = labels

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# 生成训练集和测试集的文本文件（归一化后的MOS和标签）
# 保存归一化后的MOS数据
with open(os.path.join(txt_path, 'pred_train_1.txt'), 'w') as train_mos_file:
    for index, row in train_data.iterrows():
        content = row['content']
        mos_value = row['mos_normalized']
        train_mos_file.write(f"{content},{mos_value}\n")

with open(os.path.join(txt_path, 'pred_test_1.txt'), 'w') as test_mos_file:
    for index, row in test_data.iterrows():
        content = row['content']
        mos_value = row['mos_normalized']
        test_mos_file.write(f"{content},{mos_value}\n")

# 保存标签数据
with open(os.path.join(txt_path, 'train_1.txt'), 'w') as train_label_file:
    for index, row in train_data.iterrows():
        content = row['content']
        label = row['label']
        train_label_file.write(f"{content},{label}\n")

with open(os.path.join(txt_path, 'test_1.txt'), 'w') as test_label_file:
    for index, row in test_data.iterrows():
        content = row['content']
        label = row['label']
        test_label_file.write(f"{content},{label}\n")

# 保存类别名称文件
with open(os.path.join(txt_path, 'cls_name.txt'), 'w') as cls_name_file:
    cls_name_file.write("bad\nfair\ngood\n")

# 保存修改后的DataFrame到原始Excel文件
df.to_excel(file_path, index=False)

print("归一化和标签处理完成，数据已保存到原始Excel文件中，并生成训练集与测试集的文本文件。")
