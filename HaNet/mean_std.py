import numpy as np
import h5py

# 打开 HDF5 文件
with h5py.File('./dataset/train/hsi', 'r') as file:
    # 假设数据存储在名为 'dataset_name' 的数据集中
    # 你需要替换 'dataset_name' 为你的实际数据集名称
    # 可以使用 print(file.keys()) 来查看文件中的所有数据集名称
    data = file['cube'][:]

# 计算每个通道的均值，假设通道是最后一个维度
means = np.mean(data, axis=(1, 2))

# 计算每个通道的标准差
stds = np.std(data, axis=(1, 2))

# 打印结果
print("Channel means:", means)
print("Channel standard deviations:", stds)
