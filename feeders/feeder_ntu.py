import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_config=None,
                 window_size=-1, normalization=False, debug=False, use_mmap=False, bone=False, vel=False):
        """
        :param data_path: 数据文件路径
        :param label_path: 标签文件路径
        :param split: 数据集划分（'train' 或 'test'）
        :param random_config: 数据增强配置字典，包括 'choose'、'shift'、'move' 和 'rot' 等布尔值
        :param window_size: 输出序列的长度
        :param normalization: 是否对输入序列进行归一化
        :param debug: 是否仅使用前 100 个样本进行调试
        :param bone: 是否使用骨骼模式
        :param vel: 是否使用运动模式
        """
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.p_interval = p_interval
        self.window_size = window_size
        self.normalization = normalization
        self.debug = debug
        self.use_mmap = use_mmap
        self.bone = bone
        self.vel = vel
        self.random_config = random_config or {}

        self.data, self.label = self.load_data()
        if normalization:
            self.mean_map, self.std_map = self.get_mean_std_map()

    def load_data(self):
        """根据数据集划分加载数据并转换格式"""
        npz_data = np.load(self.data_path)
        data, label = (npz_data['x_train'], np.where(npz_data['y_train'] > 0)[1]) if self.split == 'train' \
            else (npz_data['x_test'], np.where(npz_data['y_test'] > 0)[1])
        sample_names = [f"{self.split}_{i}" for i in range(len(data))]
        data = data.reshape((len(data), -1, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        return data, label

    def get_mean_std_map(self):
        """计算数据的均值和标准差映射，用于归一化"""
        N, C, T, V, M = self.data.shape
        mean_map = self.data.mean(axis=(2, 4), keepdims=True).mean(axis=0)
        std_map = self.data.transpose(0, 2, 4, 1, 3).reshape(N * T * M, C * V).std(axis=0).reshape(C, 1, V, 1)
        return mean_map, std_map

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data, label = np.array(self.data[index]), self.label[index]
        valid_frames = np.sum(data.sum(0).sum(-1).sum(-1) != 0)
        data = tools.valid_crop_resize(data, valid_frames, self.p_interval, self.window_size)
        
        # 数据增强
        if self.random_config.get('rot'):
            data = tools.random_rot(data)
        if self.bone:
            data = self.apply_bone_mode(data)
        if self.vel:
            data = self.apply_velocity_mode(data)
        
        return data, label, index

    def apply_bone_mode(self, data):
        """骨骼模式下的关节差异转换"""
        from .bone_pairs import ntu_pairs
        bone_data = np.zeros_like(data)
        for v1, v2 in ntu_pairs:
            bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
        return bone_data

    def apply_velocity_mode(self, data):
        """计算数据的运动模式，即帧之间的差分"""
        data[:, :-1] = data[:, 1:] - data[:, :-1]
        data[:, -1] = 0
        return data

    def top_k(self, score, top_k):
        """计算 top-k 准确率"""
        rank = score.argsort()
        return sum(l in rank[i, -top_k:] for i, l in enumerate(self.label)) / len(self.label)


def import_class(name):
    """动态导入模块"""
    mod = __import__(name.split('.')[0])
    for comp in name.split('.')[1:]:
        mod = getattr(mod, comp)
    return mod

