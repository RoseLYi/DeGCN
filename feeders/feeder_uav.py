import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class SkeletonDataset(Dataset):
    def __init__(self, data_path, label_path=None, interval=1, mode='train', config=None,
                 seq_length=-1, normalize=False, debug=False, mmap_mode=False):
        """
        Skeleton Dataset 初始化

        :param data_path: 数据文件路径
        :param label_path: 标签文件路径
        :param mode: 数据模式（'train' 或 'test'）
        :param config: 包含数据增强配置的字典（包括 'random_select'、'random_shift' 等布尔值）
        :param seq_length: 序列的目标长度
        :param normalize: 是否进行归一化
        :param debug: 是否启用调试模式，仅加载前 100 个样本
        :param mmap_mode: 使用内存映射加载数据
        """
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.interval = interval
        self.seq_length = seq_length
        self.normalize = normalize
        self.debug = debug
        self.mmap_mode = mmap_mode

        self.config = config or {}
        self.random_select = self.config.get('random_select', False)
        self.random_shift = self.config.get('random_shift', False)
        self.random_move = self.config.get('random_move', False)
        self.apply_rotation = self.config.get('apply_rotation', False)
        self.use_bone = self.config.get('use_bone', False)
        self.apply_velocity = self.config.get('apply_velocity', False)

        self.data, self.labels = self._load_data()
        if self.normalize:
            self.mean_map, self.std_map = self._compute_mean_std()

    def _load_data(self):
        data_content = np.load(self.data_path)
        if self.mode == 'train':
            data = data_content['x_train']
            labels = data_content['y_train']
            sample_names = [f'train_{i}' for i in range(len(data))]
        elif self.mode == 'test':
            data = data_content['x_test']
            labels = data_content['y_test']
            sample_names = [f'test_{i}' for i in range(len(data))]
        else:
            raise ValueError('模式应为 "train" 或 "test"')

        data = data.transpose(0, 4, 1, 3, 2)
        return data, labels

    def _compute_mean_std(self):
        N, C, T, V, M = self.data.shape
        mean_map = self.data.mean(axis=(2, 4), keepdims=True).mean(axis=0)
        std_map = self.data.transpose((0, 2, 4, 1, 3)).reshape(-1, C * V).std(axis=0).reshape(C, 1, V, 1)
        return mean_map, std_map

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_data = np.array(self.data[idx])
        label = self.labels[idx]

        valid_frames = np.sum(sample_data.sum(axis=(0, -1, -1)) != 0)
        sample_data = tools.valid_crop_resize(sample_data, valid_frames, self.interval, self.seq_length)

        sample_data = self._apply_augmentations(sample_data)

        return sample_data, label, idx

    def _apply_augmentations(self, data):
        if self.apply_rotation:
            data = tools.random_rot(data)
        if self.use_bone:
            data = self._calculate_bone_mode(data)
        if self.apply_velocity:
            data = self._calculate_velocity(data)
        return data

    def _calculate_bone_mode(self, data):
        from .bone_pairs import ntu_pairs
        bone_data = np.zeros_like(data)
        for start, end in ntu_pairs:
            bone_data[:, :, start - 1] = data[:, :, start - 1] - data[:, :, end - 1]
        return bone_data

    def _calculate_velocity(self, data):
        data[:, :-1] = data[:, 1:] - data[:, :-1]
        data[:, -1] = 0
        return data

    def top_k_accuracy(self, scores, top_k):
        rankings = scores.argsort()
        hit_top_k = [label in rankings[i, -top_k:] for i, label in enumerate(self.labels)]
        return sum(hit_top_k) / len(hit_top_k)


def load_class(name):
    parts = name.split('.')
    module = __import__(parts[0])
    for part in parts[1:]:
        module = getattr(module, part)
    return module
