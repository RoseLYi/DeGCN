import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, split='train', p_interval=1, random_config=None,
                 window_size=-1, normalization=False, debug=False, use_mmap=False, bone=False, vel=False):
        """
        :param data_path
        :param label_path
        :param split
        :param random_config
        :param window_size
        :param normalization
        :param debug
        :param bone
        :param vel
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

        self.data, self.label = self._load_data()
        if self.normalization:
            self.mean_map, self.std_map = self._calculate_mean_std()

    def _load_data(self):
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            data, labels = npz_data['x_train'], np.where(npz_data['y_train'] > 0)[1]
        else:
            data, labels = npz_data['x_test'], np.where(npz_data['y_test'] > 0)[1]
        
        # 调整数据格式
        data = data.reshape((len(data), -1, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        return data, labels

    def _calculate_mean_std(self):
        N, C, T, V, M = self.data.shape
        mean_map = self.data.mean(axis=(2, 4), keepdims=True).mean(axis=0)
        std_map = self.data.transpose(0, 2, 4, 1, 3).reshape(N * T * M, C * V).std(axis=0).reshape(C, 1, V, 1)
        return mean_map, std_map

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = np.array(self.data[index])
        label = self.label[index]

        valid_frames = np.sum(data.sum(axis=(0, -1, -1)) != 0)
        data = tools.valid_crop_resize(data, valid_frames, self.p_interval, self.window_size)

        # 应用数据增强
        data = self._apply_data_augmentation(data)
        
        return data, label, index

    def _apply_data_augmentation(self, data):
        if self.random_config.get('rot'):
            data = tools.random_rot(data)
        if self.bone:
            data = self._apply_bone_mode(data)
        if self.vel:
            data = self._apply_velocity_mode(data)
        return data

    def _apply_bone_mode(self, data):
        from .bone_pairs import ntu_pairs
        bone_data = np.zeros_like(data)
        for v1, v2 in ntu_pairs:
            bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
        return bone_data

    def _apply_velocity_mode(self, data):
        data[:, :-1] = data[:, 1:] - data[:, :-1]
        data[:, -1] = 0
        return data

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = sum(label in rank[i, -top_k:] for i, label in enumerate(self.label))
        return hit_top_k / len(self.label)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
