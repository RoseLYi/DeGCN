import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class SkeletonDataset(Dataset):
    def __init__(self, data_path, label_path=None, interval=1, mode='train', random_select=False, random_shift=False,
                 random_move=False, apply_rotation=False, seq_length=-1, normalize=False, debug=False, mmap_mode=False,
                 use_bone=False, apply_velocity=False):
        """
        :param data_path: Path to the data file
        :param label_path: Path to the label file
        :param mode: Mode for dataset - either 'train' or 'test'
        :param random_select: Randomly select a portion of the sequence
        :param random_shift: Randomly shift the sequence with padding
        :param random_move: Apply random movements to the sequence
        :param apply_rotation: Rotate skeleton around xyz axis
        :param seq_length: Target length for the sequence
        :param normalize: Apply normalization
        :param debug: Enable debug mode for limited samples
        :param mmap_mode: Use memory mapping for data loading
        :param use_bone: Enable bone modality
        :param apply_velocity: Enable velocity modality
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.random_select = random_select
        self.random_shift = random_shift
        self.random_move = random_move
        self.seq_length = seq_length
        self.normalize = normalize
        self.mmap_mode = mmap_mode
        self.interval = interval
        self.apply_rotation = apply_rotation
        self.use_bone = use_bone
        self.apply_velocity = apply_velocity
        # Load data during initialization
        self._load_data()

    def _load_data(self):
        # Load data based on mode
        data_content = np.load(self.data_path)
        if self.mode == 'train':
            self.data = data_content['x_train']
            self.labels = data_content['y_train']
            self.sample_names = [f'train_{i}' for i in range(len(self.data))]
        elif self.mode == 'test':
            self.data = data_content['x_test']
            self.labels = data_content['y_test']
            self.sample_names = [f'test_{i}' for i in range(len(self.data))]
        else:
            raise ValueError('Mode should be either "train" or "test"')

        self.data = self.data.transpose(0, 4, 1, 3, 2)

    def _compute_mean_std(self):
        data_shape = self.data.shape
        self.mean_map = self.data.mean(axis=(2, 4), keepdims=True).mean(axis=0)
        self.std_map = self.data.transpose((0, 2, 4, 1, 3)).reshape((-1, data_shape[1] * data_shape[3])).std(axis=0).reshape((data_shape[1], 1, data_shape[3], 1))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        label = self.labels[idx]
        sample_data = np.array(sample_data)

        if not sample_data.any():
            sample_data = np.array(self.data[0])

        valid_frames = np.sum(sample_data.sum(0).sum(-1).sum(-1) != 0)
        sample_data = tools.valid_crop_resize(sample_data, valid_frames, self.interval, self.seq_length)

        if self.apply_rotation:
            sample_data = tools.random_rot(sample_data)

        if self.use_bone:
            from .bone_pairs import ntu_pairs
            bone_data = np.zeros_like(sample_data)
            for start, end in ntu_pairs:
                bone_data[:, :, start - 1] = sample_data[:, :, start - 1] - sample_data[:, :, end - 1]
            sample_data = bone_data

        if self.apply_velocity:
            sample_data[:, :-1] = sample_data[:, 1:] - sample_data[:, :-1]
            sample_data[:, -1] = 0

        return sample_data, label, idx

    def top_k_accuracy(self, scores, top_k):
        rankings = scores.argsort()
        hit_top_k = [lbl in rankings[i, -top_k:] for i, lbl in enumerate(self.labels)]
        return sum(hit_top_k) / len(hit_top_k)


def load_class(name):
    parts = name.split('.')
    module = __import__(parts[0])
    for part in parts[1:]:
        module = getattr(module, part)
    return module

