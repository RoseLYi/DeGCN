import numpy as np
from .feeder_uav import Feeder
from feeders import tools

class UAVHumanFeeder(Feeder):
    def __init__(self, data_path, label_path=None, interval=1, mode='train', random_select=False, random_shift=False,
                 random_move=False, apply_rotation=False, seq_length=-1, normalize=False, debug=False, mmap_mode=False,
                 use_bone=False, apply_velocity=False):  
        super().__init__(data_path, label_path, interval, mode, random_select, random_shift,
                         random_move, apply_rotation, seq_length, normalize, debug, mmap_mode, 
                         use_bone, apply_velocity)
        self.load_data()
        if normalize:
            self.compute_mean_map()

    def load_data(self):
        data, labels = np.load(self.data_path), np.load(self.label_path)
        prefix = 'test_' if self.split == 'test' else 'train_'

        if not self.debug:
            self.data, self.label = data, labels
            self.sample_name = [f"{prefix}{i}" for i in range(len(data))]
        else:
            self.data, self.label = data[:100], labels[:100]
            self.sample_name = [f"{prefix}{i}" for i in range(100)]

    def __getitem__(self, index):
        data_sample = np.array(self.data[index])  
        label = self.label[index] 

        if not np.any(data_sample): 
            data_sample = np.array(self.data[0])

        valid_frames = np.sum(data_sample.sum(0).sum(-1).sum(-1) != 0)  
        
        if valid_frames == 0:
            data_sample = np.zeros((2, 64, 17, 300))

        data_sample = tools.valid_crop_resize(data_sample, valid_frames, self.p_interval, self.window_size)
        
        if self.random_rot:
            data_sample = tools.random_rot(data_sample)
        if self.bone:
            from .bone_pairs import coco_pairs
            bone_sample = np.zeros_like(data_sample)
            for start, end in coco_pairs:
                bone_sample[:, :, start - 1] = data_sample[:, :, start - 1] - data_sample[:, :, end - 1]
            data_sample = bone_sample
        if self.vel:
            data_sample[:, :-1] = data_sample[:, 1:] - data_sample[:, :-1]
            data_sample[:, -1] = 0

        return data_sample, label, index
