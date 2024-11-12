import random
import numpy as np
import torch
import torch.nn.functional as F

def crop_and_resize(data, valid_frames, interval, window_size):
    channels, frames, vertices, persons = data.shape
    total_valid = valid_frames

    cropped_data = _apply_crop(data, total_valid, interval)
    resized_data = _resize_data(cropped_data, channels, vertices, persons, window_size)

    return resized_data


def _apply_crop(data, total_valid, interval):
    if len(interval) == 1:
        p = interval[0]
        offset = int((1 - p) * total_valid / 2)
        cropped_data = data[:, offset:total_valid - offset, :, :]
    else:
        p = np.random.uniform(*interval)
        cropped_len = min(max(int(total_valid * p), 64), total_valid)
        offset = np.random.randint(0, total_valid - cropped_len + 1)
        cropped_data = data[:, offset:offset + cropped_len, :, :]
    return cropped_data


def _resize_data(data, channels, vertices, persons, window_size):
    data = torch.tensor(data, dtype=torch.float32)
    data = data.permute(0, 2, 3, 1).reshape(-1, data.shape[1])[None, None, :, :]
    resized_data = F.interpolate(data, size=(channels * vertices * persons, window_size),
                                 mode='bilinear', align_corners=False).squeeze()
    return resized_data.reshape(channels, vertices, persons, window_size).permute(0, 3, 1, 2).numpy()


def downsample_data(data, factor, randomize=True):
    start_idx = np.random.randint(factor) if randomize else 0
    return data[:, start_idx::factor, :, :]


def temporal_slice(data, factor):
    channels, frames, vertices, persons = data.shape
    return data.reshape(channels, frames // factor, factor, vertices, persons).transpose(0, 1, 3, 2, 4).reshape(
        channels, frames // factor, vertices, factor * persons)


def subtract_mean(data, mean_val):
    if mean_val == 0:
        return data
    start, end = _get_valid_frame_range(data)
    data[:, :end, :, :] -= mean_val
    return data


def _get_valid_frame_range(data):
    valid_frames = (data != 0).sum(axis=(3, 2, 0)) > 0
    start = valid_frames.argmax()
    end = len(valid_frames) - valid_frames[::-1].argmax()
    return start, end


def pad_data(data, target_size, randomize=False):
    channels, frames, vertices, persons = data.shape
    if frames < target_size:
        offset = random.randint(0, target_size - frames) if randomize else 0
        padded_data = np.zeros((channels, target_size, vertices, persons))
        padded_data[:, offset:offset + frames, :, :] = data
        return padded_data
    return data


def random_crop(data, crop_size, pad_if_needed=True):
    channels, frames, vertices, persons = data.shape
    if frames == crop_size:
        return data
    elif frames < crop_size and pad_if_needed:
        return pad_data(data, crop_size, randomize=True)
    else:
        start = random.randint(0, frames - crop_size)
        return data[:, start:start + crop_size, :, :]


def apply_random_shift(data):
    channels, frames, vertices, persons = data.shape
    shifted_data = np.zeros_like(data)
    start, end = _get_valid_frame_range(data)
    offset = random.randint(0, frames - (end - start))
    shifted_data[:, offset:offset + (end - start), :, :] = data[:, start:end, :, :]
    return shifted_data


def random_rotation(data, theta=0.3):
    data_tensor = torch.tensor(data)
    channels, frames, vertices, persons = data_tensor.shape
    data_tensor = data_tensor.permute(1, 0, 2, 3).reshape(frames, channels, vertices * persons)
    rotation = torch.rand(3).uniform_(-theta, theta)
    rotation_matrices = create_rotation_matrix(torch.stack([rotation] * frames, dim=0))
    rotated_data = torch.matmul(rotation_matrices, data_tensor)
    return rotated_data.view(frames, channels, vertices, persons).permute(1, 0, 2, 3).numpy()


def create_rotation_matrix(rotations):
    cos_vals, sin_vals = rotations.cos(), rotations.sin()
    zeros, ones = torch.zeros_like(cos_vals[:, 0:1]), torch.ones_like(cos_vals[:, 0:1])
    
    rx = _create_single_rotation_matrix(ones, zeros, cos_vals[:, 0:1], sin_vals[:, 0:1])
    ry = _create_single_rotation_matrix(cos_vals[:, 1:2], zeros, ones, sin_vals[:, 1:2])
    rz = _create_single_rotation_matrix(cos_vals[:, 2:3], sin_vals[:, 2:3], ones, zeros)
    
    return rz @ ry @ rx


def _create_single_rotation_matrix(cos_val, sin_val, ones, zeros):
    return torch.stack((cos_val, sin_val, zeros, -sin_val, cos_val, zeros, zeros, zeros, ones), dim=-1).reshape(-1, 3, 3)


def match_pose(data):
    channels, frames, vertices, persons = data.shape
    assert channels == 3
    score = data[2, :, :, :].sum(axis=1)
    rank = (-score[:-1]).argsort(axis=1).reshape(frames - 1, persons)

    forward_map = _generate_forward_map(data, rank, frames, persons)
    matched_data = _apply_forward_map(data, forward_map, frames, persons)
    return matched_data[:, :, :, (-matched_data[2].sum(axis=1).sum(axis=0)).argsort()]


def _generate_forward_map(data, rank, frames, persons):
    xy1, xy2 = data[:2, :-1].reshape(2, frames - 1, -1, persons, 1), data[:2, 1:].reshape(2, frames - 1, -1, 1, persons)
    distances = ((xy2 - xy1) ** 2).sum(axis=(0, 2))
    forward_map = np.full((frames, persons), -1, dtype=int)
    forward_map[0] = np.arange(persons)
    
    for m in range(persons):
        matched_indices = distances[rank == m].argmin(axis=1)
        for t in range(frames - 1):
            distances[t, :, matched_indices[t]] = np.inf
        forward_map[1:][rank == m] = matched_indices
    
    forward_map[1:] = forward_map[1:][forward_map[:-1]]
    return forward_map


def _apply_forward_map(data, forward_map, frames, persons):
    matched_data = np.zeros_like(data)
    for t in range(frames):
        matched_data[:, t, :, :] = data[:, t, :, forward_map[t]]
    return matched_data
