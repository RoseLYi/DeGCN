import random
import numpy as np
import torch
import torch.nn.functional as F


def crop_and_resize(data, valid_frames, interval, window_size):
    channels, frames, vertices, persons = data.shape
    start, end = 0, valid_frames
    total_valid = end - start

    # Crop
    if len(interval) == 1:
        p = interval[0]
        offset = int((1 - p) * total_valid / 2)
        cropped_data = data[:, start + offset:end - offset, :, :]
    else:
        p = np.random.uniform(*interval)
        cropped_len = min(max(int(total_valid * p), 64), total_valid)
        offset = np.random.randint(0, total_valid - cropped_len + 1)
        cropped_data = data[:, start + offset:start + offset + cropped_len, :, :]

    # Resize
    cropped_data = torch.tensor(cropped_data, dtype=torch.float32)
    cropped_data = cropped_data.permute(0, 2, 3, 1).reshape(-1, cropped_data.shape[1])[None, None, :, :]
    resized_data = F.interpolate(cropped_data, size=(channels * vertices * persons, window_size),
                                 mode='bilinear', align_corners=False).squeeze()
    resized_data = resized_data.reshape(channels, vertices, persons, window_size).permute(0, 3, 1, 2).numpy()

    return resized_data


def downsample_data(data, factor, randomize=True):
    start_idx = np.random.randint(factor) if randomize else 0
    return data[:, start_idx::factor, :, :]


def temporal_slice(data, factor):
    channels, frames, vertices, persons = data.shape
    return data.reshape(channels, frames // factor, factor, vertices, persons).transpose(0, 1, 3, 2, 4).reshape(
        channels, frames // factor, vertices, factor * persons)


def subtract_mean(data, mean_val):
    if mean_val == 0:
        return
    channels, frames, vertices, persons = data.shape
    valid_frames = (data != 0).sum(axis=(3, 2, 0)) > 0
    start, end = valid_frames.argmax(), len(valid_frames) - valid_frames[::-1].argmax()
    data[:, :end, :, :] -= mean_val
    return data


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
    valid_frames = (data != 0).sum(axis=(3, 2, 0)) > 0
    start, end = valid_frames.argmax(), len(valid_frames) - valid_frames[::-1].argmax()
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
    rx = torch.stack((ones, zeros, zeros, zeros, cos_vals[:, 0:1], sin_vals[:, 0:1],
                      zeros, -sin_vals[:, 0:1], cos_vals[:, 0:1]), dim=-1).reshape(-1, 3, 3)
    ry = torch.stack((cos_vals[:, 1:2], zeros, -sin_vals[:, 1:2], zeros, ones, zeros,
                      sin_vals[:, 1:2], zeros, cos_vals[:, 1:2]), dim=-1).reshape(-1, 3, 3)
    rz = torch.stack((cos_vals[:, 2:3], sin_vals[:, 2:3], zeros, -sin_vals[:, 2:3], cos_vals[:, 2:3],
                      zeros, zeros, zeros, ones), dim=-1).reshape(-1, 3, 3)
    return rz @ ry @ rx


def match_pose(data):
    channels, frames, vertices, persons = data.shape
    assert channels == 3
    score = data[2, :, :, :].sum(axis=1)
    rank = (-score[:-1]).argsort(axis=1).reshape(frames - 1, persons)
    xy1, xy2 = data[:2, :-1, :, :].reshape(2, frames - 1, vertices, persons, 1), \
               data[:2, 1:, :, :].reshape(2, frames - 1, vertices, 1, persons)
    distances = ((xy2 - xy1) ** 2).sum(axis=(0, 2))
    forward_map = np.full((frames, persons), -1, dtype=int)
    forward_map[0] = np.arange(persons)
    for m in range(persons):
        match = (rank == m)
        best_match = distances[match].argmin(axis=1)
        for t in range(frames - 1):
            distances[t, :, best_match[t]] = np.inf
        forward_map[1:][match] = best_match
    forward_map[1:] = forward_map[1:][forward_map[:-1]]
    matched_data = np.zeros_like(data)
    for t in range(frames):
        matched_data[:, t, :, :] = data[:, t, :, forward_map[t]]
    return matched_data[:, :, :, (-matched_data[2].sum(axis=1).sum(axis=0)).argsort()]

