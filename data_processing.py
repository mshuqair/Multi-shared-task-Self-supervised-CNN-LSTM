from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat
import numpy as np
import math


# Several data processing functions

def mean_std(data, axis=0):
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    return mean, std


# Calculate log2 of N
def next_pow_2(n):
    a = int(math.log2(n))
    return n if 2 ** a == n else 2 ** (a + 1)


def windowing(data, size, overlap_rate):
    start = 0
    while start < (len(data) - size):   # iterate through all timestamps
        yield start, start + size       # returns the start and stop indices for a given window
        start += int(size * (1 - overlap_rate))     # uses a step size


# This the main function for data segmentation
def segment(features, targets, window_size, num_cols, overlap_rate):
    step_size = int(window_size * (1 - overlap_rate))
    segments = np.zeros((int(len(features) / step_size), window_size, num_cols))
    labels = np.zeros((int(len(targets) / step_size)))
    i_segment, i_label = 0, 0

    for (start, end) in windowing(features, window_size, overlap_rate):
        if (len(features[start:end]) == window_size and
                i_segment != len(segments) and i_label != len(labels)):
            if len(np.unique(targets[start:end])) == 1:
                segments[i_segment] = features[start:end]
                labels[i_label] = np.unique(targets[start:end])[0]
                i_label += 1
                i_segment += 1

    return segments[:i_segment], labels[:i_label]       # Remove empty windows resulted from segmentation


# Data augmentation functions

# This function returns the original signal
def da_original(x):
    return x


# Adding noise
def da_jitter(x, sigma=0.05):
    return x + np.random.normal(loc=0, scale=sigma, size=x.shape)


# Apply scaling
def da_scaling(x, sigma=0.1):
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, x.shape[1]))
    return x * np.matmul(np.ones((x.shape[0], 1)), scaling_factor)


# This example using cubic splice is not the best approach to generate random curves.
# You can use other approaches, e.g., Gaussian process regression, Bézier curve, etc.
def generate_random_curves(x, sigma, knot):
    xx = (np.ones((x.shape[1], 1)) * (np.arange(0, x.shape[0], (x.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, x.shape[1]))
    x_range = np.arange(x.shape[0])
    return np.array([CubicSpline(xx[:, i], yy[:, i])(x_range) for i in range(x.shape[1])]).T


# Apply Magnitude Warping
def da_mag_warp(x, sigma=0.2, knot=4):
    return x * generate_random_curves(x, sigma, knot)


def distort_time_steps(x, sigma, knot):
    tt = generate_random_curves(x, sigma, knot)     # Regard these samples around 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)      # Add intervals to make a cumulative graph
    t_scale = [(x.shape[0] - 1) / tt_cum[-1, i] for i in range(x.shape[1])]
    for i in range(3):
        tt_cum[:, i] *= t_scale[i]
    return tt_cum


# Apply time warping
def da_time_warp(x, sigma=0.2, knot=4):
    tt_new = distort_time_steps(x, sigma, knot)
    x_range = np.arange(x.shape[0])
    return np.stack([np.interp(x_range, tt_new[:, i], x[:, i]) for i in range(3)], axis=-1)


# Apply rotation
def da_rotation(x):
    axis = np.random.uniform(low=-1, high=1, size=x.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(x, axangle2mat(axis, angle))


# Apply permutation
def da_permutation(x, n_perm=4, min_seg_length=100):
    x_new = np.zeros(x.shape)
    while True:
        segments = np.zeros(n_perm + 1, dtype=int)
        segments[1:-1] = np.sort(np.random.randint(min_seg_length, x.shape[0] - min_seg_length, n_perm - 1))
        segments[-1] = x.shape[0]
        if np.min(segments[1:] - segments[:-1]) > min_seg_length:
            break
    idx = np.random.permutation(n_perm)
    pp = 0
    for j in range(n_perm):
        seg = x[segments[idx[j]]:segments[idx[j] + 1], :]
        x_new[pp:pp + len(seg), :] = seg
        pp += len(seg)
    return x_new
