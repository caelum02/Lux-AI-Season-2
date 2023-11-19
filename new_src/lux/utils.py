import numpy as np
from itertools import product


def my_turn_to_place_factory(place_first: bool, step: int):
    if place_first:
        if step % 2 == 1:
            return True
    else:
        if step % 2 == 0:
            return True
    return False


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0

    if abs(dx) > abs(dy):
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1


def taxi_dist(a, b):
    return np.linalg.norm(a - b, ord=1, axis=-1)


def taxi_distances(a: np.ndarray, b: np.ndarray):
    return distances(a, b, ord=1)


def distances(a: np.ndarray, b: np.ndarray, ord):
    """
    Returns the distances between positions_1 and positions_2

    a: (n, 2) array
    b: (m, 2) array
    returns: (n, m) array
    """
    a = np.expand_dims(a, 1)
    b = np.expand_dims(b, 0)
    return np.linalg.norm(a - b, ord=ord, axis=-1)


factory_tile_deltas = (-1, 0, 1)
factory_tile_deltas = np.array(list(product(factory_tile_deltas, factory_tile_deltas)))


def get_factory_tiles(factory_center):
    return factory_center + factory_tile_deltas


def average_kernel(size):
    return np.ones((size, size)) / (size * size)


def conv2d(a, f, pad="zero", n=1):
    if pad == "zero":
        pad = (f.shape[0] - 1) // 2

    strd = np.lib.stride_tricks.as_strided
    a = np.pad(a, pad)
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    for i in range(n):
        if i > 0:
            a = np.pad(a, pad)
        subM = strd(a, shape=s, strides=a.strides * 2)
        a = np.einsum("ij,ijkl->kl", f, subM)
    return a
