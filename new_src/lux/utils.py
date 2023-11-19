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

    greedy = np.random.binomial(n=1, p=0.7, size=1)[0]
    
    if greedy == 1:
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
    else:
        if 0 < abs(dx) <= abs(dy):
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


factory_tile_deltas = (-1, 0, 1)
factory_tile_deltas = np.array(list(product(factory_tile_deltas, factory_tile_deltas)))
def get_factory_tiles(factory_center):
    return factory_center + factory_tile_deltas
