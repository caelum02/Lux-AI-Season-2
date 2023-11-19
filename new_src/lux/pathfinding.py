# https://gist.github.com/DanilAndreev/c6875c21c2900cd87b54fcbfebd66a4f
import sys
import numpy as np
from lux.states import Route

def taxi_dist(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy

def get_trace(position: tuple, trace_map: list) -> list:
    """
    get_trace - function for getting trace from trace map.
    :param position: Destination point | tuple(y, x)
    :param trace_map: Trace map.
    :return: Trace.
    """
    trace = []
    prev = position
    # Creating trace chain.
    while prev is not None:
        trace.append(prev)
        prev = trace_map[prev[0]][prev[1]]
        
    return trace


def a_star(start: tuple, end: tuple, weight_map: list, shifts: list, ban_set=set()) -> list:
    """
    a_star - function for calculating shortest way weight for each node using A* algorithm.
    https://en.wikipedia.org/wiki/A*_search_algorithm
    :param start: Start cords | tuple(y, x)
    :param end: Destination point cords | tuple(y, x)
    :param weight_map: Two dimensional list with connection weights.
    :param shifts: List with shifting tuples | tuple(y, x, weight)
    :return: List as track with cords.
    """
    # Generating route weights map.
    field = [[None for x in row] for row in weight_map]
    # Generating trace map. (Information to restore routes)
    trace = [[None for x in row] for row in weight_map]

    # Creating queue for nodes. Detected nods will appear here.
    queue: list = []
    # Here will appear checked nodes.
    examined: set = set()

    # Marking start point with zero weight.
    field[start[0]][start[1]] = 0
    # Adding start point to queue.
    queue.append(start)

    # While queue is not empty - process nodes.
    while len(queue):
        # Get current cords from the queue.
        current_cords = queue.pop(0)

        # If we have reached destination - stop.
        if current_cords == end:
            break

        # If this cords ware examined - skip.
        if current_cords in examined:
            continue
        # Getting current node calculated weight.
        current_weigh = field[current_cords[0]][current_cords[1]]
        # Adding this node to examined.
        examined.add(current_cords)
        # For each node connection:
        for shift in shifts:
            # Getting connected node cords.
            cords = (current_cords[0] + shift[0], current_cords[1] + shift[1])
            # If cords out of field range - continue. (For example on field border nodes)
            if cords[0] not in range(0, len(field)) or cords[1] not in range(0, len(field[0])):
                continue
            # Checking if this node hasn't examined yet.
            if cords not in examined:
                # Getting connection weight.
                weigh = weight_map[cords[0]][cords[1]]
                if weigh == -1 or (current_cords, cords) in ban_set:
                    continue
                # Adding connected not to the queue.
                # We don't check if it is already in queue because of node examination check on each step in while loop.
                queue.append(cords)
                # If calculated weight is lower - assign new weight value.
                if field[cords[0]][cords[1]] is None or field[cords[0]][cords[1]] > current_weigh + weigh:
                    field[cords[0]][cords[1]] = current_weigh + weigh
                    trace[cords[0]][cords[1]] = current_cords

                # Sort queue by sum of cord weight and expected remaining weight (for example, Cartesian distance)
                queue = sorted(queue, key=lambda cord: taxi_dist(cord, end) + field[cord[0]][cord[1]])
    if field[end[0]][end[1]] is None:
        return [], -1
    return get_trace(end, trace), field[end[0]][end[1]]

def dijkstra(start: tuple, weight_map: list, shifts: list, ban_set=set()) -> tuple:
    """
    dijkstra - function for calculating shortest way weight for each node using Dijkstra algorithm.
    https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
    :param start: Start cords | tuple(y, x)
    :param weight_map: Two dimensional list with connection weights.
    :param shifts: List with shifting tuples | tuple(y, x, weight)
    :return: Tuple with calculated route weights and trace map.
    """
    # Generating route weights map.
    field = [[None for x in row] for row in weight_map]
    # Generating trace map. (Information to restore routes)
    trace = [[None for x in row] for row in weight_map]

    # Creating queue for nodes. Detected nods will appear here.
    queue: list = []
    # Here will appear checked nodes.
    examined: set = set()

    # Marking start point with zero weight.
    field[start[0]][start[1]] = 0
    # Adding start point to queue.
    queue.append(start)

    # While queue is not empty - process nodes.
    while len(queue):
        # Get current cords from the queue.
        current_cords = queue.pop(0)
        # If this cords ware examined - skip.
        if current_cords in examined:
            continue
        # Getting current node calculated weight.
        current_weigh = field[current_cords[0]][current_cords[1]]
        # Adding this node to examined.
        examined.add(current_cords)
        # For each node connection:
        for shift in shifts:
            # Getting connected node cords.
            cords = (current_cords[0] + shift[0], current_cords[1] + shift[1])
            # If cords out of field range - continue. (For example on field border nodes)
            if cords[0] not in range(0, len(field)) or cords[1] not in range(0, len(field[0])):
                continue
            # Checking if this node hasn't examined yet.
            if cords not in examined:
                # Getting connection weight.
                weigh = weight_map[cords[0]][cords[1]]
                if weigh == -1 or (current_cords, cords) in ban_set:
                    continue
                # Adding connected not to the queue.
                # We don't check if it is already in queue because of node examination check on each step in while loop.
                queue.append(cords)
                # If calculated weight is lower - assign new weight value.
                if field[cords[0]][cords[1]] is None or field[cords[0]][cords[1]] > current_weigh + weigh:
                    field[cords[0]][cords[1]] = current_weigh + weigh
                    trace[cords[0]][cords[1]] = current_cords
    return field, trace


SHIFTS = [
    # Up
    (0, -1),
    # Right
    (1, 0),
    # Down
    (0, 1),
    # Left
    (-1, 0),
]

SHIFT_DIRECTIONS = {
    (0, -1): 1,
    (1, 0): 2,
    (0, 1): 3,
    (-1, 0): 4,
}


def get_shortest_loop(rubble_map, start, end, ban_list=[], min_length=0):
    rubble_map = rubble_map.copy()
    start = tuple(start)
    end = tuple(end)
    for pos in ban_list:
        rubble_map[pos[0], pos[1]] = -1
    if min_length > 0:
        raise NotImplementedError()
    rubble_map = rubble_map.tolist()
    trace, cost = a_star(start, end, rubble_map, SHIFTS)
    if cost == -1:
        return None
    trace = list(reversed(trace))
    for pos in trace:
        if pos != start:
            rubble_map[pos[0]][pos[1]] = -1
    trace_back, cost_back = a_star(end, start, rubble_map, SHIFTS, ban_set={(trace[1], trace[0])})
    if cost_back == -1:
        return None
    trace_back = list(reversed(trace_back))
    trace_back.pop(0)
    trace = trace + trace_back
    cost = cost + cost_back
    return Route(
        start=start,
        end=end,
        path=trace,
        cost=np.ceil(cost),
    )

# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def get_avoiding_direction(route, start):
    start = tuple(start)
    path = route.path
    if start in path:
        idx = path.index(start)
        before_idx = (idx - 1) % (len(path) - 1)
        next_idx = (idx + 1) % (len(path) - 1)
        before = path[before_idx]
        next = path[next_idx]
        scores = {}
        for shift, direction in SHIFT_DIRECTIONS.items():
            new_pos = (start[0] + shift[0], start[1] + shift[1])
            if new_pos == before or new_pos == next:
                continue
            if new_pos in path:
                scores[direction] = 1
            else:
                return direction
        for direction in scores:
            return direction
    return 0
