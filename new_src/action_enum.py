from enum import IntEnum

class ACTION_T(IntEnum):
    MOVE = 0
    TRANSFER = 1
    PICKUP = 2
    DIG = 3
    SELF_DESTRUCT = 4
    RECHARGE = 5

class RESOURCE_T(IntEnum):
    ICE = 0
    ORE = 1
    WATER = 2
    METAL = 3
    POWER = 4

class DIRECTION_T(IntEnum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4