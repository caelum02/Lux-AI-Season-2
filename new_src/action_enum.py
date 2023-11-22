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

    @property
    def opposite_direction(self):
        if self == DIRECTION_T.CENTER:
            return DIRECTION_T.CENTER
        elif self == DIRECTION_T.UP:
            return DIRECTION_T.DOWN
        elif self == DIRECTION_T.DOWN:
            return DIRECTION_T.UP
        elif self == DIRECTION_T.LEFT:
            return DIRECTION_T.RIGHT
        elif self == DIRECTION_T.RIGHT:
            return DIRECTION_T.LEFT
        else:
            raise Exception("Invalid direction")
    
    @property
    def orthogonal_directions(self):
        if self == DIRECTION_T.CENTER:
            return []
        elif self == DIRECTION_T.UP:
            return [DIRECTION_T.LEFT, DIRECTION_T.RIGHT]
        elif self == DIRECTION_T.DOWN:
            return [DIRECTION_T.LEFT, DIRECTION_T.RIGHT]
        elif self == DIRECTION_T.LEFT:
            return [DIRECTION_T.UP, DIRECTION_T.DOWN]
        elif self == DIRECTION_T.RIGHT:
            return [DIRECTION_T.UP, DIRECTION_T.DOWN]
        else:
            raise Exception("Invalid direction")

    @classmethod
    def from_float(cls, f):
        if f == 0:
            return cls.CENTER
        elif f == 1:
            return cls.UP
        elif f == 2:
            return cls.RIGHT
        elif f == 3:
            return cls.DOWN
        elif f == 4:
            return cls.LEFT
        else:
            raise Exception("Invalid direction float")