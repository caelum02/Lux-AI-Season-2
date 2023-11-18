from enum import IntEnum

class UnitStates(IntEnum):
    MOVING_TO_RESOURCE = 0
    DIGGING_RESOURCE = 1
    MOVING_TO_FACTORY = 2
    DROPPING_OFF_RESOURCE = 3
    RECHARING = 4

