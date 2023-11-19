from enum import IntEnum
from dataclasses import dataclass

Position = tuple[int, int]

@dataclass
class Route:
    start: Position
    end: Position
    path: list[Position]
    cost: float
    def __len__(self):
        return len(self.path)
    
class UnitStateEnum(IntEnum):
    MOVING_TO_START = 0
    MOVING_TO_RESOURCE = 1
    DIGGING = 2
    MOVING_TO_FACTORY = 3
    DROPPING_RESOURCE = 4
    RECHARGING = 5

@dataclass
class UnitState:    
    state: UnitStateEnum = UnitStateEnum.MOVING_TO_START
    following_route: Route | None = None

@dataclass
class ResourcePlan:
    resource_pos: Position
    resource_factory_pos: Position
    resource_route: Route
    max_resource_robots: int
    resource_threshold_light: int
    
@dataclass
class FactoryState:
    resources: dict[str, ResourcePlan]
