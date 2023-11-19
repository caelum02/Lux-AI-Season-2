from enum import IntEnum
from dataclasses import dataclass
from typing import Literal
import numpy as np

Position = tuple[int, int]
Resource = Literal["ice", "ore"]

FactoryId = str
UnitId = str

@dataclass
class Route:
    start: Position
    end: Position
    path: list[Position]
    cost: float
    def __len__(self):
        return len(self.path)
    
class UnitStateEnum(IntEnum):
    INITIAL = 0
    MOVING_TO_RESOURCE = 1
    DIGGING = 2
    MOVING_TO_FACTORY = 3
    DROPPING_RESOURCE = 4
    RECHARGING = 5
    TRANSFERING_RESOURCE = 6

class UnitMission(IntEnum):
    PIPE_FACTORY_TO_FACTORY = 0
    PIPE_FACTORY_TO_ICE = 1
    PIPE_MINE_ICE = 2
    PIPE_FACTORY_TO_ORE = 3
    PIPE_MINE_ORE = 4
    DIG_RUBBLE = 5

    @property
    def resource_type(self):
        if self in [UnitMission.PIPE_FACTORY_TO_ICE, UnitMission.PIPE_MINE_ICE]:
            return "ice"
        elif self in [UnitMission.PIPE_FACTORY_TO_ORE, UnitMission.PIPE_MINE_ORE]:
            return "ore"
        else:
            return None

class UnitRole(IntEnum):
    STATIONARY_MINER = 0
    MINER_TRANSPORTER = 1
    STATIONARY_TRANSPORTER = 2
    TRANSPORTER = 3
    RUBBLE_DIGGER = 4

    @property
    def is_stationary(self):
        return self in [UnitRole.STATIONARY_MINER, UnitRole.STATIONARY_TRANSPORTER]
    
    @property
    def is_transporter(self):
        return self in [UnitRole.MINER_TRANSPORTER, UnitRole.STATIONARY_TRANSPORTER, UnitRole.TRANSPORTER]
    
    @property
    def is_miner(self):
        return self in [UnitRole.STATIONARY_MINER, UnitRole.MINER_TRANSPORTER]

@dataclass
class UnitState:    
    state: UnitStateEnum = UnitStateEnum.INITIAL
    following_route: Route | None = None
    mission : UnitMission | None = None
    resource_type: Resource | None = None
    owner: FactoryId | None = None
    # role : UnitRole | None = None
    # stay_pos : Position | None = None
    # Invalid state when role or idle_pos changed

    def __post_init__(self):
        self.__role = None
        self.__stay_pos = None
    
    @property
    def role(self):
        return self.__role

    @property
    def stay_pos(self):
        return self.__stay_pos
    
    @role.setter
    def set_role(self, role: UnitRole):
        if self.__role != role:
            self.__role = role
            self.state = UnitStateEnum.INITIAL
    
    @stay_pos.setter
    def set_stay_pos(self, pos: Position):
        if np.any(self.__stay_pos != pos):
            self.__stay_pos = pos
            self.state = UnitStateEnum.INITIAL


@dataclass
class ResourcePlan:
    resource_pos: Position
    resource_factory_pos: Position
    resource_route: Route
    max_resource_robots: int
    resource_threshold_light: int

class FactoryRole(IntEnum):
    MAIN = 0
    SUB = 1

@dataclass
class FactoryState:
    resources: dict[str, ResourcePlan]
    robot_missions: dict[UnitMission, list[UnitId]] = None
    role : FactoryRole | None = None
    main_factory : FactoryId | None = None
    sub_factory : FactoryId | None = None

    def __post_init__(self):
        if robot_missions is None:
            robot_missions = {mission: [] for mission in UnitMission}
    
@dataclass
class EarlyStepState:
    map_size: int = 64
    rubble_score: np.ndarray = None
    factory_score: np.ndarray = None
    resource_score: np.ndarray = None
    latest_main_factory: FactoryId = None
    def  __post_init__(self):
        size = 64
        if self.rubble_score is None:
            self.rubble_score = np.zeros((size, size))
        if self.factory_score is None:
            self.factory_score = np.zeros((size, size))
        if self.resource_score is None:
            self.resource_score = np.zeros((size, size))
