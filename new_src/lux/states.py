import dataclasses
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Literal
import numpy as np

Position = tuple[int, int]
Resource = Literal["ice", "ore", "water", "metal", "power"]

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
        return self in [
            UnitRole.MINER_TRANSPORTER,
            UnitRole.STATIONARY_TRANSPORTER,
            UnitRole.TRANSPORTER,
        ]

    @property
    def is_miner(self):
        return self in [UnitRole.STATIONARY_MINER, UnitRole.MINER_TRANSPORTER]


@dataclass
class UnitState:
    state: UnitStateEnum = UnitStateEnum.INITIAL
    following_route: Route | None = None
    mission: UnitMission | None = None
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
class Plan:
    destination: Position
    source: Position
    route: Route
    max_route_robots: int

    @classmethod
    def from_plan(cls, plan: "Plan"):
        return cls(
            destination=plan.destination,
            source=plan.source,
            route=plan.route,
            max_route_robots=plan.max_route_robots,
        )


@dataclass
class ResourcePlan(Plan):
    power_per_turn: int = None
    last_power_pickup: int = None


@dataclass
class TransmitPlan(Plan):
    pass


class FactoryRole(IntEnum):
    MAIN = 0
    SUB = 1


@dataclass
class FactoryState:
    plans: dict[str, ResourcePlan | TransmitPlan] | None = None
    robot_missions: dict[UnitMission, list[UnitId]] | None = None
    role: FactoryRole | None = None
    main_factory: FactoryId | None = None
    sub_factory: FactoryId | None = None

    def __post_init__(self):
        if self.robot_missions is None:
            self.robot_missions = {mission: [] for mission in UnitMission}


@dataclass
class EarlyStepState:
    rubble_score: np.ndarray = None
    factory_score: np.ndarray = None
    resource_score: np.ndarray = None
    latest_main_factory: FactoryId = None
    sub_factory_map: dict[FactoryId, FactoryId] = field(default_factory=dict)
