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
    MOVING_TO_START = 1
    MOVING_TO_TARGET = 2
    PERFORMING_ROLE = 3
    PERFORMING_SECONDARY_ROLE = 4
    MOVING_TO_FACTORY = 5
    DROPPING_RESOURCE = 6
    PICKING_RESOURCE = 7
    MOVING_TO_RUBBLE = 8
    DIGGING_RUBBLE = 9
    RUBBLE_MOVING_TO_FACTORY = 10
    RUBBLE_RECHARGING = 11
    


class UnitMission(IntEnum):
    PIPE_FACTORY_TO_FACTORY = 0
    PIPE_FACTORY_TO_ICE = 1
    PIPE_MINE_ICE = 2
    PIPE_FACTORY_TO_ORE = 3
    PIPE_MINE_ORE = 4
    DIG_RUBBLE = 5
    NONE = 6

    @property
    def resource_type(self):
        if self in [UnitMission.PIPE_FACTORY_TO_ICE, UnitMission.PIPE_MINE_ICE]:
            return "ice"
        elif self in [UnitMission.PIPE_FACTORY_TO_ORE, UnitMission.PIPE_MINE_ORE]:
            return "ore"
        elif self == UnitMission.PIPE_FACTORY_TO_FACTORY:
            return "factory_to_factory"
        else:
            return None


class UnitRole(IntEnum):
    STATIONARY_MINER = 0
    MINER_TRANSPORTER = 1
    STATIONARY_TRANSPORTER = 2
    TRANSPORTER = 3
    RUBBLE_DIGGER = 4
    NONE = 5

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
    role : UnitRole | None = None
    target_pos : Position | None = None
    route_cache: Route | None = None
    # Invalid state when role or idle_pos changed

    def set_role(self, role: UnitRole):
        self.role = role
        self.state = UnitStateEnum.INITIAL
    
    def set_target_pos(self, pos: Position):
        self.target_pos = pos
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
    ban_list : list[Position] | None = None
    empty_factory_locs: list[Position] | None = None
    ore_disabled: bool = False
    MAX_DIGGER: int = 3

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
