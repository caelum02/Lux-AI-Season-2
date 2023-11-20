from dataclasses import dataclass
from config import resource_ids_inv

@dataclass
class UnitCargo:
    ice: int = 0
    ore: int = 0
    water: int = 0
    metal: int = 0

    def from_id(self, id_):
        return getattr(self, resource_ids_inv[id_])