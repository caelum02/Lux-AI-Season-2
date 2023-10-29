
```python
(
    Board(
        seed=ShapedArray(int32[]),
        factories_per_team=ShapedArray(int8[]),
        map=GameMap(
            rubble=ShapedArray(int8[64,64]),
            ice=ShapedArray(bool[64,64]),
            ore=ShapedArray(bool[64,64]),
            symmetry=ShapedArray(int8[])
        ),
        lichen=ShapedArray(int32[64,64]),
        lichen_strains=ShapedArray(int8[64,64]),
        units_map=ShapedArray(int16[64,64]),
        factory_map=ShapedArray(int8[64,64]),
        factory_occupancy_map=ShapedArray(int8[64,64]),
        factory_pos=ShapedArray(int8[22,2])
    ),
    Unit(
        unit_type=ShapedArray(int8[2,200]),
        action_queue=ActionQueue(
            data=UnitAction(
                action_type=ShapedArray(int8[2,200,20]),
                direction=ShapedArray(int8[2,200,20]),
                resource_type=ShapedArray(int8[2,200,20]),
                amount=ShapedArray(int16[2,200,20]),
                repeat=ShapedArray(int16[2,200,20]),
                n=ShapedArray(int16[2,200,20])
            ),
            front=ShapedArray(int8[2,200]),
            rear=ShapedArray(int8[2,200]),
            count=ShapedArray(int8[2,200])
        ),
        team_id=ShapedArray(int8[2,200]),
        unit_id=ShapedArray(int16[2,200]),
        pos=Position(pos=ShapedArray(int8[2,200,2])),
        cargo=UnitCargo(stock=ShapedArray(int32[2,200,4])),
        power=ShapedArray(int32[2,200])
    ),
    ShapedArray(int16[2000,2]), unit_id2idx
    ShapedArray(int16[2]), n_units
    Factory(
        team_id=ShapedArray(int8[2,11]),
        unit_id=ShapedArray(int8[2,11]),
        pos=Position(pos=ShapedArray(int8[2,11,2])),
        power=ShapedArray(int32[2,11]),
        cargo=UnitCargo(stock=ShapedArray(int32[2,11,4]))
    ),
    ShapedArray(int8[22,2]), factory_id2idx
    ShapedArray(int8[2]), n_factories
    Team(
        team_id=ShapedArray(int8[2]),
        faction=ShapedArray(int8[2]),
        init_water=ShapedArray(int32[2]),
        init_metal=ShapedArray(int32[2]),
        factories_to_place=ShapedArray(int32[2]),
        factory_strains=ShapedArray(int8[2,11]),
        n_factory=ShapedArray(int8[2]),
        bid=ShapedArray(int32[2])
    ),
    ShapedArray(int16[]), global_id
    ShapedArray(int8[]), place_first
)
```
