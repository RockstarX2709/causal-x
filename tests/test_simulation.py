from simulation.manufacturing_world import ManufacturingWorld


def test_manufacturing_world_initializes():
    world = ManufacturingWorld()

    assert world is not None


def test_simulation_module_imports():
    import simulation.simulator

    assert simulation.simulator is not None
