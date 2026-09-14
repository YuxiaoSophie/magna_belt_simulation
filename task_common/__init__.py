"""Task-agnostic scaffolding shared by the belt tasks.

    directives.py   shared extension directives (COMMON_DIRECTIVES) and the params helper
    scene.py        make_builder, SceneInfo and build_task_scene
    joint_state.py  seeding the finalized Model with default joint state and gains
    simulation.py   BeltTaskSimulation: solver, stepping, CUDA graph, diagnostics
"""
