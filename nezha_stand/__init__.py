"""Standalone Nezha standing task."""

TASK_NAME = "nezha_stand"


def register_task():
    """Register the standing task in the bundled task registry."""
    from legged_gym.utils.task_registry import task_registry

    if TASK_NAME in task_registry.task_classes:
        return

    from .config import NezhaStandCfg, NezhaStandCfgPPO
    from .env import NezhaStandEnv

    task_registry.register(
        TASK_NAME,
        NezhaStandEnv,
        NezhaStandCfg(),
        NezhaStandCfgPPO(),
    )


__all__ = ["TASK_NAME", "register_task"]
