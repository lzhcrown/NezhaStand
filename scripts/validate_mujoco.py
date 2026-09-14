#!/usr/bin/env python3
"""Validate the MuJoCo model, joint contract and optional exported policy."""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.nezha_stand import NezhaStandPolicyRuntime


def _resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate(args):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "MuJoCo sim-to-sim uses its own Python 3.11 environment; "
            f"current Python is {sys.version.split()[0]}"
        )
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "Activate .venv-mujoco and install mujoco/requirements.txt"
        ) from exc
    if mujoco.__version__ != "3.12.0":
        raise RuntimeError(
            f"Expected MuJoCo 3.12.0 (LZHMine-aligned), got {mujoco.__version__}"
        )

    with open(args.config, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    model_path = _resolve(cfg["model_path"])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    names = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name in cfg["joint_names"]:
            names.append(name)
    if names != cfg["joint_names"]:
        raise RuntimeError(
            f"Joint order mismatch:\nmodel={names}\nconfig={cfg['joint_names']}"
        )
    if not any(
        model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        for joint_id in range(model.njnt)
    ):
        raise RuntimeError("MuJoCo model has no free joint")

    expected_p_gains = [150.0, 220.0, 300.0, 0.0] * 4
    if cfg["p_gains"] != expected_p_gains:
        raise RuntimeError(
            "90 kg standing Kp mismatch; expected [150, 220, 300, 0] per leg"
        )
    if float(cfg["contact_force_threshold_n"]) <= 0.0:
        raise RuntimeError("contact_force_threshold_n must be positive")
    payload_body_name = cfg.get("payload_body_name", "top_box")
    payload_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, payload_body_name
    )
    if payload_body_id < 0:
        raise RuntimeError(f"Missing payload body: {payload_body_name}")
    payload_mass_kg = float(model.body_mass[payload_body_id])
    if payload_mass_kg < 0.0:
        raise RuntimeError("MJCF payload mass must be non-negative")
    if len(cfg.get("payload_inertia_per_kg", [])) != 3:
        raise RuntimeError("payload_inertia_per_kg must contain three values")
    for body_name in cfg["wheel_body_names"]:
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            raise RuntimeError(f"Missing wheel body for contact metrics: {body_name}")
        collision_geoms = [
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and int(model.geom_contype[geom_id]) != 0
        ]
        if not collision_geoms:
            raise RuntimeError(f"Wheel body has no collision geom: {body_name}")
    for geom_name in cfg["ground_geom_names"]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) < 0:
            raise RuntimeError(f"Missing ground geometry: {geom_name}")

    policy_path = _resolve(args.policy or cfg["policy_path"])
    if policy_path.is_file():
        runtime = NezhaStandPolicyRuntime(
            policy_path=str(policy_path),
            default_dof_pos=cfg["default_dof_pos"],
            p_gains=cfg["p_gains"],
            d_gains=cfg["d_gains"],
            torque_limits=cfg["torque_limits"],
            dof_lower_limits=cfg["dof_lower_limits"],
            dof_upper_limits=cfg["dof_upper_limits"],
            leg_joint_indices=cfg["leg_joint_indices"],
            wheel_joint_indices=cfg["wheel_joint_indices"],
            action_scale=cfg["action_scale"],
            clip_actions=cfg["clip_actions"],
            observation_scales=cfg["observation_scales"],
            history_length=cfg.get("history_length", 5),
        )
        observation = runtime.build_observation(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            cfg["default_dof_pos"],
            [0.0] * 16,
        )
        with torch.inference_mode():
            action = runtime.policy(
                observation.unsqueeze(0), runtime.observation_history.unsqueeze(0)
            )
        if tuple(action.shape) != (1, 12):
            raise RuntimeError(f"Unexpected policy output shape: {tuple(action.shape)}")
        print(f"policy: OK ({policy_path})")
    else:
        print(f"policy: SKIP (not found: {policy_path})")

    print(f"Python: {sys.version.split()[0]}")
    print(f"MuJoCo: {mujoco.__version__}")
    print(f"model:  OK ({model_path})")
    print(f"state:  nq={model.nq}, nv={model.nv}, joints=16 + floating base")
    print("contract: current_obs=46, history=5x46=230, action=12, control=50 Hz")
    print("control: Kp per leg=[150, 220, 300, 0]")
    print(
        "contact: wheels=FL/FR/RL/RR, threshold="
        f"{float(cfg['contact_force_threshold_n']):g} N"
    )
    print(
        f"payload: body={payload_body_name}, URDF/MJCF={payload_mass_kg:g} kg"
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "mujoco" / "nezha_stand_config.yaml"),
    )
    parser.add_argument("--policy", help="Optional policy.pt override")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(validate(parse_args()))
