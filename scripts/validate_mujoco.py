#!/usr/bin/env python3
"""Validate the MuJoCo model, joint contract and optional exported policy."""

import argparse
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_URDF = PROJECT_ROOT / "assets" / "nezha" / "urdf" / "nezha.urdf"
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
    if model.nbody != 25 or model.nu != 16:
        raise RuntimeError(
            f"Robot topology mismatch: nbody={model.nbody}, nu={model.nu}"
        )
    model_mass = float(model.body_mass.sum())
    urdf_root = ET.parse(ASSET_URDF).getroot()
    expected_mass = sum(
        float(mass.get("value"))
        for mass in urdf_root.findall(".//inertial/mass")
    )
    if not math.isclose(model_mass, expected_mass, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"Robot mass mismatch: model={model_mass}, config={expected_mass}"
        )

    for index, name in enumerate(cfg["joint_names"]):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actual = [float(value) for value in model.jnt_range[joint_id]]
        expected = [cfg["dof_lower_limits"][index], cfg["dof_upper_limits"][index]]
        if not all(
            math.isclose(a, e, rel_tol=0.0, abs_tol=1e-9)
            for a, e in zip(actual, expected)
        ):
            raise RuntimeError(
                f"Joint-limit mismatch for {name}: model={actual}, config={expected}"
            )
        default = float(cfg["default_dof_pos"][index])
        if not expected[0] <= default <= expected[1]:
            raise RuntimeError(
                f"Default position {default} is outside {name} limits {expected}"
            )

    expected_collision_sizes = {
        "trunk_collision_0": (0.5605, 0.195, 0.135),
        **{
            f"{leg}_calf_collision_0": (0.025, 0.0225, 0.195)
            for leg in ("FL", "FR", "RL", "RR")
        },
        **{
            f"{leg}_foot_collision_0": (0.105, 0.02)
            for leg in ("FL", "FR", "RL", "RR")
        },
    }
    for geom_name, expected_size in expected_collision_sizes.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise RuntimeError(f"Missing collision geometry: {geom_name}")
        actual_size = model.geom_size[geom_id][:len(expected_size)]
        if any(
            not math.isclose(float(a), e, rel_tol=0.0, abs_tol=1e-9)
            for a, e in zip(actual_size, expected_size)
        ):
            raise RuntimeError(
                f"Collision-size mismatch for {geom_name}: {actual_size}"
            )

    expected_p_gains = [150.0, 150.0, 300.0, 0.0] * 4
    if cfg["p_gains"] != expected_p_gains:
        raise RuntimeError(
            "Standing Kp contract mismatch; expected [150, 150, 300, 0] per leg"
        )
    expected_default = [
        -0.10, 0.925, -1.85, 0.0,
        0.10, 0.925, -1.85, 0.0,
        -0.10, 0.925, -1.85, 0.0,
        0.10, 0.925, -1.85, 0.0,
    ]
    if cfg["default_dof_pos"] != expected_default:
        raise RuntimeError(
            "Standing-pose contract mismatch; expected mirrored LZHMine hip angles"
        )
    if float(cfg["action_scale"]) != 0.15:
        raise RuntimeError("Flat-standing action_scale contract must remain 0.15")
    if float(cfg["contact_force_threshold_n"]) <= 0.0:
        raise RuntimeError("contact_force_threshold_n must be positive")
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
        metadata_path = policy_path.with_name("policy_metadata.json")
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing policy metadata: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        current_asset_hash = hashlib.sha256(ASSET_URDF.read_bytes()).hexdigest()
        if metadata.get("asset_urdf_sha256") != current_asset_hash:
            raise RuntimeError(
                "Exported policy was not trained with the current Nezha URDF"
            )
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
    print(f"robot:  24 bodies, mass={model_mass:.6f} kg")
    print(f"state:  nq={model.nq}, nv={model.nv}, joints=16 + floating base")
    print("contract: current_obs=46, history=5x46=230, action=12, control=50 Hz")
    print("control: Kp per leg=[150, 150, 300, 0]")
    print(
        "contact: wheels=FL/FR/RL/RR, threshold="
        f"{float(cfg['contact_force_threshold_n']):g} N"
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
