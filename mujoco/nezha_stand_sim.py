#!/usr/bin/env python3
"""Run an exported Nezha DreamWaQ standing policy in MuJoCo."""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deploy.nezha_stand import NezhaStandPolicyRuntime


def _resolve(path):
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _require_file(path, label):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _apply_payload_mass(mujoco, model, data, cfg, override_mass):
    """Use MJCF dynamics by default; optionally override for a sensitivity test."""
    body_name = cfg.get("payload_body_name", "top_box")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo model is missing payload body: {body_name}")
    if override_mass is None:
        urdf_mass = float(model.body_mass[body_id])
        print(f"Payload: {urdf_mass:g} kg on {body_name} (from URDF-derived MJCF)")
        return urdf_mass

    requested_mass = float(override_mass)
    if requested_mass < 0.0:
        raise ValueError("--payload_mass must be non-negative")
    effective_mass = max(requested_mass, 1.0e-9)
    inertia_per_kg = np.asarray(cfg["payload_inertia_per_kg"], dtype=np.float64)
    payload_com = np.asarray(cfg["payload_com"], dtype=np.float64)
    if inertia_per_kg.shape != (3,) or np.any(inertia_per_kg <= 0.0):
        raise ValueError("payload_inertia_per_kg must contain three positive values")
    if payload_com.shape != (3,):
        raise ValueError("payload_com must contain three values")
    model.body_mass[body_id] = effective_mass
    model.body_inertia[body_id] = inertia_per_kg * effective_mass
    model.body_ipos[body_id] = payload_com
    model.body_iquat[body_id] = np.asarray([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_setConst(model, data)
    print(
        f"Payload: {requested_mass:g} kg on {body_name} "
        f"(temporary MuJoCo override; simulator mass {effective_mass:g} kg)"
    )
    return requested_mass


def _joint_addresses(mujoco, model, names):
    qpos_addresses = []
    dof_addresses = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint: {name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_addresses), np.asarray(dof_addresses)


def _free_joint(mujoco, model):
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return joint_id
    raise ValueError("The Nezha MuJoCo model must contain a floating base")


def _sensor_slice(mujoco, model, name, expected_dimension):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"MuJoCo model is missing sensor: {name}")
    dimension = int(model.sensor_dim[sensor_id])
    if dimension != expected_dimension:
        raise ValueError(
            f"Sensor {name} has dimension {dimension}; expected {expected_dimension}"
        )
    address = int(model.sensor_adr[sensor_id])
    return slice(address, address + dimension)


def _contact_geometry_map(mujoco, model, wheel_body_names, ground_geom_names):
    """Map colliding wheel geometries to FL/FR/RL/RR indices."""
    wheel_body_ids = []
    for name in wheel_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model is missing wheel body: {name}")
        wheel_body_ids.append(body_id)
    body_to_wheel = {body_id: index for index, body_id in enumerate(wheel_body_ids)}
    geom_to_wheel = {
        geom_id: body_to_wheel[int(model.geom_bodyid[geom_id])]
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_to_wheel
        and int(model.geom_contype[geom_id]) != 0
    }
    if set(geom_to_wheel.values()) != set(range(4)):
        raise ValueError("Each wheel body must contain a colliding geometry")

    ground_geom_ids = set()
    for name in ground_geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"MuJoCo model is missing ground geometry: {name}")
        ground_geom_ids.add(geom_id)
    return geom_to_wheel, ground_geom_ids


def _wheel_vertical_forces(mujoco, model, data, geom_to_wheel, ground_geom_ids):
    """Sum wheel-ground normal forces for the current MuJoCo substep."""
    forces = np.zeros(4, dtype=np.float64)
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        wheel_index = None
        if geom1 in geom_to_wheel and geom2 in ground_geom_ids:
            wheel_index = geom_to_wheel[geom1]
        elif geom2 in geom_to_wheel and geom1 in ground_geom_ids:
            wheel_index = geom_to_wheel[geom2]
        if wheel_index is None:
            continue
        contact_force.fill(0.0)
        mujoco.mj_contactForce(model, data, contact_id, contact_force)
        # On a horizontal plane the contact normal is the vertical support
        # direction.  Absolute value makes the result independent of geom order.
        forces[wheel_index] += abs(float(contact_force[0]))
    return forces


def _contact_metrics(vertical_forces, threshold):
    forces = np.asarray(vertical_forces, dtype=np.float64)
    contacts = forces > float(threshold)
    total = float(forces.sum())
    fractions = forces / total if total > 1e-9 else np.zeros(4, dtype=np.float64)
    fraction_error_rms = float(np.sqrt(np.mean(np.square(fractions - 0.25))))
    return contacts, bool(np.all(contacts)), fractions, fraction_error_rms


def _reset(
    mujoco,
    model,
    data,
    runtime,
    root_qpos_address,
    qpos_addresses,
    initial_base_position,
    default_dof_pos,
):
    mujoco.mj_resetData(model, data)
    data.qpos[root_qpos_address : root_qpos_address + 7] = [
        initial_base_position[0],
        initial_base_position[1],
        initial_base_position[2],
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    data.qpos[qpos_addresses] = np.asarray(default_dof_pos, dtype=np.float64)
    data.qvel[:] = 0.0
    data.qfrc_applied[:] = 0.0
    runtime.reset()
    mujoco.mj_forward(model, data)


def _settle(
    mujoco,
    model,
    data,
    runtime,
    qpos_addresses,
    dof_addresses,
    seconds,
):
    """Let the nominal zero-action PD pose reach floor contact."""
    zero_actions = np.zeros(12, dtype=np.float32)
    for _ in range(max(0, int(round(seconds / model.opt.timestep)))):
        torques = runtime.compute_torques(
            zero_actions, data.qpos[qpos_addresses], data.qvel[dof_addresses]
        )
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof_addresses] = torques.cpu().numpy()
        mujoco.mj_step(model, data)
    runtime.reset()


def _attitude(projected_gravity):
    gravity = np.asarray(projected_gravity)
    roll = math.atan2(-gravity[1], -gravity[2])
    pitch = math.asin(float(np.clip(gravity[0], -1.0, 1.0)))
    return math.degrees(roll), math.degrees(pitch)


def _torque_pair_rms(torques, leg_indices):
    leg_torques = np.abs(np.asarray(torques)[leg_indices]).reshape(4, 3)
    differences = np.stack(
        [
            leg_torques[0] - leg_torques[1],
            leg_torques[0] - leg_torques[2],
            leg_torques[0] - leg_torques[3],
            leg_torques[1] - leg_torques[2],
            leg_torques[1] - leg_torques[3],
            leg_torques[2] - leg_torques[3],
        ]
    )
    return float(np.sqrt(np.mean(np.square(differences))))


def _normalized_torque_balance(torques, torque_limits, leg_indices):
    leg_torques = np.abs(np.asarray(torques)[leg_indices]).reshape(4, 3)
    limits = np.asarray(torque_limits)[leg_indices].reshape(4, 3)
    normalized = leg_torques / np.maximum(limits, 1e-9)
    differences = np.stack(
        [normalized[i] - normalized[j] for i in range(4) for j in range(i + 1, 4)]
    )
    # Same reduction as NezhaStandEnv._reward_torque_balance.
    return float(np.mean(np.sum(np.square(differences), axis=1)))


CSV_FIELDS = [
    "time_s", "payload_mass_kg", "base_x_m", "base_y_m", "base_z_m", "roll_deg", "pitch_deg",
    "drift_m", "wheel_rms_rad_s", "torque_pair_rms_nm",
    "normalized_torque_balance", "all_four_contact", "contact_count",
    "FL_contact", "FR_contact", "RL_contact", "RR_contact",
    "FL_vertical_force_n", "FR_vertical_force_n", "RL_vertical_force_n",
    "RR_vertical_force_n", "total_vertical_force_n", "FL_load_fraction",
    "FR_load_fraction", "RL_load_fraction", "RR_load_fraction",
    "foot_load_fraction_error_rms",
]


def _csv_row(simulated_time, payload_mass_kg, base_position, roll, pitch, drift, wheel_rms,
             torque_rms, normalized_torque_balance, contacts, all_contact,
             forces, fractions, load_error):
    names = ("FL", "FR", "RL", "RR")
    row = {
        "time_s": simulated_time,
        "payload_mass_kg": payload_mass_kg,
        "base_x_m": base_position[0],
        "base_y_m": base_position[1],
        "base_z_m": base_position[2],
        "roll_deg": roll,
        "pitch_deg": pitch,
        "drift_m": drift,
        "wheel_rms_rad_s": wheel_rms,
        "torque_pair_rms_nm": torque_rms,
        "normalized_torque_balance": normalized_torque_balance,
        "all_four_contact": int(all_contact),
        "contact_count": int(np.count_nonzero(contacts)),
        "total_vertical_force_n": float(np.sum(forces)),
        "foot_load_fraction_error_rms": load_error,
    }
    for index, name in enumerate(names):
        row[f"{name}_contact"] = int(contacts[index])
        row[f"{name}_vertical_force_n"] = forces[index]
        row[f"{name}_load_fraction"] = fractions[index]
    return row


def run(args):
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError(
            "Activate the Python 3.11 .venv-mujoco environment and run: "
            "uv pip install -r mujoco/requirements.txt"
        ) from exc
    if not hasattr(mujoco, "MjModel"):
        raise RuntimeError(
            "The official mujoco package was shadowed by the local mujoco directory"
        )

    config_path = os.path.abspath(args.config)
    cfg = _load_config(config_path)
    model_path = _resolve(args.model or cfg["model_path"])
    policy_path = _resolve(args.policy or cfg["policy_path"])
    _require_file(model_path, "MuJoCo model")
    if not policy_path or not os.path.isfile(policy_path):
        raise FileNotFoundError(
            f"Exported policy does not exist: {policy_path}\n"
            "Export it first with: python scripts/export_policy.py --checkpoint "
            "/path/to/model_20000.pt"
        )

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    payload_mass_kg = _apply_payload_mass(
        mujoco, model, data, cfg, args.payload_mass
    )
    free_joint_id = _free_joint(mujoco, model)
    root_qpos_address = int(model.jnt_qposadr[free_joint_id])
    root_body_id = int(model.jnt_bodyid[free_joint_id])
    qpos_addresses, dof_addresses = _joint_addresses(
        mujoco, model, cfg["joint_names"]
    )
    geom_to_wheel, ground_geom_ids = _contact_geometry_map(
        mujoco,
        model,
        cfg["wheel_body_names"],
        cfg["ground_geom_names"],
    )
    gyro_slice = _sensor_slice(mujoco, model, "base_gyro", 3)

    policy_dt = 1.0 / float(cfg["control_frequency_hz"])
    ratio = policy_dt / float(model.opt.timestep)
    sim_steps_per_policy = int(round(ratio))
    if sim_steps_per_policy < 1 or not np.isclose(ratio, sim_steps_per_policy):
        raise ValueError(
            "control_frequency_hz must produce an integer number of MuJoCo steps"
        )

    runtime = NezhaStandPolicyRuntime(
        policy_path=policy_path,
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
        device=args.device,
    )

    initial_position = np.asarray(cfg["initial_base_position"], dtype=np.float64)
    _reset(
        mujoco,
        model,
        data,
        runtime,
        root_qpos_address,
        qpos_addresses,
        initial_position,
        cfg["default_dof_pos"],
    )
    if not args.no_settle:
        _settle(
            mujoco,
            model,
            data,
            runtime,
            qpos_addresses,
            dof_addresses,
            float(cfg.get("settle_time_s", 0.0)),
        )
    start_xy = data.qpos[root_qpos_address : root_qpos_address + 2].copy()

    reset_requested = [False]

    def on_key(keycode):
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == "R":
            reset_requested[0] = True

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
        viewer.cam.lookat[:] = data.xpos[root_body_id]
        print("MuJoCo viewer: press R to reset the robot; close the window to stop.")
    elif args.duration <= 0:
        raise ValueError("Headless simulation requires --duration > 0")

    simulated_time = 0.0
    policy_step = 0
    last_torques = np.zeros(16)
    configured_log_interval = cfg.get("log_interval", 100)
    log_interval = max(0, int(
        args.log_interval
        if args.log_interval is not None
        else configured_log_interval
    ))
    contact_threshold = float(cfg.get("contact_force_threshold_n", 5.0))
    csv_handle = None
    csv_writer = None
    if args.csv:
        csv_path = _resolve(args.csv)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        csv_handle = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        print(f"CSV logging: {csv_path}")
    try:
        while viewer is None or viewer.is_running():
            if reset_requested[0]:
                _reset(
                    mujoco,
                    model,
                    data,
                    runtime,
                    root_qpos_address,
                    qpos_addresses,
                    initial_position,
                    cfg["default_dof_pos"],
                )
                start_xy = initial_position[:2].copy()
                reset_requested[0] = False
                print("Robot reset.")

            rotation = data.xmat[root_body_id].reshape(3, 3)
            projected_gravity = rotation.T @ np.asarray([0.0, 0.0, -1.0])
            angular_velocity = data.sensordata[gyro_slice].copy()
            dof_pos = data.qpos[qpos_addresses].copy()
            dof_vel = data.qvel[dof_addresses].copy()
            actions, _ = runtime.step(
                angular_velocity, projected_gravity, dof_pos, dof_vel
            )

            wall_start = time.time()
            vertical_force_sum = np.zeros(4, dtype=np.float64)
            for _ in range(sim_steps_per_policy):
                torques = runtime.compute_torques(
                    actions,
                    data.qpos[qpos_addresses],
                    data.qvel[dof_addresses],
                )
                last_torques = torques.cpu().numpy()
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[dof_addresses] = last_torques
                mujoco.mj_step(model, data)
                vertical_force_sum += _wheel_vertical_forces(
                    mujoco, model, data, geom_to_wheel, ground_geom_ids
                )

            simulated_time += sim_steps_per_policy * model.opt.timestep
            policy_step += 1
            vertical_forces = vertical_force_sum / sim_steps_per_policy
            contacts, all_contact, load_fractions, load_error = _contact_metrics(
                vertical_forces, contact_threshold
            )
            log_rotation = data.xmat[root_body_id].reshape(3, 3)
            log_gravity = log_rotation.T @ np.asarray([0.0, 0.0, -1.0])
            roll, pitch = _attitude(log_gravity)
            base_position = data.qpos[
                root_qpos_address : root_qpos_address + 3
            ].copy()
            drift = float(np.linalg.norm(base_position[:2] - start_xy))
            wheel_rms = float(np.sqrt(
                np.mean(np.square(data.qvel[dof_addresses][cfg["wheel_joint_indices"]]))
            ))
            torque_rms = _torque_pair_rms(last_torques, cfg["leg_joint_indices"])
            normalized_torque_balance = _normalized_torque_balance(
                last_torques, cfg["torque_limits"], cfg["leg_joint_indices"]
            )

            if csv_writer is not None:
                csv_writer.writerow(_csv_row(
                    simulated_time, payload_mass_kg, base_position, roll, pitch, drift, wheel_rms,
                    torque_rms, normalized_torque_balance, contacts, all_contact,
                    vertical_forces, load_fractions, load_error,
                ))

            if log_interval and policy_step % log_interval == 0:
                contact_bits = "".join("1" if value else "0" for value in contacts)
                print(
                    f"step={policy_step:6d} t={simulated_time:7.2f}s "
                    f"z={base_position[2]:.3f}m roll={roll:+.2f}deg "
                    f"pitch={pitch:+.2f}deg drift={drift:.3f}m "
                    f"wheel_rms={wheel_rms:.3f}rad/s "
                    f"torque_pair_rms={torque_rms:.2f}Nm\n"
                    f"  contact[FL FR RL RR]={contact_bits} "
                    f"all={'YES' if all_contact else 'NO'} "
                    f"Fz={np.array2string(vertical_forces, precision=1)}N "
                    f"load={np.array2string(load_fractions, precision=3)} "
                    f"load_error_rms={load_error:.4f}"
                )
                if csv_handle is not None:
                    csv_handle.flush()

            if viewer is not None:
                viewer.cam.lookat[:] = data.xpos[root_body_id]
                viewer.sync()
                remaining = policy_dt - (time.time() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)
            if args.duration > 0 and simulated_time + 1e-12 >= args.duration:
                break
    finally:
        if csv_handle is not None:
            csv_handle.close()
        if viewer is not None:
            viewer.close()

    final_position = data.qpos[root_qpos_address : root_qpos_address + 3]
    print(
        f"Simulation finished: t={simulated_time:.2f}s, "
        f"policy_steps={policy_step}, "
        f"base_xyz={np.array2string(final_position, precision=3)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "mujoco", "nezha_stand_config.yaml"),
    )
    parser.add_argument("--model", help="Override MuJoCo XML model path")
    parser.add_argument("--policy", help="Override exported policy.pt path")
    parser.add_argument(
        "--payload_mass", type=float,
        help="Optional MuJoCo-only override; default uses MJCF/URDF mass",
    )
    parser.add_argument("--device", default="cpu", help="Torch inference device")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--log_interval",
        type=int,
        help="Print live metrics every N policy steps; 0 disables terminal metrics",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulated seconds; <=0 runs until the viewer closes",
    )
    parser.add_argument(
        "--no-settle", action="store_true", help="Skip the nominal PD settling phase"
    )
    parser.add_argument(
        "--csv",
        help="Write one contact/force/torque row per 50 Hz policy step (relative paths use the repository root)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
