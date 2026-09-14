#!/usr/bin/env python3
"""Validate the synchronized nezha_description assets without Isaac Gym."""

import argparse
import copy
import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_PACKAGE = PROJECT_ROOT / "assets" / "nezha_description"
URDF = ASSET_PACKAGE / "urdf" / "nezha_description.urdf"
EXPECTED_DOF_NAMES = {
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf", "foot")
}
ROS_MESH_PREFIX = b"package://nezha_description/meshes/"
LOCAL_MESH_PREFIX = b"../meshes/"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_urdf_bytes(root):
    """Ignore XML formatting and mesh prefixes, but no model content."""
    root = copy.deepcopy(root)
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", Path(mesh.get("filename", "")).name)
    for element in root.iter():
        if not (element.text or "").strip():
            element.text = None
        if not (element.tail or "").strip():
            element.tail = None
    return ET.tostring(root, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional original desktop nezha directory for copy verification.",
    )
    args = parser.parse_args()

    errors = []
    root = ET.parse(URDF).getroot()
    links = root.findall("link")
    joints = root.findall("joint")
    actuated = {j.get("name") for j in joints if j.get("type") != "fixed"}

    if len(links) != 25:
        errors.append(f"expected 25 links, found {len(links)}")
    if len(joints) != 24:
        errors.append(f"expected 24 joints, found {len(joints)}")
    if actuated != EXPECTED_DOF_NAMES:
        errors.append(
            "actuated joint mismatch: "
            f"missing={sorted(EXPECTED_DOF_NAMES - actuated)}, "
            f"extra={sorted(actuated - EXPECTED_DOF_NAMES)}"
        )
    child_links = {
        joint.find("child").get("link") for joint in joints
        if joint.find("child") is not None
    }
    roots = {link.get("name") for link in links} - child_links
    if roots != {"base"}:
        errors.append(f"expected base as sole root, found {sorted(roots)}")
    payload_mass = float("nan")
    payload = root.find("link[@name='top_box']")
    if payload is None:
        errors.append("missing top_box payload link")
    else:
        payload_mass = float(payload.find("./inertial/mass").get("value"))
        if payload_mass < 0.0:
            errors.append(f"top_box mass must be non-negative, found {payload_mass}")
    if any("package://" in mesh.get("filename", "") for mesh in root.findall(".//mesh")):
        errors.append("package:// mesh URL remains in standalone URDF")

    missing_meshes = []
    for mesh in root.findall(".//mesh"):
        mesh_path = (URDF.parent / mesh.get("filename")).resolve()
        if not mesh_path.is_file():
            missing_meshes.append(str(mesh_path))
    if missing_meshes:
        errors.append(f"missing mesh references: {missing_meshes}")

    if args.source:
        source_urdf = (
            args.source.expanduser().resolve()
            / "urdf" / "nezha_description.urdf"
        )
        if not source_urdf.is_file():
            errors.append(f"source URDF not found: {source_urdf}")
        else:
            source_root = ET.parse(source_urdf).getroot()
            source_payload = source_root.find("link[@name='top_box']")
            source_mass = float(source_payload.find("./inertial/mass").get("value"))
            if source_mass < 0.0:
                errors.append(f"source top_box mass is negative: {source_mass}")
            if normalized_urdf_bytes(root) != normalized_urdf_bytes(source_root):
                errors.append(
                    "target URDF differs from source beyond mesh path rewrites"
                )
            expected_bytes = source_urdf.read_bytes().replace(
                ROS_MESH_PREFIX, LOCAL_MESH_PREFIX
            )
            if URDF.read_bytes() != expected_bytes:
                errors.append(
                    "target URDF bytes differ from source beyond mesh path rewrites"
                )

        source_mesh_dir = args.source.expanduser().resolve() / "meshes"
        source_meshes = {p.name: sha256(p) for p in source_mesh_dir.glob("*.STL")}
        copied_mesh_dir = ASSET_PACKAGE / "meshes"
        copied_meshes = {p.name: sha256(p) for p in copied_mesh_dir.glob("*.STL")}
        differing = sorted(
            name for name, digest in source_meshes.items()
            if copied_meshes.get(name) != digest
        )
        if differing:
            errors.append(f"source mesh copies differ or are missing: {differing}")
        extra = sorted(set(copied_meshes) - set(source_meshes))
        if extra:
            errors.append(f"extra target meshes not present in source: {extra}")

    print(f"URDF: {URDF}")
    print(f"robot name: {root.get('name')}")
    print(f"links/joints/actuated: {len(links)}/{len(joints)}/{len(actuated)}")
    print(f"mesh references: {len(root.findall('.//mesh'))}")
    print(f"top_box payload: {payload_mass:g} kg")
    print(f"transmission/gazebo: {len(root.findall('transmission'))}/{len(root.findall('gazebo'))}")
    print(f"bundled STL files: {len(list((ASSET_PACKAGE / 'meshes').glob('*.STL')))}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
