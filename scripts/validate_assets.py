#!/usr/bin/env python3
"""Validate the copied desktop URDF without importing Isaac Gym."""

import argparse
import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF = PROJECT_ROOT / "assets" / "nezha" / "urdf" / "nezha.urdf"
EXPECTED_DOF_NAMES = {
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf", "foot")
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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

    if len(links) != 22:
        errors.append(f"expected 22 links, found {len(links)}")
    if len(joints) != 21:
        errors.append(f"expected 21 joints, found {len(joints)}")
    if actuated != EXPECTED_DOF_NAMES:
        errors.append(
            "actuated joint mismatch: "
            f"missing={sorted(EXPECTED_DOF_NAMES - actuated)}, "
            f"extra={sorted(actuated - EXPECTED_DOF_NAMES)}"
        )

    missing_meshes = []
    for mesh in root.findall(".//mesh"):
        mesh_path = (URDF.parent / mesh.get("filename")).resolve()
        if not mesh_path.is_file():
            missing_meshes.append(str(mesh_path))
    if missing_meshes:
        errors.append(f"missing mesh references: {missing_meshes}")

    if args.source:
        source_urdf = args.source.expanduser().resolve() / "urdf" / "nezha.urdf"
        if not source_urdf.is_file():
            errors.append(f"source URDF not found: {source_urdf}")
        elif sha256(source_urdf) != sha256(URDF):
            errors.append("copied URDF differs from source")

        source_mesh_dir = args.source.expanduser().resolve() / "meshes"
        source_meshes = {p.name: sha256(p) for p in source_mesh_dir.glob("*.STL")}
        copied_mesh_dir = PROJECT_ROOT / "assets" / "nezha" / "meshes"
        copied_meshes = {p.name: sha256(p) for p in copied_mesh_dir.glob("*.STL")}
        if source_meshes != copied_meshes:
            errors.append("copied mesh set differs from source")

    print(f"URDF: {URDF}")
    print(f"robot name: {root.get('name')}")
    print(f"links/joints/actuated: {len(links)}/{len(joints)}/{len(actuated)}")
    print(f"mesh references: {len(root.findall('.//mesh'))}")
    print(f"bundled STL files: {len(list((PROJECT_ROOT / 'assets/nezha/meshes').glob('*.STL')))}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

