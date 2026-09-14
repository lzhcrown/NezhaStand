#!/usr/bin/env python3
"""Validate the imported nezha_description assets without Isaac Gym."""

import argparse
import copy
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
EXPECTED_FIXED_LINKS = {"top_box", "imu_fixed_link", "Lidar"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_element(element):
    """Normalize formatting and package-relative mesh URLs for comparison."""
    element = copy.deepcopy(element)
    for node in element.iter():
        if not (node.text or "").strip():
            node.text = None
        if not (node.tail or "").strip():
            node.tail = None
        if node.tag == "mesh" and node.get("filename"):
            node.set("filename", Path(node.get("filename")).name)
    return ET.tostring(element, encoding="unicode")


def source_urdf_path(source_root):
    for name in ("nezha_description.urdf", "nezha.urdf"):
        candidate = source_root / "urdf" / name
        if candidate.is_file():
            return candidate
    return source_root / "urdf" / "nezha_description.urdf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional source nezha_description package for semantic verification.",
    )
    args = parser.parse_args()

    errors = []
    root = ET.parse(URDF).getroot()
    links = root.findall("link")
    joints = root.findall("joint")
    actuated = {j.get("name") for j in joints if j.get("type") != "fixed"}

    if len(links) != 24:
        errors.append(f"expected 24 links, found {len(links)}")
    if len(joints) != 23:
        errors.append(f"expected 23 joints, found {len(joints)}")
    if actuated != EXPECTED_DOF_NAMES:
        errors.append(
            "actuated joint mismatch: "
            f"missing={sorted(EXPECTED_DOF_NAMES - actuated)}, "
            f"extra={sorted(actuated - EXPECTED_DOF_NAMES)}"
        )
    link_names = {link.get("name") for link in links}
    missing_fixed_links = EXPECTED_FIXED_LINKS - link_names
    if missing_fixed_links:
        errors.append(f"missing fixed accessory links: {sorted(missing_fixed_links)}")
    if "base" in link_names:
        errors.append("massless ROS base wrapper must not replace trunk as root")
    if root.findall("gazebo") or root.findall("transmission"):
        errors.append("Gazebo-only metadata was not removed from training URDF")

    missing_meshes = []
    for mesh in root.findall(".//mesh"):
        mesh_path = (URDF.parent / mesh.get("filename")).resolve()
        if not mesh_path.is_file():
            missing_meshes.append(str(mesh_path))
    if missing_meshes:
        errors.append(f"missing mesh references: {missing_meshes}")

    if args.source:
        source_root = args.source.expanduser().resolve()
        source_urdf = source_urdf_path(source_root)
        if not source_urdf.is_file():
            errors.append(f"source URDF not found: {source_urdf}")
        else:
            source_root_xml = ET.parse(source_urdf).getroot()
            source_links = {
                item.get("name"): normalized_element(item)
                for item in source_root_xml.findall("link")
                if item.get("name") != "base"
            }
            copied_links = {
                item.get("name"): normalized_element(item)
                for item in root.findall("link")
            }
            source_joints = {
                item.get("name"): normalized_element(item)
                for item in source_root_xml.findall("joint")
                if item.get("name") != "floating_base"
            }
            copied_joints = {
                item.get("name"): normalized_element(item)
                for item in root.findall("joint")
            }
            if source_links != copied_links:
                errors.append("imported physical links differ from source URDF")
            if source_joints != copied_joints:
                errors.append("imported physical joints differ from source URDF")

        source_mesh_dir = source_root / "meshes"
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
