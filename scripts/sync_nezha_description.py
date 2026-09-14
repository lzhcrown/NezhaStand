#!/usr/bin/env python3
"""Synchronize the canonical ROS Nezha description into this project.

The source URDF remains authoritative. The Isaac Gym copy preserves its raw
text byte-for-byte except for the ROS mesh path prefix. MJCF is generated as a
separate simulator derivative and never mutates the copied URDF.
"""

import argparse
import copy
import math
import os
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_PACKAGE = PROJECT_ROOT / "assets/nezha_description"
TARGET_URDF = TARGET_PACKAGE / "urdf/nezha_description.urdf"
TARGET_MESH_DIR = TARGET_PACKAGE / "meshes"
TARGET_MJCF = PROJECT_ROOT / "mujoco/models/nezha.xml"
EXPECTED_DOF_NAMES = {
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf", "foot")
}
ROS_MESH_PREFIX = b"package://nezha_description/meshes/"
LOCAL_MESH_PREFIX = b"../meshes/"


def _source_urdf(source_root):
    path = source_root / "urdf/nezha_description.urdf"
    if not path.is_file():
        raise FileNotFoundError(f"Canonical URDF not found: {path}")
    return path


def _indent(element, level=0):
    """Python 3.8-compatible ElementTree indentation."""
    prefix = "\n" + level * "  "
    child_prefix = "\n" + (level + 1) * "  "
    if len(element):
        if not (element.text or "").strip():
            element.text = child_prefix
        for child in element:
            _indent(child, level + 1)
        if not (element[-1].tail or "").strip():
            element[-1].tail = prefix
    if level and not (element.tail or "").strip():
        element.tail = prefix


def _write_tree(tree, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else 0o644
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            tree.write(stream, encoding="utf-8", xml_declaration=True)
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_bytes(content, destination):
    """Atomically write bytes so URDF comments and formatting survive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else 0o644
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _prepare_isaac_urdf(source_path):
    tree = ET.parse(source_path)
    root = tree.getroot()
    # The only permitted Isaac URDF change: resolve meshes without ROS.
    for mesh in root.findall(".//mesh"):
        filename = Path(mesh.get("filename", "")).name
        if not filename:
            raise ValueError("URDF mesh has no filename")
        mesh.set("filename", f"../meshes/{filename}")

    payload = root.find("link[@name='top_box']")
    if payload is None:
        raise ValueError("Source URDF has no top_box payload link")
    mass = payload.find("./inertial/mass")
    inertia = payload.find("./inertial/inertia")
    if mass is None or inertia is None:
        raise ValueError("top_box requires inertial/mass and inertial/inertia")
    if float(mass.get("value")) < 0.0:
        raise ValueError("top_box mass must be non-negative")

    links = {link.get("name") for link in root.findall("link")}
    joints = root.findall("joint")
    children = {joint.find("child").get("link") for joint in joints}
    roots = links - children
    dofs = {joint.get("name") for joint in joints if joint.get("type") != "fixed"}
    if roots != {"base"}:
        raise ValueError(f"Canonical URDF root must be base, got {sorted(roots)}")
    if dofs != EXPECTED_DOF_NAMES:
        raise ValueError(f"Unexpected movable joints: {sorted(dofs)}")
    _indent(root)
    return tree


def _prepare_mjcf_source(isaac_tree):
    """Create an in-memory conversion tree without editing the Isaac URDF."""
    tree = ET.ElementTree(copy.deepcopy(isaac_tree.getroot()))
    root = tree.getroot()
    for element in list(root):
        if element.tag in {"gazebo", "transmission"}:
            root.remove(element)
        elif element.tag == "link" and element.get("name") == "base":
            root.remove(element)
        elif element.tag == "joint" and element.get("name") == "floating_base":
            root.remove(element)
    return tree


def _numbers(value):
    return [float(number) for number in value.split()]


def _half_extents(value):
    return " ".join(f"{number / 2:g}" for number in _numbers(value))


def _quaternion(rpy):
    roll, pitch, yaw = _numbers(rpy)
    if roll == pitch == yaw == 0.0:
        return None
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    values = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    return " ".join(f"{value:.12g}" for value in values)


def _origin_attributes(origin):
    if origin is None:
        return {}
    attributes = {}
    xyz = origin.get("xyz", "0 0 0")
    if any(value != 0.0 for value in _numbers(xyz)):
        attributes["pos"] = xyz
    quat = _quaternion(origin.get("rpy", "0 0 0"))
    if quat:
        attributes["quat"] = quat
    return attributes


def _add_geometry(parent, geometry, attributes):
    shape = next(iter(geometry))
    attributes = dict(attributes)
    if shape.tag == "mesh":
        attributes.update(type="mesh", mesh=Path(shape.get("filename")).stem)
    elif shape.tag == "box":
        attributes.update(type="box", size=_half_extents(shape.get("size")))
    elif shape.tag == "cylinder":
        attributes.update(
            type="cylinder",
            size=f'{float(shape.get("radius")):g} {float(shape.get("length")) / 2:g}',
        )
    elif shape.tag == "sphere":
        attributes.update(type="sphere", size=f'{float(shape.get("radius")):g}')
    else:
        raise ValueError(f"Unsupported URDF geometry: {shape.tag}")
    ET.SubElement(parent, "geom", attributes)


def _generate_mjcf(urdf_tree):
    urdf = urdf_tree.getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    child_joints = {name: [] for name in links}
    child_links = set()
    for joint in urdf.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        child_joints[parent].append(joint)
        child_links.add(child)
    if set(links) - child_links != {"trunk"}:
        raise ValueError("MJCF generation requires trunk as the sole root")

    model = ET.Element("mujoco", {"model": "nezha"})
    default = ET.SubElement(model, "default")
    robot = ET.SubElement(default, "default", {"class": "robot"})
    motor = ET.SubElement(robot, "default", {"class": "motor"})
    ET.SubElement(motor, "joint")
    ET.SubElement(motor, "motor")
    visual = ET.SubElement(robot, "default", {"class": "visual"})
    ET.SubElement(visual, "geom", {
        "material": "default_material", "contype": "0", "conaffinity": "0",
        "density": "0", "group": "2",
    })
    collision = ET.SubElement(robot, "default", {"class": "collision"})
    ET.SubElement(collision, "geom", {
        "material": "collision_material", "condim": "3", "contype": "1",
        "conaffinity": "2", "solref": "0.005 1",
        "friction": "1.0 0.01 0.001", "density": "0", "group": "3",
    })
    ET.SubElement(model, "compiler", {
        "angle": "radian", "meshdir": "../../assets/nezha_description/meshes/",
        "autolimits": "true",
    })
    ET.SubElement(model, "size", {"njmax": "500", "nconmax": "100"})
    assets = ET.SubElement(model, "asset")
    ET.SubElement(assets, "material", {
        "name": "default_material", "rgba": "0.72 0.80 0.88 1",
        "specular": "0.35", "shininess": "0.45", "reflectance": "0.06",
    })
    ET.SubElement(assets, "material", {
        "name": "collision_material", "rgba": "0.0 0.4 0.8 0.2",
    })
    for mesh in urdf.findall(".//mesh"):
        filename = Path(mesh.get("filename")).name
        name = Path(filename).stem
        if assets.find(f"mesh[@name='{name}']") is None:
            ET.SubElement(assets, "mesh", {"name": name, "file": filename})

    worldbody = ET.SubElement(model, "worldbody")

    def add_link(link_name, parent, source_joint=None):
        attributes = {"name": link_name}
        if link_name == "trunk":
            attributes.update(childclass="robot", pos="0 0 0.54", quat="1 0 0 0")
        elif source_joint is not None:
            attributes.update(_origin_attributes(source_joint.find("origin")))
        body = ET.SubElement(parent, "body", attributes)
        if link_name == "trunk":
            ET.SubElement(body, "freejoint", {"name": "floating_base"})
        elif source_joint is not None and source_joint.get("type") != "fixed":
            limit = source_joint.find("limit")
            name = source_joint.get("name")
            ET.SubElement(body, "joint", {
                "name": name,
                "axis": source_joint.find("axis").get("xyz"),
                "range": f'{limit.get("lower")} {limit.get("upper")}',
                "actuatorfrcrange": f'-{limit.get("effort")} {limit.get("effort")}',
                "frictionloss": "0.025" if "hip_joint" in name else "0.05",
            })

        link = links[link_name]
        inertial = link.find("inertial")
        if inertial is not None and float(inertial.find("mass").get("value")) > 0.0:
            inertia = inertial.find("inertia")
            attributes = _origin_attributes(inertial.find("origin"))
            attributes.update(
                mass=inertial.find("mass").get("value"),
                fullinertia=" ".join(
                    inertia.get(name)
                    for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                ),
            )
            ET.SubElement(body, "inertial", attributes)
        if link_name == "trunk":
            ET.SubElement(body, "site", {
                "name": "imu_site", "pos": "0.00112019201975541 -0.0364882468746482 0.0511",
                "size": "0.005", "rgba": "1 0 0 1",
            })
            ET.SubElement(body, "site", {
                "name": "base_site", "pos": "0 0 0", "quat": "1 0 0 0",
            })
        for visual_geometry in link.findall("visual"):
            geom_attributes = {"class": "visual"}
            geom_attributes.update(_origin_attributes(visual_geometry.find("origin")))
            _add_geometry(body, visual_geometry.find("geometry"), geom_attributes)
        for index, collision_geometry in enumerate(link.findall("collision")):
            geom_attributes = {
                "class": "collision", "name": f"{link_name}_collision_{index}",
            }
            geom_attributes.update(_origin_attributes(collision_geometry.find("origin")))
            _add_geometry(body, collision_geometry.find("geometry"), geom_attributes)
        for child_joint in child_joints[link_name]:
            add_link(child_joint.find("child").get("link"), body, child_joint)

    add_link("trunk", worldbody)
    actuators = ET.SubElement(model, "actuator")
    for joint in urdf.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        name = joint.get("name")
        ET.SubElement(actuators, "motor", {
            "name": name, "joint": name, "ctrllimited": "true",
            "ctrlrange": f'-{limit.get("effort")} {limit.get("effort")}', "gear": "1",
        })
    exclusions = ET.SubElement(model, "contact")
    for joint in urdf.findall("joint"):
        ET.SubElement(exclusions, "exclude", {
            "body1": joint.find("parent").get("link"),
            "body2": joint.find("child").get("link"),
        })
    sensors = ET.SubElement(model, "sensor")
    ET.SubElement(sensors, "framequat", {
        "name": "base_quat", "objtype": "site", "objname": "base_site",
    })
    ET.SubElement(sensors, "accelerometer", {
        "name": "base_accelerometer", "site": "imu_site",
    })
    ET.SubElement(sensors, "gyro", {"name": "base_gyro", "site": "imu_site"})
    _indent(model)
    return ET.ElementTree(model)


def sync(source_root):
    source_root = source_root.expanduser().resolve()
    source_meshes = sorted((source_root / "meshes").glob("*.STL"))
    if not source_meshes:
        raise FileNotFoundError(f"No STL files found in {source_root / 'meshes'}")
    source_urdf = _source_urdf(source_root)
    tree = _prepare_isaac_urdf(source_urdf)
    source_bytes = source_urdf.read_bytes()
    if ROS_MESH_PREFIX in source_bytes:
        target_bytes = source_bytes.replace(ROS_MESH_PREFIX, LOCAL_MESH_PREFIX)
    elif source_urdf.resolve() == TARGET_URDF.resolve():
        # Allows MJCF regeneration after editing the checked-in URDF on a server.
        target_bytes = source_bytes
    else:
        raise ValueError(
            "Source URDF contains no expected package://nezha_description/meshes/ paths"
        )
    TARGET_MESH_DIR.mkdir(parents=True, exist_ok=True)
    source_names = {mesh.name for mesh in source_meshes}
    for target_mesh in TARGET_MESH_DIR.glob("*.STL"):
        if target_mesh.name not in source_names:
            target_mesh.unlink()
    for source_mesh in source_meshes:
        target_mesh = TARGET_MESH_DIR / source_mesh.name
        if source_mesh.resolve() != target_mesh.resolve():
            shutil.copy2(source_mesh, target_mesh)
    _write_bytes(target_bytes, TARGET_URDF)
    _write_tree(_generate_mjcf(_prepare_mjcf_source(tree)), TARGET_MJCF)
    print(f"source:       {source_root}")
    print(f"target URDF:  {TARGET_URDF}")
    print(f"target MJCF:  {TARGET_MJCF}")
    print(f"meshes:       {len(source_meshes)} synchronized")
    payload = tree.getroot().find("link[@name='top_box']/inertial/mass")
    print(f"payload:      {float(payload.get('value')):g} kg (copied unchanged)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Root of the ROS nezha_description package")
    return parser.parse_args()


if __name__ == "__main__":
    sync(parse_args().source)
