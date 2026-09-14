#!/usr/bin/env python3
"""Import the canonical Nezha URDF and meshes into this repository.

The ROS/Gazebo source uses a massless ``base`` wrapper and package:// mesh
URLs.  Isaac Gym needs the physical ``trunk`` to remain the root body, so the
imported URDF intentionally removes only that wrapper plus Gazebo-specific
plugins/transmissions and rewrites mesh URLs to repository-relative paths.
"""

import argparse
import math
import os
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_URDF = PROJECT_ROOT / "assets" / "nezha" / "urdf" / "nezha.urdf"
TARGET_MESH_DIR = PROJECT_ROOT / "assets" / "nezha" / "meshes"
TARGET_MJCF = PROJECT_ROOT / "mujoco" / "models" / "nezha.xml"
EXPECTED_DOF_NAMES = {
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf", "foot")
}


def _source_urdf(source_root: Path) -> Path:
    candidates = (
        source_root / "urdf" / "nezha_description.urdf",
        source_root / "urdf" / "nezha.urdf",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No canonical URDF found; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _indent(element: ET.Element, level: int = 0) -> None:
    """Python 3.8-compatible equivalent of ElementTree.indent."""
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


def _adapt_urdf(source_urdf: Path) -> ET.ElementTree:
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    root.set("name", "nezha")

    for element in list(root):
        if element.tag in {"gazebo", "transmission"}:
            root.remove(element)
        elif element.tag == "link" and element.get("name") == "base":
            root.remove(element)
        elif element.tag == "joint" and element.get("name") == "floating_base":
            root.remove(element)

    for mesh in root.findall(".//mesh"):
        source_name = Path(mesh.get("filename", "")).name
        if not source_name:
            raise ValueError("URDF contains a mesh without a filename")
        mesh.set("filename", f"../meshes/{source_name}")

    links = {link.get("name") for link in root.findall("link")}
    joints = root.findall("joint")
    dofs = {joint.get("name") for joint in joints if joint.get("type") != "fixed"}
    if "trunk" not in links or "base" in links:
        raise ValueError("Adapted URDF must have trunk, not base, as its root link")
    if dofs != EXPECTED_DOF_NAMES:
        raise ValueError(
            f"Unexpected movable-joint contract: {sorted(dofs)}"
        )

    _indent(root)
    return tree


def _write_tree_atomically(tree: ET.ElementTree, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_mode = (
        stat.S_IMODE(destination.stat().st_mode) if destination.exists() else 0o644
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            tree.write(stream, encoding="utf-8", xml_declaration=True)
        os.chmod(temporary_name, destination_mode)
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _numbers(value: str) -> List[float]:
    return [float(number) for number in value.split()]


def _half_extents(value: str) -> str:
    return " ".join(f"{number / 2:g}" for number in _numbers(value))


def _quaternion(rpy: str) -> Optional[str]:
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


def _origin_attributes(element: Optional[ET.Element]) -> Dict[str, str]:
    if element is None:
        return {}
    attributes = {}
    xyz = element.get("xyz", "0 0 0")
    if any(value != 0.0 for value in _numbers(xyz)):
        attributes["pos"] = xyz
    quat = _quaternion(element.get("rpy", "0 0 0"))
    if quat:
        attributes["quat"] = quat
    return attributes


def _add_geometry(
    parent: ET.Element, geometry: ET.Element, attributes: Dict[str, str]
) -> None:
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


def _generate_mjcf(urdf_tree: ET.ElementTree) -> ET.ElementTree:
    urdf = urdf_tree.getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    children = {name: [] for name in links}  # type: Dict[str, List[ET.Element]]
    child_links = set()
    for joint in urdf.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        children[parent].append(joint)
        child_links.add(child)
    roots = set(links) - child_links
    if roots != {"trunk"}:
        raise ValueError(f"Expected trunk as sole URDF root, found {sorted(roots)}")

    model = ET.Element("mujoco", {"model": "nezha"})
    default = ET.SubElement(model, "default")
    robot = ET.SubElement(default, "default", {"class": "robot"})
    motor = ET.SubElement(robot, "default", {"class": "motor"})
    ET.SubElement(motor, "joint")
    ET.SubElement(motor, "motor")
    visual = ET.SubElement(robot, "default", {"class": "visual"})
    ET.SubElement(
        visual,
        "geom",
        {
            "material": "default_material",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "2",
        },
    )
    collision = ET.SubElement(robot, "default", {"class": "collision"})
    ET.SubElement(
        collision,
        "geom",
        {
            "material": "collision_material",
            "condim": "3",
            "contype": "1",
            "conaffinity": "2",
            "solref": "0.005 1",
            "friction": "1.0 0.01 0.001",
            "density": "0",
            "group": "3",
        },
    )
    ET.SubElement(
        model,
        "compiler",
        {
            "angle": "radian",
            "meshdir": "../../assets/nezha/meshes/",
            "autolimits": "true",
        },
    )
    ET.SubElement(model, "size", {"njmax": "500", "nconmax": "100"})
    assets = ET.SubElement(model, "asset")
    ET.SubElement(
        assets,
        "material",
        {
            "name": "default_material",
            "rgba": "0.72 0.80 0.88 1",
            "specular": "0.35",
            "shininess": "0.45",
            "reflectance": "0.06",
        },
    )
    ET.SubElement(
        assets,
        "material",
        {"name": "collision_material", "rgba": "0.0 0.4 0.8 0.2"},
    )
    for mesh in urdf.findall(".//mesh"):
        filename = Path(mesh.get("filename")).name
        name = Path(filename).stem
        if assets.find(f"mesh[@name='{name}']") is None:
            ET.SubElement(assets, "mesh", {"name": name, "file": filename})

    worldbody = ET.SubElement(model, "worldbody")

    def add_link(
        link_name: str,
        parent: ET.Element,
        source_joint: Optional[ET.Element] = None,
    ) -> None:
        attributes = {"name": link_name}
        if link_name == "trunk":
            attributes.update(childclass="robot", pos="0 0 0.54", quat="1 0 0 0")
        elif source_joint is not None:
            attributes.update(_origin_attributes(source_joint.find("origin")))
        body = ET.SubElement(parent, "body", attributes)

        if link_name == "trunk":
            ET.SubElement(body, "freejoint", {"name": "floating_base"})
        elif source_joint is not None and source_joint.get("type") != "fixed":
            axis = source_joint.find("axis").get("xyz")
            limit = source_joint.find("limit")
            joint_attributes = {
                "name": source_joint.get("name"),
                "axis": axis,
                "range": f'{limit.get("lower")} {limit.get("upper")}',
                "actuatorfrcrange": f'-{limit.get("effort")} {limit.get("effort")}',
                # These are the passive-friction values applied by NezhaStandEnv.
                "frictionloss": (
                    "0.025" if "hip_joint" in source_joint.get("name") else "0.05"
                ),
            }
            ET.SubElement(body, "joint", joint_attributes)

        link = links[link_name]
        inertial = link.find("inertial")
        if inertial is not None and float(inertial.find("mass").get("value")) > 0.0:
            inertia = inertial.find("inertia")
            inertial_attributes = _origin_attributes(inertial.find("origin"))
            inertial_attributes.update(
                mass=inertial.find("mass").get("value"),
                fullinertia=" ".join(
                    inertia.get(name)
                    for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                ),
            )
            ET.SubElement(body, "inertial", inertial_attributes)

        if link_name == "trunk":
            ET.SubElement(
                body,
                "site",
                {"name": "imu_site", "pos": "0.00112019201975541 -0.0364882468746482 0.0511", "size": "0.005", "rgba": "1 0 0 1"},
            )
            ET.SubElement(
                body,
                "site",
                {"name": "base_site", "pos": "0 0 0", "quat": "1 0 0 0"},
            )
        for visual_geometry in link.findall("visual"):
            geom_attributes = {"class": "visual"}
            geom_attributes.update(_origin_attributes(visual_geometry.find("origin")))
            _add_geometry(body, visual_geometry.find("geometry"), geom_attributes)
        for index, collision_geometry in enumerate(link.findall("collision")):
            geom_attributes = {
                "class": "collision",
                "name": f"{link_name}_collision_{index}",
            }
            geom_attributes.update(_origin_attributes(collision_geometry.find("origin")))
            _add_geometry(body, collision_geometry.find("geometry"), geom_attributes)
        for child_joint in children[link_name]:
            add_link(child_joint.find("child").get("link"), body, child_joint)

    add_link("trunk", worldbody)

    actuators = ET.SubElement(model, "actuator")
    for joint in urdf.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        name = joint.get("name")
        ET.SubElement(
            actuators,
            "motor",
            {
                "name": name,
                "joint": name,
                "ctrllimited": "true",
                "ctrlrange": f'-{limit.get("effort")} {limit.get("effort")}',
                "gear": "1",
            },
        )

    exclusions = ET.SubElement(model, "contact")
    for joint in urdf.findall("joint"):
        ET.SubElement(
            exclusions,
            "exclude",
            {
                "body1": joint.find("parent").get("link"),
                "body2": joint.find("child").get("link"),
            },
        )
    sensors = ET.SubElement(model, "sensor")
    ET.SubElement(
        sensors,
        "framequat",
        {"name": "base_quat", "objtype": "site", "objname": "base_site"},
    )
    ET.SubElement(
        sensors, "accelerometer", {"name": "base_accelerometer", "site": "imu_site"}
    )
    ET.SubElement(sensors, "gyro", {"name": "base_gyro", "site": "imu_site"})
    tree = ET.ElementTree(model)
    _indent(model)
    return tree


def sync(source_root: Path) -> None:
    source_root = source_root.expanduser().resolve()
    source_mesh_dir = source_root / "meshes"
    source_meshes = sorted(source_mesh_dir.glob("*.STL"))
    if not source_meshes:
        raise FileNotFoundError(f"No STL files found in {source_mesh_dir}")

    source_urdf = _source_urdf(source_root)
    tree = _adapt_urdf(source_urdf)

    TARGET_MESH_DIR.mkdir(parents=True, exist_ok=True)
    source_names = {mesh.name for mesh in source_meshes}
    for stale_mesh in TARGET_MESH_DIR.glob("*.STL"):
        if stale_mesh.name not in source_names:
            stale_mesh.unlink()
    for source_mesh in source_meshes:
        shutil.copyfile(source_mesh, TARGET_MESH_DIR / source_mesh.name)
    _write_tree_atomically(tree, TARGET_URDF)
    _write_tree_atomically(_generate_mjcf(tree), TARGET_MJCF)

    print(f"source URDF: {source_urdf}")
    print(f"target URDF: {TARGET_URDF}")
    print(f"target MJCF: {TARGET_MJCF}")
    print(f"meshes copied: {len(source_meshes)}")
    print("adaptation: trunk root; package URLs rewritten; Gazebo metadata removed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/crown/nezha_description"),
        help="Root of the ROS nezha_description package",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sync(parse_args().source)
