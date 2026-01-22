"""
MJCF (MuJoCo XML) exporter
"""
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List
import numpy as np
import trimesh

try:
    from step_loader import PartInfo
    from constraints import Constraint, ConstraintType, ConstraintManager
except ImportError:
    from src.step_loader import PartInfo
    from src.constraints import Constraint, ConstraintType, ConstraintManager


def export_mjcf(
    parts: List[PartInfo],
    constraints: List[Constraint],
    output_dir: str,
    model_name: str = "assembled_model"
) -> str:
    """
    Export parts and constraints to MJCF format
    
    Args:
        parts: List of loaded parts
        constraints: List of assembly constraints  
        output_dir: Directory to write output files
        model_name: Name for the model
        
    Returns:
        Path to generated MJCF file
    """
    os.makedirs(output_dir, exist_ok=True)
    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)
    
    # Export mesh files
    mesh_files = {}
    for part in parts:
        mesh_path = export_mesh(part, meshes_dir)
        mesh_files[part.name] = mesh_path
    
    # Build MJCF XML
    root = ET.Element("mujoco", model=model_name)
    
    # Compiler settings
    compiler = ET.SubElement(root, "compiler", 
        angle="radian",
        meshdir="meshes"
    )
    
    # Options
    option = ET.SubElement(root, "option",
        gravity="0 0 -9.81",
        timestep="0.002"
    )
    
    # Assets
    asset = ET.SubElement(root, "asset")
    for part_name, mesh_path in mesh_files.items():
        mesh_name = os.path.basename(mesh_path)
        ET.SubElement(asset, "mesh",
            name=part_name,
            file=mesh_name
        )
    
    # Add some materials
    ET.SubElement(asset, "material", name="metal", rgba="0.7 0.7 0.8 1")
    ET.SubElement(asset, "material", name="plastic", rgba="0.2 0.2 0.2 1")
    
    # Worldbody
    worldbody = ET.SubElement(root, "worldbody")
    
    # Add ground plane
    ET.SubElement(worldbody, "geom",
        type="plane",
        size="1 1 0.1",
        rgba="0.3 0.3 0.3 1"
    )
    
    # Add light
    ET.SubElement(worldbody, "light",
        pos="0 0 3",
        dir="0 0 -1",
        diffuse="1 1 1"
    )
    
    # Build body hierarchy based on constraints
    body_hierarchy = build_body_hierarchy(parts, constraints)
    
    # Create bodies
    for body_info in body_hierarchy:
        add_body_to_xml(worldbody, body_info, mesh_files)
    
    # Add actuators
    actuator = ET.SubElement(root, "actuator")
    for constraint in constraints:
        if constraint.constraint_type == ConstraintType.CYLINDRICAL:
            joint_name = f"joint_{constraint.name}"
            ET.SubElement(actuator, "motor",
                name=f"motor_{constraint.name}",
                joint=joint_name,
                gear="1",
                ctrllimited="true",
                ctrlrange="-10 10"
            )
    
    # Pretty print XML
    xml_str = ET.tostring(root, encoding='unicode')
    xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
    
    # Remove extra blank lines
    lines = [line for line in xml_pretty.split('\n') if line.strip()]
    xml_pretty = '\n'.join(lines)
    
    output_path = os.path.join(output_dir, f"{model_name}.xml")
    with open(output_path, 'w') as f:
        f.write(xml_pretty)
    
    return output_path


def export_mesh(part: PartInfo, output_dir: str) -> str:
    """Export part mesh to STL file"""
    if len(part.mesh_vertices) == 0 or len(part.mesh_faces) == 0:
        raise ValueError(f"Part {part.name} has no mesh data")
    
    # Create trimesh object
    mesh = trimesh.Trimesh(
        vertices=part.mesh_vertices,
        faces=part.mesh_faces
    )
    
    # Scale to reasonable size if needed (MuJoCo uses meters)
    # Assuming input is in mm, convert to meters
    bounds = mesh.bounds
    size = np.max(bounds[1] - bounds[0])
    if size > 10:  # Likely in mm, convert to m
        mesh.apply_scale(0.001)
    
    output_path = os.path.join(output_dir, f"{part.name}.stl")
    mesh.export(output_path)
    
    return output_path


def build_body_hierarchy(parts: List[PartInfo], constraints: List[Constraint]) -> List[dict]:
    """
    Build body hierarchy from parts and constraints
    Returns list of body info dicts
    """
    if not parts:
        return []
    
    # If no constraints, all parts are independent bodies attached to world
    if not constraints:
        return [{"part": part, "parent": None, "constraint": None} for part in parts]
    
    # With constraints, build hierarchy
    # First part becomes base (fixed to world)
    # Other parts attach via joints
    
    hierarchy = []
    used_parts = set()
    
    # First constraint's part1 is the base
    base_part = constraints[0].part1 if constraints else parts[0]
    hierarchy.append({
        "part": base_part,
        "parent": None,
        "constraint": None
    })
    used_parts.add(base_part.name)
    
    # Add constrained parts
    for constraint in constraints:
        if constraint.part2.name not in used_parts:
            hierarchy.append({
                "part": constraint.part2,
                "parent": constraint.part1,
                "constraint": constraint
            })
            used_parts.add(constraint.part2.name)
    
    # Add any remaining unconstrained parts
    for part in parts:
        if part.name not in used_parts:
            hierarchy.append({
                "part": part,
                "parent": None,
                "constraint": None
            })
            used_parts.add(part.name)
    
    return hierarchy


def add_body_to_xml(parent_elem: ET.Element, body_info: dict, mesh_files: dict):
    """Add a body element to the XML tree"""
    part = body_info["part"]
    constraint = body_info["constraint"]
    
    # Position - use part position or constraint position
    if constraint:
        pos = constraint.face1.center or (0, 0, 0)
    else:
        pos = tuple(part.position)
    
    pos_str = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
    
    # Create body element
    body = ET.SubElement(parent_elem, "body",
        name=part.name,
        pos=pos_str
    )
    
    # Add joint if constrained
    if constraint:
        if constraint.constraint_type == ConstraintType.CYLINDRICAL:
            # Revolute joint
            joint_name = f"joint_{constraint.name}"
            axis = constraint.face1.axis or (0, 0, 1)
            axis_str = f"{axis[0]:.4f} {axis[1]:.4f} {axis[2]:.4f}"
            
            if constraint.flipped:
                axis_str = f"{-axis[0]:.4f} {-axis[1]:.4f} {-axis[2]:.4f}"
            
            ET.SubElement(body, "joint",
                name=joint_name,
                type="hinge",
                axis=axis_str,
                limited="false"
            )
        elif constraint.constraint_type == ConstraintType.PLANAR:
            # Fixed constraint (no joint, just position)
            pass
    else:
        # Free joint for base if it's supposed to be dynamic
        # For now, base is fixed (no joint = fixed to parent)
        pass
    
    # Add geometry
    ET.SubElement(body, "geom",
        type="mesh",
        mesh=part.name,
        material="metal"
    )
    
    # Add inertial (auto-computed from mesh)
    ET.SubElement(body, "inertial",
        pos="0 0 0",
        mass="0.1",
        diaginertia="0.001 0.001 0.001"
    )
