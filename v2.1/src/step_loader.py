"""
STEP file loader using OCP (OpenCASCADE Python)
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Face
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCP.gp import gp_Pnt, gp_Vec


@dataclass
class FaceInfo:
    """Information about a face for constraint selection"""
    face: TopoDS_Face
    face_type: str  # "plane" or "cylinder"
    normal: Optional[Tuple[float, float, float]] = None
    axis: Optional[Tuple[float, float, float]] = None
    center: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None


@dataclass
class PartInfo:
    """Information about a loaded part"""
    name: str
    filepath: str
    shape: TopoDS_Shape
    mesh_vertices: np.ndarray = field(default_factory=lambda: np.array([]))
    mesh_faces: np.ndarray = field(default_factory=lambda: np.array([]))
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    faces: List[FaceInfo] = field(default_factory=list)


def load_step(filepath: str, linear_deflection: float = 0.1) -> PartInfo:
    """
    Load a STEP file and extract mesh data
    
    Args:
        filepath: Path to the STEP file
        linear_deflection: Mesh quality (smaller = finer mesh)
    
    Returns:
        PartInfo object with mesh and face data
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    
    if status != IFSelect_RetDone:
        raise ValueError(f"Failed to read STEP file: {filepath}")
    
    reader.TransferRoots()
    shape = reader.OneShape()
    
    # Generate mesh
    BRepMesh_IncrementalMesh(shape, linear_deflection)
    
    # Extract vertices and faces from mesh
    vertices, faces = extract_mesh(shape)
    
    # Extract face information for constraints
    face_infos = extract_faces(shape)
    
    name = os.path.splitext(os.path.basename(filepath))[0]
    
    return PartInfo(
        name=name,
        filepath=filepath,
        shape=shape,
        mesh_vertices=vertices,
        mesh_faces=faces,
        faces=face_infos
    )


def extract_mesh(shape: TopoDS_Shape) -> Tuple[np.ndarray, np.ndarray]:
    """Extract triangulated mesh from shape"""
    from OCP.TopoDS import TopoDS
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    
    while explorer.More():
        # Properly cast to TopoDS_Face
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        
        if triangulation is not None:
            # Get transformation
            trsf = location.Transformation()
            
            # Extract vertices
            num_vertices = triangulation.NbNodes()
            for i in range(1, num_vertices + 1):
                pnt = triangulation.Node(i)
                pnt = pnt.Transformed(trsf)
                all_vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
            
            # Extract triangles
            num_triangles = triangulation.NbTriangles()
            for i in range(1, num_triangles + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                # Convert to 0-indexed and apply offset
                all_faces.append([
                    n1 - 1 + vertex_offset,
                    n2 - 1 + vertex_offset,
                    n3 - 1 + vertex_offset
                ])
            
            vertex_offset += num_vertices
        
        explorer.Next()
    
    if len(all_vertices) == 0:
        return np.array([]), np.array([])
    
    return np.array(all_vertices, dtype=np.float32), np.array(all_faces, dtype=np.int32)


def extract_faces(shape: TopoDS_Shape) -> List[FaceInfo]:
    """Extract face geometry information for constraint selection"""
    from OCP.TopoDS import TopoDS
    
    face_infos = []
    
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        adaptor = BRepAdaptor_Surface(face)
        surf_type = adaptor.GetType()
        
        if surf_type == GeomAbs_Plane:
            # Planar face
            plane = adaptor.Plane()
            axis = plane.Axis()
            direction = axis.Direction()
            location = plane.Location()
            
            face_infos.append(FaceInfo(
                face=face,
                face_type="plane",
                normal=(direction.X(), direction.Y(), direction.Z()),
                center=(location.X(), location.Y(), location.Z())
            ))
            
        elif surf_type == GeomAbs_Cylinder:
            # Cylindrical face
            cylinder = adaptor.Cylinder()
            axis = cylinder.Axis()
            direction = axis.Direction()
            location = cylinder.Location()
            radius = cylinder.Radius()
            
            face_infos.append(FaceInfo(
                face=face,
                face_type="cylinder",
                axis=(direction.X(), direction.Y(), direction.Z()),
                center=(location.X(), location.Y(), location.Z()),
                radius=radius
            ))
        
        explorer.Next()
    
    return face_infos
