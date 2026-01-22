"""
Constraint system for CAD assembly
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import numpy as np

try:
    from step_loader import PartInfo, FaceInfo
except ImportError:
    from src.step_loader import PartInfo, FaceInfo


class ConstraintType(Enum):
    CYLINDRICAL = "cylindrical"  # Revolute joint (hinge)
    PLANAR = "planar"  # Fixed/weld


@dataclass
class Constraint:
    """A mate constraint between two parts"""
    constraint_type: ConstraintType
    part1: PartInfo
    face1: FaceInfo
    part2: PartInfo
    face2: FaceInfo
    flipped: bool = False
    
    @property
    def name(self) -> str:
        return f"{self.constraint_type.value}_{self.part1.name}_{self.part2.name}"


class ConstraintManager:
    """Manages assembly constraints"""
    
    def __init__(self):
        self.constraints: List[Constraint] = []
        self.pending_constraint: Optional[dict] = None
        
    def start_constraint(self, constraint_type: ConstraintType):
        """Start creating a new constraint"""
        self.pending_constraint = {
            "type": constraint_type,
            "part1": None,
            "face1": None,
            "part2": None,
            "face2": None
        }
        
    def add_face_selection(self, part: PartInfo, face: FaceInfo) -> bool:
        """
        Add a face selection to the pending constraint
        Returns True if constraint is complete
        """
        if self.pending_constraint is None:
            return False
            
        # Validate face type for constraint
        ct = self.pending_constraint["type"]
        if ct == ConstraintType.CYLINDRICAL and face.face_type != "cylinder":
            print(f"Warning: Cylindrical mate requires cylindrical face, got {face.face_type}")
            return False
        if ct == ConstraintType.PLANAR and face.face_type != "plane":
            print(f"Warning: Planar mate requires planar face, got {face.face_type}")
            return False
        
        if self.pending_constraint["part1"] is None:
            self.pending_constraint["part1"] = part
            self.pending_constraint["face1"] = face
            return False
        elif self.pending_constraint["part2"] is None:
            # Ensure different parts
            if part.name == self.pending_constraint["part1"].name:
                print("Warning: Select a face on a different part")
                return False
                
            self.pending_constraint["part2"] = part
            self.pending_constraint["face2"] = face
            return True  # Constraint complete
            
        return False
        
    def finalize_constraint(self, flipped: bool = False) -> Optional[Constraint]:
        """Finalize the pending constraint"""
        if self.pending_constraint is None:
            return None
            
        if self.pending_constraint["part2"] is None:
            return None
            
        constraint = Constraint(
            constraint_type=self.pending_constraint["type"],
            part1=self.pending_constraint["part1"],
            face1=self.pending_constraint["face1"],
            part2=self.pending_constraint["part2"],
            face2=self.pending_constraint["face2"],
            flipped=flipped
        )
        
        self.constraints.append(constraint)
        self.pending_constraint = None
        
        return constraint
        
    def cancel_constraint(self):
        """Cancel the pending constraint"""
        self.pending_constraint = None
        
    def get_constraints(self) -> List[Constraint]:
        """Get all finalized constraints"""
        return self.constraints.copy()
        
    def compute_joint_axis(self, constraint: Constraint) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute joint position and axis from constraint
        Returns (position, axis)
        """
        if constraint.constraint_type == ConstraintType.CYLINDRICAL:
            # Use the axis from face1
            pos = np.array(constraint.face1.center or [0, 0, 0])
            axis = np.array(constraint.face1.axis or [0, 0, 1])
            
            if constraint.flipped:
                axis = -axis
                
            return pos, axis
            
        elif constraint.constraint_type == ConstraintType.PLANAR:
            # Use the normal from face1
            pos = np.array(constraint.face1.center or [0, 0, 0])
            axis = np.array(constraint.face1.normal or [0, 0, 1])
            
            if constraint.flipped:
                axis = -axis
                
            return pos, axis
            
        return np.array([0, 0, 0]), np.array([0, 0, 1])
