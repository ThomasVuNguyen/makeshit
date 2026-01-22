"""
3D Viewer widget using VTK for rendering OpenCASCADE shapes
"""
import numpy as np
from typing import List, Optional, Callable
from dataclasses import dataclass

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFrame
from PyQt6.QtCore import Qt, pyqtSignal

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

try:
    from step_loader import PartInfo, FaceInfo
except ImportError:
    from src.step_loader import PartInfo, FaceInfo


@dataclass
class PartActor:
    """VTK actor representing a part"""
    part_info: PartInfo
    actor: vtk.vtkActor
    original_color: tuple = (0.8, 0.8, 0.9)


class Viewer3D(QFrame):
    """3D viewer widget with part selection and face highlighting"""
    
    face_selected = pyqtSignal(object, object)  # (part_info, face_info)
    part_selected = pyqtSignal(object)  # part_info
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        
        self.parts: List[PartActor] = []
        self.selected_part: Optional[PartActor] = None
        self.selection_mode = "part"  # "part" or "face"
        self.highlighted_face_actor: Optional[vtk.vtkActor] = None
        
        self._setup_vtk()
        
    def _setup_vtk(self):
        """Initialize VTK rendering"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        
        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.15, 0.15, 0.2)  # Dark background
        self.renderer.SetBackground2(0.05, 0.05, 0.1)
        self.renderer.GradientBackgroundOn()
        
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        
        # Interactor
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # Picker for selection
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        
        # Add axis indicator
        self._add_axes()
        
        # Add lighting
        self._add_lighting()
        
        # Connect mouse events
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_left_click)
        self.interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
        
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        
    def _add_axes(self):
        """Add coordinate axes indicator"""
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(10, 10, 10)
        
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self.interactor)
        widget.SetViewport(0, 0, 0.2, 0.2)
        widget.EnabledOn()
        widget.InteractiveOff()
        self.axes_widget = widget
        
    def _add_lighting(self):
        """Setup lighting"""
        light1 = vtk.vtkLight()
        light1.SetLightTypeToSceneLight()
        light1.SetPosition(1, 1, 1)
        light1.SetIntensity(0.8)
        self.renderer.AddLight(light1)
        
        light2 = vtk.vtkLight()
        light2.SetLightTypeToSceneLight()
        light2.SetPosition(-1, -1, 0.5)
        light2.SetIntensity(0.4)
        self.renderer.AddLight(light2)
        
    def add_part(self, part_info: PartInfo, color: tuple = (0.8, 0.8, 0.9)):
        """Add a part to the scene"""
        if len(part_info.mesh_vertices) == 0:
            print(f"Warning: Part {part_info.name} has no mesh data")
            return
            
        # Create polydata from mesh
        polydata = vtk.vtkPolyData()
        
        # Points
        points = vtk.vtkPoints()
        for v in part_info.mesh_vertices:
            points.InsertNextPoint(v[0], v[1], v[2])
        polydata.SetPoints(points)
        
        # Triangles
        triangles = vtk.vtkCellArray()
        for f in part_info.mesh_faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, int(f[0]))
            triangle.GetPointIds().SetId(1, int(f[1]))
            triangle.GetPointIds().SetId(2, int(f[2]))
            triangles.InsertNextCell(triangle)
        polydata.SetPolys(triangles)
        
        # Compute normals for smooth shading
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()
        
        # Mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(normals.GetOutput())
        
        # Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        
        self.renderer.AddActor(actor)
        
        part_actor = PartActor(
            part_info=part_info,
            actor=actor,
            original_color=color
        )
        self.parts.append(part_actor)
        
        # Reset camera on first part
        if len(self.parts) == 1:
            self.renderer.ResetCamera()
            
        self.vtk_widget.GetRenderWindow().Render()
        
    def set_selection_mode(self, mode: str):
        """Set selection mode: 'part' or 'face'"""
        self.selection_mode = mode
        self._clear_highlight()
        
    def _on_left_click(self, obj, event):
        """Handle left click for selection"""
        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        
        picked_actor = self.picker.GetActor()
        
        if picked_actor:
            for part_actor in self.parts:
                if part_actor.actor == picked_actor:
                    if self.selection_mode == "part":
                        self._select_part(part_actor)
                    else:
                        # Face selection - use pick position to find closest face
                        pick_pos = self.picker.GetPickPosition()
                        face_info = self._find_closest_face(part_actor.part_info, pick_pos)
                        if face_info:
                            self.face_selected.emit(part_actor.part_info, face_info)
                    break
        
        # Forward the event
        self.interactor.GetInteractorStyle().OnLeftButtonDown()
        
    def _on_mouse_move(self, obj, event):
        """Handle mouse move for hover highlighting"""
        if self.selection_mode == "face":
            click_pos = self.interactor.GetEventPosition()
            self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
            
            picked_actor = self.picker.GetActor()
            
            if picked_actor:
                for part_actor in self.parts:
                    if part_actor.actor == picked_actor:
                        pick_pos = self.picker.GetPickPosition()
                        self._highlight_face_at(part_actor.part_info, pick_pos)
                        break
            else:
                self._clear_highlight()
        
        # Forward the event
        self.interactor.GetInteractorStyle().OnMouseMove()
        
    def _select_part(self, part_actor: PartActor):
        """Select a part"""
        # Reset previous selection
        if self.selected_part:
            self.selected_part.actor.GetProperty().SetColor(*self.selected_part.original_color)
        
        # Highlight new selection
        self.selected_part = part_actor
        part_actor.actor.GetProperty().SetColor(0.2, 0.8, 0.4)  # Green highlight
        
        self.vtk_widget.GetRenderWindow().Render()
        self.part_selected.emit(part_actor.part_info)
        
    def _find_closest_face(self, part_info: PartInfo, pick_pos: tuple) -> Optional[FaceInfo]:
        """Find the closest face to a pick position"""
        if not part_info.faces:
            return None
            
        pick_point = np.array(pick_pos)
        min_dist = float('inf')
        closest_face = None
        
        for face_info in part_info.faces:
            if face_info.center:
                center = np.array(face_info.center)
                dist = np.linalg.norm(pick_point - center)
                if dist < min_dist:
                    min_dist = dist
                    closest_face = face_info
        
        return closest_face
        
    def _highlight_face_at(self, part_info: PartInfo, pick_pos: tuple):
        """Highlight face at position with visual indicator"""
        face_info = self._find_closest_face(part_info, pick_pos)
        
        if face_info and face_info.center:
            self._clear_highlight()
            
            # Create a sphere at the face center as highlight
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*face_info.center)
            sphere.SetRadius(2.0)
            sphere.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(sphere.GetOutput())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            if face_info.face_type == "cylinder":
                actor.GetProperty().SetColor(1.0, 0.5, 0.0)  # Orange for cylinder
            else:
                actor.GetProperty().SetColor(0.0, 0.5, 1.0)  # Blue for plane
                
            actor.GetProperty().SetOpacity(0.8)
            
            self.renderer.AddActor(actor)
            self.highlighted_face_actor = actor
            self.vtk_widget.GetRenderWindow().Render()
            
    def _clear_highlight(self):
        """Clear face highlight"""
        if self.highlighted_face_actor:
            self.renderer.RemoveActor(self.highlighted_face_actor)
            self.highlighted_face_actor = None
            self.vtk_widget.GetRenderWindow().Render()
            
    def reset_camera(self):
        """Reset camera to fit all parts"""
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        
    def clear(self):
        """Clear all parts from scene"""
        for part_actor in self.parts:
            self.renderer.RemoveActor(part_actor.actor)
        self.parts.clear()
        self.selected_part = None
        self._clear_highlight()
        self.vtk_widget.GetRenderWindow().Render()
