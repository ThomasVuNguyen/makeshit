"""
Main PyQt6 Application Window
"""
import os
import sys
from typing import List, Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QStatusBar, QMessageBox,
    QGroupBox, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt

try:
    from viewer import Viewer3D
    from step_loader import load_step, PartInfo
    from constraints import ConstraintManager, ConstraintType, Constraint
    from mjcf_exporter import export_mjcf
    from simulator import launch_viewer, run_random_validation, run_rl_training
except ImportError:
    from src.viewer import Viewer3D
    from src.step_loader import load_step, PartInfo
    from src.constraints import ConstraintManager, ConstraintType, Constraint
    from src.mjcf_exporter import export_mjcf
    from src.simulator import launch_viewer, run_random_validation, run_rl_training


class MakeshitApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Makeshit v2.1 - STEP to MuJoCo Assembly Tool")
        self.setMinimumSize(1200, 800)
        
        self.parts: List[PartInfo] = []
        self.constraint_manager = ConstraintManager()
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Setup the UI layout"""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        
        # 3D Viewer (left, takes most space)
        self.viewer = Viewer3D()
        layout.addWidget(self.viewer, stretch=3)
        
        # Control Panel (right)
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel, stretch=1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Parts will auto-load from public/")
        
    def _create_control_panel(self) -> QWidget:
        """Create the control panel with buttons"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Parts list
        parts_group = QGroupBox("Parts")
        parts_layout = QVBoxLayout(parts_group)
        self.parts_list = QListWidget()
        parts_layout.addWidget(self.parts_list)
        layout.addWidget(parts_group)
        
        # Constraints section
        constraints_group = QGroupBox("Constraints")
        constraints_layout = QVBoxLayout(constraints_group)
        
        self.btn_cylindrical = QPushButton("Cylindrical Mate")
        self.btn_cylindrical.setToolTip("Create revolute joint between cylindrical faces")
        constraints_layout.addWidget(self.btn_cylindrical)
        
        self.btn_planar = QPushButton("Planar Mate")
        self.btn_planar.setToolTip("Create fixed connection between planar faces")
        constraints_layout.addWidget(self.btn_planar)
        
        self.btn_flip = QPushButton("Flip")
        self.btn_flip.setToolTip("Flip the constraint direction")
        self.btn_flip.setEnabled(False)
        constraints_layout.addWidget(self.btn_flip)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setToolTip("Cancel current constraint")
        self.btn_cancel.setEnabled(False)
        constraints_layout.addWidget(self.btn_cancel)
        
        layout.addWidget(constraints_group)
        
        # Constraints list
        self.constraints_list = QListWidget()
        layout.addWidget(self.constraints_list)
        
        # Actions section
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.btn_simulate = QPushButton("Simulate")
        self.btn_simulate.setToolTip("Export to MJCF and run MuJoCo simulation")
        self.btn_simulate.setStyleSheet("font-weight: bold; padding: 10px;")
        actions_layout.addWidget(self.btn_simulate)
        
        layout.addWidget(actions_group)
        
        # Status label
        self.constraint_status = QLabel("Select a constraint type")
        self.constraint_status.setWordWrap(True)
        layout.addWidget(self.constraint_status)
        
        layout.addStretch()
        
        return panel
        
    def _connect_signals(self):
        """Connect UI signals"""
        self.btn_cylindrical.clicked.connect(self._on_cylindrical_mate)
        self.btn_planar.clicked.connect(self._on_planar_mate)
        self.btn_flip.clicked.connect(self._on_flip)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_simulate.clicked.connect(self._on_simulate)
        
        self.viewer.face_selected.connect(self._on_face_selected)
        self.viewer.part_selected.connect(self._on_part_selected)
        
    def load_step_file(self, filepath: str):
        """Load a STEP file and add to scene"""
        try:
            self.status_bar.showMessage(f"Loading {os.path.basename(filepath)}...")
            
            part_info = load_step(filepath)
            self.parts.append(part_info)
            
            # Get alternating colors
            colors = [
                (0.7, 0.7, 0.8),  # Light gray-blue
                (0.3, 0.3, 0.35),  # Dark gray
            ]
            color = colors[len(self.parts) % len(colors)]
            
            self.viewer.add_part(part_info, color)
            
            # Add to parts list
            item = QListWidgetItem(part_info.name)
            self.parts_list.addItem(item)
            
            self.status_bar.showMessage(
                f"Loaded {part_info.name}: {len(part_info.mesh_vertices)} vertices, "
                f"{len(part_info.faces)} selectable faces"
            )
            
        except Exception as e:
            self.status_bar.showMessage(f"Error loading {filepath}: {str(e)}")
            print(f"Error: {e}")
            
    def _on_cylindrical_mate(self):
        """Start cylindrical mate constraint"""
        self.constraint_manager.start_constraint(ConstraintType.CYLINDRICAL)
        self.viewer.set_selection_mode("face")
        self._update_constraint_ui(True)
        self.constraint_status.setText("Select first cylindrical face...")
        self.status_bar.showMessage("Cylindrical Mate: Click on a cylindrical face")
        
    def _on_planar_mate(self):
        """Start planar mate constraint"""
        self.constraint_manager.start_constraint(ConstraintType.PLANAR)
        self.viewer.set_selection_mode("face")
        self._update_constraint_ui(True)
        self.constraint_status.setText("Select first planar face...")
        self.status_bar.showMessage("Planar Mate: Click on a planar face")
        
    def _on_flip(self):
        """Flip the pending constraint direction"""
        if self.constraint_manager.pending_constraint:
            constraint = self.constraint_manager.finalize_constraint(flipped=True)
            if constraint:
                self._add_constraint_to_list(constraint)
                self._reset_constraint_mode()
                self.status_bar.showMessage(f"Created flipped constraint: {constraint.name}")
        
    def _on_cancel(self):
        """Cancel current constraint"""
        self.constraint_manager.cancel_constraint()
        self._reset_constraint_mode()
        self.status_bar.showMessage("Constraint cancelled")
        
    def _on_face_selected(self, part_info: PartInfo, face_info):
        """Handle face selection for constraints"""
        complete = self.constraint_manager.add_face_selection(part_info, face_info)
        
        if complete:
            self.constraint_status.setText(
                f"Faces selected! Click 'Flip' to reverse direction, or selecting will confirm."
            )
            self.btn_flip.setEnabled(True)
            
            # Auto-finalize without flip
            constraint = self.constraint_manager.finalize_constraint(flipped=False)
            if constraint:
                self._add_constraint_to_list(constraint)
                self._reset_constraint_mode()
                self.status_bar.showMessage(f"Created constraint: {constraint.name}")
        else:
            self.constraint_status.setText(
                f"Selected face on {part_info.name} ({face_info.face_type}). "
                f"Select second face on different part..."
            )
            
    def _on_part_selected(self, part_info: PartInfo):
        """Handle part selection"""
        self.status_bar.showMessage(f"Selected part: {part_info.name}")
        
    def _add_constraint_to_list(self, constraint: Constraint):
        """Add constraint to the UI list"""
        text = f"{constraint.constraint_type.value}: {constraint.part1.name} ↔ {constraint.part2.name}"
        if constraint.flipped:
            text += " (flipped)"
        item = QListWidgetItem(text)
        self.constraints_list.addItem(item)
        
    def _update_constraint_ui(self, active: bool):
        """Update UI for constraint creation mode"""
        self.btn_cylindrical.setEnabled(not active)
        self.btn_planar.setEnabled(not active)
        self.btn_cancel.setEnabled(active)
        self.btn_flip.setEnabled(False)
        
    def _reset_constraint_mode(self):
        """Reset to normal mode"""
        self.viewer.set_selection_mode("part")
        self._update_constraint_ui(False)
        self.constraint_status.setText("Select a constraint type")
        
    def _on_simulate(self):
        """Export and run simulation"""
        if not self.parts:
            QMessageBox.warning(self, "No Parts", "Load some STEP files first!")
            return
            
        self.status_bar.showMessage("Exporting to MJCF...")
        
        try:
            # Export
            constraints = self.constraint_manager.get_constraints()
            mjcf_path = export_mjcf(
                self.parts,
                constraints,
                self.output_dir,
                "servo_assembly"
            )
            
            self.status_bar.showMessage(f"Exported to {mjcf_path}")
            
            # Validate
            result = run_random_validation(mjcf_path)
            
            if result["success"]:
                self.status_bar.showMessage(
                    f"Validation passed! Running RL training..."
                )
                
                # Run short training
                run_rl_training(mjcf_path, total_timesteps=5000)
                
                self.status_bar.showMessage("Training complete! Launching viewer...")
                
                # Launch viewer
                launch_viewer(mjcf_path)
            else:
                QMessageBox.critical(
                    self, "Validation Failed", 
                    f"Model validation failed:\n{result.get('error', 'Unknown error')}"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulation failed:\n{str(e)}")
            self.status_bar.showMessage(f"Error: {str(e)}")
