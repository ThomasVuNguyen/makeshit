#!/usr/bin/env python3
"""
Makeshit v2.1 - STEP to MuJoCo Assembly Tool
Entry point for the application
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt6.QtWidgets import QApplication
from app import MakeshitApp


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Makeshit v2.1")
    app.setApplicationDisplayName("STEP to MuJoCo Assembly Tool")
    
    window = MakeshitApp()
    window.show()
    
    # Auto-load STEP files from public folder
    public_dir = os.path.join(os.path.dirname(__file__), 'public')
    if os.path.exists(public_dir):
        step_files = [f for f in os.listdir(public_dir) if f.endswith('.step')]
        for step_file in step_files:
            window.load_step_file(os.path.join(public_dir, step_file))
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
