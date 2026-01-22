# Makeshit v2.1 - STEP to MuJoCo Assembly Tool

A Python desktop application to import STEP CAD files, assemble them with constraints, and export to MuJoCo for simulation and reinforcement learning.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **STEP Import**: Load STEP CAD files with full geometry extraction
- **3D Viewer**: Interactive VTK-based viewer with part selection
- **Constraint System**: Cylindrical and Planar mate constraints
- **MuJoCo Export**: Generate MJCF XML with proper joint definitions
- **RL Training**: Built-in validation and training with Stable-Baselines3

## Installation

### Prerequisites
- Python 3.11 (required for CAD library compatibility)
- macOS/Linux (Windows may require additional setup)

### Setup

```bash
# Clone the repository
git clone https://github.com/ThomasVuNguyen/makeshit.git
cd makeshit/v2.1

# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Activate environment
source venv/bin/activate

# Launch the application
python main.py
```

### End-to-End Workflow

1. **Launch Application**
   ```bash
   python main.py
   ```
   The app auto-loads STEP files from the `public/` folder.

2. **View Parts**
   - Parts appear in the 3D viewer (left panel)
   - Rotate: Left-click + drag
   - Pan: Middle-click + drag
   - Zoom: Scroll wheel

3. **Create Constraints**
   - Click **"Cylindrical Mate"** button
   - Hover over parts to see face highlights (orange = cylinder, blue = plane)
   - Click on a cylindrical face (e.g., motor shaft)
   - Click on matching face on the other part (e.g., horn hole)
   - Constraint appears in the list

4. **Simulate**
   - Click **"Simulate"** button
   - Application exports MJCF XML to `output/`
   - Runs random validation (500 steps)
   - Runs PPO training (5000 timesteps)
   - Launches MuJoCo viewer

5. **Output Files**
   - `output/servo_assembly.xml` - MuJoCo model
   - `output/meshes/*.stl` - Exported mesh files

### Adding Your Own STEP Files

Place your STEP files in the `public/` folder:
```
public/
├── your_part1.step
├── your_part2.step
└── ...
```

They will auto-load when you launch the application.

## Project Structure

```
v2.1/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── public/                 # STEP files (input)
│   ├── motor.step
│   └── horn.step
├── output/                 # Generated files
│   ├── servo_assembly.xml
│   └── meshes/
├── src/
│   ├── app.py              # PyQt6 main window
│   ├── viewer.py           # VTK 3D viewer
│   ├── step_loader.py      # STEP parsing (OpenCASCADE)
│   ├── constraints.py      # Mate constraint system
│   ├── mjcf_exporter.py    # MJCF XML generation
│   └── simulator.py        # MuJoCo + RL training
└── venv/                   # Virtual environment
```

## Constraint Types

| Type | Description | MuJoCo Joint |
|------|-------------|--------------|
| Cylindrical Mate | Aligns cylindrical faces | `hinge` (revolute) |
| Planar Mate | Aligns planar faces | Fixed (weld) |

## Dependencies

- **cadquery-ocp** - OpenCASCADE for STEP parsing
- **PyQt6** - Desktop UI framework
- **VTK** - 3D visualization
- **MuJoCo** - Physics simulation
- **Stable-Baselines3** - Reinforcement learning
- **trimesh** - Mesh processing

## Troubleshooting

### "No module named 'step_loader'"
Make sure you're running from the project root:
```bash
cd /path/to/makeshit/v2.1
python main.py
```

### VTK display issues on macOS
```bash
export QT_MAC_WANTS_LAYER=1
python main.py
```

### Memory usage
The app targets <1GB RAM. For large STEP files, increase mesh deflection:
```python
# In step_loader.py, increase linear_deflection
load_step(filepath, linear_deflection=0.5)  # coarser mesh
```

## License

MIT License
