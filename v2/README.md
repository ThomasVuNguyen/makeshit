# Robot Design Studio

A web-based CAD assembly tool for designing robots, with direct export to MuJoCo for physics simulation and training.

![Robot Assembly → MuJoCo Simulation](https://img.shields.io/badge/STEP%20Files-→%20MuJoCo-blue)

## Overview

This tool allows you to:
1. **Import STEP files** - Load 3D CAD models of robot parts
2. **Assemble parts** - Use CAD-like mating features (Cylindrical Mate)
3. **Export to MuJoCo** - Generate MJCF + real mesh STL files
4. **Simulate & Train** - Run physics simulation with visual feedback

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.12+

### Installation

```bash
# Install web app dependencies
npm install

# Set up Python environment
cd train
python -m venv venv
./venv/bin/pip install -r requirements.txt
cd ..
```

### Running

**Terminal 1 - Web App:**
```bash
npm run dev
```
Opens at http://localhost:5173

**Terminal 2 - Save Server:**
```bash
cd train
./venv/bin/python save_server.py
```
Runs at http://localhost:3001

## Designing a New Robot

### 1. Add Your STEP Files

Place `.step` or `.stp` files in:
```
public/models/
```

### 2. Assemble in Browser

1. Open http://localhost:5173
2. Click **Import** to load your STEP files
3. Click **Mate Mode** to enter assembly mode
4. Select **Cylindrical Mate** for rotational joints
5. Click on faces to connect parts:
   - First click → selects source face
   - Second click → selects target face
   - Parts align automatically

### 3. Export & Simulate

1. Click **Simulate** button
2. Files are exported to:
   - `output/model.xml` - MuJoCo MJCF
   - `output/meshes/*.stl` - Real geometry
3. Run simulation:
   ```bash
   cd train
   ./venv/bin/mjpython train.py
   ```

### 4. Iterate

- Modify assembly → Click Simulate → See changes in MuJoCo
- Each export overwrites previous files for quick iteration

## Project Structure

```
v2/
├── public/models/          # STEP files go here
├── src/
│   ├── components/
│   │   ├── Viewer.tsx      # 3D viewport + assembly logic
│   │   └── ControlPanel.tsx # UI controls
│   ├── stores/
│   │   └── useStore.ts     # State management + mate logic
│   └── utils/
│       ├── stepLoader.ts   # STEP file parser
│       └── mujocoExporter.ts # MJCF + STL export
├── train/
│   ├── save_server.py      # Receives exports from web app
│   ├── train.py            # MuJoCo simulation runner
│   ├── requirements.txt    # Python dependencies
│   └── venv/               # Python virtual environment
└── output/                 # Exported files (auto-created)
    ├── model.xml
    └── meshes/
        ├── motor.stl
        └── horn.stl
```

## Mate Types

| Mate Type | Description | Use Case |
|-----------|-------------|----------|
| Cylindrical | Aligns faces + allows rotation | Motor shafts, hinges |
| *(more coming)* | | |

## Troubleshooting

### STEP files not loading
- Ensure `.step` or `.stp` extension
- Check browser console (F12) for errors

### Simulate downloads files instead of exporting
- Start the save server: `./venv/bin/python save_server.py`

### MuJoCo viewer not opening (macOS)
- Use `mjpython` instead of `python`:
  ```bash
  ./venv/bin/mjpython train.py
  ```

### Parts not aligning correctly
- Try different face selections
- Cylindrical mate works best on flat circular faces

## Tech Stack

- **Frontend**: React + TypeScript + Vite + Three.js
- **CAD Import**: occt-import-js (OpenCASCADE)
- **Physics**: MuJoCo (Python)
- **State**: Zustand

## License

MIT
