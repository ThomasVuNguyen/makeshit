---
description: How to design and simulate a new robot from STEP files
---

# New Robot Design Workflow

## Prerequisites
Ensure you have:
- Node.js installed
- Python 3.12+ with MuJoCo (`pip install mujoco`)
- STEP files for your robot parts

---

## Step 1: Prepare STEP Files

Place your `.step` or `.stp` files in:
```
public/models/
```

Example structure:
```
public/models/
├── motor.step
├── arm.step
├── gripper.step
└── base.step
```

---

## Step 2: Start the Development Environment

// turbo
### 2a. Start the web app
```bash
npm run dev
```
Web app runs at http://localhost:5173

// turbo
### 2b. Start the save server (in a separate terminal)
```bash
cd train
./venv/bin/python save_server.py
```
Save server runs at http://localhost:3001

---

## Step 3: Assemble Your Robot

1. Open http://localhost:5173 in your browser
2. **Import parts**: Click "Import" to load your STEP files (or they auto-load if configured as defaults)
3. **Enter Mate Mode**: Click "Mate Mode" button
4. **Select mate type**: Choose "Cylindrical Mate" for rotational joints
5. **Click surfaces**: 
   - First click: Select face on first part
   - Second click: Select face on second part
   - Parts will align automatically
6. **Repeat** for all joints in your robot

---

## Step 4: Export and Simulate

1. Click the **"Simulate"** button in the web app
2. This exports:
   - `output/model.xml` - MuJoCo MJCF file
   - `output/meshes/*.stl` - Real mesh geometry for each part

// turbo
3. Run the MuJoCo simulation:
```bash
cd train
./venv/bin/mjpython train.py
```

The MuJoCo viewer will open showing your robot with real geometry!

---

## Step 5: Iterate

- Make changes in the web app
- Click "Simulate" again to re-export
- Re-run `mjpython train.py` to see updates

Each export overwrites the previous files, so you can iterate quickly.

---

## Troubleshooting

### STEP files not loading
- Ensure files have `.step` or `.stp` extension
- Check browser console for errors

### Simulate button downloads files instead of exporting
- Make sure `save_server.py` is running on port 3001

### MuJoCo viewer not opening
- Use `mjpython` instead of `python` on macOS
- Ensure MuJoCo is properly installed: `pip install mujoco`

### Parts not mating correctly
- Try clicking different faces
- Cylindrical mate works best on flat circular faces
