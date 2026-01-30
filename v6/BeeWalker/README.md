# BeeWalker Simulation

A rigorous MJCF (MuJoCo) simulation of a 6-DOF bipedal robot powered by MG996R servos.

## Structure
- `model.xml`: The MuJoCo robot definition (Iteratively refined to match hardware).
- `web_view.py`: Browser-based viewer with camera tracking and joint testing.
- `simulate.py`: Headless/Passive viewer script.

## Quick Start

1. **Install Dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Web Viewer:**
   ```bash
   python3 web_view.py
   ```
   Open `http://localhost:5000` to see the robot.

## Hardware Specs
- **Servos:** MG996R (Black)
- **Brackets:** Standard U-Brackets (Purple)
- **Geometry:** 65mm leg segments, 25mm x 30mm feet.
