Plan: SolveSpace → MuJoCo PipelineGoal
Build a UI tool to import STEP parts, assemble with constraints, export to MJCF.StackSolveSpace + Python bindings + custom MJCF exporter

The goal of this application is to create an application that:

- import step files
- has a 3D viewer to display the step files
- has capability to assemble those step files like Planar Mate & Cylindrical Mate of traditional CAD software assembly mode
- Can export the whole thing to Mujoco-compatible model
- Run light on RAM under 1GB

User flow:

- Import step files (example 2 parts)
- See the parts and drag them around in a 3d scene
- Click on 'Cylindrical Mate' button
- Hover on faces of the parts, see highlighted face, and click on the face. The face picked will be highlighted clearly
- Do the same on the other part
- Optionally click 'flip' to flip the mate direction
- Click 'Simulate' button and run a mujoco training run on that

