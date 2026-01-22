# MakeShit

**MakeShit** is a modern, intuitive web-based CAD tool designed for rapidly assembling and configuring robot mechanisms for **MuJoCo** simulation.

> **Simplify your simulation workflow.**  
> Drag, drop, align, and export directly to MJCF.

## 🚀 Features

- **Intuitive Assembly**: 
  - Drag-and-drop parts (STL/Mesh support).
  - **Smart Alignment**: Easily align faces, edges, and axes.
  - **Snap-to-Joint**: Create Hinge, Slide, Ball, and specialized Cylindrical joints instantly.

- **Physics-Aware Tooling**:
  - **Kinematic Preview**: Move parts and see connected bodies follow.
  - **Joint Visualization**: Inspect joint axes and limits directly in the 3D viewport.

- **Seamless Export**:
  - **One-Click MJCF**: Generates clean, ready-to-simulate `.xml` files.
  - **Embedded Meshes**: Automatically handles geometry conversion and referencing.

## 🛠️ Tech Stack

- **Core**: React 19, TypeScript, Vite
- **3D Engine**: Three.js, @react-three/fiber, @react-three/drei
- **State Management**: Zustand + Zundo (Undo/Redo)
- **Physics**: (Planned) WASM-based validation

## 📦 Getting Started

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Run Development Server**
   ```bash
   npm run dev
   ```

3. **Build for Production**
   ```bash
   npm run build
   ```

## 🤝 Contributing

This project is a work in progress. Feel free to open issues or PRs for new joint types, exporters, or UI improvements.

---

*Built with ❤️ for the Robotics Community.*
