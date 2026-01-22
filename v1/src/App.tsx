import { useState, useEffect } from 'react';
import { Viewport } from './components/Viewport';
import { Toolbar } from './components/Toolbar';
import { PartsList } from './components/PartsList';
import { PropertiesPanel } from './components/PropertiesPanel';
import { SimulationPreview } from './components/SimulationPreview';
import { useAssemblyStore } from './hooks/useAssemblyStore';
import type { JointType } from './types/assembly';
import './App.css';

function App() {
  const [simulating, setSimulating] = useState(false);

  const parts = useAssemblyStore((state) => state.parts);
  const joints = useAssemblyStore((state) => state.joints);
  const setTransformMode = useAssemblyStore((state) => state.setTransformMode);
  const jointCreationMode = useAssemblyStore((state) => state.jointCreationMode);
  const jointCreationParentId = useAssemblyStore((state) => state.jointCreationParentId);
  const pendingJointType = useAssemblyStore((state) => state.pendingJointType);
  const startJointCreation = useAssemblyStore((state) => state.startJointCreation);
  const cancelJointCreation = useAssemblyStore((state) => state.cancelJointCreation);
  const setPendingJointType = useAssemblyStore((state) => state.setPendingJointType);

  const addPart = useAssemblyStore((state) => state.addPart);
  const clearAssembly = useAssemblyStore((state) => state.clearAssembly);

  // Auto-load sample parts
  useEffect(() => {
    // Only load if empty to prevent double loading
    if (useAssemblyStore.getState().parts.length > 0) return;

    const loadSample = async () => {
      try {
        const { loadCADFile } = await import('./lib/loaders');

        // Load Motor
        const motorRes = await fetch('/motor.step');
        const motorBlob = await motorRes.blob();
        const motorFile = new File([motorBlob], 'motor.step', { type: 'application/octet-stream' });
        const motorGeo = await loadCADFile(motorFile);
        addPart(motorGeo, 'motor.step');

        // Load Horn
        const hornRes = await fetch('/horn.step');
        const hornBlob = await hornRes.blob();
        const hornFile = new File([hornBlob], 'horn.step', { type: 'application/octet-stream' });
        const hornGeo = await loadCADFile(hornFile);
        addPart(hornGeo, 'horn.step');

        console.log("Auto-loaded sample parts");
      } catch (err) {
        console.error("Failed to auto-load samples:", err);
      }
    };
    loadSample();
  }, []);

  const parentPart = parts.find((p) => p.id === jointCreationParentId);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) {
        return;
      }

      switch (e.key.toLowerCase()) {
        case 'g':
          if (!jointCreationMode) setTransformMode('translate');
          break;
        case 'r':
          if (!jointCreationMode) setTransformMode('rotate');
          break;
        case 's':
          if (!jointCreationMode) setTransformMode('scale');
          break;
        case 'j':
          if (parts.length >= 2 && !jointCreationMode) {
            startJointCreation();
          }
          break;
        case 'escape':
          if (jointCreationMode) {
            cancelJointCreation();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [parts.length, jointCreationMode, setTransformMode, startJointCreation, cancelJointCreation]);

  return (
    <div className="app">
      <Toolbar onSimulate={() => setSimulating(true)} />

      <div className="main-content">
        <aside className="sidebar left">
          <PartsList />

          {parts.length >= 2 && !jointCreationMode && (
            <button
              className="add-joint-button"
              onClick={() => startJointCreation()}
            >
              🔗 Create Joint
            </button>
          )}

          {/* Existing joints list */}
          {joints.length > 0 && (
            <div className="joints-summary">
              <div className="joints-count">{joints.length} joint{joints.length > 1 ? 's' : ''}</div>
            </div>
          )}
        </aside>

        <main className="viewport-container">
          <Viewport />

          {/* Joint creation mode overlay */}
          {jointCreationMode && (
            <div className="joint-mode-overlay">
              <div className="joint-mode-status">
                <div className="joint-mode-title">
                  🔗 Joint Creation Mode
                </div>
                <div className="joint-mode-instruction">
                  {!jointCreationParentId
                    ? "Click the PARENT part (the base)"
                    : `Parent: ${parentPart?.name} — Now click the CHILD part`
                  }
                </div>
                <div className="joint-mode-type">
                  <label>Type:</label>
                  <select
                    value={pendingJointType}
                    onChange={(e) => setPendingJointType(e.target.value as JointType)}
                  >
                    <option value="hinge">Hinge</option>
                    <option value="slide">Slide</option>
                    <option value="cylindrical">Cylindrical</option>
                    <option value="ball">Ball</option>
                    <option value="fixed">Fixed</option>
                  </select>
                </div>
                <button className="cancel-joint-btn" onClick={cancelJointCreation}>
                  Cancel (Esc)
                </button>
              </div>
            </div>
          )}
        </main>

        <aside className="sidebar right">
          <PropertiesPanel />
        </aside>
      </div>

      {/* Keyboard shortcuts hint */}
      {!jointCreationMode && (
        <div className="shortcuts-hint">
          <span>G: Move</span>
          <span>R: Rotate</span>
          <span>S: Scale</span>
          <span>J: Joint</span>
        </div>
      )}

      {/* Simulation Overlay */}
      {simulating && (
        <SimulationPreview onClose={() => setSimulating(false)} />
      )}
    </div>
  );
}

export default App;
