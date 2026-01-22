import { useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment, Grid } from '@react-three/drei'
import { Viewer } from './components/Viewer'
import { ControlPanel } from './components/ControlPanel'
import { MujocoSimulator } from './components/MujocoSimulator'

function App() {
  const [showSimulator, setShowSimulator] = useState(false)
  const [mjcfContent, setMjcfContent] = useState<string | null>(null)

  // Listen for simulate event from Viewer
  useEffect(() => {
    const handleSimulate = (e: CustomEvent<string>) => {
      setMjcfContent(e.detail)
      setShowSimulator(true)
    }

    window.addEventListener('startSimulation', handleSimulate as EventListener)
    return () => window.removeEventListener('startSimulation', handleSimulate as EventListener)
  }, [])

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Canvas camera={{ position: [50, 50, 50], fov: 50 }}>
        <color attach="background" args={['#1a1a1a']} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
        <Viewer />
        <OrbitControls makeDefault />
        <Environment preset="city" />
        <Grid infiniteGrid cellSize={1} sectionSize={10} fadeDistance={100} />
      </Canvas>

      {/* Header */}
      <div style={{ position: 'absolute', top: 20, left: 20, pointerEvents: 'none', color: 'white' }}>
        <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 'bold' }}>Makeshit V2</h1>
        <p style={{ margin: 0, opacity: 0.7 }}>STEP Assembler & MuJoCo Exporter</p>
      </div>

      {/* Control Panel */}
      <ControlPanel />

      {/* MuJoCo Simulation Modal */}
      {showSimulator && mjcfContent && (
        <MujocoSimulator
          mjcf={mjcfContent}
          onClose={() => {
            setShowSimulator(false)
            setMjcfContent(null)
          }}
        />
      )}
    </div>
  )
}

export default App
