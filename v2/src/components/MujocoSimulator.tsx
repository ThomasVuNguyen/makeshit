import { useEffect, useRef, useState, useCallback } from 'react'
import loadMujoco from 'mujoco-js'
import './MujocoSimulator.css'

interface MujocoSimulatorProps {
    mjcf: string
    onClose: () => void
}

// Create a simple test model that works without external meshes
function createSimpleMjcf(hasMate: boolean): string {
    if (hasMate) {
        return `
<mujoco model="servo_assembly">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  
  <worldbody>
    <light name="top" pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.3 0.3 0.3 1"/>
    
    <!-- Motor body (fixed) -->
    <body name="motor" pos="0 0 0.1">
      <geom name="motor_body" type="box" size="0.03 0.02 0.04" rgba="0.3 0.3 0.3 1"/>
      <geom name="motor_shaft" type="cylinder" size="0.005 0.02" pos="0 0 0.06" rgba="0.5 0.5 0.5 1"/>
      
      <!-- Horn attached to motor with revolute joint -->
      <body name="horn" pos="0 0 0.08">
        <joint name="horn_joint" type="hinge" axis="0 0 1" damping="0.1"/>
        <geom name="horn_arm" type="capsule" size="0.003" fromto="0 0 0 0.04 0 0" rgba="0.9 0.9 0.9 1"/>
        <geom name="horn_center" type="cylinder" size="0.008 0.003" rgba="0.9 0.9 0.9 1"/>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="horn_motor" joint="horn_joint" gear="0.1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>`
    } else {
        return `
<mujoco model="free_parts">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  
  <worldbody>
    <light name="top" pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.3 0.3 0.3 1"/>
    
    <!-- Motor body -->
    <body name="motor" pos="0 0 0.1">
      <geom name="motor_body" type="box" size="0.03 0.02 0.04" rgba="0.3 0.3 0.3 1"/>
    </body>
    
    <!-- Horn (free) -->
    <body name="horn" pos="0.1 0 0.05">
      <freejoint name="horn_free"/>
      <geom name="horn_arm" type="capsule" size="0.003" fromto="0 0 0 0.04 0 0" rgba="0.9 0.9 0.9 1"/>
    </body>
  </worldbody>
</mujoco>`
    }
}

export function MujocoSimulator({ mjcf, onClose }: MujocoSimulatorProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const [status, setStatus] = useState<'loading' | 'running' | 'error'>('loading')
    const [error, setError] = useState<string | null>(null)
    const [simTime, setSimTime] = useState(0)
    const animationRef = useRef<number>(0)
    const mujocoRef = useRef<any>(null)
    const modelRef = useRef<any>(null)
    const dataRef = useRef<any>(null)

    // Initialize MuJoCo
    useEffect(() => {
        let mounted = true

        async function initMujoco() {
            try {
                setStatus('loading')
                console.log('Loading MuJoCo WASM...')

                const mujoco = await loadMujoco()
                if (!mounted) return

                mujocoRef.current = mujoco
                console.log('MuJoCo loaded successfully')

                // Mount virtual file system
                const FS = (mujoco as any).FS
                try {
                    FS.mkdir('/working')
                } catch (e) {
                    // Directory might already exist
                }
                try {
                    FS.mount((mujoco as any).MEMFS, { root: '.' }, '/working')
                } catch (e) {
                    // Already mounted
                }

                // Determine if we have mates (check for "revolute" or "joint" in MJCF)
                const hasMate = mjcf.includes('revolute') || mjcf.includes('hinge')

                // Use simplified model for browser simulation
                const simpleMjcf = createSimpleMjcf(hasMate)
                console.log('Using simplified MJCF for simulation:', simpleMjcf)

                FS.writeFile('/working/model.xml', simpleMjcf)

                // Load model
                console.log('Loading model from XML...')
                const model = mujoco.MjModel.loadFromXML('/working/model.xml')
                if (!model) {
                    throw new Error('Failed to load model from XML')
                }
                modelRef.current = model

                // Create simulation data
                const data = new mujoco.MjData(model)
                if (!data) {
                    throw new Error('Failed to create MjData')
                }
                dataRef.current = data

                console.log('MuJoCo simulation initialized!')
                console.log('Model has', model.nbody, 'bodies')
                setStatus('running')

            } catch (err) {
                console.error('MuJoCo initialization error:', err)
                setError(err instanceof Error ? err.message : 'Unknown error')
                setStatus('error')
            }
        }

        initMujoco()

        return () => {
            mounted = false
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current)
            }
            try {
                dataRef.current?.delete()
                modelRef.current?.delete()
            } catch (e) {
                // Ignore cleanup errors
            }
        }
    }, [mjcf])

    // Run simulation loop
    useEffect(() => {
        if (status !== 'running') return

        const mujoco = mujocoRef.current
        const model = modelRef.current
        const data = dataRef.current
        const canvas = canvasRef.current

        if (!mujoco || !model || !data || !canvas) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        let lastTime = performance.now()
        let controlTime = 0

        function step() {
            const now = performance.now()
            const dt = (now - lastTime) / 1000
            lastTime = now
            controlTime += dt

            // Apply oscillating control to motor (if exists)
            try {
                const nCtrl = model.nu
                if (nCtrl > 0) {
                    // Oscillate the motor control
                    const ctrl = Math.sin(controlTime * 2) * 0.5
                    data.ctrl[0] = ctrl
                }
            } catch (e) {
                // Ignore control errors
            }

            // Step simulation
            try {
                const steps = Math.min(Math.floor(dt / 0.002), 10)
                for (let i = 0; i < Math.max(1, steps); i++) {
                    mujoco.mj_step(model, data)
                }
            } catch (e) {
                console.error('Simulation step error:', e)
            }

            setSimTime(data.time)
            renderSimulation(ctx, model, data, mujoco)

            animationRef.current = requestAnimationFrame(step)
        }

        animationRef.current = requestAnimationFrame(step)

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current)
            }
        }
    }, [status])

    // 2D visualization
    const renderSimulation = useCallback((ctx: CanvasRenderingContext2D, model: any, data: any, mujoco: any) => {
        const width = ctx.canvas.width
        const height = ctx.canvas.height

        // Clear
        ctx.fillStyle = '#1a1a2e'
        ctx.fillRect(0, 0, width, height)

        // Ground
        ctx.fillStyle = '#333366'
        ctx.fillRect(0, height - 60, width, 60)

        // Grid lines
        ctx.strokeStyle = '#444488'
        ctx.lineWidth = 1
        for (let x = 0; x < width; x += 50) {
            ctx.beginPath()
            ctx.moveTo(x, 0)
            ctx.lineTo(x, height)
            ctx.stroke()
        }

        const scale = 1500 // pixels per meter
        const centerX = width / 2
        const groundY = height - 60

        // Draw bodies
        try {
            const nBodies = model.nbody

            for (let i = 1; i < nBodies; i++) {
                // Get body position (xpos is flat array: x,y,z for each body)
                const xIdx = i * 3
                const x = data.xpos[xIdx]
                const z = data.xpos[xIdx + 2]

                const canvasX = centerX + x * scale
                const canvasY = groundY - z * scale

                // Color based on body index
                const colors = ['#4a90d9', '#e07070', '#70e070', '#e0e070']
                ctx.fillStyle = colors[(i - 1) % colors.length]

                // Draw body as rectangle/circle
                if (i === 1) {
                    // Motor - draw as box
                    ctx.fillRect(canvasX - 30, canvasY - 40, 60, 80)
                    ctx.strokeStyle = '#ffffff'
                    ctx.lineWidth = 2
                    ctx.strokeRect(canvasX - 30, canvasY - 40, 60, 80)

                    // Shaft
                    ctx.fillStyle = '#666'
                    ctx.fillRect(canvasX - 8, canvasY - 60, 16, 20)
                } else {
                    // Horn - draw as arm
                    ctx.save()
                    ctx.translate(canvasX, canvasY - 60)

                    // Get rotation from xquat if available
                    try {
                        const qIdx = i * 4
                        const qw = data.xquat[qIdx]
                        const qz = data.xquat[qIdx + 3]
                        const angle = 2 * Math.atan2(qz, qw)
                        ctx.rotate(angle)
                    } catch (e) {
                        // Ignore rotation errors
                    }

                    ctx.fillStyle = '#e0e0e0'
                    ctx.fillRect(0, -5, 60, 10)
                    ctx.beginPath()
                    ctx.arc(0, 0, 12, 0, Math.PI * 2)
                    ctx.fill()
                    ctx.strokeStyle = '#aaa'
                    ctx.lineWidth = 2
                    ctx.stroke()
                    ctx.restore()
                }
            }
        } catch (e) {
            // Ignore render errors
        }

        // Info overlay
        ctx.fillStyle = '#ffffff'
        ctx.font = '14px monospace'
        ctx.textAlign = 'left'
        ctx.fillText(`Time: ${data.time.toFixed(3)}s`, 10, 25)
        ctx.fillText(`Bodies: ${model.nbody}`, 10, 45)

        // Draw legend
        ctx.fillStyle = '#888'
        ctx.font = '12px sans-serif'
        ctx.fillText('Horn oscillates automatically', 10, height - 20)
    }, [])

    return (
        <div className="mujoco-overlay">
            <div className="mujoco-modal">
                <div className="mujoco-header">
                    <h2>🔬 MuJoCo Simulation</h2>
                    <button className="mujoco-close" onClick={onClose}>✕</button>
                </div>

                <div className="mujoco-content">
                    {status === 'loading' && (
                        <div className="mujoco-loading">
                            <div className="spinner"></div>
                            <p>Loading MuJoCo WASM...</p>
                        </div>
                    )}

                    {status === 'error' && (
                        <div className="mujoco-error">
                            <p>❌ Error: {error}</p>
                            <pre style={{ fontSize: '12px', maxWidth: '400px', overflow: 'auto' }}>{error}</pre>
                            <button onClick={onClose}>Close</button>
                        </div>
                    )}

                    {status === 'running' && (
                        <>
                            <canvas
                                ref={canvasRef}
                                width={600}
                                height={400}
                                className="mujoco-canvas"
                            />
                            <div className="mujoco-info">
                                <span>⏱ Sim Time: {simTime.toFixed(3)}s</span>
                                <span style={{ marginLeft: '20px' }}>🔄 Real-time physics simulation</span>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}
