
import { useEffect, useRef, useState, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import loadMujoco from 'mujoco-js';
import { useAssemblyStore } from '../hooks/useAssemblyStore';
import { exportToMJCF } from '../lib/mjcfExporter';
import type { Part } from '../types/assembly';

// Helper to sanitize names exactly as the exporter does
const sanitizeName = (name: string) => name.replace(/\s+/g, '_');

interface SimPartProps {
    part: Part;
    model: any;
    data: any;
    bodyId: number;
}

function SimPart({ part, model, data, bodyId }: SimPartProps) {
    const meshRef = useRef<THREE.Mesh>(null);

    useFrame(() => {
        if (!model || !data || !meshRef.current || bodyId === -1) return;

        // MuJoCo data access
        // xpos is 3 * nbody
        const xpos = data.xpos;
        const xquat = data.xquat;

        // Update Position
        meshRef.current.position.set(
            xpos[bodyId * 3 + 0],
            xpos[bodyId * 3 + 1],
            xpos[bodyId * 3 + 2]
        );

        // Update Rotation
        meshRef.current.quaternion.set(
            xquat[bodyId * 4 + 1], // x
            xquat[bodyId * 4 + 2], // y
            xquat[bodyId * 4 + 3], // z
            xquat[bodyId * 4 + 0]  // w
        );
    });

    return (
        <mesh
            ref={meshRef}
            geometry={part.geometry}
            scale={part.scale}
        >
            <meshStandardMaterial color={part.color} />
        </mesh>
    );
}

export function SimulationPreview({ onClose }: { onClose: () => void }) {
    const parts = useAssemblyStore(state => state.parts);
    const joints = useAssemblyStore(state => state.joints);
    const name = useAssemblyStore(state => state.name);

    const [mujoco, setMujoco] = useState<any>(null);
    const [model, setModel] = useState<any>(null);
    const [data, setData] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    // Initialize MuJoCo WASM
    useEffect(() => {
        const init = async () => {
            try {
                console.log("Loading MuJoCo...");
                const _mujoco = await loadMujoco();
                console.log("MuJoCo Loaded:", _mujoco);

                // Initialize FS and allocate memory
                if (_mujoco.FS) {
                    _mujoco.FS.mkdir('/working');
                    _mujoco.FS.mount(_mujoco.MEMFS, { root: '.' }, '/working');
                }

                setMujoco(_mujoco);
            } catch (err: any) {
                console.error("Failed to load MuJoCo WASM:", err);
                setError(`Failed to load Physics Engine: ${err.message}`);
            } finally {
                setLoading(false);
            }
        };
        init();
    }, []);

    // Load Model when ready
    useEffect(() => {
        if (!mujoco) return;

        // 1. Export current assembly to MJCF
        const { xml, meshFiles } = exportToMJCF(name, parts, joints);

        try {
            // 2. Write assets to virtual FS
            // Need to clean up previous optional?

            // Create meshes directory
            try { mujoco.FS.mkdir('/working/meshes'); } catch (e) { /* ignore exist */ }

            // Write meshes
            meshFiles.forEach((buffer, filename) => {
                mujoco.FS.writeFile(`/working/meshes/${filename}`, new Uint8Array(buffer));
            });

            // Write XML
            mujoco.FS.writeFile('/working/model.xml', xml);

            // 3. Load XML
            const validXML = '/working/model.xml';
            const errorBuf = new Uint8Array(1000);

            // Free previous
            if (model) mujoco.mj_deleteModel(model);
            if (data) mujoco.mj_deleteData(data);

            const _model = mujoco.mj_loadXML(validXML, 0, errorBuf, 1000);

            if (!_model) {
                const errStr = new TextDecoder().decode(errorBuf).replace(/\0/g, '');
                throw new Error(errStr || "Unknown MuJoCo Load Error");
            }

            const _data = mujoco.mj_makeData(_model);

            setModel(_model);
            setData(_data);

        } catch (err: any) {
            console.error("Simulation Setup Error:", err);
            setError(err.message);
        }

        return () => {
            // Cleanup if unmounting (handled by state setting null but we should be careful with memory)
            if (model) mujoco.mj_deleteModel(model);
            if (data) mujoco.mj_deleteData(data);
        }
    }, [mujoco, parts, joints, name]);

    // Mapping from part ID to Body ID
    const partBodyIds = useMemo(() => {
        if (!model || !mujoco) return new Map<string, number>();

        const map = new Map<string, number>();
        parts.forEach(part => {
            const bodyName = sanitizeName(part.name);
            const id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bodyName);
            map.set(part.id, id);
        });
        return map;
    }, [model, mujoco, parts]);

    // Physics Loop
    useFrame((_, delta) => {
        if (model && data && mujoco) {
            // sub-steps for stability? 
            // Standard mj_step
            mujoco.mj_step(model, data);
        }
    });

    return (
        <div style={{
            position: 'absolute',
            top: 0, left: 0, width: '100%', height: '100%',
            zIndex: 2000,
            background: 'rgba(0, 0, 0, 0.9)',
            display: 'flex',
            flexDirection: 'column'
        }}>
            <div style={{
                padding: '16px',
                background: '#222',
                color: 'white',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <h3 style={{ margin: 0 }}>Physics Simulation</h3>
                {error && <span style={{ color: '#ff4444' }}>{error}</span>}
                <button onClick={onClose} style={{ padding: '8px 16px', background: '#555', color: 'white', border: 'none', cursor: 'pointer' }}>
                    Close
                </button>
            </div>

            <div style={{ flex: 1, position: 'relative' }}>
                <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
                    <ambientLight intensity={0.5} />
                    <directionalLight position={[10, 10, 5]} intensity={1} />
                    <Grid infiniteGrid />
                    <OrbitControls makeDefault />

                    {/* Simulation Scene */}
                    {model && data && parts.map(part => (
                        <SimPart
                            key={part.id}
                            part={part}
                            model={model}
                            data={data}
                            bodyId={partBodyIds.get(part.id) ?? -1}
                        />
                    ))}
                </Canvas>

                {loading && (
                    <div style={{
                        position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                        color: 'white', fontSize: '20px'
                    }}>
                        Loading Physics Engine...
                    </div>
                )}
            </div>
        </div>
    );
}
