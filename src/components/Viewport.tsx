import { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import { OrbitControls, TransformControls, Grid, GizmoHelper, GizmoViewport, Html } from '@react-three/drei';
import * as THREE from 'three';
import { useAssemblyStore } from '../hooks/useAssemblyStore';
import { detectSurfaceType } from '../lib/geometryInference';
import type { Part, Joint } from '../types/assembly';

// Calculate preview transform for a child part based on joint settings
function getJointPreviewTransform(
    joint: Joint,
    parentPart: Part,
    childPart: Part
): { position: [number, number, number]; rotation: [number, number, number] } {
    const previewValue = joint.previewValue;
    const range = joint.limitUpper - joint.limitLower;
    const currentValue = joint.limitLower + range * previewValue;

    // If pivot is not set, show child at its original position
    if (!joint.pivotSet) {
        return {
            position: childPart.position,
            rotation: childPart.rotation,
        };
    }

    // Pivot is set - rotate/slide around the anchor point
    const axisIndex = { x: 0, y: 1, z: 2 }[joint.axis];

    // Anchor position in world space (parent position + anchor offset)
    const pivotWorld: [number, number, number] = [
        parentPart.position[0] + joint.anchorPosition[0],
        parentPart.position[1] + joint.anchorPosition[1],
        parentPart.position[2] + joint.anchorPosition[2],
    ];

    if (joint.type === 'hinge') {
        // Rotate child around the pivot point
        const rotation: [number, number, number] = [...childPart.rotation];
        rotation[axisIndex] = currentValue;

        // For a hinge, the position stays at the pivot
        return {
            position: pivotWorld,
            rotation,
        };
    } else if (joint.type === 'slide' || joint.type === 'cylindrical') {
        // Slide along the axis from the pivot
        const position: [number, number, number] = [...pivotWorld];
        position[axisIndex] += currentValue;

        return {
            position,
            rotation: childPart.rotation,
        };
    }

    // Ball joint or fixed - just show at pivot
    return {
        position: pivotWorld,
        rotation: childPart.rotation,
    };
}

// Individual part mesh component
function PartMesh({ part, jointPreview }: { part: Part; jointPreview?: { position: [number, number, number]; rotation: [number, number, number] } }) {
    const meshRef = useRef<THREE.Mesh>(null);
    const {
        selectedPartId,
        selectPart,
        updatePartPosition,
        updatePartRotation,
        transformMode,
        jointCreationMode,
        jointCreationParentId,
        handlePartClickForJoint,
        alignMode,
        alignType,
        alignStep,
        alignPoints,
        addAlignPoint,
        setAlignTargetPoint,
        applyAlignment,
        applyAxisAlignment,
        commitAlignAxis,
        previewAxisAlignment,
    } = useAssemblyStore();

    const isSelected = selectedPartId === part.id;
    const isJointParent = jointCreationParentId === part.id;

    const handleClick = (e: ThreeEvent<MouseEvent>) => {
        e.stopPropagation();

        if (alignMode) {
            const point: [number, number, number] = [e.point.x, e.point.y, e.point.z];

            // 1. Point Mode
            if (alignType === 'point') {
                if (alignStep === 'target') {
                    setAlignTargetPoint(point, part.id);
                } else if (alignStep === 'source') {
                    applyAlignment(point, part.id);
                }

                // 2. Smart Cylinder/Plane Mode
            } else if (alignType === 'cylinder') {
                const geometry = (e.object as THREE.Mesh).geometry;
                const brepFaces = geometry.userData.brepFaces;

                if (brepFaces && e.faceIndex !== undefined) {
                    const bFace = brepFaces.find((f: any) => e.faceIndex! >= f.first && e.faceIndex! <= f.last);
                    if (bFace) {
                        const surface = detectSurfaceType(geometry, bFace.first, bFace.last);

                        // Check valid surface type
                        if (surface.type === 'cylinder' || surface.type === 'plane') {
                            const worldMatrix = e.object.matrixWorld;
                            let axis = new THREE.Vector3();
                            let center = new THREE.Vector3();

                            if (surface.type === 'cylinder') {
                                axis.copy(surface.axis!).transformDirection(worldMatrix).normalize();
                                center.copy(surface.origin!).applyMatrix4(worldMatrix);
                            } else {
                                axis.copy(surface.normal!).transformDirection(worldMatrix).normalize();
                                center.copy(e.point);
                            }

                            if (alignStep === 'target') {
                                // Set Persistent Highlight for Target
                                const tempSubsetGeo = new THREE.BufferGeometry();
                                tempSubsetGeo.setAttribute('position', geometry.getAttribute('position'));
                                const indices = [];
                                const indexAttr = geometry.index;
                                if (indexAttr) {
                                    for (let i = bFace.first * 3; i < (bFace.last + 1) * 3; i++) {
                                        indices.push(indexAttr.getX(i));
                                    }
                                }
                                tempSubsetGeo.setIndex(indices);
                                // Store in global store for persistent rendering
                                useAssemblyStore.getState().setAlignTargetHighlight(tempSubsetGeo);

                                commitAlignAxis({ center: center.toArray(), normal: axis.toArray(), partId: part.id });
                            } else {
                                previewAxisAlignment({ center: center.toArray(), normal: axis.toArray() }, part.id);
                            }
                        }
                    }
                }

                // 3. Axis (2-Point Rim) Mode - Fallback/Manual
            } else if (alignType === 'axis') {
                const pointNormal = e.face?.normal.clone().applyQuaternion(e.object.quaternion).toArray() || [0, 1, 0];

                if (alignStep === 'source' && alignPoints.length === 1) {
                    const p1Obj = alignPoints[0] as unknown as { point: [number, number, number], normal: [number, number, number] };
                    const p1 = new THREE.Vector3(...p1Obj.point);
                    const n1 = new THREE.Vector3(...p1Obj.normal);
                    const p2 = new THREE.Vector3(...point);
                    const center = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
                    const axisNormal = n1.normalize();
                    applyAxisAlignment({ center: center.toArray(), normal: axisNormal.toArray() }, part.id);
                } else {
                    addAlignPoint(point, pointNormal, part.id);
                }
            }
            return;
        }

        if (jointCreationMode) {
            handlePartClickForJoint(part.id);
        } else {
            selectPart(part.id);
        }
    };

    // Determine material color based on state
    let emissiveColor = '#000';
    const hasPreview = !!jointPreview;

    if (jointCreationMode) {
        if (isJointParent) {
            emissiveColor = '#0a5';
        } else if (jointCreationParentId) {
            emissiveColor = '#05a';
        }
    } else if (hasPreview) {
        emissiveColor = '#530'; // Orange glow for previewing part
    } else if (isSelected) {
        emissiveColor = '#444';
    }

    // Use preview transform if available, otherwise use part's stored transform
    const displayPosition = jointPreview?.position ?? part.position;
    const displayRotation = jointPreview?.rotation ?? part.rotation;

    // Hover State for Highlighting
    const [highlightGeo, setHighlightGeo] = useState<THREE.BufferGeometry | null>(null);

    const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();

        // Highlighting for Cylindrical/Plane align mode
        if (!alignMode || alignType !== 'cylinder') {
            if (highlightGeo) setHighlightGeo(null);
            return;
        }

        const geometry = (e.object as THREE.Mesh).geometry;
        const brepFaces = geometry.userData.brepFaces;

        if (brepFaces && e.faceIndex !== undefined) {
            const bFace = brepFaces.find((f: any) => e.faceIndex! >= f.first && e.faceIndex! <= f.last);
            if (bFace) {
                // Check if cylindrical or plane before highlighting?
                const surface = detectSurfaceType(geometry, bFace.first, bFace.last);
                if (surface.type !== 'cylinder' && surface.type !== 'plane') {
                    setHighlightGeo(null);
                    return;
                }

                // Create highlight geometry subset
                if (!highlightGeo || highlightGeo.userData.faceId !== bFace.first) {
                    const subsetGeo = new THREE.BufferGeometry();
                    subsetGeo.setAttribute('position', geometry.getAttribute('position'));

                    // Create indices for the range
                    const indices = [];
                    const indexAttr = geometry.index;
                    if (indexAttr) {
                        for (let i = bFace.first * 3; i < (bFace.last + 1) * 3; i++) {
                            indices.push(indexAttr.getX(i));
                        }
                    }
                    subsetGeo.setIndex(indices);
                    subsetGeo.userData.faceId = bFace.first; // Cache key
                    setHighlightGeo(subsetGeo);
                }
                return;
            }
        }
        setHighlightGeo(null);
    };

    return (
        <>
            <mesh
                ref={meshRef}
                geometry={part.geometry}
                position={displayPosition}
                rotation={displayRotation}
                scale={part.scale}
                visible={part.visible}
                onClick={handleClick}
                onPointerMove={handlePointerMove}
                onPointerOut={() => setHighlightGeo(null)}
            >
                <meshStandardMaterial
                    color={part.color}
                    emissive={emissiveColor}
                    metalness={0.3}
                    roughness={0.7}
                />

                {/* Highlight Overlay (Hover) */}
                {highlightGeo && (
                    <mesh geometry={highlightGeo}>
                        <meshBasicMaterial
                            color="#ffff00"
                            transparent
                            opacity={0.3}
                            depthTest={false}
                            depthWrite={false}
                            side={THREE.DoubleSide}
                        />
                    </mesh>
                )}
            </mesh>

            {isSelected && !jointCreationMode && !hasPreview && meshRef.current && (
                <TransformControls
                    object={meshRef.current}
                    mode={transformMode}
                    onObjectChange={() => {
                        if (meshRef.current) {
                            const pos = meshRef.current.position;
                            const rot = meshRef.current.rotation;
                            updatePartPosition(part.id, [pos.x, pos.y, pos.z]);
                            updatePartRotation(part.id, [rot.x, rot.y, rot.z]);
                        }
                    }}
                />
            )}
        </>
    );
}

// Scene setup component
function Scene() {
    const parts = useAssemblyStore((state) => state.parts);
    const joints = useAssemblyStore((state) => state.joints);
    const selectPart = useAssemblyStore((state) => state.selectPart);
    const jointCreationMode = useAssemblyStore((state) => state.jointCreationMode);
    const cancelJointCreation = useAssemblyStore((state) => state.cancelJointCreation);

    // Undo/Redo Listener
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.ctrlKey && e.key === 'z') {
                e.preventDefault();
                useAssemblyStore.temporal.getState().undo();
            } else if (e.ctrlKey && e.key === 'y') {
                e.preventDefault();
                useAssemblyStore.temporal.getState().redo();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);


    // Alignment state
    const alignMode = useAssemblyStore((state) => state.alignMode);
    const alignStep = useAssemblyStore((state) => state.alignStep);
    const alignType = useAssemblyStore((state) => state.alignType);
    const alignTargetPoint = useAssemblyStore((state) => state.alignTargetPoint);
    const alignPoints = useAssemblyStore((state) => state.alignPoints);
    const alignTargetAxis = useAssemblyStore((state) => state.alignTargetAxis);
    const alignTargetHighlight = useAssemblyStore((state) => state.alignTargetHighlight);
    const alignFlip = useAssemblyStore((state) => state.alignFlip);

    // Actions
    const cancelAlignMode = useAssemblyStore((state) => state.cancelAlignMode);
    const confirmAlignment = useAssemblyStore((state) => state.confirmAlignment);
    const toggleAlignFlip = useAssemblyStore((state) => state.toggleAlignFlip);

    // Calculate joint previews for child parts
    const jointPreviews = useMemo(() => {
        const previews = new Map<string, { position: [number, number, number]; rotation: [number, number, number] }>();

        for (const joint of joints) {
            if (joint.type === 'fixed') continue; // No preview for fixed joints

            const parentPart = parts.find((p) => p.id === joint.parentId);
            const childPart = parts.find((p) => p.id === joint.childId);
            if (!parentPart || !childPart) continue;

            const preview = getJointPreviewTransform(joint, parentPart, childPart);
            previews.set(joint.childId, preview);
        }

        return previews;
    }, [joints, parts]);

    const handleGroundClick = () => {
        if (jointCreationMode) {
            cancelJointCreation();
        } else if (alignMode) {
            cancelAlignMode();
        } else {
            selectPart(null);
        }
    };

    return (
        <>
            {/* Lighting */}
            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
            <directionalLight position={[-10, -10, -5]} intensity={0.3} />

            {/* Grid */}
            <Grid
                args={[20, 20]}
                cellSize={0.5}
                cellThickness={0.5}
                cellColor={jointCreationMode ? "#4a6e4a" : "#6e6e6e"}
                sectionSize={2}
                sectionThickness={1}
                sectionColor={jointCreationMode ? "#4b9d4b" : "#9d4b4b"}
                fadeDistance={30}
                fadeStrength={1}
                followCamera={false}
                infiniteGrid
            />

            {/* Ground plane for click-to-deselect/cancel */}
            <mesh
                rotation={[-Math.PI / 2, 0, 0]}
                position={[0, -0.01, 0]}
                onClick={handleGroundClick}
            >
                <planeGeometry args={[100, 100]} />
                <meshBasicMaterial visible={false} />
            </mesh>

            {alignMode && alignType === 'point' && alignTargetPoint && (
                <mesh position={alignTargetPoint.point}>
                    <sphereGeometry args={[0.2, 16, 16]} />
                    <meshBasicMaterial color="#00ff00" depthTest={false} transparent opacity={0.8} />
                </mesh>
            )}

            {/* Visual Markers for Axis Alignment */}
            {alignMode && alignType === 'axis' && (
                <>
                    {/* Draw collected points */}
                    {alignPoints.map((p: any, i) => (
                        <mesh key={i} position={p.point || p}>
                            <sphereGeometry args={[0.05, 8, 8]} />
                            <meshBasicMaterial color="#ffff00" depthTest={false} />
                        </mesh>
                    ))}

                    {/* Draw Target Axis if defined */}
                    {alignTargetAxis && (
                        <group position={alignTargetAxis.center}>
                            {/* Represents the axis vector (using a cylinder aligned to normal) */}
                            <mesh quaternion={new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...alignTargetAxis.normal))}>
                                <cylinderGeometry args={[0.02, 0.02, 2, 8]} />
                                <meshBasicMaterial color="#00ffff" transparent opacity={0.5} depthTest={false} />
                            </mesh>
                            <mesh quaternion={new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...alignTargetAxis.normal))}>
                                <ringGeometry args={[0.1, 0.12, 32]} />
                                <meshBasicMaterial color="#00ffff" side={THREE.DoubleSide} depthTest={false} />
                            </mesh>
                        </group>
                    )}
                </>
            )}

            {/* Persistent Selection Highlight (Green) */}
            {alignTargetHighlight && (
                <mesh geometry={alignTargetHighlight}>
                    <meshBasicMaterial
                        color="#00ff00"
                        transparent
                        opacity={0.5}
                        depthTest={false}
                        depthWrite={false}
                        side={THREE.DoubleSide}
                    />
                </mesh>
            )}

            {/* Render parts with joint previews */}
            {parts.map((part) => (
                <PartMesh
                    key={part.id}
                    part={part}
                    jointPreview={jointPreviews.get(part.id)}
                />
            ))}

            {/* Camera controls */}
            <OrbitControls makeDefault />

            {/* Gizmo helper */}
            <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
                <GizmoViewport labelColor="white" axisHeadScale={1} />
            </GizmoHelper>

            {/* Confirmation Overlay within 3D View (Using HTML) */}
            {alignMode && alignStep === 'confirm' && (
                <Html position={[0, 0, 0]} style={{ pointerEvents: 'none' }} calculatePosition={() => [window.innerWidth / 2, window.innerHeight * 0.8]}>
                    <div style={{
                        pointerEvents: 'auto',
                        background: 'rgba(0, 0, 0, 0.8)',
                        padding: '16px',
                        borderRadius: '8px',
                        border: '1px solid #444',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '8px',
                        alignItems: 'center',
                        transform: 'translate(-50%, -100%)',
                        minWidth: '200px'
                    }}>
                        <h3 style={{ margin: 0, color: 'white', fontSize: '14px' }}>Confirm Alignment</h3>
                        <div style={{ display: 'flex', gap: '8px' }}>
                            <button
                                onClick={toggleAlignFlip}
                                style={{
                                    padding: '6px 12px',
                                    background: alignFlip ? '#555' : '#333',
                                    color: 'white',
                                    border: '1px solid #666',
                                    borderRadius: '4px',
                                    cursor: 'pointer'
                                }}
                            >
                                {alignFlip ? '🔃 Flipped' : '🔃 Flip'}
                            </button>
                            <button
                                onClick={confirmAlignment}
                                style={{
                                    padding: '6px 12px',
                                    background: '#28a745',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '4px',
                                    cursor: 'pointer',
                                    fontWeight: 'bold'
                                }}
                            >
                                ✓ Check
                            </button>
                            <button
                                onClick={cancelAlignMode}
                                style={{
                                    padding: '6px 12px',
                                    background: '#dc3545',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '4px',
                                    cursor: 'pointer'
                                }}
                            >
                                ✕ Cancel
                            </button>
                        </div>
                    </div>
                </Html>
            )}
        </>
    );
}

// Main viewport component
export function Viewport() {
    return (
        <div style={{ width: '100%', height: '100%', background: '#1a1a2e' }}>
            <Canvas
                camera={{ position: [5, 5, 5], fov: 50 }}
                shadows
                gl={{ antialias: true }}
            >
                <Scene />
            </Canvas>
        </div>
    );
}
