import { create } from 'zustand';
import { temporal, type TemporalState } from 'zundo';

import { v4 as uuidv4 } from 'uuid';
import * as THREE from 'three';
import type { Part, Joint, JointType } from '../types/assembly';

interface AssemblyState {
    // State
    name: string;
    parts: Part[];
    joints: Joint[];
    selectedPartId: string | null;
    selectedJointId: string | null;
    transformMode: 'translate' | 'rotate' | 'scale';

    // Joint creation mode
    jointCreationMode: boolean;
    jointCreationParentId: string | null;
    pendingJointType: JointType;

    // Part actions
    addPart: (geometry: THREE.BufferGeometry, fileName: string) => string;
    removePart: (id: string) => void;
    updatePartPosition: (id: string, position: [number, number, number]) => void;
    updatePartRotation: (id: string, rotation: [number, number, number]) => void;
    updatePartScale: (id: string, scale: [number, number, number]) => void;
    updatePartName: (id: string, name: string) => void;
    updatePartColor: (id: string, color: string) => void;
    togglePartVisibility: (id: string) => void;
    selectPart: (id: string | null) => void;

    // Joint actions
    addJoint: (parentId: string, childId: string, type: JointType) => string;
    removeJoint: (id: string) => void;
    updateJoint: (id: string, updates: Partial<Joint>) => void;
    selectJoint: (id: string | null) => void;
    setPivotFromCurrentPosition: (jointId: string) => void;

    // Joint creation mode actions
    startJointCreation: (type?: JointType) => void;
    cancelJointCreation: () => void;
    handlePartClickForJoint: (partId: string) => string | null;
    setPendingJointType: (type: JointType) => void;

    // Transform mode
    setTransformMode: (mode: 'translate' | 'rotate' | 'scale') => void;

    // Alignment actions
    alignMode: boolean;
    alignType: 'point' | 'axis' | 'cylinder';
    alignStep: 'target' | 'source' | 'confirm';
    alignTargetPoint: { point: [number, number, number]; partId?: string } | null; // For point mode

    // Axis Alignment State
    alignPoints: { point: [number, number, number]; normal: [number, number, number]; partId?: string }[]; // Buffer for 2 points
    alignTargetAxis: { center: [number, number, number]; normal: [number, number, number]; partId?: string } | null;
    alignPendingSource: {
        axis: { center: [number, number, number]; normal: [number, number, number] };
        partId: string;
        originalPosition: [number, number, number];
        originalRotation: [number, number, number];
    } | null;

    // Persistent Highlight for Target Selection
    alignTargetHighlight: any | null; // Typed loosely to avoid importing THREE in interface if possible, or use THREE.BufferGeometry
    setAlignTargetHighlight: (geo: any | null) => void;

    alignFlip: boolean;
    toggleAlignFlip: () => void;

    startAlignMode: (type?: 'point' | 'axis' | 'cylinder') => void;
    cancelAlignMode: () => void;

    // Point Mode Handlers
    setAlignTargetPoint: (point: [number, number, number], partId?: string) => void;
    applyAlignment: (sourcePoint: [number, number, number], partId: string) => void;

    // Axis Mode Handlers
    addAlignPoint: (point: [number, number, number], normal?: [number, number, number], partId?: string) => void;
    commitAlignAxis: (axis: { center: [number, number, number]; normal: [number, number, number]; partId?: string }) => void;
    applyAxisAlignment: (sourceAxis: { center: [number, number, number]; normal: [number, number, number] }, partId: string) => void;

    // Reversible Flow
    previewAxisAlignment: (sourceAxis: { center: [number, number, number]; normal: [number, number, number] }, partId: string) => void;
    confirmAlignment: () => void;

    // Assembly actions
    setName: (name: string) => void;
    clearAssembly: () => void;
}

// Generate random pastel color
const randomColor = () => {
    const hue = Math.floor(Math.random() * 360);
    return `hsl(${hue}, 70%, 60%)`;
};

export const useAssemblyStore = create<AssemblyState>()(
    temporal(
        (set, get) => ({
            // Initial state
            name: 'Untitled Assembly',
            parts: [],
            joints: [],
            selectedPartId: null,
            selectedJointId: null,
            transformMode: 'translate',
            jointCreationMode: false,
            jointCreationParentId: null,
            pendingJointType: 'hinge',

            alignMode: false,
            alignType: 'point',
            alignStep: 'target',
            alignTargetPoint: null,
            alignPoints: [],
            alignTargetAxis: null,
            alignPendingSource: null,
            alignTargetHighlight: null,
            alignFlip: false, // Default to Normal (Anti-Parallel for mating faces)
            setAlignTargetHighlight: (geo) => set({ alignTargetHighlight: geo }),

            // Part actions
            addPart: (geometry, fileName) => {
                const id = uuidv4();
                const baseName = fileName.replace(/\.(stl|step|stp)$/i, '');

                // Center the geometry
                geometry.computeBoundingBox();
                geometry.center();

                const newPart: Part = {
                    id,
                    name: baseName,
                    fileName,
                    geometry,
                    position: [0, 0, 0],
                    rotation: [0, 0, 0],
                    scale: [1, 1, 1],
                    color: randomColor(),
                    visible: true,
                };

                set((state) => ({
                    parts: [...state.parts, newPart],
                    selectedPartId: id,
                }));

                return id;
            },

            removePart: (id) => {
                set((state) => ({
                    parts: state.parts.filter((p) => p.id !== id),
                    joints: state.joints.filter((j) => j.parentId !== id && j.childId !== id),
                    selectedPartId: state.selectedPartId === id ? null : state.selectedPartId,
                }));
            },

            updatePartPosition: (id, position) => {
                const { parts, joints } = get();
                const partIndex = parts.findIndex((p) => p.id === id);
                if (partIndex === -1) return;

                const oldPosition = new THREE.Vector3(...parts[partIndex].position);
                const newPosition = new THREE.Vector3(...position);
                const delta = newPosition.clone().sub(oldPosition);

                // 1. Identify Rigid Group (Undirected Search on 'fixed' joints)
                // This finds all parts physically welded to the moving part.
                const movingParts = new Set<string>();
                const rigidQueue = [id];
                movingParts.add(id);

                let head = 0;
                while (head < rigidQueue.length) {
                    const currentId = rigidQueue[head++];

                    joints.forEach(joint => {
                        if (joint.type !== 'fixed') return;

                        let neighborId: string | null = null;
                        if (joint.parentId === currentId) neighborId = joint.childId;
                        if (joint.childId === currentId) neighborId = joint.parentId;

                        if (neighborId && !movingParts.has(neighborId)) {
                            movingParts.add(neighborId);
                            rigidQueue.push(neighborId);
                        }
                    });
                }

                // 2. Identify Descendants (Directed Search from the Rigid Group)
                // This moves any children attached to the moving rigid body (e.g., proper kinematics).
                const descendantQueue = [...movingParts];
                head = 0;

                while (head < descendantQueue.length) {
                    const currentId = descendantQueue[head++];

                    joints.forEach(joint => {
                        // Check if this joint starts from the current part (Directed)
                        if (joint.parentId === currentId) {
                            if (!movingParts.has(joint.childId)) {
                                movingParts.add(joint.childId);
                                descendantQueue.push(joint.childId);
                            }
                        }
                    });
                }

                // 3. Apply Delta to ALL identified moving parts
                const newPositions = new Map<string, [number, number, number]>();

                movingParts.forEach(partId => {
                    const part = parts.find(p => p.id === partId);
                    if (part) {
                        const currentPos = new THREE.Vector3(...part.position);
                        const nextPos = currentPos.add(delta);
                        newPositions.set(partId, nextPos.toArray());
                    }
                });

                set((state) => ({
                    parts: state.parts.map((p) =>
                        newPositions.has(p.id) ? { ...p, position: newPositions.get(p.id)! } : p
                    ),
                }));
            },


            updatePartRotation: (id, rotation) => {
                const { parts, joints } = get();
                const part = parts.find((p) => p.id === id);
                if (!part) return;

                // 1. Calculate Rotation Delta (Quaternion)
                const oldRot = new THREE.Euler(...part.rotation);
                const newRot = new THREE.Euler(...rotation);
                const qOld = new THREE.Quaternion().setFromEuler(oldRot);
                const qNew = new THREE.Quaternion().setFromEuler(newRot);
                const qDelta = qNew.clone().multiply(qOld.clone().invert());

                // 2. Center of Rotation (The Parent's Position)
                const center = new THREE.Vector3(...part.position);

                // 3. Track updates
                const posUpdates = new Map<string, [number, number, number]>();
                const rotUpdates = new Map<string, [number, number, number]>();

                rotUpdates.set(id, rotation);

                const processChildren = (parentId: string, parentQDelta: THREE.Quaternion, pivot: THREE.Vector3) => {
                    const childJoints = joints.filter(j => j.parentId === parentId);
                    childJoints.forEach(joint => {
                        const childPart = parts.find(p => p.id === joint.childId);
                        if (childPart && !rotUpdates.has(joint.childId)) {
                            // Update Rotation
                            const childRot = new THREE.Euler(...childPart.rotation);
                            const childQ = new THREE.Quaternion().setFromEuler(childRot);
                            const newChildQ = parentQDelta.clone().multiply(childQ);
                            rotUpdates.set(joint.childId, new THREE.Euler().setFromQuaternion(newChildQ).toArray() as [number, number, number]);

                            // Update Position (Orbit around parent)
                            const childPos = new THREE.Vector3(...childPart.position);
                            const offset = childPos.clone().sub(pivot);
                            offset.applyQuaternion(parentQDelta);
                            const newChildPos = pivot.clone().add(offset);
                            posUpdates.set(joint.childId, newChildPos.toArray());

                            // Recurse
                            // Important: Pivot for next child is the SAME parent pivot? 
                            // NO. If I rotate the shoulder, the elbow moves. 
                            // If I rotate the elbow, the hand moves.
                            // Here we are rotating the PARENT (id). 
                            // So the Pivot is ALWAYS the Parent's origin (center).
                            processChildren(joint.childId, parentQDelta, pivot);
                        }
                    });
                };

                processChildren(id, qDelta, center);

                set((state) => ({
                    parts: state.parts.map((p) => {
                        if (posUpdates.has(p.id) && rotUpdates.has(p.id)) {
                            return { ...p, position: posUpdates.get(p.id)!, rotation: rotUpdates.get(p.id)! };
                        }
                        if (rotUpdates.has(p.id)) { // Root parent only changes rotation
                            return { ...p, rotation: rotUpdates.get(p.id)! };
                        }
                        return p;
                    }),
                }));
            },

            updatePartScale: (id, scale) => {
                set((state) => ({
                    parts: state.parts.map((p) =>
                        p.id === id ? { ...p, scale } : p
                    ),
                }));
            },

            updatePartName: (id, name) => {
                set((state) => ({
                    parts: state.parts.map((p) =>
                        p.id === id ? { ...p, name } : p
                    ),
                }));
            },

            updatePartColor: (id, color) => {
                set((state) => ({
                    parts: state.parts.map((p) =>
                        p.id === id ? { ...p, color } : p
                    ),
                }));
            },

            togglePartVisibility: (id) => {
                set((state) => ({
                    parts: state.parts.map((p) =>
                        p.id === id ? { ...p, visible: !p.visible } : p
                    ),
                }));
            },

            selectPart: (id) => {
                set({ selectedPartId: id, selectedJointId: null });
            },

            // Joint actions
            addJoint: (parentId, childId, type) => {
                const id = uuidv4();
                const parentPart = get().parts.find((p) => p.id === parentId);
                const childPart = get().parts.find((p) => p.id === childId);

                if (!parentPart || !childPart) return '';

                const newJoint: Joint = {
                    id,
                    name: `${parentPart.name}_to_${childPart.name}`,
                    type,
                    parentId,
                    childId,
                    axis: 'z',
                    limitLower: type === 'hinge' ? -Math.PI : -1,
                    limitUpper: type === 'hinge' ? Math.PI : 1,
                    anchorPosition: [...childPart.position] as [number, number, number], // Default to child's current position
                    childOffset: [0, 0, 0],
                    pivotSet: false,
                    previewValue: 0.5,
                };

                set((state) => ({
                    joints: [...state.joints, newJoint],
                    selectedJointId: id,
                }));

                return id;
            },

            removeJoint: (id) => {
                set((state) => ({
                    joints: state.joints.filter((j) => j.id !== id),
                    selectedJointId: state.selectedJointId === id ? null : state.selectedJointId,
                }));
            },

            updateJoint: (id, updates) => {
                set((state) => ({
                    joints: state.joints.map((j) =>
                        j.id === id ? { ...j, ...updates } : j
                    ),
                }));
            },

            selectJoint: (id) => {
                set({ selectedJointId: id, selectedPartId: null });
            },

            // Set pivot from current child position relative to parent
            setPivotFromCurrentPosition: (jointId) => {
                const { joints, parts } = get();
                const joint = joints.find((j) => j.id === jointId);
                if (!joint) return;

                const parentPart = parts.find((p) => p.id === joint.parentId);
                const childPart = parts.find((p) => p.id === joint.childId);
                if (!parentPart || !childPart) return;

                // Calculate child's position relative to parent (this becomes the pivot)
                const anchorPosition: [number, number, number] = [
                    childPart.position[0] - parentPart.position[0],
                    childPart.position[1] - parentPart.position[1],
                    childPart.position[2] - parentPart.position[2],
                ];

                set((state) => ({
                    joints: state.joints.map((j) =>
                        j.id === jointId
                            ? {
                                ...j,
                                anchorPosition,
                                childOffset: [0, 0, 0] as [number, number, number],
                                pivotSet: true,
                                previewValue: 0.5, // Reset preview to center
                            }
                            : j
                    ),
                }));
            },
            toggleAlignFlip: () => {
                const { alignMode, alignStep, alignPendingSource, previewAxisAlignment } = get();
                const newFlip = !get().alignFlip;
                set({ alignFlip: newFlip });

                // Live Update if in Confirmation Step
                if (alignMode && alignStep === 'confirm' && alignPendingSource) {
                    // Re-run preview with new flip state
                    previewAxisAlignment(alignPendingSource.axis, alignPendingSource.partId);
                }
            },

            // Alignment actions
            startAlignMode: (type = 'point') => {
                set({
                    alignMode: true,
                    alignType: type,
                    alignStep: 'target',
                    alignTargetPoint: null,
                    alignPoints: [],
                    alignTargetAxis: null,
                    alignPendingSource: null,
                    alignTargetHighlight: null,
                    selectedPartId: null,
                    transformMode: 'translate', // Disable gizmo during align
                });
            },

            cancelAlignMode: () => {
                const { alignStep, alignPendingSource, parts } = get();

                // If cancelling during confirmation, revert the part position
                if (alignStep === 'confirm' && alignPendingSource) {
                    const { partId, originalPosition, originalRotation } = alignPendingSource;
                    set({
                        parts: parts.map(p =>
                            p.id === partId ? {
                                ...p,
                                position: originalPosition,
                                rotation: originalRotation
                            } : p
                        )
                    });
                }

                set({
                    alignMode: false,
                    alignStep: 'target',
                    alignTargetPoint: null,
                    alignPoints: [],
                    alignTargetAxis: null,
                    alignPendingSource: null,
                    alignTargetHighlight: null,
                    transformMode: 'translate',
                });
            },

            // ... Point handlers ...

            // Axis Mode Handlers
            addAlignPoint: (point, normal, partId) => {
                const { alignPoints, alignStep } = get();
                const n = normal || [0, 1, 0];
                const newPoints = [...alignPoints, { point, normal: n, partId }];

                // 2-point Check
                // We have 3 points, calculate the axis
                if (alignStep === 'target' && newPoints.length === 2) {

                    const p1 = new THREE.Vector3(...newPoints[0].point);
                    const p2 = new THREE.Vector3(...newPoints[1].point);
                    const n1 = new THREE.Vector3(...newPoints[0].normal);

                    // Center is midpoint
                    const center = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
                    // Normal - use the first one from face (assuming user clicked same face)
                    const axisNormal = n1.normalize();
                    // Part ID from the first point
                    const targetPartId = newPoints[0].partId;

                    set({
                        alignTargetAxis: { center: center.toArray(), normal: axisNormal.toArray(), partId: targetPartId },
                        alignPoints: [], // Clear for source step
                        alignStep: 'source'
                    });
                } else {
                    set({ alignPoints: newPoints });
                }
            },

            commitAlignAxis: (axis) => {
                set({
                    alignPoints: [],
                    alignTargetAxis: axis,
                    alignStep: 'source',
                });
            },

            // New: Preview Alignment (Moves part but doesn't constrain yet)
            previewAxisAlignment: (sourceAxis, partId) => {
                const { alignTargetAxis, parts, alignFlip, alignPendingSource } = get();
                if (!alignTargetAxis) return;

                const part = parts.find((p) => p.id === partId);
                if (!part) return;

                // Save original state if not already saved (first preview)
                let originalState = alignPendingSource;
                if (!originalState) {
                    originalState = {
                        axis: sourceAxis,
                        partId: partId,
                        originalPosition: part.position,
                        originalRotation: part.rotation
                    };
                }

                // Math: Align Source Axis to Target Axis
                const S = new THREE.Vector3(...sourceAxis.center);
                const N_s = new THREE.Vector3(...sourceAxis.normal).normalize();
                const T = new THREE.Vector3(...alignTargetAxis.center);
                let N_t = new THREE.Vector3(...alignTargetAxis.normal).normalize();

                if (!alignFlip) {
                    N_t.negate();
                }

                const qRot = new THREE.Quaternion().setFromUnitVectors(N_s, N_t);

                // Current Part Transform (Use Original to keep it stable during live updates if needed, though here we use currentPART which is fine if we assume clean state, BUT better to use original rotation if we are flipping back and forth to avoid drift)
                // Actually, let's use the ORIGINAL rotation from the pending state if available to be safe against drift.
                const sourcePart = alignPendingSource ? { position: alignPendingSource.originalPosition, rotation: alignPendingSource.originalRotation } : part;

                const partPos = new THREE.Vector3(...sourcePart.position);
                const partRot = new THREE.Euler(...sourcePart.rotation);
                const partQuat = new THREE.Quaternion().setFromEuler(partRot);

                // New Rotation = qRot * partQuat
                const newPartQuat = qRot.clone().multiply(partQuat);

                // Position Alignment
                // We pivot around S (Source Axis Center relative to Part).
                // Wait, S is in World Space.
                // If we use the ORIGINAL part position, S is valid for THAT position.
                // So yes, using original state is safer.

                const V = new THREE.Vector3().subVectors(partPos, S);
                V.applyQuaternion(qRot);
                const newPartPos = new THREE.Vector3().addVectors(T, V);

                set((state) => ({
                    parts: state.parts.map((p) =>
                        p.id === partId ? {
                            ...p,
                            position: newPartPos.toArray(),
                            rotation: new THREE.Euler().setFromQuaternion(newPartQuat).toArray() as [number, number, number]
                        } : p
                    ),
                    alignStep: 'confirm',
                    alignPendingSource: originalState,
                    selectedPartId: partId,
                }));
            },

            // New: Confirm Alignment
            confirmAlignment: () => {
                const { alignPendingSource, alignTargetAxis, addJoint, parts } = get();
                if (!alignPendingSource || !alignTargetAxis) return;

                const partId = alignPendingSource.partId;

                // Auto-Constrain: Create Fixed Joint
                if (alignTargetAxis.partId) {
                    const jointId = addJoint(alignTargetAxis.partId, partId, 'cylindrical');

                    const joint = get().joints.find(j => j.id === jointId);
                    if (joint) {
                        const parentPart = parts.find(p => p.id === alignTargetAxis.partId);
                        if (parentPart) {
                            const relAnchor = [
                                alignTargetAxis.center[0] - parentPart.position[0],
                                alignTargetAxis.center[1] - parentPart.position[1],
                                alignTargetAxis.center[2] - parentPart.position[2],
                            ] as [number, number, number];

                            get().updateJoint(jointId, {
                                anchorPosition: relAnchor,
                                pivotSet: true
                            });
                        }
                    }
                }

                // Cleanup
                set({
                    alignMode: false,
                    alignStep: 'target',
                    alignType: 'point',
                    alignPoints: [],
                    alignTargetAxis: null,
                    alignPendingSource: null,
                    alignTargetHighlight: null,
                    transformMode: 'translate',
                    selectedPartId: partId,
                });
            },

            applyAxisAlignment: (sourceAxis, partId) => {
                // Legacy / Manual Axis Mode can still use this, OR we can route it to preview->confirm too?
                // For now, let's keep it direct for the "Axis (2-Pt)" mode unless user wants that reversible too.
                // The user specifically asked for "Cylindrical alignment".
                // But logically, consistency is good.
                // Let's forward this to previewAxisAlignment for consistency!
                get().previewAxisAlignment(sourceAxis, partId);
            },

            // Point Mode Handlers
            setAlignTargetPoint: (point, partId) => {
                set({
                    alignTargetPoint: { point, partId },
                    alignStep: 'source',
                });
            },

            applyAlignment: (sourcePoint, partId) => {
                const { alignTargetPoint, parts, addJoint } = get();
                if (!alignTargetPoint) return;

                // Calculate translation vector (Target - Source)
                const translation = [
                    alignTargetPoint.point[0] - sourcePoint[0],
                    alignTargetPoint.point[1] - sourcePoint[1],
                    alignTargetPoint.point[2] - sourcePoint[2],
                ];

                // Find part and apply translation
                const part = parts.find((p) => p.id === partId);
                if (part) {
                    const newPosition: [number, number, number] = [
                        part.position[0] + translation[0],
                        part.position[1] + translation[1],
                        part.position[2] + translation[2],
                    ];

                    set((state) => ({
                        parts: state.parts.map((p) =>
                            p.id === partId ? { ...p, position: newPosition } : p
                        ),
                        // Reset/End alignment mode
                        alignMode: false,
                        alignStep: 'target',
                        alignTargetPoint: null,
                        selectedPartId: partId, // Select the moved part
                        transformMode: 'translate', // Re-enable gizmo
                    }));

                    // Auto-Constrain: Create Fixed Joint if target has a part ID
                    if (alignTargetPoint.partId) {
                        // We use the alignment point (Target Point) as the anchor
                        const jointId = addJoint(alignTargetPoint.partId, partId, 'fixed');

                        // Set the anchor to the world position where they met
                        // For 'fixed', anchor technically doesn't matter for logic, but good for visualization
                        // Updating joint with anchor position
                        const joint = get().joints.find(j => j.id === jointId);
                        if (joint) {
                            const parentPart = parts.find(p => p.id === alignTargetPoint.partId);
                            if (parentPart) {
                                // Anchor relative to parent
                                const relAnchor = [
                                    alignTargetPoint.point[0] - parentPart.position[0],
                                    alignTargetPoint.point[1] - parentPart.position[1],
                                    alignTargetPoint.point[2] - parentPart.position[2],
                                ] as [number, number, number];

                                get().updateJoint(jointId, {
                                    anchorPosition: relAnchor,
                                    pivotSet: true
                                });
                            }
                        }
                    }
                }
            },


            // Joint creation mode actions
            startJointCreation: (type = 'hinge') => {
                set({
                    jointCreationMode: true,
                    jointCreationParentId: null,
                    pendingJointType: type,
                    selectedPartId: null,
                    selectedJointId: null,
                });
            },

            cancelJointCreation: () => {
                set({
                    jointCreationMode: false,
                    jointCreationParentId: null,
                });
            },

            handlePartClickForJoint: (partId) => {
                const { jointCreationParentId, pendingJointType, addJoint } = get();

                if (!jointCreationParentId) {
                    // First click - select parent
                    set({ jointCreationParentId: partId });
                    return null;
                } else if (partId !== jointCreationParentId) {
                    // Second click - create joint
                    const jointId = addJoint(jointCreationParentId, partId, pendingJointType);
                    set({
                        jointCreationMode: false,
                        jointCreationParentId: null,
                        selectedJointId: jointId,
                    });
                    return jointId;
                }
                return null;
            },

            setPendingJointType: (type) => {
                set({ pendingJointType: type });
            },

            // Transform mode
            setTransformMode: (mode) => {
                set({ transformMode: mode });
            },

            // Assembly actions
            setName: (name) => {
                set({ name });
            },

            clearAssembly: () => {
                set({
                    parts: [],
                    joints: [],
                    selectedPartId: null,
                    selectedJointId: null,
                });
            },
        }),
        {
            partialize: (state) => ({ parts: state.parts, joints: state.joints }),
        }
    )
);
