import { create } from 'zustand'
import { Group, Vector3, Quaternion, Matrix4 } from 'three'
import { v4 as uuidv4 } from 'uuid'
import type { SurfaceData } from '../utils/geometry'

export interface Part {
    id: string
    name: string
    object: Group
    fileName: string
}

export interface SelectedFace extends SurfaceData {
    partId: string
    meshUuid: string
}

export interface Mate {
    id: string
    type: 'cylindrical' | 'planar'
    face1: SelectedFace
    face2: SelectedFace
    flipped: boolean
}

type MateMode = null | 'cylindrical' | 'planar'

interface AppState {
    parts: Part[]
    selectedPartId: string | null
    selectedFace: SelectedFace | null
    mateMode: MateMode
    mateFace1: SelectedFace | null
    mateFace2: SelectedFace | null
    mateFlipped: boolean
    mates: Mate[]

    addPart: (object: Group, fileName: string) => void
    selectPart: (id: string | null) => void
    selectFace: (face: SelectedFace | null) => void
    removePart: (id: string) => void
    startMate: (type: 'cylindrical' | 'planar') => void
    cancelMate: () => void
    toggleFlip: () => void
    confirmMate: () => void
    clearMates: () => void
}

export const useStore = create<AppState>((set, get) => ({
    parts: [],
    selectedPartId: null,
    selectedFace: null,
    mateMode: null,
    mateFace1: null,
    mateFace2: null,
    mateFlipped: false,
    mates: [],

    addPart: (object, fileName) => {
        const id = uuidv4()
        object.userData.id = id
        object.name = fileName
        set((state) => ({ parts: [...state.parts, { id, name: fileName, object, fileName }] }))
    },

    selectPart: (id) => set({ selectedPartId: id, selectedFace: null }),

    selectFace: (face) => {
        const { mateMode, mateFace1 } = get()

        if (mateMode && face) {
            if (!mateFace1) {
                set({ mateFace1: face, selectedFace: face })
            } else if (face.partId !== mateFace1.partId) {
                set({ mateFace2: face, selectedFace: face })
            } else {
                set({ mateFace1: face, mateFace2: null, selectedFace: face })
            }
        } else {
            set({ selectedFace: face, selectedPartId: face ? face.partId : null })
        }
    },

    removePart: (id) => set((state) => ({
        parts: state.parts.filter(p => p.id !== id),
        selectedPartId: state.selectedPartId === id ? null : state.selectedPartId,
        selectedFace: state.selectedFace?.partId === id ? null : state.selectedFace
    })),

    startMate: (type) => set({
        mateMode: type,
        mateFace1: null,
        mateFace2: null,
        mateFlipped: false,
        selectedFace: null
    }),

    cancelMate: () => set({
        mateMode: null,
        mateFace1: null,
        mateFace2: null,
        mateFlipped: false
    }),

    toggleFlip: () => set((state) => ({ mateFlipped: !state.mateFlipped })),

    confirmMate: () => {
        const { mateMode, mateFace1, mateFace2, mateFlipped, parts } = get()

        if (!mateMode || !mateFace1 || !mateFace2) return

        const mate: Mate = {
            id: uuidv4(),
            type: mateMode,
            face1: mateFace1,
            face2: mateFace2,
            flipped: mateFlipped
        }

        const part1 = parts.find(p => p.id === mateFace1.partId)
        const part2 = parts.find(p => p.id === mateFace2.partId)

        if (part1 && part2) {
            // Get the mesh within each part to get proper world matrix
            let mesh1: any = null
            let mesh2: any = null

            part1.object.traverse(o => {
                if (o.uuid === mateFace1.meshUuid) mesh1 = o
            })
            part2.object.traverse(o => {
                if (o.uuid === mateFace2.meshUuid) mesh2 = o
            })

            if (!mesh1 || !mesh2) {
                console.error('Could not find meshes for mate')
                return
            }

            // Update world matrices
            mesh1.updateWorldMatrix(true, false)
            mesh2.updateWorldMatrix(true, false)

            // Get face normals in WORLD space
            const worldRotation1 = new Matrix4().extractRotation(mesh1.matrixWorld)
            const worldRotation2 = new Matrix4().extractRotation(mesh2.matrixWorld)

            const normal1World = mateFace1.normal.clone().applyMatrix4(worldRotation1).normalize()
            const normal2World = mateFace2.normal.clone().applyMatrix4(worldRotation2).normalize()

            if (mateMode === 'cylindrical') {
                // CYLINDRICAL MATE: 
                // For servo horn on shaft, we want face2's normal to point OPPOSITE to face1's normal
                // This makes the surfaces face each other

                // Target: face2 normal should point opposite to face1 normal (surfaces meet)
                const targetNormal = mateFlipped ? normal1World.clone() : normal1World.clone().negate()

                // Calculate rotation to align normal2World -> targetNormal
                const rotationQuat = new Quaternion()
                rotationQuat.setFromUnitVectors(normal2World, targetNormal)

                // Apply rotation to part2
                part2.object.quaternion.premultiply(rotationQuat)
                part2.object.updateMatrixWorld(true)

                // Recalculate mesh2 world matrix after rotation
                mesh2.updateWorldMatrix(true, false)

                // Get face centers in world space
                const center1World = mateFace1.point.clone()
                mesh1.localToWorld(center1World)

                const center2World = mateFace2.point.clone()
                mesh2.localToWorld(center2World)

                // Move part2 so centers align
                const offset = center1World.clone().sub(center2World)
                part2.object.position.add(offset)

                console.log('Cylindrical mate applied:', {
                    normal1World: normal1World.toArray(),
                    normal2World: normal2World.toArray(),
                    targetNormal: targetNormal.toArray(),
                    center1: center1World.toArray(),
                    center2: center2World.toArray(),
                    offset: offset.toArray()
                })

            } else if (mateMode === 'planar') {
                // PLANAR MATE: Same logic - align faces to meet
                const targetNormal = mateFlipped ? normal1World.clone() : normal1World.clone().negate()

                const rotationQuat = new Quaternion()
                rotationQuat.setFromUnitVectors(normal2World, targetNormal)
                part2.object.quaternion.premultiply(rotationQuat)
                part2.object.updateMatrixWorld(true)

                mesh2.updateWorldMatrix(true, false)
                const center1World = mateFace1.point.clone()
                mesh1.localToWorld(center1World)
                const center2World = mateFace2.point.clone()
                mesh2.localToWorld(center2World)

                const offset = center1World.clone().sub(center2World)
                part2.object.position.add(offset)
            }
        }

        set((state) => ({
            mates: [...state.mates, mate],
            mateMode: null,
            mateFace1: null,
            mateFace2: null,
            mateFlipped: false
        }))
    },

    clearMates: () => set({ mates: [] })
}))
