import { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { TransformControls } from '@react-three/drei'
import { useStore } from '../stores/useStore'
import { loadStepFile } from '../utils/stepLoader'
import { detectSurface } from '../utils/geometry'
import { exportToMJCF, exportPartsToSTL } from '../utils/mujocoExporter'
import * as THREE from 'three'

interface HoveredSurface {
    meshUuid: string
    faceIndex: number
    surface: ReturnType<typeof detectSurface>
}

export function Viewer() {
    const { parts, addPart, selectedPartId, selectPart, selectedFace, selectFace, mateMode, mateFace1, mateFace2, mates } = useStore()
    const { scene, raycaster, pointer, camera } = useThree()
    const loadedRefs = useRef<Set<string>>(new Set())
    const [hoveredSurface, setHoveredSurface] = useState<HoveredSurface | null>(null)
    const lastHoverRef = useRef<{ meshUuid: string; faceIndex: number } | null>(null)

    // Load default parts on mount
    useEffect(() => {
        const loadDefaultParts = async () => {
            const files = ['motor.step', 'horn.step']

            for (const file of files) {
                if (loadedRefs.current.has(file)) continue

                // Mark as loading IMMEDIATELY to prevent StrictMode race condition
                loadedRefs.current.add(file)

                const url = `/${file}`
                try {
                    console.log(`Loading ${file}...`)
                    const group = await loadStepFile(url)

                    if (file === 'horn.step') {
                        group.position.set(50, 0, 0)
                    }

                    addPart(group, file)
                } catch (e) {
                    console.error(`Failed to load ${file}`, e)
                    loadedRefs.current.delete(file) // Allow retry on failure
                }
            }
        }

        loadDefaultParts()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []) // Empty deps - only run once on mount

    // Listen for simulate event - exports MJCF + STL to output folder
    useEffect(() => {
        const handleSimulate = async () => {
            const mjcf = exportToMJCF(parts, mates)
            const stlFiles = exportPartsToSTL(parts)

            console.log('Exporting to output folder...')
            console.log('MJCF:', mjcf)
            console.log('STL files:', [...stlFiles.keys()])

            // Helper to encode ArrayBuffer to base64 (handles large files)
            const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
                const bytes = new Uint8Array(buffer)
                let binary = ''
                const chunkSize = 8192
                for (let i = 0; i < bytes.length; i += chunkSize) {
                    const chunk = bytes.subarray(i, i + chunkSize)
                    binary += String.fromCharCode.apply(null, Array.from(chunk))
                }
                return btoa(binary)
            }

            // Send to save_server
            try {
                console.log('Sending MJCF to save server...')

                // Export MJCF
                const mjcfResponse = await fetch('http://localhost:3001/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ type: 'mjcf', content: mjcf })
                })

                if (!mjcfResponse.ok) throw new Error('Failed to save MJCF')
                console.log('MJCF saved successfully')

                // Export STL files
                for (const [filename, data] of stlFiles) {
                    console.log(`Saving ${filename} (${data.byteLength} bytes)...`)
                    const base64 = arrayBufferToBase64(data)
                    const stlResponse = await fetch('http://localhost:3001/save', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ type: 'stl', filename, content: base64 })
                    })
                    if (!stlResponse.ok) throw new Error(`Failed to save ${filename}`)
                    console.log(`${filename} saved successfully`)
                }

                alert('✅ Exported to output/ folder!\n\nRun: cd train && ./venv/bin/mjpython train.py')

            } catch (e) {
                console.error('Export error:', e)
                // Fallback: download files
                const blob = new Blob([mjcf], { type: 'application/xml' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = 'model.xml'
                a.click()
                URL.revokeObjectURL(url)

                alert('Save server not running.\n\nStart it with: python train/save_server.py\n\nOr use downloaded model.xml')
            }
        }

        window.addEventListener('simulate', handleSimulate)
        return () => window.removeEventListener('simulate', handleSimulate)
    }, [parts, mates])

    // Listen for importStep event from file picker
    useEffect(() => {
        const handleImportStep = async (e: CustomEvent<{ file: File }>) => {
            const file = e.detail.file
            const fileName = file.name

            if (loadedRefs.current.has(fileName)) {
                console.log(`${fileName} already loaded, skipping`)
                return
            }

            // Mark as loading immediately
            loadedRefs.current.add(fileName)

            try {
                console.log(`Loading imported file: ${fileName}...`)

                // Read file as ArrayBuffer and create blob URL
                const arrayBuffer = await file.arrayBuffer()
                const blob = new Blob([arrayBuffer])
                const url = URL.createObjectURL(blob)

                const group = await loadStepFile(url)

                // Position new parts to the right of existing parts
                const offset = parts.length * 50
                group.position.set(offset, 0, 0)

                addPart(group, fileName)

                URL.revokeObjectURL(url)
                console.log(`Successfully loaded ${fileName}`)
            } catch (err) {
                console.error(`Failed to load ${fileName}:`, err)
                loadedRefs.current.delete(fileName) // Allow retry
                alert(`Failed to load ${fileName}. Make sure it's a valid STEP file.`)
            }
        }

        window.addEventListener('importStep', handleImportStep as unknown as EventListener)
        return () => window.removeEventListener('importStep', handleImportStep as unknown as EventListener)
    }, [parts.length, addPart])

    // Update hover state on each frame - detect FULL surface on hover
    useFrame(() => {
        if (parts.length === 0) return

        raycaster.setFromCamera(pointer, camera)

        const meshes: THREE.Mesh[] = []
        parts.forEach(part => {
            part.object.traverse(obj => {
                if (obj instanceof THREE.Mesh) meshes.push(obj)
            })
        })

        const intersects = raycaster.intersectObjects(meshes)

        if (intersects.length > 0 && intersects[0].faceIndex !== undefined) {
            const hit = intersects[0]
            const mesh = hit.object as THREE.Mesh
            const faceIndex = hit.faceIndex

            // Only recompute surface if we moved to a different face/mesh
            const last = lastHoverRef.current
            if (!last || last.meshUuid !== mesh.uuid || last.faceIndex !== faceIndex) {
                const surface = detectSurface(mesh.geometry, faceIndex)
                setHoveredSurface({ meshUuid: mesh.uuid, faceIndex, surface })
                lastHoverRef.current = { meshUuid: mesh.uuid, faceIndex }
            }
        } else {
            if (lastHoverRef.current !== null) {
                setHoveredSurface(null)
                lastHoverRef.current = null
            }
        }
    })

    // Helper to create highlight geometry from face indices
    const createHighlightGeometry = useCallback((
        meshUuid: string,
        indices: number[]
    ): { geo: THREE.BufferGeometry; worldMatrix: THREE.Matrix4 } | null => {
        let mesh: THREE.Mesh | null = null
        scene.traverse((o) => {
            if (o.uuid === meshUuid) mesh = o as THREE.Mesh
        })

        if (!mesh) return null

        const sourceGeo = (mesh as THREE.Mesh).geometry
        const positionAttr = sourceGeo.attributes.position
        const indexAttr = sourceGeo.index

        const newPositions: number[] = []

        indices.forEach(faceIndex => {
            if (indexAttr) {
                const a = indexAttr.getX(faceIndex * 3)
                const b = indexAttr.getX(faceIndex * 3 + 1)
                const c = indexAttr.getX(faceIndex * 3 + 2)

                newPositions.push(
                    positionAttr.getX(a), positionAttr.getY(a), positionAttr.getZ(a),
                    positionAttr.getX(b), positionAttr.getY(b), positionAttr.getZ(b),
                    positionAttr.getX(c), positionAttr.getY(c), positionAttr.getZ(c)
                )
            } else {
                const a = faceIndex * 3
                newPositions.push(
                    positionAttr.getX(a), positionAttr.getY(a), positionAttr.getZ(a),
                    positionAttr.getX(a + 1), positionAttr.getY(a + 1), positionAttr.getZ(a + 1),
                    positionAttr.getX(a + 2), positionAttr.getY(a + 2), positionAttr.getZ(a + 2)
                )
            }
        })

        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.Float32BufferAttribute(newPositions, 3))
        geo.computeVertexNormals()

            ; (mesh as THREE.Mesh).updateWorldMatrix(true, false)
        const worldMatrix = (mesh as THREE.Mesh).matrixWorld.clone()

        return { geo, worldMatrix }
    }, [scene])

    // Create hover highlight geometry (FULL SURFACE, works in all modes)
    const hoverGeometry = useMemo(() => {
        if (!hoveredSurface) return null
        const result = createHighlightGeometry(hoveredSurface.meshUuid, hoveredSurface.surface.indices)
        if (!result) return null
        return { ...result, surfaceType: hoveredSurface.surface.type }
    }, [hoveredSurface, createHighlightGeometry])

    // Create selection highlight geometry
    const selectionGeometry = useMemo(() => {
        if (!selectedFace) return null
        return createHighlightGeometry(selectedFace.meshUuid, selectedFace.indices)
    }, [selectedFace, createHighlightGeometry])

    // Mate face highlights
    const mateFace1Geometry = useMemo(() => {
        if (!mateFace1) return null
        return createHighlightGeometry(mateFace1.meshUuid, mateFace1.indices)
    }, [mateFace1, createHighlightGeometry])

    const mateFace2Geometry = useMemo(() => {
        if (!mateFace2) return null
        return createHighlightGeometry(mateFace2.meshUuid, mateFace2.indices)
    }, [mateFace2, createHighlightGeometry])

    return (
        <>
            <group
                onClick={(e) => {
                    e.stopPropagation()

                    if (e.object instanceof THREE.Mesh && e.faceIndex !== undefined) {
                        const mesh = e.object as THREE.Mesh
                        console.log("Detecting surface from face:", e.faceIndex)

                        const surface = detectSurface(mesh.geometry, e.faceIndex)
                        console.log("Detected Surface:", surface)

                        let target: any = e.object
                        while (target && !target.userData.id && target.parent) {
                            target = target.parent
                        }

                        if (target && target.userData.id) {
                            selectFace({
                                ...surface,
                                partId: target.userData.id,
                                meshUuid: mesh.uuid
                            })
                        }
                    } else {
                        selectFace(null)
                        selectPart(null)
                    }
                }}
            >
                {parts.map((part) => (
                    <primitive key={part.id} object={part.object} />
                ))}
            </group>

            {/* Hover Highlight (FULL surface, works in ALL modes including mate mode) */}
            {hoverGeometry && (
                <mesh
                    geometry={hoverGeometry.geo}
                    matrixAutoUpdate={false}
                    matrix={hoverGeometry.worldMatrix}
                >
                    <meshBasicMaterial
                        color={hoverGeometry.surfaceType === 'cylinder' ? '#fbbf24' : '#22d3ee'}
                        opacity={0.4}
                        transparent
                        depthTest={false}
                        side={THREE.DoubleSide}
                    />
                </mesh>
            )}

            {/* Selection Highlight (full surface) - only when NOT in mate mode */}
            {selectionGeometry && !mateMode && (
                <mesh
                    geometry={selectionGeometry.geo}
                    matrixAutoUpdate={false}
                    matrix={selectionGeometry.worldMatrix}
                >
                    <meshBasicMaterial
                        color={selectedFace?.type === 'cylinder' ? '#fbbf24' : '#22d3ee'}
                        opacity={0.6}
                        transparent
                        depthTest={false}
                        side={THREE.DoubleSide}
                    />
                </mesh>
            )}

            {/* Mate Face 1 Highlight (orange) */}
            {mateFace1Geometry && (
                <mesh
                    geometry={mateFace1Geometry.geo}
                    matrixAutoUpdate={false}
                    matrix={mateFace1Geometry.worldMatrix}
                >
                    <meshBasicMaterial color="#f97316" opacity={0.7} transparent depthTest={false} side={THREE.DoubleSide} />
                </mesh>
            )}

            {/* Mate Face 2 Highlight (purple) */}
            {mateFace2Geometry && (
                <mesh
                    geometry={mateFace2Geometry.geo}
                    matrixAutoUpdate={false}
                    matrix={mateFace2Geometry.worldMatrix}
                >
                    <meshBasicMaterial color="#a855f7" opacity={0.7} transparent depthTest={false} side={THREE.DoubleSide} />
                </mesh>
            )}

            {selectedPartId && !selectedFace && !mateMode && parts.find(p => p.id === selectedPartId) && (
                <TransformControls
                    object={parts.find(p => p.id === selectedPartId)?.object}
                    mode="translate"
                />
            )}
        </>
    )
}
