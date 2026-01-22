import type { Part, Mate } from '../stores/useStore'
import * as THREE from 'three'

/**
 * Convert a Three.js BufferGeometry to STL binary format
 */
function geometryToSTL(geometry: THREE.BufferGeometry, name: string): ArrayBuffer {
    const positions = geometry.attributes.position
    const indices = geometry.index

    let triangleCount = 0
    if (indices) {
        triangleCount = indices.count / 3
    } else {
        triangleCount = positions.count / 3
    }

    // STL header (80 bytes) + triangle count (4 bytes) + triangles (50 bytes each)
    const bufferSize = 80 + 4 + triangleCount * 50
    const buffer = new ArrayBuffer(bufferSize)
    const view = new DataView(buffer)

    // Write header (80 bytes, can be anything)
    const header = `STL exported from Makeshit V2 - ${name}`
    for (let i = 0; i < 80; i++) {
        view.setUint8(i, i < header.length ? header.charCodeAt(i) : 0)
    }

    // Write triangle count
    view.setUint32(80, triangleCount, true)

    // Helper vectors
    const vA = new THREE.Vector3()
    const vB = new THREE.Vector3()
    const vC = new THREE.Vector3()
    const normal = new THREE.Vector3()
    const cb = new THREE.Vector3()
    const ab = new THREE.Vector3()

    let offset = 84
    for (let i = 0; i < triangleCount; i++) {
        let a: number, b: number, c: number

        if (indices) {
            a = indices.getX(i * 3)
            b = indices.getX(i * 3 + 1)
            c = indices.getX(i * 3 + 2)
        } else {
            a = i * 3
            b = i * 3 + 1
            c = i * 3 + 2
        }

        vA.fromBufferAttribute(positions, a)
        vB.fromBufferAttribute(positions, b)
        vC.fromBufferAttribute(positions, c)

        // Compute normal
        cb.subVectors(vC, vB)
        ab.subVectors(vA, vB)
        normal.crossVectors(cb, ab).normalize()

        // Write normal (3 floats)
        view.setFloat32(offset, normal.x, true); offset += 4
        view.setFloat32(offset, normal.y, true); offset += 4
        view.setFloat32(offset, normal.z, true); offset += 4

        // Write vertices (3 vertices, 3 floats each)
        view.setFloat32(offset, vA.x, true); offset += 4
        view.setFloat32(offset, vA.y, true); offset += 4
        view.setFloat32(offset, vA.z, true); offset += 4

        view.setFloat32(offset, vB.x, true); offset += 4
        view.setFloat32(offset, vB.y, true); offset += 4
        view.setFloat32(offset, vB.z, true); offset += 4

        view.setFloat32(offset, vC.x, true); offset += 4
        view.setFloat32(offset, vC.y, true); offset += 4
        view.setFloat32(offset, vC.z, true); offset += 4

        // Attribute byte count (unused)
        view.setUint16(offset, 0, true); offset += 2
    }

    return buffer
}

/**
 * Extract combined geometry from a part
 */
function getPartGeometry(part: Part): THREE.BufferGeometry | null {
    const geometries: THREE.BufferGeometry[] = []

    part.object.traverse((child) => {
        if (child instanceof THREE.Mesh && child.geometry) {
            // Clone and apply world transform
            const geo = child.geometry.clone()
            geo.applyMatrix4(child.matrixWorld)
            geometries.push(geo)
        }
    })

    if (geometries.length === 0) return null
    if (geometries.length === 1) return geometries[0]

    // Merge geometries
    const merged = new THREE.BufferGeometry()
    const positions: number[] = []
    const indices: number[] = []
    let indexOffset = 0

    for (const geo of geometries) {
        const pos = geo.attributes.position
        const idx = geo.index

        for (let i = 0; i < pos.count; i++) {
            positions.push(pos.getX(i), pos.getY(i), pos.getZ(i))
        }

        if (idx) {
            for (let i = 0; i < idx.count; i++) {
                indices.push(idx.getX(i) + indexOffset)
            }
        } else {
            for (let i = 0; i < pos.count; i++) {
                indices.push(i + indexOffset)
            }
        }

        indexOffset += pos.count
    }

    merged.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    merged.setIndex(indices)
    return merged
}

/**
 * Export parts to STL files
 * Returns a map of filename -> ArrayBuffer
 */
export function exportPartsToSTL(parts: Part[]): Map<string, ArrayBuffer> {
    const files = new Map<string, ArrayBuffer>()

    parts.forEach((part, index) => {
        const geometry = getPartGeometry(part)
        if (geometry) {
            const name = part.name.replace('.step', '')
            const stlData = geometryToSTL(geometry, name)
            files.set(`${name}.stl`, stlData)
        }
    })

    return files
}

/**
 * Export the assembly to MuJoCo MJCF format with proper mesh references
 */
export function exportToMJCF(parts: Part[], mates: Mate[]): string {
    const lines: string[] = []

    lines.push('<?xml version="1.0" encoding="utf-8"?>')
    lines.push('<mujoco model="makeshit_assembly">')
    lines.push('  <compiler angle="radian" coordinate="local" meshdir="meshes"/>')
    lines.push('')
    lines.push('  <option gravity="0 0 -9.81" timestep="0.002"/>')
    lines.push('')

    // Default settings
    lines.push('  <default>')
    lines.push('    <joint damping="0.5"/>')
    lines.push('    <geom contype="1" conaffinity="1" friction="1 0.5 0.5"/>')
    lines.push('  </default>')
    lines.push('')

    // Assets - meshes (STL files)
    lines.push('  <asset>')
    parts.forEach((part) => {
        const meshName = part.name.replace('.step', '')
        // Scale from mm to meters (STEP files are typically in mm)
        lines.push(`    <mesh name="${meshName}" file="${meshName}.stl" scale="0.001 0.001 0.001"/>`)
    })
    lines.push('  </asset>')
    lines.push('')

    // Worldbody
    lines.push('  <worldbody>')
    lines.push('    <light name="top" pos="0 0 1" dir="0 0 -1" diffuse="1 1 1"/>')
    lines.push('    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.3 0.3 0.3 1"/>')
    lines.push('')

    // Parts as bodies
    parts.forEach((part, index) => {
        const pos = part.object.position
        const meshName = part.name.replace('.step', '')

        // Convert position from mm to meters
        const px = (pos.x * 0.001).toFixed(4)
        const py = (pos.y * 0.001).toFixed(4)
        const pz = (pos.z * 0.001 + 0.05).toFixed(4) // Lift above ground

        if (index === 0) {
            // First part is fixed (motor)
            lines.push(`    <body name="${meshName}" pos="${px} ${py} ${pz}">`)
            lines.push(`      <geom type="mesh" mesh="${meshName}" rgba="0.4 0.4 0.4 1"/>`)

            // Add child bodies for mated parts
            mates.forEach((mate, mateIdx) => {
                if (mate.face1.partId === part.id) {
                    const childPart = parts.find(p => p.id === mate.face2.partId)
                    if (childPart) {
                        const childMeshName = childPart.name.replace('.step', '')
                        const childPos = childPart.object.position
                        // Relative position
                        const relX = ((childPos.x - pos.x) * 0.001).toFixed(4)
                        const relY = ((childPos.y - pos.y) * 0.001).toFixed(4)
                        const relZ = ((childPos.z - pos.z) * 0.001).toFixed(4)

                        lines.push('')
                        lines.push(`      <!-- Mated: ${childMeshName} -->`)
                        lines.push(`      <body name="${childMeshName}" pos="${relX} ${relY} ${relZ}">`)

                        if (mate.type === 'cylindrical') {
                            lines.push(`        <joint name="joint_${mateIdx}" type="hinge" axis="0 0 1" range="-3.14159 3.14159"/>`)
                        } else {
                            lines.push(`        <joint name="joint_${mateIdx}" type="slide" axis="0 0 1" range="-0.01 0.01"/>`)
                        }

                        lines.push(`        <geom type="mesh" mesh="${childMeshName}" rgba="0.9 0.9 0.9 1"/>`)
                        lines.push(`      </body>`)
                    }
                }
            })

            lines.push('    </body>')
        }
    })

    lines.push('')
    lines.push('  </worldbody>')
    lines.push('')

    // Actuators
    lines.push('  <actuator>')
    mates.forEach((mate, index) => {
        if (mate.type === 'cylindrical') {
            lines.push(`    <motor name="motor_${index}" joint="joint_${index}" gear="1" ctrllimited="true" ctrlrange="-1 1"/>`)
        }
    })
    lines.push('  </actuator>')
    lines.push('')

    lines.push('</mujoco>')

    return lines.join('\n')
}
