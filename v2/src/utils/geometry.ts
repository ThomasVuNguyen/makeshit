import * as THREE from 'three';

export interface SurfaceData {
    type: 'plane' | 'cylinder' | 'unknown'
    indices: number[]
    normal: THREE.Vector3  // For plane: face normal. For cylinder: axis direction
    point: THREE.Vector3   // Center point of the surface
    axis?: THREE.Vector3   // For cylinder: estimated rotation axis
    radius?: number
}

// Helper to get triangle normal
const _vA = new THREE.Vector3();
const _vB = new THREE.Vector3();
const _vC = new THREE.Vector3();
const _cb = new THREE.Vector3();
const _ab = new THREE.Vector3();

function getFaceNormal(geometry: THREE.BufferGeometry, faceIndex: number, target: THREE.Vector3) {
    const pos = geometry.attributes.position;
    const idx = geometry.index;

    if (idx) {
        const a = idx.getX(faceIndex * 3);
        const b = idx.getX(faceIndex * 3 + 1);
        const c = idx.getX(faceIndex * 3 + 2);
        _vA.fromBufferAttribute(pos, a);
        _vB.fromBufferAttribute(pos, b);
        _vC.fromBufferAttribute(pos, c);
    } else {
        const a = faceIndex * 3;
        _vA.fromBufferAttribute(pos, a);
        _vB.fromBufferAttribute(pos, a + 1);
        _vC.fromBufferAttribute(pos, a + 2);
    }

    _cb.subVectors(_vC, _vB);
    _ab.subVectors(_vA, _vB);
    _cb.cross(_ab);
    _cb.normalize();
    target.copy(_cb);
    return target;
}

function getFaceCentroid(geometry: THREE.BufferGeometry, faceIndex: number, target: THREE.Vector3) {
    const pos = geometry.attributes.position;
    const idx = geometry.index;

    if (idx) {
        const a = idx.getX(faceIndex * 3);
        const b = idx.getX(faceIndex * 3 + 1);
        const c = idx.getX(faceIndex * 3 + 2);
        _vA.fromBufferAttribute(pos, a);
        _vB.fromBufferAttribute(pos, b);
        _vC.fromBufferAttribute(pos, c);
    } else {
        const a = faceIndex * 3;
        _vA.fromBufferAttribute(pos, a);
        _vB.fromBufferAttribute(pos, a + 1);
        _vC.fromBufferAttribute(pos, a + 2);
    }

    target.copy(_vA).add(_vB).add(_vC).divideScalar(3);
    return target;
}

// Estimate cylinder axis from a set of normals using cross products
function estimateCylinderAxis(normals: THREE.Vector3[]): THREE.Vector3 {
    // For a cylinder, all normals are perpendicular to the axis
    // Take cross products of pairs of normals to estimate the axis
    const axis = new THREE.Vector3();
    let count = 0;

    for (let i = 0; i < Math.min(normals.length - 1, 20); i++) {
        const n1 = normals[i];
        const n2 = normals[i + 1];
        const cross = new THREE.Vector3().crossVectors(n1, n2);

        if (cross.length() > 0.01) { // Skip nearly parallel normals
            cross.normalize();

            // Ensure consistent direction
            if (axis.length() > 0 && axis.dot(cross) < 0) {
                cross.negate();
            }

            axis.add(cross);
            count++;
        }
    }

    if (count > 0) {
        axis.divideScalar(count).normalize();
    } else {
        // Fallback to Y axis if we can't determine
        axis.set(0, 1, 0);
    }

    return axis;
}

export function detectSurface(geometry: THREE.BufferGeometry, startFaceIndex: number): SurfaceData {
    const startNormal = new THREE.Vector3();
    getFaceNormal(geometry, startFaceIndex, startNormal);

    const indices = geometry.index;

    if (!indices) {
        const centroid = new THREE.Vector3();
        getFaceCentroid(geometry, startFaceIndex, centroid);
        return {
            type: 'plane',
            indices: [startFaceIndex],
            normal: startNormal,
            point: centroid
        };
    }

    // Build/Get Vertex Map
    if (!geometry.userData.vertexMap) {
        const map = new Map<number, number[]>();
        for (let i = 0; i < indices.count / 3; i++) {
            const a = indices.getX(i * 3);
            const b = indices.getX(i * 3 + 1);
            const c = indices.getX(i * 3 + 2);
            if (!map.has(a)) map.set(a, []);
            map.get(a)!.push(i);
            if (!map.has(b)) map.set(b, []);
            map.get(b)!.push(i);
            if (!map.has(c)) map.set(c, []);
            map.get(c)!.push(i);
        }
        geometry.userData.vertexMap = map;
    }

    const vMap = geometry.userData.vertexMap as Map<number, number[]>;

    const visited = new Set<number>();
    const queue = [startFaceIndex];
    visited.add(startFaceIndex);

    const patchIndices: number[] = [];
    const patchNormals: THREE.Vector3[] = [];
    const patchCentroids: THREE.Vector3[] = [];

    let count = 0;
    const limit = 2000;

    while (queue.length > 0 && count < limit) {
        count++;
        const f = queue.shift()!;
        patchIndices.push(f);

        const n = new THREE.Vector3();
        getFaceNormal(geometry, f, n);
        patchNormals.push(n);

        const centroid = new THREE.Vector3();
        getFaceCentroid(geometry, f, centroid);
        patchCentroids.push(centroid);

        // Neighbors
        const a = indices.getX(f * 3);
        const b = indices.getX(f * 3 + 1);
        const c = indices.getX(f * 3 + 2);
        const vs = [a, b, c];

        for (const v of vs) {
            const neighbors = vMap.get(v);
            if (neighbors) {
                for (const nf of neighbors) {
                    if (visited.has(nf)) continue;

                    const n2 = new THREE.Vector3();
                    getFaceNormal(geometry, nf, n2);
                    const angle = n.angleTo(n2);

                    if (angle < THREE.MathUtils.degToRad(30)) {
                        visited.add(nf);
                        queue.push(nf);
                    }
                }
            }
        }
    }

    // Calculate center point of surface
    const centerPoint = new THREE.Vector3();
    for (const c of patchCentroids) centerPoint.add(c);
    centerPoint.divideScalar(patchCentroids.length);

    // Analyze normal variance
    const avgNormal = new THREE.Vector3();
    for (const n of patchNormals) avgNormal.add(n);
    avgNormal.divideScalar(patchNormals.length).normalize();

    let maxAngle = 0;
    for (const n of patchNormals) {
        maxAngle = Math.max(maxAngle, n.angleTo(avgNormal));
    }

    if (maxAngle < 0.1) { // ~5 degrees = Planar
        return {
            type: 'plane',
            indices: patchIndices,
            normal: avgNormal,
            point: centerPoint
        };
    }

    // Cylindrical - estimate axis
    const axis = estimateCylinderAxis(patchNormals);

    return {
        type: 'cylinder',
        indices: patchIndices,
        normal: avgNormal,
        point: centerPoint,
        axis: axis
    };
}
