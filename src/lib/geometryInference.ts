
import * as THREE from 'three';
import { calculateCircleFrom3Points } from './geometryUtils';

export interface SurfaceData {
    type: 'plane' | 'cylinder' | 'unknown';
    // For Plane
    normal?: THREE.Vector3;
    center?: THREE.Vector3; // Some point on the plane
    // For Cylinder
    axis?: THREE.Vector3;
    origin?: THREE.Vector3; // Point on axis
    radius?: number;
}

export const detectSurfaceType = (
    geometry: THREE.BufferGeometry,
    startIndex: number, // Triangle index
    endIndex: number
): SurfaceData => {
    const posAttr = geometry.getAttribute('position');
    const indexAttr = geometry.index;
    if (!indexAttr) return { type: 'unknown' };

    // Helper to get vertex
    const getPoint = (triIdx: number, vertIdx: number): THREE.Vector3 => {
        const i = indexAttr.getX((startIndex + triIdx) * 3 + vertIdx);
        return new THREE.Vector3().fromBufferAttribute(posAttr, i);
    };

    const numTriangles = endIndex - startIndex + 1;
    if (numTriangles < 1) return { type: 'unknown' };

    // 1. Check Planarity
    // Sample a few normals
    // Calculate normal of first triangle
    const p0 = getPoint(0, 0);
    const p1 = getPoint(0, 1);
    const p2 = getPoint(0, 2);
    const n0 = new THREE.Vector3().subVectors(p1, p0).cross(new THREE.Vector3().subVectors(p2, p0)).normalize();

    let isPlane = true;
    const numSamples = Math.min(10, numTriangles);
    const step = Math.floor(numTriangles / numSamples);

    for (let i = 1; i < numSamples; i++) {
        const triIdx = i * step;
        const v0 = getPoint(triIdx, 0);
        const v1 = getPoint(triIdx, 1);
        const v2 = getPoint(triIdx, 2);
        const n = new THREE.Vector3().subVectors(v1, v0).cross(new THREE.Vector3().subVectors(v2, v0)).normalize();

        if (n.dot(n0) < 0.99) { // 0.99 allows small noise
            isPlane = false;
            break;
        }
    }

    if (isPlane) {
        // Compute centroid approx
        return {
            type: 'plane',
            normal: n0,
            center: p0 // Just return a point on the plane
        };
    }

    // 2. Check Cylindricality
    // Method: Robustly sample 3 points to find a circle. 
    // We try multiple combinations of indices and vertices to avoid collinear points.

    if (numTriangles < 4) return { type: 'unknown' };

    // Helper to try to finding a valid circle from a set of relative offsets
    const findCircle = (offsets: number[], vertIndices: number[]): { circle: any, points: THREE.Vector3[] } | null => {
        const p0 = getPoint(Math.floor(numTriangles * offsets[0]), vertIndices[0]);
        const p1 = getPoint(Math.floor(numTriangles * offsets[1]), vertIndices[1]);
        const p2 = getPoint(Math.floor(numTriangles * offsets[2]), vertIndices[2]);

        const circle = calculateCircleFrom3Points(p0, p1, p2);
        if (!circle) return null;
        if (circle.radius > 10000 || circle.radius < 0.001) return null; // Sanity check
        return { circle, points: [p0, p1, p2] };
    };

    // Attempt strategies for Set 1
    const strategies = [
        { offsets: [0, 0.33, 0.66], verts: [0, 0, 0] },     // Spread out, same vertex
        { offsets: [0, 0.33, 0.66], verts: [0, 1, 2] },     // Spread out, diff vertices (avoids edge alignment)
        { offsets: [0.1, 0.5, 0.9], verts: [0, 0, 0] },     // Wider spread
        { offsets: [0.1, 0.5, 0.9], verts: [1, 1, 1] },     // Spread, diff vertex index
        { offsets: [0, 0.25, 0.5], verts: [0, 1, 2] },      // Closer cluster
    ];

    let c1Data = null;
    for (const strat of strategies) {
        c1Data = findCircle(strat.offsets, strat.verts);
        if (c1Data) break;
    }

    if (!c1Data) return { type: 'unknown' };

    // Attempt strategies for Set 2 (must be distinct from Set 1 to check along axis)
    // We shift the offsets slightly
    const straegies2 = [
        { offsets: [0.15, 0.45, 0.75], verts: [1, 1, 1] },
        { offsets: [0.15, 0.45, 0.75], verts: [2, 0, 1] },
        { offsets: [0.2, 0.6, 0.8], verts: [1, 1, 1] },
        { offsets: [0.05, 0.35, 0.65], verts: [2, 2, 2] },
    ];

    let c2Data = null;
    for (const strat of straegies2) {
        c2Data = findCircle(strat.offsets, strat.verts);
        if (c2Data) break;
    }

    if (!c2Data) return { type: 'unknown' };

    const c1 = c1Data.circle;
    const c2 = c2Data.circle;

    // Compare
    const axisMatch = Math.abs(new THREE.Vector3(...c1.normal).dot(new THREE.Vector3(...c2.normal))) > 0.90; // Relaxed tolerance
    const radiusMatch = Math.abs(c1.radius - c2.radius) < (Math.max(c1.radius, c2.radius) * 0.1); // 10% tolerance

    if (axisMatch && radiusMatch) {
        return {
            type: 'cylinder',
            axis: new THREE.Vector3(...c1.normal),
            origin: new THREE.Vector3(...c1.center),
            radius: c1.radius
        };
    }

    return { type: 'unknown' };
};
