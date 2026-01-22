import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';

// occt-import-js is loaded dynamically to avoid blocking initial load
let occtImportJs: typeof import('occt-import-js') | null = null;

/**
 * Load an STL file and return the geometry
 */
export async function loadSTL(file: File): Promise<THREE.BufferGeometry> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (event) => {
            try {
                const contents = event.target?.result;
                if (!contents) {
                    reject(new Error('Failed to read file'));
                    return;
                }

                const loader = new STLLoader();
                const geometry = loader.parse(contents as ArrayBuffer);

                // Compute normals if not present
                if (!geometry.attributes.normal) {
                    geometry.computeVertexNormals();
                }

                resolve(geometry);
            } catch (error) {
                reject(error);
            }
        };

        reader.onerror = () => {
            reject(new Error('Failed to read file'));
        };

        reader.readAsArrayBuffer(file);
    });
}

/**
 * Load a STEP file and return the geometry using occt-import-js
 */
export async function loadSTEP(file: File): Promise<THREE.BufferGeometry> {
    // Dynamically import occt-import-js if not already loaded
    if (!occtImportJs) {
        try {
            // @ts-ignore
            occtImportJs = await import('occt-import-js');
        } catch (error) {
            throw new Error('Failed to load STEP file support. Please ensure occt-import-js is installed.');
        }
    }

    // Initialize the WASM module
    // @ts-ignore
    const occt = await occtImportJs.default();

    // Read the file as ArrayBuffer
    const fileBuffer = await file.arrayBuffer();
    const fileContent = new Uint8Array(fileBuffer);

    // Import the STEP file
    const result = occt.ReadStepFile(fileContent, null);

    if (!result.success) {
        throw new Error('Failed to parse STEP file. The file may be corrupted or in an unsupported format.');
    }

    // Convert to Three.js geometry
    const geometry = new THREE.BufferGeometry();

    // Collect all vertices and indices from all meshes
    const allVertices: number[] = [];
    const allNormals: number[] = [];
    const allIndices: number[] = [];
    let vertexOffset = 0;

    // Geometry Metadata
    const allBrepFaces: { first: number; last: number; color: number[] | null }[] = [];
    let triangleOffset = 0;

    for (const mesh of result.meshes as any[]) {
        // Add vertices
        for (let i = 0; i < mesh.attributes.position.array.length; i++) {
            allVertices.push(mesh.attributes.position.array[i]);
        }

        // Add normals if available
        if (mesh.attributes.normal) {
            for (let i = 0; i < mesh.attributes.normal.array.length; i++) {
                allNormals.push(mesh.attributes.normal.array[i]);
            }
        }

        // Add indices with offset
        let meshTriangleCount = 0;
        if (mesh.index) {
            for (let i = 0; i < mesh.index.array.length; i++) {
                allIndices.push(mesh.index.array[i] + vertexOffset);
            }
            meshTriangleCount = mesh.index.array.length / 3;
        } else {
            // If no indices, we can't easily merge without generating them or assuming non-indexed
            // OCCT usually returns indexed meshes. If not, we might need to handle it.
            // For now, assume indexed.
            console.warn('Mesh missing indices, geometry inference might fail.');
        }

        // Merge Brep Faces
        if (mesh.brep_faces) {
            for (const face of mesh.brep_faces) {
                allBrepFaces.push({
                    first: face.first + triangleOffset,
                    last: face.last + triangleOffset,
                    color: face.color
                });
            }
        }

        vertexOffset += mesh.attributes.position.array.length / 3;
        triangleOffset += meshTriangleCount;
    }

    // Set geometry attributes
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(allVertices, 3));

    if (allNormals.length > 0) {
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(allNormals, 3));
    } else {
        geometry.computeVertexNormals();
    }

    if (allIndices.length > 0) {
        geometry.setIndex(allIndices);
    }

    // Attach Metadata for Alignment System
    geometry.userData.brepFaces = allBrepFaces;

    // Center geometry for better UX
    geometry.computeBoundingBox();
    geometry.center();

    return geometry;
}

/**
 * Determine file type and load appropriately
 */
export async function loadCADFile(file: File): Promise<THREE.BufferGeometry> {
    const extension = file.name.split('.').pop()?.toLowerCase();

    switch (extension) {
        case 'stl':
            return loadSTL(file);
        case 'step':
        case 'stp':
            return loadSTEP(file);
        default:
            throw new Error(`Unsupported file format: .${extension}`);
    }
}

/**
 * Export geometry to STL ArrayBuffer for MuJoCo mesh reference
 */
export function geometryToSTL(geometry: THREE.BufferGeometry): ArrayBuffer {
    const positions = geometry.attributes.position;
    const indices = geometry.index;

    let triangleCount: number;

    if (indices) {
        triangleCount = indices.count / 3;
    } else {
        triangleCount = positions.count / 3;
    }

    // STL binary format
    const bufferLength = 84 + triangleCount * 50;
    const buffer = new ArrayBuffer(bufferLength);
    const view = new DataView(buffer);

    // Header (80 bytes)
    for (let i = 0; i < 80; i++) {
        view.setUint8(i, 0);
    }

    // Triangle count
    view.setUint32(80, triangleCount, true);

    let offset = 84;
    const normal = new THREE.Vector3();
    const v0 = new THREE.Vector3();
    const v1 = new THREE.Vector3();
    const v2 = new THREE.Vector3();

    for (let i = 0; i < triangleCount; i++) {
        let i0: number, i1: number, i2: number;

        if (indices) {
            i0 = indices.getX(i * 3);
            i1 = indices.getX(i * 3 + 1);
            i2 = indices.getX(i * 3 + 2);
        } else {
            i0 = i * 3;
            i1 = i * 3 + 1;
            i2 = i * 3 + 2;
        }

        v0.fromBufferAttribute(positions, i0);
        v1.fromBufferAttribute(positions, i1);
        v2.fromBufferAttribute(positions, i2);

        // Compute normal
        const edge1 = v1.clone().sub(v0);
        const edge2 = v2.clone().sub(v0);
        normal.crossVectors(edge1, edge2).normalize();

        // Normal
        view.setFloat32(offset, normal.x, true); offset += 4;
        view.setFloat32(offset, normal.y, true); offset += 4;
        view.setFloat32(offset, normal.z, true); offset += 4;

        // Vertices
        view.setFloat32(offset, v0.x, true); offset += 4;
        view.setFloat32(offset, v0.y, true); offset += 4;
        view.setFloat32(offset, v0.z, true); offset += 4;

        view.setFloat32(offset, v1.x, true); offset += 4;
        view.setFloat32(offset, v1.y, true); offset += 4;
        view.setFloat32(offset, v1.z, true); offset += 4;

        view.setFloat32(offset, v2.x, true); offset += 4;
        view.setFloat32(offset, v2.y, true); offset += 4;
        view.setFloat32(offset, v2.z, true); offset += 4;

        // Attribute byte count
        view.setUint16(offset, 0, true); offset += 2;
    }

    return buffer;
}
