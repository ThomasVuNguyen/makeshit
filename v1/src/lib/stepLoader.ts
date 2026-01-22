
import * as THREE from 'three';

// Define the shape of the OCCT result based on documentation
interface OCCTMesh {
    name: string;
    color?: number[];
    brep_faces: {
        first: number;
        last: number;
        color: number[] | null;
    }[];
    attributes: {
        position: { array: any }; // Float32Array
        normal?: { array: any };
    };
    index: {
        array: any; // Int32Array or similar
    };
}

interface OCCTResult {
    success: boolean;
    root: any;
    meshes: OCCTMesh[];
}

export const loadStepFile = async (file: File): Promise<THREE.BufferGeometry> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const buffer = e.target?.result as ArrayBuffer;
                const fileBuffer = new Uint8Array(buffer);

                // Dynamically load the global occtimportjs function if it's not available
                // We rely on the script being in public/ and loaded, OR we can try to rely on the bundler.
                // Since we copied it to public, let's assume we load it via a utility or it's global.
                // Ideally, we'd use the npm package, but it seems designed for a script tag or node. 
                // Let's try to use the global window object.

                // Note: In a real prod app, we'd manage this script loading better.
                if (!(window as any).occtimportjs) {
                    // Lazy load the script
                    await new Promise<void>((resolveScript, rejectScript) => {
                        const script = document.createElement('script');
                        script.src = '/occt-import-js.js';
                        script.onload = () => resolveScript();
                        script.onerror = () => rejectScript(new Error('Failed to load occt-import-js.js'));
                        document.body.appendChild(script);
                    });
                }

                const occtinit = (window as any).occtimportjs;
                if (!occtinit) {
                    throw new Error('OCCT Import JS not found');
                }

                const occt = await occtinit();
                const result: OCCTResult = occt.ReadStepFile(fileBuffer, null);

                if (!result.success) {
                    throw new Error('Failed to parse STEP file');
                }

                if (result.meshes.length === 0) {
                    throw new Error('No meshes found in STEP file');
                }

                // For MVP, populate the first mesh.
                // Ideally we handle the full scene graph.
                const meshData = result.meshes[0];

                const geometry = new THREE.BufferGeometry();

                const positions = new Float32Array(meshData.attributes.position.array);
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                if (meshData.attributes.normal) {
                    const normals = new Float32Array(meshData.attributes.normal.array);
                    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
                } else {
                    geometry.computeVertexNormals();
                }

                // Indices
                if (meshData.index) {
                    const indices = new Uint32Array(meshData.index.array);
                    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                }

                // Attach metadata
                geometry.userData.brepFaces = meshData.brep_faces;

                // Center logic similar to STL
                geometry.computeBoundingBox();
                geometry.center();

                resolve(geometry);

            } catch (err) {
                reject(err);
            }
        };
        reader.onerror = (err) => reject(err);
        reader.readAsArrayBuffer(file);
    });
};
