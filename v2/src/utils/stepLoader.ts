import initOCCT from 'occt-import-js';
import * as THREE from 'three';

let occtInstance: any = null;

export const initStepLoader = async () => {
    if (occtInstance) return occtInstance;

    occtInstance = await initOCCT({
        locateFile: (name: string) => {
            // Assuming wasm is at the root (public folder)
            return `/${name}`;
        },
    });
    return occtInstance;
};

export const loadStepFile = async (url: string): Promise<THREE.Group> => {
    const occt = await initStepLoader();

    const response = await fetch(url);
    const buffer = await response.arrayBuffer();

    // Read the STEP file
    const fileContent = new Uint8Array(buffer);

    // result structure from occt-import-js
    const result = occt.ReadStepFile(fileContent, null);

    const group = new THREE.Group();

    if (result && result.meshes) {
        for (const meshData of result.meshes) {
            const geometry = new THREE.BufferGeometry();

            // Positions
            if (meshData.attributes.position) {
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(meshData.attributes.position.array, 3));
            }

            // Normals
            if (meshData.attributes.normal) {
                geometry.setAttribute('normal', new THREE.Float32BufferAttribute(meshData.attributes.normal.array, 3));
            }

            // Indices
            if (meshData.index) {
                geometry.setIndex(new THREE.Uint16BufferAttribute(meshData.index.array, 1));
            } else {
                // geometry.setIndex([...Array(positions.length / 3).keys()]); // simplified
            }

            // Color (optional)
            let color = new THREE.Color(0xcccccc);
            if (meshData.color) {
                // meshData.color is usually [r, g, b] 0-1
                color.setRGB(meshData.color[0], meshData.color[1], meshData.color[2]);
            }

            const material = new THREE.MeshStandardMaterial({
                color: color,
                metalness: 0.3,
                roughness: 0.4
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            mesh.name = meshData.name || 'Part';

            // Store metadata for mates integration later
            mesh.userData = {
                ...meshData,
                originalColor: color.clone()
            };

            group.add(mesh);
        }
    }

    return group;
};
