import type { Part, Joint, JointType, Axis } from '../types/assembly';
import { geometryToSTL } from './loaders';

/**
 * Convert axis letter to MuJoCo axis vector
 */
function axisToVector(axis: Axis): string {
    switch (axis) {
        case 'x': return '1 0 0';
        case 'y': return '0 1 0';
        case 'z': return '0 0 1';
    }
}

/**
 * Convert joint type to MuJoCo joint type
 */
function jointTypeToMJCF(type: JointType): string {
    switch (type) {
        case 'hinge': return 'hinge';
        case 'slide': return 'slide';
        case 'ball': return 'ball';
        case 'fixed': return 'fixed';
        case 'cylindrical': return 'slide'; // TODO: Implement composite (slide + hinge) for true cylindrical
    }
}

/**
 * Escape XML special characters
 */
function escapeXml(str: string): string {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&apos;');
}

/**
 * Build body hierarchy from parts and joints
 */
interface BodyNode {
    part: Part;
    children: BodyNode[];
    jointToParent?: Joint;
}

function buildBodyHierarchy(parts: Part[], joints: Joint[]): BodyNode[] {
    // Find parts that are children (have a joint where they are the child)
    const childPartIds = new Set(joints.map(j => j.childId));

    // Root parts are those not connected as children
    const rootParts = parts.filter(p => !childPartIds.has(p.id));

    // Build tree recursively
    function buildNode(part: Part, jointToParent?: Joint): BodyNode {
        const childJoints = joints.filter(j => j.parentId === part.id);
        const children = childJoints.map(j => {
            const childPart = parts.find(p => p.id === j.childId);
            if (!childPart) return null;
            return buildNode(childPart, j);
        }).filter((n): n is BodyNode => n !== null);

        return {
            part,
            children,
            jointToParent,
        };
    }

    return rootParts.map(p => buildNode(p));
}

/**
 * Generate MJCF body element recursively
 */
function generateBodyXML(node: BodyNode, indent: string = '    '): string {
    const { part, children, jointToParent } = node;
    const partName = escapeXml(part.name.replace(/\s+/g, '_'));
    const meshName = `${partName}_mesh`;

    const pos = part.position.map(v => v.toFixed(6)).join(' ');
    const euler = part.rotation.map(v => (v * 180 / Math.PI).toFixed(2)).join(' ');

    let xml = `${indent}<body name="${partName}" pos="${pos}" euler="${euler}">\n`;

    // Add joint if this body has a connection to parent
    if (jointToParent && jointToParent.type !== 'fixed') {
        const joint = jointToParent;
        const jointName = escapeXml(joint.name.replace(/\s+/g, '_'));
        const axis = axisToVector(joint.axis);
        const type = jointTypeToMJCF(joint.type);

        xml += `${indent}  <joint name="${jointName}" type="${type}" axis="${axis}"`;

        if (joint.type === 'hinge' || joint.type === 'slide') {
            xml += ` range="${joint.limitLower.toFixed(4)} ${joint.limitUpper.toFixed(4)}"`;
        }

        xml += `/>\n`;
    }

    // Add geometry
    xml += `${indent}  <geom type="mesh" mesh="${meshName}" rgba="0.7 0.7 0.7 1"/>\n`;

    // Add child bodies
    for (const child of children) {
        xml += generateBodyXML(child, indent + '  ');
    }

    xml += `${indent}</body>\n`;

    return xml;
}

export interface MJCFExportResult {
    xml: string;
    meshFiles: Map<string, ArrayBuffer>; // filename -> STL data
}

/**
 * Export assembly to MuJoCo MJCF format
 */
export function exportToMJCF(
    assemblyName: string,
    parts: Part[],
    joints: Joint[]
): MJCFExportResult {
    const meshFiles = new Map<string, ArrayBuffer>();

    // Generate mesh assets
    let assetXML = '  <asset>\n';

    for (const part of parts) {
        const partName = escapeXml(part.name.replace(/\s+/g, '_'));
        const meshName = `${partName}_mesh`;
        const fileName = `${partName}.stl`;

        // Convert geometry to STL
        meshFiles.set(fileName, geometryToSTL(part.geometry));

        // Add mesh asset reference
        const scale = part.scale.map(v => v.toFixed(6)).join(' ');
        assetXML += `    <mesh name="${meshName}" file="${fileName}" scale="${scale}"/>\n`;
    }

    assetXML += '  </asset>\n';

    // Build body hierarchy
    const rootBodies = buildBodyHierarchy(parts, joints);

    // Generate worldbody
    let worldbodyXML = '  <worldbody>\n';
    worldbodyXML += '    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>\n';
    worldbodyXML += '    <geom type="plane" size="10 10 0.1" rgba="0.9 0.9 0.9 1"/>\n';

    for (const root of rootBodies) {
        worldbodyXML += generateBodyXML(root);
    }

    worldbodyXML += '  </worldbody>\n';

    // Compile full MJCF
    const modelName = escapeXml(assemblyName.replace(/\s+/g, '_'));

    const xml = `<?xml version="1.0" encoding="utf-8"?>
<mujoco model="${modelName}">
  <compiler angle="degree" meshdir="meshes"/>
  
  <option gravity="0 0 -9.81"/>
  
${assetXML}
${worldbodyXML}
</mujoco>
`;

    return { xml, meshFiles };
}

/**
 * Download the MJCF export as a zip file
 */
export async function downloadMJCFExport(
    assemblyName: string,
    parts: Part[],
    joints: Joint[]
): Promise<void> {
    const { xml, meshFiles } = exportToMJCF(assemblyName, parts, joints);

    // For simplicity, we'll download the XML file directly
    // and create a separate download for the meshes folder

    // Download XML
    const xmlBlob = new Blob([xml], { type: 'application/xml' });
    const xmlUrl = URL.createObjectURL(xmlBlob);
    const xmlLink = document.createElement('a');
    xmlLink.href = xmlUrl;
    xmlLink.download = `${assemblyName.replace(/\s+/g, '_')}.xml`;
    xmlLink.click();
    URL.revokeObjectURL(xmlUrl);

    // Download meshes as individual files
    // In a real app, we'd use a zip library here
    for (const [fileName, data] of meshFiles) {
        const blob = new Blob([data], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = fileName;
        // Delay downloads slightly to avoid browser blocking
        await new Promise(resolve => setTimeout(resolve, 100));
        link.click();
        URL.revokeObjectURL(url);
    }
}
