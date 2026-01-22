import * as THREE from 'three';

// Joint types supported by MuJoCo
export type JointType = 'hinge' | 'slide' | 'ball' | 'fixed' | 'cylindrical';

// Axis options
export type Axis = 'x' | 'y' | 'z';

export interface Part {
    id: string;
    name: string;
    fileName: string;
    geometry: THREE.BufferGeometry;
    position: [number, number, number];
    rotation: [number, number, number]; // Euler angles in radians
    scale: [number, number, number];
    color: string;
    visible: boolean;
}

export interface Joint {
    id: string;
    name: string;
    type: JointType;
    parentId: string;  // Part ID
    childId: string;   // Part ID
    axis: Axis;
    limitLower: number;
    limitUpper: number;
    anchorPosition: [number, number, number]; // Pivot position relative to parent
    childOffset: [number, number, number]; // Child position relative to pivot when at rest
    pivotSet: boolean; // Whether pivot has been configured
    previewValue: number; // 0-1 normalized value for previewing joint movement
}

export interface Assembly {
    name: string;
    parts: Part[];
    joints: Joint[];
    selectedPartId: string | null;
    selectedJointId: string | null;
}
