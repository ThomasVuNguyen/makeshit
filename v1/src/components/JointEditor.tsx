import { useState } from 'react';
import { useAssemblyStore } from '../hooks/useAssemblyStore';
import type { JointType } from '../types/assembly';
import './JointEditor.css';

interface JointEditorProps {
    isOpen: boolean;
    onClose: () => void;
}

export function JointEditor({ isOpen, onClose }: JointEditorProps) {
    const { parts, joints, addJoint, removeJoint, selectJoint } = useAssemblyStore();
    const [parentId, setParentId] = useState<string>('');
    const [childId, setChildId] = useState<string>('');
    const [jointType, setJointType] = useState<JointType>('hinge');

    if (!isOpen) return null;

    const handleCreate = () => {
        if (!parentId || !childId) {
            alert('Please select both parent and child parts.');
            return;
        }

        if (parentId === childId) {
            alert('Parent and child must be different parts.');
            return;
        }

        const id = addJoint(parentId, childId, jointType);
        if (id) {
            selectJoint(id);
            setParentId('');
            setChildId('');
            setJointType('hinge');
            onClose();
        }
    };

    return (
        <div className="joint-editor-overlay" onClick={onClose}>
            <div className="joint-editor-modal" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>Create Joint</h3>
                    <button className="close-button" onClick={onClose}>
                        ✕
                    </button>
                </div>

                <div className="modal-body">
                    <div className="form-group">
                        <label>Parent Part</label>
                        <select
                            value={parentId}
                            onChange={(e) => setParentId(e.target.value)}
                        >
                            <option value="">Select parent...</option>
                            {parts.map((part) => (
                                <option key={part.id} value={part.id}>
                                    {part.name}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Child Part</label>
                        <select
                            value={childId}
                            onChange={(e) => setChildId(e.target.value)}
                        >
                            <option value="">Select child...</option>
                            {parts.filter((p) => p.id !== parentId).map((part) => (
                                <option key={part.id} value={part.id}>
                                    {part.name}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Joint Type</label>
                        <select
                            value={jointType}
                            onChange={(e) => setJointType(e.target.value as JointType)}
                        >
                            <option value="hinge">Hinge (Revolute)</option>
                            <option value="slide">Slide (Prismatic)</option>
                            <option value="ball">Ball (Spherical)</option>
                            <option value="fixed">Fixed</option>
                        </select>
                    </div>
                </div>

                <div className="modal-footer">
                    <button className="cancel-button" onClick={onClose}>
                        Cancel
                    </button>
                    <button className="create-button" onClick={handleCreate}>
                        Create Joint
                    </button>
                </div>

                {/* Existing Joints List */}
                {joints.length > 0 && (
                    <div className="existing-joints">
                        <h4>Existing Joints</h4>
                        <div className="joints-list">
                            {joints.map((joint) => {
                                const parent = parts.find((p) => p.id === joint.parentId);
                                const child = parts.find((p) => p.id === joint.childId);
                                return (
                                    <div
                                        key={joint.id}
                                        className="joint-item"
                                        onClick={() => {
                                            selectJoint(joint.id);
                                            onClose();
                                        }}
                                    >
                                        <span className="joint-info">
                                            <span className="joint-type">{joint.type}</span>
                                            {parent?.name} → {child?.name}
                                        </span>
                                        <button
                                            className="delete-joint"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                removeJoint(joint.id);
                                            }}
                                        >
                                            ✕
                                        </button>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
