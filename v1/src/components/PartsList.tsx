import { useAssemblyStore } from '../hooks/useAssemblyStore';
import './PartsList.css';

export function PartsList() {
    const {
        parts,
        selectedPartId,
        selectPart,
        removePart,
        togglePartVisibility,
        updatePartName,
        // Joint stuff
        joints,
        selectJoint,
        removeJoint,
        selectedJointId,
        // Joint creation
        jointCreationMode,
        handlePartClickForJoint
    } = useAssemblyStore();

    if (parts.length === 0) {
        return (
            <div className="parts-list">
                <h3 className="panel-title">Parts</h3>
                <div className="empty-message">
                    No parts loaded.<br />
                    Import STL files to get started.
                </div>
            </div>
        );
    }

    return (
        <div className="parts-list">
            <h3 className="panel-title">Parts ({parts.length})</h3>
            <div className="parts-items">
                {parts.map((part) => (
                    <div
                        key={part.id}
                        className={`part-item ${selectedPartId === part.id ? 'selected' : ''}`}
                        onClick={() => {
                            if (jointCreationMode) {
                                handlePartClickForJoint(part.id);
                            } else {
                                selectPart(part.id);
                            }
                        }}
                    >
                        <div
                            className="part-color"
                            style={{ backgroundColor: part.color }}
                        />
                        <input
                            type="text"
                            className="part-name"
                            value={part.name}
                            onChange={(e) => updatePartName(part.id, e.target.value)}
                            onClick={(e) => e.stopPropagation()}
                        />
                        <button
                            className="part-visibility"
                            onClick={(e) => {
                                e.stopPropagation();
                                togglePartVisibility(part.id);
                            }}
                            title={part.visible ? 'Hide' : 'Show'}
                        >
                            {part.visible ? '👁️' : '👁️‍🗨️'}
                        </button>
                        <button
                            className="part-delete"
                            onClick={(e) => {
                                e.stopPropagation();
                                removePart(part.id);
                            }}
                            title="Delete"
                        >
                            ✕
                        </button>
                    </div>
                ))}
            </div>

            {
                joints.length > 0 && (
                    <>
                        <h3 className="panel-title" style={{ marginTop: '16px', borderTop: '1px solid #333', paddingTop: '16px' }}>
                            Joints ({joints.length})
                        </h3>
                        <div className="parts-items">
                            {joints.map((joint) => {
                                const parentPart = parts.find(p => p.id === joint.parentId);
                                const childPart = parts.find(p => p.id === joint.childId);
                                const jointName = joint.name || `${parentPart?.name || 'Part'} - ${childPart?.name || 'Part'}`;

                                return (
                                    <div
                                        key={joint.id}
                                        className={`part-item ${selectedJointId === joint.id ? 'selected' : ''}`}
                                        onClick={() => selectJoint(joint.id)}
                                    >
                                        <div className="part-color" style={{ backgroundColor: '#646cff', borderRadius: '50%' }} />
                                        <div className="part-name" style={{ fontSize: '0.85em', color: '#aaa' }}>
                                            <span style={{ color: '#fff' }}>{joint.type}</span>: {jointName}
                                        </div>
                                        <button
                                            className="part-delete"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                removeJoint(joint.id);
                                            }}
                                            title="Delete Joint"
                                        >
                                            ✕
                                        </button>
                                    </div>
                                );
                            })}
                        </div>
                    </>
                )
            }
        </div >
    );
}
