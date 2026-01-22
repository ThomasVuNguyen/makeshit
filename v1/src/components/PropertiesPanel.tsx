import { useAssemblyStore } from '../hooks/useAssemblyStore';
import './PropertiesPanel.css';

export function PropertiesPanel() {
    const {
        parts,
        joints,
        selectedPartId,
        selectedJointId,
        updatePartPosition,
        updatePartRotation,
        updatePartScale,
        updatePartColor,
        updateJoint,
        setPivotFromCurrentPosition,
    } = useAssemblyStore();

    const selectedPart = parts.find((p) => p.id === selectedPartId);
    const selectedJoint = joints.find((j) => j.id === selectedJointId);

    if (!selectedPart && !selectedJoint) {
        return (
            <div className="properties-panel">
                <h3 className="panel-title">Properties</h3>
                <div className="empty-message">
                    Select a part or joint to view properties.
                </div>
            </div>
        );
    }

    if (selectedPart) {
        return (
            <div className="properties-panel">
                <h3 className="panel-title">Part Properties</h3>
                <div className="properties-content">
                    <div className="property-group">
                        <label className="property-label">Color</label>
                        <input
                            type="color"
                            className="color-input"
                            value={selectedPart.color.startsWith('hsl') ? '#888888' : selectedPart.color}
                            onChange={(e) => updatePartColor(selectedPart.id, e.target.value)}
                        />
                    </div>

                    <div className="property-group">
                        <label className="property-label">Position</label>
                        <div className="vector-inputs">
                            <div className="vector-input">
                                <span className="axis-label x">X</span>
                                <input
                                    type="number"
                                    step="0.1"
                                    value={selectedPart.position[0].toFixed(2)}
                                    onChange={(e) =>
                                        updatePartPosition(selectedPart.id, [
                                            parseFloat(e.target.value) || 0,
                                            selectedPart.position[1],
                                            selectedPart.position[2],
                                        ])
                                    }
                                />
                            </div>
                            <div className="vector-input">
                                <span className="axis-label y">Y</span>
                                <input
                                    type="number"
                                    step="0.1"
                                    value={selectedPart.position[1].toFixed(2)}
                                    onChange={(e) =>
                                        updatePartPosition(selectedPart.id, [
                                            selectedPart.position[0],
                                            parseFloat(e.target.value) || 0,
                                            selectedPart.position[2],
                                        ])
                                    }
                                />
                            </div>
                            <div className="vector-input">
                                <span className="axis-label z">Z</span>
                                <input
                                    type="number"
                                    step="0.1"
                                    value={selectedPart.position[2].toFixed(2)}
                                    onChange={(e) =>
                                        updatePartPosition(selectedPart.id, [
                                            selectedPart.position[0],
                                            selectedPart.position[1],
                                            parseFloat(e.target.value) || 0,
                                        ])
                                    }
                                />
                            </div>
                        </div>
                    </div>

                    <div className="property-group">
                        <label className="property-label">Rotation (deg)</label>
                        <div className="vector-inputs">
                            <div className="vector-input">
                                <span className="axis-label x">X</span>
                                <input
                                    type="number"
                                    step="5"
                                    value={((selectedPart.rotation[0] * 180) / Math.PI).toFixed(1)}
                                    onChange={(e) =>
                                        updatePartRotation(selectedPart.id, [
                                            ((parseFloat(e.target.value) || 0) * Math.PI) / 180,
                                            selectedPart.rotation[1],
                                            selectedPart.rotation[2],
                                        ])
                                    }
                                />
                            </div>
                            <div className="vector-input">
                                <span className="axis-label y">Y</span>
                                <input
                                    type="number"
                                    step="5"
                                    value={((selectedPart.rotation[1] * 180) / Math.PI).toFixed(1)}
                                    onChange={(e) =>
                                        updatePartRotation(selectedPart.id, [
                                            selectedPart.rotation[0],
                                            ((parseFloat(e.target.value) || 0) * Math.PI) / 180,
                                            selectedPart.rotation[2],
                                        ])
                                    }
                                />
                            </div>
                            <div className="vector-input">
                                <span className="axis-label z">Z</span>
                                <input
                                    type="number"
                                    step="5"
                                    value={((selectedPart.rotation[2] * 180) / Math.PI).toFixed(1)}
                                    onChange={(e) =>
                                        updatePartRotation(selectedPart.id, [
                                            selectedPart.rotation[0],
                                            selectedPart.rotation[1],
                                            ((parseFloat(e.target.value) || 0) * Math.PI) / 180,
                                        ])
                                    }
                                />
                            </div>
                        </div>
                    </div>

                    <div className="property-group">
                        <label className="property-label">Scale</label>
                        <div className="vector-inputs">
                            <div className="vector-input">
                                <span className="axis-label x">X</span>
                                <input
                                    type="number"
                                    step="0.1"
                                    min="0.01"
                                    value={selectedPart.scale[0].toFixed(2)}
                                    onChange={(e) =>
                                        updatePartScale(selectedPart.id, [
                                            parseFloat(e.target.value) || 0.01,
                                            selectedPart.scale[1],
                                            selectedPart.scale[2],
                                        ])
                                    }
                                />
                            </div>
                            <div className="vector-input">
                                <span className="axis-label y">Y</span>
                                <input
                                    type="number"
                                    step="0.1"
                                    min="0.01"
                                    value={selectedPart.scale[1].toFixed(2)}
                                    onChange={(e) =>
                                        updatePartScale(selectedPart.id, [
                                            selectedPart.scale[0],
                                            parseFloat(e.target.value) || 0.01,
                                            selectedPart.scale[2],
                                        ])
                                    }
                                />
                            </div>
                            <div className="vector-input">
                                <span className="axis-label z">Z</span>
                                <input
                                    type="number"
                                    step="0.1"
                                    min="0.01"
                                    value={selectedPart.scale[2].toFixed(2)}
                                    onChange={(e) =>
                                        updatePartScale(selectedPart.id, [
                                            selectedPart.scale[0],
                                            selectedPart.scale[1],
                                            parseFloat(e.target.value) || 0.01,
                                        ])
                                    }
                                />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    if (selectedJoint) {
        const parentPart = parts.find((p) => p.id === selectedJoint.parentId);
        const childPart = parts.find((p) => p.id === selectedJoint.childId);

        return (
            <div className="properties-panel">
                <h3 className="panel-title">Joint Properties</h3>
                <div className="properties-content">
                    <div className="property-group">
                        <label className="property-label">Name</label>
                        <input
                            type="text"
                            className="text-input"
                            value={selectedJoint.name}
                            onChange={(e) =>
                                updateJoint(selectedJoint.id, { name: e.target.value })
                            }
                        />
                    </div>

                    <div className="property-group">
                        <label className="property-label">Type</label>
                        <select
                            className="select-input"
                            value={selectedJoint.type}
                            onChange={(e) =>
                                updateJoint(selectedJoint.id, {
                                    type: e.target.value as 'hinge' | 'slide' | 'ball' | 'fixed',
                                })
                            }
                        >
                            <option value="hinge">Hinge (Revolute)</option>
                            <option value="slide">Slide (Prismatic)</option>
                            <option value="ball">Ball (Spherical)</option>
                            <option value="fixed">Fixed</option>
                        </select>
                    </div>

                    <div className="property-group">
                        <label className="property-label">Parent</label>
                        <div className="info-text">{parentPart?.name || 'Unknown'}</div>
                    </div>

                    <div className="property-group">
                        <label className="property-label">Child</label>
                        <div className="info-text">{childPart?.name || 'Unknown'}</div>
                    </div>

                    <div className="property-group">
                        <label className="property-label">Axis</label>
                        <select
                            className="select-input"
                            value={selectedJoint.axis}
                            onChange={(e) =>
                                updateJoint(selectedJoint.id, {
                                    axis: e.target.value as 'x' | 'y' | 'z',
                                })
                            }
                        >
                            <option value="x">X</option>
                            <option value="y">Y</option>
                            <option value="z">Z</option>
                        </select>
                    </div>

                    {/* Set Pivot Button */}
                    <div className="property-group">
                        <button
                            className={`set-pivot-button ${selectedJoint.pivotSet ? 'pivot-set' : ''}`}
                            onClick={() => setPivotFromCurrentPosition(selectedJoint.id)}
                        >
                            {selectedJoint.pivotSet ? '✓ Pivot Set' : '📍 Set Pivot Here'}
                        </button>
                        <div className="pivot-hint">
                            {selectedJoint.pivotSet
                                ? 'Pivot locked at current child position'
                                : 'Position the child part, then click to lock pivot'}
                        </div>
                    </div>

                    {selectedJoint.type !== 'fixed' && selectedJoint.type !== 'ball' && (
                        <div className="property-group">
                            <label className="property-label">
                                Limits {selectedJoint.type === 'hinge' ? '(rad)' : '(m)'}
                            </label>
                            <div className="limit-inputs">
                                <div className="limit-input">
                                    <span className="limit-label">Min</span>
                                    <input
                                        type="number"
                                        step={selectedJoint.type === 'hinge' ? 0.1 : 0.01}
                                        value={selectedJoint.limitLower.toFixed(2)}
                                        onChange={(e) =>
                                            updateJoint(selectedJoint.id, {
                                                limitLower: parseFloat(e.target.value) || 0,
                                            })
                                        }
                                    />
                                </div>
                                <div className="limit-input">
                                    <span className="limit-label">Max</span>
                                    <input
                                        type="number"
                                        step={selectedJoint.type === 'hinge' ? 0.1 : 0.01}
                                        value={selectedJoint.limitUpper.toFixed(2)}
                                        onChange={(e) =>
                                            updateJoint(selectedJoint.id, {
                                                limitUpper: parseFloat(e.target.value) || 0,
                                            })
                                        }
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Joint Preview Slider */}
                    {selectedJoint.type !== 'fixed' && (
                        <div className="property-group preview-group">
                            <label className="property-label">🎬 Preview Motion</label>
                            <input
                                type="range"
                                className="preview-slider"
                                min="0"
                                max="1"
                                step="0.01"
                                value={selectedJoint.previewValue}
                                onChange={(e) =>
                                    updateJoint(selectedJoint.id, {
                                        previewValue: parseFloat(e.target.value),
                                    })
                                }
                            />
                            <div className="preview-value">
                                {selectedJoint.type === 'hinge'
                                    ? `${((selectedJoint.limitLower + (selectedJoint.limitUpper - selectedJoint.limitLower) * selectedJoint.previewValue) * 180 / Math.PI).toFixed(1)}°`
                                    : selectedJoint.type === 'slide'
                                        ? `${(selectedJoint.limitLower + (selectedJoint.limitUpper - selectedJoint.limitLower) * selectedJoint.previewValue).toFixed(3)}m`
                                        : 'Drag to preview'
                                }
                            </div>
                        </div>
                    )}
                </div>
            </div>
        );
    }

    return null;
}
