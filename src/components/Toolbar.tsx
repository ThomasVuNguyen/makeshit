import { useRef } from 'react';
import { useAssemblyStore } from '../hooks/useAssemblyStore';
import { loadCADFile } from '../lib/loaders';
import { downloadMJCFExport } from '../lib/mjcfExporter';
import './Toolbar.css';

export function Toolbar() {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const {
        name,
        parts,
        joints,
        transformMode,
        setTransformMode,
        addPart,
        setName,
        clearAssembly,
        alignMode,
        alignType,
        startAlignMode,
        cancelAlignMode,
        alignFlip,
        toggleAlignFlip,
    } = useAssemblyStore();

    const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files) return;

        for (const file of Array.from(files)) {
            try {
                const geometry = await loadCADFile(file);
                addPart(geometry, file.name);
            } catch (error) {
                console.error('Failed to load file:', error);
                alert(`Failed to load ${file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
            }
        }

        // Reset input
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleExport = async () => {
        if (parts.length === 0) {
            alert('Add some parts before exporting!');
            return;
        }

        try {
            await downloadMJCFExport(name, parts, joints);
        } catch (error) {
            console.error('Export failed:', error);
            alert(`Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    };

    return (
        <div className="toolbar">
            <div className="toolbar-section">
                <input
                    type="text"
                    className="assembly-name-input"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Assembly Name"
                />
            </div>

            <div className="toolbar-section">
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".stl,.step,.stp"
                    multiple
                    onChange={handleImport}
                    style={{ display: 'none' }}
                    id="file-import"
                />
                <label htmlFor="file-import" className="toolbar-button">
                    📥 Import CAD
                </label>

                <button className="toolbar-button export" onClick={handleExport}>
                    📤 Export MJCF
                </button>
            </div>

            <div className="toolbar-section">
                <span className="toolbar-label">Transform:</span>
                <button
                    className={`toolbar-button ${transformMode === 'translate' ? 'active' : ''}`}
                    onClick={() => setTransformMode('translate')}
                    title="Translate (G)"
                >
                    ↔️ Move
                </button>
                <button
                    className={`toolbar-button ${transformMode === 'rotate' ? 'active' : ''}`}
                    onClick={() => setTransformMode('rotate')}
                    title="Rotate (R)"
                >
                    🔄 Rotate
                </button>
                <button
                    className={`toolbar-button ${transformMode === 'scale' ? 'active' : ''}`}
                    onClick={() => setTransformMode('scale')}
                    title="Scale (S)"
                >
                    📐 Scale
                </button>
            </div>

            <div className="toolbar-section">
                <span className="toolbar-label">Connections:</span>
                <button
                    className={`toolbar-button ${alignMode && alignType === 'point' ? 'active' : ''}`}
                    onClick={() => alignMode ? cancelAlignMode() : startAlignMode('point')}
                    title="Align Points (A)"
                >
                    {alignMode && alignType === 'point' ? '❌' : '🎯 Point'}
                </button>
                <button
                    className={`toolbar-button ${alignMode && alignType === 'cylinder' ? 'active' : ''}`}
                    onClick={() => alignMode ? cancelAlignMode() : startAlignMode('cylinder')}
                    title="Smart Cylindrical/Plane Align"
                >
                    {alignMode && alignType === 'cylinder' ? '❌' : '⚙️ Cylindrical'}
                </button>
                <button
                    className={`toolbar-button ${alignMode && alignType === 'axis' ? 'active' : ''}`}
                    onClick={() => alignMode ? cancelAlignMode() : startAlignMode('axis')}
                    title="Align 2 Points (Rim)"
                >
                    {alignMode && alignType === 'axis' ? '❌' : '📏 Axis (2-Pt)'}
                </button>

                {alignMode && (
                    <button
                        className={`toolbar-button ${alignFlip ? 'active' : ''}`}
                        onClick={toggleAlignFlip}
                        title="Flip Alignment Direction"
                        style={{ marginLeft: '8px', border: '1px solid #aaa' }}
                    >
                        {alignFlip ? '🔃 Flipped' : '🔃 Flip'}
                    </button>
                )}
            </div>

            <div className="toolbar-section">
                <button className="toolbar-button danger" onClick={clearAssembly}>
                    🗑️ Clear All
                </button>
            </div>
        </div>
    );
}
