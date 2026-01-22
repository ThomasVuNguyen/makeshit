import { useRef } from 'react'
import { useStore } from '../stores/useStore'
import './ControlPanel.css'

export function ControlPanel() {
    const {
        mateMode,
        mateFace1,
        mateFace2,
        mateFlipped,
        startMate,
        cancelMate,
        toggleFlip,
        confirmMate,
        mates
    } = useStore()

    const fileInputRef = useRef<HTMLInputElement>(null)

    const canConfirm = mateFace1 && mateFace2

    const handleImportClick = () => {
        fileInputRef.current?.click()
    }

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files
        if (!files || files.length === 0) return

        // Dispatch event for each file to be loaded by Viewer
        Array.from(files).forEach(file => {
            window.dispatchEvent(new CustomEvent('importStep', { detail: { file } }))
        })

        // Reset input so the same file can be imported again if needed
        e.target.value = ''
    }

    return (
        <div className="control-panel">
            {/* Hidden file input */}
            <input
                ref={fileInputRef}
                type="file"
                accept=".step,.stp,.STEP,.STP"
                multiple
                style={{ display: 'none' }}
                onChange={handleFileChange}
            />

            {!mateMode ? (
                <>
                    <button
                        className="btn btn-import"
                        onClick={handleImportClick}
                    >
                        📁 Import STEP
                    </button>
                    <button
                        className="btn btn-primary"
                        onClick={() => startMate('cylindrical')}
                    >
                        Cylindrical Mate
                    </button>
                    <button
                        className="btn btn-primary"
                        onClick={() => startMate('planar')}
                    >
                        Planar Mate
                    </button>
                    <button
                        className="btn btn-success"
                        onClick={() => {
                            // Simulate button - will trigger MuJoCo export
                            window.dispatchEvent(new CustomEvent('simulate'))
                        }}
                    >
                        Simulate
                    </button>
                    {mates.length > 0 && (
                        <div className="mate-count">{mates.length} mate(s)</div>
                    )}
                </>
            ) : (
                <>
                    <div className="mode-indicator">
                        {mateMode === 'cylindrical' ? '🔵 Cylindrical Mate' : '🟢 Planar Mate'}
                    </div>

                    <div className="face-status">
                        <div className={`face-slot ${mateFace1 ? 'selected' : ''}`}>
                            Face 1: {mateFace1 ? `${mateFace1.type} (${mateFace1.indices.length} tris)` : 'Click a face...'}
                        </div>
                        <div className={`face-slot ${mateFace2 ? 'selected' : ''}`}>
                            Face 2: {mateFace2 ? `${mateFace2.type} (${mateFace2.indices.length} tris)` : 'Click another part...'}
                        </div>
                    </div>

                    <button
                        className="btn btn-secondary"
                        onClick={toggleFlip}
                    >
                        Flip {mateFlipped ? '↔️' : '↕️'}
                    </button>

                    <button
                        className="btn btn-success"
                        onClick={confirmMate}
                        disabled={!canConfirm}
                    >
                        Confirm
                    </button>

                    <button
                        className="btn btn-danger"
                        onClick={cancelMate}
                    >
                        Cancel
                    </button>
                </>
            )}
        </div>
    )
}
