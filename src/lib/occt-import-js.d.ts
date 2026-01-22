declare module 'occt-import-js' {
    interface Position {
        array: number[];
    }

    interface Normal {
        array: number[];
    }

    interface Index {
        array: number[];
    }

    interface Attributes {
        position: Position;
        normal?: Normal;
    }

    interface Mesh {
        attributes: Attributes;
        index?: Index;
    }

    interface ReadResult {
        success: boolean;
        meshes: Mesh[];
    }

    interface OcctModule {
        ReadStepFile(fileContent: Uint8Array, options: null): ReadResult;
    }

    function init(): Promise<OcctModule>;
    export default init;
}
