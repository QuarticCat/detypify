import { Detypify, ortEnv, type SymbolInfo, type Strokes } from "detypify-service";
import { derived, get, writable } from "svelte/store";

ortEnv.wasm.numThreads = 1;
ortEnv.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";

export type Sample = {
    id: string;
    name: string;
    strokes: Strokes;
};

export const session = writable<Detypify | undefined>(undefined);
export const isContribMode = writable(false);
export const strokes = writable<Strokes>([]);
export const input = writable("");
export const samples = writable<Sample[]>([]);

Detypify.create().then((created: Detypify) => {
    session.set(created);
    strokes.set(get(strokes)); // trigger drawing
});

export const candidates = derived(
    strokes,
    ($strokes, set) => {
        const sess = get(session);
        if (get(isContribMode) || !sess || $strokes.length === 0) {
            set([]);
            return;
        }
        sess.candidates($strokes, 5).then((result) => set(result));
    },
    [] as SymbolInfo[],
);

export const imgUrl = derived(strokes, ($strokes) => {
    const sess = get(session);
    if (!get(isContribMode) || !sess || $strokes.length === 0) return undefined;
    return sess.draw($strokes)?.toDataURL();
});
