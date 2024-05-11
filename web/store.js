import { Detypify, ortEnv } from "detypify-service";
import { writable, derived, get } from "svelte/store";

ortEnv.wasm.numThreads = 1;
ortEnv.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

export const session = writable();

export const isContribMode = writable(false);

export const strokes = writable([]);
export const inputText = writable("");
export const savedSamples = writable([]);

Detypify.create().then((s) => {
    session.set(s);
    strokes.set(get(strokes)); // trigger drawing
});

export const candidates = derived(strokes, async ($strokes, set) => {
    let sess = get(session);
    // not loaded or clear
    if (get(isContribMode) || !sess || $strokes.length === 0) return set([]);
    set(await sess.candidates($strokes, 5));
});

export const imgUrl = derived(strokes, ($strokes) => {
    let sess = get(session);
    // not loaded or clear
    if (!get(isContribMode) || !sess|| $strokes.length === 0) return;
    return sess.draw($strokes).toDataURL();
});
