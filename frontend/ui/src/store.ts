import { Detypify, ortEnv } from "detypify-service";
import type { Strokes } from "detypify-service";
import { get, writable } from "svelte/store";

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
