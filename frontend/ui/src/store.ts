import type { Strokes } from "detypify-service";
import { writable } from "svelte/store";

export type Sample = {
    id: string;
    name: string;
    strokes: Strokes;
};

export const strokes = writable<Strokes>([]);
export const samples = writable<Sample[]>([]);
