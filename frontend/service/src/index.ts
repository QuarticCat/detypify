import contribSymsRaw from "../train/contrib.json";
import inferSymsRaw from "../train/infer.json";
import { InferenceSession, Tensor } from "onnxruntime-web";

export { env as ortEnv } from "onnxruntime-web";

const modelUrl = new URL("../train/model.onnx", import.meta.url).href;

export interface SymbolInfo {
    char: string;
    names: string[];
    shorthand?: string;
    mathShorthand?: string;
    markupShorthand?: string;
}

export type Point = [number, number];
export type Stroke = Point[];
export type Strokes = Stroke[];

/**
 * Symbol metadata used by the model.
 */
export const inferSyms = inferSymsRaw as SymbolInfo[];

/**
 * Mapping from Typst symbol names to characters.
 */
export const contribSyms = contribSymsRaw as Record<string, string>;

/**
 * Normalize strokes and draw them to canvas.
 */
export function drawStrokes(strokes: Strokes): HTMLCanvasElement | undefined {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 224;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
        throw new Error("Failed to get 2D canvas context.");
    }
    ctx.fillStyle = "black";
    ctx.strokeStyle = "white";
    ctx.lineWidth = 8;

    // Find bounding rect.
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const stroke of strokes) {
        for (const [x, y] of stroke) {
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }
    }

    // Normalize.
    const padding = 10;
    const targetSize = canvas.width - 2 * padding;
    let width = Math.max(maxX - minX, maxY - minY);
    const scale = width > 1e-6 ? targetSize / width : 1;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    // Draw to canvas.
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (const stroke of strokes) {
        ctx.beginPath();
        for (const [x, y] of stroke) {
            const targetX = (x - centerX) * scale + canvas.width / 2;
            const targetY = (y - centerY) * scale + canvas.width / 2;
            ctx.lineTo(Math.round(targetX), Math.round(targetY));
        }
        ctx.stroke();
    }

    return canvas;
}

/**
 * Typst symbol classifier.
 */
export class Detypify {
    private sess: InferenceSession;

    constructor(sess: InferenceSession) {
        this.sess = sess;
    }

    /**
     * Load ONNX runtime and the model.
     */
    static async create(): Promise<Detypify> {
        return new Detypify(await InferenceSession.create(modelUrl));
    }

    /**
     * Inference top `k` candidates.
     */
    async candidates(strokes: Strokes, k: number): Promise<SymbolInfo[]> {
        const canvas = drawStrokes(strokes);
        if (!canvas) return [];

        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) {
            throw new Error("Failed to get 2D canvas context.");
        }

        // To greyscale.
        const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        const grey = new Float32Array(rgba.length / 4);
        for (let i = 0; i < grey.length; ++i) {
            grey[i] = rgba[i * 4] / 255;
        }

        // Infer.
        const tensor = new Tensor("float32", grey, [1, 1, canvas.width, canvas.height]);
        const outputs = await this.sess.run({ [this.sess.inputNames[0]]: tensor });
        const output = Array.from(outputs[this.sess.outputNames[0]].data as Iterable<number>);

        // Select top K.
        const withIdx = output.map((x, i) => [x, i] as const);
        withIdx.sort((a, b) => b[0] - a[0]);
        return withIdx.slice(0, k).map(([, i]) => inferSyms[i]);
    }
}
