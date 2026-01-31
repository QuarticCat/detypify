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

export interface DrawOptions {
    canvasSize?: number;
    lineWidth?: number;
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
export function drawStrokes(strokes: Strokes, options: DrawOptions = {}): HTMLCanvasElement | undefined {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = options.canvasSize ?? 32;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
        throw new Error("Failed to get 2D canvas context.");
    }
    ctx.fillStyle = "white";
    ctx.lineWidth = options.lineWidth ?? 1;

    // Find bounding rect.
    let minX = Infinity;
    let maxX = 0;
    let minY = Infinity;
    let maxY = 0;
    for (const stroke of strokes) {
        for (const [x, y] of stroke) {
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }
    }

    // Normalize.
    let width = Math.max(maxX - minX, maxY - minY);
    if (width === 0) return;
    width = width * 1.2 + 20;
    const zeroX = (minX + maxX - width) / 2;
    const zeroY = (minY + maxY - width) / 2;
    const scale = canvas.width / width;

    // Draw to canvas.
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.translate(0.5, 0.5);
    for (const stroke of strokes) {
        ctx.beginPath();
        for (const [x, y] of stroke) {
            ctx.lineTo(Math.round((x - zeroX) * scale), Math.round((y - zeroY) * scale));
        }
        ctx.stroke();
    }
    ctx.translate(-0.5, -0.5);

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
    async candidates(strokes: Strokes, k: number, options?: DrawOptions): Promise<SymbolInfo[]> {
        const canvas = drawStrokes(strokes, options);
        if (!canvas) return [];

        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) {
            throw new Error("Failed to get 2D canvas context.");
        }

        // To greyscale.
        const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        const grey = new Float32Array(rgba.length / 4);
        for (let i = 0; i < grey.length; ++i) {
            grey[i] = rgba[i * 4] === 255 ? 1 : 0;
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
