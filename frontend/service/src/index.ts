import inferSyms from "../train/infer.json";
import { InferenceSession, Tensor } from "onnxruntime-web";

export { env as ortEnv } from "onnxruntime-web";

const modelUrl = new URL("../train/model.onnx", import.meta.url).href;

export interface DetypifySymbol {
    char: string;
    names: string[];
    shorthand?: string;
    mathShorthand?: string;
    markupShorthand?: string;
}

export type Point = [number, number];
export type Stroke = Point[];
export type Strokes = Stroke[];

export class Detypify {
    private sess: InferenceSession;
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;

    constructor(sess: InferenceSession) {
        this.sess = sess;

        this.canvas = document.createElement("canvas");
        this.canvas.width = this.canvas.height = 32;

        const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) {
            throw new Error("Failed to get 2D canvas context.");
        }
        this.ctx = ctx;
        this.ctx.fillStyle = "white";
    }

    /**
     * Load ONNX runtime and the model.
     */
    static async create(): Promise<Detypify> {
        return new Detypify(await InferenceSession.create(modelUrl));
    }

    /**
     * Normalize strokes and draw to inner canvas.
     */
    draw(strokes: Strokes): HTMLCanvasElement | undefined {
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
        const canvasWidth = this.canvas.width;
        let width = Math.max(maxX - minX, maxY - minY);
        if (width === 0) return;
        width = width * 1.2 + 20;
        const zeroX = (minX + maxX - width) / 2;
        const zeroY = (minY + maxY - width) / 2;
        const scale = canvasWidth / width;

        // Draw to inner canvas.
        this.ctx.fillRect(0, 0, canvasWidth, canvasWidth);
        this.ctx.translate(0.5, 0.5);
        for (const stroke of strokes) {
            this.ctx.beginPath();
            for (const [x, y] of stroke) {
                this.ctx.lineTo(Math.round((x - zeroX) * scale), Math.round((y - zeroY) * scale));
            }
            this.ctx.stroke();
        }
        this.ctx.translate(-0.5, -0.5);

        return this.canvas;
    }

    /**
     * Inference top `k` candidates.
     */
    async candidates(strokes: Strokes, k: number): Promise<DetypifySymbol[]> {
        this.draw(strokes);

        // To greyscale.
        const canvasWidth = this.canvas.width;
        const rgba = this.ctx.getImageData(0, 0, canvasWidth, canvasWidth).data;
        const grey = new Float32Array(rgba.length / 4);
        for (let i = 0; i < grey.length; ++i) {
            grey[i] = rgba[i * 4] === 255 ? 1 : 0;
        }

        // Infer.
        const tensor = new Tensor("float32", grey, [1, 1, 32, 32]);
        const outputs = await this.sess.run({ [this.sess.inputNames[0]]: tensor });
        const output = Array.from(outputs[this.sess.outputNames[0]].data as Iterable<number>);

        // Select top K.
        const withIdx = output.map((x, i) => [x, i] as const);
        withIdx.sort((a, b) => b[0] - a[0]);
        return withIdx.slice(0, k).map(([, i]) => inferSyms[i]);
    }
}
