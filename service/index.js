import inferSyms from "./train/infer.json";
import modelUrl from "./train/model.onnx";
import { InferenceSession, Tensor } from "onnxruntime-web";

// To generate .d.ts:
//   $ bunx -p typescript *.js --declaration --allowJs --emitDeclarationOnly

/**
 * @typedef {object} DetypifySymbol
 * @property {string[]} names
 * @property {number} codepoint
 */

export { env as ortEnv } from "onnxruntime-web";

export class Detypify {
    /**
     * @param {InferenceSession} sess
     */
    constructor(sess) {
        this.sess = sess;

        this.canvas = document.createElement("canvas");
        this.canvas.width = this.canvas.height = 32;

        this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
        this.ctx.fillStyle = "white";
    }

    /**
     * Load ONNX runtime and the model.
     *
     * @returns {Promise<Detypify>}
     */
    static async create() {
        return new Detypify(await InferenceSession.create(modelUrl));
    }

    /**
     * Normalize strokes and draw to inner canvas.
     *
     * @param {[number, number][][]} strokes
     * @returns {HTMLCanvasElement}
     */
    draw(strokes) {
        // find bounding rect
        let minX = Infinity;
        let maxX = 0;
        let minY = Infinity;
        let maxY = 0;
        for (let stroke of strokes) {
            for (let [x, y] of stroke) {
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
            }
        }

        // normalize
        let canvasWidth = this.canvas.width;
        let width = Math.max(maxX - minX, maxY - minY);
        if (width == 0) return;
        width = width * 1.2 + 20;
        let zeroX = (minX + maxX - width) / 2;
        let zeroY = (minY + maxY - width) / 2;
        let scale = canvasWidth / width;

        // draw to inner canvas
        this.ctx.fillRect(0, 0, canvasWidth, canvasWidth);
        this.ctx.translate(0.5, 0.5);
        for (let stroke of strokes) {
            this.ctx.beginPath();
            for (let [x, y] of stroke) {
                this.ctx.lineTo(Math.round((x - zeroX) * scale), Math.round((y - zeroY) * scale));
            }
            this.ctx.stroke();
        }
        this.ctx.translate(-0.5, -0.5);

        return this.canvas;
    }

    /**
     * Inference top `k` candidates.
     *
     * @param {[number, number][][]} strokes
     * @param {number} k
     * @returns {Promise<DetypifySymbol[]>}
     */
    async candidates(strokes, k) {
        this.draw(strokes);

        // to greyscale
        let canvasWidth = this.canvas.width;
        let rgba = this.ctx.getImageData(0, 0, canvasWidth, canvasWidth).data;
        let grey = new Float32Array(rgba.length / 4);
        for (let i = 0; i < grey.length; ++i) {
            grey[i] = rgba[i * 4] == 255 ? 1 : 0;
        }

        // infer
        let tensor = new Tensor("float32", grey, [1, 1, 32, 32]);
        let output = await this.sess.run({ [this.sess.inputNames[0]]: tensor });
        output = Array.prototype.slice.call(output[this.sess.outputNames[0]].data);

        // select top K
        let withIdx = output.map((x, i) => [x, i]);
        withIdx.sort((a, b) => b[0] - a[0]);
        return withIdx.slice(0, k).map(([_, i]) => inferSyms[i]);
    }
}
