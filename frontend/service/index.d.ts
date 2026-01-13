import { InferenceSession } from "onnxruntime-web";

export { env as ortEnv } from "onnxruntime-web";
export class Detypify {
    /**
     * Load ONNX runtime and the model.
     *
     * @returns {Promise<Detypify>}
     */
    static create(): Promise<Detypify>;
    /**
     * @param {InferenceSession} sess
     */
    constructor(sess: InferenceSession);
    sess: InferenceSession;
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
    /**
     * Normalize strokes and draw to inner canvas.
     *
     * @param {[number, number][][]} strokes
     * @returns {HTMLCanvasElement}
     */
    draw(strokes: [number, number][][]): HTMLCanvasElement;
    /**
     * Inference top `k` candidates.
     *
     * @param {[number, number][][]} strokes
     * @param {number} k
     * @returns {Promise<DetypifySymbol[]>}
     */
    candidates(strokes: [number, number][][], k: number): Promise<DetypifySymbol[]>;
}
export type DetypifySymbol = {
    names: string[];
    codepoint: number;
};
