import inferSyms from "../train-out/infer.json";
import modelUrl from "../train-out/model.onnx";
import { InferenceSession, Tensor, env as ortConfig } from "onnxruntime-web";
import { writable, derived, get } from "svelte/store";

ortConfig.wasm.numThreads = 1;
ortConfig.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

export const session = writable();

export const isContribMode = writable(false);

export const strokes = writable([]);
export const inputText = writable("");
export const savedSamples = writable([]);

let dstCanvas = document.createElement("canvas");
dstCanvas.width = dstCanvas.height = 32;
let dstCtx = dstCanvas.getContext("2d", { willReadFrequently: true });
dstCtx.fillStyle = "white";

function drawToDst($strokes) {
    // find rect
    let minX = Infinity;
    let maxX = 0;
    let minY = Infinity;
    let maxY = 0;
    for (let stroke of $strokes) {
        for (let [x, y] of stroke) {
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }
    }

    // normalize
    let dstWidth = dstCanvas.width;
    let width = Math.max(maxX - minX, maxY - minY);
    if (width == 0) return;
    width = width * 1.2 + 20;
    let zeroX = (minX + maxX - width) / 2;
    let zeroY = (minY + maxY - width) / 2;
    let scale = dstWidth / width;

    // draw to dstCanvas
    dstCtx.fillRect(0, 0, dstWidth, dstWidth);
    dstCtx.translate(0.5, 0.5);
    for (let stroke of $strokes) {
        dstCtx.beginPath();
        for (let [x, y] of stroke) {
            dstCtx.lineTo(Math.round((x - zeroX) * scale), Math.round((y - zeroY) * scale));
        }
        dstCtx.stroke();
    }
    dstCtx.translate(-0.5, -0.5);
}

InferenceSession.create(modelUrl).then((s) => {
    session.set(s);
    strokes.set(get(strokes)); // trigger drawing
});

export const candidates = derived(strokes, async ($strokes, set) => {
    let sess = get(session);

    // not loaded or clear
    if (get(isContribMode) || !sess || $strokes.length === 0) return set([]);

    drawToDst($strokes);

    // to greyscale
    let dstWidth = dstCanvas.width;
    let rgba = dstCtx.getImageData(0, 0, dstWidth, dstWidth).data;
    let grey = new Float32Array(rgba.length / 4);
    for (let i = 0; i < grey.length; ++i) {
        grey[i] = rgba[i * 4] == 255 ? 1 : 0;
    }

    // infer
    let tensor = new Tensor("float32", grey, [1, 1, 32, 32]);
    let output = await sess.run({ [sess.inputNames[0]]: tensor });
    output = Array.prototype.slice.call(output[sess.outputNames[0]].data);

    // select top K
    let withIdx = output.map((x, i) => [x, i]);
    withIdx.sort((a, b) => b[0] - a[0]);
    set(withIdx.slice(0, 5).map(([_, i]) => inferSyms[i]));
});

export const imgUrl = derived(strokes, ($strokes) => {
    // not loaded or clear
    if (!get(isContribMode) || $strokes.length === 0) return;

    drawToDst($strokes);
    return dstCanvas.toDataURL();
});
