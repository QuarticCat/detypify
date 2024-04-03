import classes from "../train-out/classes.json";
import modelUrl from "../train-out/model.onnx";
import { InferenceSession, Tensor, env as ortConfig } from "onnxruntime-web";
import { writable, derived, get } from "svelte/store";

ortConfig.wasm.numThreads = 1;
ortConfig.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

export const session = writable();

InferenceSession.create(modelUrl).then((s) => {
    session.set(s);
});

export const isContribMode = writable(false);

export const strokes = writable([]);
export const inputText = writable("");
export const savedSamples = writable([]);

let dstCanvas = document.createElement("canvas");
dstCanvas.width = dstCanvas.height = 32;
let dstCtx = dstCanvas.getContext("2d", { willReadFrequently: true });
dstCtx.fillStyle = "white";

let minX = Infinity;
let minY = Infinity;
let maxX = 0;
let maxY = 0;

export const greyscale = derived(strokes, ($strokes) => {
    if ($strokes.length === 0) {
        minX = minY = Infinity;
        maxX = maxY = 0;
        return null;
    }

    // update
    let stroke = $strokes[$strokes.length - 1];
    let xs = stroke.map((p) => p[0]);
    minX = Math.min(minX, ...xs);
    maxX = Math.max(maxX, ...xs);
    let ys = stroke.map((p) => p[1]);
    minY = Math.min(minY, ...ys);
    maxY = Math.max(maxY, ...ys);

    // normalize
    let dstWidth = dstCanvas.width;
    let width = Math.max(maxX - minX, maxY - minY);
    if (width == 0) return;
    width *= 1.2;
    let zeroX = (maxX + minX) / 2 - width / 2;
    let zeroY = (maxY + minY) / 2 - width / 2;
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

    // // [debug] download dstCanvas image
    // let img = document.createElement("a");
    // img.href = dstCanvas.toDataURL();
    // img.download = "test.png";
    // img.click();

    // to greyscale
    let rgba = dstCtx.getImageData(0, 0, dstWidth, dstWidth).data;
    let grey = new Float32Array(rgba.length / 4);
    for (let i = 0; i < grey.length; ++i) {
        grey[i] = rgba[i * 4] == 255 ? 1 : 0;
    }
    return grey;
});

export const candidates = derived(greyscale, async ($greyscale, set) => {
    let sess = get(session);

    if (get(isContribMode) || !sess || !$greyscale) {
        set([]);
        return;
    }

    // infer
    let tensor = new Tensor("float32", $greyscale, [1, 1, 32, 32]);
    let output = await sess.run({ [sess.inputNames[0]]: tensor });
    output = Array.prototype.slice.call(output[sess.outputNames[0]].data);

    // select top K
    let withIdx = output.map((x, i) => [x, i]);
    withIdx.sort((a, b) => b[0] - a[0]);
    set(withIdx.slice(0, 5).map(([_, i]) => classes[i]));
});
