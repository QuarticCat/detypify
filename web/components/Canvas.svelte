<script>
    import { onMount } from "svelte";
    import { Tensor, InferenceSession, env } from "onnxruntime-web";
    import { Button, Tooltip } from "flowbite-svelte";
    import { CloseOutline } from "flowbite-svelte-icons";
    import modelUrl from "../../train-out/model.onnx";

    env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

    export let results;

    let srcCanvas; // user painting
    let srcCtx;
    let dstCanvas; // normalized image
    let dstCtx;

    let isDrawing;
    let currP;
    let stroke;

    let strokes = [];
    let minX = Infinity;
    let minY = Infinity;
    let maxX = 0;
    let maxY = 0;

    onMount(() => {
        srcCtx = srcCanvas.getContext("2d");
        srcCtx.lineWidth = 5;
        srcCtx.lineJoin = "round";
        srcCtx.lineCap = "round";

        dstCanvas = document.createElement("canvas");
        dstCanvas.width = dstCanvas.height = 32;
        dstCtx = dstCanvas.getContext("2d", { willReadFrequently: true });
    });

    function drawStart({ offsetX, offsetY }) {
        isDrawing = true;
        currP = [offsetX, offsetY];
        stroke = [currP];
    }

    function drawMove({ offsetX, offsetY }) {
        if (!isDrawing) return;

        let prevP = currP;
        currP = [offsetX, offsetY];
        stroke.push(currP);

        srcCtx.beginPath();
        srcCtx.moveTo(...prevP);
        srcCtx.lineTo(...currP);
        srcCtx.closePath();
        srcCtx.stroke();
    }

    async function drawEnd() {
        if (!isDrawing) return; // normal mouse leave
        isDrawing = false;

        // update
        strokes.push(stroke);
        minX = Math.min(minX, ...stroke.map((p) => p[0]));
        minY = Math.min(minY, ...stroke.map((p) => p[1]));
        maxX = Math.max(maxX, ...stroke.map((p) => p[0]));
        maxY = Math.max(maxY, ...stroke.map((p) => p[1]));

        // normalize
        let dstWidth = dstCanvas.width;
        let width = Math.max(maxX - minX, maxY - minY);
        if (width == 0) return;
        width *= 1.2;
        let zeroX = (maxX + minX) / 2 - width / 2;
        let zeroY = (maxY + minY) / 2 - width / 2;
        let scale = dstWidth / width;

        // draw to dstCanvas
        dstCtx.clearRect(0, 0, dstWidth, dstWidth);
        dstCtx.beginPath();
        for (let stroke of strokes) {
            let [x, y] = stroke[0];
            dstCtx.moveTo((x - zeroX) * scale, (y - zeroY) * scale);
            for (let [x, y] of stroke.slice(1)) {
                dstCtx.lineTo((x - zeroX) * scale, (y - zeroY) * scale);
            }
        }
        dstCtx.closePath();
        dstCtx.stroke();

        // to tensor
        let data = dstCtx.getImageData(0, 0, dstWidth, dstWidth).data;
        let tensor = new Float32Array(data.length / 4);
        for (let i = 0; i < tensor.length; ++i) {
            tensor[i] = data[i * 4 + 3] ? 1 : 0;
        }
        tensor = new Tensor("float32", tensor, [1, 1, 32, 32]);

        // infer
        let session = await InferenceSession.create(modelUrl);
        let output = await session.run({ [session.inputNames[0]]: tensor });
        output = Array.prototype.slice.call(output[session.outputNames[0]].data);
        let withIdx = output.map((x, i) => [x, i]);
        withIdx.sort((a, b) => b[0] - a[0]);
        results = withIdx.slice(0, 5).map(([_, i]) => i);
    }

    function drawClear() {
        srcCtx.clearRect(0, 0, srcCanvas.width, srcCanvas.height);
        strokes = [];
        minX = minY = Infinity;
        maxX = maxY = 0;
        results = [];
    }
</script>

<div class="relative size-[320px]">
    <canvas
        width="320"
        height="320"
        class="rounded-lg border border-gray-200 bg-gray-100 shadow-md dark:border-0 dark:bg-gray-600"
        bind:this={srcCanvas}
        on:mousedown={drawStart}
        on:mousemove={drawMove}
        on:mouseup={drawEnd}
        on:mouseleave={drawEnd}
    />
    <!-- TODO: adjust hover & focus colors of the button under light theme -->
    <Button
        pill
        outline
        color="alternative"
        class="absolute right-1 top-1 border-0 p-2 focus:ring-0"
        on:click={drawClear}
    >
        <CloseOutline class="size-6" />
    </Button>
    <Tooltip class="dark:bg-gray-900">Clear</Tooltip>
</div>
