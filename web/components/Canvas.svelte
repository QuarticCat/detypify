<script>
    import { Tooltip } from "flowbite-svelte";
    import { CloseOutline } from "flowbite-svelte-icons";
    import { onMount } from "svelte";
    import Button from "../utils/Button.svelte";

    export let greyscale;

    let srcCanvas; // user painting
    let srcCtx;
    let dstCanvas; // normalized image
    let dstCtx;

    let touchL;
    let touchT;

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
        dstCtx.fillStyle = "white";

        ({ left: touchL, top: touchT } = srcCanvas.getBoundingClientRect());
    });

    const touchCall = (fn) => (e) => {
        fn({
            offsetX: e.touches[0].clientX - touchL,
            offsetY: e.touches[0].clientY - touchT,
        });
    };

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
        srcCtx.stroke();
    }

    function drawEnd() {
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
        dstCtx.fillRect(0, 0, dstWidth, dstWidth);
        dstCtx.translate(0.5, 0.5);
        for (let stroke of strokes) {
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
        greyscale = grey;
    }

    function drawClear() {
        srcCtx.clearRect(0, 0, srcCanvas.width, srcCanvas.height);
        strokes = [];
        minX = minY = Infinity;
        maxX = maxY = 0;
        greyscale = null;
    }
</script>

<div class="relative w-[320px]">
    <canvas
        width="320"
        height="320"
        class="rounded-lg border border-gray-200 bg-gray-100 shadow-md dark:border-0 dark:bg-gray-600"
        bind:this={srcCanvas}
        on:mousedown={drawStart}
        on:mousemove={drawMove}
        on:mouseup={drawEnd}
        on:mouseleave={drawEnd}
        on:touchstart|preventDefault={touchCall(drawStart)}
        on:touchmove|preventDefault={touchCall(drawMove)}
        on:touchend|preventDefault={drawEnd}
        on:touchcancel|preventDefault={drawEnd}
    />
    <Button class="absolute right-1 top-1 p-2" on:click={drawClear}>
        <CloseOutline class="size-6" />
    </Button>
    <Tooltip class="dark:bg-gray-900">Clear</Tooltip>
</div>
