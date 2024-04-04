<script>
    import { Tooltip } from "flowbite-svelte";
    import { CloseOutline } from "flowbite-svelte-icons";
    import { onMount } from "svelte";
    import { strokes } from "../store";
    import Button from "../utils/Button.svelte";

    let srcCanvas;
    let srcCtx;

    let isDrawing;
    let currP;
    let stroke;

    onMount(() => {
        srcCtx = srcCanvas.getContext("2d");
        srcCtx.lineWidth = 5;
        srcCtx.lineJoin = "round";
        srcCtx.lineCap = "round";
    });

    const touchCall = (fn) => (e) => {
        let rect = srcCanvas.getBoundingClientRect();
        fn({
            offsetX: e.touches[0].clientX - rect.left,
            offsetY: e.touches[0].clientY - rect.top,
        });
    };

    function drawStart({ offsetX, offsetY }) {
        isDrawing = true;
        currP = [offsetX, offsetY];
        stroke = [currP];
    }

    function drawMove({ offsetX, offsetY }) {
        if (!isDrawing) return;

        srcCtx.beginPath();
        srcCtx.moveTo(currP[0], currP[1]);
        srcCtx.lineTo(offsetX, offsetY);
        srcCtx.stroke();

        currP = [offsetX, offsetY];
        stroke.push(currP);
    }

    function drawEnd() {
        if (!isDrawing) return; // normal mouse leave
        isDrawing = false;
        if (stroke.length === 1) return; // no line
        $strokes = [...$strokes, stroke];
    }

    function drawClear() {
        $strokes = [];
    }

    $: if (!!srcCanvas && $strokes.length === 0) {
        srcCtx.clearRect(0, 0, srcCanvas.width, srcCanvas.height);
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
        <Tooltip class="dark:bg-gray-900">Clear</Tooltip>
    </Button>
</div>
