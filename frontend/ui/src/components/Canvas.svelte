<script lang="ts">
    import { strokes } from "../store";
    import Button from "../utils/Button.svelte";
    import { Tooltip } from "flowbite-svelte";
    import { CloseOutline } from "flowbite-svelte-icons";
    import { onMount } from "svelte";

    let srcCanvas = $state<HTMLCanvasElement | null>(null);
    let srcCtx = $state<CanvasRenderingContext2D | null>(null);

    let isDrawing = false;
    let currP: [number, number] | undefined;
    let stroke: Array<[number, number]> = [];

    onMount(() => {
        if (!srcCanvas) return;
        srcCtx = srcCanvas.getContext("2d");
        if (!srcCtx) return;
        srcCtx.lineWidth = 5;
        srcCtx.lineJoin = "round";
        srcCtx.lineCap = "round";
    });

    const touchCall = (fn: ({ offsetX, offsetY }: { offsetX: number; offsetY: number }) => void) => (e: TouchEvent) => {
        e.preventDefault();
        const touch = e.touches[0];
        if (!touch || !srcCanvas) return;
        const rect = srcCanvas.getBoundingClientRect();
        fn({
            offsetX: touch.clientX - rect.left,
            offsetY: touch.clientY - rect.top,
        });
    };

    function drawStart({ offsetX, offsetY }: { offsetX: number; offsetY: number }) {
        if (!srcCtx) return;
        isDrawing = true;

        const roundedX = Math.round(offsetX);
        const roundedY = Math.round(offsetY);

        currP = [roundedX, roundedY];
        stroke = [currP];
    }

    function drawMove({ offsetX, offsetY }: { offsetX: number; offsetY: number }) {
        if (!isDrawing || !srcCtx || !currP) return;

        const roundedX = Math.round(offsetX);
        const roundedY = Math.round(offsetY);

        srcCtx.beginPath();
        srcCtx.moveTo(currP[0], currP[1]);
        srcCtx.lineTo(roundedX, roundedY);
        srcCtx.stroke();

        currP = [roundedX, roundedY];
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

    $effect(() => {
        if (srcCanvas && srcCtx && $strokes.length === 0) {
            srcCtx.clearRect(0, 0, srcCanvas.width, srcCanvas.height);
        }
    });
</script>

<div class="relative w-80">
    <canvas
        width="320"
        height="320"
        class="rounded-lg border border-gray-200 bg-gray-100 shadow-md dark:border-0 dark:bg-gray-600"
        bind:this={srcCanvas}
        onmousedown={drawStart}
        onmousemove={drawMove}
        onmouseup={drawEnd}
        onmouseleave={drawEnd}
        ontouchstart={touchCall(drawStart)}
        ontouchmove={touchCall(drawMove)}
        ontouchend={drawEnd}
        ontouchcancel={drawEnd}
    ></canvas>
    <Button class="absolute right-1 top-1 p-2" onclick={drawClear}>
        <CloseOutline class="size-6" />
        <Tooltip class="dark:bg-gray-900">Clear</Tooltip>
    </Button>
</div>
