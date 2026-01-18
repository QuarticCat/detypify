<script lang="ts">
    import { strokes } from "../store";
    import type { Stroke } from "detypify-service";
    import { Tooltip } from "flowbite-svelte";
    import { CloseOutline } from "flowbite-svelte-icons";

    let canvas: HTMLCanvasElement | undefined;
    let ctx: CanvasRenderingContext2D | null | undefined;
    let stroke: Stroke = [];

    // Initialize canvas context on mount.
    $effect(() => {
        if (!canvas) return;
        ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.lineWidth = 5;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
    });

    // Every time stroke clears (refresh contrib, change mode), clear canvas.
    $effect(() => {
        if ($strokes.length > 0 || !ctx) return;
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    });

    function drawStart(event: PointerEvent) {
        if (event.button !== 0 || !ctx) return;

        const roundedX = Math.round(event.offsetX);
        const roundedY = Math.round(event.offsetY);
        stroke = [[roundedX, roundedY]];

        ctx.beginPath();
        ctx.moveTo(roundedX, roundedY);
        ctx.lineTo(roundedX, roundedY);
        ctx.stroke();
    }

    function drawMove(event: PointerEvent) {
        if (stroke.length === 0 || !ctx) return;

        const roundedX = Math.round(event.offsetX);
        const roundedY = Math.round(event.offsetY);
        const [lastX, lastY] = stroke[stroke.length - 1];
        stroke.push([roundedX, roundedY]);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(roundedX, roundedY);
        ctx.stroke();
    }

    function drawEnd() {
        if (stroke.length === 0) return;
        $strokes = [...$strokes, stroke];
        stroke = [];
    }

    function drawClear() {
        $strokes = [];
    }
</script>

<div class="relative w-80">
    <canvas
        width="320"
        height="320"
        class="touch-none rounded-lg border border-gray-200 bg-gray-100 shadow-md dark:border-0 dark:bg-gray-600"
        bind:this={canvas}
        onpointerdown={drawStart}
        onpointermove={drawMove}
        onpointerup={drawEnd}
        onpointerleave={drawEnd}
        onpointercancel={drawEnd}
    ></canvas>
    <button type="button" class="ui-hover-btn ui-close-btn" onclick={drawClear}>
        <CloseOutline class="size-6" />
        <Tooltip class="dark:bg-gray-900">Clear</Tooltip>
    </button>
</div>
