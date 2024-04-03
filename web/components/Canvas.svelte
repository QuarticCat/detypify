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
        ({ left, top } = srcCanvas.getBoundingClientRect());
        fn({
            offsetX: e.touches[0].clientX - left,
            offsetY: e.touches[0].clientY - top,
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
        if (stroke.length === 1) return; // no line
        $strokes = [...$strokes, stroke];
    }

    function drawClear() {
        srcCtx.clearRect(0, 0, srcCanvas.width, srcCanvas.height);
        $strokes = [];
    }

    function redraw(strokes) {
        if (!srcCanvas) return;
        srcCtx.clearRect(0, 0, srcCanvas.width, srcCanvas.height);
        for (let stroke of strokes) {
            srcCtx.beginPath();
            for (let [x, y] of stroke) {
                srcCtx.lineTo(x, y);
            }
            srcCtx.stroke();
        }
    }

    $: redraw($strokes);
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
