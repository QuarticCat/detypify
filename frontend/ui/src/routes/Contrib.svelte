<script lang="ts">
    import Canvas from "../lib/Canvas.svelte";
    import ContribPanel from "../lib/ContribPanel.svelte";
    import type { Sample } from "../lib/ContribPanel.svelte";
    import Preview from "../lib/Preview.svelte";
    import type { Strokes } from "detypify-service";
    import { drawStrokes } from "detypify-service";
    import { Hr, Alert } from "flowbite-svelte";

    let input = $state("");
    let strokes: Strokes = $state([]);
    let samples: Sample[] = $state([]);

    function draw(strokes: Strokes): string | undefined {
        if (strokes.length === 0) return;
        return drawStrokes(strokes)?.toDataURL();
    }
</script>

<div class="ui-sub-container w-80">
    <Canvas bind:strokes />
    <ContribPanel bind:input bind:strokes bind:samples />
    <Alert color="blue" border dismissable>
        Select a symbol, draw it, submit your contribution and make Detypify better!
    </Alert>
</div>

<div class="ui-sub-container w-100">
    <Preview name={input} img={draw(strokes)} />
    <Hr class="mx-auto h-2 w-60 rounded" />
    {#each samples as { id, name, strokes }, idx (id)}
        <Preview {name} img={draw(strokes)} ondelete={() => samples.splice(idx, 1)} />
    {/each}
</div>
