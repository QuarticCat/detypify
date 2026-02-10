<script lang="ts">
    import Candidate from "../lib/Candidate.svelte";
    import Canvas from "../lib/Canvas.svelte";
    import type { Strokes } from "detypify-service";
    import { Detypify, inferSyms } from "detypify-service";
    import { Alert } from "flowbite-svelte";

    const { session }: { session: Detypify } = $props();

    let strokes: Strokes = $state([]);
    let candidates: number[] = $state([]);
    let numToShow = $state(5);

    // Every time stroke changes, calculate candidates.
    $effect(() => {
        if (strokes.length === 0) {
            candidates = [];
            return;
        }

        session.infer(strokes).then((scores) => {
            const keys = Array.from(scores.keys());
            keys.sort((a, b) => scores[b] - scores[a]);
            candidates = keys;
        });
    });
</script>

<div class="ui-sub-container w-80">
    <Canvas bind:strokes />
    {#if "brave" in navigator}
        <Alert color="yellow" border dismissable>
            If you are using Brave, please turn off Shields for this site, or it won't work properly.
        </Alert>
    {/if}
</div>

<div class="ui-sub-container w-100">
    {#each candidates.slice(0, numToShow) as i (i)}
        <Candidate info={inferSyms[i]} />
    {/each}
</div>
