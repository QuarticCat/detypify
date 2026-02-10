<script lang="ts">
    import Candidate from "../lib/Candidate.svelte";
    import Canvas from "../lib/Canvas.svelte";
    import type { Strokes } from "detypify-service";
    import { Detypify, inferSyms } from "detypify-service";
    import { Alert, Button } from "flowbite-svelte";
    import { fly } from "svelte/transition";

    const { session }: { session: Detypify } = $props();

    let strokes: Strokes = $state([]);
    let candidates: number[] = $state([]);
    let numToShow = $state(5);

    // Every time stroke changes, reset numToShow and infer candidates.
    $effect(() => {
        numToShow = 5;

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
    {#if candidates.length > 0}
        <div
            in:fly|local={{ x: 20, duration: 50, delay: 50 }}
            out:fly|local={{ x: 20, duration: 50 }}
            class="w-fit self-center"
        >
            <Button outline size="sm" onclick={() => (numToShow += 5)}>Show More</Button>
        </div>
    {/if}
</div>
