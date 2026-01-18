<script lang="ts">
    import Candidate from "../lib/Candidate.svelte";
    import Canvas from "../lib/Canvas.svelte";
    import type { Strokes, SymbolInfo } from "detypify-service";
    import { Detypify } from "detypify-service";

    const { session }: { session: Detypify } = $props();

    let strokes: Strokes = $state([]);
    let candidates: SymbolInfo[] = $state([]);

    // Every time stroke changes, calculate candidates.
    $effect(() => {
        if (strokes.length > 0) {
            session.candidates(strokes, 5).then((res) => (candidates = res));
        } else {
            candidates = [];
        }
    });
</script>

<div class="ui-sub-container w-80">
    <Canvas bind:strokes />
    <!-- TODO: Brave alert here. -->
</div>

<div class="ui-sub-container w-100">
    {#each candidates as info}
        <Candidate {info} />
    {/each}
</div>
