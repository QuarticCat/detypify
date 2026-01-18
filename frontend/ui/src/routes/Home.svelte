<script lang="ts">
    import Candidate from "../lib/Candidate.svelte";
    import Canvas from "../lib/Canvas.svelte";
    import { strokes } from "../store";
    import type { Strokes, SymbolInfo } from "detypify-service";
    import { Detypify } from "detypify-service";
    import { Spinner } from "flowbite-svelte";

    const { session }: { session: Detypify } = $props();

    async function candidates(strokes: Strokes): Promise<SymbolInfo[]> {
        if (strokes.length === 0) return [];
        return await session.candidates(strokes, 5);
    }
</script>

<div class="flex flex-col gap-4 w-80">
    <Canvas />
    <!-- TODO: Brave alert here. -->
</div>

<div class="flex flex-col gap-4 w-100">
    {#await candidates($strokes)}
        <Spinner size="12" class="self-center" />
    {:then infoList}
        <!-- FIXME: Fly-in animation is not working. -->
        {#each infoList as info}
            <Candidate {info} />
        {/each}
    {/await}
</div>
