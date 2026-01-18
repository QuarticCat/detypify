<script lang="ts">
    import Candidate from "./lib/Candidate.svelte";
    import Canvas from "./lib/Canvas.svelte";
    import ContribPanel from "./lib/ContribPanel.svelte";
    import NavBar from "./lib/NavBar.svelte";
    import Preview from "./lib/Preview.svelte";
    import { strokes, isContribMode, samples } from "./store";
    import type { Strokes, SymbolInfo } from "detypify-service";
    import { Detypify, ortEnv } from "detypify-service";
    import { Alert, Hr, Spinner } from "flowbite-svelte";
    import { FireSolid } from "flowbite-svelte-icons";

    ortEnv.wasm.numThreads = 1;
    ortEnv.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";

    let contribSymName = $state("");

    async function candidates(session: Detypify, strokes: Strokes): Promise<SymbolInfo[]> {
        if (strokes.length === 0) return [];
        return await session.candidates(strokes, 5);
    }

    function draw(session: Detypify, strokes: Strokes): string | undefined {
        if (strokes.length === 0) return;
        return session.draw(strokes)?.toDataURL();
    }

    function deletePreview(id: string) {
        $samples = $samples.filter((s) => s.id !== id);
    }
</script>

<NavBar />

<div class="flex flex-wrap justify-center gap-x-16 gap-y-4 p-[2vw]">
    <div class="flex flex-col gap-4 w-80">
        <Canvas />
        {#if !$isContribMode}
            <Alert color="blue" border dismissable>
                Cannot identify your symbol? Click <FireSolid class="inline size-4 align-[-3px]" /> to help us improve our
                dataset!
            </Alert>
        {:else}
            <ContribPanel bind:value={contribSymName} />
        {/if}
    </div>
    <div class="flex flex-col gap-4 w-100">
        {#await Detypify.create()}
            <Spinner size="12" class="self-center" />
        {:then session}
            {#if !$isContribMode}
                {#await candidates(session, $strokes)}
                    <Spinner size="12" class="self-center" />
                {:then infoList}
                    {#each infoList as info}
                        <Candidate {info} />
                    {/each}
                {/await}
            {:else}
                <Preview name={contribSymName} img={draw(session, $strokes)} />
                <Hr class="mx-auto h-2 w-60 rounded" />
                {#each $samples as { id, name, strokes } (id)}
                    <Preview {name} img={draw(session, strokes)} ondelete={() => deletePreview(id)} />
                {/each}
            {/if}
        {/await}
    </div>
</div>
