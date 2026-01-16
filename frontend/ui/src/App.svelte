<script lang="ts">
    import Alert from "./components/Alert.svelte";
    import Candidate from "./components/Candidate.svelte";
    import Canvas from "./components/Canvas.svelte";
    import ContribPanel from "./components/ContribPanel.svelte";
    import NavBar from "./components/NavBar.svelte";
    import Preview from "./components/Preview.svelte";
    import { candidates, imgUrl, inputText, isContribMode, savedSamples, session } from "./store";
    import { contribSyms } from "detypify-service";
    import { Hr, Spinner } from "flowbite-svelte";

    async function candidates(strokes: Strokes): Promise<SymbolInfo[] | undefined> {
        if (strokes.length === 0) return;
        return await $session?.candidates(strokes, 5);
    }

    function draw(strokes: Strokes): string | undefined {
        if (strokes.length === 0) return;
        return $session?.draw(strokes)?.toDataURL();
    }
</script>

<NavBar />

<div class="flex flex-wrap justify-center gap-x-16 gap-y-4 px-[4vw] py-[1vw]">
    <div class="flex flex-col gap-4 w-80">
        <Canvas />
        {#if !$isContribMode}
            <Alert color="blue" border dismissable>
                Cannot identify your symbol? Click <FireSolid class="inline size-4 align-[-3px]" /> to help us improve our
                dataset!
            </Alert>
        {:else}
            <ContribPanel />
        {/if}
    </div>
    <div class="flex flex-col gap-4 w-100">
        {#if !$session}
            <Spinner size="12" class="self-center" />
        {:else if !$isContribMode}
            {#await candidates($strokes)}
                <Spinner size="12" class="self-center" />
            {:then infoList}
                {#each infoList as info}
                    <Candidate {info} />
                {/each}
            {/await}
        {:else}
            <Preview logo={contribSyms[$inputText] ?? ""} imgUrl={$imgUrl ?? BLANK} />
            <Hr class="mx-auto h-2 w-60 rounded" />
            {#each $samples as { id, name, strokes } (id)}
                <Preview {name} img={draw(strokes)} ondelete={() => ($samples = $samples.filter((s) => s.id !== id))} />
            {/each}
        {/if}
    </div>
</div>
