<script lang="ts">
    import contribSyms from "../../service/train/contrib.json";
    import Alert from "./components/Alert.svelte";
    import Candidate from "./components/Candidate.svelte";
    import Canvas from "./components/Canvas.svelte";
    import ContribPanel from "./components/ContribPanel.svelte";
    import NavBar from "./components/NavBar.svelte";
    import Preview from "./components/Preview.svelte";
    import { candidates, imgUrl, inputText, isContribMode, savedSamples, session } from "./store";
    import { Hr, Spinner } from "flowbite-svelte";

    const BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAP7//wAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==";
</script>

<NavBar />

<div class="flex flex-wrap justify-evenly space-y-4 pb-8 md:pt-8">
    <div class="hidden md:block"></div>
    <div class="w-80 space-y-4">
        <Canvas />
        {#if !$isContribMode}
            <Alert />
        {:else}
            <ContribPanel />
        {/if}
    </div>
    <div class="w-90 space-y-4 text-center">
        {#if !$session}
            <Spinner size="12" />
        {:else if !$isContribMode}
            {#each $candidates as info}
                <Candidate {info} />
            {/each}
        {:else}
            <Preview logo={contribSyms[$inputText as keyof typeof contribSyms] ?? ""} imgUrl={$imgUrl ?? BLANK} />
            <Hr class="mx-auto h-2 w-60 rounded" />
            {#each $savedSamples as { id, logo, imgUrl } (id)}
                <Preview {id} {logo} {imgUrl} />
            {/each}
        {/if}
    </div>
    <div class="hidden md:block"></div>
</div>
