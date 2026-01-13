<script>
    import { Hr, Spinner } from "flowbite-svelte";
    import contribSyms from "../service/train/contrib.json";
    import "./app.pcss";
    import Alert from "./components/Alert.svelte";
    import Candidate from "./components/Candidate.svelte";
    import Canvas from "./components/Canvas.svelte";
    import ContribPanel from "./components/ContribPanel.svelte";
    import NavBar from "./components/NavBar.svelte";
    import Preview from "./components/Preview.svelte";
    import { candidates, imgUrl, inputText, isContribMode, savedSamples, session } from "./store";

    const BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAP7//wAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==";
</script>

<NavBar />

<div class="flex flex-wrap justify-evenly space-y-4 pb-8 md:pt-8">
    <div class="hidden md:block"></div>
    <div class="w-[320px] space-y-4">
        <Canvas />
        {#if !$isContribMode}
            <Alert />
        {:else}
            <ContribPanel />
        {/if}
    </div>
    <div class="w-[360px] space-y-4 text-center">
        {#if !$session}
            <Spinner size="12" />
        {:else if !$isContribMode}
            {#each $candidates as info}
                <Candidate {info} />
            {/each}
        {:else}
            <Preview logo={contribSyms[$inputText] ?? ""} imgUrl={$imgUrl ?? BLANK} />
            <Hr classHr="w-[240px] h-2 rounded mx-auto" />
            {#each $savedSamples as { id, logo, imgUrl } (id)}
                <Preview {id} {logo} {imgUrl} />
            {/each}
        {/if}
    </div>
    <div class="hidden md:block"></div>
</div>
