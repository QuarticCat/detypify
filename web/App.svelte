<script>
    import { Spinner } from "flowbite-svelte";
    import "./app.pcss";
    import Candidate from "./components/Candidate.svelte";
    import Canvas from "./components/Canvas.svelte";
    import EditArea from "./components/EditArea.svelte";
    import NavBar from "./components/NavBar.svelte";
    import { candidates, isContribMode, session } from "./store";
</script>

<NavBar />

<div class="flex flex-wrap justify-evenly space-y-4 pb-8 md:pt-8">
    <div class="hidden md:block"></div>
    <div class="w-[320px] space-y-4">
        <Canvas />
        {#if $isContribMode}
            <EditArea />
        {/if}
    </div>
    <div class="w-[360px] space-y-4 text-center">
        {#if !$session}
            <Spinner size="12" />
        {:else if !$isContribMode}
            {#each $candidates as [logo, info]}
                <Candidate {logo} {info} />
            {/each}
        {:else}
            <Spinner size="12" />
        {/if}
    </div>
    <div class="hidden md:block"></div>
</div>
