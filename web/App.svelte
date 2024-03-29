<script>
    import "./app.pcss";
    import NavBar from "./components/NavBar.svelte";
    import Canvas from "./components/Canvas.svelte";
    import Candidate from "./components/Candidate.svelte";
    import classes from "../train-out/classes.js";

    let results = [];
    $: candidates = results.map((i) => classes[i]);
</script>

<NavBar />

<div class="flex flex-wrap justify-evenly py-16 md:px-32">
    <Canvas bind:results />
    <div class="w-[400px] space-y-4">
        {#each candidates as [logo, info]}
            <Candidate {logo} {info} />
        {/each}
    </div>
</div>
