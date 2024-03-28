<script>
    import "./app.pcss";
    import NavBar from "./components/NavBar.svelte";
    import Canvas from "./components/Canvas.svelte";
    import Candidate from "./components/Candidate.svelte";

    let candidates = [
        {
            logo: "⧛",
            info: [
                ["Name", "fence.r.double"],
                ["Escape", "\\u{29DB}"],
            ],
        },
        {
            logo: "∗",
            info: [
                ["Name", "ast.op"],
                ["Escape", "\\u{2217}"],
                ["Shorthand", "*"],
            ],
        },
    ];
</script>

<NavBar />

<div class="flex flex-wrap justify-evenly py-16 md:px-32">
    <Canvas />
    <div class="space-y-4">
        {#each candidates as { logo, info }}
            <Candidate {logo} {info} />
        {/each}
    </div>
</div>
