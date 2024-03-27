<script>
    import "./app.pcss";
    import NavBar from "./components/NavBar.svelte";
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

<div class="my-8 grid grid-cols-2 py-4">
    <div class="col-span-1"></div>
    <div class="col-span-1 space-y-4">
        {#each candidates as { logo, info }}
            <Candidate {logo} {info} />
        {/each}
    </div>
</div>
