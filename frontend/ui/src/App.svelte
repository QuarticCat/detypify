<script lang="ts">
    import Contrib from "./routes/Contrib.svelte";
    import FAQ from "./routes/FAQ.svelte";
    import Home from "./routes/Home.svelte";
    import { Detypify, ortEnv } from "detypify-service";
    import { Navbar, NavBrand, NavLi, NavUl, NavHamburger } from "flowbite-svelte";
    import { Spinner, DarkMode, Tooltip, ToolbarButton, Heading } from "flowbite-svelte";
    import { GithubSolid } from "flowbite-svelte-icons";
    import { onMount } from "svelte";
    import { fade } from "svelte/transition";

    ortEnv.wasm.numThreads = 1;
    ortEnv.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";

    let activeHash = $state("#");

    onMount(() => {
        const updateHash = () => {
            activeHash = window.location.hash || "#";
        };

        updateHash();
        window.addEventListener("hashchange", updateHash);
        return () => window.removeEventListener("hashchange", updateHash);
    });
</script>

<Navbar>
    <NavBrand href="/">
        <!--  TODO: logo -->
        <span class="self-center text-2xl font-semibold whitespace-nowrap dark:text-white">Detypify</span>
    </NavBrand>
    <div class="flex">
        <NavUl activeUrl={activeHash}>
            <NavLi href="#">Home</NavLi>
            <NavLi href="#contrib">Contrib</NavLi>
            <NavLi href="#faq">FAQ</NavLi>
        </NavUl>

        <ToolbarButton size="lg" class="my-auto" href="https://github.com/QuarticCat/detypify">
            <GithubSolid size="lg" />
        </ToolbarButton>
        <Tooltip class="dark:bg-gray-900" placement="bottom">View on GitHub</Tooltip>

        <DarkMode size="lg" class="my-auto" />
        <Tooltip class="dark:bg-gray-900" placement="bottom">Toggle dark mode</Tooltip>

        <NavHamburger />
    </div>
</Navbar>

{#await Detypify.create()}
    <div class="ui-container min-h-80">
        <Spinner size="16" class="self-center" />
    </div>
{:then session}
    {#key activeHash}
        <div class="ui-container" out:fade={{ duration: 50 }} in:fade={{ duration: 50, delay: 50 }}>
            {#if activeHash === "#"}
                <Home {session} />
            {:else if activeHash === "#contrib"}
                <Contrib {session} />
            {:else if activeHash === "#faq"}
                <FAQ />
            {:else}
                <Heading>Not Found</Heading>
            {/if}
        </div>
    {/key}
{/await}
