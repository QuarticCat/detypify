<script>
    import "./app.pcss";
    import { Navbar, NavBrand, ToolbarButton, Tooltip, DarkMode } from "flowbite-svelte";
    import { GithubSolid } from "flowbite-svelte-icons";
</script>

<Navbar>
    <NavBrand>
        <!-- <img src={logo} class="me-3 h-8" alt="Detexify Logo" /> -->
        <span class="self-center whitespace-nowrap text-2xl font-semibold dark:text-white">
            Detypify
        </span>
    </NavBrand>
    <div class="ms-auto flex items-center">
        <ToolbarButton
            size="lg"
            class="inline-block hover:text-gray-900 dark:hover:text-white"
            href="https://github.com/QuarticCat/detypify"
        >
            <GithubSolid size="lg" />
        </ToolbarButton>
        <Tooltip class="dark:bg-gray-900" placement="bottom">View on GitHub</Tooltip>
        <DarkMode size="lg" class="inline-block hover:text-gray-900 dark:hover:text-white" />
        <Tooltip class="dark:bg-gray-900" placement="bottom">Toggle dark mode</Tooltip>
    </div>
</Navbar>
