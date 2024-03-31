<script>
    import { Navbar, NavBrand, ToolbarButton, Tooltip, DarkMode, Modal, Heading, P, A } from "flowbite-svelte";
    import { QuestionCircleSolid, GithubSolid } from "flowbite-svelte-icons";

    let openHelpModal = false;
</script>

<Navbar>
    <NavBrand>
        <span class="self-center whitespace-nowrap text-2xl font-semibold dark:text-white">Detypify</span>
    </NavBrand>

    <div class="ms-auto flex items-center">
        <ToolbarButton
            size="lg"
            class="inline-block hover:text-gray-900 dark:hover:text-white"
            on:click={() => (openHelpModal = true)}
        >
            <QuestionCircleSolid size="lg" />
        </ToolbarButton>
        <Tooltip class="z-10 dark:bg-gray-900" placement="bottom">Help</Tooltip>

        <ToolbarButton
            size="lg"
            class="inline-block hover:text-gray-900 dark:hover:text-white"
            href="https://github.com/QuarticCat/detypify"
        >
            <GithubSolid size="lg" />
        </ToolbarButton>
        <Tooltip class="z-10 dark:bg-gray-900" placement="bottom">View on GitHub</Tooltip>

        <DarkMode size="lg" class="inline-block hover:text-gray-900 dark:hover:text-white" />
        <Tooltip class="z-10 dark:bg-gray-900" placement="bottom">Toggle dark mode</Tooltip>
    </div>
</Navbar>

<Modal bind:open={openHelpModal} outsideclose>
    <Heading tag="h4">What symbols does it support?</Heading>
    <P>
        Check
        <A href="https://github.com/QuarticCat/detypify/blob/main/supported-symbols.txt">supported-symbols.txt</A>.
    </P>

    <Heading tag="h4">Like it?</Heading>
    <P>
        Star me on
        <A href="https://github.com/QuarticCat/detypify">GitHub</A>!
    </P>
</Modal>
