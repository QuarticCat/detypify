<script>
    import { A, DarkMode, Heading, Modal, NavBrand, Navbar, P, ToolbarButton, Tooltip } from "flowbite-svelte";
    import { FireOutline, FireSolid, GithubSolid, QuestionCircleSolid } from "flowbite-svelte-icons";
    import { isContribMode, strokes } from "../store";

    let openHelpModal = false;

    function toggleContribMode() {
        $isContribMode = !$isContribMode;
        $strokes = [];
        // should I reset inputText and savedSamples?
    }
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

        <ToolbarButton
            size="lg"
            class="inline-block hover:text-gray-900 dark:hover:text-white"
            on:click={toggleContribMode}
        >
            {#if $isContribMode}
                <FireOutline size="lg" />
            {:else}
                <FireSolid size="lg" />
            {/if}
        </ToolbarButton>
        <Tooltip class="z-10 dark:bg-gray-900" placement="bottom">Toggle contrib mode</Tooltip>

        <DarkMode size="lg" class="inline-block hover:text-gray-900 dark:hover:text-white" />
        <Tooltip class="z-10 dark:bg-gray-900" placement="bottom">Toggle dark mode</Tooltip>
    </div>
</Navbar>

<Modal bind:open={openHelpModal} outsideclose>
    <Heading tag="h4">Cannot find some symbols?</Heading>
    <P>
        Supported symbols are listed in
        <A href="https://github.com/QuarticCat/detypify/blob/main/assets/supported-symbols.txt">
            supported-symbols.txt
        </A>.
    </P>
    <P>
        You can click <FireSolid class="inline align-text-top" /> to contribute your drawings to the dataset.
    </P>
    <P>
        If Detexify supports a symbol but Detypify doesn't, PR to
        <A href="https://github.com/QuarticCat/detypify/blob/main/assets/tex_to_typ_extra.csv">tex_to_typ_extra.csv</A>.
    </P>

    <Heading tag="h4">Want to use it offline?</Heading>
    <P>
        Check the
        <A href="https://support.google.com/chrome/answer/9658361">guide</A>.
    </P>

    <Heading tag="h4">Like it?</Heading>
    <P>
        Star me on
        <A href="https://github.com/QuarticCat/detypify">GitHub</A>!
    </P>
</Modal>
