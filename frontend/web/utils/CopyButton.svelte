<script>
    import { Tooltip } from "flowbite-svelte";
    import Button from "./Button.svelte";

    let tip;

    function reset() {
        tip = "Copy";
    }

    async function copy() {
        await navigator.clipboard.writeText(this.children[0].textContent);
        tip = "Copied!";
    }
</script>

<Button on:click={copy}>
    <slot />
    <Tooltip class="dark:bg-gray-900" on:show={reset}>
        {tip}
    </Tooltip>
</Button>
