<script lang="ts">
    import Button from "./Button.svelte";
    import { Tooltip } from "flowbite-svelte";

    let { children } = $props();
    let tip = $state("Copy");

    function reset() {
        tip = "Copy";
    }

    async function copy(event: MouseEvent) {
        const target = event.currentTarget as HTMLElement | null;
        const text = target?.firstElementChild?.textContent?.trim() ?? target?.textContent?.trim();
        if (!text) return;
        await navigator.clipboard.writeText(text);
        tip = "Copied!";
    }
</script>

<Button onclick={copy}>
    {@render children?.()}
    <Tooltip class="dark:bg-gray-900" onbeforetoggle={reset}>
        {tip}
    </Tooltip>
</Button>
