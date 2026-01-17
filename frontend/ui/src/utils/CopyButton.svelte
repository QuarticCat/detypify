<script lang="ts">
    import { Tooltip } from "flowbite-svelte";
    import type { Snippet } from "svelte";

    let { children, text }: { children?: Snippet; text: string } = $props();
    let tip = $state("Copy");

    async function copy() {
        await navigator.clipboard.writeText(text);
        tip = "Copied!";
    }

    function reset() {
        tip = "Copy";
    }
</script>

<button type="button" class="ui-hover-btn" onclick={copy}>
    {@render children?.()}
    <Tooltip class="dark:bg-gray-900" onbeforetoggle={reset}>
        {tip}
    </Tooltip>
</button>
