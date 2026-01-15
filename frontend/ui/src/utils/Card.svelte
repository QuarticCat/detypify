<script lang="ts">
    import { fly } from "svelte/transition";

    let { class: className = "", children, ...restProps } = $props();
</script>

<div
    class={`flex flex-row items-center space-x-4 rounded-lg border border-gray-200 bg-white p-4 text-gray-500 shadow-md dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 ${className}`}
    in:fly={{ x: -20, duration: 200 }}
    out:fly={{ x: 20, duration: 200 }}
    {...restProps}
>
    {@render children?.()}
</div>
