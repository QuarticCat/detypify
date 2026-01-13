<script>
    import { fly } from "svelte/transition";
</script>

<div
    class="flex flex-row items-center space-x-4 rounded-lg border border-gray-200 bg-white p-4 text-gray-500 shadow-md dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 {$$props.class}"
    in:fly={{ x: -20, duration: 200 }}
    out:fly={{ x: 20, duration: 200 }}
>
    <slot />
</div>
