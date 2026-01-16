<script lang="ts">
    let { class: className = "", children, ...restProps } = $props();
</script>

<button
    type="button"
    class={`text-gray-600 hover:text-black dark:text-gray-400 dark:hover:text-white ${className}`}
    {...restProps}
>
    {@render children?.()}
</button>
