<script lang="ts">
    let { class: className = "", children, ...restProps } = $props();
</script>

<button
    type="button"
    class={`text-black hover:text-primary-700 dark:text-gray-400 dark:hover:text-white ${className}`}
    {...restProps}
>
    {@render children?.()}
</button>
