<script>
    import { Avatar, P, Tooltip } from "flowbite-svelte";
    import { fly } from "svelte/transition";
    import Button from "../utils/Button.svelte";

    export let logo, info;

    let tip;

    async function copy() {
        await navigator.clipboard.writeText(this.textContent);
        tip = "Copied!";
    }
</script>

<div
    class="flex flex-row items-center space-x-4 rounded-lg border border-gray-200 bg-white p-4 text-gray-500 shadow-md dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400"
    in:fly={{ x: -20, duration: 200 }}
    out:fly={{ x: 20, duration: 200 }}
>
    <Button on:click={copy}>
        <Avatar rounded size="lg" class="text-5xl" style="font-family: NewCMMath-Detypify;">
            {logo}
        </Avatar>
        <Tooltip class="dark:bg-gray-900" on:show={() => (tip = "Copy")}>
            {tip}
        </Tooltip>
    </Button>
    <div class="space-y-1">
        {#each info as [key, value]}
            <P>
                {key}:
                <Button on:click={copy}>
                    <code class="text-base font-medium">
                        {value}
                    </code>
                </Button>
                <Tooltip class="dark:bg-gray-900" on:show={() => (tip = "Copy")}>
                    {tip}
                </Tooltip>
            </P>
        {/each}
    </div>
</div>
