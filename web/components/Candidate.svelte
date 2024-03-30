<script>
    import { fly } from "svelte/transition";
    import { Avatar, Button, Tooltip, P } from "flowbite-svelte";

    export let logo, info;

    let tip;

    async function copy(text) {
        await navigator.clipboard.writeText(text);
        tip = "Copied!";
    }
</script>

<div
    class="flex flex-row items-center space-x-4 rounded-lg border border-gray-200 bg-white p-4 text-gray-500 shadow-md dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400"
    in:fly={{ x: -20, duration: 200 }}
    out:fly={{ x: 20, duration: 200 }}
>
    <Button color="alternative" class="p-0 focus:ring-0" on:click={copy(logo)}>
        <Avatar rounded size="lg" class="text-5xl" style="font-family: NewCMMath-Detypify;">
            {logo}
        </Avatar>
    </Button>
    <Tooltip class="dark:bg-gray-900" on:show={() => (tip = "Copy")}>
        {tip}
    </Tooltip>
    <div class="space-y-1">
        {#each info as [key, value]}
            <P>
                {key}:
                <Button color="alternative" class="rounded-sm border-0 p-0 focus:ring-0" on:click={copy(value)}>
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
