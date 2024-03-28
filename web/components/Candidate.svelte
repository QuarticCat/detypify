<script>
    import { Card, Avatar, Button, Tooltip, P } from "flowbite-svelte";

    export let logo, info;

    let tip;

    async function copy(text) {
        await navigator.clipboard.writeText(text);
        tip = "Copied!";
    }
</script>

<Card horizontal padding="sm" class="flex items-center space-x-4">
    <Button color="alternative" class="p-0 focus:ring-0" on:click={copy(logo)}>
        <Avatar rounded size="lg" class="text-5xl">
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
                    <code class="font-medium">
                        {value}
                    </code>
                </Button>
                <Tooltip class="dark:bg-gray-900" on:show={() => (tip = "Copy")}>
                    {tip}
                </Tooltip>
            </P>
        {/each}
    </div>
</Card>
