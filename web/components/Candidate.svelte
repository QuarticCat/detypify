<script>
    import { Avatar, P } from "flowbite-svelte";
    import { fly } from "svelte/transition";
    import CopyButton from "../utils/CopyButton.svelte";

    export let logo, info;
</script>

<div
    class="flex flex-row items-center space-x-4 rounded-lg border border-gray-200 bg-white p-4 text-gray-500 shadow-md dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400"
    in:fly={{ x: -20, duration: 200 }}
    out:fly={{ x: 20, duration: 200 }}
>
    <CopyButton>
        <Avatar rounded size="lg" class="text-5xl" style="font-family: NewCMMath-Detypify;">
            {logo}
        </Avatar>
    </CopyButton>
    <div class="space-y-1">
        {#each info as [key, value]}
            <P>
                {key}:
                <CopyButton>
                    <code class="text-base font-medium">
                        {value}
                    </code>
                </CopyButton>
            </P>
        {/each}
    </div>
</div>
