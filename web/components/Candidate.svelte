<script>
    import { Avatar, P } from "flowbite-svelte";
    import Card from "../utils/Card.svelte";
    import CopyButton from "../utils/CopyButton.svelte";

    export let logo, info;
</script>

<Card>
    <CopyButton>
        <Avatar rounded size="lg" class="font-[NewCMMath-Detypify] text-5xl">
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
</Card>
