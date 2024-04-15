<script>
    import { Avatar, P } from "flowbite-svelte";
    import Card from "../utils/Card.svelte";
    import CopyButton from "../utils/CopyButton.svelte";

    export let info;
</script>

<Card>
    <CopyButton>
        <Avatar rounded size="lg" class="font-[NewCMMath-Detypify] text-5xl">
            {String.fromCodePoint(info.codepoint)}
        </Avatar>
    </CopyButton>
    <div class="space-y-1">
        <P>
            Name:
            {#each info.names as name, i}
                {i === 0 ? "" : " | "}
                <CopyButton>
                    <code class="text-base font-medium">
                        {name}
                    </code>
                </CopyButton>
            {/each}
        </P>
        <P>
            Escape:
            <CopyButton>
                <code class="text-base font-medium">
                    {"\\u{" + info.codepoint.toString(16).toUpperCase().padStart(4, "0") + "}"}
                </code>
            </CopyButton>
        </P>
        {#if info.shorthand}
            <P>
                Shorthand:
                <CopyButton>
                    <code class="text-base font-medium">
                        {info.shorthand}
                    </code>
                </CopyButton>
            </P>
        {:else if info.markupShorthand}
            <P>
                Shorthand:
                <CopyButton>
                    <code class="text-base font-medium">
                        {info.markupShorthand}
                    </code>
                </CopyButton>
                <span class="text-sm text-gray-500">(markup)</span>
            </P>
        {:else if info.mathShorthand}
            <P>
                Shorthand:
                <CopyButton>
                    <code class="text-base font-medium">
                        {info.mathShorthand}
                    </code>
                </CopyButton>
                <span class="text-sm text-gray-500">(math)</span>
            </P>
        {/if}
    </div>
</Card>
