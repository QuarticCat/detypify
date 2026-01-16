<script lang="ts">
    import Card from "../utils/Card.svelte";
    import CopyButton from "../utils/CopyButton.svelte";
    import type { SymbolInfo } from "detypify-service";
    import { Avatar, P } from "flowbite-svelte";

    const { info }: { info: SymbolInfo } = $props();
</script>

<Card>
    <CopyButton>
        <Avatar cornerStyle="rounded" size="lg" class="font-[NewCMMath-Detypify] text-5xl">
            {info.char}
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
                    \u{info.char.codePointAt(0)?.toString(16).toUpperCase().padStart(4, "0")}
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
