<script lang="ts">
    import Card from "../utils/Card.svelte";
    import CopyButton from "../utils/CopyButton.svelte";
    import { Avatar, P } from "flowbite-svelte";

    type CandidateInfo = {
        char?: string;
        codepoint?: number;
        names: string[];
        shorthand?: string;
        markupShorthand?: string;
        mathShorthand?: string;
    };

    const { info } = $props<{ info: CandidateInfo }>();

    const symbolChar = $derived(info.char ?? (info.codepoint ? String.fromCodePoint(info.codepoint) : ""));
    const codepoint = $derived(
        info.codepoint ?? (symbolChar ? (symbolChar.codePointAt(0) ?? undefined) : undefined),
    );
    const escapeCode = $derived(
        codepoint !== undefined ? `\\u{${codepoint.toString(16).toUpperCase().padStart(4, "0")}}` : "",
    );
</script>

<Card>
    <CopyButton>
        <Avatar cornerStyle="rounded" size="lg" class="font-[NewCMMath-Detypify] text-5xl">
            {symbolChar}
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
                    {escapeCode}
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
