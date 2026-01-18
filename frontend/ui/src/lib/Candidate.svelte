<script lang="ts">
    import Card from "./Card.svelte";
    import CopyButton from "./CopyButton.svelte";
    import type { SymbolInfo } from "detypify-service";
    import { Avatar, P } from "flowbite-svelte";

    const { info }: { info: SymbolInfo } = $props();
</script>

<Card>
    {@const escape = `\\u{${info.char.codePointAt(0)?.toString(16).toUpperCase().padStart(4, "0")}}`}
    {@const shorthand = info.shorthand ?? info.markupShorthand ?? info.mathShorthand}
    {@const shorthandKind = info.markupShorthand ? "markup" : info.mathShorthand ? "math" : ""}
    <CopyButton text={info.char}>
        <Avatar cornerStyle="rounded" size="lg" class="font-[NewCMMath-Detypify] text-5xl">
            {info.char}
        </Avatar>
    </CopyButton>
    <div class="flex flex-col gap-y-1">
        <P>
            Name:
            {#each info.names as name, i}
                {i === 0 ? "" : " | "}
                <CopyButton text={name}>
                    <code class="text-base font-medium">
                        {name}
                    </code>
                </CopyButton>
            {/each}
        </P>
        <P>
            Escape:
            <CopyButton text={escape}>
                <code class="text-base font-medium">
                    {escape}
                </code>
            </CopyButton>
        </P>
        {#if shorthand}
            <P>
                Shorthand:
                <CopyButton text={shorthand}>
                    <code class="text-base font-medium">
                        {shorthand}
                    </code>
                </CopyButton>
                {#if shorthandKind}
                    <span class="text-sm text-gray-500">({shorthandKind})</span>
                {/if}
            </P>
        {/if}
    </div>
</Card>
