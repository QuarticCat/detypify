<script lang="ts">
    import Card from "../utils/Card.svelte";
    import { contribSyms } from "detypify-service";
    import { Avatar, Hr, Tooltip } from "flowbite-svelte";
    import { CloseOutline } from "flowbite-svelte-icons";

    const BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAP7//wAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==";
    const { name, img = BLANK, ondelete }: { name: string; img?: string; ondelete?: () => void } = $props();
</script>

<Card class="relative flex justify-center">
    <Avatar cornerStyle="rounded" size="lg" class="font-[NewCMMath-Detypify] text-5xl">
        <!-- Workaround Safari font rendering issues. -->
        <span class="-m-5 p-5">{contribSyms[name]}</span>
    </Avatar>
    <Hr class="h-1 w-12 rounded" />
    <Avatar cornerStyle="rounded" size="lg" src={img} />

    {#if ondelete}
        <button type="button" class="ui-hover-btn ui-close-btn" onclick={ondelete}>
            <CloseOutline class="size-6" />
            <Tooltip class="dark:bg-gray-900">Delete</Tooltip>
        </button>
    {/if}
</Card>
