<script lang="ts">
    import { savedSamples } from "../store";
    import Button from "../utils/Button.svelte";
    import Card from "../utils/Card.svelte";
    import { Avatar, Hr, Tooltip } from "flowbite-svelte";
    import { CloseOutline } from "flowbite-svelte-icons";

    const { id, logo, imgUrl }: { id?: number; logo: string; imgUrl?: string; } = $props();

    function deleteSelf() {
        if (id === undefined) return;
        $savedSamples = $savedSamples.filter((s) => s.id !== id);
    }
</script>

<Card class="relative flex justify-center">
    <Avatar cornerStyle="rounded" size="lg" class="font-[NewCMMath-Detypify] text-5xl">
        <!-- Workaround Safari font rendering issues. -->
        <span class="-m-5 p-5">{logo}</span>
    </Avatar>
    <Hr class="h-1 w-12 rounded" />
    <Avatar cornerStyle="rounded" size="lg" src={imgUrl} />

    <Button class={`absolute right-1 top-1 p-2 ${id === undefined ? "hidden" : ""}`} onclick={deleteSelf}>
        <CloseOutline class="size-6" />
        <Tooltip class="dark:bg-gray-900">Delete</Tooltip>
    </Button>
</Card>
