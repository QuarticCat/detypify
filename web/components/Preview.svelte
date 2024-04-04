<script>
    import { Avatar, Hr, Tooltip } from "flowbite-svelte";
    import { CloseOutline } from "flowbite-svelte-icons";
    import Button from "../utils/Button.svelte";
    import Card from "../utils/Card.svelte";
    import { savedSamples } from "../store";

    export let id, imgUrl, logo;
    export let closeable = false;

    function deleteSelf() {
        $savedSamples = $savedSamples.filter((s) => s.id !== id);
    }
</script>

<Card class="relative flex justify-center">
    <Avatar rounded size="lg" class="font-[NewCMMath-Detypify] text-5xl">
        {logo}
    </Avatar>
    <Hr classHr="w-12 h-1 rounded" />
    <Avatar rounded size="lg" src={imgUrl} />

    <Button class="absolute right-1 top-1 p-2 {closeable ? '' : 'hidden'}" on:click={deleteSelf}>
        <CloseOutline class="size-6" />
        <Tooltip class="dark:bg-gray-900">Delete</Tooltip>
    </Button>
</Card>
