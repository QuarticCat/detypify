<script>
    import { Avatar, Hr } from "flowbite-svelte";
    import Card from "../utils/Card.svelte";

    export let imgUrl, logo;
</script>

<Card class="flex justify-center">
    <Avatar rounded size="lg" class="font-[NewCMMath-Detypify] text-5xl">
        {logo}
    </Avatar>
    <Hr classHr="w-12 h-1 rounded" />
    <Avatar rounded size="lg" src={imgUrl} />
</Card>
