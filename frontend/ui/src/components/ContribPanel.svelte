<script lang="ts">
    import { imgUrl, inputText, savedSamples, strokes } from "../store";
    import MyButton from "../utils/Button.svelte";
    import { contribSyms } from "detypify-service";
    import { Button, Input, Modal, Spinner } from "flowbite-svelte";
    import { RefreshOutline } from "flowbite-svelte-icons";

    const symKeys = Object.keys(contribSyms);
    let isSubmitting = $state(false);
    let modalOpen = $state(false);
    let modalOk = $state(false);
    let modalText = $state("");

    const inputColor = $derived<"green" | "red">(contribSyms[$inputText] ? "green" : "red");
    const disableSave = $derived(inputColor !== "green" || $strokes.length === 0);
    const disableSubmit = $derived(isSubmitting || $savedSamples.length === 0);

    function refreshInput() {
        const old = $inputText;
        while (($inputText = symKeys[Math.floor(symKeys.length * Math.random())]) === old);
        $strokes = [];
    }

    let sampleId = 0;
    function save() {
        const name = $inputText;
        const logo = contribSyms[name];
        if (!logo) return;
        $savedSamples = [
            {
                id: sampleId,
                name,
                logo,
                strokes: $strokes,
                imgUrl: $imgUrl,
            },
            ...$savedSamples,
        ];
        sampleId += 1;
        $strokes = [];
    }

    if (!localStorage.getItem("token")) {
        localStorage.setItem("token", window.crypto.getRandomValues(new Uint32Array(1))[0].toString());
    }

    async function submit() {
        isSubmitting = true;

        const payload = {
            ver: 3,
            token: Number(localStorage.getItem("token")),
            samples: $savedSamples.map(({ name, strokes }) => [name, strokes]),
        };
        $savedSamples = [];

        try {
            const response = await fetch("https://detypify.quarticcat.workers.dev/contrib", {
                method: "POST",
                headers: {
                    Accept: "application/json",
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });
            modalOk = response.ok;
            modalText = await response.text();
        } catch (err) {
            modalOk = false;
            modalText = err instanceof Error ? err.message : String(err);
        }
        modalOpen = true;

        isSubmitting = false;
    }
</script>

<Input type="text" placeholder="symbol" color={inputColor} bind:value={$inputText}>
    {#snippet right()}
        <MyButton onclick={refreshInput}><RefreshOutline /></MyButton>
    {/snippet}
</Input>

<div class="flex justify-around gap-4">
    <Button class="w-full" disabled={disableSave} onclick={save}>Save</Button>
    <Button class="w-full" disabled={disableSubmit} onclick={submit}>
        {#if isSubmitting}
            <Spinner size="5" />
        {:else}
            Submit
        {/if}
    </Button>
</div>

<Modal size="xs" bind:open={modalOpen} dismissable>
    {#snippet header()}
        <h3>Result</h3>
    {/snippet}
    <div class={`text-center text-xl ${modalOk ? "text-green-600" : "text-red-600"}`}>
        {modalText}
    </div>
</Modal>
