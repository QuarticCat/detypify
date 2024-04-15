<script>
    import { Button, Input, Modal, Spinner } from "flowbite-svelte";
    import { RefreshOutline } from "flowbite-svelte-icons";
    import contribSyms from "../../train-out/contrib.json";
    import { imgUrl, inputText, savedSamples, strokes } from "../store";
    import MyButton from "../utils/Button.svelte";

    let symKeys = Object.keys(contribSyms);
    let inputColor, disableSave, disableSubmit;

    function validateInput(input) {
        if (contribSyms[input]) {
            inputColor = "green";
        } else {
            inputColor = "red";
        }
    }

    function refreshInput() {
        let old = $inputText;
        while (($inputText = symKeys[(symKeys.length * Math.random()) << 0]) === old);
        $strokes = [];
    }

    let sampleId = 0;
    function save() {
        $savedSamples = [
            {
                id: sampleId,
                name: $inputText,
                logo: contribSyms[$inputText],
                strokes: $strokes,
                imgUrl: $imgUrl,
            },
            ...$savedSamples,
        ];
        sampleId += 1;
        $strokes = [];
    }

    if (!localStorage.getItem("token")) {
        localStorage.setItem("token", window.crypto.getRandomValues(new Uint32Array(1))[0]);
    }

    let isSubmitting = false;
    let modalOpen = false;
    let modalOk, modalText;
    async function submit() {
        isSubmitting = true;

        let payload = {
            ver: 3,
            token: Number(localStorage.getItem("token")),
            samples: $savedSamples.map(({ name, strokes }) => [name, strokes]),
        };
        $savedSamples = [];

        try {
            let response = await fetch("https://detypify.quarticcat.workers.dev/contrib", {
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
            modalText = err;
        }
        modalOpen = true;

        isSubmitting = false;
    }

    $: validateInput($inputText);
    $: disableSave = inputColor !== "green" || $strokes.length === 0;
    $: disableSubmit = isSubmitting || $savedSamples.length === 0;
</script>

<Input type="text" placeholder="symbol" color={inputColor} bind:value={$inputText}>
    <MyButton slot="right" on:click={refreshInput}><RefreshOutline /></MyButton>
</Input>

<div class="flex justify-around gap-4">
    <Button class="w-full" disabled={disableSave} on:click={save}>Save</Button>
    <Button class="w-full" disabled={disableSubmit} on:click={submit}>
        {#if isSubmitting}
            <Spinner size="5" />
        {:else}
            Submit
        {/if}
    </Button>
</div>

<Modal title="Result" size="xs" color={modalOk ? "green" : "red"} bind:open={modalOpen} outsideclose>
    <div class="text-center text-xl">
        {modalText}
    </div>
</Modal>
