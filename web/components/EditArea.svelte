<script>
    import { Button, Input, Spinner } from "flowbite-svelte";
    import { RefreshOutline } from "flowbite-svelte-icons";
    import symbols from "../../train-out/symbols.json";
    import { imgUrl, inputText, savedSamples, strokes } from "../store";
    import MyButton from "../utils/Button.svelte";

    let symbolKeys = Object.keys(symbols);

    let inputColor = "base";
    let disableSave = true;
    let disableSubmit = true;

    let sampleId = 0;

    let isSubmitting = false;

    function validateInput(input) {
        if (input === "") {
            inputColor = "base";
        } else if (symbols[input]) {
            inputColor = "green";
        } else {
            inputColor = "red";
        }
    }

    function refreshInput() {
        let oldInput = $inputText;
        let newInput;
        do {
            newInput = symbolKeys[(symbolKeys.length * Math.random()) << 0];
        } while (newInput === oldInput);
        $inputText = newInput;
    }

    function save() {
        $savedSamples = [
            {
                id: sampleId,
                name: $inputText,
                logo: symbols[$inputText],
                strokes: $strokes,
                imgUrl: $imgUrl,
            },
            ...$savedSamples,
        ];
        sampleId += 1;
        $strokes = [];
    }

    // async function submit() {
    //     isSubmitting = true;

    //     let samples = $savedSamples.map(({ name, strokes }) => [name, strokes]);
    //     $savedSamples = [];

    //     let form = new FormData();
    //     form.append("c", JSON.stringify(samples));
    //     let response = await fetch(`https://corsproxy.io/?${encodeURIComponent("https://pb.mgt.moe/")}`, {
    //         method: "POST",
    //         body: form,
    //     });
    //     let text = await response.text();

    //     let pasteUrl = text.match(/url: (.*?)\n/)[1];
    //     window.open(
    //         `https://github.com/QuarticCat/detypify-data/issues/new` +
    //             `?title=${encodeURIComponent("Samples 0.1.0")}` +
    //             `&body=${encodeURIComponent(`${pasteUrl}\n\n<!-- Do not modify, just submit -->`)}`,
    //     );

    //     isSubmitting = false;
    // }

    async function submit() {
        isSubmitting = true;

        let samples = $savedSamples.map(({ name, strokes }) => [name, strokes]);
        $savedSamples = [];

        await navigator.clipboard.writeText(JSON.stringify(samples));
        window.open(
            `https://github.com/QuarticCat/detypify-data/issues/new` +
                `?title=${encodeURIComponent("Samples 0.2.0")}` +
                `&body=${encodeURIComponent(`<!-- Data has been saved to your clipboard. Paste them just this line and submit. -->\n`)}`,
        );

        isSubmitting = false;
    }

    $: validateInput($inputText);
    $: disableSave = inputColor !== "green" || $strokes.length === 0;
    $: disableSubmit = isSubmitting || $savedSamples.length === 0;
</script>

<div>
    <Input type="text" placeholder="Symbol Name" color={inputColor} bind:value={$inputText}>
        <MyButton slot="right" on:click={refreshInput}><RefreshOutline /></MyButton>
    </Input>
</div>

<div class="flex justify-around">
    <Button class="w-5/12" disabled={disableSave} on:click={save}>Save</Button>
    <Button class="w-5/12" disabled={disableSubmit} on:click={submit}>
        {#if isSubmitting}
            <Spinner size="5" />
        {:else}
            Submit
        {/if}
    </Button>
</div>
