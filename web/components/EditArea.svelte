<script>
    import { Button, Input, Spinner } from "flowbite-svelte";
    import { RefreshOutline } from "flowbite-svelte-icons";
    import symbols from "../../train-out/symbols.json";
    import { imgUrl, inputText, savedSamples, strokes } from "../store";
    import MyButton from "../utils/Button.svelte";

    // Ref: https://github.com/toeverything/blocksuite/blob/master/packages/framework/global/src/env/index.ts
    const IS_SAFARI = window.navigator && /Apple Computer/.test(window.navigator.vendor);

    let symbolKeys = Object.keys(symbols);

    let inputColor = "base";
    let disableSave = true;
    let disableSubmit = true;

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
        let old = $inputText;
        while (($inputText = symbolKeys[(symbolKeys.length * Math.random()) << 0]) === old);
    }

    let sampleId = 0;
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

    let copyText = "Copy";
    async function copyOnSafari() {
        await navigator.clipboard.writeText(JSON.stringify(samples));
        copyText = "Copied!";
        setTimeout(() => (copyText = "Copy"), 500);
    }

    let isSubmitting = false;
    async function submit() {
        isSubmitting = true;

        let samples = $savedSamples.map(({ name, strokes }) => [name, strokes]);
        $savedSamples = [];

        if (!IS_SAFARI) await navigator.clipboard.writeText(JSON.stringify(samples));
        let title = encodeURIComponent("Samples 0.2.0");
        let body = encodeURIComponent(`<!--
- Data has been saved to your clipboard (Safari users need to click copy button)
- Paste it below and submit
- Don't modify the title or add extra description (use comments instead)
-->\n`);
        window.open(`https://github.com/QuarticCat/detypify-data/issues/new?title=${title}&body=${body}`);

        isSubmitting = false;
    }

    $: validateInput($inputText);
    $: disableSave = inputColor !== "green" || $strokes.length === 0;
    $: disableSubmit = isSubmitting || $savedSamples.length === 0;
</script>

<div>
    <Input type="text" placeholder="Symbol" color={inputColor} bind:value={$inputText}>
        <MyButton slot="right" on:click={refreshInput}><RefreshOutline /></MyButton>
    </Input>
</div>

<div class="flex justify-around gap-4">
    <Button class="w-full" disabled={disableSave} on:click={save}>Save</Button>
    {#if IS_SAFARI}
        <Button class="w-full" disabled={disableSubmit} on:click={copyOnSafari}>{copyText}</Button>
    {/if}
    <Button class="w-full" disabled={disableSubmit} on:click={submit}>
        {#if isSubmitting}
            <Spinner size="5" />
        {:else}
            Submit
        {/if}
    </Button>
</div>
