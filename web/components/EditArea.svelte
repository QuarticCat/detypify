<script>
    import { Button, Input } from "flowbite-svelte";
    import { RefreshOutline } from "flowbite-svelte-icons";
    import { imgUrl, inputText, savedSamples, strokes } from "../store";
    import MyButton from "../utils/Button.svelte";

    let symbols = {
        "fence.l.double": "⧚",
        "ast.circle": "⊛",
    };
    let symbolKeys = Object.keys(symbols);

    let inputColor = "base";
    let disableSave = true;
    let disableSubmit = true;

    let sampleId = 0;

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

    function submit() {}

    $: validateInput($inputText);
    $: disableSave = inputColor !== "green" || $strokes.length === 0;
    $: disableSubmit = $savedSamples.length === 0;
</script>

<div>
    <Input type="text" placeholder="Symbol Name" color={inputColor} bind:value={$inputText}>
        <MyButton slot="right" on:click={refreshInput}><RefreshOutline /></MyButton>
    </Input>
</div>

<div class="flex justify-around">
    <Button class="w-5/12" disabled={disableSave} on:click={save}>Save</Button>
    <Button class="w-5/12" disabled={disableSubmit} on:click={submit}>Submit</Button>
</div>
