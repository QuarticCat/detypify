<script>
    import { Button, Input } from "flowbite-svelte";
    import { RefreshOutline } from "flowbite-svelte-icons";
    import MyButton from "../utils/Button.svelte";
    import { greyscale, inputText, savedSamples } from "../store";

    let symbols = {
        "fence.l.double": "⧚",
        "ast.circle": "⊛",
    };
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
        let oldInput = $inputText;
        let newInput;
        do {
            newInput = symbolKeys[(symbolKeys.length * Math.random()) << 0];
        } while (newInput === oldInput);
        $inputText = newInput;
    }

    $: validateInput($inputText);
    $: disableSave = inputColor !== "green" || !$greyscale;
    $: disableSubmit = $savedSamples.length === 0;
</script>

<div>
    <Input type="text" placeholder="Symbol Name" color={inputColor} bind:value={$inputText}>
        <MyButton slot="right" on:click={refreshInput}><RefreshOutline /></MyButton>
    </Input>
</div>

<div class="flex justify-around">
    <Button class="w-5/12" disabled={disableSave}>Save</Button>
    <Button class="w-5/12" disabled={disableSubmit}>Submit</Button>
</div>
