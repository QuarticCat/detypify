<script lang="ts">
    import { contribSyms } from "detypify-service";
    import type { Strokes } from "detypify-service";
    import { Button, Input } from "flowbite-svelte";
    import { RefreshOutline } from "flowbite-svelte-icons";

    export type Sample = {
        id: string;
        name: string;
        strokes: Strokes;
    };

    const symKeys = Object.keys(contribSyms);
    const contribUrl = import.meta.env.DEV
        ? "http://localhost:8787/contrib"
        : "https://detypify.quarticcat.workers.dev/contrib";

    let {
        input = $bindable(),
        strokes = $bindable(),
        samples = $bindable(),
    }: {
        input: string;
        strokes: Strokes;
        samples: Sample[];
    } = $props();

    let submitting = $state(false);
    const isValid = $derived(Boolean(contribSyms[input]));

    function refresh() {
        const old = input;
        while ((input = symKeys[Math.floor(symKeys.length * Math.random())]) === old);
        strokes = [];
    }

    function save() {
        const sample = {
            id: crypto.randomUUID(),
            name: input,
            strokes,
        };
        samples = [sample, ...samples];
        strokes = [];
    }

    function getToken(): number {
        const existing = localStorage.getItem("token");
        if (existing) return Number(existing);

        const token = crypto.getRandomValues(new Uint32Array(1))[0].toString();
        localStorage.setItem("token", token);
        return Number(token);
    }

    async function submit() {
        submitting = true;
        try {
            const response = await fetch(contribUrl, {
                method: "POST",
                headers: {
                    Accept: "application/json",
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    ver: 3,
                    token: getToken(),
                    samples: samples.map(({ name, strokes }) => [name, strokes]),
                }),
            });
            samples = []; // clear samples only if success
            window.alert(await response.text());
        } catch (err) {
            window.alert(err instanceof Error ? err.message : String(err));
        }
        submitting = false;
    }
</script>

<Input type="text" placeholder="symbol" color={isValid ? "green" : "red"} bind:value={input}>
    {#snippet right()}
        <button type="button" class="ui-hover-btn" onclick={refresh}>
            <RefreshOutline />
        </button>
    {/snippet}
</Input>

<div class="flex justify-around gap-4">
    <Button class="w-full" disabled={!isValid || strokes.length === 0} onclick={save}>Save</Button>
    <Button class="w-full" disabled={samples.length === 0} onclick={submit} loading={submitting}>Submit</Button>
</div>
