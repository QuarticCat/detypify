<script>
    import "./app.pcss";
    import NavBar from "./components/NavBar.svelte";
    import Canvas from "./components/Canvas.svelte";
    import Candidate from "./components/Candidate.svelte";
    import { Spinner } from "flowbite-svelte";
    import { env as ortConfig, InferenceSession, Tensor } from "onnxruntime-web";
    import modelUrl from "../train-out/model.onnx";
    import classes from "../train-out/classes.json";

    ortConfig.wasm.numThreads = 1;
    ortConfig.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

    let session, greyscale;
    let isLoading = true;
    let candidates = [];

    function infer(grey) {
        if (isLoading || !grey) {
            candidates = [];
            return;
        }
        let tensor = new Tensor("float32", grey, [1, 1, 32, 32]);
        session.run({ [session.inputNames[0]]: tensor }).then((output) => {
            output = Array.prototype.slice.call(output[session.outputNames[0]].data);
            let withIdx = output.map((x, i) => [x, i]);
            withIdx.sort((a, b) => b[0] - a[0]);
            candidates = withIdx.slice(0, 5).map(([_, i]) => classes[i]);
        });
    }

    InferenceSession.create(modelUrl).then((s) => {
        session = s;
        isLoading = false;
        infer(greyscale);
    });

    $: infer(greyscale);
</script>

<NavBar />

<div class="flex flex-wrap justify-evenly space-y-4 md:py-4">
    <div class="hidden md:block"></div>
    <Canvas bind:greyscale />
    <div class="w-[360px] space-y-4 text-center">
        {#if isLoading}
            <Spinner size="12" />
        {:else}
            {#each candidates as [logo, info]}
                <Candidate {logo} {info} />
            {/each}
        {/if}
    </div>
    <div class="hidden md:block"></div>
</div>
