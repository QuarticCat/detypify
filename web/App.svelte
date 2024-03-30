<script>
    import "./app.pcss";
    import NavBar from "./components/NavBar.svelte";
    import Canvas from "./components/Canvas.svelte";
    import Infer from "./components/Infer.svelte";
    import { Spinner } from "flowbite-svelte";
    import { env, InferenceSession } from "onnxruntime-web";
    import modelUrl from "../train-out/model.onnx";

    env.wasm.numThreads = 1;
    env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/"; // match the version in package.json

    let sessionPromise = InferenceSession.create(modelUrl);
    let greyscale;
</script>

<NavBar />

<div class="flex flex-wrap justify-evenly py-16 md:px-32">
    <Canvas bind:greyscale />
    <div class="w-[400px] text-center">
        {#await sessionPromise}
            <Spinner size="12" />
        {:then session}
            <Infer {session} {greyscale} />
        {/await}
    </div>
</div>
