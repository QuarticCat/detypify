<script>
    import Candidate from "./Candidate.svelte";
    import { Tensor } from "onnxruntime-web";
    import classes from "../../train-out/classes.json";

    export let session, greyscale;

    let candidates;

    function infer(data) {
        if (!data) {
            candidates = [];
            return;
        }
        let tensor = new Tensor("float32", data, [1, 1, 32, 32]);
        session.run({ [session.inputNames[0]]: tensor }).then((output) => {
            output = Array.prototype.slice.call(output[session.outputNames[0]].data);
            let withIdx = output.map((x, i) => [x, i]);
            withIdx.sort((a, b) => b[0] - a[0]);
            candidates = withIdx.slice(0, 5).map(([_, i]) => classes[i]);
        });
    }

    $: infer(greyscale);
</script>

{#each candidates as [logo, info]}
    <Candidate {logo} {info} />
{/each}
