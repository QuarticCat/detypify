<script lang="ts">
    import Canvas from "../lib/Canvas.svelte";
    import ContribPanel from "../lib/ContribPanel.svelte";
    import Preview from "../lib/Preview.svelte";
    import { strokes, samples } from "../store";
    import type { Strokes } from "detypify-service";
    import { Detypify } from "detypify-service";
    import { Hr } from "flowbite-svelte";

    const { session }: { session: Detypify } = $props();

    let contribSymName = $state("");

    function draw(strokes: Strokes): string | undefined {
        if (strokes.length === 0) return;
        return session.draw(strokes)?.toDataURL();
    }

    function deletePreview(id: string) {
        $samples = $samples.filter((s) => s.id !== id);
    }
</script>

<div class="flex flex-col gap-4 w-80">
    <Canvas />
    <ContribPanel bind:value={contribSymName} />
</div>

<div class="flex flex-col gap-4 w-100">
    <Preview name={contribSymName} img={draw($strokes)} />
    <Hr class="mx-auto h-2 w-60 rounded" />
    {#each $samples as { id, name, strokes } (id)}
        <Preview {name} img={draw(strokes)} ondelete={() => deletePreview(id)} />
    {/each}
</div>
