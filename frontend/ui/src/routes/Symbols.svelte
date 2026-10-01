<script lang="ts">
    import { inferSyms } from "detypify-service";

    import Candidate from "../lib/Candidate.svelte";

    // Sort symbols by their primary name so the grid reads alphabetically.
    const symbols = [...inferSyms].sort((a, b) => a.names[0].localeCompare(b.names[0]));
</script>

<div class="ui-grid">
    {#each symbols as info (info.char)}
        <Candidate {info} />
    {/each}
</div>
