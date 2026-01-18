<script lang="ts">
    import { inferSyms } from "detypify-service";
    import { AccordionItem, Accordion, A, P, Li, List } from "flowbite-svelte";

    const supportedSyms = inferSyms.flatMap((info) => info.names);
    supportedSyms.sort();
</script>

<div class="ui-sub-container w-180">
    <Accordion>
        <AccordionItem>
            {#snippet header()}Supported symbols{/snippet}
            <List>
                {#each supportedSyms as name}
                    <Li>{name}</Li>
                {/each}
            </List>
        </AccordionItem>
        <AccordionItem>
            {#snippet header()}Use it offline{/snippet}
            <P>
                Check the <A href="https://support.google.com/chrome/answer/9658361">guide</A>.
            </P>
        </AccordionItem>
        <AccordionItem>
            {#snippet header()}Support us{/snippet}
            <P>
                Star this project on <A href="https://github.com/QuarticCat/detypify">GitHub</A>!
            </P>
        </AccordionItem>
    </Accordion>
</div>
