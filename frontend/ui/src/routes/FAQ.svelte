<script lang="ts">
    import { inferSyms } from "detypify-service";
    import { AccordionItem, Accordion, A, P, Li, List } from "flowbite-svelte";

    const supportedSyms = inferSyms.flatMap((info) => info.names);
    supportedSyms.sort();
</script>

<div class="ui-sub-container w-180">
    <Accordion>
        <AccordionItem>
            {#snippet header()}Can't find your symbol?{/snippet}
            <P>
                Supported symbols are listed in section below. To recognize characters out of this scope, consider
                giving <A href="https://shapecatcher.com/">shapecatcher</A> a try!
            </P>
        </AccordionItem>
        <AccordionItem>
            {#snippet header()}Supported symbols?{/snippet}
            <List>
                {#each supportedSyms as name}
                    <Li>{name}</Li>
                {/each}
            </List>
        </AccordionItem>
        <AccordionItem>
            {#snippet header()}Results are inaccurate?{/snippet}
            <P>
                Some privacy settings (like <A href="https://brave.com/shields/">Brave Shields</A>) have canvas
                anti-fingerprinting, which might confuse the classifier. Make sure they are turned off for this site.
                Don't worry! We don't collect any privacy data. This project is fully open-source and works locally on
                your machine.
            </P>
        </AccordionItem>
        <AccordionItem>
            {#snippet header()}Use it offline?{/snippet}
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
