import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// https://vitejs.dev/config/
export default defineConfig({
    root: "web",
    base: "/detypify",
    build: {
        outDir: "../web-out",
        emptyOutDir: true,
    },
    plugins: [svelte()],
});
