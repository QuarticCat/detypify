import { svelte } from "@sveltejs/vite-plugin-svelte";

/** @type {import('vite').UserConfig} */
export default {
    root: "web",
    build: {
        outDir: "../web-out",
        emptyOutDir: true,
    },
    plugins: [svelte()],
    assetsInclude: ["**/*.onnx"],
};
