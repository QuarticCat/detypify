import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
    resolve: {
        conditions: ["module", "browser", "onnxruntime-web-use-extern-wasm"],
    },
    assetsInclude: ["**/*.onnx"],
    server: {
        fs: {
            allow: [".."],
        },
    },
    plugins: [tailwindcss(), svelte()],
});
