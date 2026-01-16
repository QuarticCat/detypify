import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
    resolve: {
        conditions: ["module", "browser", "onnxruntime-web-use-extern-wasm"],
    },
    assetsInclude: ["**/*.onnx"],
    plugins: [
        tailwindcss(),
        svelte(),
        VitePWA({
            registerType: "autoUpdate",
            workbox: {
                globPatterns: ["**/*.{js,css,html,ico,png,svg,onnx,woff2}"],
                runtimeCaching: [
                    {
                        urlPattern: ({ url }) => url.pathname.endsWith(".wasm"),
                        handler: "CacheFirst",
                        options: { cacheName: "wasm-cache" },
                    },
                ],
            },
            manifest: {
                name: "Detypify",
                short_name: "Detypify",
                description: "Typst symbol classifier",
                theme_color: "#ffffff",
            },
            pwaAssets: {
                image: "public/favicon.svg",
            },
        }),
    ],
});
