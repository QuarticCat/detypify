import { svelte } from "@sveltejs/vite-plugin-svelte";
import { VitePWA } from "vite-plugin-pwa";

/** @type {import('vite').UserConfig} */
export default {
    resolve: {
        conditions: ["onnxruntime-web-use-extern-wasm"],
    },
    optimizeDeps: {
        esbuildOptions: {
            conditions: ["onnxruntime-web-use-extern-wasm"],
        },
    },
    assetsInclude: ["**/*.onnx"],
    plugins: [
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
                icons: [
                    {
                        src: "icons/pwa-192x192.png",
                        sizes: "192x192",
                        type: "image/png",
                    },
                    {
                        src: "icons/pwa-512x512.png",
                        sizes: "512x512",
                        type: "image/png",
                    },
                    {
                        src: "icons/pwa-512x512.png",
                        sizes: "512x512",
                        type: "image/png",
                        purpose: "any",
                    },
                    {
                        src: "icons/pwa-512x512.png",
                        sizes: "512x512",
                        type: "image/png",
                        purpose: "maskable",
                    },
                ],
            },
        }),
    ],
};
