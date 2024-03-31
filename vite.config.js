import { svelte } from "@sveltejs/vite-plugin-svelte";
import { VitePWA } from "vite-plugin-pwa";

/** @type {import('vite').UserConfig} */
export default {
    root: "web",
    build: {
        outDir: "../web-out",
        emptyOutDir: true,
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
                        urlPattern: ({ url }) => url.origin === "https://cdn.jsdelivr.net",
                        handler: "CacheFirst",
                        options: {
                            cacheName: "jsdelivr",
                        },
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
                        src: "pwa-192x192.png",
                        sizes: "192x192",
                        type: "image/png",
                    },
                    {
                        src: "pwa-512x512.png",
                        sizes: "512x512",
                        type: "image/png",
                    },
                    {
                        src: "pwa-512x512.png",
                        sizes: "512x512",
                        type: "image/png",
                        purpose: "any",
                    },
                    {
                        src: "pwa-512x512.png",
                        sizes: "512x512",
                        type: "image/png",
                        purpose: "maskable",
                    },
                ],
            },
        }),
    ],
};
