import servicePackage from "../service/package.json" with { type: "json" };
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

const ortVersion = servicePackage.dependencies["onnxruntime-web"];
if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(ortVersion)) {
    throw new Error(`Expected onnxruntime-web version to be pinned, received ${ortVersion}`);
}
const ortDistUrl = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/`;

export default defineConfig({
    define: {
        "import.meta.env.VITE_ORT_DIST_URL": JSON.stringify(ortDistUrl),
    },
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
                        urlPattern: ({ url }) => url.pathname.includes("onnxruntime-web"),
                        handler: "CacheFirst",
                        options: { cacheName: "ort-cache" },
                    },
                ],
                maximumFileSizeToCacheInBytes: 10 * 1024 * 1024,
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
