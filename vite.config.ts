import { defineConfig } from "vite-plus";

export default defineConfig({
    lint: {
        jsPlugins: [{ name: "vite-plus", specifier: "vite-plus/oxlint-plugin" }],
        rules: { "vite-plus/prefer-vite-plus-imports": "error" },
        options: { typeAware: true, typeCheck: true },
    },
    fmt: {
        tabWidth: 4,
        printWidth: 120,
        sortImports: true,
        sortPackageJson: false,
        sortTailwindcss: {
            stylesheet: "frontend/ui/src/app.css",
        },
        svelte: true,
        ignorePatterns: ["*.md", "*.toml", "*.yml", "*.yaml", "worker-configuration.d.ts"],
    },
});
