import { contribSyms } from "detypify-service";
import { execSync } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const FONT_URL = "https://mirrors.ctan.org/fonts/newcomputermodern/otf/NewCMMath-Regular.otf";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const cacheDir = path.join(root, "node_modules", ".cache", "detypify-fonts");
const fontPath = path.join(cacheDir, "NewCMMath-Regular.otf");
const textPath = path.join(cacheDir, "NewCMMath-Detypify.txt");
const outputDir = path.join(root, "public");
const outputPath = path.join(outputDir, "NewCMMath-Detypify.woff2");

// Download font or read from cache.
await fs.access(fontPath).catch(async () => {
    await fs.mkdir(cacheDir, { recursive: true });
    const res = await fetch(FONT_URL);
    if (!res.ok) throw new Error(`Failed to download font: ${res.status} ${res.statusText}`);
    await fs.writeFile(fontPath, Buffer.from(await res.arrayBuffer()));
});

// Generate font subset.
await fs.writeFile(textPath, Object.values(contribSyms).join(""));
execSync(
    `uvx --from=fonttools[woff] pyftsubset ${fontPath} --text-file=${textPath} --flavor=woff2 --output-file=${outputPath}`,
    { stdio: "inherit" },
);
