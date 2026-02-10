# Detypify Service

Integrate Detypify into your own projects.

## Example

```typescript
import { Detypify, inferSyms } from "detypify-service";

const session = await Detypify.create();
const storkes = [[[0, 0], [1, 1]]];
const scores = await session.infer(strokes);
const candidates = Array.from(scores.keys());
candidates.sort((a, b) => scores[b] - scores[a]);
console.log(inferSyms[candidates[0]]);
```

## API Reference

- `ortEnv`: Re-export of [`onnxruntime-web.env`](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html). Used to configure onnxruntime.

  Note: Recent `onnxruntime-web` builds ship bundled wasm by default; use the `onnxruntime-web-use-extern-wasm` export condition to opt into external wasm loading (see the official `exports` in https://github.com/microsoft/onnxruntime/blob/main/js/web/package.json).

- `inferSyms`: Model's output symbol data.

- `contribSyms`: Mapping from Typst symbol names to characters.

- `Detypify`: The main type.

  - Use `Detypify.create()` to create an instance.

  - Use `instance.infer(strokes)` to inference scores of each symbol.

    The higher `scores[i]` is, the more likely your strokes is `inferSyms[i]`.
