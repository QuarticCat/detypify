# Detypify Service

Let you integrate Detypify into your own projects easily.

## Getting Started

1. Install it as a dependency.

    ```console
    $ bun add detypify-service
    ```

1. This javascript module exports:

    - `ortEnv`: Re-export of [`onnxruntime-web.env`](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html). Used to configure onnxruntime.

      Note: Recent `onnxruntime-web` builds ship bundled wasm by default; use the `onnxruntime-web-use-extern-wasm` export condition to opt into external wasm loading (see the official `exports` in https://github.com/microsoft/onnxruntime/blob/main/js/web/package.json).

    - `Detypify`: The main type.

      - Use `Detypify.create()` to create an instance.

      - Use `instance.candidates(strokes, k)` to inference top `k` candidates.

    - `inferSyms`: Symbol metadata used by the model.

    - `contribSyms`: Mapping from Typst symbol names to characters.
