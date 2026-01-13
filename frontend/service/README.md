# Detypify Service

Let you integrate Detypify into your own projects easily.

## Getting Started

1. Install it as a dependency.

    ```console
    $ bun add detypify-service
    ```

1. This javascript module has only 2 exports:

    - `ortEnv`: Re-export of [`onnxruntime-web.env`](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html). Used to configure onnxruntime.

      Note: ONNX Runtime Web will request wasm files on the fly. Default URLs are unlikely to match yours. You might need to configure `ortEnv.wasm.wasmPaths`.

    - `Detypify`: The main type.

      - Use `Detypify.create()` to create an instance.

      - Use `instance.candidates(strokes, k)` to inference top `k` candidates.


[Vite](https://vitejs.dev/) is recommended. I'm not sure whether other build tools can resolve bundled assets or not.
