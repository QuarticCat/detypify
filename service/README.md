# Detypify Service

Let you integrate Detypify into your own projects easily.

## Getting Started

1. Install it as a dependency.

    ```console
    $ bun add detypify-service
    ```

1. This javascript module has only 2 exports:

  - `ortEnv`: Re-export of [`onnxruntime-web.env`](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html). Used to configureonnxruntime.

    Note: ONNX Runtime Web will request wasm files on the fly. That often causes problems. You might need to configure `ortEnv.wasm.wasmPaths` to a right value.

  - `Detypify`: The main object.

    - Use `Detypify.load()` to create a session.

    - Use `session.candidates(strokes, k)` to inference top K candidates.


[Vite](https://vitejs.dev/) is recommended. I'm not sure whether other build tools can resolve bundled assets or not.
