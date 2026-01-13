# Detypify Worker

A Cloudflare Worker that accepts contributions from the web page and stores them in a D1 database.

## Development

```console
$ bun run dev         # start local dev server
$ bun run deploy      # deploy to Cloudflare
$ bun run cf-typegen  # generate Cloudflare bindings/types (worker-configuration.d.ts)
```
