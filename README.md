<div align="center">
    <img src="./assets/logo.svg" alt="logo" width="150"/>
    <h1>Detypify</h1>
    <p>
        Can't remember some Typst symbol?
        <a href="https://detypify.quarticcat.com/">Draw it!</a>
    </p>
</div>

## Features

- **PWA**: installable and works offline
- **Tiny model**: 1.3 MiB (ONNX), fast to load and run
- **Decent symbol set**: support 400+ symbols

> [!WARNING]
> For some unknown reason, the webpage doesn't function properly on Brave.

## News

- 2024-04-06: This project has been integrated into [Tinymist](https://github.com/Myriad-Dreamin/tinymist).

## Associated Repos

- [detypify-data](https://github.com/QuarticCat/detypify-data): Detypify's own dataset (your contributions on the website finally go here)
- [detypify-external](https://github.com/QuarticCat/detypify-external): Necessary external data to bootstrap Detypify

## Use As A Library

Use the [detypify-service](https://www.npmjs.com/package/detypify-service) NPM package.

## Self Deployment

Download files from [gh-pages](https://github.com/QuarticCat/detypify/tree/gh-pages) branch and host them using any HTTP server.

## Development

If you want to train the model, you need to pull submodules. ([Git LFS](https://git-lfs.com/) is required)

```console
$ git submodule update --init --recursive
```

Otherwise, if you just want to develop the webpage, you can download `train` folder from [NPM](https://www.npmjs.com/package/detypify-service?activeTab=code) to `./build/train`.

### Preprocessing

```console
$ uv sync                     # install venv and dependencies
$ uv run python/proc_data.py  # preprocess data
$ uv run python/proc_font.py  # preprocess font
```

### Training

```console
$ uv sync                 # install venv and dependencies
$ uv run python/train.py  # train
```

### Web Page

```console
$ bun run --cwd=service copy  # copy train folder
$ bun install                 # install dependencies
$ bun run dev                 # start dev server
$ bun run build               # build for production
```

### Cloudflare Worker

It collects contributions from the website.

```console
$ bun run worker:dev     # start dev server
$ bun run worker:deploy  # deploy to cloudflare
```

### Logo & Favicons

1. Install *NewComputerModernMath* font ([guide](https://wiki.archlinux.org/title/TeX_Live#Making_fonts_available_to_Fontconfig)).

1. Convert `manuscript.svg`.

    ```console
    $ cd assets
    $ inkscape manuscript.svg --export-text-to-path --export-filename=logo.svg
    $ bunx svgo --multipass logo.svg
    ```

1. Generate favicons by [Favicon InBrowser.App](https://favicon.inbrowser.app/tools/favicon-generator) using `logo.svg`.

1. Move them to [web/public/icons](web/public/icons).

## License

MIT
