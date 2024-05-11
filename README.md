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

## News

- 2024-04-06: This project has been integrated into Tinymist. [🔗](https://github.com/Myriad-Dreamin/tinymist/releases/tag/v0.11.3)

## Associated Repos

- [detypify-data](https://github.com/QuarticCat/detypify-data): Detypify's own dataset (your contributions on the website finally go here)
- [detypify-external](https://github.com/QuarticCat/detypify-external): Necessary external data to bootstrap Detypify

## Integrate Into Your Project

See [service/README.md](service/README.md).

## Self Deployment

Download files from [gh-pages](https://github.com/QuarticCat/detypify/tree/gh-pages) branch and host them using any HTTP server.

## Development

If you want to build `migrate` or `train`, you need to pull submodules. ([Git LFS](https://git-lfs.com/) is required)

```console
$ git submodule update --init --recursive
```

If you just want to build `web`, you can download `train-out` from [NPM](https://www.npmjs.com/package/detypify-service?activeTab=code) to project root.

### Migrating

```console
$ rye sync              # install venv and denpendencies
$ rye run migrate       # migrate
$ rye run migrate-font  # strip font (optional)
```

### Training

```console
$ rye sync       # install venv and denpendencies
$ rye run train  # train
```

### Web Page

```console
$ bun run --cwd=service copy  # copy train-out folder
$ bun install                 # install dependencies
$ bun run dev                 # start dev server
$ bun run build               # build for production
```

### Logo & Favicons (Optional)

1. Install *NewComputerModernMath* font ([guide](https://wiki.archlinux.org/title/TeX_Live#Making_fonts_available_to_Fontconfig))

1. Convert `manuscript.svg`

    ```console
    $ cd assets
    $ inkscape manuscript.svg --export-text-to-path --export-filename=logo.svg
    $ bunx svgo --multipass logo.svg
    ```

1. Generate favicons by [Favicon InBrowser.App](https://favicon.inbrowser.app/tools/favicon-generator) using `logo.svg`

## License

MIT
