<div align="center">
    <img src="./assets/logo.svg" alt="logo" width="150"/>
    <h1>Detypify</h1>
    <p>
        Can't remember Typst symbols?
        <a href="https://detypify.quarticcat.com/">Draw it!</a>
    </p>
</div>

## Features

- **PWA**: installable and works offline
- **Tiny model**: 1.1 MiB (ONNX), fast to load and run
- **Decent symbol set**: support 350+ symbols

## Help Wanted

I'm neither an AI scientist nor a frontend engineer. This project is not what I am good at. So I hope to get some help from the community. If you have interest, please take a look at [my issues](https://github.com/QuarticCat/detypify/issues/created_by/QuarticCat).

And of course, PRs are welcome. 🥰

## Development

Before building the project, you need to prepare necessary data ([Git LFS](https://git-lfs.com/) required).

```console
$ git clone --depth=1 https://github.com/QuarticCat/detypify-data data
$ git clone --depth=1 https://github.com/QuarticCat/detypify-external external
```

### Migration

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
$ bun install    # install dependencies
$ bun run dev    # start dev server
$ bun run build  # build for production
```

### Logo & Favicons (Optional)

1. Install *NewComputerModernMath* font ([guide](https://wiki.archlinux.org/title/TeX_Live#Making_fonts_available_to_Fontconfig))

1. Convert `detypify.svg`

    ```console
    $ cd assets
    $ inkscape detypify.svg --export-text-to-path --export-filename=logo.svg
    $ bunx svgo --multipass logo.svg
    ```

1. Generate favicons by [Favicon InBrowser.App](https://favicon.inbrowser.app/tools/favicon-generator) using `logo.svg`

## License

MIT
