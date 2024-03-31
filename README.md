<div align="center">
    <img src="./assets/logo.svg" alt="logo" width="150"/>
    <h1>Detypify (WIP)</h1>
    <p>
        Can't remember Typst symbols?
        <a href="https://detypify.quarticcat.com/">Draw it!</a>
    </p>
</div>

## Features

- **Static website**: works offline.
- **Tiny model**: 1.1 MiB (ONNX), fast to load and run.
- **Decent symbol set**: support 350+ common symbols. ([list](./supported-symbols.txt))

## Development

### Preparation

#### Training Data

1. Download `detexify.sql.gz` and `symbols.json` from [detexify-data](https://github.com/kirel/detexify-data) to `data` folder

1. Import training data to a PostgreSQL database named `detypify`

    ```console
    $ createdb detypify
    $ gunzip -c data/detexify.sql.gz | psql detypify
    ```

#### Symbol Mapping

1. Clone [mitex](https://github.com/mitex-rs/mitex) and build `mitex-spec-gen`

    ```console
    $ git clone https://github.com/mitex-rs/mitex
    $ cd mitex
    $ cargo build --package=mitex-spec-gen
    ```

1. Move `target/mitex-artifacts/spec/default.json` to `data` folder

#### Typst Symbol Page

1. Access https://typst.app/docs/reference/symbols/sym/ in your browser

1. Right click -> Save as -> `data/typ_sym.html`

#### Symbol Font

1. Download *NewComputerModern* from [CTAN](https://ctan.org/pkg/newcomputermodern?lang=en)

1. Extract and move `otf/NewCMMath-Regular.otf` to `data` folder

### Migration

```console
$ rye sync         # prepare venv and denpendencies
$ rye run migrate  # migrate
```

### Training

```console
$ rye sync       # prepare venv and denpendencies
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

1. Generate favicons by https://favicon-generator.s2n.tech/ using `logo.svg`

## License

MIT
