# Detypify (WIP)

Typst symbol classifier.

## Development

### Migration

1. Prepare training data

    1. Download `detexify.sql.gz` and `symbols.json` from [detexify-data](https://github.com/kirel/detexify-data) to `data` folder

    1. Import training data to a PostgreSQL database named `detypify`

        ```console
        $ createdb detypify
        $ gunzip -c data/detexify.sql.gz | psql detypify
        ```

1. Prepare symbol mapping

    1. Clone [mitex](https://github.com/mitex-rs/mitex) and build `mitex-spec-gen`

        ```console
        $ git clone https://github.com/mitex-rs/mitex
        $ cd mitex
        $ cargo build --package=mitex-spec-gen
        ```

    1. Copy `default.json` to `data` folder

        ```console
        $ cp target/mitex-artifacts/spec/default.json <project-root>/data
        ```

1. Prepare Typst symbol page

    1. Access https://typst.app/docs/reference/symbols/sym/ in your browser

    1. Right click -> Save as -> `<project-root>/data/typ_sym.html`

1. Run code in project root

    ```console
    $ rye sync         # prepare venv and denpendencies
    $ rye run migrate  # migrate
    ```

    Outputs will be in `migrate-out` folder.

### Training

In project root:

```console
$ rye sync       # prepare venv and denpendencies
$ rye run train  # train
```

Outputs will be in `train-out` folder.

### Web Page

In project root:

```console
$ bun install    # install dependencies
$ bun run dev    # start dev server
$ bun run build  # build for production
```

Outputs will be in `web-out` folder.

## License

MIT
