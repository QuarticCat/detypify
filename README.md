<div align="center">
    <img src="./assets/logo.svg" alt="logo" width="150"/>
    <h1>Detypify</h1>
    <p>
        Can't find some Typst symbol?
        <a href="https://detypify.quarticcat.com/">Draw it!</a>
    </p>
</div>

## Features

- **PWA**: installable and works offline
- **Tiny model**: fast to load and run
- **Decent symbol set**: support 400+ symbols

You can also use it in [Tinymist](https://github.com/Myriad-Dreamin/tinymist).

## Development

### File Structure

```text
- python     # training scripts
- frontend
  - service  # inference lib
  - ui       # web UI
  - worker   # Cloudflare worker
```

Check corresponding folders for more information.

Before you build frontend projects, make sure you have the `train` folder in [frontend/service](./frontend/service) by either:

- Train your own one, or
- Download from [NPM](https://www.npmjs.com/package/detypify-service?activeTab=code).

### Logo

Source: [assets/manuscript.svg](./assets/manuscript.svg) (requires [*NewComputerModernMath*](https://ctan.org/pkg/newcomputermodern) font)

To compile it for production:

```console
$ cd assets
$ inkscape manuscript.svg --export-text-to-path --export-filename=logo.svg  # convert text to path
$ bunx svgo --multipass logo.svg  # optimize SVG
```

## License

MIT
