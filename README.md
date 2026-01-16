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
> If you are using Brave browser, please turn off Shields for this site, or it won't function properly.

## News

- 2024-04-06: This project has been integrated into [Tinymist](https://github.com/Myriad-Dreamin/tinymist).

## Associated Repos

- [detypify-data](https://github.com/QuarticCat/detypify-data): Detypify's own dataset (your contributions on the website finally go here)
- [detypify-external](https://github.com/QuarticCat/detypify-external): Necessary external data to bootstrap Detypify

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

The logo is a hand-written SVG file in [assets/manuscript.svg](./assets/manuscript.svg).

It requires *NewComputerModernMath* font ([install guide](https://wiki.archlinux.org/title/TeX_Live#Making_fonts_available_to_Fontconfig)).

To strip the font requirement and optimize for production:

```console
$ cd assets
$ inkscape manuscript.svg --export-text-to-path --export-filename=logo.svg
$ bunx svgo --multipass logo.svg
```

## License

MIT
