# Third-party notices

Copyright © 2026 HAKU. The original notebook, tests, documentation, and project artwork in this repository are licensed under **AGPL-3.0-only**; see [LICENSE](LICENSE).

This is an independent community project. It is not an official release of MangaJaNai, IllustrationJaNai, Google Colab, or the libraries below.

## Model weights

The notebook downloads unmodified model weights from official releases. This repository contains model names, URLs, and SHA-256 checksums; it does not include weights or upstream example images.

| Upstream project | Author | Source and terms |
| --- | --- | --- |
| MangaJaNai V1 | the-database | [Project](https://github.com/the-database/MangaJaNai), [V1 release](https://github.com/the-database/MangaJaNai/releases/tag/1.0.0) |
| IllustrationJaNai V3 | the-database | [Project](https://github.com/the-database/MangaJaNai), [V3 release](https://github.com/the-database/MangaJaNai/releases/tag/3.0.0) |

The upstream MangaJaNai repository uses [Creative Commons Attribution-NonCommercial 4.0 International](https://github.com/the-database/MangaJaNai/blob/main/LICENSE). Model use remains subject to those terms. This project's source-code license does not replace the weights' license or grant rights to third-party manga and illustrations.

## Runtime dependencies

Dependencies are installed from their upstream distributions at runtime; their source code and binaries are not bundled in this repository.

| Dependency | Upstream license | Source |
| --- | --- | --- |
| Spandrel | MIT | [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel) |
| PyMuPDF / MuPDF | GNU AGPL v3; upstream also offers commercial licensing | [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [COPYING](https://github.com/pymupdf/PyMuPDF/blob/main/COPYING) |
| PyTorch / Torchvision | BSD-style | [PyTorch](https://github.com/pytorch/pytorch), [Torchvision](https://github.com/pytorch/vision) |
| Pillow | HPND | [python-pillow/Pillow](https://github.com/python-pillow/Pillow) |
| NumPy | BSD-3-Clause | [numpy/numpy](https://github.com/numpy/numpy) |
| tqdm | MPL-2.0 and MIT portions | [tqdm/tqdm](https://github.com/tqdm/tqdm) |
| nbformat (development only) | BSD-3-Clause | [jupyter/nbformat](https://github.com/jupyter/nbformat) |

Refer to each upstream distribution for its complete license and notices, including transitive dependencies. Model architectures are provided by Spandrel rather than copied into this project.

## Project artwork

The banner is original SVG artwork created for this repository. It is a symbolic illustration, not an example of model output. Tests use synthetic images and PDF fixtures generated in temporary folders; no third-party manga pages are distributed.
