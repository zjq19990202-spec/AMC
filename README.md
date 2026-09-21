# AMC project website

This is the **website branch** for [Atomic Motion Coordinate](https://zjq19990202-spec.github.io/AMC/).
It is intentionally separate from the code release.

| Branch | Purpose |
| --- | --- |
| [`main`](https://github.com/zjq19990202-spec/AMC/tree/main) | Minimal AMC training and spherical inference source code. |
| `gh-pages` (this branch) | Static project page, paper PDF, figures, logos and demonstration videos. |

GitHub Pages is configured to publish this branch from its repository root, so
changes pushed here update the public site at:

<https://zjq19990202-spec.github.io/AMC/>

## Local preview

```bash
python3 -m http.server 8877
```

Then open <http://127.0.0.1:8877>. This preview serves the page directly; no
build step or web framework is required.

## Main website files

- `index.html`, `styles.css`: static page and layout.
- `assets/`: paper PDF, four task videos, and institution marks.
- `images/`: steering, method, task and video-poster figures.

Do not put model checkpoints, robot datasets, training logs, or private
calibration assets on this branch.
