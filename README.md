# Atomic Motion Coordinate

Project page and supplementary demo for **Atomic Motion Coordinate for
Language-Steerable and Force-Responsive Manipulation**.

The site is a dependency-free static page. It includes the paper PDF, a
four-task robot demonstration, method illustrations, and task snapshots.

## Local preview

```bash
cd /home/admin123/下载/atomic-motion-coordinate
python3 -m http.server 8877
```

Open `http://127.0.0.1:8877`.

## Publishing with GitHub Pages

Push the `main` branch, then in the repository's **Settings → Pages**, select
**Deploy from a branch** and choose `main` / `(root)`. The public URL will be
`https://Robbyant.github.io/atomic-motion-coordinate/`.

## Repository contents

- `index.html`, `styles.css`: static project page
- `assets/amc-four-task-demo.mp4`: H.264 demo video
- `assets/amc-paper.pdf`: paper PDF
- `images/`: method and task figures
