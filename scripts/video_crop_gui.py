#!/usr/bin/env python3
"""Small interactive video crop and rotation tool backed by FFmpeg.

Rotate the preview, optionally drag a crop rectangle, then export either the
current video or all loaded videos. Outputs are written to a ``cropped``
subdirectory and inputs are never overwritten.
"""

from __future__ import annotations

import argparse
import math
import queue
import shutil
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


CANVAS_WIDTH = 960
CANVAS_HEIGHT = 540
EVEN = 2
def even_floor(value: float) -> int:
    return max(EVEN, int(value) // EVEN * EVEN)


def even_coordinate(value: float) -> int:
    return max(0, int(value) // EVEN * EVEN)


def even_ceil(value: float) -> int:
    return max(EVEN, math.ceil(value / EVEN) * EVEN)


def normalized_rotation(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def rotation_filters(angle: float) -> list[str]:
    angle = normalized_rotation(angle)
    if abs(angle) < 0.05:
        return []
    if abs(angle - 90.0) < 0.05:
        return ["transpose=clock"]
    if abs(angle + 90.0) < 0.05:
        return ["transpose=cclock"]
    if abs(abs(angle) - 180.0) < 0.05:
        return ["hflip", "vflip"]
    angle_expression = f"{angle:.6f}*PI/180"
    return [
        f"rotate={angle_expression}:"
        f"ow=ceil(rotw({angle_expression})/2)*2:"
        f"oh=ceil(roth({angle_expression})/2)*2:c=black"
    ]


class VideoCropGUI:
    def __init__(self, root: tk.Tk, paths: list[Path], gpu: int) -> None:
        self.root = root
        self.paths = paths
        self.gpu = gpu
        self.current_path = paths[0]
        self.capture: cv2.VideoCapture | None = None
        self.source_width = 0
        self.source_height = 0
        self.frame_width = 0
        self.frame_height = 0
        self.duration = 0.0
        self.display_scale = 1.0
        self.display_x = 0
        self.display_y = 0
        self.display_width = 0
        self.display_height = 0
        self.drag_start: tuple[int, int] | None = None
        self.crop_rect: tuple[int, int, int, int] | None = None
        self.crop_normalized: tuple[float, float, float, float] | None = None
        self.preview_image: Image.Image | None = None
        self.source_preview_image: Image.Image | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.worker_events: queue.Queue[tuple[str, str]] = queue.Queue()

        root.title("视频旋转与画面裁剪工具")
        root.resizable(False, False)
        self._build_ui()
        self._open_video(self.current_path)
        self.root.after(150, self._poll_worker_events)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="视频：").pack(side=tk.LEFT)
        self.video_var = tk.StringVar(value=self.current_path.name)
        self.video_combo = ttk.Combobox(
            top,
            textvariable=self.video_var,
            values=[path.name for path in self.paths],
            state="readonly",
            width=60,
        )
        self.video_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.video_combo.bind("<<ComboboxSelected>>", self._on_video_selected)

        ttk.Label(top, text="比例：").pack(side=tk.LEFT)
        self.aspect_var = tk.StringVar(value="自由")
        ttk.Combobox(
            top,
            textvariable=self.aspect_var,
            values=["自由", "16:9", "9:16", "1:1", "4:3"],
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT)

        ttk.Label(top, text="旋转：").pack(side=tk.LEFT, padx=(12, 0))
        self.rotation_var = tk.DoubleVar(value=0.0)
        self.rotation_scale = ttk.Scale(
            top,
            from_=-180.0,
            to=180.0,
            orient=tk.HORIZONTAL,
            variable=self.rotation_var,
            length=150,
            command=self._on_rotation_dragged,
        )
        self.rotation_scale.pack(side=tk.LEFT)
        self.rotation_scale.bind("<ButtonRelease-1>", self._on_rotation_released)
        self.rotation_label = ttk.Label(top, text="0.0°", width=7)
        self.rotation_label.pack(side=tk.LEFT)
        ttk.Button(top, text="归零", command=self._reset_rotation).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            background="#151515",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(padx=8)
        self.canvas.bind("<ButtonPress-1>", self._drag_begin)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)

        timeline = ttk.Frame(self.root, padding=(8, 8, 8, 2))
        timeline.pack(fill=tk.X)
        ttk.Label(timeline, text="预览时刻：").pack(side=tk.LEFT)
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_scale = ttk.Scale(
            timeline,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.time_var,
            length=790,
        )
        self.time_scale.pack(side=tk.LEFT, padx=6)
        self.time_scale.bind("<ButtonRelease-1>", self._on_timeline_release)
        self.time_label = ttk.Label(timeline, text="00:00 / 00:00", width=17)
        self.time_label.pack(side=tk.LEFT)

        info = ttk.Frame(self.root, padding=(8, 4))
        info.pack(fill=tk.X)
        self.roi_label = ttk.Label(info, text="拖动鼠标框选要保留的区域")
        self.roi_label.pack(side=tk.LEFT)
        ttk.Label(
            info,
            text="白色半透明区域会保留，框外会被裁掉",
            foreground="#666666",
        ).pack(side=tk.RIGHT)

        buttons = ttk.Frame(self.root, padding=8)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="重置为完整画面", command=self._reset_crop).pack(
            side=tk.LEFT
        )
        self.current_button = ttk.Button(
            buttons, text="导出当前视频", command=self._crop_current
        )
        self.current_button.pack(side=tk.RIGHT)
        self.all_button = ttk.Button(
            buttons,
            text=f"批量导出全部 {len(self.paths)} 个视频",
            command=self._crop_all,
        )
        self.all_button.pack(side=tk.RIGHT, padx=8)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(8, 4),
        ).pack(fill=tk.X)

    def _open_video(self, path: Path) -> None:
        if self.capture is not None:
            self.capture.release()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            messagebox.showerror("无法打开", str(path))
            return

        self.capture = capture
        self.current_path = path
        self.source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._update_rotated_dimensions()
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration = max(0.0, frame_count / fps)
        self.time_scale.configure(to=max(0.1, self.duration))
        self.time_var.set(min(1.0, self.duration / 2.0))
        self.crop_rect = None
        self.crop_normalized = None
        self._load_preview(self.time_var.get())
        self._update_roi_label()

    def _load_preview(self, seconds: float) -> None:
        if self.capture is None:
            return
        self.capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if not ok:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.source_preview_image = Image.fromarray(rgb)
        self._render_rotated_preview(exact=True)
        self.time_label.configure(
            text=f"{self._format_time(seconds)} / {self._format_time(self.duration)}"
        )

    def _render_rotated_preview(self, exact: bool = False) -> None:
        if self.source_preview_image is None:
            return
        angle = self.rotation_var.get()
        self._update_rotated_dimensions()
        self._restore_crop_from_normalized()
        source = self.source_preview_image
        filters = rotation_filters(angle)
        if not filters:
            image = source.copy()
        elif exact:
            source_array = np.asarray(source)
            command = [
                "ffmpeg",
                "-hide_banner",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.source_width}x{self.source_height}",
                "-i",
                "pipe:0",
                "-vf",
                ",".join(filters),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "pipe:1",
            ]
            result = subprocess.run(
                command, input=source_array.tobytes(), capture_output=True
            )
            expected_size = self.frame_width * self.frame_height * 3
            if result.returncode != 0 or len(result.stdout) != expected_size:
                error = result.stderr.decode(errors="replace").strip()
                raise RuntimeError(f"FFmpeg 旋转预览失败：{error}")
            rotated = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
                self.frame_height, self.frame_width, 3
            )
            image = Image.fromarray(rotated)
        else:
            source_array = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR)
            matrix = cv2.getRotationMatrix2D(
                (self.source_width / 2.0, self.source_height / 2.0), -angle, 1.0
            )
            matrix[0, 2] += self.frame_width / 2.0 - self.source_width / 2.0
            matrix[1, 2] += self.frame_height / 2.0 - self.source_height / 2.0
            rotated = cv2.warpAffine(
                source_array,
                matrix,
                (self.frame_width, self.frame_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            image = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        self.display_scale = min(
            CANVAS_WIDTH / self.frame_width, CANVAS_HEIGHT / self.frame_height
        )
        self.display_width = round(self.frame_width * self.display_scale)
        self.display_height = round(self.frame_height * self.display_scale)
        self.display_x = (CANVAS_WIDTH - self.display_width) // 2
        self.display_y = (CANVAS_HEIGHT - self.display_height) // 2
        self.preview_image = image.resize(
            (self.display_width, self.display_height), Image.Resampling.LANCZOS
        )
        self._render_preview()

    def _render_preview(self) -> None:
        if self.preview_image is None:
            return
        self.preview_photo = ImageTk.PhotoImage(self.preview_image)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.display_x,
            self.display_y,
            anchor=tk.NW,
            image=self.preview_photo,
            tags="preview",
        )
        self._draw_crop_overlay()

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0, round(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _canvas_to_frame(self, x: int, y: int) -> tuple[int, int]:
        x = min(max(x, self.display_x), self.display_x + self.display_width)
        y = min(max(y, self.display_y), self.display_y + self.display_height)
        frame_x = round((x - self.display_x) / self.display_scale)
        frame_y = round((y - self.display_y) / self.display_scale)
        return min(frame_x, self.frame_width), min(frame_y, self.frame_height)

    def _drag_begin(self, event: tk.Event) -> None:
        self.drag_start = self._canvas_to_frame(event.x, event.y)

    def _drag_move(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        self.crop_rect = self._rect_from_points(
            self.drag_start, self._canvas_to_frame(event.x, event.y)
        )
        self._remember_crop_normalized()
        self._render_preview()
        self._update_roi_label()

    def _drag_end(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        self.crop_rect = self._rect_from_points(
            self.drag_start, self._canvas_to_frame(event.x, event.y)
        )
        self._remember_crop_normalized()
        self.drag_start = None
        self._render_preview()
        self._update_roi_label()

    def _rect_from_points(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> tuple[int, int, int, int] | None:
        x0, x1 = sorted((start[0], end[0]))
        y0, y1 = sorted((start[1], end[1]))
        width, height = x1 - x0, y1 - y0
        if width < 8 or height < 8:
            return None

        aspect = self.aspect_var.get()
        ratios = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3}
        ratio = ratios.get(aspect)
        if ratio is not None:
            if width / height > ratio:
                width = round(height * ratio)
            else:
                height = round(width / ratio)

        x = even_coordinate(x0)
        y = even_coordinate(y0)
        width = even_floor(min(width, self.frame_width - x))
        height = even_floor(min(height, self.frame_height - y))
        return x, y, width, height

    def _draw_crop_overlay(self) -> None:
        if self.crop_rect is None:
            return
        x, y, width, height = self.crop_rect
        x0 = self.display_x + round(x * self.display_scale)
        y0 = self.display_y + round(y * self.display_scale)
        x1 = self.display_x + round((x + width) * self.display_scale)
        y1 = self.display_y + round((y + height) * self.display_scale)

        shade = "#000000"
        self.canvas.create_rectangle(
            self.display_x, self.display_y, self.display_x + self.display_width, y0,
            fill=shade, stipple="gray50", outline=""
        )
        self.canvas.create_rectangle(
            self.display_x, y1, self.display_x + self.display_width,
            self.display_y + self.display_height, fill=shade, stipple="gray50", outline=""
        )
        self.canvas.create_rectangle(
            self.display_x, y0, x0, y1, fill=shade, stipple="gray50", outline=""
        )
        self.canvas.create_rectangle(
            x1, y0, self.display_x + self.display_width, y1,
            fill=shade, stipple="gray50", outline=""
        )
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00ff88", width=3)

    def _reset_crop(self) -> None:
        self.crop_rect = (0, 0, even_floor(self.frame_width), even_floor(self.frame_height))
        self._remember_crop_normalized()
        self._render_preview()
        self._update_roi_label()

    def _remember_crop_normalized(self) -> None:
        if self.crop_rect is None or self.frame_width <= 0 or self.frame_height <= 0:
            self.crop_normalized = None
            return
        x, y, width, height = self.crop_rect
        self.crop_normalized = (
            x / self.frame_width,
            y / self.frame_height,
            width / self.frame_width,
            height / self.frame_height,
        )

    def _restore_crop_from_normalized(self) -> None:
        if self.crop_normalized is None:
            self.crop_rect = None
            return
        x_ratio, y_ratio, width_ratio, height_ratio = self.crop_normalized
        x = even_coordinate(x_ratio * self.frame_width)
        y = even_coordinate(y_ratio * self.frame_height)
        width = even_floor(min(width_ratio * self.frame_width, self.frame_width - x))
        height = even_floor(min(height_ratio * self.frame_height, self.frame_height - y))
        self.crop_rect = (x, y, width, height)

    def _update_roi_label(self) -> None:
        if self.crop_rect is None:
            self.roi_label.configure(text="拖动鼠标框选要保留的区域")
            return
        x, y, width, height = self.crop_rect
        self.roi_label.configure(
            text=f"保留区域：x={x}, y={y}, 宽={width}, 高={height}"
        )

    def _on_video_selected(self, _event: tk.Event) -> None:
        selected = self.video_var.get()
        path = next(path for path in self.paths if path.name == selected)
        self._open_video(path)

    def _on_timeline_release(self, _event: tk.Event) -> None:
        self._load_preview(self.time_var.get())

    def _update_rotated_dimensions(self) -> None:
        angle = normalized_rotation(self.rotation_var.get())
        if abs(angle) < 0.05 or abs(abs(angle) - 180.0) < 0.05:
            self.frame_width = self.source_width
            self.frame_height = self.source_height
            return
        if abs(abs(angle) - 90.0) < 0.05:
            self.frame_width = self.source_height
            self.frame_height = self.source_width
            return
        radians = math.radians(angle)
        cos_value = abs(math.cos(radians))
        sin_value = abs(math.sin(radians))
        self.frame_width = even_ceil(
            self.source_width * cos_value + self.source_height * sin_value
        )
        self.frame_height = even_ceil(
            self.source_width * sin_value + self.source_height * cos_value
        )

    def _on_rotation_dragged(self, value: str) -> None:
        self.rotation_label.configure(text=f"{float(value):.1f}°")
        self._render_rotated_preview()
        self._update_roi_label()

    def _on_rotation_released(self, _event: tk.Event) -> None:
        self._render_rotated_preview(exact=True)
        self._update_roi_label()

    def _reset_rotation(self) -> None:
        self.rotation_var.set(0.0)
        self._on_rotation_dragged("0")

    def _crop_current(self) -> None:
        self._start_crop([self.current_path])

    def _crop_all(self) -> None:
        self._start_crop(self.paths)

    def _start_crop(self, paths: list[Path]) -> None:
        if self.crop_rect is None:
            self.crop_rect = (
                0,
                0,
                even_floor(self.frame_width),
                even_floor(self.frame_height),
            )
            self._remember_crop_normalized()
            self._render_preview()
            self._update_roi_label()
        self.current_button.configure(state=tk.DISABLED)
        self.all_button.configure(state=tk.DISABLED)
        self.status_var.set(f"准备导出 {len(paths)} 个视频……")
        threading.Thread(
            target=self._crop_worker,
            args=(paths, self.crop_rect, float(self.rotation_var.get())),
            daemon=True,
        ).start()

    def _crop_worker(
        self,
        paths: list[Path],
        crop_rect: tuple[int, int, int, int],
        rotation: float,
    ) -> None:
        x, y, width, height = crop_rect
        try:
            for index, path in enumerate(paths, start=1):
                self.worker_events.put(
                    ("status", f"正在导出 {index}/{len(paths)}：{path.name}")
                )
                output_dir = path.parent / "cropped"
                output_dir.mkdir(parents=True, exist_ok=True)
                output = output_dir / f"{path.stem}_cropped.mp4"
                temporary = output_dir / f".{path.stem}_cropped.tmp.mp4"
                filters = rotation_filters(rotation)
                filters.append(f"crop={width}:{height}:{x}:{y}")
                command = [
                    "ffmpeg", "-hide_banner", "-y", "-i", str(path),
                    "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", ",".join(filters),
                    "-c:v", "hevc_nvenc", "-gpu", str(self.gpu),
                    "-preset", "p6", "-rc:v", "vbr", "-cq:v", "18",
                    "-b:v", "0", "-c:a", "copy", "-movflags", "+faststart",
                    str(temporary),
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip().splitlines()[-1])
                temporary.replace(output)
            self.worker_events.put(
                ("done", f"完成：文件保存在 {paths[0].parent / 'cropped'}")
            )
        except Exception as exc:  # UI must surface FFmpeg failures.
            self.worker_events.put(("error", str(exc)))

    def _poll_worker_events(self) -> None:
        try:
            while True:
                kind, message = self.worker_events.get_nowait()
                self.status_var.set(message)
                if kind in {"done", "error"}:
                    self.current_button.configure(state=tk.NORMAL)
                    self.all_button.configure(state=tk.NORMAL)
                if kind == "done":
                    messagebox.showinfo("裁剪完成", message)
                elif kind == "error":
                    messagebox.showerror("裁剪失败", message)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_worker_events)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI video crop tool")
    parser.add_argument("videos", nargs="*", type=Path, help="videos to load")
    parser.add_argument("--gpu", type=int, default=1, help="NVENC GPU index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed")
    paths = [path.expanduser().resolve() for path in args.videos]
    if not paths:
        default_dir = Path("/media/admin123/T5 EVO/20260829_154128_clips")
        paths = sorted(default_dir.glob("*.mp4"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Video not found: {missing[0]}")
    if not paths:
        raise SystemExit("No videos were supplied")

    root = tk.Tk()
    VideoCropGUI(root, paths, args.gpu)
    root.mainloop()


if __name__ == "__main__":
    main()
