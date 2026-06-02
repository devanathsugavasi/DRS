"""Tkinter umpire dashboard for live review and buffered replay."""

from __future__ import annotations

from math import ceil, sqrt
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from core.integration import DRSPipeline, PipelineState


class UmpireDashboard(tk.Tk):
    """Touch-friendly dashboard with camera wall, review layout, and replay controls."""

    def __init__(self, camera_ids: list[int] | None = None, record: bool = False):
        super().__init__()
        self.title("Cricket DRS Umpire Dashboard")
        self.geometry("1440x900")
        self.minsize(1180, 720)
        self.configure(bg="#0b0f16")

        self.pipeline = DRSPipeline(camera_ids=camera_ids, record=record)
        self.camera_ids = sorted(self.pipeline.camera_manager.camera_ids)
        self.replay = None
        self.playback_speed = 1.0
        self.paused = False

        default_main = self.camera_ids[0] if self.camera_ids else 0
        default_small = self.camera_ids[1] if len(self.camera_ids) > 1 else default_main
        self.layout_var = tk.StringVar(value="Wall")
        self.main_camera_var = tk.StringVar(value=f"CAM {default_main}")
        self.small_camera_var = tk.StringVar(value=f"CAM {default_small}")
        self.status_var = tk.StringVar(value="Ready")
        self.decision_var = tk.StringVar(value="REVIEW INCONCLUSIVE")

        self.image_refs: dict[int, ImageTk.PhotoImage] = {}
        self.video_labels: dict[int, tk.Label] = {}
        self._build_ui()

    def start(self) -> None:
        self.pipeline.start()
        self.status_var.set("Live")
        self.after(20, self._update_live)
        self.mainloop()

    def destroy(self) -> None:
        self.pipeline.stop()
        super().destroy()

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#0A0E1A", foreground="#E8ECF0", font=("Rajdhani", 11))
        style.configure("TFrame", background="#0A0E1A")
        style.configure("TLabel", background="#0A0E1A", foreground="#E8ECF0", font=("Rajdhani", 11))
        style.configure("TButton", background="#1B2333", foreground="#FFD700", borderwidth=0, focuscolor="#FFD700", padding=(12, 6), font=("Rajdhani", 11, "bold"))
        style.map("TButton", background=[("active", "#2A3A5C")])
        style.configure("Decision.TLabel", background="#0A0E1A", foreground="#FFD700", font=("Rajdhani", 28, "bold"))
        style.configure("Status.TLabel", background="#101827", foreground="#00E5FF", font=("Rajdhani", 11, "bold"))

        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        top.columnconfigure(99, weight=1)

        ttk.Label(top, text="Layout").grid(row=0, column=0, padx=(0, 4))
        self.layout_selector = ttk.Combobox(
            top,
            textvariable=self.layout_var,
            values=["Wall", "Review", "Focus"],
            state="readonly",
            width=10,
        )
        self.layout_selector.grid(row=0, column=1, padx=4)
        self.layout_selector.bind("<<ComboboxSelected>>", self._layout_changed)

        ttk.Label(top, text="Big").grid(row=0, column=2, padx=(14, 4))
        self.main_selector = ttk.Combobox(
            top,
            textvariable=self.main_camera_var,
            values=[f"CAM {camera_id}" for camera_id in self.camera_ids],
            state="readonly",
            width=10,
        )
        self.main_selector.grid(row=0, column=3, padx=4)
        self.main_selector.bind("<<ComboboxSelected>>", self._layout_changed)

        ttk.Label(top, text="Small").grid(row=0, column=4, padx=(14, 4))
        self.small_selector = ttk.Combobox(
            top,
            textvariable=self.small_camera_var,
            values=[f"CAM {camera_id}" for camera_id in self.camera_ids],
            state="readonly",
            width=10,
        )
        self.small_selector.grid(row=0, column=5, padx=4)
        self.small_selector.bind("<<ComboboxSelected>>", self._layout_changed)

        ttk.Button(top, text="Wall", command=lambda: self._set_layout("Wall")).grid(row=0, column=6, padx=(16, 4))
        ttk.Button(top, text="Review", command=lambda: self._set_layout("Review")).grid(row=0, column=7, padx=4)
        ttk.Button(top, text="Focus", command=lambda: self._set_layout("Focus")).grid(row=0, column=8, padx=4)
        ttk.Label(top, textvariable=self.decision_var, style="Decision.TLabel").grid(row=0, column=9, padx=(18, 8))
        ttk.Label(top, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=99, sticky="e")

        self.video_grid = ttk.Frame(self)
        self.video_grid.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        self._build_camera_tiles()

        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 10))
        ttk.Button(controls, text="Live", command=self._go_live).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Pause", command=self._pause).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Play", command=self._play).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="0.25x", command=self._slow).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Frame -1", command=lambda: self._step(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Frame +1", command=lambda: self._step(1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Instant Replay", command=self._instant_replay).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Save Replay", command=self._save_replay).pack(side=tk.LEFT, padx=4)
        ttk.Label(controls, text="Appeals: LBW | Edge | No Ball | Run Out", style="Status.TLabel").pack(side=tk.RIGHT, padx=4)

    def _update_live(self) -> None:
        if self.replay is None and not self.paused:
            state = self.pipeline.process_once()
            self._render_state(state)
        self.after(20, self._update_live)

    def _update_replay(self) -> None:
        if self.replay is None:
            return
        frames = self.replay.current_frames()
        self._render_raw({camera_id: item.frame for camera_id, item in frames.items()})
        if self.replay.playing:
            self.replay.tick()
            self.after(self.replay.frame_delay_ms(), self._update_replay)

    def _render_state(self, state: PipelineState) -> None:
        if state.sync_report:
            dropped = sum(state.sync_report.dropped_frames.values())
            suffix = f" | dropped {dropped}" if dropped else ""
            self.status_var.set(f"Live | sync spread {state.sync_report.spread_ms:.1f} ms{suffix}")
            decision = getattr(state, "decision", None)
            if decision:
                self.decision_var.set(getattr(decision, "decision", "REVIEW INCONCLUSIVE"))
        self._render_raw({camera_id: output.annotated for camera_id, output in state.frames.items()})

    def _render_raw(self, frames: dict[int, object]) -> None:
        health = self.pipeline.camera_manager.health()
        for camera_id in self._visible_camera_ids():
            label = self.video_labels.get(camera_id)
            if label is None:
                continue
            frame = frames.get(camera_id)
            if frame is None:
                self._show_waiting_tile(camera_id, health.get(camera_id, {}))
                continue
            frame = self._draw_camera_health(frame.copy(), health.get(camera_id, {}))
            tile_w, tile_h = self._label_size(label)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = self._resize_to_tile(rgb, tile_w, tile_h)
            image = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.image_refs[camera_id] = image
            label.configure(image=image, text="")

    def _go_live(self) -> None:
        self.replay = None
        self.paused = False
        self.status_var.set("Live")

    def _pause(self) -> None:
        if self.replay:
            self.replay.pause()
        self.paused = True
        self.status_var.set("Paused")

    def _play(self) -> None:
        self.paused = False
        if self.replay:
            self.replay.play(self.playback_speed)
            self._update_replay()

    def _slow(self) -> None:
        self.playback_speed = 0.25
        if self.replay:
            self.replay.play(self.playback_speed)
            self.status_var.set("Replay 0.25x")
            self._update_replay()

    def _step(self, delta: int) -> None:
        if self.replay is None:
            self._instant_replay()
        if self.replay:
            self.replay.pause()
            self.replay.step(delta)
            self._update_replay()
            self.status_var.set(f"Replay frame {self.replay.cursor + 1}/{self.replay.total_frames}")

    def _instant_replay(self) -> None:
        self.replay = self.pipeline.camera_manager.create_replay()
        self.replay.seek(max(0, self.replay.total_frames - 180))
        self.replay.pause()
        self.layout_var.set("Review")
        self._layout_camera_tiles()
        self.status_var.set("Instant replay")
        self._update_replay()

    def _save_replay(self) -> None:
        path = self.pipeline.camera_manager.save_replay()
        self.status_var.set(f"Replay saved: {path}")

    def _build_camera_tiles(self) -> None:
        for camera_id in self.camera_ids:
            label = tk.Label(
                self.video_grid,
                text=f"CAM {camera_id}\nwaiting for feed",
                bg="#0b0f16",
                fg="#FFD700",
                font=("Segoe UI", 15, "bold"),
                bd=2,
                relief=tk.SOLID,
                compound=tk.CENTER,
            )
            label.bind("<Double-Button-1>", lambda _event, cid=camera_id: self._make_main_camera(cid))
            self.video_labels[camera_id] = label
        self._layout_camera_tiles()

    def _layout_camera_tiles(self) -> None:
        for label in self.video_labels.values():
            label.grid_forget()

        for row in range(8):
            self.video_grid.rowconfigure(row, weight=0, uniform="")
        for col in range(8):
            self.video_grid.columnconfigure(col, weight=0, uniform="")

        mode = self.layout_var.get()
        if mode == "Focus":
            self._layout_focus()
        elif mode == "Review":
            self._layout_review()
        else:
            self._layout_wall()

    def _layout_wall(self) -> None:
        visible = self.camera_ids
        rows, cols = self._grid_shape(len(visible))
        for row in range(rows):
            self.video_grid.rowconfigure(row, weight=1, uniform="wall_rows")
        for col in range(cols):
            self.video_grid.columnconfigure(col, weight=1, uniform="wall_cols")
        for index, camera_id in enumerate(visible):
            row, col = divmod(index, cols)
            self.video_labels[camera_id].grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

    def _layout_focus(self) -> None:
        camera_id = self._camera_from_var(self.main_camera_var)
        self.video_grid.rowconfigure(0, weight=1)
        self.video_grid.columnconfigure(0, weight=1)
        self.video_labels[camera_id].grid(row=0, column=0, padx=6, pady=6, sticky="nsew")

    def _layout_review(self) -> None:
        main_camera = self._camera_from_var(self.main_camera_var)
        thumbnails = [camera_id for camera_id in self._review_thumbnail_order() if camera_id != main_camera]
        thumb_cols = max(1, len(thumbnails))

        self.video_grid.rowconfigure(0, weight=4, uniform="review_rows")
        self.video_grid.rowconfigure(1, weight=1, uniform="review_rows")
        for col in range(thumb_cols):
            self.video_grid.columnconfigure(col, weight=1, uniform="review_cols")

        self.video_labels[main_camera].grid(
            row=0,
            column=0,
            columnspan=thumb_cols,
            padx=6,
            pady=6,
            sticky="nsew",
        )
        for index, camera_id in enumerate(thumbnails):
            self.video_labels[camera_id].grid(row=1, column=index, padx=6, pady=6, sticky="nsew")

    def _visible_camera_ids(self) -> list[int]:
        mode = self.layout_var.get()
        if mode == "Focus":
            return [self._camera_from_var(self.main_camera_var)]
        if mode == "Review":
            return [self._camera_from_var(self.main_camera_var)] + [
                camera_id for camera_id in self._review_thumbnail_order() if camera_id != self._camera_from_var(self.main_camera_var)
            ]
        return self.camera_ids

    def _review_thumbnail_order(self) -> list[int]:
        small = self._camera_from_var(self.small_camera_var)
        return [small] + [camera_id for camera_id in self.camera_ids if camera_id != small]

    def _grid_shape(self, count: int) -> tuple[int, int]:
        if count <= 1:
            return 1, 1
        if count <= 2:
            return 1, 2
        if count <= 4:
            return 2, 2
        cols = min(3, ceil(sqrt(count)))
        return ceil(count / cols), cols

    def _label_size(self, label: tk.Label) -> tuple[int, int]:
        width = max(240, label.winfo_width() - 4)
        height = max(160, label.winfo_height() - 4)
        return width, height

    def _resize_to_tile(self, frame, tile_w: int, tile_h: int):
        h, w = frame.shape[:2]
        scale = min(tile_w / max(1, w), tile_h / max(1, h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return cv2.copyMakeBorder(
            resized,
            (tile_h - new_h) // 2,
            tile_h - new_h - ((tile_h - new_h) // 2),
            (tile_w - new_w) // 2,
            tile_w - new_w - ((tile_w - new_w) // 2),
            cv2.BORDER_CONSTANT,
            value=(10, 14, 20),
        )

    def _layout_changed(self, _event=None) -> None:
        self._layout_camera_tiles()
        self.status_var.set(f"{self.layout_var.get()} | big {self.main_camera_var.get()} | small {self.small_camera_var.get()}")

    def _set_layout(self, layout: str) -> None:
        self.layout_var.set(layout)
        self._layout_changed()

    def _make_main_camera(self, camera_id: int) -> None:
        self.main_camera_var.set(f"CAM {camera_id}")
        self.layout_var.set("Review")
        self._layout_changed()

    def _camera_from_var(self, var: tk.StringVar) -> int:
        try:
            return int(var.get().replace("CAM ", ""))
        except ValueError:
            return self.camera_ids[0]

    def _show_waiting_tile(self, camera_id: int, health: dict[str, float]) -> None:
        fps = health.get("fps", 0.0)
        synthetic = bool(health.get("synthetic", 0.0))
        source = "synthetic fallback" if synthetic else "waiting for feed"
        label = self.video_labels.get(camera_id)
        if label is not None:
            label.configure(
                image="",
                text=f"CAM {camera_id}\n{source}\nfps {fps:.1f}",
                bg="#0b0f16",
                fg="#f4f5f7",
            )

    def _draw_camera_health(self, frame, health: dict[str, float]):
        fps = health.get("fps", 0.0)
        source = "SYN" if health.get("synthetic", 0.0) else "LIVE"
        text = f"{source} | {fps:.1f} fps"
        cv2.rectangle(frame, (8, frame.shape[0] - 30), (230, frame.shape[0] - 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (14, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 240, 220), 1, cv2.LINE_AA)
        return frame


def run_dashboard(camera_ids: list[int] | None = None, record: bool = False) -> None:
    UmpireDashboard(camera_ids=camera_ids, record=record).start()
