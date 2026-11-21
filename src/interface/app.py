"""ACS - GUI for candy detection and classification.

Run: python -m src.interface.app
Optional: python -m src.interface.app --source photo.jpg --device cpu
Uses the existing src.detection.integrated_pipeline (stage 2.5.1).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Event, Thread
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from src.interface.service import (
    DEFAULT_COLOR, DEFAULT_DETECTOR, DEFAULT_OUTPUT, DEFAULT_TYPE,
    ModelCache, Settings, analyze_batch, collect_images, validate_settings,
)

IMAGE_FILTER = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    ("All files", "*.*"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACS - Demo GUI")
    parser.add_argument("--source", type=Path, help="One image or an image folder")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--type-checkpoint", type=Path, default=DEFAULT_TYPE)
    parser.add_argument("--color-model", type=Path, default=DEFAULT_COLOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "interface")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--check-only", action="store_true", help="Check model paths without opening a window")
    return parser.parse_args(argv)


class ACSApplication:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        root.title("ACS - Automatic Candy Selector")
        root.geometry("1490x880")
        root.minsize(1100, 680)

        self.files: list[Path] = []
        self.reports: dict[Path, dict[str, Any]] = {}
        self.messages: Queue[tuple[str, dict[str, Any]]] = Queue()
        self.cancel_event = Event()
        self.cache = ModelCache()
        self.busy = False
        self.close_after_work = False
        self.displayed: Path | None = None
        self.original_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.fit_mode = True
        self.zoom = 1.0
        self.resize_task: str | None = None

        self.detector_var = tk.StringVar(value=str(args.detector))
        self.type_var = tk.StringVar(value=str(args.type_checkpoint))
        self.color_var = tk.StringVar(value=str(args.color_model))
        self.output_var = tk.StringVar(value=str(args.output))
        self.device_var = tk.StringVar(value=args.device)
        self.conf_var = tk.StringVar(value=str(args.conf))
        self.save_crops_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Choose one or more images to begin.")
        self.current_title = tk.StringVar(value="No image selected")
        self.count_var = tk.StringVar(value="0 images")

        self._build_ui()
        if args.source:
            self._add_sources([args.source])
        root.after(120, self._poll_messages)
        root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Header.TLabel", font=("DejaVu Sans", 16, "bold"))
        style.configure("Section.TLabel", font=("DejaVu Sans", 10, "bold"))
        style.configure("Action.TButton", font=("DejaVu Sans", 10, "bold"))

        header = ttk.Frame(self.root, padding=(16, 12, 16, 8))
        header.pack(fill="x")
        ttk.Label(header, text="ACS | Automatic Candy Selector", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header, text="YOLOv8n (detection) -> MobileNetV3 (type) -> HSV/KNN (color)").pack(anchor="w", pady=(3, 0))

        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        left = ttk.Frame(panes, padding=(4, 3, 7, 3))
        right = ttk.Frame(panes, padding=(5, 3, 5, 3))
        panes.add(left, weight=1)
        panes.add(right, weight=3)

        source_box = ttk.LabelFrame(left, text="Images", padding=8)
        source_box.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(source_box)
        row.pack(fill="x")
        self.add_btn = ttk.Button(row, text="+ Images", command=self._choose_images)
        self.add_btn.pack(side="left")
        self.folder_btn = ttk.Button(row, text="+ Folder", command=self._choose_folder)
        self.folder_btn.pack(side="left", padx=4)
        self.remove_btn = ttk.Button(row, text="Remove", command=self._remove_selected)
        self.remove_btn.pack(side="left", padx=(0, 4))
        self.clear_btn = ttk.Button(row, text="Clear", command=self._clear_files)
        self.clear_btn.pack(side="left")
        ttk.Label(source_box, textvariable=self.count_var).pack(anchor="w", pady=(5, 2))
        list_frame = ttk.Frame(source_box)
        list_frame.pack(fill="x")
        self.listbox = tk.Listbox(list_frame, height=5, selectmode="extended", exportselection=False,
                                 activestyle="dotbox", font=("DejaVu Sans", 9))
        source_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=source_scroll.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        source_scroll.pack(side="right", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self._select_image)

        options_box = ttk.LabelFrame(left, text="Inference settings", padding=8)
        options_box.pack(fill="x", pady=(0, 8))
        row2 = ttk.Frame(options_box)
        row2.pack(fill="x")
        ttk.Label(row2, text="YOLO confidence:").pack(side="left")
        self.conf_entry = ttk.Entry(row2, textvariable=self.conf_var, width=7)
        self.conf_entry.pack(side="left", padx=(6, 14))
        ttk.Label(row2, text="Device:").pack(side="left")
        self.device_combo = ttk.Combobox(row2, textvariable=self.device_var,
                                         values=("cpu", "auto", "cuda"), state="readonly", width=6)
        self.device_combo.pack(side="left", padx=5)
        ttk.Checkbutton(options_box, text="Save image crops", variable=self.save_crops_var).pack(anchor="w", pady=(7, 2))
        ttk.Label(options_box, text="The classifier recognizes only four trained candy types.", foreground="#80691e", wraplength=360).pack(anchor="w")

        path_box = ttk.LabelFrame(left, text="Models and output folder", padding=8)
        path_box.pack(fill="x", pady=(0, 8))
        for title, variable, ext, is_dir in (
            ("YOLO detector (.pt)", self.detector_var, "*.pt", False),
            ("MobileNet classifier (.pth)", self.type_var, "*.pth", False),
            ("Color model (.joblib)", self.color_var, "*.joblib", False),
            ("Results", self.output_var, "", True),
        ):
            ttk.Label(path_box, text=title).pack(anchor="w")
            line = ttk.Frame(path_box)
            line.pack(fill="x", pady=(1, 4))
            ttk.Entry(line, textvariable=variable).pack(side="left", fill="x", expand=True)
            ttk.Button(line, text="Browse", width=8,
                       command=lambda var=variable, e=ext, directory=is_dir:
                       self._browse_path(var, e, directory)).pack(side="right", padx=(4, 0))

        run_row = ttk.Frame(left)
        run_row.pack(fill="x", pady=(0, 3))
        self.start_btn = ttk.Button(run_row, text="Analyze", style="Action.TButton", command=self._start)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.cancel_btn = ttk.Button(run_row, text="Stop", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=(7, 0))
        self.progress = ttk.Progressbar(left, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(3, 5))

        results_frame = ttk.LabelFrame(left, text="Candies in selected image", padding=7)
        results_frame.pack(fill="both", expand=True, pady=(0, 2))
        columns = ("id", "variante", "d", "t", "etat")
        self.table = ttk.Treeview(results_frame, columns=columns, show="headings", height=6)
        for code, title, width in (
            ("id", "#", 35), ("variante", "Type / color", 145),
            ("d", "D", 42), ("t", "T", 42), ("etat", "Status", 88),
        ):
            self.table.heading(code, text=title)
            self.table.column(code, width=width, minwidth=32, stretch=(code == "variante"))
        table_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=table_scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        table_scroll.pack(side="right", fill="y")

        preview_header = ttk.Frame(right)
        preview_header.pack(fill="x", pady=(0, 6))
        ttk.Label(preview_header, textvariable=self.current_title, style="Section.TLabel").pack(side="left", fill="x", expand=True)
        ttk.Button(preview_header, text="Previous", width=8, command=lambda: self._move(-1)).pack(side="left", padx=2)
        ttk.Button(preview_header, text="Next", width=6, command=lambda: self._move(1)).pack(side="left", padx=2)
        ttk.Button(preview_header, text="Zoom -", width=7, command=lambda: self._zoom_by(1 / 1.25)).pack(side="left", padx=2)
        ttk.Button(preview_header, text="Zoom +", width=7, command=lambda: self._zoom_by(1.25)).pack(side="left", padx=2)
        ttk.Button(preview_header, text="Fit", command=self._fit).pack(side="left", padx=2)
        ttk.Button(preview_header, text="100%", command=self._fullsize).pack(side="left", padx=2)
        ttk.Button(preview_header, text="Open", command=self._open_preview).pack(side="left", padx=(8, 0))

        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, background="#202831", highlightthickness=0)
        yscroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        xscroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        ttk.Label(right, text="Green: prediction available  |  Orange: review  |  D: detection  |  T: type",
                  foreground="#596878").pack(anchor="w", pady=(5, 0))

        footer = ttk.Frame(self.root, padding=(14, 2, 14, 8))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status_var).pack(side="left", fill="x", expand=True)
        self.output_btn = ttk.Button(footer, text="Open results folder", command=self._open_output)
        self.output_btn.pack(side="right")
        self._set_busy(False)

    def _browse_path(self, variable: tk.StringVar, ext: str, directory: bool) -> None:
        if directory:
            choice = filedialog.askdirectory(title="Select a folder")
        else:
            choice = filedialog.askopenfilename(title="Select a model",
                                                filetypes=[("Model files", ext), ("All files", "*.*")])
        if choice:
            variable.set(choice)

    def _choose_images(self) -> None:
        selected = filedialog.askopenfilenames(title="Select images", filetypes=IMAGE_FILTER)
        if selected:
            self._add_sources(selected)

    def _choose_folder(self) -> None:
        selected = filedialog.askdirectory(title="Select an image folder")
        if selected:
            self._add_sources([selected])

    def _add_sources(self, sources: list[Path | str]) -> None:
        try:
            files = collect_images(sources)
        except Exception as exc:
            messagebox.showerror("Images", str(exc))
            return
        existing = set(self.files)
        new_files = [p for p in files if p not in existing]
        self.files.extend(new_files)
        for path in new_files:
            self.listbox.insert("end", f"{path.name}  [{path.parent.name}]")
        self.count_var.set(f"{len(self.files)} images")
        if new_files and not self.listbox.curselection():
            self._choose_index(0)

    def _remove_selected(self) -> None:
        indices = list(self.listbox.curselection())
        for index in reversed(indices):
            path = self.files.pop(index)
            self.reports.pop(path, None)
            self.listbox.delete(index)
        self.count_var.set(f"{len(self.files)} images")
        self._choose_index(0) if self.files else self._show_image(None)

    def _clear_files(self) -> None:
        self.files.clear()
        self.reports.clear()
        self.listbox.delete(0, "end")
        self.count_var.set("0 images")
        self._show_image(None)

    def _choose_index(self, index: int) -> None:
        if not self.files:
            self._show_image(None)
            return
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(index)
        self.listbox.see(index)
        self._show_image(self.files[index])

    def _select_image(self, _event: object | None = None) -> None:
        selection = self.listbox.curselection()
        if selection:
            self._show_image(self.files[selection[0]])

    def _move(self, delta: int) -> None:
        if not self.files:
            return
        selection = self.listbox.curselection()
        current = selection[0] if selection else 0
        self._choose_index((current + delta) % len(self.files))

    def _show_image(self, path: Path | None) -> None:
        self.displayed = path
        self.original_image = None
        self.tk_image = None
        self.canvas.delete("all")
        for child in self.table.get_children():
            self.table.delete(child)
        if path is None:
            self.current_title.set("No image selected")
            return
        report = self.reports.get(path)
        display_path = Path(report["image_annotee"]) if report else path
        self.current_title.set(f"{path.name}  -  {'ACS result' if report else 'Original'}")
        try:
            with Image.open(display_path) as opened:
                self.original_image = ImageOps.exif_transpose(opened).convert("RGB")
        except Exception as exc:
            self.status_var.set(f"Unable to display {display_path.name}: {exc}")
            return
        if report:
            for item in report.get("detections", []):
                self.table.insert("", "end", values=(
                    item.get("id", ""),
                    f"{item.get('type_predit', '?')}/{item.get('couleur_predite', '?')}",
                    f"{item.get('confiance_detection', 0):.2f}",
                    f"{item.get('confiance_type', 0):.2f}",
                    "Review" if item.get("avertissements") else "OK",
                ))
        self.fit_mode = True
        self._redraw()

    def _redraw(self) -> None:
        if self.original_image is None:
            return
        available_w = max(1, self.canvas.winfo_width() - 12)
        available_h = max(1, self.canvas.winfo_height() - 12)
        ow, oh = self.original_image.size
        if self.fit_mode:
            ratio = min(available_w / ow, available_h / oh, 1.0)
        else:
            ratio = self.zoom
        w, h = max(1, round(ow * ratio)), max(1, round(oh * ratio))
        if w * h > 25_000_000:
            self.status_var.set("Zoom is too large for this image")
            return
        resized = self.original_image.resize((w, h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.configure(scrollregion=(0, 0, w, h))

    def _on_canvas_resize(self, _event: object) -> None:
        if self.resize_task is not None:
            self.root.after_cancel(self.resize_task)
        self.resize_task = self.root.after(110, self._resize_if_needed)

    def _resize_if_needed(self) -> None:
        self.resize_task = None
        if self.fit_mode:
            self._redraw()

    def _zoom_by(self, factor: float) -> None:
        if self.original_image is None:
            return
        if self.fit_mode:
            ow, oh = self.original_image.size
            self.zoom = min(max(1, self.canvas.winfo_width() - 12) / ow,
                            max(1, self.canvas.winfo_height() - 12) / oh, 1.0)
        self.fit_mode = False
        self.zoom = min(3.0, max(0.15, self.zoom * factor))
        self._redraw()

    def _fit(self) -> None:
        self.fit_mode = True
        self._redraw()

    def _fullsize(self) -> None:
        self.fit_mode = False
        self.zoom = 1.0
        self._redraw()

    def _settings(self) -> Settings:
        try:
            conf = float(self.conf_var.get().replace(",", "."))
        except ValueError as exc:
            raise ValueError("Confidence must be a number, e.g. 0.45") from exc
        return Settings(
            detector=Path(self.detector_var.get()).expanduser().resolve(),
            type_checkpoint=Path(self.type_var.get()).expanduser().resolve(),
            color_model=Path(self.color_var.get()).expanduser().resolve(),
            output=Path(self.output_var.get()).expanduser().resolve(),
            device=self.device_var.get(), confidence=conf,
            save_crops=self.save_crops_var.get(),
        )

    def _start(self) -> None:
        if self.busy:
            return
        try:
            settings = self._settings()
            validate_settings(settings)
            if not self.files:
                raise ValueError("Select one or more images first")
        except Exception as exc:
            messagebox.showerror("Settings", str(exc))
            return
        self.cancel_event.clear()
        self.progress.configure(maximum=len(self.files), value=0)
        self.status_var.set("Initializing pipeline...")
        self._set_busy(True)
        images = list(self.files)

        def work() -> None:
            try:
                analyze_batch(images, settings, cache=self.cache,
                              cancelled=self.cancel_event,
                              emit=lambda name, payload: self.messages.put((name, payload)))
            except Exception as exc:
                self.messages.put(("error", {"message": str(exc)}))

        Thread(target=work, name="acs-inference", daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        for btn in (self.start_btn, self.add_btn, self.folder_btn, self.remove_btn, self.clear_btn):
            btn.configure(state="disabled" if busy else "normal")
        self.cancel_btn.configure(state="normal" if busy else "disabled")
        self.device_combo.configure(state="disabled" if busy else "readonly")
        self.conf_entry.configure(state="disabled" if busy else "normal")

    def _cancel(self) -> None:
        if self.busy:
            self.cancel_event.set()
            self.status_var.set("Cancellation requested: stopping after current image...")

    def _poll_messages(self) -> None:
        try:
            while True:
                name, payload = self.messages.get_nowait()
                if name == "status":
                    self.status_var.set(payload["message"])
                elif name == "progress":
                    self.status_var.set(f"Image {payload['current']}/{payload['total']}: "
                                        f"{Path(payload['image']).name}")
                elif name == "result":
                    report = payload["report"]
                    path = Path(report["image_source"]).resolve()
                    self.reports[path] = report
                    self.progress.configure(value=payload["current"])
                    if self.displayed is None or self.displayed == path:
                        self._show_image(path)
                    self.status_var.set(f"{payload['current']}/{payload['total']}: "
                                        f"{report['nombre_detections']} candies detected")
                elif name in {"done", "cancelled", "error"}:
                    self._set_busy(False)
                    if name == "done":
                        self.status_var.set(f"Completed: {payload['processed']} image(s). "
                                            f"Results: {payload['output']}")
                        if self.files:
                            selected = self.listbox.curselection()
                            if not selected or self.files[selected[0]] not in self.reports:
                                first_result = next((i for i, f in enumerate(self.files)
                                                     if f in self.reports), 0)
                                self._choose_index(first_result)
                    elif name == "cancelled":
                        self.status_var.set(f"Stopped after {payload['processed']} "
                                            f"image(s) of {payload['total']}.")
                    else:
                        self.status_var.set("Error during inference")
                        messagebox.showerror("ACS - Error", payload["message"])
                    if self.close_after_work:
                        self.root.destroy()
                        return
        except Empty:
            pass
        self.root.after(120, self._poll_messages)

    def _preview_path(self) -> Path | None:
        if self.displayed is None:
            return None
        report = self.reports.get(self.displayed)
        return Path(report["image_annotee"]) if report else self.displayed

    @staticmethod
    def _open_external(path: Path) -> None:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _open_preview(self) -> None:
        path = self._preview_path()
        if path and path.exists():
            try:
                self._open_external(path)
            except OSError as exc:
                messagebox.showerror("Open file", str(exc))

    def _open_output(self) -> None:
        path = Path(self.output_var.get()).expanduser().resolve()
        if path.is_dir():
            try:
                self._open_external(path)
            except OSError as exc:
                messagebox.showerror("Folder", str(exc))
        else:
            messagebox.showinfo("Results", "The output folder does not exist yet.")

    def _close(self) -> None:
        if not self.busy:
            self.root.destroy()
            return
        if messagebox.askyesno("Exit", "An analysis is running.\n"
                                "Stop after the current image and exit?"):
            self.close_after_work = True
            self._cancel()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.check_only:
        settings = Settings(
            detector=args.detector, type_checkpoint=args.type_checkpoint,
            color_model=args.color_model, output=args.output,
            device=args.device, confidence=args.conf,
        )
        validate_settings(settings)
        if args.source:
            images = collect_images([args.source])
            print(f"ACS - Images ready: {len(images)}")
        print("ACS - Configuration validated. No window opened.")
        return
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit("GUI unavailable: check $DISPLAY and Tkinter. "
                         f"Details: {exc}") from exc
    ACSApplication(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
