"""
A more robust Quantum Art Generator.


Dependencies
------------
* qiskit
* pillow (PIL)
* PyQt5
* numpy

If Qiskit is not available the generator will automatically fall back to a classical RNG.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json
from datetime import datetime
import colorsys

from PIL import Image, ImageDraw

# -------------------------- Optional Qiskit import ---------------------------
try:
    import qiskit  # type: ignore
    from qiskit_aer import AerSimulator  # type: ignore
    from qiskit import transpile  # type: ignore

    _QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover – any ImportError or runtime error
    _QISKIT_AVAILABLE = False

# ------------------------------- Qt Imports ---------------------------------
from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

# --------------------------- Logging configuration --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------ Data classes --------------------------------


@dataclass
class ArtConfig:
    """Configuration for art generation."""

    width: int = 800
    height: int = 800
    num_quantum_bits: int = 32
    min_shapes: int = 5
    max_extra_shapes: int = 10  # total shapes = min_shapes + seed % max_extra_shapes
    style: int = 0              # 0 = original, 1 = fractal, 2 = inkblot


# -------------------------- Quantum randomness ------------------------------

def _quantum_seed(num_bits: int) -> str:
    """Return a bitstring obtained from a quantum simulator.

    If Qiskit or the simulator is not available, a ``RuntimeError`` is raised.
    """

    if not _QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit with Aer simulator is not available.")

    logger.debug("Generating %d-bit quantum seed via Qiskit", num_bits)

    backend = AerSimulator(method="statevector")  # type: ignore
    max_qubits: int = backend.configuration().num_qubits  # typically 29

    bits: str = ""
    remaining = num_bits

    while remaining > 0:
        chunk = min(remaining, max_qubits)

        # simple superposition circuit
        qc = qiskit.QuantumCircuit(chunk, chunk)  # type: ignore
        qc.h(range(chunk))
        qc.measure(range(chunk), range(chunk))

        tqc = transpile(qc, backend, optimization_level=0)  # type: ignore
        result = backend.run(tqc, shots=1).result()  # type: ignore
        counts = result.get_counts()
        bitstring = list(counts.keys())[0].zfill(chunk)  # ensure correct length for chunk

        bits += bitstring
        remaining -= chunk

    # We might have generated a few extra bits if num_bits wasn't a multiple of chunk
    return bits[:num_bits]


def _fallback_seed(num_bits: int) -> str:
    """Return a pseudo-random bitstring using ``random.getrandbits``."""

    bits = random.getrandbits(num_bits)
    return format(bits, f"0{num_bits}b")


def get_seed(num_bits: int) -> str:
    """Get a bitstring of *num_bits* using quantum randomness when possible."""

    try:
        return _quantum_seed(num_bits)
    except Exception as exc:
        logger.warning("Quantum seed failed – falling back to pseudo-random (%s)", exc)
        return _fallback_seed(num_bits)


# --------------------------- Art generation logic ---------------------------


def bitstring_to_ints(bitstring: str, segment_len: int = 8) -> List[int]:
    """Split *bitstring* into *segment_len*-sized pieces and convert to int."""

    return [int(bitstring[i : i + segment_len], 2) for i in range(0, len(bitstring), segment_len)]


def seed_to_color(seed: int) -> Tuple[int, int, int]:
    """Map a seed integer to an RGB color deterministically."""

    rng = random.Random(seed)  # independent RNG

    # Choose a pleasant HSL color then convert to RGB for a nicer palette
    h = rng.random()               # 0-1  hue
    s = 0.6 + rng.random() * 0.4   # 0.6-1.0  saturation (vivid)
    l = 0.35 + rng.random() * 0.3  # 0.35-0.65 lightness (avoid extremes)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def _generate_original(cfg: ArtConfig, draw: ImageDraw.ImageDraw, seed_values: List[int]):
    """Original random art style (semi-transparent shapes)."""

    for i in range(cfg.min_shapes + (seed_values[0] % cfg.max_extra_shapes)):
        x0 = random.randint(0, cfg.width)
        y0 = random.randint(0, cfg.height)
        x1 = random.randint(x0, cfg.width)
        y1 = random.randint(y0, cfg.height)

        base_color = seed_to_color(seed_values[i % len(seed_values)])
        alpha = 120 + (seed_values[i % len(seed_values)] % 100)  # semi-transparent 120-220
        color = base_color + (alpha,)

        shape_selector = seed_values[i % len(seed_values)] % 4

        if shape_selector == 0:
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=None)
        elif shape_selector == 1:
            draw.ellipse([x0, y0, x1, y1], fill=color, outline=None)
        elif shape_selector == 2:
            draw.polygon([(x0, y0), (x1, y0), ((x0 + x1) // 2, y1)], fill=color, outline=None)
        else:
            thickness = 1 + seed_values[i % len(seed_values)] % 8
            draw.line([(x0, y0), (x1, y1)], fill=color, width=thickness)


# ---------------------- Fractal & Inkblot generators ----------------------

import math

def _generate_fractal(cfg: ArtConfig, draw: ImageDraw.ImageDraw, seed_values: List[int]):
    """Recursive square fractal (Pythagoras tree–like) for higher complexity."""

    max_depth = 5

    def recurse(x: int, y: int, size: int, depth: int, angle: float = 0.0):
        if depth == 0 or size < 4:
            return

        idx = depth % len(seed_values)
        color = seed_to_color(seed_values[idx]) + (200,)

        # compute rectangle corners with optional rotation
        half = size / 2
        cx, cy = x + half, y + half
        points = [(-half, -half), (half, -half), (half, half), (-half, half)]
        rot = angle
        rot_points = []
        for px, py in points:
            rx = px * math.cos(rot) - py * math.sin(rot)
            ry = px * math.sin(rot) + py * math.cos(rot)
            rot_points.append((cx + rx, cy + ry))

        draw.polygon(rot_points, outline=color, width=2)

        # recurse on each corner
        new_size = int(size * 0.5)
        for dx, dy in [(-half, -half), (half, -half), (half, half), (-half, half)]:
            nx = int(cx + dx * 0.5 - new_size / 2)
            ny = int(cy + dy * 0.5 - new_size / 2)
            recurse(nx, ny, new_size, depth - 1, rot + (seed_values[idx] % 360) * math.pi / 180)

    recurse(0, 0, min(cfg.width, cfg.height), max_depth)


def _generate_inkblot(cfg: ArtConfig, img: Image.Image, draw: ImageDraw.ImageDraw, seed_values: List[int]):
    """Generate a mirrored grayscale inkblot on white background with high complexity."""

    rng = random.Random(seed_values[0])
    half_w = cfg.width // 2
    for _ in range(300):  # lots of blotches for complexity
        shape = rng.choice(["ellipse", "rect", "polygon"])
        max_size = rng.randint(10, half_w)
        w = rng.randint(10, max_size)
        h = rng.randint(10, max_size)
        x0 = rng.randint(0, half_w - w)
        y0 = rng.randint(0, cfg.height - h)
        x1 = x0 + w
        y1 = y0 + h

        gray = rng.randint(0, 200)
        color = (gray, gray, gray, 255)

        if shape == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=color, outline=None)
        elif shape == "rect":
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=None)
        else:  # random polygon
            points = [(rng.randint(x0, x1), rng.randint(y0, y1)) for _ in range(5)]
            draw.polygon(points, fill=color)

        mirror_box = img.crop((x0, y0, x1, y1)).transpose(Image.FLIP_LEFT_RIGHT)
        img.paste(mirror_box, (cfg.width - x1, y0))


def generate_art(cfg: ArtConfig, output_path: Path | None = None) -> Path:
    """Generate an image and return the output path.

    If *output_path* is not provided a temporary file in the OS temp dir will be
    created. The filename encodes the quantum seed so output images are
    reproducible.
    """

    bitstring = get_seed(cfg.num_quantum_bits)
    seed_values = bitstring_to_ints(bitstring)

    logger.debug("Seed bits: %s", bitstring)

    # choose base background depending on style
    if cfg.style == 2:  # inkblot – pure white background
        bg_color = (255, 255, 255, 255)
    else:
        bg_color = seed_to_color(seed_values[-1]) + (255,)

    img = Image.new("RGBA", (cfg.width, cfg.height), bg_color)
    draw = ImageDraw.Draw(img, "RGBA")

    # choose style
    if cfg.style == 0:
        _generate_original(cfg, draw, seed_values)
    elif cfg.style == 1:
        _generate_fractal(cfg, draw, seed_values)
    else:
        _generate_inkblot(cfg, img, draw, seed_values)

    if output_path is None:
        output_path = Path(os.getenv("TEMP", ".")).resolve() / f"quantum_art_{bitstring}.png"
    else:
        output_path = Path(output_path)

    # Convert to RGB before saving (removes alpha but keeps composition)
    img.convert("RGB").save(output_path)

    # ----------------------- metadata / provenance -----------------------
    metadata = {
        "seed_bitstring": bitstring,
        "num_qubits": cfg.num_quantum_bits,
        "quantum_random": _QISKIT_AVAILABLE,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "qiskit_version": qiskit.__version__ if _QISKIT_AVAILABLE else None,
        "aer_simulator": "AerSimulator" if _QISKIT_AVAILABLE else None,
        "image_file": str(output_path.name),
        "style": cfg.style,
    }

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Art saved to %s", output_path)
    logger.info("Metadata saved to %s", json_path)
    return output_path

# ---------------------------- Qt Worker Thread ------------------------------

class ArtWorker(QtCore.QObject):
    """Worker object that generates art in a background thread."""

    finished = QtCore.pyqtSignal(Path)
    error = QtCore.pyqtSignal(str)

    def __init__(self, cfg: ArtConfig):
        super().__init__()
        self.cfg = cfg

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            path = generate_art(self.cfg)
            self.finished.emit(path)
        except Exception as exc:
            logger.exception("Error in ArtWorker")
            self.error.emit(str(exc))

# ------------------------------- Main Window --------------------------------

class MainWindow(QtWidgets.QWidget):
    """Main application window."""

    def __init__(self, cfg: ArtConfig | None = None):
        super().__init__()
        self.cfg = cfg or ArtConfig()
        self.setWindowTitle("Quantum Art Generator")
        self.setFixedSize(self.cfg.width + 50, self.cfg.height + 100)

        self.label = QtWidgets.QLabel("Click the button to generate quantum art")
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(self.cfg.width, self.cfg.height)

        # Style selector
        self.style_combo = QtWidgets.QComboBox()
        self.style_combo.addItems(["Original", "Fractal", "Inkblot"])

        self.btn_generate = QtWidgets.QPushButton("Generate Quantum Art")
        self.btn_generate.clicked.connect(self._start_generation)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.style_combo)
        layout.addWidget(self.btn_generate)
        self.setLayout(layout)

        self._thread: QtCore.QThread | None = None
        self._worker: ArtWorker | None = None

    # --------------------------- Generation logic ---------------------------

    def _start_generation(self) -> None:
        self.btn_generate.setEnabled(False)
        self.label.setText("Generating… please wait (this can take ~30-90 s)")

        self._thread = QtCore.QThread()
        # Pass selected style to worker via a fresh ArtConfig copy
        cfg_copy = ArtConfig(**self.cfg.__dict__)
        cfg_copy.style = self.style_combo.currentIndex()

        self._worker = ArtWorker(cfg_copy)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_art_ready)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    @QtCore.pyqtSlot(Path)
    def _on_art_ready(self, path: Path) -> None:
        pixmap = QtGui.QPixmap(str(path)).scaled(
            self.cfg.width,
            self.cfg.height,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save art as…",
            str(path.name),
            "PNG Images (*.png)",
        )
        if save_path:
            try:
                Path(save_path).write_bytes(Path(path).read_bytes())
                # copy metadata JSON sidecar
                src_json = Path(path).with_suffix(".json")
                if src_json.exists():
                    dst_json = Path(save_path).with_suffix(".json")
                    dst_json.write_bytes(src_json.read_bytes())
                logger.info("Image saved to %s", save_path)
            except Exception as exc:
                self._show_error(f"Failed to save image: {exc}")

        self.label.setText("Art generation completed!")
        self.btn_generate.setEnabled(True)

    @QtCore.pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self._show_error(message)
        self.btn_generate.setEnabled(True)

    def _show_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Error", message)
        self.label.setText("An error occurred – check logs.")


# ----------------------------- CLI interface --------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate quantum art via GUI or CLI")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    args = parser.parse_args()

    cfg = ArtConfig()

    if args.cli:
        path = generate_art(cfg)
        print(f"Image saved to {path}")
        return

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(cfg)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
