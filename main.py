"""
A more robust Quantum Art Generator.

Key improvements over the original script:
1. Clear separation between UI and generation logic.
2. Graceful fallback to the Python ``random`` module if Qiskit or the Aer backend is unavailable.
3. Uses a worker thread (``QtCore.QThread``) so the GUI remains responsive while art is generated.
4. Added type annotations, extensive docstrings, and logging for easier debugging.
5. The user is prompted with a *save* dialog after the art is generated instead of silently writing into a folder.
6. Extra command-line interface: run ``python quantum_art_generator.py --cli`` to generate art without a GUI.
7. Support for IBM Quantum computers with API key authentication.

Dependencies
------------
* qiskit
* qiskit-ibm-runtime (for IBM Quantum access)
* pillow (PIL)
* PyQt5
* numpy

If Qiskit is not available the generator will automatically fall back to a classical RNG.
IBM Quantum computers require a valid API key from https://quantum-computing.ibm.com/
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import json
from datetime import datetime
import colorsys
import math
import numpy as np
try:
    from noise import pnoise2  # Perlin noise library
except ImportError:  # fallback if noise not installed
    def pnoise2(x, y, repeatx=1024, repeaty=1024, base=0):
        return random.random() * 2 - 1

from PIL import Image, ImageDraw

# -------------------------- Optional Qiskit import ---------------------------
try:
    import qiskit  # type: ignore
    from qiskit_aer import AerSimulator  # type: ignore
    from qiskit import transpile  # type: ignore
    
    _QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover – any ImportError or runtime error
    _QISKIT_AVAILABLE = False

# -------------------------- Optional IBM Quantum import ---------------------------
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler  # type: ignore
    
    _IBM_QUANTUM_AVAILABLE = True
except Exception as e:  # pragma: no cover – any ImportError or runtime error
    _IBM_QUANTUM_AVAILABLE = False

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
    seed_value: int | None = None  # user-defined seed (None = random)
    complexity: int = 5            # 1-10 complexity multiplier
    palette: str = "Vivid"         # Vivid / Pastel / Neon / Dark
    high_res_factor: int = 1       # 1 = native, 2 = 2×, 4 = 4×
    ibm_api_key: str | None = None  # IBM Quantum API key
    use_ibm_quantum: bool = False   # Whether to use IBM quantum computers
    ibm_backend: str = "ibmq_qasm_simulator"  # IBM backend name
    soft_mode: bool = False          # Whether to use soft/artistic mode


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


def _ibm_quantum_seed(num_bits: int, api_key: str, backend_name: str) -> str:
    """Return a bitstring obtained from IBM quantum computers.
    
    If IBM Quantum is not available or authentication fails, a ``RuntimeError`` is raised.
    """
    
    if not _IBM_QUANTUM_AVAILABLE:
        raise RuntimeError("IBM Quantum Runtime is not available.")
    
    logger.debug("Generating %d-bit quantum seed via IBM Quantum", num_bits)
    
    try:
        # Authenticate with IBM Quantum
        service = QiskitRuntimeService(channel="ibm_cloud", token=api_key)
        
        # Get available backends
        backends = service.backends()
        logger.info("Available IBM backends: %s", [b.name for b in backends])
        
        if not backends:
            logger.warning("No backends available. This might be a permissions issue or account limitation.")
            raise RuntimeError("No IBM Quantum backends available. Check your account permissions.")
        
        # Find the requested backend
        backend = None
        for b in backends:
            if b.name == backend_name:
                backend = b
                break
        
        if backend is None:
            # Fallback to simulator if requested backend not found
            logger.warning("Backend %s not found, using first available backend", backend_name)
            # Use the first available backend or a default simulator
            backend = backends[0]
            logger.info("Using first available backend: %s", backend.name)
        
        logger.info("Using IBM backend: %s", backend.name)
        
        # Create quantum circuit
        qc = qiskit.QuantumCircuit(num_bits, num_bits)
        qc.h(range(num_bits))  # Hadamard gates for superposition
        qc.measure(range(num_bits), range(num_bits))
        
        # Transpile for the backend
        tqc = transpile(qc, backend, optimization_level=0)
        
        # Run on quantum computer
        sampler = Sampler(session=service, options={"backend": backend})
        job = sampler.run(tqc, shots=1)
        result = job.result()
        
        # Extract bitstring from result
        counts = result.quasi_dists[0]
        bitstring = ""
        for bit, count in counts.items():
            if count > 0:  # Get the measured bitstring
                bitstring = format(bit, f"0{num_bits}b")
                break
        
        if not bitstring:
            raise RuntimeError("No valid measurement result obtained")
        
        logger.info("IBM Quantum job completed successfully")
        return bitstring
        
    except Exception as exc:
        logger.error("IBM Quantum authentication or execution failed: %s", exc)
        raise RuntimeError(f"IBM Quantum failed: {exc}")


def _fallback_seed(num_bits: int) -> str:
    """Return a pseudo-random bitstring using ``random.getrandbits``."""

    bits = random.getrandbits(num_bits)
    return format(bits, f"0{num_bits}b")


def get_seed(num_bits: int, cfg: ArtConfig | None = None) -> str:
    """Get a bitstring of *num_bits* using quantum randomness when possible."""
    
    # If IBM Quantum is configured and enabled, try that first
    if cfg and cfg.use_ibm_quantum and cfg.ibm_api_key:
        try:
            return _ibm_quantum_seed(num_bits, cfg.ibm_api_key, cfg.ibm_backend)
        except Exception as exc:
            logger.warning("IBM Quantum seed failed – falling back to local quantum (%s)", exc)
    
    # Try local quantum simulator
    try:
        return _quantum_seed(num_bits)
    except Exception as exc:
        logger.warning("Quantum seed failed – falling back to pseudo-random (%s)", exc)
        return _fallback_seed(num_bits)


# --------------------------- Art generation logic ---------------------------


def bitstring_to_ints(bitstring: str, segment_len: int = 8) -> List[int]:
    """Split *bitstring* into *segment_len*-sized pieces and convert to int."""

    return [int(bitstring[i : i + segment_len], 2) for i in range(0, len(bitstring), segment_len)]


def seed_to_color(seed: int, palette: str = "Vivid") -> Tuple[int, int, int]:
    """Map a seed integer to an RGB color deterministically."""

    rng = random.Random(seed)  # independent RNG

    # Choose a pleasant HSL color then convert to RGB for a nicer palette
    h = rng.random()               # 0-1  hue
    # adjust by palette
    palette_map = {
        "Vivid": (0.6, 0.4, 0.35, 0.3),
        "Pastel": (0.3, 0.2, 0.7, 0.2),
        "Neon": (0.9, 0.1, 0.45, 0.1),
        "Dark": (0.5, 0.3, 0.15, 0.15),
    }
    s_base, s_var, l_base, l_var = palette_map.get(palette, (0.6, 0.4, 0.35, 0.3))

    s = s_base + rng.random() * s_var
    l = l_base + rng.random() * l_var
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def _generate_original_soft(cfg: ArtConfig, draw: ImageDraw.ImageDraw, seed_values: List[int]):
    """Soft/artistic version of original random art style with painterly effects."""

    total_shapes = (cfg.min_shapes + (seed_values[0] % cfg.max_extra_shapes)) * cfg.complexity
    
    # Create a base layer with soft gradients
    for i in range(total_shapes // 3):  # Fewer base shapes for soft effect
        x0 = random.randint(0, cfg.width)
        y0 = random.randint(0, cfg.height)
        x1 = random.randint(x0, cfg.width)
        y1 = random.randint(y0, cfg.height)

        base_color = seed_to_color(seed_values[i % len(seed_values)], cfg.palette)
        alpha = 30 + (seed_values[i % len(seed_values)] % 50)  # Very transparent for blending
        
        # Create soft gradient shapes
        for step in range(5):  # Multiple layers for softness
            fade_factor = step / 4.0
            faded_color = tuple(int(c * fade_factor) for c in base_color) + (alpha,)
            
            # Slightly offset positions for organic feel
            offset_x = random.randint(-10, 10)
            offset_y = random.randint(-10, 10)
            
            draw.ellipse([
                x0 + offset_x, y0 + offset_y, 
                x1 + offset_x, y1 + offset_y
            ], fill=faded_color, outline=None)
    
    # Add organic brush strokes
    for i in range(total_shapes):
        x0 = random.randint(0, cfg.width)
        y0 = random.randint(0, cfg.height)
        
        # Create curved brush strokes
        points = []
        num_points = random.randint(3, 8)
        for j in range(num_points):
            x = x0 + j * random.randint(10, 30) + random.randint(-15, 15)
            y = y0 + j * random.randint(10, 30) + random.randint(-15, 15)
            points.append((x, y))
        
        if len(points) > 2:
            base_color = seed_to_color(seed_values[i % len(seed_values)], cfg.palette)
            alpha = 80 + (seed_values[i % len(seed_values)] % 120)
            color = base_color + (alpha,)
            
            # Draw organic brush stroke
            draw.line(points, fill=color, width=random.randint(3, 12))
            
            # Add soft highlights
            highlight_color = tuple(min(255, c + 50) for c in base_color) + (alpha // 2,)
            draw.line(points, fill=highlight_color, width=random.randint(1, 4))


def _generate_original(cfg: ArtConfig, draw: ImageDraw.ImageDraw, seed_values: List[int]):
    """Original random art style (semi-transparent shapes)."""

    # Use soft mode if enabled
    if cfg.soft_mode:
        _generate_original_soft(cfg, draw, seed_values)
        return

    total_shapes = (cfg.min_shapes + (seed_values[0] % cfg.max_extra_shapes)) * cfg.complexity
    for i in range(total_shapes):
        x0 = random.randint(0, cfg.width)
        y0 = random.randint(0, cfg.height)
        x1 = random.randint(x0, cfg.width)
        y1 = random.randint(y0, cfg.height)

        base_color = seed_to_color(seed_values[i % len(seed_values)], cfg.palette)
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

def _generate_fractal_soft(cfg: ArtConfig, draw: ImageDraw.ImageDraw, seed_values: List[int]):
    """Soft/artistic version of fractal with organic, flowing shapes."""

    max_depth = 2 + cfg.complexity  # scale depth

    def recurse_soft(x: int, y: int, size: int, depth: int, angle: float = 0.0):
        if depth == 0 or size < 8:
            return

        idx = depth % len(seed_values)
        base_color = seed_to_color(seed_values[idx], cfg.palette)
        
        # Create soft, organic shapes
        for layer in range(3):  # Multiple layers for softness
            alpha = 60 + layer * 20  # Increasing transparency
            color = base_color + (alpha,)
            
            # Organic shape with curved edges
            points = []
            num_points = 8
            for i in range(num_points):
                angle_offset = (i / num_points) * 2 * math.pi
                radius = size / 2 + random.randint(-size//4, size//4)
                px = x + size/2 + radius * math.cos(angle_offset + angle)
                py = y + size/2 + radius * math.sin(angle_offset + angle)
                points.append((int(px), int(py)))
            
            if len(points) > 2:
                draw.polygon(points, fill=color, outline=None)

        # Recurse with organic branching
        new_size = int(size * 0.6)
        for i in range(4):
            angle_offset = i * math.pi / 2 + random.uniform(-0.3, 0.3)
            nx = int(x + size/2 + math.cos(angle_offset) * size/3)
            ny = int(y + size/2 + math.sin(angle_offset) * size/3)
            recurse_soft(nx, ny, new_size, depth - 1, angle + random.uniform(-0.5, 0.5))

    recurse_soft(0, 0, min(cfg.width, cfg.height), max_depth)


def _generate_fractal(cfg: ArtConfig, draw: ImageDraw.ImageDraw, seed_values: List[int]):
    """Recursive square fractal (Pythagoras tree–like) for higher complexity."""

    # Use soft mode if enabled
    if cfg.soft_mode:
        _generate_fractal_soft(cfg, draw, seed_values)
        return

    max_depth = 2 + cfg.complexity  # scale depth

    def recurse(x: int, y: int, size: int, depth: int, angle: float = 0.0):
        if depth == 0 or size < 4:
            return

        idx = depth % len(seed_values)
        color = seed_to_color(seed_values[idx], cfg.palette) + (200,)

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
    """Generate a mirrored black & white inkblot on white background."""
    rng = random.Random(seed_values[0])
    half_w = cfg.width // 2

    for _ in range(100 * cfg.complexity):
        # shape = rng.choice(["ellipse", "rect", "polygon"])
        shape = "ellipse"
        max_size = rng.randint(10, half_w)
        w = rng.randint(10, max_size)
        h = rng.randint(10, max_size)
        x0 = rng.randint(0, half_w - w)
        y0 = rng.randint(0, cfg.height - h)
        x1 = x0 + w
        y1 = y0 + h

        color = rng.choice([(0, 0, 0), (255, 255, 255)])  # black or white

        if shape == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=color)
        elif shape == "rect":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            points = [(rng.randint(x0, x1), rng.randint(y0, y1)) for _ in range(5)]
            draw.polygon(points, fill=color)

        mirror_box = img.crop((x0, y0, x1, y1)).transpose(Image.FLIP_LEFT_RIGHT)
        img.paste(mirror_box, (cfg.width - x1, y0))


def _generate_mondrian(cfg: ArtConfig, draw: ImageDraw.ImageDraw, seed_values: List[int]):
    """Piet Mondrian style coloured rectangles with thick black borders."""

    rng = random.Random(seed_values[1])
    rects = 20 * cfg.complexity
    for _ in range(rects):
        x0 = rng.randint(0, cfg.width - 20)
        y0 = rng.randint(0, cfg.height - 20)
        x1 = x0 + rng.randint(20, cfg.width // 2)
        y1 = y0 + rng.randint(20, cfg.height // 2)

        color = rng.choice([(239, 221, 111), (227, 38, 54), (44, 117, 255), (255, 255, 255)]) + (255,)
        draw.rectangle([x0, y0, x1, y1], fill=color, outline="black", width=6)


def _generate_galaxy(cfg: ArtConfig, draw: ImageDraw.ImageDraw, seed_values: List[int]):
    """Logarithmic colourful spiral of dots resembling a galaxy."""

    rng = random.Random(seed_values[2])
    num_stars = 1000 * cfg.complexity
    center_x, center_y = cfg.width / 2, cfg.height / 2
    for i in range(num_stars):
        angle = i * 0.1
        radius = 0.3 * cfg.width * math.exp(-angle / (5 * cfg.complexity))
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        size = rng.randint(1, 3)
        color = seed_to_color(rng.randint(0, 1 << 30), cfg.palette) + (220,)
        draw.ellipse([x, y, x + size, y + size], fill=color, outline=None)


def _generate_nebula(cfg: ArtConfig, img: Image.Image, seed_values: List[int], palette: str):
    """Perlin noise based colourful nebula background."""

    w, h = cfg.width, cfg.height
    pixels = img.load()
    scale = 100.0 / cfg.complexity
    
    # Use seed for consistent noise generation
    noise_seed = seed_values[0] if seed_values else 0
    
    for y in range(h):
        for x in range(w):
            # Generate perlin noise for nebula effect
            n = pnoise2(x / scale, y / scale, repeatx=1024, repeaty=1024, base=noise_seed)
            n = (n + 1) / 2  # Normalize to 0..1
            
            # Create nebula colors based on noise
            if n < 0.3:
                # Dark space regions
                color = (int(n * 50), int(n * 30), int(n * 80))
            elif n < 0.6:
                # Nebula gas regions
                hue = n * 0.8 + 0.2  # Purple to blue range
                r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.6)
                color = (int(r * 255), int(g * 255), int(b * 255))
            else:
                # Bright star regions
                color = seed_to_color(int(n * 1e6), palette)
            
            pixels[x, y] = color


def _generate_wave_collapse_soft(cfg: ArtConfig, draw: ImageDraw.ImageDraw, seed_values: List[int]):
    """Soft/artistic version of wave collapse with organic, flowing quantum visualizations."""
    
    rng = random.Random(seed_values[0])
    
    # Clear the existing background and create black quantum vacuum
    draw.rectangle([0, 0, cfg.width, cfg.height], fill=(0, 0, 0, 255))
    
    # Create organic quantum wave functions
    num_waves = 2 + cfg.complexity // 3
    waves = []
    
    for i in range(num_waves):
        center_y = rng.randint(100, cfg.height - 100)
        amplitude = rng.randint(30, 80)
        frequency = rng.randint(2, 6)
        phase = rng.random() * 2 * math.pi
        
        waves.append({
            'center_y': center_y,
            'amplitude': amplitude,
            'frequency': frequency,
            'phase': phase,
            'color': seed_to_color(seed_values[i % len(seed_values)], "Neon")
        })
    
    # Draw organic wave functions with soft gradients
    for wave in waves:
        # Create multiple layers for softness
        for layer in range(4):
            alpha = 40 + layer * 20
            width = 8 - layer * 2
            
            points = []
            for x in range(0, cfg.width, 3):
                # Add organic variation to the wave
                variation = rng.uniform(-0.3, 0.3)
                y = wave['center_y'] + wave['amplitude'] * math.sin(
                    (x / cfg.width) * wave['frequency'] * 2 * math.pi + wave['phase']
                ) * (1 + variation)
                points.append((x, int(y)))
            
            if len(points) > 1:
                color = wave['color'][:3] + (alpha,)
                draw.line(points, fill=color, width=width)
    
    # Add soft quantum clouds (probability distributions)
    for wave in waves:
        for i in range(20):
            x = rng.randint(0, cfg.width)
            wave_y = wave['center_y'] + wave['amplitude'] * math.sin(
                (x / cfg.width) * wave['frequency'] * 2 * math.pi + wave['phase']
            )
            
            # Create soft probability clouds
            cloud_size = rng.randint(10, 30)
            alpha = rng.randint(20, 60)
            color = wave['color'][:3] + (alpha,)
            
            # Multiple overlapping circles for soft effect
            for j in range(3):
                offset_x = rng.randint(-cloud_size//2, cloud_size//2)
                offset_y = rng.randint(-cloud_size//2, cloud_size//2)
                size = cloud_size - j * 5
                
                draw.ellipse([
                    x + offset_x - size//2, int(wave_y) + offset_y - size//2,
                    x + offset_x + size//2, int(wave_y) + offset_y + size//2
                ], fill=color, outline=None)
    
    # Add organic measurement apparatus
    num_detectors = 1 + cfg.complexity // 4
    detectors = []
    
    for i in range(num_detectors):
        detector_x = rng.randint(100, cfg.width - 100)
        detector_y = rng.randint(50, cfg.height - 50)
        detector_size = rng.randint(25, 40)
        
        detectors.append({
            'x': detector_x,
            'y': detector_y,
            'size': detector_size,
            'active': rng.random() > 0.4,
            'color': seed_to_color(seed_values[i % len(seed_values)], "Vivid")
        })
    
    # Draw soft measurement apparatus
    for detector in detectors:
        # Soft detector glow
        glow_color = (255, 255, 255, 30) if detector['active'] else (100, 100, 100, 20)
        for glow in range(3):
            size_offset = glow * 5
            draw.ellipse([
                detector['x'] - size_offset, detector['y'] - size_offset,
                detector['x'] + detector['size'] + size_offset, detector['y'] + detector['size'] + size_offset
            ], fill=glow_color, outline=None)
        
        # Detector frame
        frame_color = (255, 255, 255, 200) if detector['active'] else (100, 100, 100, 150)
        draw.ellipse([
            detector['x'], detector['y'],
            detector['x'] + detector['size'], detector['y'] + detector['size']
        ], outline=frame_color, width=2)
        
        # Soft crosshairs
        center_x = detector['x'] + detector['size'] // 2
        center_y = detector['y'] + detector['size'] // 2
        
        for line_width in [3, 1]:  # Multiple lines for softness
            alpha = 150 if line_width == 3 else 100
            line_color = frame_color[:3] + (alpha,)
            
            draw.line([
                (center_x - 12, center_y),
                (center_x + 12, center_y)
            ], fill=line_color, width=line_width)
            
            draw.line([
                (center_x, center_y - 12),
                (center_x, center_y + 12)
            ], fill=line_color, width=line_width)
    
    # Soft quantum collapse effects
    for detector in detectors:
        if detector['active']:
            for wave in waves:
                wave_x = detector['x']
                wave_y = wave['center_y'] + wave['amplitude'] * math.sin(
                    (wave_x / cfg.width) * wave['frequency'] * 2 * math.pi + wave['phase']
                )
                
                # Create soft collapsed particles
                particle_size = rng.randint(8, 15)
                particle_color = detector['color'][:3] + (200,)
                
                # Soft particle glow
                for glow in range(4):
                    glow_size = particle_size + glow * 3
                    glow_alpha = 50 - glow * 10
                    glow_color = particle_color[:3] + (glow_alpha,)
                    
                    draw.ellipse([
                        int(wave_x) - glow_size//2, int(wave_y) - glow_size//2,
                        int(wave_x) + glow_size//2, int(wave_y) + glow_size//2
                    ], fill=glow_color, outline=None)
                
                # Main particle
                draw.ellipse([
                    int(wave_x) - particle_size//2, int(wave_y) - particle_size//2,
                    int(wave_x) + particle_size//2, int(wave_y) + particle_size//2
                ], fill=particle_color, outline=(255, 255, 255, 150), width=1)
    
    # Add soft quantum interference patterns
    for i in range(cfg.complexity):
        if len(waves) >= 2:
            wave1 = rng.choice(waves)
            wave2 = rng.choice(waves)
            
            if wave1 != wave2:
                interference_x = rng.randint(100, cfg.width - 100)
                interference_y = (wave1['center_y'] + wave2['center_y']) // 2
                
                # Soft interference rings
                interference_color = seed_to_color(seed_values[i % len(seed_values)], "Neon")[:3] + (80,)
                for ring in range(5):
                    radius = 20 + ring * 15
                    alpha = 100 - ring * 15
                    ring_color = interference_color[:3] + (alpha,)
                    
                    draw.ellipse([
                        interference_x - radius, interference_y - radius,
                        interference_x + radius, interference_y + radius
                    ], outline=ring_color, width=2)


def _generate_wave_collapse(cfg: ArtConfig, draw: ImageDraw.ImageDraw, seed_values: List[int]):
    """Generate art representing quantum wave function collapse - proper waveform visualization."""
    
    # Use soft mode if enabled
    if cfg.soft_mode:
        _generate_wave_collapse_soft(cfg, draw, seed_values)
        return
    
    logger.info("Wave collapse function called with complexity: %d, seed_values: %s", cfg.complexity, seed_values[:5])
    
    rng = random.Random(seed_values[0])
    
    # Clear the existing background and create black quantum vacuum
    draw.rectangle([0, 0, cfg.width, cfg.height], fill=(0, 0, 0, 255))
    
    # Create quantum wave functions as actual waveforms
    num_waves = 2 + cfg.complexity // 3
    waves = []
    
    logger.info("Creating %d wave functions", num_waves)
    
    for i in range(num_waves):
        # Each wave function has a center line and amplitude
        center_y = rng.randint(100, cfg.height - 100)
        amplitude = rng.randint(20, 60)
        frequency = rng.randint(3, 8)
        phase = rng.random() * 2 * math.pi
        
        waves.append({
            'center_y': center_y,
            'amplitude': amplitude,
            'frequency': frequency,
            'phase': phase,
            'color': seed_to_color(seed_values[i % len(seed_values)], "Neon")
        })
    
    logger.info("Drawing %d wave functions", len(waves))
    
    # Draw wave functions as actual waveforms
    for wave in waves:
        points = []
        for x in range(0, cfg.width, 2):
            # Create sine wave with quantum characteristics
            y = wave['center_y'] + wave['amplitude'] * math.sin(
                (x / cfg.width) * wave['frequency'] * 2 * math.pi + wave['phase']
            )
            points.append((x, int(y)))
        
        # Draw the waveform
        if len(points) > 1:
            # Main wave line
            draw.line(points, fill=wave['color'][:3] + (255,), width=3)
            
            # Add wave envelope (probability distribution)
            for i, (x, y) in enumerate(points):
                if i % 10 == 0:  # Draw envelope points less frequently
                    envelope_size = wave['amplitude'] // 3
                    draw.ellipse([
                        x - envelope_size, y - envelope_size,
                        x + envelope_size, y + envelope_size
                    ], fill=wave['color'][:3] + (60,), outline=None)
    
    # Add measurement apparatus (detectors) at specific points
    num_detectors = 1 + cfg.complexity // 4
    detectors = []
    
    logger.info("Creating %d detectors", num_detectors)
    
    for i in range(num_detectors):
        detector_x = rng.randint(100, cfg.width - 100)
        detector_y = rng.randint(50, cfg.height - 50)
        detector_size = rng.randint(30, 50)
        
        detectors.append({
            'x': detector_x,
            'y': detector_y,
            'size': detector_size,
            'active': rng.random() > 0.4,  # Most detectors are active
            'color': seed_to_color(seed_values[i % len(seed_values)], "Vivid")
        })
    
    # Draw measurement apparatus
    for detector in detectors:
        # Detector frame
        frame_color = (255, 255, 255, 255) if detector['active'] else (100, 100, 100, 200)
        draw.rectangle([
            detector['x'], detector['y'],
            detector['x'] + detector['size'], detector['y'] + detector['size']
        ], outline=frame_color, width=3)
        
        # Detector crosshairs
        center_x = detector['x'] + detector['size'] // 2
        center_y = detector['y'] + detector['size'] // 2
        
        # Horizontal crosshair
        draw.line([
            (center_x - 15, center_y),
            (center_x + 15, center_y)
        ], fill=frame_color, width=2)
        
        # Vertical crosshair
        draw.line([
            (center_x, center_y - 15),
            (center_x, center_y + 15)
        ], fill=frame_color, width=2)
        
        # Detector readings (quantum states)
        if detector['active']:
            # Draw measurement result
            state = rng.choice(['|0⟩', '|1⟩', '|+⟩', '|-⟩'])
            # Represent state with colored circle
            state_color = detector['color'][:3] + (255,)
            draw.ellipse([
                center_x - 8, center_y - 8,
                center_x + 8, center_y + 8
            ], fill=state_color, outline=(255, 255, 255, 255), width=2)
    
    # Simulate wave function collapse - particles appearing at measurement points
    collapsed_particles = []
    
    for detector in detectors:
        if detector['active']:
            # When detector is active, collapse nearby wave functions
            for wave in waves:
                # Find the wave value at detector position
                wave_x = detector['x']
                wave_y = wave['center_y'] + wave['amplitude'] * math.sin(
                    (wave_x / cfg.width) * wave['frequency'] * 2 * math.pi + wave['phase']
                )
                
                # Create collapsed particle at wave position
                particle_x = int(wave_x)
                particle_y = int(wave_y)
                particle_size = rng.randint(5, 12)
                
                collapsed_particles.append({
                    'x': particle_x,
                    'y': particle_y,
                    'size': particle_size,
                    'color': detector['color'][:3] + (255,),
                    'detector': detector
                })
    
    logger.info("Created %d collapsed particles", len(collapsed_particles))
    
    # Draw collapsed particles (definite quantum states)
    for particle in collapsed_particles:
        # Sharp, bright particle representing definite state
        draw.ellipse([
            particle['x'], particle['y'],
            particle['x'] + particle['size'],
            particle['y'] + particle['size']
        ], fill=particle['color'], outline=(255, 255, 255, 255), width=2)
        
        # Measurement line from detector to particle
        detector = particle['detector']
        detector_center_x = detector['x'] + detector['size'] // 2
        detector_center_y = detector['y'] + detector['size'] // 2
        particle_center_x = particle['x'] + particle['size'] // 2
        particle_center_y = particle['y'] + particle['size'] // 2
        
        # Draw measurement line
        draw.line([
            (detector_center_x, detector_center_y),
            (particle_center_x, particle_center_y)
        ], fill=(255, 255, 255, 200), width=2)
    
    # Add quantum interference patterns (where waves overlap)
    for i in range(cfg.complexity):
        if len(waves) >= 2:
            wave1 = rng.choice(waves)
            wave2 = rng.choice(waves)
            
            if wave1 != wave2:
                # Create interference pattern
                interference_x = rng.randint(100, cfg.width - 100)
                interference_y = (wave1['center_y'] + wave2['center_y']) // 2
                
                # Draw interference rings
                interference_color = seed_to_color(seed_values[i % len(seed_values)], "Neon")[:3] + (120,)
                for ring in range(4):
                    radius = 15 + ring * 10
                    draw.ellipse([
                        interference_x - radius, interference_y - radius,
                        interference_x + radius, interference_y + radius
                    ], outline=interference_color, width=2)
    
    # Add quantum uncertainty regions (Heisenberg uncertainty principle)
    for i in range(cfg.complexity // 2):
        x = rng.randint(50, cfg.width - 50)
        y = rng.randint(50, cfg.height - 50)
        
        # Uncertainty cloud
        uncertainty_color = (150, 150, 255, 80)
        for j in range(6):
            offset_x = rng.randint(-20, 20)
            offset_y = rng.randint(-20, 20)
            draw.ellipse([
                x + offset_x, y + offset_y,
                x + offset_x + 8, y + offset_y + 8
            ], fill=uncertainty_color, outline=None)
    
    logger.info("Wave collapse function completed")


def generate_art(cfg: ArtConfig, output_path: Path | None = None, progress_cb: callable | None = None) -> Path:
    """Generate an image and return the output path.

    If *output_path* is not provided a temporary file in the OS temp dir will be
    created. The filename encodes the quantum seed so output images are
    reproducible.
    """

    bitstring = get_seed(cfg.num_quantum_bits, cfg)
    seed_values = bitstring_to_ints(bitstring)

    logger.debug("Seed bits: %s", bitstring)

    # choose base background depending on style
    if cfg.style == 2:  # inkblot – pure white background
        bg_color = (255, 255, 255, 255)
    elif cfg.style == 5:  # wave collapse – pure black background
        bg_color = (0, 0, 0, 255)
    else:
        bg_color = seed_to_color(seed_values[-1], cfg.palette) + (255,)

    img = Image.new("RGBA", (cfg.width, cfg.height), bg_color)
    draw = ImageDraw.Draw(img, "RGBA")

    # choose style
    if cfg.style == 0:
        _generate_original(cfg, draw, seed_values)
    elif cfg.style == 1:
        _generate_fractal(cfg, draw, seed_values)
    elif cfg.style == 2:
        _generate_inkblot(cfg, img, draw, seed_values)
    elif cfg.style == 3:
        _generate_mondrian(cfg, draw, seed_values)
    elif cfg.style == 4:
        _generate_galaxy(cfg, draw, seed_values)
    elif cfg.style == 5: # Nebula
        _generate_nebula(cfg, img, seed_values, cfg.palette)
    elif cfg.style == 6: # Wave Function Collapse
        _generate_wave_collapse(cfg, draw, seed_values)
    else:
        _generate_original(cfg, draw, seed_values)  # fallback

    if progress_cb:
        progress_cb(100)

    if output_path is None:
        output_path = Path(os.getenv("TEMP", ".")).resolve() / f"quantum_art_{bitstring}.png"
    else:
        output_path = Path(output_path)

    # Convert to RGB before saving (removes alpha but keeps composition)
    final_img = img.convert("RGB")
    if cfg.high_res_factor > 1:
        final_img = final_img.resize((cfg.width * cfg.high_res_factor, cfg.height * cfg.high_res_factor), Image.LANCZOS)
    final_img.save(output_path)

    # ----------------------- GIF Preview -----------------------------------
    def make_preview_gif(src_img: Image.Image, dst_path: Path, frames: int = 15):
        w, h = src_img.size
        imgs: list[Image.Image] = []
        for i in range(frames):
            # zoom in first half then out
            if i < frames // 2:
                scale = 1.0 + (i / (frames // 2)) * 0.3  # up to 1.3x
            else:
                scale = 1.3 - ((i - frames // 2) / (frames // 2)) * 0.3
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = src_img.resize((new_w, new_h), Image.LANCZOS)
            # crop or pad to original size
            if scale >= 1.0:
                # crop center
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                frame = resized.crop((left, top, left + w, top + h))
            else:
                frame = Image.new("RGB", (w, h), (0, 0, 0))
                frame.paste(resized, ((w - new_w)//2, (h - new_h)//2))
            imgs.append(frame)
        imgs[0].save(dst_path, save_all=True, append_images=imgs[1:], duration=80, loop=0)

    gif_path = output_path.with_suffix('.gif')
    try:
        make_preview_gif(final_img, gif_path)
    except Exception as _gif_exc:
        gif_path = None  # skip if fails

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
        "high_res_factor": cfg.high_res_factor,
        "gif_preview": gif_path.name if gif_path else None,
    }
    
    # Add IBM Quantum metadata if used
    if cfg.use_ibm_quantum and cfg.ibm_api_key:
        metadata.update({
            "ibm_quantum_used": True,
            "ibm_backend": cfg.ibm_backend,
            "ibm_api_key_provided": True,
            "ibm_quantum_available": _IBM_QUANTUM_AVAILABLE,
        })
    else:
        metadata.update({
            "ibm_quantum_used": False,
            "ibm_quantum_available": _IBM_QUANTUM_AVAILABLE,
        })

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
    progress = QtCore.pyqtSignal(int)

    def __init__(self, cfg: ArtConfig):
        super().__init__()
        self.cfg = cfg

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            path = generate_art(self.cfg, progress_cb=self.progress.emit)
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
        # Make window size more flexible for different image sizes
        self.setFixedSize(900, 1000)  # Fixed reasonable size for the GUI

        self.label = QtWidgets.QLabel("Click the button to generate quantum art")
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(400, 300)  # Fixed preview size
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")

        # Style selector
        self.style_combo = QtWidgets.QComboBox()
        self.style_combo.addItems(["Original", "Fractal", "Inkblot", "Mondrian", "Galaxy", "Nebula", "Wave Collapse"])

        # Seed input
        self.seed_edit = QtWidgets.QLineEdit()
        self.seed_edit.setPlaceholderText("Random Seed (blank = random)")

        # Complexity slider
        self.comp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.comp_slider.setRange(1, 10)
        self.comp_slider.setValue(5)
        self.comp_label = QtWidgets.QLabel("Complexity: 5")
        self.comp_slider.valueChanged.connect(lambda v: self.comp_label.setText(f"Complexity: {v}"))

        # Palette picker
        self.palette_combo = QtWidgets.QComboBox()
        self.palette_combo.addItems(["Vivid", "Pastel", "Neon", "Dark"])

        # High-res factor combo
        self.hires_combo = QtWidgets.QComboBox()
        self.hires_combo.addItems(["1×", "2×", "4×"])

        # Artistic mode toggle
        self.artistic_group = QtWidgets.QGroupBox("Artistic Style")
        artistic_layout = QtWidgets.QVBoxLayout()
        
        self.soft_mode_checkbox = QtWidgets.QCheckBox("Soft/Artistic Mode")
        self.soft_mode_checkbox.setToolTip("Creates painterly, organic art with blended colors and natural shapes instead of sharp geometric lines")
        self.soft_mode_checkbox.setChecked(False)
        
        artistic_layout.addWidget(self.soft_mode_checkbox)
        self.artistic_group.setLayout(artistic_layout)

        # Image size and aspect ratio controls
        self.size_group = QtWidgets.QGroupBox("Image Size & Aspect Ratio")
        size_layout = QtWidgets.QVBoxLayout()
        
        # Aspect ratio selector
        self.aspect_label = QtWidgets.QLabel("Aspect Ratio:")
        self.aspect_combo = QtWidgets.QComboBox()
        self.aspect_combo.addItems([
            "Square (1:1) - 800×800",
            "Widescreen (16:9) - 1920×1080", 
            "Classic Photo (3:2) - 1800×1200",
            "Mobile Portrait (9:16) - 1080×1920",
            "Tablet (4:3) - 1600×1200",
            "Ultra-wide (21:9) - 2560×1080",
            "Instagram (4:5) - 1080×1350",
            "Custom Size"
        ])
        self.aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        
        # Custom size inputs
        self.custom_size_label = QtWidgets.QLabel("Custom Size (only for 'Custom Size' above):")
        self.custom_size_label.setVisible(False)
        
        self.width_label = QtWidgets.QLabel("Width:")
        self.width_edit = QtWidgets.QSpinBox()
        self.width_edit.setRange(100, 4000)
        self.width_edit.setValue(800)
        self.width_edit.setVisible(False)
        
        self.height_label = QtWidgets.QLabel("Height:")
        self.height_edit = QtWidgets.QSpinBox()
        self.height_edit.setRange(100, 4000)
        self.height_edit.setValue(800)
        self.height_edit.setVisible(False)
        
        # Add widgets to size group layout
        size_layout.addWidget(self.aspect_label)
        size_layout.addWidget(self.aspect_combo)
        size_layout.addWidget(self.custom_size_label)
        size_layout.addWidget(self.width_label)
        size_layout.addWidget(self.width_edit)
        size_layout.addWidget(self.height_label)
        size_layout.addWidget(self.height_edit)
        self.size_group.setLayout(size_layout)

        # IBM Quantum controls
        self.ibm_group = QtWidgets.QGroupBox("IBM Quantum Computer")
        ibm_layout = QtWidgets.QVBoxLayout()
        
        # IBM API Key input
        self.ibm_api_label = QtWidgets.QLabel("IBM API Key:")
        self.ibm_api_edit = QtWidgets.QLineEdit()
        self.ibm_api_edit.setPlaceholderText("Enter your IBM Quantum API key")
        self.ibm_api_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ibm_api_edit.setToolTip("Get your API key from https://quantum-computing.ibm.com/")
        
        # IBM Backend selection
        self.ibm_backend_label = QtWidgets.QLabel("IBM Backend:")
        self.ibm_backend_combo = QtWidgets.QComboBox()
        self.ibm_backend_combo.addItems([
            "ibmq_qasm_simulator",
            "ibm_manila", 
            "ibm_lima",
            "ibm_belem",
            "ibm_quito",
            "ibm_oslo",
            "ibm_perth",
            "ibm_brisbane",
            "ibm_kyoto",
            "ibm_osaka"
        ])
        
        # Enable IBM Quantum checkbox
        self.ibm_enable_checkbox = QtWidgets.QCheckBox("Use IBM Quantum Computer")
        self.ibm_enable_checkbox.setToolTip("Check to use real IBM quantum computers instead of local simulator")
        
        # IBM status label
        self.ibm_status_label = QtWidgets.QLabel("IBM Quantum: Not Available" if not _IBM_QUANTUM_AVAILABLE else "IBM Quantum: Available")
        self.ibm_status_label.setStyleSheet("color: red;" if not _IBM_QUANTUM_AVAILABLE else "color: green;")
        
        # Add widgets to IBM group layout
        ibm_layout.addWidget(self.ibm_api_label)
        ibm_layout.addWidget(self.ibm_api_edit)
        ibm_layout.addWidget(self.ibm_backend_label)
        ibm_layout.addWidget(self.ibm_backend_combo)
        ibm_layout.addWidget(self.ibm_enable_checkbox)
        ibm_layout.addWidget(self.ibm_status_label)
        self.ibm_group.setLayout(ibm_layout)

        self.btn_generate = QtWidgets.QPushButton("Generate Quantum Art")
        self.btn_generate.clicked.connect(self._start_generation)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.palette_combo)
        layout.addWidget(self.comp_label)
        layout.addWidget(self.comp_slider)
        layout.addWidget(self.seed_edit)
        layout.addWidget(self.hires_combo)
        layout.addWidget(self.artistic_group) # Add the new group box to the layout
        layout.addWidget(self.style_combo)
        layout.addWidget(self.size_group) # Add the new group box to the layout
        layout.addWidget(self.ibm_group) # Add the new group box to the layout
        layout.addWidget(self.btn_generate)

        # Progress bar
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.setLayout(layout)

        self._thread: QtCore.QThread | None = None
        self._worker: ArtWorker | None = None

    # --------------------------- Generation logic ---------------------------

    def _start_generation(self) -> None:
        self.btn_generate.setEnabled(False)
        
        # Check if IBM Quantum is being used
        if self.ibm_enable_checkbox.isChecked() and self.ibm_api_edit.text().strip():
            self.label.setText("Generating on IBM Quantum Computer… please wait (this can take 1-5 minutes)")
        else:
            self.label.setText("Generating… please wait (this can take ~30-90 s)")
        
        # Validate IBM Quantum settings
        if self.ibm_enable_checkbox.isChecked() and not self.ibm_api_edit.text().strip():
            self._show_error("IBM Quantum is enabled but no API key provided. Please enter your IBM Quantum API key.")
            self.btn_generate.setEnabled(True)
            return

        self._thread = QtCore.QThread()
        # Pass selected style to worker via a fresh ArtConfig copy
        cfg_copy = ArtConfig(**self.cfg.__dict__)
        cfg_copy.style = self.style_combo.currentIndex()
        seed_text = self.seed_edit.text().strip()
        cfg_copy.seed_value = int(seed_text) if seed_text.isdigit() else None
        cfg_copy.complexity = self.comp_slider.value()
        cfg_copy.palette = self.palette_combo.currentText()
        cfg_copy.high_res_factor = int(self.hires_combo.currentText().replace("×", ""))

        # Update IBM Quantum settings from GUI
        cfg_copy.ibm_api_key = self.ibm_api_edit.text()
        cfg_copy.use_ibm_quantum = self.ibm_enable_checkbox.isChecked()
        cfg_copy.ibm_backend = self.ibm_backend_combo.currentText()

        # Update image size from GUI
        cfg_copy.width = self.width_edit.value()
        cfg_copy.height = self.height_edit.value()

        # Update artistic mode from GUI
        cfg_copy.soft_mode = self.soft_mode_checkbox.isChecked()

        self._worker = ArtWorker(cfg_copy)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_art_ready)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._worker.progress.connect(self.progress.setValue)

        self._thread.start()

    @QtCore.pyqtSlot(Path)
    def _on_art_ready(self, path: Path) -> None:
        pixmap = QtGui.QPixmap(str(path)).scaled(
            400, 300,  # Fixed preview size
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
                # copy GIF preview sidecar if exists
                src_gif = Path(path).with_suffix(".gif")
                if src_gif.exists():
                    dst_gif = Path(save_path).with_suffix(".gif")
                    dst_gif.write_bytes(src_gif.read_bytes())
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

    def _on_aspect_changed(self, index: int) -> None:
        if index == 7: # Custom Size
            self.custom_size_label.setVisible(True)
            self.width_label.setVisible(True)
            self.width_edit.setVisible(True)
            self.height_label.setVisible(True)
            self.height_edit.setVisible(True)
        else:
            self.custom_size_label.setVisible(False)
            self.width_label.setVisible(False)
            self.width_edit.setVisible(False)
            self.height_label.setVisible(False)
            self.height_edit.setVisible(False)
            # Set default values for common aspect ratios
            if index == 0: # Square (1:1)
                self.width_edit.setValue(800)
                self.height_edit.setValue(800)
            elif index == 1: # Widescreen (16:9)
                self.width_edit.setValue(1920)
                self.height_edit.setValue(1080)
            elif index == 2: # Classic Photo (3:2)
                self.width_edit.setValue(1800)
                self.height_edit.setValue(1200)
            elif index == 3: # Mobile Portrait (9:16)
                self.width_edit.setValue(1080)
                self.height_edit.setValue(1920)
            elif index == 4: # Tablet (4:3)
                self.width_edit.setValue(1600)
                self.height_edit.setValue(1200)
            elif index == 5: # Ultra-wide (21:9)
                self.width_edit.setValue(2560)
                self.height_edit.setValue(1080)
            elif index == 6: # Instagram (4:5)
                self.width_edit.setValue(1080)
                self.height_edit.setValue(1350)


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
