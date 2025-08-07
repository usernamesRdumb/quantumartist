# 🎨 Quantum Art Generator

A robust and customizable art generator using **quantum randomness** to create generative visual artwork — with both **GUI** and **CLI** support.

Supports:
- Local Qiskit simulators 🧪
- Real IBM Quantum computers 🔬
- Classical fallback RNG 🧠

---

## ✨ Features

- 🎛️ Separation of concerns — GUI and generation logic are cleanly separated
- ⚛️ Quantum randomness via Qiskit and IBM Quantum backends
- 🧵 Multithreaded GUI (using QThread) for smooth user experience
- 🖼️ Multiple art styles:
  - Original (transparent shapes)
  - Fractal
  - Inkblot (black & white only)
  - Mondrian
  - Galaxy
  - Nebula
  - Wave Collapse
- 🖌️ Soft mode — painterly, organic variants of every style
- 💾 Save prompt after generation (no silent overwrite)
- 🧰 CLI support via `--cli`
- 🔐 Fallback to classical RNG if quantum backends are unavailable
- 🔑 IBM Quantum support with backend selection and API key auth

---

## 🛠️ Installation

Install required dependencies:

```bash
pip install qiskit qiskit-ibm-runtime pillow PyQt5 numpy
```

Optional (for Perlin noise in Nebula style):

```bash
pip install noise
```

---

## 🚀 Usage

### GUI Mode

```bash
python quantum_art_generator.py
```

### CLI Mode

```bash
python quantum_art_generator.py --cli
```

---

## 🔗 IBM Quantum Integration

1. Get your API key from: https://quantum-computing.ibm.com/
2. Paste it into the GUI field
3. Select a backend (e.g. `ibmq_qasm_simulator`, `ibm_perth`, etc.)
4. Toggle "Use IBM Quantum"

If unavailable or invalid, the app gracefully falls back to local or classical RNG.

---

## 📁 Output Files

Each artwork comes with associated metadata and preview:

| File Type | Description                                |
|-----------|--------------------------------------------|
| `.png`    | Final high-resolution image                |
| `.gif`    | Animated zoom preview                      |
| `.json`   | Metadata including quantum seed + settings |

---

## 🧠 Philosophy

Each piece of art is:

- **Unique** — seeded by quantum measurement data
- **Reproducible** — exact outputs via saved seeds and config
- **Interpretable** — artistic styles inspired by quantum theory

---

## 🔮 Planned Features

- NFT minting integration
- Custom palette editor
- Sound-reactive visualization
- Procedural audio generation from seeds

---

## 👨‍💻 Author

Built with chaotic precision by usernamesRdumb.
Pull requests welcome (unless you're a chud).

---

## 📄 License

Licensed under the MIT License
