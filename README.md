# Rapid-Porkchop

GPU-accelerated Lambert solver for interplanetary trajectory design (porkchop-style analysis).

This project implements the Lambert problem (two-point boundary value problem) as presented in *Orbital Mechanics for Engineering Students* by Howard Curtis, and parallelizes the lambert solver loop on NVIDIA GPUs using Numba-CUDA.

## Requirements

- Python **3.8 – 3.13**
- Numpy
- Matplotlib
- Numba
- An **NVIDIA GPU** with **CUDA 12 or newer**
- CUDAtoolkit

---

## Installation (Conda)

Conda is recommended for managing Python and CUDA-related dependencies.

```powershell
conda install numpy
conda install matplotlib
conda install -c conda-forge numba-cuda "cuda-version=12"
```

---

## Usage

1. Open `src/` and edit `startPorkchop.py`.
2. Set:
   - Departure dates / arrival dates
   - Number of points
   - Departure planet ID / arrival planet ID
3. Note: Asteroid **257323** is mapped to planet ID **11** for convenience.

Run:

```powershell
python startPorkchop.py
```

The output image should appear in the same directory as the run file.

<img src="/src/outputs-example/Jupiter-Asteroid-MegaZoom.png" width="500" alt="Diagram">

