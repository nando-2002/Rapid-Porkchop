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

## Installation
Install Python, preferably through [Anaconda](https://www.anaconda.com/docs/getting-started/miniconda/install) so that you get the conda package manager.

Clone the repo
```powershell
git clone https://github.com/nando-2002/Rapid-Porkchop.git
```
### Package Installs

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
