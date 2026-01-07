# Rapid-Porkchop

GPU-accelerated Lambert solver for interplanetary trajectory design (porkchop-style analysis).

This project implements the Lambert problem (two-point boundary value problem) as presented in *Orbital Mechanics for Engineering Students* by Howard Curtis, and parallelizes key computations on NVIDIA GPUs using Numba-CUDA.

---

## What it does

Rapid-Porkchop solves the **Lambert problem**:

Given two position vectors (departure and arrival) and a time-of-flight (TOF), compute the feasible transfer orbit(s) (velocity vectors).

This implementation uses the **Universal Variables** approach.

---

## Requirements

- Python **3.8 – 3.13**
- An **NVIDIA GPU** with **CUDA version 12 or newer**

---

## Quick start (Conda)

It is recommended to use Conda for managing Python and CUDA-related dependencies.

```powershell
conda install numpy
conda install matplotlib
conda install -c conda-forge numba-cuda "cuda-version=12"
