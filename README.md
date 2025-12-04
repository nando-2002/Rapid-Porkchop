# Rapid-Porkchop

Python implementation of the Lambert problem (interplanetary trajectory planning) solved with the Universal Variables formulation.

## Overview

Rapid-Porkchop solves the two-point boundary value Lambert problem:
given two position vectors (departure and arrival) and a time-of-flight (TOF),
compute the feasible transfer orbit(s) (velocity vectors) connecting them under a central inverse-square gravity field.

This implementation uses the Universal Variables approach for robust handling of elliptic, parabolic and hyperbolic cases.

## Features

- Universal-variables Lambert solver supporting multi-revolution and single-revolution transfers
- Handles elliptical, parabolic, and hyperbolic geometries
- Returns both prograde and retrograde solutions when they exist
- Lightweight; minimal dependencies (NumPy/SciPy)
- Example scripts for travel-time maps and porkchop plots (if present in `examples/`)

## Quick Start

1. Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt