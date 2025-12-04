# Rapid-Porkchop

Python implementation of the Lambert problem (interplanetary trajectory planning) solved with the Universal Variables formulation.

## Overview

Rapid-Porkchop solves the two-point boundary value Lambert problem:
given two position vectors (departure and arrival) and a time-of-flight (TOF),
compute the feasible transfer orbit(s) (velocity vectors).

This implementation uses the Universal Variables approach to solve the Lambert problem.

As of 04/12/2025 there is nothing "rapid" about Rapid-Porkchop. That will come later.

## Quick Start

1. Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt