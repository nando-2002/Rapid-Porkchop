# Rapid-Porkchop

This is an implementation of the lambert problem for interplanetary spaceflight, taken from "Orbital Mechanics for Engineering Students" by Howard Curtis. Since this is a boundary value problem, it has been converted to run in parallel on Nvidia GPUs, using Numba - CUDA, an easy to use Python frontend for the CUDA programming language. 

## Overview

Rapid-Porkchop solves the two-point boundary value Lambert problem:
given two position vectors (departure and arrival) and a time-of-flight (TOF),
compute the feasible transfer orbit(s) (velocity vectors).

This implementation uses the Universal Variables approach to solve the Lambert problem.

## Quick Start

Install Python (3.8 - 3.13) and requirements. It is recommended to install Python with the 'conda' package manager

```powershell
conda install numpy
conda install matplotlib
conda install -c conda-forge numba-cuda "cuda-version=12"