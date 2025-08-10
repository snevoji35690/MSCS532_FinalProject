# MSCS532_FinalProject
# AoS vs SoA Benchmark – Data Layout Optimization in HPC

## Overview
This project demonstrates the performance impact of changing data layout
from **Array of Structures (AoS)** to **Structure of Arrays (SoA)** in a
simple high-performance computing (HPC) particle update kernel.

The transformation improves:
- **Cache locality** (unit-stride memory access)
- **Vectorization/SIMD efficiency**
- **Overall runtime for large datasets**

The benchmark is implemented in **Python** using:
- **NumPy** for numerical arrays
- **Numba** for JIT compilation, parallel loops, and vectorization

## Requirements
- Python 3.9 or later
- NumPy
- Numba

Install dependencies:
pip install numpy numba

Usage
Run the benchmark:

python aos_soa_benchmark.py
Optional: Change problem size or parameters by editing these variables in main():
N  = 5_000_000  # number of particles
dt = 1e-3       # time step
k  = 0.7        # spring constant

Expected Output
Example run:

AoS step: 0.210 s

SoA step: 0.085 s

Speedup: 2.47×

Actual times vary depending on your CPU, number of threads, and Numba version.

Project Structure

aos_soa_benchmark.py   # Benchmark source code

README.md              # This file
