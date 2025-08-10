import math, time
import numpy as np
from numba import njit, prange

# ----------------------------
# Data generation
# ----------------------------
def make_aos(N):
    # NumPy structured dtype (AoS-like)
    dtype = np.dtype([('x','f8'),('y','f8'),('z','f8'),
                      ('vx','f8'),('vy','f8'),('vz','f8'),
                      ('m','f8')])
    aos = np.zeros(N, dtype=dtype)
    rng = np.random.default_rng(42)
    aos['x'] = rng.standard_normal(N)
    aos['y'] = rng.standard_normal(N)
    aos['z'] = rng.standard_normal(N)
    aos['vx'] = rng.standard_normal(N)*0.1
    aos['vy'] = rng.standard_normal(N)*0.1
    aos['vz'] = rng.standard_normal(N)*0.1
    aos['m']  = rng.uniform(0.5, 2.0, N)
    return aos

def make_soa(N):
    rng = np.random.default_rng(42)
    x  = rng.standard_normal(N)
    y  = rng.standard_normal(N)
    z  = rng.standard_normal(N)
    vx = rng.standard_normal(N)*0.1
    vy = rng.standard_normal(N)*0.1
    vz = rng.standard_normal(N)*0.1
    m  = rng.uniform(0.5, 2.0, N)
    return x,y,z,vx,vy,vz,m

# ----------------------------
# AoS kernel (Numba)
# ----------------------------
@njit(parallel=True, fastmath=True)
def step_aos(aos, dt, k):
    N = aos.shape[0]
    for i in prange(N):
        # spring-like force towards origin: F = -k * r
        fx = -k * aos['x'][i]
        fy = -k * aos['y'][i]
        fz = -k * aos['z'][i]
        invm = 1.0 / aos['m'][i]
        aos['vx'][i] += fx * invm * dt
        aos['vy'][i] += fy * invm * dt
        aos['vz'][i] += fz * invm * dt
        aos['x'][i]  += aos['vx'][i] * dt
        aos['y'][i]  += aos['vy'][i] * dt
        aos['z'][i]  += aos['vz'][i] * dt

# ----------------------------
# SoA kernel (Numba)
# ----------------------------
@njit(parallel=True, fastmath=True)
def step_soa(x,y,z,vx,vy,vz,m, dt, k):
    N = x.shape[0]
    for i in prange(N):
        fx = -k * x[i]
        fy = -k * y[i]
        fz = -k * z[i]
        invm = 1.0 / m[i]
        vx[i] += fx * invm * dt
        vy[i] += fy * invm * dt
        vz[i] += fz * invm * dt
        x[i]  += vx[i] * dt
        y[i]  += vy[i] * dt
        z[i]  += vz[i] * dt

# ----------------------------
# Benchmark harness
# ----------------------------
def benchmark():
    N  = 5_000_000  # adjust for your machine
    dt = 1e-3
    k  = 0.7

    print(f"N={N}")
    aos = make_aos(N)
    x,y,z,vx,vy,vz,m = make_soa(N)

    # warmup JIT
    step_aos(aos, dt, k)
    step_soa(x,y,z,vx,vy,vz,m, dt, k)

    # time AoS
    t0 = time.perf_counter()
    step_aos(aos, dt, k)
    t1 = time.perf_counter()

    # time SoA
    t2 = time.perf_counter()
    step_soa(x,y,z,vx,vy,vz,m, dt, k)
    t3 = time.perf_counter()

    print(f"AoS step: {t1 - t0:.3f} s")
    print(f"SoA step: {t3 - t2:.3f} s")
    print(f"Speedup (AoS/SoA): {(t1 - t0)/(t3 - t2):.2f}×")

if __name__ == "__main__":
    benchmark()
