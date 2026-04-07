# Parameters & Configuration

## Flight Conditions

Passed via the orchestrator CLI or as `flight_conditions` dict:

| Parameter | CLI Flag | Default | Units |
|-----------|----------|---------|-------|
| Mach number | `--mach` | 0.78 | — |
| Angle of attack | `--aoa` | 2.0 | degrees |
| Altitude | `--altitude` | 35000 | feet |

## Mission Profile

| Parameter | Default | Units | Notes |
|-----------|---------|-------|-------|
| Range | 1500 | nmi | Aviary default |
| Passengers | 162 | — | Used for Aviary payload |
| Cruise Mach | 0.78 | — | From flight conditions |
| Cruise altitude | 35000 | ft | From flight conditions |
| Optimizer max iter | 50 | — | Aviary SNOPT iterations |

## Dependency Versions

> **Critical pinning** for Aviary compatibility.

| Package | Required Version | Notes |
|---------|-----------------|-------|
| `openmdao` | 3.36.0 | Aviary 0.9.10 hard requirement |
| `dymos` | 1.13.1 | Aviary 0.9.10 hard requirement |
| `aviary` | 0.9.10 | Latest stable |
| `numpy` | >= 1.26 | Required by all |
| `pyCycle` | latest | Works with openmdao 3.36.0 |

## SU2 Configuration

The SU2 adapter generates a config file with:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MATH_PROBLEM` | DIRECT | Euler simulation |
| `NUM_METHOD_GRAD` | GREEN_GAUSS | Gradient method |
| `CONV_NUM_METHOD_FLOW` | JST | Convective flux scheme |
| `ITER` | 250 | Max iterations |
| `CFL_NUMBER` | 10.0 | CFL for stability |

## pyCycle Engine Model

Default high-bypass turbofan parameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Fan PR | 1.5 | Fan pressure ratio |
| LPC PR | 3.0 | Low-pressure compressor |
| HPC PR | 14.0 | High-pressure compressor |
| T4 max | 2900 R | Turbine inlet temperature |
| BPR | ~5.0 | Bypass ratio (solved) |
