# Eq.(2) existing-window identifiability audit

Paper: https://arxiv.org/html/2603.21908v1 section 4.1.2, Eq.(2):
P = alpha(S_comp) V^2 f + (k1 T + k2) V.

The audit integrates the three basis terms over each Windows host envelope,
using separate query midpoint timestamps for voltage, temperature and frequency.
It does not replace mean(V^2 f) with mean(V)^2 mean(f).
Frequency is converted from MHz to Hz; temperature uses the existing project's
Celsius convention. NVML temperature is not independently certified junction temperature.

Implementation choices (not paper parameters): exactly reference-1/2/3 train;
pilot excluded; holdout-1/2/3 only read after a training decision is saved.
Column L2 scaling, numerical SVD rank, condition <=1000; energy-budget corner
perturbations must change every coefficient by at most 20%. Negative alpha/k1
or leakage in the observed range is rejected, never clipped or regularized.
These conservative diagnostic gates do not establish statistical confidence.
Three independent training windows and three coefficients give zero residual
degrees of freedom. Sampling points are not independent training observations.

Constant activity over a mixed inference window is an extra assumption.
Whole-board energy includes memory and other domains; reported core voltage
does not identify all those domains. Sensor update lag and feature uncertainty
are unknown; energy-budget perturbations are only a partial sensitivity check.
Even a passing solve/holdout remains conditional board-level Eq.(2)-basis
regression, not identified operator alpha/k1/k2 or a DVFS certificate.
Holdout predictions use measured V/T/f/duration, not future-known inputs.
Existing window and latency models are never changed by this tool.

## Minimal supplemental diagnostic

Two additional windows request 2527 then 2250 MHz via existing bridge sessions.
This is explicitly not the fixed-frequency acceptance path: actual synchronous
V/f readings form the features; no requested frequency is used as a model label.
Each card must exhibit a loaded median at least 100 MHz below baseline, below
the requested ceiling plus 15 MHz, and the second condition at least 100 MHz
below the first. These are excitation checks, not exact-clock certificates.
Both settings use the same 512/64, 20-request workload. Each has a workload
warmup outside its measured host envelope. All samples remain in raw evidence.
Reset and loaded restoration must pass before supplemental data can be audited.
The caller must acknowledge restoration to default AUTO policy; an arbitrary
preexisting manual lock is not preserved. No power limit or memory clock changes.

With --extra, all five training windows are used (2 residual degrees of freedom),
and all 32 energy-budget corners are checked. Thresholds are unchanged. Old
holdouts remain original-policy-only and previously viewed; they do not validate
new frequency conditions, which would require fresh held-out measurements.
