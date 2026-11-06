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
