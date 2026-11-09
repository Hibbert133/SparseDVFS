# Frozen offline stage to selection and partition: 2026-11-09

Branch: 5090_eager. Target remains the user's separate Windows/WSL machine.
This increment is offline only and does not assume the Nov 6 service is alive.

## Paper and existing paths

Source: https://arxiv.org/html/2603.21908v1 section 4.2 Algorithm 1 and Eq.(3).
The algorithm merges if the current block is short OR frequencies are similar,
takes the componentwise higher frequency, and appends the final block even when
short. Algorithm 1 uses `<` while Eq.(3) requires strict `>`: equality and the
short tail are retained and explicitly reported, not silently reinterpreted.
N=5 follows the paper's example; a 5% similarity rule and energy-minimizing
selector subject to a 5% latency constraint are explicit project choices.

`sparsedvfs/paper_core.py:greedy_partition` already implements those literal
semantics and validates topological order. The frequency factory supports a
GPU-pair adapter without inventing CPU/EMC placeholders.

`rtx5090/eager_model.py:plan` requires same-path unit models and real validation,
coverage, and measured switching evidence. The current latency table and
board-window Eq2 candidate do not satisfy that contract. Its strict input gates
remain intact. This change preserves partition decisions and Eq3 outcomes in
the generated output; previously the returned decision list was discarded.

`rtx5090/planning.py` is the historical module adapter, using placeholder
CPU/EMC values internally and latency-selected rates pending energy selection.
It is not the new TP2 path and is not used for the present artifacts.

## Delivered chain and boundaries

`rtx5090.stage_freeze_plan` checks the identity, model/evidence digests, repaired
holdout provenance and restoration results. It copies frozen models into
`stage-freeze.json` with byte hashes and paths of all consumed evidence. It
leaves existing source artifacts unchanged. It does not claim live identity
or current hardware restoration from historical evidence.

Whole-window selection uses ONLY original training windows: minimum board
energy among requested settings whose mean HTTP request latency is within 5%
of the original current-policy reference. It reports held-out comparisons
without changing the choice. This analysis is retrospective, the prompts are
not paired between conditions, and the reference was not certified AUTO.
One training/holdout window per reduced setting is not a benefit confidence
interval. This screen is not Eq1/Eq2 sparse operator selection.

`rtx5090.partition_dryrun` accepts complete paired latency/energy predictions
at both rates plus a switching time. It runs selection, the existing Algorithm
1, order/coverage checks, Eq3 flags and a serial timing estimate. It does not
emit an eager executable plan. All outputs are non-deployable, including when
inputs are supplied. Unknown initial/final switching, coordination/readback,
HTTP/scheduler costs and switching energy remain explicit.

The real 68-unit observation order produces a one-context input template
with missing frequency-dependent costs marked null. No kernel DAG is inferred
from the module order; each prefill/decode context needs its own cost table.
The fixed-shape layer-0 linear timings cannot fill other layers/units or rates.
Whole-board request energy is never divided among units. Missing switching
evidence is not replaced with a guessed hardware bound.

A separate clearly synthetic example exercises nontrivial OR merging,
conservative high-frequency merging, multiple blocks and a short tail. Its
success establishes software behavior only, not hardware validity.

## Remaining minimum evidence before actual dynamic planning

1. Frequency-dependent latency and energy predictions for every intended
   controllable unit/context, or a smaller explicitly instrumented boundary
   set with equivalent evidence; actual V/f and model scope must agree.
2. Same-service dual-rank boundary protocol and measured switching bound with
   loaded readback/restoration. Existing diagnostic frequency shifts are not
   exact-clock acceptance or a switching-latency measurement.
3. Independent checks of modeled serial costs and end-to-end <=5% vs a verified
   reference. CPU/EMC and look-ahead remain unsupported/unverified.

Do not bypass those gates to make current board-window candidates deployable.
