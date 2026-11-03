# Clock grid correction

Candidates now use the union of reported graphics-clock lists. Idle memory states no longer truncate the grid. Candidates are not proof of lock support; loaded readback calibration and successful reset remain required.

Host regression: 194 tests, 179 passed, 15 skipped (12 Linux native, 3 CUDA). No new physical GPU execution. Dummy TP2 algorithm execution remains unimplemented.
