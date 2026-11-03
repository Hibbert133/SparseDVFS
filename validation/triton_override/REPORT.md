# Independent Triton override validation

Implementation: b3a6d01 (explicit existing Triton 3.6.0 path; no package installation).

Host Python/mock suite: 193 total, 178 passed, 0 failures, 15 skipped (12 Linux-native and 3 CUDA). Five new import/path/metadata/hash/subprocess tests passed. Fixtures do not execute real Triton kernels or vLLM. No new QEMU run was performed for this update. Earlier native/CI evidence in validation/conda applies to its recorded commit, not this update. Target GPU compatibility and savings remain unverified.

See summary.json for exact tested source hashes and tests.txt for the test list.
