# v292 Deep Research: Colab Gate Parity

## Question

Can Colab be the primary environment for the same work currently associated with the Kaggle gate, rather than running only the 91 source and Python contracts?

## Comparison with the original MixLLM organization

The upstream MixLLM gate organization separates source provisioning, hardware detection, operator correctness, and measured native benchmarks. Its benchmark path constructs packed three-level tensors, calls the production SM75 backend, and measures native operator and dense FP16 reference timings with CUDA events. The current SHMQ `colab_v284_gpu_check.py` does not execute that benchmark path: it clones the repository, compiles the extension through the tests, and runs the seven unittest modules only. Therefore its `COLAB_V291_SM75_CHECK_PASS` marker proves compile/correctness contracts but cannot establish a performance baseline.

The gate notebook itself contains the missing cells. Cell 8 runs the same `benchmark_sm75_backend` scenarios for smoke mixed precision, Qwen-shaped mixed 4/8/16 partitions, pure INT4, pure INT8, and pure FP16. Cell 9 applies the unchanged correctness, decode, prefill, timing-integrity, and production thresholds. Cell 7 adds the full Qwen2.5-0.5B quality gate when the exact model files are available. Cell 5 applies the pinned vLLM 0.9.0 patch and runs its smoke path.

## Finding

The next safe iteration is not an algorithmic optimization. It is a validation-parity correction: make the official Colab CLI runner execute the exact gate notebook cells on a fresh T4, provide the Qwen model through a deterministic `/kaggle/input` compatibility mount, and preserve the notebook’s model, data, precision, thresholds, and scenario shapes. This prevents us from calling contract-only Colab results a performance baseline and gives the next kernel optimization a trustworthy feedback loop.

## Safety boundary

No benchmark setting, model architecture, quality threshold, precision, or arithmetic is changed. The only environment adaptation is mapping Colab’s local filesystem and dependency availability to the paths expected by the existing gate notebook. Kaggle is not launched in this iteration.
