# SHMQ-Ultimate

**Experimental reimplementation of SHMQ (Sparse-Hessian Mixed-precision Quantization) for LLMs**, combining components from several sources:

| Source | Component | Role |
|--------|-----------|------|
| **HAWQ-V3** | PyHessian + ILP solver (PULP) | Inter-layer sensitivity + optimal bit allocation |
| **SliM-LLM** | GPTQ/OBS + SQC references | Calibration and quantization references |
| **SHMQ paper** | Decoupled permutation + RMSNorm fusion + parallel constraint | Core SHMQ algorithm |
| **AutoRound** | SignSGD learnable rounding | Per-weight rounding optimization (200 steps) |
| **SmoothQuant** | Activation outlier migration | Pre-processing for A8 stability |

**Format**: Experimental three-level `{4, 8, 16}` weights. The current fused kernel uses FP16 activations; it is not a W4.8A8 implementation.

**Target**: reproducible Qwen benchmark on a pinned GPU environment. Accuracy, throughput, and end-to-end vLLM compatibility are not established by this repository yet.

> **Important architecture limitation:** SHMQ's permutation selects input
> channels (`Cin`, weight columns), while the current fused kernel partitions
> output rows (`Cout`). The adapter now rejects SHMQ input-axis cluster sizes
> instead of silently projecting them onto output rows. A faithful SHMQ result
> therefore requires a `Cin`-partitioned kernel or a separately documented
> output-channel algorithm.

---

## What's Inside

### Code: 5,710 lines across 38 files

```
shmq-ultimate/
├── src/shmq/                          # Main package (3,506 lines)
│   ├── config.py                      # SHMQConfig dataclass
│   ├── pipeline.py                    # 9-step orchestrator (475 lines)
│   ├── model_loader.py                # HuggingFace model + LayerInfo
│   ├── calibration.py                 # WikiText/Pile calibration data
│   ├── utils.py                       # Helpers (get_module_by_name, etc.)
│   ├── smooth/                        # Step 1: SmoothQuant
│   │   ├── smooth.py                  #   Activation scale capture + smoothing
│   │   └── calibration.py             #   Scale calibration
│   ├── sensitivity/                   # Step 2: Sensitivity computation
│   │   ├── fisher.py                  #   Inter-layer Fisher H ≈ (1/|D|)Σ g·gᵀ
│   │   ├── pyhessian_trace.py         #   Alt: PyHessian trace (from HAWQ-V3)
│   │   ├── obs.py                     #   Intra-layer OBS (GPTQ Hessian)
│   │   ├── manhattan.py               #   Manhattan norm aggregation
│   │   └── parallel.py                #   Parallel constraint (q/k/v same bits)
│   ├── ilp/                           # Step 3: ILP bit allocation
│   │   └── solver.py                  #   PULP solver (2 levels {4,8})
│   ├── permutation/                   # Steps 4-5: Decoupled permutation
│   │   ├── metric.py                  #   Permutation metric (act × weight l∞)
│   │   ├── decoupled.py               #   Sort→partition→sort (SHMQ Eq.12)
│   │   └── rmsnorm_fusion.py          #   PermutedRMSNorm wrapper
│   ├── autoround/                     # Step 6: AutoRound (Intel)
│   │   ├── sign_sgd.py                #   SignSGD optimizer
│   │   ├── autoround_block.py         #   200-step per-block optimization
│   │   ├── wrapper.py                 #   Learnable V rounding wrapper
│   │   └── baking.py                  #   Bake V into weights (zero overhead)
│   ├── quantize/                      # Steps 7-8: SQC + GPTQ
│   │   ├── sqc.py                     #   Salience-Weighted Quantizer Calibration
│   │   ├── gptq.py                    #   GPTQ (OBS) per-element Hessian
│   │   └── mixed.py                   #   Mixed INT4/INT8 dispatcher
│   └── inference/                     # Step 9: REAL INT4/INT8 inference
│       ├── shmq_matmul_kernel.cu      #   CUSTOM CUDA kernel (353 lines) ★
│       ├── kernel_loader.py           #   JIT compile + CPU fallback
│       ├── weight_packing.py          #   INT4 pack/unpack + per-group scales
│       ├── shmq_quant_linear.py       #   SHMQQuantLinear nn.Module
│       └── model_converter.py         #   Replace nn.Linear → SHMQQuantLinear
├── tests/                             # 37 tests (26 unit + 11 E2E)
│   ├── test_smoke.py                  #   15 unit tests for each component
│   ├── test_real_int4_inference.py    #   11 tests for INT4 packing + kernel
│   └── test_e2e_pytest.py             #   11 E2E tests (Qwen2.5-0.5B, 2 blocks)
├── scripts/gpu/                       # GPU deployment scripts
│   ├── setup_gpu.sh                   #   Full environment setup
│   ├── build_cuda_kernel.py           #   Compile + verify CUDA kernel
│   ├── benchmark_qwen7b.py            #   Full pipeline on Qwen2.5-7B-Instruct
│   ├── eval_perplexity.py             #   WikiText-2 perplexity
│   └── eval_zeroshot.py               #   HellaSwag/ARC/PIQA zero-shot
├── configs/                           # Preset configurations
│   ├── qwen7b_paper.json              #   Paper defaults (128 samples × 2048)
│   └── quick_test.json                #   Quick CPU test (4 samples × 128)
├── external/                          # Cloned source repos (reference)
│   ├── HAWQ-V3/                       #   PyHessian + ILP source
│   ├── SliM-LLM/                      #   AutoGPTQ + Marlin source
│   ├── AutoRound/                     #   SignSGD source
│   └── SmoothQuant/                   #   Outlier migration source
└── paper/
    ├── shmq_paper.pdf                 #   SHMQ paper
    └── shmq_paper.txt                 #   Extracted text
```

### CUDA Kernel

`src/shmq/inference/shmq_3level_kernel.py` implements an experimental CuPy
`RawKernel` path for contiguous output-row regions:

- One launch combines FP16, INT8, and INT4 output-row regions.
- INT4 values are packed as signed two's-complement nibbles with per-group scales.
- The CUDA path uses scalar CUDA-core accumulation; it does not use `mma.sync`,
  and is not a verified tensor-core or MixLLM performance path.
- A PyTorch reference path exists for contract tests.

The main architectural limitation is important: SHMQ's permutation selects input
channels (`Cin`, weight columns), while this fused kernel partitions output rows
(`Cout`). The adapter rejects input-axis cluster metadata. The
`SHMQKPartitionLinear` path preserves input-axis semantics and is a correctness
reference, but it is not a production vLLM kernel.

---

## Quick Start

### Option A: CPU contract tests (requires PyTorch and pytest)

```bash
cd shmq-ultimate
pip install torch transformers datasets pulp scipy numpy
python -m pytest tests/ -v
```

The test count is intentionally not hard-coded here; run pytest to see the
current collection. Tests requiring optional model downloads or CUDA may be
skipped or require additional dependencies.

### Option B: Strict CUDA kernel gate (T4)

```bash
cd shmq-ultimate

python ..\scripts\build_kaggle_gpu_gate.py
# Upload shmq-ultimate/kaggle_gpu_gate/shmq_gpu_gate.ipynb to a T4 Kaggle session.
# The gate fails closed if CUDA, CuPy, NVRTC, or the RawKernel path is unavailable.
```

### Option C: Step-by-step manual run

```python
from shmq.config import SHMQConfig
from shmq.pipeline import SHMQPipeline

config = SHMQConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device="cuda", dtype="float16",
    n_samples=128, sequence_length=2048,
    target_hp_ratio=0.20, base_hp_ratio=0.125,
    enable_autoround=True, autoround_iters=200,
    enable_sqc=True,
)
pipeline = SHMQPipeline(config)
pipeline.run()                           # Experimental pipeline
pipeline.save_model("./download/qwen7b_shmq")
```

---

## The Experimental Pipeline

| Step | Module | What it does | Time (7B, A100) |
|------|--------|--------------|-----------------|
| 0 | `model_loader` | Load Qwen2.5-7B + 128×2048 calibration tokens | 30s |
| 1 | `smooth/smooth` | SmoothQuant: migrate activation outliers to weights | 15s |
| 2 | `sensitivity/fisher` + `obs` | Inter-layer Fisher H + intra-layer OBS Hessian | 416s |
| 3 | `ilp/solver` | ILP bit allocation (the active solver must be checked for `{4,8,16}` support) | <1s |
| 4 | `permutation/decoupled` | Sort→partition→sort by magnitude (SHMQ Eq.12) | 60s |
| 5 | `permutation/rmsnorm_fusion` | Fuse permutation into RMSNorm (zero overhead) | 1s |
| 6 | `autoround` | 200-step SignSGD learnable rounding per block | 480s |
| 7 | `quantize/sqc` | SQC scale calibration (salience-weighted) | 120s |
| 8 | `quantize/gptq` + `mixed` | GPTQ + mixed INT4/INT8 fake-quant | 180s |
| 9+ | `mixllm/adapter` + `inference/shmq_3level_kernel` | Experimental output-row `{16,8,4}` adapter and CUDA reference path | unverified |

The timings and total are historical estimates, not a benchmark result for the
current implementation. T4 execution, memory use, and model quality must be
measured independently.

---

## Configuration Parameters (from SHMQ paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| Format | W4.8A8 in the paper; current fused path uses FP16 activations | SHMQ §4 / implementation |
| Inter-layer Hessian | Fisher `H ≈ (1/|D|)Σ g·gᵀ` | SHMQ Eq.6, App A.2 |
| Intra-layer sensitivity | `S_{i,j} = ½ · h_{i,j} · (w-Q(w))²` | SHMQ Eq.5 |
| Base high-precision ratio (UB) | 12.5% | SHMQ §4 |
| Dampening factor (λ) | 0.1 | SHMQ §4 |
| Permutation metric | activations × weights l∞ norm | SHMQ §3.2.3 |
| Permutation approach | Decoupled (identify → sort by magnitude) | SHMQ §3.2.3 |
| Fusion | q/k/v → RMSNorm; up/gate → prior activation | SHMQ §3.2 |
| Parallel constraint | q/k/v same bits; up/gate same bits | SHMQ Eq.4 |
| Calibration | 128 samples × 2048 tokens | SHMQ §4 |
| AutoRound iters | 200 | AutoRound paper |
| Group size | 128 | SliM-LLM |

---

## What Makes This "Ultimate" (vs original SHMQ)

| Feature | Original SHMQ | SHMQ-Ultimate |
|---------|---------------|---------------|
| Bit allocation | Proportion mapping (Eq.8) | **ILP** (PULP) — mathematically optimal |
| Rounding | Round-to-Nearest | **AutoRound** SignSGD (200 steps) |
| Activation outliers | Not addressed | **SmoothQuant** pre-processing |
| Scale calibration | Basic | **SQC** salience-weighted |
| Inter-layer Hessian | Fisher only | Fisher **+ PyHessian** (switchable) |
| Sensitivity backend | Custom | **GPTQ/OBS** from SliM-LLM (Frantar 2023) |

Expected: SHMQ-Ultimate should **match or slightly exceed** original SHMQ.

---

## Testing

### Test status

Run `python -m pytest tests/ -v` in an environment containing the declared
dependencies. Do not treat a CPU reference pass as validation of the CUDA
kernel, Qwen-3 7B quality, throughput, or vLLM integration.

### What's tested

- **Component tests** (`test_smoke.py`): Every module (Fisher, OBS, ILP, permutation, RMSNorm fusion, AutoRound, SQC, GPTQ, mixed quantizer) has a dedicated test.
- **INT4 packing tests** (`test_real_int4_inference.py`): Verifies `pack_int4`/`unpack_int4` roundtrip, INT8 quantization, SHMQ matmul correctness vs fake-quant reference, model converter swaps Linear → SHMQQuantLinear, forward pass produces valid output.
- **E2E tests** (`test_e2e_pytest.py`): Full 9-step pipeline on Qwen2.5-0.5B (2 blocks), verifying each step's output, memory compression (3.21×), and real INT4 inference.

### What requires a GPU (cannot test here)

- CuPy/NVRTC CUDA kernel compilation and execution (use the strict T4 gate)
- CUDA kernel correctness (needs CUDA-capable GPU)
- Inference speedup measurement (needs GPU)
- Qwen2.5-7B-Instruct evaluation (needs ≥24GB VRAM)

- Run the generated `kaggle_gpu_gate/shmq_gpu_gate.ipynb` on an actual T4 to
  verify the implemented scalar RawKernel. This does not establish tensor-core
  speedup or production vLLM compatibility.

---

## Repository Sources

| Repo | URL | Used for |
|------|-----|---------|
| HAWQ-V3 | https://github.com/Zhen-Dong/HAWQ | PyHessian, ILP solver |
| SliM-LLM | https://github.com/Aaronhuang-778/SliM-LLM | AutoGPTQ, Marlin, SQC |
| AutoRound | https://github.com/intel/auto-round | SignSGD learnable rounding |
| SmoothQuant | https://github.com/mit-han-lab/smoothquant | Activation outlier migration |
| SHMQ paper | https://aclanthology.org/2025.emnlp-industry.175/ | Algorithm specification |

---

## Honest Status Disclosure

**What is implemented:**
- ✅ CPU packing/reference contract tests and a generated Kaggle GPU gate.
- ✅ Fail-fast validation preventing Cin cluster metadata from being treated as Cout.
- ✅ Experimental three-level kernel and prototype vLLM patch.

**What requires a GPU machine to verify:**
- ⚠️ CUDA kernel compilation (can't compile without `nvcc` + GPU)
- ⚠️ CUDA kernel correctness (can't run without CUDA)
- ⚠️ Any speedup measurement (needs GPU and a valid faithful artifact)
- ⚠️ Qwen2.5-7B-Instruct results (needs ≥24GB VRAM)
- ⚠️ 0.13% accuracy gap (needs full calibration + eval)

To complete verification, implement and validate the Cin-partitioned kernel,
then run the Kaggle gate and full benchmarks on a pinned T4 environment.
