# FPGA Acceleration of Quantized Neural Networks

Lab coursework from **Neural Network Acceleration on FPGAs**, Chair of Dependable
Nano Computing (CDNC), Karlsruhe Institute of Technology (KIT) — M.Sc. coursework,
2026.

This repository documents a full pipeline for deploying quantized neural networks
on FPGA hardware: quantization-aware training with Brevitas, hand-written HDL
neuron implementations, FINN-based dataflow accelerator generation and folding
optimization, and end-to-end deployment/evaluation on a PYNQ-Z2 board.

> Instructors: Vincent Meyers, Johannes Reibold

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| PyTorch | 2.8.0 | Model definition & training |
| Brevitas | 0.12.1 | Quantization-aware training (QAT) |
| QONNX | 1.0.0 | Quantized ONNX IR / export format |
| ONNX Runtime | 1.20.1 | Functional model verification |
| FINN(+) | 1.4.0 | Dataflow HLS/RTL accelerator compiler |
| Vivado / Vitis | 2022.1 | FPGA synthesis, implementation, bitstream generation |
| GHDL | — | VHDL simulation (Task 2) |
| PYNQ-Z2 | — | Target deployment board (Zynq-7020 SoC) |

Full environment setup (conda env, exact package versions, Vivado install
notes) is documented in [`task0/`](./task0).

---

## Repository Structure
```
├── task0/ # Environment setup: conda, Brevitas, Vivado
│ └── README.md
├── task1/ # Quantization-aware training: keyword spotting
│ ├── code/ # training/eval scripts + Speech Commands data
│ ├── results/ # accuracy/latency plots, model_comparison.csv
│ ├── dump/
│ └── README.md
├── task2/ # Hand-written VHDL MLP accelerator (Iris dataset)
│ ├── mlp/ # neuron.vhdl, layer.vhdl, neuralnetwork.vhdl, ...
│ ├── dump/
│ └── README.md
├── task3/ # FINN folding optimization (TFC-W1A1 MNIST MLP)
│ ├── models/ # tfc-w1a1(-flattened)/w1a2/w2a2 .onnx
│ ├── results/ # output_auto/, output_custom_0..3/
│ ├── custom_0..3_folding_config.json
│ ├── build.yaml
│ └── README.md
├── task4/ # GTSRB traffic sign classifier on PYNQ-Z2
│ ├── task4/ # training + FINN build working directory
│ ├── task4-Pynq-Z2/ # self-contained on-board deployment package
│ ├── dump/
│ └── README.md
└── README.md
```
Each `taskN/` is self-contained with its own `README.md` covering setup,
usage, and results in detail — this top-level README is a summary and
index.

---

## Task Summaries

### Task 0 — Environment Setup
Conda environment (`nnlab`, Python 3.11) with PyTorch, Brevitas, QONNX,
ONNX Runtime, and `finn-plus`; Vivado/Vitis 2022.1 WebPack installation and
verification against the reference `tfc-w1a1`/`tfc-w2a2` models.

### Task 1 — Quantization-Aware Training: Keyword Spotting
Trained an 11-class keyword-spotting CNN (32→64→64 channels, spectrogram
input) on Google Speech Commands, first in FP32 and then via QAT at three
bit-width configurations (8w8a, 4w4a, 2w4a) using Brevitas.

| Config | Test Accuracy | BOPS | Weight Bits |
|---|---|---|---|
| FP32 baseline | 93.79% | 3.52 B | 8.1 M |
| **8w8a (best)** | **94.28%** | 219.9 M (**16× ↓**) | 2.0 M (**4× ↓**) |
| 4w4a | 88.72% | 55.0 M | 1.0 M |
| 2w4a | 85.15% | 27.5 M | 0.5 M |

Also includes an investigation into a PyTorch/QONNX evaluation
discrepancy at 2w4a (58.29% vs. 85.15% on the same checkpoint), traced to
an evaluation-script bug rather than a real accuracy drop, and an analysis
of why CPU latency *increases* under quantization despite a ~130× BOPS
reduction — see [`task1/README.md`](./task1/README.md).

### Task 2 — Hand-Written VHDL MLP Accelerator
Implemented a fully-connected binary/bipolar-weight neural network
(4 → 8 → 8 → 3 neurons, Iris dataset) directly in VHDL: FSM-based neuron
datapath (idle/mult/sum/act states, single-cycle XNOR+popcount MAC for
binary weights), layer instantiation, and network-level control/argmax
logic — no HLS or automated flow, to build intuition for what tools like
FINN generate under the hood. Simulated with GHDL; predictions verified
against the task's expected output labels. See
[`task2/README.md`](./task2/README.md).

### Task 3 — FINN Folding Optimization
Took a pre-trained MNIST MLP (`tfc-w1a1`, 784→64→64→64→10, 1-bit weights/
activations) and iteratively tuned FINN's per-layer folding factors
(`PE`/`SIMD`) across 5 builds (automatic + 4 manual folds) to trade
hardware resources for throughput, including:
- Diagnosing why a fixed-cycle `Reshape` node capped end-to-end throughput
  regardless of downstream folding.
- Root-causing the folding legality constraint against the model's actual
  `NCHW` (`[1,1,28,28]`) input tensor shape, after an initial wrong
  assumption about data layout.
- Resolving the bottleneck by flattening the model's input to `[1,784]`
  ahead of the FINN build, removing the `Reshape` node entirely.
- Discovering that rtlsim throughput and the estimate report increasingly
  diverge under aggressive folding (a 2× *estimated* gain translated to
  only +5.8% real rtlsim throughput in the final fold).

| Configuration | RTL-Sim Throughput | Bottleneck Layer |
|---|---|---|
| Auto folding | 30,534 img/s | `MVAU_hls_0` (896 cyc, estimate) |
| Best custom fold | **1,923,077 img/s** | `Thresholding` / `MVAU_hls_0` tied (7 cyc, estimate) |

~63× rtlsim throughput improvement over the automatic folding baseline.
Note the best fold's post-synthesis F<sub>max</sub> came in at **~83 MHz**,
not the 100 MHz assumed by cycle-accurate rtlsim — real deployable
throughput is correspondingly lower than the rtlsim figure suggests. Full
process journal (including the wrong turns and how they were caught) in
[`task3/README.md`](./task3/README.md).

### Task 4 — GTSRB Traffic Sign Classifier on PYNQ-Z2
End-to-end pipeline, designed independently: CNN architecture/bit-width
design-space exploration, Brevitas QAT, QONNX export, in-graph pre/
post-processing (hardware normalization + Top-K argmax via `InsertTopK`),
FINN+ dataflow build, and on-board evaluation.

**Note:** This task was carried out as part of a final competition in which the goal was to maximize a (Throughput x Accuracy) Figure-of-Merit (FoM), while ensuring Accuracy is above 80%. The model parameters were chosen with that in mind.  

**Model:** 3 conv layers (12/24/36 channels), 12×12 input, 2-bit weights /
3-bit activations, power-of-two quantization — chosen from a 16-way
design-space sweep over image size, channel width, and bit-width (see
[`task4/README.md`](./task4/README.md)).

| Stage | Metric | Value |
|---|---|---|
| Training | Validation accuracy | 95.90% |
| Training | Test accuracy | 89.22% |
| FINN build | Target throughput | 800,000 img/s |
| FINN build | RTL-sim throughput | 447,275 img/s |
| Synthesis | LUT / FF / DSP | 40,660 / 41,191 / 0 |
| **On-board (PYNQ-Z2)** | **Throughput** | **~145,000 FPS** |
| **On-board (PYNQ-Z2)** | **Accuracy** | **89.29%** |
| **On-board (PYNQ-Z2)** | **Latency (batch of 1000)** | **0.0825 s** |

On-board throughput lands at ~30% of the RTL-sim figure — attributed to
PS–PL (ARM↔fabric) AXI DMA bandwidth and Python/PYNQ driver overhead
rather than any shortfall in the compiled accelerator itself; see the
task README for the full breakdown and discussion of paths to closing
that gap. Deployment package (bitstream, driver, evaluation notebook) is
self-contained in
[`task4/Pynq-Z2-deployment/`](./task4/Pynq-Z2-deployment).

---

## Cross-Task Notes

A few things that came up repeatedly across tasks, worth knowing before
diving into any one of them:

- **Estimate reports vs. rtlsim vs. post-synthesis timing are three
  different sources of truth**, and they diverge more as designs get
  pushed harder (Tasks 3 & 4). Cycle-based estimates assume ideal
  pipelining; rtlsim is cycle-accurate but assumes the target clock
  closes timing; only post-synthesis/implementation confirms the actual
  achievable F<sub>max</sub>. Task 3 in particular is a worked example of
  this gap widening under aggressive folding.
- **On-board throughput is a system-level number**, not just an
  accelerator-level one — DMA/interconnect bandwidth and host-side driver
  overhead (Task 4) can dominate over the compiled hardware's own
  cycle-level performance.
- **Don't assume a tensor's data layout or shape** — verify it directly
  against the actual model/intermediate graph before relying on it for a
  legality constraint (Task 3's `Reshape` node investigation is the
  concrete example of this going wrong once and getting caught).

---

## Getting Started

1. Set up the environment following [`task0/`](./task0) (conda env, Brevitas,
   Vivado 2022.1).
2. Each `taskN/` folder is self-contained with its own scripts/configs and
   a `README.md` describing setup, usage, and results in detail.
3. Task 2 requires GHDL (or Vivado's ISim) for VHDL simulation. Tasks 3
   and 4 require a working FINN(+) installation and, for on-board
   evaluation, a PYNQ-Z2 board on the local network.

---

## Notes

- This repository documents coursework submitted for academic credit at
  KIT. If you're a current student in this course, please treat this as a
  reference rather than a source to copy from directly.
- Result numbers throughout are pulled directly from build/eval logs; see
  each task's own README for full methodology and raw data.
