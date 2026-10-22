# The MoE Memory Wall: Predicting and Prefetching Experts for LLM Inference
## Mid-Evaluation Setup & Execution Guide

This repository contains the complete simulation framework, analytical models, real routing traces, and evaluation scripts for **Project 2: The MoE Memory Wall**.

---

## 1. System Requirements & Prerequisites

The framework requires a Linux environment (Native Ubuntu 20.04/22.04+ or **WSL2 on Windows**):

### Hardware Requirements
- **CPU:** x86_64 processor with multi-core support (for parallel C++ simulation)
- **RAM:** Minimum 8 GB (16 GB recommended)
- **Disk Space:** ~2 GB free space

### Software Packages
- **C++ Compiler:** GCC/G++ with C++20 support (`g++ >= 11.4.0`)
- **Build System:** CMake (`cmake >= 3.22`) and Make (`build-essential`)
- **Python:** Python 3.10 or higher with `pip`

---

## 2. Quick Setup Instructions

Once you have downloaded/cloned the project files, open a terminal and follow these steps:

### Step 1: Install System Dependencies (Linux / WSL)
```bash
sudo apt update
sudo apt install -y build-essential cmake g++ python3 python3-pip
```

### Step 2: Set Up Python Virtual Environment
Navigate to the project root (`ashu_`):
```bash
cd ashu_
python3 -m venv venv
source venv/bin/activate       # On Linux / WSL
# Or on Windows PowerShell: .\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Build the `LLMSimulator` C++ Engine
The primary simulator models GPU execution, Logic-PIM, and integrates the **Ramulator 2.0** cycle-accurate DRAM engine:
```bash
cd LLMSimulator
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../..
```
*Verification:* Ensure the executable `LLMSimulator/build/run` is created:
```bash
ls -lh LLMSimulator/build/run
```

---

## 3. Repository Directory Structure

```text
ashu_/
├── LLMSimulator/              # C++ Simulator (with Ramulator 2.0 DRAM microarchitecture)
│   ├── build/                 # Compiled binaries (`run`) and CMake cache
│   ├── src/                   # C++ engine source code
│   └── data/                  # Raw per-run simulation CSV outputs
├── SSD-offloading/            # Reference offload modeling extension
├── requirements.txt           # Python library dependencies
├── README.md                  # This setup and execution manual
└── mid_evaluation/            # Mid-evaluation experiment suite
    ├── scripts/               # All evaluation and analysis Python scripts
    ├── data/                  # Real Hugging Face routing traces (JSONL)
    ├── results/               # Formatted summary CSV results
    ├── plots/                 # Generated publication-quality figures
    ├── logs/                  # Raw execution logs from simulation runs
    └── MoE Memory Wall...pdf  # Mid-evaluation report document
```

---

## 4. Running the Evaluation Scripts

All Python scripts are designed to be run from the `ashu_` root directory with the virtual environment activated.

```bash
# Make sure you are in ashu_ directory:
cd ashu_
```

---

### Stage 1: Duplex Baseline Reproduction & DRAM Verification

#### 1. Run Baseline GPU vs. Logic-PIM Simulation
```bash
python mid_evaluation/scripts/reproduce_duplex.py
```
- **What it does:** Configures Mixtral 8x7B (batch size 32, decode iteration) and calls `LLMSimulator` to simulate:
  1. Baseline 4× NVIDIA H100 GPUs
  2. Duplex (GPU + Logic-PIM in HBM dies)
  3. Duplex + PE (Co-processing enabled)
- **Outputs Generated:**
  - `mid_evaluation/results/duplex_reproduction_summary.csv`
  - Per-run execution logs in `mid_evaluation/logs/`

#### 2. Run Cycle-Accurate Ramulator 2.0 DRAM Microarchitecture Check
```bash
python mid_evaluation/scripts/exp1_ramulator_check.py
```
- **What it does:** Executes `LLMSimulator` with Ramulator 2.0 enabled (`use_ramulator: true`) to model realistic DRAM bank conflicts, row-buffer miss latencies, and $t_{\text{CCD\_L}}$ constraints.
- **Outputs Generated:**
  - `mid_evaluation/results/ramulator_check.csv`

#### 3. Generate Multi-System Throughput and Latency Plots
```bash
python mid_evaluation/scripts/exp1_throughput_latency_scaling.py
```
- **What it does:** Parses the simulation results and creates comparative throughput and latency scaling charts across 4× H100 and 8× B100 systems.
- **Outputs Generated:**
  - `mid_evaluation/plots/fig11_throughput_comparison.png` (Generation throughput)
  - `mid_evaluation/plots/fig12_latency_comparison.png` (Per-token decode latency)
  - `mid_evaluation/results/system_performance_comparison.csv`
  - `mid_evaluation/results/e2e_latency_comparison.csv`

---

### Stage 2: Single-GPU MoE Parameter & Off-Chip Modeling

#### 4. Run Analytical Single-GPU MoE Parameter & Timing Model
```bash
python mid_evaluation/scripts/model_offchip_moe.py
```
- **What it does:** Calculates exact Mixtral 8x7B parameter sizing ($352.32 \text{ MB}$ per expert, $90.19 \text{ GB}$ total). Computes ALU compute duration vs. PCIe Gen 5 ($64 \text{ GB/s}$) and PCIe Gen 4 ($32 \text{ GB/s}$) streaming latency across batch sizes $1 \le B \le 64$.
- **Outputs Generated:**
  - `mid_evaluation/results/offchip_dram_expert_metrics.csv`
  - `mid_evaluation/plots/moe_offchip_loading_vs_compute.png` (Linear scale comparison)

#### 5. Run Single-GPU Decode Simulation Sweep
```bash
python mid_evaluation/scripts/sim_single_gpu_sweep.py
```
- **What it does:** Calls `LLMSimulator` to simulate full single-GPU execution of Mixtral 8x7B (32 layers) across batch sizes $1 \le B \le 64$, pairing GPU compute/attention with off-chip DRAM transfers.
- **Outputs Generated:**
  - `mid_evaluation/results/simulated_single_gpu_offchip.csv`

---

### Stage 3: Interconnect Scaling & Prefetching Upper Bound

#### 6. Benchmark Interconnect Bandwidth Scaling
```bash
python mid_evaluation/scripts/exp2_interconnect_decode_benchmarks.py
```
- **What it does:** Evaluates decode step latency across 5 interconnect technologies (PCIe 4.0 @ $32 \text{ GB/s}$, PCIe 5.0 @ $64 \text{ GB/s}$, CXL 2.0 @ $128 \text{ GB/s}$, NVLink-C2C Low @ $450 \text{ GB/s}$, NVLink-C2C High @ $900 \text{ GB/s}$) on Mixtral 8x7B and DeepSeek-V3. Calculates the theoretical prefetching speedup ceiling.
- **Outputs Generated:**
  - `mid_evaluation/plots/expert_loading_dominates_decode.png` (End-to-end breakdown)
  - `mid_evaluation/plots/expert_copy_vs_compute_logscale.png` (Log-scale compute vs transfer)
  - `mid_evaluation/plots/interconnect_scaling_prefetch_bound.png` (Prefetching upper bound)

---

### Stage 4: Spatio-Temporal Correlation Analysis (Real Hugging Face Data)

#### 7. Download / Refresh Real Hugging Face Routing Traces (Optional)
```bash
python mid_evaluation/scripts/download_hf_traces.py
```
- **What it does:** Streams real routing decisions from the Hugging Face dataset `allenai/analysis_mixtral` (inference on the C4 pre-training corpus for `mistralai/Mixtral-8x7B`).
- *Note:* A pre-extracted trace file is already provided at `mid_evaluation/data/hf_mixtral_c4_routing.jsonl` (320,000 decisions across 10,000 tokens).

#### 8. Inspect Routing Traces (Sanity Check)
```bash
python mid_evaluation/scripts/inspect_hf_trace.py
```
- **What it does:** Reads the first few tokens of the trace file and prints layer-by-layer router selections.

#### 9. Run Comprehensive Spatio-Temporal Correlation Suite
```bash
python mid_evaluation/scripts/correlation_analysis.py
```
- **What it does:** Analyzes 320,000 real routing points:
  1. Computes global Shannon entropy and Gini coefficient.
  2. Constructs the spatial Markov transition matrix $P(E_{L+1} \mid E_L)$ across adjacent layers.
  3. Evaluates temporal self-persistence $P(E_{t+1} \mid E_t)$ across consecutive tokens.
  4. Measures pairwise router co-activation strength.
- **Outputs Generated:**
  - `mid_evaluation/results/correlation_statistics.csv`
  - `mid_evaluation/plots/spatio_temporal_correlation_dashboard.png` (4-panel dashboard)

#### 10. Auxiliary Skew Diagnostics
```bash
python mid_evaluation/scripts/check_layer_skew.py
python mid_evaluation/scripts/check_seq_skew.py
```
- **What they do:** Print layer-wise Gini specialization values and sequence-wise variance across different prompt sequences.

---

## 5. Summary of Generated Artifacts

### Summary CSV Files (`mid_evaluation/results/`)
| File Name | Description |
| :--- | :--- |
| `duplex_reproduction_summary.csv` | Baseline 4× H100 vs. Duplex simulation metrics (latency, energy, speedup) |
| `ramulator_check.csv` | Comparison between analytical peak and cycle-accurate DRAM simulation |
| `offchip_dram_expert_metrics.csv` | Parameter sizes, compute times, PCIe load times, and stall ratios |
| `simulated_single_gpu_offchip.csv` | End-to-end simulated decode latencies across batch sizes 1 to 64 |
| `correlation_statistics.csv` | Entropy, Gini, spatial prediction accuracy, and temporal persistence rates |
| `system_performance_comparison.csv`| Multi-system throughput, TBT, and speedup summary table |
| `e2e_latency_comparison.csv` | End-to-end request latencies (generating 1024 tokens from 1024-token prompt) |

### Plot Figures (`mid_evaluation/plots/`)
| File Name | Description |
| :--- | :--- |
| `fig11_throughput_comparison.png` | Generation throughput (tokens/s) across Baseline vs Duplex |
| `fig12_latency_comparison.png` | Per-token decode latency breakdown (MoE vs Non-MoE time) |
| `moe_offchip_loading_vs_compute.png` | Linear bar chart of PCIe transfer vs GPU compute time |
| `expert_copy_vs_compute_logscale.png`| Logarithmic comparison showing the orders-of-magnitude gap |
| `expert_loading_dominates_decode.png`| Stacked percentage showing >96% of decode time spent on transfers |
| `interconnect_scaling_prefetch_bound.png` | Interconnect bandwidth scaling & theoretical prefetching speedup ceiling |
| `spatio_temporal_correlation_dashboard.png` | 4-panel dashboard: popularity, spatial transitions, temporal persistence, co-activation |

---

## 6. Key Scientific Findings & Expected Values

When running the suite, the outputs will confirm the following key metrics:

1. **Duplex Baseline Reproduction:**
   - Baseline 4× H100 latency: **$7,975.6 \ \mu\text{s}$**
   - Duplex (Logic-PIM) latency: **$3,111.1 \ \mu\text{s}$**
   - Speedup: **$2.564\times$** (matches published MICRO '24 paper range of $2.51\times\text{--}2.67\times$).

2. **DRAM Microarchitecture Fidelity (Ramulator 2.0):**
   - Cycle-accurate timing is **$23\%\text{--}26\%$ slower** than analytical peak due to realistic DRAM bank conflicts and row-buffer misses.
   - Duplex maintains a **$2.61\times$ speedup** under Ramulator timing, with DRAM dynamic energy reduced by **$42.2\%$** ($1,978 \text{ mJ}$ vs $3,421 \text{ mJ}$).

3. **The MoE Memory Wall:**
   - Mixtral 8x7B expert weights require **$90.19 \text{ GB}$**, exceeding an 80 GB A100 GPU.
   - At batch size 1, computing an expert takes **$1.13 \ \mu\text{s}$**, but transferring it across PCIe 5.0 takes **$5,505 \ \mu\text{s}$**.
   - Load-to-execution ratio is **$\sim 32\times$** (load vs local HBM read of $172.8 \ \mu\text{s}$). The GPU is stalled **$96.96\%$** of the time.

4. **The PCIe Prefetching Paradox:**
   - On commodity PCIe ($32\text{--}64 \text{ GB/s}$), loading expert weights takes hundreds of milliseconds ($365\text{--}717 \text{ ms}$), while computation takes only $12\text{--}18 \text{ ms}$.
   - Prefetching alone can only hide the compute time, capping maximum speedup at **$1.02\times\text{--}1.05\times$**. Prefetching **must be paired with hot-expert caching**.

5. **Predictability from Real Traces:**
   - Global Shannon Entropy: **$2.9996 \text{ bits}$** (globally balanced across C4).
   - Spatial Layer-to-Layer ($L \rightarrow L+1$) Top-2 Prediction Accuracy: **$51.76\%$** (more than **$2\times$ the random chance of $25\%$**).
   - Temporal Self-Persistence ($t \rightarrow t+1$): **$23.84\%$** (nearly **$2\times$ the random chance of $12.5\%$**).
