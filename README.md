<div align="center">

# 🏔️ Alpamayo 1.5

### Supercharging Autonomous Driving with Interactive, Steerable Reasoning

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Alpamayo--1.5--10B-blue)](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

## Updates

- [May 2026] SFT and RL post-training scripts are available in [Alpamayo Recipes](https://github.com/NVlabs/alpamayo-recipes): [Alpamayo 1.5 SFT](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_5_sft) and [Alpamayo 1.x RL post-training](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_x_rl).

**📖 Please read the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) first!**
The model card contains comprehensive details on model architecture, inputs/outputs, licensing, and tested hardware configurations. This GitHub README focuses on setup, usage, and frequently asked questions.

## Support

📣 **Usage questions and discussion about Alpamayo 1.5**: please join us on the [Alpamayo NV Developer Forum](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766).

🐛 **Code-level bugs, documentation issues, and feature requests**: file a [GitHub issue](../../issues/new/choose) using the appropriate template (Bug report, Documentation request, or Feature request). The relevant NVIDIA responder is auto-assigned via the `assignees:` field on the template.

🚨 **Security vulnerabilities**: please use [NVIDIA's Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). Do not file security issues publicly here.

## Prerequisites

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit 12.x** with `nvcc` (required to compile `flash-attn` from source). If you don't have it, see [Troubleshooting](#flash-attention-issues) for a fallback using PyTorch's built-in SDPA.
- **Python 3.12**

### Hardware requirements

| Configuration                                           | VRAM   |
| ------------------------------------------------------- | ------ |
| Single-sample inference (`num_traj_samples=1`)          | ~24 GB |
| Multi-sample inference (`num_traj_samples=16`)          | ~40 GB |
| Multi-sample inference with CFG (`num_traj_samples=16`) | ~60 GB |

Measured on an NVIDIA H100 80GB GPU.

## Getting Started

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Set up the environment

```bash
uv venv a1_5_venv
source a1_5_venv/bin/activate
uv sync --active
```

> **Note:** If `uv sync` fails on `flash-attn`, see [Troubleshooting](#flash-attention-issues) below.

### 3. Authenticate with HuggingFace

The model and dataset require access to gated resources. Request access here:

- 🤗 [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- 🤗 [Alpamayo-1.5-10B Model](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

Then authenticate:

```bash
hf auth login
```

Get your token at: https://huggingface.co/settings/tokens

> **Note:** The `physical_ai_av` package (auto-installed via dependencies) streams data from the HuggingFace dataset. You must have accepted the dataset access request above before running inference.

## Running Inference

### Test script

NOTE: This script will download both some example data (relatively small) and the model weights (22 GB).
The latter can be particularly slow depending on network bandwidth.
For reference, it takes around 2.5 minutes on a 100 MB/s wired connection.

```bash
python src/alpamayo1_5/test_inference.py
```

In case you would like to obtain more trajectories and reasoning traces, please feel free to increase
the `num_traj_samples` argument in the script.

### Tracked runs (ML_Platform)

`scripts/run_inference_tracked.py` is the test script above, wired to the MLflow
tracking server so a run can be traced back later: which commit, which data, which
model, how well.

```bash
pip install "mlflow>=3"
export MLFLOW_TRACKING_URI=http://ysh-jetson-orin-nano.tail4570ef.ts.net:5000
git commit -am "..."                       # dirty runs are not reproducible
python scripts/run_inference_tracked.py --notes "baseline before KD"
```

Run it from the repository root -- MLflow reads `cwd` to autolog the git commit, and
that tag outranks anything set by hand. The tracking server is Tailscale-only; from
outside the tailnet, forward it with
`ssh -N -L 5000:100.81.70.49:5000 <relay>` and use `http://localhost:5000`.

| Flag | Default | |
|---|---|---|
| `--clip-id` | one example clip | repeat for several clips |
| `--num-traj-samples` | `1` | trajectories sampled per clip |
| `--model` | `nvidia/Alpamayo-1.5-10B` | HF repo id, or a local checkpoint path |
| `--run-type` | `eval` | `infer` when predictions are not scored |
| `--experiment` / `--project` | `alpamayo-kd` | the project, not the run |
| `--evals-repo` | `YSHRobotics/alpamayo-kd-evals` | private HF repo receiving the run directory |
| `--notes` | -- | one or two human sentences; this is what makes a run readable later |
| `--include-gt` | off | also store the logged future (gated-dataset content) |
| `--no-samples` / `--no-upload` / `--no-track` | off | skip visualizations / Hugging Face / MLflow |

The dataset carries the logged future trajectory, so minADE is computable and the run
is recorded as an `eval` with `score` = mean minADE in meters (lower is better).

Each run writes a directory under `out/` and pushes it to the private evals repo:

```
runs/<YYYY-MM-DD>_<run-name>/
├── summary.json        scores, config, CoC traces -- also kept as an MLflow artifact
├── trajectories.npz    predicted trajectories per clip
└── samples/            one BEV + camera figure per clip
```

The run then carries `output_uri = hf:<repo>@<40-char sha>#runs/<dir>/`, so the numbers
on MLflow and the files on Hugging Face point at each other. Uploads above 10MB per file
or 50MB per run never go through MLflow -- the tracking server is a Jetson Orin Nano with
one worker, and a single large upload stalls it for everyone.

Ground truth is **not** stored by default. It is content from a gated NVIDIA dataset, and
`score` already captures everything the comparison needs; `--include-gt` overrides this
for local debugging.

On exit the script prints `PASS` or `FAIL` with the missing coordinates. A `FAIL` is a
bug in the script, not in the run.

#### Recorded metrics (EC-254)

Every run also logs the Tier 1 metrics the depth-pruning demonstration is judged on.
They are all derived from the inference pass that already ran, so they cost nothing extra.
Implementation lives in `src/alpamayo1_5/prune/metrics.py`, purely additive to upstream.

| Group | Metrics |
|---|---|
| Accuracy | `min_ade` / `min_fde` / `mean_ade` / `mean_fde`, plus `ade_<h>` and `de_<h>` at 1, 2, 3, 4, 5 and 6.4s |
| Kinematics | `jerk_mean` / `jerk_p95`, `lat_accel_mean` / `lat_accel_p95`, `lat_accel_over_4_ratio`, `accel_violation_rate`, `curvature_violation_rate`, `within_bounds_ratio` |
| Reasoning | `cot_tokens_mean` / `_min` / `_max`; `meta_action` kept per clip in `summary.json` |
| Scene | `net_heading_deg`, `lateral_offset_m`, and a `scene.<type>.count` mix over straight / curve / lane_change |
| Size and speed | `params_billions`, `param_bytes_gb`, `vram_peak_gb`, `latency_sec_mean` / `_p95` |

The horizon breakdown is the point of the accuracy group: cutting layers is expected to
fail at distance first, and a single aggregate ADE hides exactly that.

Kinematics are computed in **physical units** by projecting the prediction back through
`traj_to_action` and denormalizing. `is_within_bounds` is reported as `within_bounds_ratio`
but only as a hard canary -- its gates sit at roughly 12-14 sigma and it collapses all 64
waypoints into one boolean, so it answers "did anything explode", never "is this
comfortable to ride in".

`--inference-step` is validated rather than passed through: `flow_matching.py` resolves it
as `inference_step or self.num_inference_steps`, so `0` silently becomes the default 10 and
quietly corrupts any latency measurement.

> **Not yet covered.** Runs are **not** comparable across pruned configurations yet:
> generation samples at `temperature=0.6`, so two runs of the same clip produce different
> reasoning and different trajectories. Fixing the conditioning (`prune/conditioning.py`)
> has to land before the pruning-curve table is meaningful. Also outstanding: the
> three-way latency split (`prune/profile.py`), the egomotion scene pre-pass
> (`prune/scene_labels.py`), trajectory-CoT agreement and hallucination
> (`prune/cot_quality.py`), and paired bootstrap CIs.


### Interactive notebooks

We provide notebooks that demonstrate the different capabilities of Alpamayo 1.5 under `notebooks/`, including standard model inference, incorporating navigation guidance, modifying the number of cameras, and visual question answering.

### Inference methods

Alpamayo 1.5 provides two inference methods:

- **`sample_trajectories_from_data_with_vlm_rollout`** -- Full pipeline: the VLM generates chain-of-causation reasoning, then a diffusion expert produces trajectory predictions conditioned on the VLM's hidden states. This is the primary inference method used by the test script and most notebooks.

- **`generate_text`** -- Text-only generation for visual question answering (VQA). Returns extracted text fields.

### Optional CUDA graph acceleration

Repeated trajectory inference can replay the diffusion expert with exact-shape CUDA graphs. Enable
this after moving the model to CUDA and calling `eval()`:

```python
model.eval()
model.enable_diffusion_expert_cuda_graph(
    max_batch_size=16,
    max_graphs=4,
)
```

Set `max_batch_size` to at least `batch_size * num_traj_samples * num_traj_sets`. The first
supported input shape is captured lazily; up to `max_graphs` exact shape signatures are retained,
and additional signatures fall back to eager execution. Captured graphs keep static CUDA buffers,
so this option trades additional GPU memory for lower diffusion-expert launch overhead. Inspect
`model.diffusion_expert_cuda_graph_stats` for capture, replay, and fallback counts.


## Fine-tuning and Post-training Recipes

SFT and RL post-training scripts are maintained in [Alpamayo Recipes](https://github.com/NVlabs/alpamayo-recipes):

- [Alpamayo 1.5 SFT](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_5_sft)
- [Alpamayo 1.x RL post-training](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_x_rl), including Alpamayo 1.5

## Project Structure

```
alpamayo_1.5_release/
├── notebooks/
│   ├── inference.ipynb                  # Standard model inference
│   ├── inference_cam_num.ipynb          # Inference with different camera counts
│   ├── inference_nav.ipynb              # Inference with navigation guidance
│   └── inference_vqa.ipynb              # Visual question answering
├── src/
│   └── alpamayo1_5/
│       ├── action_space/
│       │   └── ...                      # Action space definitions
│       ├── diffusion/
│       │   └── ...                      # Diffusion model components
│       ├── geometry/
│       │   └── ...                      # Geometry utilities and modules
│       ├── models/
│       │   ├── ...                      # Model components and utils functions
│       ├── __init__.py                  # Package marker
│       ├── config.py                    # Model and experiment configuration
│       ├── helper.py                    # Utility functions
│       ├── load_physical_aiavdataset.py # Dataset loader
│       ├── test_inference.py            # Inference test script
├── pyproject.toml                       # Project dependencies
└── uv.lock                              # Locked dependency versions
```

## Troubleshooting

### Flash Attention issues

The model uses Flash Attention 2 by default. `flash-attn` requires CUDA Toolkit (specifically `nvcc`) at build time. If you see build errors during `uv sync`:

**Option A: Install without flash-attn and use SDPA fallback**

```bash
uv sync --active --no-install-package flash-attn
```

Then load the model with PyTorch's built-in scaled dot-product attention:

```python
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

model = Alpamayo1_5.from_pretrained(
    "nvidia/Alpamayo-1.5-10B",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda")
```

**Option B: Install CUDA Toolkit, then retry**

Install CUDA Toolkit 12.x (e.g., via your package manager or [NVIDIA's install guide](https://developer.nvidia.com/cuda-downloads)), ensure `nvcc` is on your PATH, then re-run:

```bash
uv sync --active
```

## Frequently Asked Questions (FAQ)

<details>
<summary><strong>How does Alpamayo 1.5 relate to Alpamayo 1?</strong></summary>

Alpamayo 1.5 expands upon the architecture released in Alpamayo 1 and fully realizes what is described in our paper [*"Alpamayo 1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail
"*](https://arxiv.org/abs/2511.00088). Specifically:

| Feature                                 | Description                                                      | Alpamayo 1             | Alpamayo 1.5       |
| --------------------------------------- | ---------------------------------------------------------------- | ---------------------- | ------------------ |
| **Chain-of-Causation (CoC) reasoning**  | Hybrid auto-labeling with human in the loop for reasoning traces | ✅ Included            | ✅ Included        |
| **Vision-Language-Action architecture** | Cosmos-Reason backbone + action expert                           | ✅ Included            | ✅ Included        |
| **Trajectory prediction**               | 6.4s horizon, 64 waypoints at 10 Hz                              | ✅ Supported           | ✅ Supported       |
| **RL post-training**                    | Reinforcement learning for reasoning/action consistency          | ❌ Not RL post-trained | ✅ RL post-trained |
| **Navigation conditioning**             | Explicit navigation inputs                                       | ❌ Not supported       | ✅ Supported       |
| **General VQA**                         | Supports visual question answering                               | ❌ Not supported       | ✅ Supported       |
| **Flexible multi-camera support**       | Supports a variable number of input cameras                      | ❌ Not supported       | ✅ Supported       |

</details>

<details>
<summary><strong>Does Alpamayo 1.5 accept navigation inputs?</strong></summary>

Yes! Please see `notebooks/inference_nav.ipynb` for examples.

</details>

<details>
<summary><strong>Does Alpamayo 1.5 support general VQA?</strong></summary>

Yes! Please see `notebooks/inference_vqa.ipynb` for examples.

</details>

<details>
<summary><strong>Was Alpamayo 1.5 post-trained with Reinforcement Learning (RL)?</strong></summary>

Yes! Alpamayo 1.5 has undergone RL post-training, achieving improvements in reasoning quality and reasoning-trajectory alignment as a result.

</details>

<details>
<summary><strong>Does Alpamayo 1.5 accept different numbers of cameras?</strong></summary>

Yes! Please see `notebooks/inference_cam_num.ipynb` for examples. Note that model accuracy may degrade with fewer cameras, the magnitude of which will depend on the specific scenario. For instance, it is expected that Alpamayo 1.5 would struggle to see cross-traffic in a right turn if only provided a front-facing camera.

</details>

<details>
<summary><strong>What are the minimum GPU requirements?</strong></summary>

You need an NVIDIA GPU with at least **24 GB VRAM** for inference. Tested configurations include RTX 3090, A100, H100, and B200. Running on GPUs with less memory (e.g., 16 GB) will likely result in CUDA out-of-memory errors. Please refer to our [hardware requirements](#hardware-requirements) for more information.

</details>

<details>
<summary><strong>Can I use this model in production / commercial applications?</strong></summary>

Yes. See the [License](#license) section and the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) for details.

</details>

## License

- **Inference code**: Apache License 2.0 - see [LICENSE](./LICENSE) for details.
- **Model weights**: OpenMDW-1.1 - see the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) for details.

## Disclaimer

Alpamayo 1.5 is a pre-trained reasoning model designed to accelerate research and development in the autonomous vehicle (AV) domain. It is intended to serve as a foundation for a range of AV-related use cases-from instantiating an end-to-end backbone for autonomous driving to enabling reasoning-based auto-labeling tools. In short, it should be viewed as a building block for developing customized AV applications.

Important notes:

- Alpamayo 1.5 is provided solely for research, experimentation, and evaluation purposes.
- Alpamayo 1.5 is not a fully fledged driving stack. Among other limitations, it lacks access to critical real-world sensor inputs, does not incorporate required diverse and redundant safety mechanisms, and has not undergone automotive-grade validation for deployment.

By using this model, you acknowledge that it is a research tool intended to support scientific inquiry, benchmarking, and exploration—not a substitute for a certified AV stack. The developers and contributors disclaim any responsibility or liability for the use of the model or its outputs.

## Citation

If you use Alpamayo 1.5 in your research, please cite:

```bibtex
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```
# YSH-KD-Alpamayo-1.5
# YSH-KD-Alpamayo-1.5
