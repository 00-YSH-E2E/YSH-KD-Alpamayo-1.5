# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Alpamayo 1.5 inference, recorded on ML_Platform.

Same pipeline as ``src/alpamayo1_5/test_inference.py`` -- load clips, roll out
the VLM, sample trajectories, score them against the logged future -- but every
run lands on the MLflow tracking server with the coordinates needed to trace it
back later: which code, which data, which model, how well.

Outputs go to a private Hugging Face dataset repo, and the run carries the
resulting commit sha as ``output_uri``. Numbers live on MLflow; files live on
Hugging Face; neither holds a copy of the other.

The dataset ships a ground-truth future trajectory, so minADE is computable and
the run is an ``eval`` (``score`` = mean minADE in meters, lower is better).
Pass ``--run-type infer`` when predictions are produced without scoring them.

Run it from the repository root so MLflow picks up the right git commit::

    export MLFLOW_TRACKING_URI=http://ysh-jetson-orin-nano.tail4570ef.ts.net:5000
    python scripts/run_inference_tracked.py --notes "baseline before KD"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ml_platform_track as mlp  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402
from alpamayo1_5.prune import metrics as pm  # noqa: E402

DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
MODEL_REPO = "nvidia/Alpamayo-1.5-10B"
DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
EVALS_REPO = "YSHRobotics/alpamayo-kd-evals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", action="append", default=None,
                        help="Clip to run. Repeat for several clips (default: one example clip).")
    parser.add_argument("--t0-us", type=int, default=5_100_000,
                        help="Trajectory sampling timestamp in microseconds.")
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--inference-step", type=int, default=None,
                        help="Euler steps for the diffusion expert. The model falls back to "
                             "its default (10) for 0 or None, silently -- so 0 is rejected "
                             "here rather than quietly measuring the wrong thing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=MODEL_REPO,
                        help="HF model repo, or a local path for a distilled checkpoint.")
    parser.add_argument("--experiment", default="alpamayo-kd",
                        help="MLflow experiment: the project, not the run.")
    parser.add_argument("--project", default="alpamayo-kd",
                        help="Hub project slug (web/projects.config.json).")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--notes", default=None,
                        help="One or two human sentences: why this run exists.")
    parser.add_argument("--run-type", choices=["eval", "infer"], default="eval",
                        help="eval scores against the logged future; infer only writes outputs.")
    parser.add_argument("--out-dir", default="out",
                        help="Where predictions are written before upload.")
    parser.add_argument("--evals-repo", default=EVALS_REPO,
                        help="Private HF dataset repo that receives the run directory.")
    parser.add_argument("--include-gt", action="store_true",
                        help="Also store the ground-truth trajectory. Off by default: it is "
                             "content from a gated dataset, and the score already captures it.")
    parser.add_argument("--no-samples", action="store_true",
                        help="Skip the per-clip BEV visualizations.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Write outputs locally without pushing to Hugging Face.")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--no-track", action="store_true",
                        help="Run the inference without touching MLflow.")
    return parser.parse_args()


def slugify(text: str) -> str:
    """Reduce a run name to something safe for a directory and a repo path."""
    return re.sub(r"[^0-9A-Za-z._-]+", "-", text).strip("-") or "run"


def load_model(source: str) -> Alpamayo1_5:
    """Load from a HF repo id or a local checkpoint directory."""
    return Alpamayo1_5.from_pretrained(source, dtype=torch.bfloat16).to("cuda")


def render_sample(result: dict, data: dict, path: Path) -> bool:
    """Write one BEV + camera figure for a clip.

    The skill is blunt about this: a run archived without viewable samples shows
    nothing on the Hub, which defeats the point of keeping the outputs at all.
    A failed plot must not take the run down with it.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from alpamayo1_5 import viz_utils

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 1.3]})
        grid = viz_utils.make_camera_grid(data["image_frames"], data["camera_indices"])
        axes[0].imshow(grid)
        axes[0].axis("off")

        viz_utils.plot_condition(
            axes[1], result["pred_xy"], color="tab:blue", label="prediction"
        )
        if result.get("gt_xy") is not None:
            gt = result["gt_xy"]
            axes[1].plot(gt[:, 0], gt[:, 1], color="k", linestyle="--", linewidth=2,
                         label="logged future")
        axes[1].set_aspect("equal", adjustable="datalim")
        axes[1].set_xlabel("x [m]")
        axes[1].set_ylabel("y [m]")
        axes[1].legend(loc="best", fontsize=8)
        title = result["clip_id"]
        if result["min_ade"] is not None:
            title += f"  |  minADE {result['min_ade']:.2f} m"
        axes[1].set_title(title, fontsize=10)

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return True
    except Exception as exc:  # visualization is never worth failing a run over
        print(f"[samples] could not render {result['clip_id']}: {exc}", file=sys.stderr)
        return False


def measure_kinematics(model: Alpamayo1_5, data: dict, pred_xyz, pred_rot) -> dict:
    """Project the prediction back into the action space and read off feasibility.

    A numerical failure here must not take the run down: the trajectory is
    already produced and scored by the time this runs.
    """
    try:
        n_samples = pred_xyz.shape[2]
        history_xyz = data["ego_history_xyz"][:, -1].repeat(n_samples, 1, 1)
        history_rot = data["ego_history_rot"][:, -1].repeat(n_samples, 1, 1, 1)
        future_xyz = pred_xyz[0, 0].to(history_xyz.device)
        future_rot = pred_rot[0, 0].to(history_rot.device)
        stats = pm.kinematics(
            model.action_space, history_xyz, history_rot, future_xyz, future_rot
        )
        stats.update(pm.heading_summary(future_xyz, future_rot))
        return stats
    except Exception as exc:
        print(f"[metrics] kinematics unavailable: {exc}", file=sys.stderr)
        return {}


def run_clip(model: Alpamayo1_5, processor, clip_id: str, args, out_dir: Path) -> dict:
    """Run one clip and return its predictions, reasoning trace and minADE."""
    data = load_physical_aiavdataset(clip_id, t0_us=args.t0_us)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )

    diffusion_kwargs = {}
    if args.inference_step is not None:
        diffusion_kwargs["inference_step"] = args.inference_step

    torch.cuda.manual_seed_all(args.seed)
    # Without an explicit sync the timer measures kernel queueing, not compute.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
            diffusion_kwargs=diffusion_kwargs,
            return_extra=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    cot = [str(c) for c in np.asarray(extra["cot"]).reshape(-1)]
    meta_action = [str(m) for m in np.asarray(extra.get("meta_action", [])).reshape(-1)]
    result = {
        "clip_id": clip_id,
        "t0_us": args.t0_us,
        "pred_xy": pred_xyz.cpu().numpy()[0, 0, :, :, :2],  # [K, T, 2]
        "cot": cot,
        "meta_action": meta_action,
        "latency_sec": elapsed,
        "min_ade": None,
        "gt_xy": None,
        "metrics": {},
    }
    result["metrics"].update(pm.cot_stats(model.tokenizer, cot))

    # The dataset carries the logged future, so the run can be scored.
    gt = data.get("ego_future_xyz")
    if gt is not None and args.run_type == "eval":
        gt_xy = gt.cpu()[0, 0, :, :2].numpy()  # [T, 2]
        accuracy = pm.displacement_metrics(result["pred_xy"], gt_xy)
        result["metrics"].update(accuracy)
        result["min_ade"] = accuracy["min_ade"]
        result["gt_xy"] = gt_xy

    result["metrics"].update(measure_kinematics(model, data, pred_xyz, pred_rot))
    # No heading means no scene: a default of 0/0 would silently label everything "straight".
    if "net_heading_abs_deg" in result["metrics"]:
        result["scene"] = pm.classify_scene(
            result["metrics"]["net_heading_abs_deg"],
            result["metrics"]["lateral_offset_abs_m"],
        )
    else:
        result["scene"] = None

    if not args.no_samples:
        render_sample(result, data, out_dir / "samples" / f"{clip_id}.png")
    return result


def write_outputs(results: list[dict], out_dir: Path, include_gt: bool) -> None:
    """Write predictions and reasoning traces into the run directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for result in results:
        arrays[f"{result['clip_id']}.pred_xy"] = result["pred_xy"]
        if include_gt and result["gt_xy"] is not None:
            arrays[f"{result['clip_id']}.gt_xy"] = result["gt_xy"]
    np.savez_compressed(out_dir / "trajectories.npz", **arrays)

    (out_dir / "cot.txt").write_text(
        "\n\n".join(f"=== {r['clip_id']} ===\n" + "\n---\n".join(r["cot"]) for r in results)
    )
    summary = {
        "clips": [
            {
                "clip_id": r["clip_id"],
                "t0_us": r["t0_us"],
                "min_ade": r["min_ade"],
                "latency_sec": round(r["latency_sec"], 3),
                "scene": r.get("scene"),
                "meta_action": r.get("meta_action"),
                "metrics": r.get("metrics", {}),
                "cot": r["cot"],
            }
            for r in results
        ]
    }
    scored = [r["min_ade"] for r in results if r["min_ade"] is not None]
    if scored:
        summary["mean_min_ade"] = float(np.mean(scored))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def upload_run(out_dir: Path, repo: str, run_dir: str) -> str | None:
    """Push the run directory to the private evals repo, returning the commit sha."""
    try:
        from huggingface_hub import HfApi

        commit = HfApi().upload_folder(
            folder_path=str(out_dir),
            path_in_repo=f"runs/{run_dir}",
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"Add run {run_dir}",
        )
        return getattr(commit, "oid", None)
    except Exception as exc:
        print(f"[upload] failed, outputs stay local: {exc}", file=sys.stderr)
        return None


def log_results(run: mlp.Run, results: list[dict], out_dir: Path, args, run_dir: str,
                model_stats: dict) -> None:
    """Attach metrics, the per-clip breakdown and the output coordinate."""
    scored = [r["min_ade"] for r in results if r["min_ade"] is not None]
    run.metric("n_clips", len(results))
    run.metric("n_traj_samples", args.num_traj_samples)
    run.metric("latency_sec_mean", float(np.mean([r["latency_sec"] for r in results])))
    run.metric("latency_sec_p95", float(np.percentile([r["latency_sec"] for r in results], 95)))
    for key, value in model_stats.items():
        run.metric(key, value)

    # EC-254 Tier 1: accuracy by horizon, kinematic feasibility, CoT length.
    keys = sorted({k for r in results for k in r.get("metrics", {})})
    for key in keys:
        values = [r["metrics"][key] for r in results if key in r["metrics"]]
        if values:
            run.metric(key, float(np.mean(values)))

    # Scene mix, so a headline number is never read without knowing what it covered.
    scenes = [r.get("scene") for r in results if r.get("scene")]
    for scene in set(scenes):
        run.metric(f"scene.{scene}.count", scenes.count(scene))

    if scored:
        run.score(float(np.mean(scored)))  # minADE in meters, lower is better
        run.metric("min_ade_max", float(np.max(scored)))
        run.by_scenario({r["clip_id"]: r["min_ade"] for r in results if r["min_ade"] is not None})

    # summary.json is small and worth having in the MLflow UI; the bulk lives on HF.
    run.artifact(str(out_dir / "summary.json"), name=args.run_type)

    if args.no_upload:
        run.result_path(f"path:{out_dir}")
        return
    sha = upload_run(out_dir, args.evals_repo, run_dir)
    if sha:
        run.result_path(f"hf:{args.evals_repo}@{sha}#runs/{run_dir}/")
    else:
        run.result_path(f"path:{out_dir}")


def main() -> None:
    args = parse_args()
    if args.inference_step is not None and args.inference_step < 1:
        raise SystemExit(
            "--inference-step must be >= 1. The model treats 0 as 'unset' and silently "
            "runs its default instead, which quietly corrupts any latency measurement."
        )
    clip_ids = args.clip_id or [DEFAULT_CLIP_ID]
    run_name = args.run_name or f"a1.5-10b-{len(clip_ids)}clip-n{args.num_traj_samples}"
    run_dir = f"{time.strftime('%Y-%m-%d')}_{slugify(run_name)}"
    out_dir = Path(args.out_dir) / run_dir

    params = {
        "model": args.model,
        "clip_ids": clip_ids,
        "t0_us": args.t0_us,
        "num_traj_samples": args.num_traj_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_generation_length": args.max_generation_length,
        "seed": args.seed,
        "dtype": "bfloat16",
        "device": "cuda",
        "inference_method": "sample_trajectories_from_data_with_vlm_rollout",
        "inference_step": args.inference_step,
        "include_gt": args.include_gt,
    }

    def execute(run: mlp.Run | None) -> None:
        model = load_model(args.model)
        processor = helper.get_processor(model.tokenizer)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        results = [run_clip(model, processor, clip_id, args, out_dir) for clip_id in clip_ids]
        model_stats = pm.model_size(model)
        for result in results:
            print(f"\n=== {result['clip_id']} ===")
            print("Chain-of-Causation:\n", result["cot"][0])
            if result["min_ade"] is not None:
                print("minADE:", result["min_ade"], "meters")
        write_outputs(results, out_dir, args.include_gt)
        print(f"\nOutputs written to {out_dir}")
        if run is not None:
            log_results(run, results, out_dir, args, run_dir, model_stats)

    if args.no_track:
        execute(None)
        return

    # A local path means a distilled checkpoint; anything else is a HF repo id.
    is_local = Path(args.model).exists()
    model_source = f"path:{args.model}" if is_local else f"hf:{args.model}@main"
    context = mlp.evaluate if args.run_type == "eval" else mlp.infer
    with context(
        args.experiment,
        run_name=run_name,
        project=args.project,
        model=model_source,
        hf_datasets=[f"{DATASET_REPO}@main"],
        params=params,
        split=f"clip:{clip_ids[0]}" if len(clip_ids) == 1 else f"{len(clip_ids)}-clips",
        seed=args.seed,
        notes=args.notes,
        tracking_uri=args.tracking_uri,
    ) as run:
        execute(run)


if __name__ == "__main__":
    main()
