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

"""Tier 1 metrics for the depth-pruning demonstration (EC-254).

Everything here is computed from one inference pass that already happened, so
adding it to a run costs almost nothing. Three groups:

* **Accuracy** -- ADE / FDE / minADE, broken down by horizon. Cutting layers is
  expected to fail at distance first, so the aggregate number alone hides the
  failure being looked for.
* **Kinematic feasibility** -- jerk, lateral acceleration and per-waypoint
  bound violations, computed in physical units. This is the axis a vehicle
  engineer reads first.
* **Heading summary** -- net heading change and lateral offset per trajectory:
  the raw material for scene stratification and for checking a stated maneuver
  against the one actually driven.

Only xy is used. ``action_to_traj`` fixes z to a constant from t0, so z error
carries no information.
"""

from __future__ import annotations

import numpy as np
import torch

from alpamayo1_5.action_space.action_space import ActionSpace
from alpamayo1_5.geometry.rotation import so3_to_yaw_torch

# The logged future starts at t0 + 0.1s at 10Hz, so index i is time (i + 1) * 0.1s.
HORIZON_INDICES = {"1.0s": 9, "2.0s": 19, "3.0s": 29, "4.0s": 39, "5.0s": 49, "6.4s": 63}

# Comfort threshold, not a physical limit: above this a passenger notices.
LATERAL_ACCEL_COMFORT_MS2 = 4.0


def displacement_metrics(pred_xy: np.ndarray, gt_xy: np.ndarray) -> dict[str, float]:
    """Displacement error of K sampled trajectories against the logged future.

    Args:
        pred_xy: Predicted trajectories, shape ``[K, T, 2]``.
        gt_xy: Logged future, shape ``[T, 2]``.

    Returns:
        ADE/FDE for the best and the mean sample, plus ADE per horizon.
    """
    error = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=-1)  # [K, T]
    ade_per_sample = error.mean(axis=-1)
    fde_per_sample = error[:, -1]
    best = int(np.argmin(ade_per_sample))

    metrics = {
        "min_ade": float(ade_per_sample.min()),
        "min_fde": float(fde_per_sample.min()),
        "mean_ade": float(ade_per_sample.mean()),
        "mean_fde": float(fde_per_sample.mean()),
    }
    # Horizon breakdown follows the best sample, so it stays consistent with min_ade.
    for name, index in HORIZON_INDICES.items():
        if index < error.shape[1]:
            metrics[f"ade@{name}"] = float(error[best, : index + 1].mean())
            metrics[f"de@{name}"] = float(error[best, index])
    return metrics


def _denormalize(action: torch.Tensor, action_space: ActionSpace) -> tuple[torch.Tensor, ...]:
    """Undo the action-space normalization. The two channels differ ~26x in sigma."""
    accel = action[..., 0] * action_space.accel_std.to(action) + action_space.accel_mean.to(action)
    kappa = action[..., 1] * action_space.curvature_std.to(action) + action_space.curvature_mean.to(
        action
    )
    return accel, kappa


def kinematics(
    action_space: ActionSpace,
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
) -> dict[str, float]:
    """Kinematic feasibility of predicted trajectories, in physical units.

    The trajectory is projected back into the unicycle action space, then
    denormalized -- ``is_within_bounds`` is reported alongside as a hard canary
    only. Its gates (accel +-9.8 m/s2, curvature +-0.33 /m) sit around 12-14
    sigma and it collapses all 64 waypoints with ``all(dim=-1)``, so it answers
    "did anything explode", never "is this comfortable to ride in".

    Args:
        action_space: The model's action space, holding the checkpoint's
            normalization constants and dt.
        history_xyz: Ego history, shape ``[K, N_hist, 3]``.
        history_rot: Ego history rotations, shape ``[K, N_hist, 3, 3]``.
        pred_xyz: Predicted trajectories, shape ``[K, T, 3]``.
        pred_rot: Predicted rotations, shape ``[K, T, 3, 3]``.

    Returns:
        Jerk, lateral acceleration and per-waypoint violation rates.
    """
    action, states = action_space.traj_to_action(
        traj_history_xyz=history_xyz,
        traj_history_rot=history_rot,
        traj_future_xyz=pred_xyz,
        traj_future_rot=pred_rot,
        output_all_states=True,
    )
    accel, kappa = _denormalize(action, action_space)
    speed = states[..., 0]  # v is already physical; states[..., 1] is the normalized accel

    dt = action_space.dt
    jerk = torch.diff(accel, dim=-1) / dt
    lateral_accel = speed**2 * kappa

    accel_lo, accel_hi = action_space.accel_bounds
    kappa_lo, kappa_hi = action_space.curvature_bounds
    accel_violation = ((accel < accel_lo) | (accel > accel_hi)).float()
    kappa_violation = ((kappa < kappa_lo) | (kappa > kappa_hi)).float()

    def stat(tensor: torch.Tensor) -> tuple[float, float]:
        flat = tensor.abs().flatten().float()
        return float(flat.mean()), float(torch.quantile(flat, 0.95))

    jerk_mean, jerk_p95 = stat(jerk)
    lat_mean, lat_p95 = stat(lateral_accel)
    return {
        "jerk_mean": jerk_mean,
        "jerk_p95": jerk_p95,
        "lat_accel_mean": lat_mean,
        "lat_accel_p95": lat_p95,
        "lat_accel_over_4_ratio": float(
            (lateral_accel.abs() > LATERAL_ACCEL_COMFORT_MS2).float().mean()
        ),
        "accel_violation_rate": float(accel_violation.mean()),
        "curvature_violation_rate": float(kappa_violation.mean()),
        # Hard canary: fraction of samples where every waypoint passed.
        "within_bounds_ratio": float(action_space.is_within_bounds(action).float().mean()),
        "speed_mean": float(speed.abs().mean()),
    }


def heading_summary(pred_xyz: torch.Tensor, pred_rot: torch.Tensor) -> dict[str, float]:
    """Net heading change and lateral offset of the best-effort mean trajectory.

    These two numbers are what separates a straight run from a curve and, more
    importantly, a lane change (small heading, large offset) from either. A lane
    change is the maneuver most likely to break first under aggressive pruning
    and the one a heading-only rule silently misses.
    """
    yaw = so3_to_yaw_torch(pred_rot)  # [K, T]
    net_heading = yaw[..., -1] - yaw[..., 0]
    net_heading = torch.atan2(torch.sin(net_heading), torch.cos(net_heading))
    lateral_offset = pred_xyz[..., -1, 1]
    return {
        "net_heading_deg": float(torch.rad2deg(net_heading).mean()),
        "net_heading_abs_deg": float(torch.rad2deg(net_heading).abs().mean()),
        "lateral_offset_m": float(lateral_offset.mean()),
        "lateral_offset_abs_m": float(lateral_offset.abs().mean()),
    }


def classify_scene(net_heading_abs_deg: float, lateral_offset_abs_m: float) -> str:
    """Label a maneuver from egomotion alone, per the EC-254 thresholds."""
    if net_heading_abs_deg > 20.0:
        return "curve"
    if lateral_offset_abs_m > 2.0:
        return "lane_change"
    if net_heading_abs_deg < 5.0 and lateral_offset_abs_m < 1.0:
        return "straight"
    return "other"


def cot_stats(tokenizer, cot_texts: list[str]) -> dict[str, float]:
    """Length of the generated reasoning, in tokens.

    A pruned backbone that simply stops reasoning looks fast and scores fine on
    trajectories; the token count is what exposes it.
    """
    lengths = []
    for text in cot_texts:
        try:
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
        except Exception:
            lengths.append(len(text.split()))
    if not lengths:
        return {}
    return {
        "cot_tokens_mean": float(np.mean(lengths)),
        "cot_tokens_min": float(np.min(lengths)),
        "cot_tokens_max": float(np.max(lengths)),
    }


def model_size(model: torch.nn.Module) -> dict[str, float]:
    """Parameter count and peak inference memory -- the "does it fit" column."""
    params = sum(p.numel() for p in model.parameters())
    stats = {
        "params_billions": params / 1e9,
        "param_bytes_gb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9,
    }
    if torch.cuda.is_available():
        stats["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return stats
