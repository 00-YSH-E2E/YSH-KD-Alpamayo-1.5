"""Self-contained ML_Platform tracking helper.

Implements the recording rules of the ``ml-platform`` skill: pick the run type,
attach the lineage tags that make a run reproducible, and refuse to push large
artifacts through the MLflow proxy (the tracking server runs on a Jetson Orin
Nano with 1GB of memory and a single worker).

This is a standalone replacement for ``examples/ml_platform_track.py``. It
closes the gaps documented in the skill's section 13: ``run_type`` is always
tagged, ``hf_models=`` is a real argument, ``infer()`` exists as its own
context, environment tags are collected automatically, artifact sizes are
enforced rather than suggested, and unresolved Hugging Face revisions are
reported loudly instead of being written through as-is.

Usage::

    with evaluate("my-project", model="hf:nvidia/Some-Model@main",
                  hf_datasets=["nvidia/Some-Dataset@main"]) as ev:
        ev.score(0.42)

The run must be started from inside the git repository that holds the code
being run: MLflow reads ``cwd`` to autolog ``mlflow.source.git.commit``, and
the hub trusts that tag over anything set by hand.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Iterable, Mapping

try:
    import mlflow
except ImportError:  # only required when a run is actually recorded
    mlflow = None

DEFAULT_TRACKING_URI = "http://ysh-jetson-orin-nano.tail4570ef.ts.net:5000"

# Size ceilings from the skill's section 4. A single upload that exceeds these
# stalls the whole tracking server, and MLflow artifacts are invisible on the
# hub UI (:8000) anyway -- anything bigger belongs on Hugging Face.
MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
MAX_RUN_ARTIFACT_BYTES = 50 * 1024 * 1024

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_METRIC_KEY_RE = re.compile(r"[^0-9A-Za-z_\-./ ]")
_SOURCE_PREFIXES = ("hf:", "path:", "dvc:", "mlflow:", "s3:", "http:", "https:")


def env_tags(seed: int | None = None, notes: str | None = None) -> dict[str, str]:
    """Collect environment and narrative tags. Never raises."""
    tags = {
        "env.host": socket.gethostname(),
        "env.python": platform.python_version(),
        "entrypoint": sys.argv[0],
        "cmd": " ".join(shlex.quote(a) for a in sys.argv),
    }
    try:
        import torch

        tags["env.torch"] = torch.__version__
        tags["env.cuda"] = torch.version.cuda or ""
        if torch.cuda.is_available():
            tags["env.gpu"] = torch.cuda.get_device_name(0)
            tags["env.gpu_count"] = str(torch.cuda.device_count())
    except Exception:
        pass
    if seed is not None:
        tags["seed"] = str(seed)
    if notes is not None:
        tags["notes"] = notes
    return {k: v for k, v in tags.items() if v}


def git_tags() -> dict[str, str]:
    """Read git coordinates from ``cwd``. Never raises."""

    def run(*args: str) -> str:
        try:
            out = subprocess.run(
                args, capture_output=True, text=True, timeout=10, check=False
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            return ""

    tags = {
        "git_commit": run("git", "rev-parse", "HEAD"),
        "git_branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_remote": run("git", "config", "--get", "remote.origin.url"),
    }
    if tags["git_commit"]:
        tags["git_dirty"] = "true" if run("git", "status", "--porcelain") else "false"
    return {k: v for k, v in tags.items() if v}


def _hf_token() -> str | None:
    """Return an HF token from the environment or the CLI cache, if any."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    path = os.path.expanduser("~/.cache/huggingface/token")
    try:
        token = open(path).read().strip()
        return token or None
    except OSError:
        return None


def parse_coord(coord: str) -> tuple[str, str, str]:
    """Split ``owner/name[:kind][@rev][#subpath]`` into (repo, revision, subpath)."""
    coord = coord.strip()
    for prefix in ("hf:", "hf/"):
        if coord.startswith(prefix):
            coord = coord[len(prefix) :]
    subpath = ""
    if "#" in coord:
        coord, subpath = coord.split("#", 1)
    revision = "main"
    if "@" in coord:
        coord, revision = coord.rsplit("@", 1)
    return coord, revision, subpath


def hf_sha(repo: str, revision: str = "main", kind: str = "model") -> str | None:
    """Resolve a Hugging Face revision to its 40-character commit sha.

    Returns ``None`` when the hub cannot be reached or the repo is gated
    without a token -- the caller decides whether that is fatal.
    """
    if _SHA_RE.match(revision):
        return revision
    api_repo = repo.split(":", 1)[0]
    section = "models" if kind == "model" else "datasets"
    url = f"https://huggingface.co/api/{section}/{api_repo}/revision/{revision}"
    request = urllib.request.Request(url)
    token = _hf_token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            sha = json.load(response).get("sha", "")
        return sha if _SHA_RE.match(sha) else None
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return None


def flatten_params(obj: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a nested config into dotted ``key: str`` pairs for MLflow params."""
    flat: dict[str, str] = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            flat.update(flatten_params(value, f"{prefix}{key}."))
    elif isinstance(obj, (list, tuple)):
        flat[prefix.rstrip(".")] = json.dumps(list(obj))
    else:
        flat[prefix.rstrip(".")] = str(obj)
    return flat


def _load_params(params: Any) -> dict[str, str]:
    """Accept a dict, or a path to a YAML/JSON config, and flatten it."""
    if params is None:
        return {}
    if isinstance(params, str):
        with open(params) as handle:
            if params.endswith((".yaml", ".yml")):
                import yaml

                params = yaml.safe_load(handle)
            else:
                params = json.load(handle)
    return flatten_params(params)


class Run:
    """Thin wrapper over an active MLflow run that enforces the recording rules."""

    def __init__(self, run_type: str) -> None:
        self.run_type = run_type
        self.run_id = mlflow.active_run().info.run_id
        self._metrics: set[str] = set()
        self._tags: dict[str, str] = {}
        self._artifact_bytes = 0
        self.unresolved: list[str] = []

    # -- writing -----------------------------------------------------------
    def tag(self, key: str, value: Any) -> None:
        """Set one tag. Tags are free -- when in doubt, leave one."""
        self._tags[key] = str(value)
        mlflow.set_tag(key, str(value))

    def param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    def params(self, params: Any) -> None:
        """Log a dict or a config file path as params, flattened."""
        flat = _load_params(params)
        if flat:
            mlflow.log_params(flat)

    def metric(self, key: str, value: float, step: int | None = None) -> None:
        key = _METRIC_KEY_RE.sub("_", key)  # MLflow rejects anything outside this alphabet
        self._metrics.add(key)
        mlflow.log_metric(key, float(value), step=step)

    def score(self, value: float) -> None:
        """Log the single representative number for an eval run."""
        self.metric("score", value)

    def by_scenario(self, scores: Mapping[str, Any]) -> None:
        """Log per-scenario breakdowns as ``scenario.<key>.score`` / ``.count``."""
        for name, value in scores.items():
            key = _METRIC_KEY_RE.sub("_", str(name))
            if isinstance(value, (tuple, list)) and len(value) == 2:
                self.metric(f"scenario.{key}.score", value[0])
                self.metric(f"scenario.{key}.count", value[1])
            else:
                self.metric(f"scenario.{key}.score", value)

    def result_path(self, uri: str) -> None:
        """Record where the outputs actually landed."""
        self.tag("output_uri", uri)
        self.tag("eval_result_path", uri)

    def artifact(self, path: str, name: str | None = None) -> bool:
        """Upload a small file. Refuses anything over the size ceilings."""
        size = os.path.getsize(path)
        if size > MAX_ARTIFACT_BYTES:
            print(
                f"[ml_platform] SKIP artifact {path} ({size / 1e6:.1f}MB > 10MB). "
                "Put it on Hugging Face and tag output_uri instead.",
                file=sys.stderr,
            )
            return False
        if self._artifact_bytes + size > MAX_RUN_ARTIFACT_BYTES:
            print(
                f"[ml_platform] SKIP artifact {path}: run would exceed the 50MB budget.",
                file=sys.stderr,
            )
            return False
        mlflow.log_artifact(path, artifact_path=name)
        self._artifact_bytes += size
        return True

    def checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Checkpoints always go to Hugging Face -- never through MLflow."""
        raise RuntimeError(
            "Checkpoints never go to MLflow (skill section 4). Upload to Hugging Face "
            "and record the coordinate with run.tag('output_uri', 'hf:owner/name@<sha>')."
        )

    # -- lineage -----------------------------------------------------------
    def hf_datasets(self, coords: Iterable[str]) -> None:
        self._hf_coords(coords, kind="dataset", tag_key="hf_datasets")

    def hf_models(self, coords: Iterable[str]) -> None:
        self._hf_coords(coords, kind="model", tag_key="hf_models")

    def _hf_coords(self, coords: Iterable[str], kind: str, tag_key: str) -> None:
        pinned: list[str] = []
        for raw in coords:
            repo, revision, _ = parse_coord(raw)
            if kind == "model" and ":" not in repo:
                repo = f"{repo}:model"  # the hub keys model repos with a :model suffix
            sha = hf_sha(repo, revision, kind=kind)
            if sha is None:
                self.unresolved.append(f"{repo}@{revision}")
                sha = revision
            pinned.append(f"{repo}@{sha}")
            self.tag(f"hf.{repo}", sha)
        if pinned:
            self.tag(tag_key, json.dumps(pinned))

    def model_source(self, source: str) -> str:
        """Tag the evaluated/inferred model, pinning ``hf:`` coordinates to a sha."""
        if source.startswith("hf:"):
            repo, revision, subpath = parse_coord(source)
            if ":" not in repo:
                repo = f"{repo}:model"
            sha = hf_sha(repo, revision, kind="model")
            if sha is None:
                self.unresolved.append(f"{repo}@{revision}")
                sha = revision
            source = f"hf:{repo}@{sha}" + (f"#{subpath}" if subpath else "")
            self.hf_models([f"{repo}@{sha}"])
        elif not source.startswith(_SOURCE_PREFIXES):
            source = f"path:{source}"
        self.tag("model_source", source)
        return source

    # -- validation --------------------------------------------------------
    def check(self) -> list[str]:
        """Return the list of rule violations for this run (empty means it passes)."""
        problems: list[str] = []
        if self.run_type not in ("train", "eval", "infer"):
            problems.append(f"run_type must be train|eval|infer, got {self.run_type!r}")
        if not self._tags.get("git_commit"):
            problems.append("git_commit missing -- was this run started outside a git repo?")
        if not (self._tags.get("hf_datasets") or self._tags.get("hf_models")):
            problems.append("no data coordinate: needs hf_datasets or hf_models")
        if self.run_type in ("eval", "infer") and not self._tags.get("model_source"):
            problems.append("model_source required for eval/infer runs")
        if self.run_type == "eval" and "score" not in self._metrics:
            problems.append("eval runs need a 'score' metric")
        if self.run_type == "infer" and not self._tags.get("output_uri"):
            problems.append("infer runs need an output_uri -- otherwise nothing was produced")
        if self.run_type == "train" and not (
            self._metrics & {"val_loss", "train_loss", "loss"}
        ):
            problems.append("train runs need val_loss / train_loss / loss")
        for coord in dict.fromkeys(self.unresolved):
            problems.append(f"unresolved HF revision (not a 40-char sha): {coord}")
        return problems


@contextmanager
def _run(
    experiment: str,
    run_type: str,
    run_name: str | None = None,
    project: str | None = None,
    params: Any = None,
    hf_datasets: Iterable[str] = (),
    hf_models: Iterable[str] = (),
    model: str | None = None,
    split: str | None = None,
    challenge: str | None = None,
    seed: int | None = None,
    notes: str | None = None,
    tracking_uri: str | None = None,
):
    """Open an MLflow run with the mandatory lineage tags already attached."""
    if mlflow is None:
        raise RuntimeError("MLflow is not installed. pip install 'mlflow>=3'")
    mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
                            or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment)
    started = time.time()
    with mlflow.start_run(run_name=run_name):
        run = Run(run_type)
        run.tag("run_type", run_type)
        for key, value in git_tags().items():
            run.tag(key, value)
        for key, value in env_tags(seed=seed, notes=notes).items():
            run.tag(key, value)
        if project:
            run.tag("project", project)
        if split:
            run.tag("eval_split", split)
        if challenge:
            run.tag("eval_challenge", challenge)
        if model:
            run.model_source(model)
        run.hf_datasets(hf_datasets)
        if hf_models:
            run.hf_models(hf_models)
        run.params(params)
        try:
            yield run
        finally:
            run.tag("duration_sec", str(int(time.time() - started)))
            problems = run.check()
            print(f"\n[ml_platform] run_id={run.run_id}  experiment={experiment}")
            if problems:
                print("[ml_platform] FAIL -- fix the script, not the run:")
                for problem in problems:
                    print(f"  - {problem}")
            else:
                print("[ml_platform] PASS -- all required coordinates recorded")
            if run._tags.get("git_dirty") == "true":
                print("[ml_platform] WARNING: repo is dirty, this run is not reproducible")


def track(experiment: str, **kwargs: Any):
    """Open a ``train`` run."""
    return _run(experiment, "train", **kwargs)


def evaluate(experiment: str, **kwargs: Any):
    """Open an ``eval`` run: labelled data, a representative ``score``."""
    return _run(experiment, "eval", **kwargs)


def infer(experiment: str, **kwargs: Any):
    """Open an ``infer`` run: no labels, outputs only, ``output_uri`` required."""
    return _run(experiment, "infer", **kwargs)
