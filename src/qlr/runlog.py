"""Run directory bookkeeping: ``config.json``, ``env.json``, ``metrics.json``.

The layout is the one `cluster.local.md` prescribes, so `slurm/sync.sh pull`
finds these three files and leaves the heavy artifacts on the cluster.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import subprocess
import time

__all__ = ["RunDir"]


class RunDir:
    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def config(self, cfg: dict) -> None:
        self._write("config.json", cfg)

    def env(self) -> None:
        info = {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        try:
            import torch

            info |= {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except ImportError:  # pragma: no cover
            pass
        # The remote tree carries no .git, so provenance comes from the stamp
        # slurm/sync.sh writes at transfer time.
        stamp = pathlib.Path(".git_commit")
        if stamp.exists():
            info["git"] = dict(
                line.split("=", 1) for line in stamp.read_text().strip().splitlines() if "=" in line
            )
        else:
            try:
                info["git"] = {
                    "commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
                    ).strip()
                }
            except Exception:
                info["git"] = None
        self._write("env.json", info)

    def metrics(self, m: dict) -> None:
        m = dict(m)
        m["wall_seconds"] = time.time() - self.t0
        self._write("metrics.json", m)

    def artifact(self, name: str) -> pathlib.Path:
        return self.path / name

    def _write(self, name: str, obj: dict) -> None:
        with open(self.path / name, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
