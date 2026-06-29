"""Colab bridge client.

Drive a Colab kernel from outside. The remote side is a tiny Flask app
running in the user's Colab kernel; we talk to it over HTTPS via a
cloudflared tunnel. The user's token guards the tunnel.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    result: Optional[str]
    error: Optional[str]
    truncated: bool = False
    elapsed_s: float = 0.0
    raw: dict = field(default_factory=dict)

    def __str__(self) -> str:
        out = []
        if self.stdout:
            out.append(self.stdout.rstrip())
        if self.result is not None and self.result != "None":
            out.append(f"=> {self.result}")
        if self.stderr:
            out.append(f"[stderr]\n{self.stderr.rstrip()}")
        if self.error:
            out.append(f"[error]\n{self.error.rstrip()}")
        if self.truncated:
            out.append("...[output truncated]...")
        return "\n".join(out) if out else "(no output)"


class ColabClient:
    """Drive a Colab kernel exposed by the bridge notebook."""

    def __init__(self, url: str, token: str, *, timeout: float = 300.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["X-Bridge-Token"] = token
        # Note: do NOT set Content-Type here. /upload needs multipart/form-data
        # (auto-set by requests when files= is used); /exec sets it per-call.

    # ---- low-level ----

    def health(self) -> dict:
        r = self.session.get(f"{self.url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def exec(self, code: str, *, timeout: Optional[float] = None) -> ExecResult:
        t0 = time.time()
        r = self.session.post(
            f"{self.url}/exec",
            data=json.dumps({"code": code}),
            headers={"Content-Type": "application/json"},
            timeout=timeout or self.timeout,
        )
        elapsed = time.time() - t0
        r.raise_for_status()
        body = r.json()
        return ExecResult(
            ok=body.get("ok", False),
            stdout=body.get("stdout", ""),
            stderr=body.get("stderr", ""),
            result=body.get("result"),
            error=body.get("error"),
            truncated=body.get("truncated", False),
            elapsed_s=elapsed,
            raw=body,
        )

    def upload(self, local_path: str, remote_path: Optional[str] = None) -> dict:
        p = Path(local_path)
        remote = remote_path or f"/content/{p.name}"
        with p.open("rb") as f:
            r = self.session.post(
                f"{self.url}/upload",
                files={"file": (p.name, f)},
                data={"path": remote},
                timeout=300,
            )
        r.raise_for_status()
        return r.json()

    def download(self, remote_path: str, local_path: Optional[str] = None) -> Path:
        r = self.session.get(
            f"{self.url}/download",
            params={"path": remote_path},
            timeout=300,
        )
        r.raise_for_status()
        out = Path(local_path or Path(remote_path).name)
        out.write_bytes(r.content)
        return out

    # ---- high-level ----

    def run_notebook(self, nb_path: str, *, stop_on_error: bool = True) -> list[ExecResult]:
        """Execute every code cell in a Jupyter notebook against the remote kernel."""
        import nbformat
        nb = nbformat.read(nb_path, as_version=4)
        results: list[ExecResult] = []
        n = len(nb.cells)
        for i, cell in enumerate(nb.cells, 1):
            if cell.cell_type != "code":
                continue
            src = "".join(cell.source)
            first = src.splitlines()[0] if src else "(empty)"
            print(f"[{i}/{n}] {first[:60]}")
            res = self.exec(src)
            print(res)
            results.append(res)
            if not res.ok and stop_on_error:
                print(f"  stopping at cell {i} due to error")
                break
        return results

    def set_var(self, name: str, code: str) -> ExecResult:
        return self.exec(f"{name} = {code}")

    def get_var(self, name: str) -> ExecResult:
        return self.exec(name)
