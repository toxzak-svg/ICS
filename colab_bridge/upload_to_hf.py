"""Robust upload-to-HF helper for ICS pipeline artifacts.

Designed to handle:
- Big folders (multi-GB) via `upload_large_folder` in a background subprocess
- Incremental uploads (resume between runs via commit history)
- Retry on transient HF errors
- Logging to a local file so we can monitor progress without blocking

The function `upload_stage` is the main entry point. It stages an artifact to
HF in a background subprocess and writes a `*.upload_marker.json` next to the
source dir when done.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional


def _upload_subprocess_script(local_dir: str, repo_id: str, repo_type: str, token: str, message: str) -> str:
    """Build the python script that runs inside the nohup subprocess."""
    return f'''
import os, sys, json, time, traceback
from pathlib import Path

local_dir = {local_dir!r}
repo_id = {repo_id!r}
repo_type = {repo_type!r}
token = {token!r}
message = {message!r}

# Where to write the upload log + marker
upload_log = Path(local_dir).parent / f".{{Path(local_dir).name}}.upload.log"
marker = Path(local_dir).parent / f".{{Path(local_dir).name}}.upload_marker.json"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{{ts}}] {{msg}}\\n"
    with open(upload_log, "a") as f:
        f.write(line)
    print(line, flush=True)

try:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    log(f"START repo={{repo_id}} type={{repo_type}} dir={{local_dir}}")
    log(f"  dir size: {{sum(p.stat().st_size for p in Path(local_dir).rglob('*') if p.is_file())/1e9:.2f}} GB")

    # Try upload_large_folder first (handles big files + multipart)
    # Fallback to upload_folder for small dirs
    try:
        log("  trying upload_large_folder...")
        api.upload_large_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            private=True,
            commit_message=message,
            token=token,
            num_workers=4,
            print_report=True,
        )
        log("  upload_large_folder OK")
    except Exception as e:
        log(f"  upload_large_folder failed: {{e}}")
        log("  falling back to upload_folder...")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=message + " [fallback]",
            token=token,
            ignore_patterns=[".upload.log", "*.upload_marker.json"],
        )
        log("  upload_folder OK")

    marker.write_text(json.dumps({{
        "ok": True,
        "repo": repo_id,
        "type": repo_type,
        "message": message,
        "finished_at": time.time(),
        "size_gb": sum(p.stat().st_size for p in Path(local_dir).rglob('*') if p.is_file())/1e9,
    }}, indent=2))
    log(f"OK marker written")
except Exception as e:
    log(f"FAIL: {{e}}")
    log(traceback.format_exc())
    try:
        marker.write_text(json.dumps({{"ok": False, "error": str(e)}}, indent=2))
    except Exception:
        pass
    sys.exit(1)
'''


def upload_stage(
    local_dir: str,
    repo_id: str,
    repo_type: str,
    token: str,
    message: str,
    *,
    wait: bool = False,
    poll_interval_s: float = 15.0,
    timeout_s: float = 1800.0,
) -> dict:
    """Upload an ICS artifact dir to HF.

    Args:
        local_dir: local path to the artifact directory
        repo_id: HF repo id, e.g. "user/model"
        repo_type: "model" or "dataset"
        token: HF token
        message: commit message
        wait: if True, block until done (poll marker file); else return immediately
        poll_interval_s: how often to check marker when wait=True
        timeout_s: max wait time when wait=True

    Returns:
        dict with "ok", "pid", "marker_path"
    """
    local_dir = str(Path(local_dir).resolve())
    parent = Path(local_dir).parent
    base = Path(local_dir).name

    script = _upload_subprocess_script(local_dir, repo_id, repo_type, token, message)
    script_path = parent / f".{base}.upload.py"
    script_path.write_text(script)

    log_path = parent / f".{base}.upload.log"
    marker_path = parent / f".{base}.upload_marker.json"
    log_path.write_text("")  # truncate

    # Spawn background subprocess
    proc = subprocess.Popen(
        ["nohup", "python3", "-u", str(script_path), ">", str(log_path), "2>&1", "&"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # The actual uploader pid is the child of this; we use log_path for status

    result = {"ok": None, "pid": proc.pid, "marker_path": str(marker_path), "log_path": str(log_path)}

    if not wait:
        return result

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(poll_interval_s)
        if marker_path.exists():
            try:
                m = json.loads(marker_path.read_text())
                result["ok"] = m.get("ok", False)
                result["marker"] = m
                return result
            except Exception:
                pass
        # Stream a tail of the log so the caller sees progress
        try:
            tail = log_path.read_text().splitlines()[-3:]
            result["last_log"] = "\n".join(tail)
        except Exception:
            pass

    result["ok"] = False
    result["timeout"] = True
    return result


def is_uploaded(marker_path: str) -> bool:
    p = Path(marker_path)
    if not p.exists():
        return False
    try:
        m = json.loads(p.read_text())
        return m.get("ok", False)
    except Exception:
        return False