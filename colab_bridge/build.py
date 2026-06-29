"""Generate the Colab bridge notebook.

The bridge runs a Flask HTTP server inside the Colab kernel, then tunnels
it out via cloudflared (no auth, no signup). Any Python code sent to
the /exec endpoint runs in the same kernel as the user's other cells,
so state is shared and the user can keep working in their notebook
while Mavis drives it from outside.
"""
from __future__ import annotations

import json
from pathlib import Path


def md(source: str, *, cell_id: str) -> dict:
    return {
        "id": cell_id,
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str, *, cell_id: str) -> dict:
    return {
        "id": cell_id,
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"collapsed": False},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# --- cells -----------------------------------------------------------------

INTRO_MD = """# Colab Bridge — let Mavis drive your notebook

When you run the cells below, a small HTTP server starts inside the Colab kernel
and gets tunneled out via `cloudflared` to a public URL. Send Python code to that
URL and it executes in the same kernel as the rest of your notebook — state is shared,
imports stick, variables persist.

**You only need to do 3 things:**

1. Run cells 1–3 in order.
2. Copy the printed `Public URL` and `Token` to Mavis.
3. Keep this tab open. The bridge dies if Colab disconnects.

> Security: the URL is public. The token is the only thing standing between
> anyone on the internet and your Colab. Don't share the URL without the token.
"""

INSTALL_CODE = r'''# @title 1. Install + download cloudflared
!pip install -q flask flask-cors requests

import os, sys, platform, shutil, urllib.request, tarfile, tempfile, subprocess

# cloudflared: a single static binary, no auth, no signup. Anonymous tunnel.
CFD_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CFD_PATH = "/usr/local/bin/cloudflared"

if not os.path.exists(CFD_PATH):
    print("Downloading cloudflared...")
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as t:
        urllib.request.urlretrieve(CFD_URL, t.name)
    # It's a bare binary, not actually a tarball, despite the .deb/.tgz tricks.
    # Some releases ship a raw binary; just copy and chmod.
    shutil.copy(t.name, CFD_PATH)
    os.chmod(CFD_PATH, 0o755)
    print("Installed:", CFD_PATH)
else:
    print("cloudflared already at", CFD_PATH)

print(subprocess.check_output([CFD_PATH, "--version"]).decode().strip())
'''

SERVER_CODE = r'''# @title 2. Define the bridge server
import os, sys, io, ast, json, time, traceback, secrets, threading, subprocess, re
from flask import Flask, request, jsonify
from flask_cors import CORS

# A random token guards the tunnel. Anyone with this token can run code in the kernel.
TOKEN = secrets.token_urlsafe(24)

app = Flask(__name__)
CORS(app)


def _unauth():
    return jsonify({"ok": False, "error": "unauthorized"}), 401


def _check():
    tok = request.headers.get("X-Bridge-Token") or request.args.get("token")
    if tok != TOKEN:
        return _unauth()
    return None


@app.route("/health", methods=["GET"])
def health():
    err = _check()
    if err:
        return err
    return jsonify({
        "ok": True,
        "kernel": "python3",
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "globals": sorted([k for k in globals().keys() if not k.startswith("_")])[:50],
    })


@app.route("/exec", methods=["POST"])
def exec_code():
    err = _check()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    code = body.get("code", "")
    timeout = int(body.get("timeout", 300))

    if not isinstance(code, str) or not code.strip():
        return jsonify({"ok": False, "error": "empty code"}), 400

    # Cap runaway output (~2 MB total stdout+stderr).
    MAX_OUT = 2 * 1024 * 1024

    old_stdout, old_stderr = sys.stdout, sys.stderr
    buf_out, buf_err = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = buf_out, buf_err

    result_repr = None
    error = None
    try:
        tree = ast.parse(code, mode="exec")
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()
        if tree.body:
            exec(compile(tree, "<bridge>", "exec"), globals())
        if last_expr is not None:
            value = eval(compile(ast.Expression(last_expr.value), "<bridge>", "eval"), globals())
            result_repr = repr(value)
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    out = buf_out.getvalue()
    err_out = buf_err.getvalue()
    truncated = False
    if len(out) + len(err_out) > MAX_OUT:
        out = out[:MAX_OUT // 2]
        err_out = err_out[:MAX_OUT // 2]
        truncated = True

    return jsonify({
        "ok": error is None,
        "stdout": out,
        "stderr": err_out,
        "result": result_repr,
        "error": error,
        "truncated": truncated,
        "elapsed_s": None,  # filled by client if needed
    })


@app.route("/upload", methods=["POST"])
def upload():
    err = _check()
    if err:
        return err
    # multipart: file field, optional 'path' field
    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "error": "no file"}), 400
    dest = request.form.get("path") or f"/content/{f.filename}"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    f.save(dest)
    return jsonify({"ok": True, "path": dest, "size": os.path.getsize(dest)})


@app.route("/download", methods=["GET"])
def download():
    err = _check()
    if err:
        return err
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": f"not found: {path}"}), 404
    from flask import send_file
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


print(f"BRIDGE_TOKEN={TOKEN}")
print("Server defined. Run cell 3 to start it.")
'''

LAUNCH_CODE = r'''# @title 3. Start server + open tunnel
import threading, time, re, os, subprocess, signal

PORT = 9000
LOG = "/tmp/cloudflared.log"

# Wipe the previous log so URL extraction is clean on re-runs.
open(LOG, "w").close()

# Start the Flask server in a background thread. It shares the kernel
# with the rest of the notebook, so any code we exec via /exec lands here.
def _run():
    # use_reloader=False is critical; the reloader would fork and lose
    # access to this kernel's globals.
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False, threaded=True)

t = threading.Thread(target=_run, daemon=True)
t.start()
time.sleep(2)

# Sanity check
import urllib.request
try:
    r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
    print(f"Server health: {r.status} (in-kernel)")
except Exception as e:
    print(f"Server health FAILED: {e}")
    raise

# Launch cloudflared
proc = subprocess.Popen(
    [CFD_PATH, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
    stdout=open(LOG, "w"),
    stderr=subprocess.STDOUT,
    text=True,
    preexec_fn=os.setsid,
)

# Poll the log for the public URL. cloudflared takes 1-3s to negotiate.
url = None
for i in range(60):
    time.sleep(1)
    try:
        content = open(LOG).read()
    except FileNotFoundError:
        continue
    m = re.search(r"(https://[\w-]+\.trycloudflare\.com)", content)
    if m:
        url = m.group(1)
        break

if not url:
    print("Failed to get cloudflared URL. Last 50 lines of log:")
    print("\n".join(open(LOG).read().splitlines()[-50:]))
    raise RuntimeError("cloudflared did not produce a URL in 60s")

# Stash the URL/token in the kernel so the user can grab them programmatically.
BRIDGE_URL = url
BRIDGE_TOKEN = TOKEN

print()
print("=" * 70)
print("  BRIDGE READY")
print("=" * 70)
print(f"  Public URL:  {BRIDGE_URL}")
print(f"  Token:       {BRIDGE_TOKEN}")
print("=" * 70)
print()
print("Copy both to Mavis. Keep this tab open. If Colab disconnects, re-run cells 2+3.")
print()
print(f"Quick test: curl {BRIDGE_URL}/health -H 'X-Bridge-Token: {BRIDGE_TOKEN}'")
'''

NOTES_MD = """## Notes

- The bridge shares the kernel with the rest of your notebook. State (`import`s, variables, GPU context) persists across calls.
- Free Colab sessions time out after 90 min idle / 12 h max. Re-run cell 2+3 if the URL dies.
- The cloudflared tunnel is anonymous; URLs rotate on every start.
- If cloudflared is blocked on your network, edit cell 3 to use `ngrok` (requires a free authtoken) or `localtunnel` (npm-based).
- To stop: interrupt cell 3, or `pkill -f cloudflared` in a new cell.
"""


cells = [
    md(INTRO_MD, cell_id="intro"),
    code(INSTALL_CODE, cell_id="install"),
    code(SERVER_CODE, cell_id="server"),
    code(LAUNCH_CODE, cell_id="launch"),
    md(NOTES_MD, cell_id="notes"),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {
            "provenance": [],
            "authorship_tag": "ABX9TyM9LZ4yL1z7Vb6p1Z7X",
            "include_colab_link": True,
        },
        "accelerator": "CPU",
        "gpuClass": "standard",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "colab_bridge.ipynb"
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
