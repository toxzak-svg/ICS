"""Upload a local script to colab and exec it.

Usage: BRIDGE_URL=... BRIDGE_TOKEN=... python scripts/run_remote.py <local> <remote>
Default remote = /content/<filename>
"""
import os
import sys
import urllib.request
import urllib.error
import json


def upload(local: str, remote: str, url: str, tok: str) -> dict:
    import requests
    sess = requests.Session()
    sess.headers["X-Bridge-Token"] = tok
    with open(local, "rb") as f:
        r = sess.post(
            f"{url}/upload",
            files={"file": (os.path.basename(local), f)},
            data={"path": remote},
            timeout=600,
        )
    r.raise_for_status()
    return r.json()


def call(code: str, url: str, tok: str, timeout: int = 300) -> dict:
    import requests
    sess = requests.Session()
    sess.headers["X-Bridge-Token"] = tok
    r = sess.post(
        f"{url}/exec",
        json={"code": code, "timeout": timeout},
        timeout=timeout + 60,
    )
    r.raise_for_status()
    return r.json()


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: run_remote.py <local-script> [remote-path] [timeout-s]")
    local = sys.argv[1]
    remote = sys.argv[2] if len(sys.argv) >= 3 else f"/content/{os.path.basename(local)}"
    timeout = int(sys.argv[3]) if len(sys.argv) >= 4 else 1800

    url = os.environ.get("BRIDGE_URL", "").rstrip("/")
    tok = os.environ.get("BRIDGE_TOKEN", "")
    if not url or not tok:
        sys.exit("BRIDGE_URL and BRIDGE_TOKEN must be set")

    print(f"BRIDGE_URL  = {url}")
    print(f"BRIDGE_TOKEN= {tok[:8]}...{tok[-4:]}")
    print(f"local  = {local}  ({os.path.getsize(local)} bytes)")
    print(f"remote = {remote}")
    print(f"timeout= {timeout}s")

    print("\nUploading...")
    res = upload(local, remote, url, tok)
    print(f"  ok={res.get('ok')} size={res.get('size')}")

    print(f"\nExec (timeout={timeout}s)...")
    code = f"exec(open({remote!r}).read())"
    t0 = __import__("time").time()
    resp = call(code, url, tok, timeout=timeout)
    dt = __import__("time").time() - t0
    print(f"  ok={resp.get('ok')} in {dt:.1f}s truncated={resp.get('truncated')}")
    if resp.get("stdout"):
        print("--- STDOUT ---")
        print(resp["stdout"][-12000:])
    if resp.get("stderr"):
        print("--- STDERR ---")
        print(resp["stderr"][-3000:])
    if resp.get("error"):
        print("--- ERROR ---")
        print(resp["error"][-3000:])
    if not resp.get("ok"):
        sys.exit(2)


if __name__ == "__main__":
    main()
