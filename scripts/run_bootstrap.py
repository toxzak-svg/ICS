"""Push _bootstrap_qwen06.py to /content/ and run it via /exec.

Usage: set BRIDGE_URL + BRIDGE_TOKEN, then run this script.
"""
import os
import sys
import urllib.request
import urllib.error


def upload(local_path: str, remote_path: str, url: str, tok: str) -> dict:
    import requests  # type: ignore
    sess = requests.Session()
    sess.headers["X-Bridge-Token"] = tok
    with open(local_path, "rb") as f:
        r = sess.post(
            f"{url}/upload",
            files={"file": (os.path.basename(local_path), f)},
            data={"path": remote_path},
            timeout=600,
        )
    r.raise_for_status()
    return r.json()


def call(code: str, url: str, tok: str, timeout: int = 300) -> dict:
    import requests  # type: ignore
    sess = requests.Session()
    sess.headers["X-Bridge-Token"] = tok
    r = sess.post(
        f"{url}/exec",
        json={"code": code, "timeout": timeout},
        timeout=timeout + 30,
    )
    r.raise_for_status()
    return r.json()


def main():
    url = os.environ.get("BRIDGE_URL", "").rstrip("/")
    tok = os.environ.get("BRIDGE_TOKEN", "")
    if not url or not tok:
        sys.exit("BRIDGE_URL and BRIDGE_TOKEN must be set")
    print(f"Bridge: {url}")

    bootstrap_local = "colab_bridge/_bootstrap_qwen06.py"
    bootstrap_remote = "/content/_bootstrap_qwen06.py"
    print(f"\nUploading {bootstrap_local} -> {bootstrap_remote}")
    res = upload(bootstrap_local, bootstrap_remote, url, tok)
    print(f"  ok={res.get('ok')} size={res.get('size')}")

    # Run it. Bootstrap may take a while (download + install).
    print("\nExecuting bootstrap on remote...")
    code = f"exec(open({bootstrap_remote!r}).read())"
    resp = call(code, url, tok, timeout=900)
    print(f"  ok={resp.get('ok')}")
    print(f"  truncated={resp.get('truncated')}")
    if resp.get("stdout"):
        print("--- STDOUT ---")
        print(resp["stdout"])
    if resp.get("stderr"):
        print("--- STDERR ---")
        print(resp["stderr"])
    if resp.get("error"):
        print("--- ERROR ---")
        print(resp["error"])
    if not resp.get("ok"):
        sys.exit(2)


if __name__ == "__main__":
    main()