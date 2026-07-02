"""Tiny bridge client — uses env vars BRIDGE_URL + BRIDGE_TOKEN.

Usage:
  python scripts/bridge.py 'import torch; torch.cuda.is_available()'
  python scripts/bridge.py < code.py
  python scripts/bridge.py --json '{"timeout": 600, "code": "..."}'
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error


def call(code: str, timeout: int = 300) -> dict:
    url = os.environ.get("BRIDGE_URL", "").rstrip("/")
    tok = os.environ.get("BRIDGE_TOKEN", "")
    if not url or not tok:
        sys.exit("BRIDGE_URL and BRIDGE_TOKEN must be set in env")
    body = json.dumps({"code": code, "timeout": timeout}).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/exec",
        data=body,
        method="POST",
        headers={"X-Bridge-Token": tok, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout + 30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}"}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--json":
        payload = json.loads(sys.argv[2])
        resp = call(payload.get("code", ""), int(payload.get("timeout", 300)))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--file":
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            code = f.read()
        timeout = int(sys.argv[3]) if len(sys.argv) >= 4 else 300
        resp = call(code, timeout)
    elif len(sys.argv) >= 2:
        resp = call(sys.argv[1])
    else:
        resp = call(sys.stdin.read())
    print(json.dumps(resp, indent=2)[:8000])
    if not resp.get("ok"):
        sys.exit(2)


if __name__ == "__main__":
    main()