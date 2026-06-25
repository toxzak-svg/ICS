"""Local smoke test for the bridge.

Spins up the same Flask app on a local port (no cloudflared), then drives
it via the client. Verifies the contract: /health, /exec (statements,
expressions, errors, large output), and a round-trip variable.
"""
import io
import json
import os
import sys
import threading
import time
import traceback

# Run the server's source inline so we share its globals.
SERVER_PATH = os.path.join(os.path.dirname(__file__), "build.py")
# The server source lives in cell 3 of the generated notebook.
import nbformat
NB_PATH = os.path.join(os.path.dirname(__file__), "colab_bridge.ipynb")
nb = nbformat.read(NB_PATH, as_version=4)
SERVER_CELL = "".join(nb.cells[2].source)  # 0=md, 1=install, 2=server

ns = {"__name__": "__main__"}
exec(SERVER_CELL, ns)
app = ns["app"]
TOKEN = ns["TOKEN"]

# Start server in a background thread
PORT = 9876
def _run():
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False, threaded=True)
t = threading.Thread(target=_run, daemon=True)
t.start()
time.sleep(1.5)

# Patch the server's globals so our injected variables survive /exec calls
# (the server already uses globals(); we just need to set the same ones).
ns["my_var"] = 42
ns["my_list"] = [1, 2, 3]

# Import the client from this folder
sys.path.insert(0, os.path.dirname(__file__))
from client import ColabClient  # noqa: E402

c = ColabClient(f"http://127.0.0.1:{PORT}", TOKEN)

# 1. health
h = c.health()
print("health:", h)
assert h["ok"], "health check failed"
assert h["kernel"] == "python3"

# 2. simple statement
r = c.exec("x = 7 + 3")
print("x = 7 + 3 ->", r)
assert r.ok
assert r.result is None  # statement, no expression

# 3. expression result (last line)
r = c.exec("a = 5\nb = 9\na * b")
print("a*b ->", r)
assert r.ok
assert r.result == "45"

# 4. capture stdout
r = c.exec("print('hello from colab')\nprint('line 2')")
print("print ->", r)
assert r.ok
assert "hello from colab" in r.stdout
assert "line 2" in r.stdout

# 5. error propagation
r = c.exec("1/0")
print("1/0 ->", r)
assert not r.ok
assert "ZeroDivisionError" in r.error

# 6. variable persistence across calls (shared kernel)
r = c.exec("my_var * 2")
print("my_var*2 ->", r)
assert r.result == "84"

# 7. import sticks
r = c.exec("import math\nmath.pi")
print("math.pi ->", r)
assert r.result[:6] == "3.1415"
r = c.exec("math.e")
print("math.e ->", r)
assert r.result[:6] == "2.7182"

# 8. unauthorized request is rejected
import requests
bad = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
print("unauth status:", bad.status_code)
assert bad.status_code == 401

# 9. large output is capped
r = c.exec("print('x' * (3 * 1024 * 1024))")
print("large output ->", len(r.stdout), "chars, truncated =", r.truncated)
assert r.truncated
assert len(r.stdout) <= 1.5 * 1024 * 1024

print("\nAll smoke tests passed.")
