"""Validate the generated notebook."""
import json
import sys
from pathlib import Path

p = Path(__file__).resolve().parent / "MiniMax_colab.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))

assert nb["nbformat"] == 4, "wrong nbformat"
assert nb["nbformat_minor"] == 5, "wrong nbformat_minor"
assert len(nb["cells"]) == 6, f"expected 6 cells, got {len(nb['cells'])}"

print(f"OK: {p.name} ({p.stat().st_size} bytes)")
print(f"   nbformat {nb['nbformat']}.{nb['nbformat_minor']}, kernel={nb['metadata']['kernelspec']['name']}")
print(f"   cells ({len(nb['cells'])}):")
for i, c in enumerate(nb["cells"], 1):
    src = "".join(c["source"])
    first_line = src.splitlines()[0] if src else "(empty)"
    print(f"     {i}. {c['cell_type']:8s} - {first_line[:70]}")

# Quick sanity: every cell has source, code cells have empty outputs list
for c in nb["cells"]:
    assert c["source"], f"empty source in {c['cell_type']} cell"
    if c["cell_type"] == "code":
        assert "outputs" in c
        assert "execution_count" in c

print("\nAll checks passed.")
