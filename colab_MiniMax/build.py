"""Generate the MiniMax-in-Colab notebook.

Run once: `python build.py` -> writes `MiniMax_colab.ipynb` next to this file.
"""
from __future__ import annotations

import json
from pathlib import Path

# --- cell helpers ----------------------------------------------------------


def md(source: str, *, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"id": cell_id},
        "source": source.splitlines(keepends=True),
    }


def code(source: str, *, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": cell_id, "collapsed": False},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# --- cells -----------------------------------------------------------------

INTRO_MD = """# MiniMax in Google Colab

A chat interface for MiniMax's M-series models, running entirely in Google Colab.

MiniMax exposes an **OpenAI-compatible** API, so we use the `openai` SDK pointed
at MiniMax's `base_url` — no MiniMax-specific client needed.

## Quick start

1. Get a MiniMax API key from <https://platform.minimax.io/user-center/apikeys>
2. Paste it in the form field in cell 2 (or store it in Colab secrets — recommended)
3. `Runtime > Run all`
4. Click the public Gradio link to open the chat UI

Free Colab tier is fine — we call MiniMax's API, no GPU needed.
"""

INSTALL_CODE = """# @title 1. Install dependencies
!pip install -q openai gradio
"""

KEY_CODE = r'''# @title 2. Configure API key
# @markdown **Option A**: paste your key directly (visible in the notebook)
MINIMAX_API_KEY = ""  # @param {type:"string"}

# @markdown **Option B (recommended)**: store it in Colab secrets, then leave the field above empty.
# @markdown - Click the key icon in the left sidebar
# @markdown - Add a secret named `MINIMAX_API_KEY` and toggle "Notebook access" on

import os

try:
    from google.colab import userdata
    if not MINIMAX_API_KEY:
        MINIMAX_API_KEY = userdata.get('MINIMAX_API_KEY')
except (ImportError, Exception):
    pass

if not MINIMAX_API_KEY:
    raise ValueError(
        "No API key found. Either paste it above or add it to Colab secrets."
    )

os.environ["MINIMAX_API_KEY"] = MINIMAX_API_KEY
print("\u2713 API key loaded")
'''

CONFIG_CODE = r'''# @title 3. Configure model
# @markdown Pick a model and tune the behavior:
SYSTEM_PROMPT = "You are a helpful assistant."  # @param {type:"string"}
MODEL = "MiniMax-M3"  # @param ["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.5", "MiniMax-M2.5-highspeed", "MiniMax-M2.1", "MiniMax-M2.1-highspeed", "MiniMax-M2"]
TEMPERATURE = 1.0  # @param {type:"slider", min:0, max:2, step:0.1}
MAX_TOKENS = 2048  # @param {type:"integer"}
SHOW_THINKING = True  # @param {type:"boolean"}
# @markdown Toggle `SHOW_THINKING` to expose M3's `reasoning_split` mode — separates
# @markdown the model's chain-of-thought from the final answer. M2.x models always
# @markdown think (cannot be disabled); the toggle has no effect for them.

from openai import OpenAI

# MiniMax exposes an OpenAI-compatible endpoint. Pick the right base URL for your region.
# Global:   https://api.minimax.io/v1
# Mainland: https://api.MiniMax.chat/v1
BASE_URL = "https://api.minimax.io/v1"

client = OpenAI(api_key=MINIMAX_API_KEY, base_url=BASE_URL)
print(f"Ready: {MODEL} @ temp={TEMPERATURE}, max_tokens={MAX_TOKENS}, thinking={SHOW_THINKING}")
print(f"Endpoint: {BASE_URL}")
'''

LAUNCH_CODE = r'''# @title 4. Launch chat UI
import gradio as gr


def _build_messages(message, history, system_prompt):
    msgs = [{"role": "system", "content": system_prompt}]
    for user_msg, bot_msg in history:
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": bot_msg})
    msgs.append({"role": "user", "content": message})
    return msgs


def chat(message, history):
    """Stream a response from MiniMax given the running conversation."""
    messages = _build_messages(message, history, SYSTEM_PROMPT)

    extra_body = {"reasoning_split": True} if SHOW_THINKING else {}

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
        extra_body=extra_body or None,
    )

    # MiniMax streams reasoning_details[].text as *cumulative* strings
    # (each delta contains the full reasoning text so far). We track the
    # previous cumulative length and emit only the new tail.
    reasoning_seen_len = 0
    text_parts = []
    thinking_parts = []
    show_thinking = SHOW_THINKING

    for chunk in stream:
        delta = chunk.choices[0].delta

        if show_thinking:
            details = getattr(delta, "reasoning_details", None) or []
            for detail in details:
                # detail can be a pydantic model or a dict depending on SDK path
                text = None
                if isinstance(detail, dict):
                    text = detail.get("text")
                else:
                    text = getattr(detail, "text", None)
                if text and len(text) > reasoning_seen_len:
                    tail = text[reasoning_seen_len:]
                    reasoning_seen_len = len(text)
                    thinking_parts.append(tail)
                    # Stream the thinking pane live
                    yield (
                        "**Thinking:**\n\n"
                        + "".join(thinking_parts).strip()
                        + ("\n\n---\n\n" if text_parts else "")
                        + "".join(text_parts)
                    )

        if delta.content:
            text_parts.append(delta.content)
            if show_thinking and thinking_parts:
                yield (
                    "**Thinking:**\n\n"
                    + "".join(thinking_parts).strip()
                    + "\n\n---\n\n"
                    + "".join(text_parts)
                )
            else:
                yield "".join(text_parts)

    # M2.x always thinks but never exposes reasoning_details via the
    # OpenAI-compatible endpoint. Fall back to the final content (which
    # may contain embedded <think>...</think> tags).
    if not thinking_parts and not text_parts:
        yield "(no response)"


gr.ChatInterface(
    chat,
    title="MiniMax in Colab",
    description=(
        f"Model: `{MODEL}` \u00b7 Temperature: {TEMPERATURE} "
        f"\u00b7 Max tokens: {MAX_TOKENS} \u00b7 Thinking: {SHOW_THINKING}"
    ),
    examples=[
        "Explain the difference between linear and logarithmic attention.",
        "Write a Python function to detect a cycle in a directed graph.",
        "I'm planning a 3-day trip to Kyoto in November. What should I prioritize?",
    ],
).launch(share=True, debug=False, height=600)
'''

NOTES_MD = """## Notes

- `share=True` produces a public Gradio URL valid for 72 hours.
- Chat history lives in the Gradio session only. Re-running cells resets it.
- To stop the app: interrupt the cell (`Runtime > Interrupt execution`) or close the tab.
- Free Colab tier is sufficient — this calls MiniMax's API, no GPU used.

## Endpoints

| Region    | Base URL                          |
| --------- | --------------------------------- |
| Global    | `https://api.minimax.io/v1`       |
| Mainland  | `https://api.MiniMax.chat/v1`     |

To switch, edit `BASE_URL` in cell 3.

## Models

| Model                 | Context   | Notes                                                  |
| --------------------- | --------- | ------------------------------------------------------ |
| `MiniMax-M3`             | 1,000,000 | Latest. Agentic reasoning, tool use, long context.     |
| `MiniMax-M2.7`           | 204,800   | Self-improving, ~60 tps.                               |
| `MiniMax-M2.7-highspeed` | 204,800   | Same as M2.7, ~100 tps.                                |
| `MiniMax-M2.5`           | 204,800   | Peak performance / value.                              |
| `MiniMax-M2.5-highspeed` | 204,800   | ~100 tps.                                              |
| `MiniMax-M2.1`           | 204,800   | Multilingual code, enhanced dev experience.            |
| `MiniMax-M2.1-highspeed` | 204,800   | ~100 tps.                                              |
| `MiniMax-M2`             | 204,800   | Agentic, advanced reasoning.                           |

M2.x always thinks (cannot be disabled). M3 honors `reasoning_split` via the `SHOW_THINKING` toggle.

## Troubleshooting

- **`NameError: name 'MINIMAX_API_KEY' is not defined`** — you ran the chat cell without running the key cell first.
- **`AuthenticationError` / 401** — bad API key, or key has no credit. Check <https://platform.minimax.io/user-center/basic-information/balance>.
- **`temperature` out of range** — MiniMax rejects values outside `[0, 2]`. Default is `1.0`.
- **Public URL never shows** — Colab sometimes blocks Gradio sharing on certain networks. Try `share=False` and use the in-notebook preview instead.
- **Thinking pane empty for M2.x** — expected; M2.x always thinks but the toggle is only meaningful for M3.
"""


cells = [
    md(INTRO_MD, cell_id="intro"),
    code(INSTALL_CODE, cell_id="install"),
    code(KEY_CODE, cell_id="key"),
    code(CONFIG_CODE, cell_id="config"),
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
    out = Path(__file__).resolve().parent / "MiniMax_colab.ipynb"
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
