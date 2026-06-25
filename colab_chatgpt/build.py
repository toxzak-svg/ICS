"""Generate the ChatGPT-in-Colab notebook.

Run once: `python build.py` -> writes `chatgpt_colab.ipynb` next to this file.
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

INTRO_MD = """# ChatGPT in Google Colab

A simple chat interface for OpenAI's ChatGPT models, running entirely in Google Colab.

## Quick start

1. Get an OpenAI API key from <https://platform.openai.com/api-keys>
2. Paste it in the form field in cell 2 (or store it in Colab secrets — recommended)
3. `Runtime > Run all`
4. Click the public Gradio link to open the chat UI

Free Colab tier is fine — we call OpenAI's API, no GPU needed.
"""

INSTALL_CODE = """# @title 1. Install dependencies
!pip install -q openai gradio
"""

KEY_CODE = r'''# @title 2. Configure API key
# @markdown **Option A**: paste your key directly (visible in the notebook)
OPENAI_API_KEY = ""  # @param {type:"string"}

# @markdown **Option B (recommended)**: store it in Colab secrets, then leave the field above empty.
# @markdown - Click the key icon in the left sidebar
# @markdown - Add a secret named `OPENAI_API_KEY` and toggle "Notebook access" on

import os

try:
    from google.colab import userdata
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
except (ImportError, Exception):
    pass

if not OPENAI_API_KEY:
    raise ValueError(
        "No API key found. Either paste it above or add it to Colab secrets."
    )

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
print("\u2713 API key loaded")
'''

CONFIG_CODE = r'''# @title 3. Configure model
# @markdown Pick a model and tune the behavior:
SYSTEM_PROMPT = "You are a helpful assistant."  # @param {type:"string"}
MODEL = "gpt-4o-mini"  # @param ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
TEMPERATURE = 0.7  # @param {type:"slider", min:0, max:2, step:0.1}
MAX_TOKENS = 1024  # @param {type:"integer"}

from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
print(f"Ready: {MODEL} @ temp={TEMPERATURE}, max_tokens={MAX_TOKENS}")
'''

LAUNCH_CODE = r'''# @title 4. Launch chat UI
import gradio as gr


def chat(message, history):
    """Stream a response from OpenAI given the running conversation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            response += delta
            yield response


gr.ChatInterface(
    chat,
    title="ChatGPT in Colab",
    description=(
        f"Model: `{MODEL}` \u00b7 Temperature: {TEMPERATURE} "
        f"\u00b7 Max tokens: {MAX_TOKENS}"
    ),
    examples=[
        "Explain quantum entanglement like I'm five.",
        "Write a Python function to flatten a nested list.",
        "What are three good names for a coffee shop?",
    ],
).launch(share=True, debug=False, height=600)
'''

NOTES_MD = """## Notes

- `share=True` produces a public Gradio URL valid for 72 hours.
- Chat history lives in the Gradio session only. Re-running cells resets it.
- To stop the app: interrupt the cell (`Runtime > Interrupt execution`) or close the tab.
- Free Colab tier is sufficient — this calls OpenAI's API, no GPU used.

## Troubleshooting

- **`NameError: name 'OPENAI_API_KEY' is not defined`** — you ran the chat cell without running the key cell first.
- **`AuthenticationError`** — bad API key, or key has no credit. Check <https://platform.openai.com/usage>.
- **Public URL never shows** — Colab sometimes blocks Gradio sharing on certain networks. Try `share=False` and use the in-notebook preview instead.
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
    out = Path(__file__).resolve().parent / "chatgpt_colab.ipynb"
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")