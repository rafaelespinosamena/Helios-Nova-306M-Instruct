#!/usr/bin/env python3
"""
Helios Nova 306M-Instruct  ·  Interactive Chat
================================================
Downloads the instruct-tuned model from HuggingFace Hub and provides an
interactive chat experience with the same template used during SFT.

Chat template (no special tokens added — uses existing vocabulary)
------------------------------------------------------------------
    ### System:
    You are a helpful assistant.
    ### User:
    What is the capital of France?
    ### Assistant:
    The capital of France is Paris.</s>

Usage
-----
    python instruct_chat.py
    python instruct_chat.py --temperature 0.7 --top-k 40
    python instruct_chat.py --system "You are a pirate. Answer in pirate speak."
    python instruct_chat.py --repo path/to/local/checkpoint

Controls
--------
    Type any message and press Enter to chat.
    Type "!temp 0.5"     to change temperature on the fly.
    Type "!topk 30"      to change top-k on the fly.
    Type "!max 512"      to change max generation length.
    Type "!rep 1.1"      to change repetition penalty.
    Type "!stream"       to toggle streaming output.
    Type "!system ..."   to change the system prompt.
    Type "!reset"        to clear conversation history.
    Type "!single"       to toggle single-turn mode (no history).
    Type "quit" or "exit" or Ctrl+C to leave.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from HeliosNova import HeliosNova, HeliosNovaConfig

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_REPO    = "respinosamena/Helios-Nova-306M-Instruct"
DEFAULT_SYSTEM  = "You are a helpful assistant."

# Chat template markers
SYS_MARKER  = "### System:\n"
USER_MARKER = "### User:\n"
ASST_MARKER = "### Assistant:\n"


def load_model(repo_id: str, device: torch.device) -> tuple[HeliosNova, AutoTokenizer]:
    """Download (or use cached) instruct model + tokenizer."""
    print(f"Loading tokenizer from {repo_id} …")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    print(f"Loading model from {repo_id} …")
    model = HeliosNova.from_pretrained(repo_id, device=str(device))
    model = model.to(device).eval()

    n = model.param_count()
    cfg = model.cfg
    print(f"  Model:    {n:,} params ({n / 1e6:.1f}M)")
    print(f"  Layers:   {cfg.n_layers}")
    print(f"  Context:  {cfg.max_seq_len} tokens")
    print(f"  GQA:      {cfg.n_heads}q / {cfg.n_kv_heads}kv")

    # Load training metadata if available
    try:
        from huggingface_hub import hf_hub_download
        import json
        meta_path = hf_hub_download(repo_id, "training_state.json")
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  SFT step: {meta.get('step', '?')}")
        print(f"  Val loss: {meta.get('val_loss', '?')}")
    except Exception:
        pass

    return model, tokenizer


def build_prompt(
    system: str,
    history: list[dict],
    user_msg: str,
) -> str:
    """
    Build the full prompt string from system message, conversation history,
    and the new user message.  Ends with the assistant marker so the model
    knows it should generate a response.
    """
    parts = [f"{SYS_MARKER}{system}\n"]

    for turn in history:
        if turn["role"] == "user":
            parts.append(f"{USER_MARKER}{turn['content']}\n")
        elif turn["role"] == "assistant":
            parts.append(f"{ASST_MARKER}{turn['content']}\n")

    # Current user turn + assistant prompt
    parts.append(f"{USER_MARKER}{user_msg}\n")
    parts.append(ASST_MARKER)

    return "".join(parts)


@torch.no_grad()
def generate(
    model: HeliosNova,
    input_ids: torch.Tensor,
    tokenizer,
    max_new: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float = 1.2,
    stream: bool = True,
) -> str:
    """Generate tokens with streaming, stopping at EOS or turn markers."""
    generated: list[str] = []
    buffer = ""

    # Tokens that signal we should stop (the model is trying to start a new turn)
    stop_strings = ["### User:", "### System:", "### Assistant:"]

    for _ in range(max_new):
        ctx = input_ids[:, -model.cfg.max_seq_len:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(input_ids[0].tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

        token_id = next_id.item()

        # Stop on EOS
        if token_id == tokenizer.eos_token_id:
            break

        # Skip BOS / PAD
        if token_id in (tokenizer.bos_token_id, tokenizer.pad_token_id):
            continue

        tok = tokenizer.decode([token_id], skip_special_tokens=False)
        generated.append(tok)
        buffer += tok

        # Check if the model is generating a new turn marker → stop
        should_stop = False
        for stop in stop_strings:
            if stop in buffer:
                # Remove the partial marker from output
                idx = buffer.find(stop)
                # Trim generated to remove the marker
                clean = buffer[:idx].rstrip()
                if stream:
                    # We need to rewrite — clear and reprint
                    print(f"\r\033[K", end="")
                generated = [clean]
                should_stop = True
                break

        if should_stop:
            break

        if stream:
            print(tok, end="", flush=True)

    if stream:
        print()

    result = "".join(generated).strip()

    # Final cleanup: remove any trailing turn markers
    for stop in stop_strings:
        if stop in result:
            result = result[:result.find(stop)].strip()

    return result


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Helios Nova Instruct Chat")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO,
                        help="HuggingFace repo or local path (default: %(default)s)")
    parser.add_argument("--max-tokens", "-m", type=int, default=1024,
                        help="Max tokens to generate per response")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-k", "-k", type=int, default=40,
                        help="Top-k sampling")
    parser.add_argument("--repetition-penalty", "-r", type=float, default=1.2,
                        help="Repetition penalty (>1.0 discourages repeats)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Print all at once instead of streaming")
    parser.add_argument("--system", "-s", type=str, default=DEFAULT_SYSTEM,
                        help="System prompt")
    parser.add_argument("--single-turn", action="store_true",
                        help="No conversation history (each prompt is independent)")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}\n")

    model, tokenizer = load_model(args.repo, device)

    # Mutable settings
    temperature = args.temperature
    top_k = args.top_k
    max_tokens = args.max_tokens
    rep_penalty = args.repetition_penalty
    stream = not args.no_stream
    system = args.system
    single_turn = args.single_turn
    history: list[dict] = []

    print(f"\n{'─' * 60}")
    print(f"  Helios Nova 306M-Instruct")
    print(f"  temp={temperature}  top_k={top_k}  max={max_tokens}  rep={rep_penalty}")
    print(f"  stream={stream}  single_turn={single_turn}")
    print(f"  System: {system[:60]}{'…' if len(system) > 60 else ''}")
    print(f"  Commands: !temp !topk !max !rep !stream !system !reset !single")
    print(f"{'─' * 60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        # ── Commands ─────────────────────────────────────────────────
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        if user_input.startswith("!temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"  → temperature={temperature}")
            except (IndexError, ValueError):
                print("  Usage: !temp 0.5")
            continue

        if user_input.startswith("!topk "):
            try:
                top_k = int(user_input.split()[1])
                print(f"  → top_k={top_k}")
            except (IndexError, ValueError):
                print("  Usage: !topk 30")
            continue

        if user_input.startswith("!max "):
            try:
                max_tokens = int(user_input.split()[1])
                print(f"  → max_tokens={max_tokens}")
            except (IndexError, ValueError):
                print("  Usage: !max 512")
            continue

        if user_input.startswith("!rep "):
            try:
                rep_penalty = float(user_input.split()[1])
                print(f"  → repetition_penalty={rep_penalty}")
            except (IndexError, ValueError):
                print("  Usage: !rep 1.2")
            continue

        if user_input == "!stream":
            stream = not stream
            print(f"  → stream={stream}")
            continue

        if user_input.startswith("!system "):
            system = user_input[8:].strip()
            history.clear()
            print(f"  → system prompt updated, history cleared")
            continue

        if user_input == "!reset":
            history.clear()
            print(f"  → conversation history cleared")
            continue

        if user_input == "!single":
            single_turn = not single_turn
            if single_turn:
                history.clear()
            print(f"  → single_turn={single_turn}")
            continue

        # ── Build prompt ─────────────────────────────────────────────
        current_history = [] if single_turn else history
        prompt_text = build_prompt(system, current_history, user_input)

        # Tokenise
        ids = [tokenizer.bos_token_id] + tokenizer.encode(
            prompt_text, add_special_tokens=False
        )

        # Check context length
        remaining = model.cfg.max_seq_len - len(ids)
        gen_len = min(max_tokens, remaining)

        if gen_len <= 0:
            print("  (context window full — use !reset to clear history)")
            # Trim history
            if history:
                history = history[-4:]  # keep last 2 turns
                print("  (auto-trimmed history to last 2 turns, try again)")
            continue

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        # ── Generate ─────────────────────────────────────────────────
        print(f"\nHelios Nova: ", end="", flush=True)

        if not stream:
            response = generate(
                model, input_ids, tokenizer, gen_len,
                temperature, top_k, rep_penalty, stream=False,
            )
            print(response)
        else:
            response = generate(
                model, input_ids, tokenizer, gen_len,
                temperature, top_k, rep_penalty, stream=True,
            )

        # Save to history
        if not single_turn and response:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

        print()


if __name__ == "__main__":
    main()
