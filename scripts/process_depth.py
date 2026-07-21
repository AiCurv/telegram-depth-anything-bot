#!/usr/bin/env python3
"""
scripts/process_depth.py

Telegram <-> Depth Anything V2 bridge.

Pipeline:
    1. Download the original photo from Telegram via getFile + file download URL.
    2. Load the requested Hugging Face pipeline (small for /fast, large for /hd).
    3. Run monocular depth estimation.
    4. Render the depth map. Two modes:
         - 'gray'    : normalize depth tensor to 0-255 uint8, save as a pure
                       single-channel grayscale PNG. No colormap applied. This
                       is the cleanest representation of relative depth.
         - 'inferno' : apply matplotlib's 'inferno' colormap, save as RGB PNG.
    5. POST the result back to Telegram via sendPhoto, replying to the original
       message so the chat stays readable.

This script is invoked by .github/workflows/depth_pipeline.yml. It does NOT
load any model on import — only when run as __main__ — so unit imports stay
cheap.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import traceback
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from transformers import pipeline

# ---------------------------------------------------------------------------
# Telegram API glue
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    print("FATAL: TELEGRAM_BOT_TOKEN is not set.", file=sys.stderr)
    sys.exit(2)

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"


def download_telegram_file(file_id: str, dest_path: str) -> None:
    """Fetch the binary content of a Telegram photo by its file_id.

    Telegram's `getFile` endpoint returns a `file_path` that we can then GET
    from https://api.telegram.org/file/bot<token>/<file_path>. Large files
    (>20MB) require the bot to be hosted on a server with the proper file
    bandwidth — Depth Anything V2 handles images well below this limit.
    """
    r = requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(f"getFile failed: {payload}")

    file_path = payload["result"]["file_path"]
    download_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"

    resp = requests.get(download_url, timeout=60)
    resp.raise_for_status()

    with open(dest_path, "wb") as fh:
        fh.write(resp.content)

    print(f"[telegram] downloaded {len(resp.content)} bytes -> {dest_path}")


def send_telegram_photo(chat_id: str, photo_bytes: bytes, caption: str,
                        reply_to: str | None = None) -> None:
    """POST a PNG photo to Telegram's sendPhoto endpoint.

    If `reply_to` is set but Telegram rejects it (e.g. message_id was from a
    forwarded/synthetic payload and doesn't exist in this chat), we retry
    without the reply link rather than failing the whole run.
    """
    files = {"photo": ("depth.png", photo_bytes, "image/png")}
    data = {
        "chat_id": chat_id,
        "caption": caption,
        "parse_mode": "Markdown",
    }
    if reply_to:
        data["reply_to_message_id"] = reply_to

    r = requests.post(f"{TELEGRAM_API}/sendPhoto", files=files, data=data, timeout=60)
    if r.status_code != 200 and reply_to:
        # Retry without reply_to — common case: reply_to message_id is stale or
        # was never a real Telegram message_id (synthetic test, forwarded update, etc.)
        print(f"[telegram] sendPhoto with reply_to={reply_to} failed ({r.status_code}); retrying without reply")
        files = {"photo": ("depth.png", photo_bytes, "image/png")}
        data.pop("reply_to_message_id", None)
        r = requests.post(f"{TELEGRAM_API}/sendPhoto", files=files, data=data, timeout=60)

    if r.status_code != 200:
        # Fall back to a text error so the user is never left hanging.
        send_telegram_text(chat_id, f"Depth finished but sendPhoto failed: {r.text[:300]}")
        raise RuntimeError(f"sendPhoto failed: {r.status_code} {r.text[:300]}")


def send_telegram_text(chat_id: str, text: str) -> None:
    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[telegram] sendMessage failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Depth inference
# ---------------------------------------------------------------------------

def run_depth_inference(image_path: str, model_name: str, cmap: str) -> Tuple[np.ndarray, Image.Image]:
    """Run Depth Anything V2 on the image, returning (depth_array, source_pil).

    The `cmap` argument is passed through purely so we can log it next to the
    model name; rendering happens in `render_depth_map`. Returning the raw
    float32 depth array gives the renderer maximum precision to work with
    before normalization.

    The HF pipeline returns a dict with 'predicted_depth' (a tensor) and
    'depth' (a PIL image, post-processed and interpolated back to source
    resolution). We use 'depth' as the source of truth for the array so the
    output dimensions always match the input photo.
    """
    print(f"[infer] loading model: {model_name}  (cmap={cmap})")
    t0 = time.time()

    # CPU-only runner; GitHub ubuntu-latest has no GPU.
    pipe = pipeline(
        task="depth-estimation",
        model=model_name,
        device="cpu",
    )
    print(f"[infer] model loaded in {time.time() - t0:.1f}s")

    image = Image.open(image_path).convert("RGB")
    print(f"[infer] source image: {image.size[0]}x{image.size[1]}")

    t1 = time.time()
    result = pipe(image)
    print(f"[infer] inference took {time.time() - t1:.1f}s")

    depth_pil = result["depth"]
    # Use the post-processed 'depth' PIL image — its dimensions already match
    # the source image, so the renderer doesn't need to resize anything.
    depth_arr = np.array(depth_pil).astype(np.float32)
    return depth_arr, image


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# Cap the longest side of the output PNG at this many pixels. Telegram
# downscales anything bigger anyway, and keeping the cap means a single
# 4K source photo doesn't produce a 4K PNG that takes forever to upload.
MAX_OUTPUT_SIDE = 1280


def _resize_to_cap(source: Image.Image) -> tuple[int, int]:
    """Return (target_w, target_h) preserving aspect ratio, longest side <= cap."""
    w, h = source.size
    scale = MAX_OUTPUT_SIDE / max(w, h) if max(w, h) > MAX_OUTPUT_SIDE else 1.0
    return int(round(w * scale)), int(round(h * scale))


def render_depth_map(depth_arr: np.ndarray, source: Image.Image, cmap: str) -> bytes:
    """Render the depth array as a PNG, returning the raw bytes.

    Two modes:
      - 'gray'    : min-max normalize the depth to 0-255, save as a single-
                    channel grayscale PNG (mode='L' in PIL). No colormap is
                    applied — the brightness directly encodes relative depth,
                    which is the most faithful 2D representation.
      - 'inferno' : apply matplotlib's inferno colormap, save as RGB PNG.
                    Better for human visual scanning of depth gradients.

    Both modes resize the output to <= MAX_OUTPUT_SIDE px on the longest side
    using LANCZOS resampling, preserving aspect ratio.
    """
    target_w, target_h = _resize_to_cap(source)

    # Resize the depth array to match the target output dimensions. We use
    # PIL's LANCZOS for a clean, non-jagged downscale.
    depth_pil_full = Image.fromarray(depth_arr, mode="F")
    depth_pil_resized = depth_pil_full.resize((target_w, target_h), Image.LANCZOS)
    depth_resized = np.array(depth_pil_resized)

    if cmap == "gray":
        # Min-max normalize to 0-255 uint8.
        dmin = float(depth_resized.min())
        dmax = float(depth_resized.max())
        # Guard against divide-by-zero on a degenerate (flat) depth map.
        rng = dmax - dmin
        if rng < 1e-6:
            normalized = np.zeros_like(depth_resized, dtype=np.uint8)
        else:
            normalized = ((depth_resized - dmin) / rng * 255.0)
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        gray_img = Image.fromarray(normalized, mode="L")
        buf = io.BytesIO()
        gray_img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return buf.read()

    # cmap == 'inferno'
    # Size the matplotlib figure so its pixel output matches (target_w, target_h)
    # exactly. dpi=100 -> figsize = (pixels / 100).
    fig_w = target_w / 100.0
    fig_h = target_h / 100.0

    fig, ax = plt.subplots(
        figsize=(fig_w, fig_h),
        constrained_layout=True,
    )
    ax.imshow(depth_resized, cmap="inferno", aspect="auto")
    ax.axis("off")

    buf = io.BytesIO()
    # pad_inches=0 + constrained_layout keeps the canvas exactly the image.
    fig.savefig(buf, format="png", dpi=100, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

# Friendly display names for the final caption.
MODEL_LABELS = {
    "depth-anything/Depth-Anything-V2-Small-hf": "Small (fast)",
    "depth-anything/Depth-Anything-V2-Large-hf": "Large (HD)",
}
CMAP_LABELS = {
    "gray": "grayscale",
    "inferno": "inferno colormap",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Depth Anything V2 on a Telegram photo.")
    p.add_argument("--chat-id", required=True)
    p.add_argument("--file-id", required=True)
    p.add_argument(
        "--model",
        default="depth-anything/Depth-Anything-V2-Large-hf",
        help="Hugging Face model id. Defaults to the Large (HD) model.",
    )
    p.add_argument(
        "--cmap",
        default="gray",
        choices=["gray", "inferno"],
        help="Render mode: 'gray' for pure 0-255 grayscale (default), 'inferno' for colormap.",
    )
    p.add_argument("--reply-to", default=None, help="Telegram message_id to reply to")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    workdir = "/tmp/depth_work"
    os.makedirs(workdir, exist_ok=True)
    src_path = os.path.join(workdir, "source.jpg")

    model_label = MODEL_LABELS.get(args.model, args.model)
    cmap_label = CMAP_LABELS.get(args.cmap, args.cmap)

    try:
        # 1. Acknowledge we're working on it.
        send_telegram_text(
            args.chat_id,
            f"📥 Got it. Running *{model_label}* depth estimation with *{cmap_label}* output now…",
        )

        # 2. Pull the original photo.
        download_telegram_file(args.file_id, src_path)

        # 3. Run inference.
        depth_arr, source = run_depth_inference(src_path, args.model, args.cmap)

        # 4. Render + reply.
        png_bytes = render_depth_map(depth_arr, source, args.cmap)

        stats = (
            f"✅ *Depth map ready.*\n"
            f"Model: `{args.model}` ({model_label})\n"
            f"Render: `{args.cmap}` ({cmap_label})\n"
            f"Source: {source.size[0]}×{source.size[1]} px\n"
            f"Depth range: `{float(depth_arr.min()):.1f}`–`{float(depth_arr.max()):.1f}` (relative)"
        )
        send_telegram_photo(args.chat_id, png_bytes, stats, reply_to=args.reply_to)
        print("[main] done.")
        return 0

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        send_telegram_text(
            args.chat_id,
            f"❌ Depth run failed: `{str(exc)[:200]}`\nTry `/fast` instead of `/hd`, "
            f"or send a smaller image.",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
