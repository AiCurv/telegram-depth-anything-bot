#!/usr/bin/env python3
"""
scripts/process_depth.py

Telegram <-> Depth Anything V2 bridge, with album (batch) support.

Pipeline:
    1. Parse --photo-ids (a JSON array of Telegram file_id strings, OR a single
       file_id string for backward compatibility).
    2. Download all source photos from Telegram via getFile + file download URL.
       Partial download failures are tolerated — we process whatever subset
       came through.
    3. Load the requested Hugging Face pipeline ONCE (small for /fast, large
       for /hd).
    4. Loop through downloaded photos, run monocular depth estimation per
       image, and render each to a depth PNG (grayscale or inferno based on
       --cmap).
    5. POST the results back to Telegram:
         - If multiple photos: use sendMediaGroup to deliver them as a single
           album reply. Telegram caps albums at 10 items.
         - If single photo: fall back to sendPhoto.
       In both cases we reply to the original message so the chat stays
       readable.
    6. Report final status (success/partial/failed) with per-photo stats.

This script is invoked by .github/workflows/depth_pipeline.yml. It does NOT
load any model on import — only when run as __main__ — so unit imports stay
cheap.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from typing import List, Tuple

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

# Telegram's sendMediaGroup endpoint hard-limits albums to 10 items.
# We also enforce it in the Vercel webhook, but defend here too.
MAX_MEDIA_GROUP_SIZE = 10


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


def download_photo_batch(photo_ids: List[str], workdir: str) -> List[Tuple[str, str]]:
    """Download all photos in the batch.

    Returns a list of (file_id, local_path) tuples for photos that downloaded
    successfully. Failed downloads are logged and skipped so the rest of the
    batch can still be processed.
    """
    results: List[Tuple[str, str]] = []
    for idx, file_id in enumerate(photo_ids, start=1):
        local_path = os.path.join(workdir, f"source_{idx:02d}.jpg")
        try:
            download_telegram_file(file_id, local_path)
            results.append((file_id, local_path))
        except Exception as exc:  # noqa: BLE001
            print(f"[download] failed for photo #{idx} (file_id={file_id}): {exc}",
                  file=sys.stderr)
    return results


def send_telegram_text(chat_id: str, text: str) -> None:
    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[telegram] sendMessage failed: {exc}", file=sys.stderr)


def send_telegram_photo(chat_id: str, photo_bytes: bytes, caption: str,
                        reply_to: str | None = None) -> None:
    """POST a single PNG photo to Telegram's sendPhoto endpoint.

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
        print(f"[telegram] sendPhoto with reply_to={reply_to} failed ({r.status_code}); retrying without reply")
        files = {"photo": ("depth.png", photo_bytes, "image/png")}
        data.pop("reply_to_message_id", None)
        r = requests.post(f"{TELEGRAM_API}/sendPhoto", files=files, data=data, timeout=60)

    if r.status_code != 200:
        send_telegram_text(chat_id, f"Depth finished but sendPhoto failed: {r.text[:300]}")
        raise RuntimeError(f"sendPhoto failed: {r.status_code} {r.text[:300]}")


def send_telegram_media_group(chat_id: str, photo_paths: List[str],
                              caption: str | None = None,
                              reply_to: str | None = None) -> None:
    """POST an album of PNG photos to Telegram's sendMediaGroup endpoint.

    Telegram limits albums to 10 items; if we have more (defensive), we send
    the first 10 and log a warning.

    The caption is attached to the FIRST photo only — that's how Telegram
    renders album captions natively.

    Same reply_to fallback as send_telegram_photo: if the reply_to message_id
    is rejected, retry without it.
    """
    if not photo_paths:
        raise RuntimeError("send_telegram_media_group called with empty photo_paths")

    if len(photo_paths) > MAX_MEDIA_GROUP_SIZE:
        print(f"[telegram] truncating album from {len(photo_paths)} to {MAX_MEDIA_GROUP_SIZE}")
        photo_paths = photo_paths[:MAX_MEDIA_GROUP_SIZE]

    files = []
    for idx, path in enumerate(photo_paths):
        # Telegram wants the multipart field name to be unique per photo AND
        # match the `attach://` reference in the JSON. We use photo_0, photo_1, ...
        files.append((f"photo_{idx}", (f"depth_{idx}.png", open(path, "rb"), "image/png")))

    # Build the media JSON array. The first photo gets the caption.
    media = []
    for idx in range(len(photo_paths)):
        item = {"type": "photo", "media": f"attach://photo_{idx}"}
        if idx == 0 and caption:
            item["caption"] = caption
            item["parse_mode"] = "Markdown"
        media.append(item)

    data = {
        "chat_id": chat_id,
        "media": json.dumps(media),
    }
    if reply_to:
        data["reply_to_message_id"] = reply_to

    r = requests.post(f"{TELEGRAM_API}/sendMediaGroup", files=files, data=data, timeout=120)

    # Close the file handles we opened.
    for _, (_, fh, _) in files:
        fh.close()

    if r.status_code != 200 and reply_to:
        print(f"[telegram] sendMediaGroup with reply_to={reply_to} failed ({r.status_code}); retrying without reply")
        # Reopen file handles for the retry.
        files = []
        for idx, path in enumerate(photo_paths):
            files.append((f"photo_{idx}", (f"depth_{idx}.png", open(path, "rb"), "image/png")))
        data.pop("reply_to_message_id", None)
        r = requests.post(f"{TELEGRAM_API}/sendMediaGroup", files=files, data=data, timeout=120)
        for _, (_, fh, _) in files:
            fh.close()

    if r.status_code != 200:
        send_telegram_text(chat_id, f"Depth finished but sendMediaGroup failed: {r.text[:300]}")
        raise RuntimeError(f"sendMediaGroup failed: {r.status_code} {r.text[:300]}")


# ---------------------------------------------------------------------------
# Depth inference (batch-aware)
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


def render_depth_map(depth_arr: np.ndarray, source: Image.Image, cmap: str,
                     out_path: str) -> Tuple[float, float]:
    """Render the depth array to a PNG file on disk.

    Two modes:
      - 'gray'    : min-max normalize the depth to 0-255, save as a single-
                    channel grayscale PNG (mode='L' in PIL). No colormap is
                    applied — the brightness directly encodes relative depth,
                    which is the most faithful 2D representation.
      - 'inferno' : apply matplotlib's inferno colormap, save as RGB PNG.
                    Better for human visual scanning of depth gradients.

    Both modes resize the output to <= MAX_OUTPUT_SIDE px on the longest side
    using LANCZOS resampling, preserving aspect ratio.

    Returns (min_depth, max_depth) for stats reporting.
    """
    target_w, target_h = _resize_to_cap(source)

    # Resize the depth array to match the target output dimensions. We use
    # PIL's LANCZOS for a clean, non-jagged downscale.
    depth_pil_full = Image.fromarray(depth_arr, mode="F")
    depth_pil_resized = depth_pil_full.resize((target_w, target_h), Image.LANCZOS)
    depth_resized = np.array(depth_pil_resized)

    dmin = float(depth_resized.min())
    dmax = float(depth_resized.max())

    if cmap == "gray":
        # Min-max normalize to 0-255 uint8.
        rng = dmax - dmin
        if rng < 1e-6:
            normalized = np.zeros_like(depth_resized, dtype=np.uint8)
        else:
            normalized = ((depth_resized - dmin) / rng * 255.0)
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        gray_img = Image.fromarray(normalized, mode="L")
        gray_img.save(out_path, format="PNG", optimize=True)
    else:
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

        # pad_inches=0 + constrained_layout keeps the canvas exactly the image.
        fig.savefig(out_path, format="png", dpi=100, pad_inches=0)
        plt.close(fig)

    return dmin, dmax


def run_batch_inference(image_paths: List[str], model_name: str, cmap: str,
                        workdir: str) -> List[Tuple[str, float, float]]:
    """Load the model once and run inference across all images in the batch.

    Returns a list of (output_png_path, min_depth, max_depth) tuples for each
    successfully processed image. Failed images are logged and skipped.
    """
    print(f"[infer] loading model: {model_name}  (cmap={cmap})  batch_size={len(image_paths)}")
    t0 = time.time()

    # CPU-only runner; GitHub ubuntu-latest has no GPU.
    pipe = pipeline(
        task="depth-estimation",
        model=model_name,
        device="cpu",
    )
    print(f"[infer] model loaded in {time.time() - t0:.1f}s")

    results: List[Tuple[str, float, float]] = []
    for idx, image_path in enumerate(image_paths, start=1):
        try:
            image = Image.open(image_path).convert("RGB")
            print(f"[infer] photo #{idx}/{len(image_paths)}: {image.size[0]}x{image.size[1]}")

            t1 = time.time()
            result = pipe(image)
            print(f"[infer]   inference took {time.time() - t1:.1f}s")

            depth_pil = result["depth"]
            depth_arr = np.array(depth_pil).astype(np.float32)

            out_path = os.path.join(workdir, f"depth_{idx:02d}.png")
            dmin, dmax = render_depth_map(depth_arr, image, cmap, out_path)
            results.append((out_path, dmin, dmax))
        except Exception as exc:  # noqa: BLE001
            print(f"[infer] FAILED on photo #{idx} ({image_path}): {exc}", file=sys.stderr)
            traceback.print_exc()

    return results


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
    p = argparse.ArgumentParser(description="Run Depth Anything V2 on Telegram photos (batch-aware).")
    p.add_argument("--chat-id", required=True)
    p.add_argument(
        "--photo-ids",
        required=True,
        help='JSON array of Telegram file_ids, e.g. \'["id1","id2"]\'. '
             'For backward compat, a bare string is treated as a single-element array.',
    )
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


def parse_photo_ids(raw: str) -> List[str]:
    """Parse the --photo-ids argument into a list of file_id strings.

    Accepts:
      - JSON array:        '["id1","id2"]'
      - Bare string:       'id1'  (treated as single-element list)
      - Comma-separated:   'id1,id2'  (defensive fallback)

    Empty strings are filtered out.
    """
    raw = (raw or "").strip()
    if not raw:
        return []

    # Try JSON first (the canonical format from the workflow).
    if raw.startswith("["):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return [str(x) for x in arr if x]
        except json.JSONDecodeError:
            pass

    # Bare string fallback.
    if "," in raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return [raw]


def main() -> int:
    args = parse_args()

    photo_ids = parse_photo_ids(args.photo_ids)
    if not photo_ids:
        send_telegram_text(args.chat_id, "❌ No photo_ids provided to the depth runner.")
        return 1

    workdir = "/tmp/depth_work"
    os.makedirs(workdir, exist_ok=True)

    model_label = MODEL_LABELS.get(args.model, args.model)
    cmap_label = CMAP_LABELS.get(args.cmap, args.cmap)
    is_album = len(photo_ids) > 1

    try:
        # 1. Acknowledge we're working on it.
        batch_label = f"album of {len(photo_ids)} photos" if is_album else "photo"
        send_telegram_text(
            args.chat_id,
            f"📥 Got it. Running *{model_label}* depth estimation with *{cmap_label}* output "
            f"on your {batch_label} now…"
        )

        # 2. Pull all source photos (partial failures tolerated).
        downloaded = download_photo_batch(photo_ids, workdir)
        if not downloaded:
            raise RuntimeError("All photo downloads failed")
        print(f"[main] downloaded {len(downloaded)}/{len(photo_ids)} photos")

        # 3. Run batch inference (model loaded ONCE).
        image_paths = [p for (_, p) in downloaded]
        results = run_batch_inference(image_paths, args.model, args.cmap, workdir)
        if not results:
            raise RuntimeError("All inference attempts failed")

        output_paths = [r[0] for r in results]
        depth_ranges = [(r[1], r[2]) for r in results]

        # 4. Send results back to Telegram.
        if is_album and len(output_paths) > 1:
            # Album reply via sendMediaGroup.
            caption = (
                f"✅ *Depth maps ready.*\n"
                f"Model: `{args.model}` ({model_label})\n"
                f"Render: `{args.cmap}` ({cmap_label})\n"
                f"Photos: {len(output_paths)}/{len(photo_ids)} processed"
            )
            send_telegram_media_group(args.chat_id, output_paths, caption=caption, reply_to=args.reply_to)
        else:
            # Single photo reply via sendPhoto.
            dmin, dmax = depth_ranges[0]
            # Read the source size from the downloaded image for stats.
            src_img = Image.open(image_paths[0])
            caption = (
                f"✅ *Depth map ready.*\n"
                f"Model: `{args.model}` ({model_label})\n"
                f"Render: `{args.cmap}` ({cmap_label})\n"
                f"Source: {src_img.size[0]}×{src_img.size[1]} px\n"
                f"Depth range: `{dmin:.1f}`–`{dmax:.1f}` (relative)"
            )
            with open(output_paths[0], "rb") as fh:
                send_telegram_photo(args.chat_id, fh.read(), caption, reply_to=args.reply_to)

        print(f"[main] done. {len(output_paths)}/{len(photo_ids)} photos processed.")
        return 0

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        send_telegram_text(
            args.chat_id,
            f"❌ Depth run failed: `{str(exc)[:200]}`\nTry `/fast` instead of `/hd`, "
            f"or send fewer photos.",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
