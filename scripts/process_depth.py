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
    3. Load the Depth Anything V2 Large Hugging Face pipeline ONCE.
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
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

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
# Model (hardcoded)
# ---------------------------------------------------------------------------
#
# The Small model (Depth-Anything-V2-Small-hf) has been retired. We always
# run Depth-Anything-V2-Large-hf for maximum quality. The Cloudflare Worker
# no longer parses /fast or /small flags — model selection is gone from the
# entire pipeline.

MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"

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

    Implements strict independent per-image float32 min-max normalization,
    matching the official Hugging Face Depth Anything V2 implementation:

        d_min = depth.min()
        d_max = depth.max()
        normalized = (depth - d_min) / (d_max - d_min)

    d_min / d_max are computed fresh for THIS image only. No tensor bounds are
    cached or leaked across loop iterations, so batch albums and repeated
    images each get their own full dynamic range stretched to [0, 1].

    Two render modes (both saved as 3-channel RGB PNGs):
      - 'gray'    : normalized * 255 -> uint8, broadcast to RGB (R=G=B).
                    This is the most faithful 2D representation — brightness
                    directly encodes relative depth, no colormap applied.
      - 'inferno' : matplotlib.colormaps['inferno'](normalized) maps every
                    float pixel directly to an RGB triple. Bypasses
                    plt.imshow / savefig entirely, so there is no figure
                    padding, no DPI scaling, no axis spine, no aspect-ratio
                    stretching — pixel-exact output matching the HF Space.

    Both modes resize the depth array to <= MAX_OUTPUT_SIDE px on the longest
    side using LANCZOS resampling BEFORE normalization, preserving aspect
    ratio. Resize happens on the float32 tensor so no quantization is
    introduced before the colormap is applied.

    Returns (min_depth, max_depth) for stats reporting.
    """
    target_w, target_h = _resize_to_cap(source)

    # Resize the float32 depth tensor to the target output dimensions.
    # PIL mode="F" keeps full float32 precision through LANCZOS resampling.
    depth_pil_full = Image.fromarray(depth_arr, mode="F")
    depth_pil_resized = depth_pil_full.resize((target_w, target_h), Image.LANCZOS)
    depth_resized = np.asarray(depth_pil_resized, dtype=np.float32)

    # STRICT per-image min-max normalization. No clipping, no shared bounds,
    # no cached min/max from previous loop iterations.
    dmin = float(depth_resized.min())
    dmax = float(depth_resized.max())
    rng = dmax - dmin
    if rng < 1e-6:
        # Degenerate flat depth — render as all-black rather than dividing by ~0.
        normalized = np.zeros_like(depth_resized, dtype=np.float32)
    else:
        normalized = (depth_resized - dmin) / rng  # float32 in [0, 1]

    if cmap == "gray":
        # Scale normalized floats to uint8 [0, 255] and broadcast to 3-channel
        # RGB so the output PNG has consistent shape with the inferno path.
        depth_uint8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        rgb = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
        Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG", optimize=True)
    else:
        # cmap == 'inferno' — apply the colormap directly to the normalized
        # float tensor. matplotlib.colormaps['inferno'] returns an RGBA array
        # of shape (H, W, 4) with values in [0, 1]. We drop the alpha channel
        # and quantize to uint8. This is the exact pipeline the HF Space uses,
        # bypassing pyplot's figure/axes machinery entirely.
        rgba = matplotlib.colormaps["inferno"](normalized)
        rgb = rgba[..., :3]  # drop alpha
        rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(rgb_uint8, mode="RGB").save(out_path, format="PNG", optimize=True)

    return dmin, dmax


def run_batch_inference(image_paths: List[str], model_name: str, cmap: str,
                        workdir: str) -> List[Tuple[str, float, float]]:
    """Load the model once and run inference across all images in the batch.

    Bypasses `transformers.pipeline("depth-estimation")` entirely. The pipeline
    post-processes raw depth with `(depth * 255 / np.max(depth)).astype("uint8")`,
    which (a) divides by `max` instead of `(max - min)` so negative depth values
    get clipped to zero, and (b) quantizes to uint8 BEFORE downstream code can
    do per-image min-max normalization. The result is salt-and-pepper noise in
    dark regions and crushed mid-tones — exactly the artifacts we were seeing
    vs. the Hugging Face Space.

    Native path (matches the official HF Space for Depth Anything V2):
      1. processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
         model     = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
      2. inputs = processor(images=image, return_tensors="pt")
      3. outputs = model(**inputs)  # no_grad
      4. predicted_depth = outputs.predicted_depth           # raw float32 tensor
      5. resize to (H, W) of the ORIGINAL input image via
         torch.nn.functional.interpolate(..., mode="bicubic", align_corners=False)
      6. depth_arr = prediction.squeeze().cpu().numpy()      # float32, unquantized

    render_depth_map then applies strict per-image min-max normalization on
    the unquantized float32 tensor — exactly what the HF Space does.

    Returns a list of (output_png_path, min_depth, max_depth) tuples for each
    successfully processed image. Failed images are logged and skipped.
    """
    print(f"[infer] loading model (native): {model_name}  (cmap={cmap})  batch_size={len(image_paths)}")
    t0 = time.time()

    # CPU-only runner; GitHub ubuntu-latest has no GPU.
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.to("cpu").eval()
    print(f"[infer] model loaded in {time.time() - t0:.1f}s")

    results: List[Tuple[str, float, float]] = []
    for idx, image_path in enumerate(image_paths, start=1):
        try:
            image = Image.open(image_path).convert("RGB")
            src_w, src_h = image.size  # PIL: (width, height)
            print(f"[infer] photo #{idx}/{len(image_paths)}: {src_w}x{src_h}")

            t1 = time.time()

            # --- Native inference, bypassing the pipeline post-processing ---
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Raw predicted_depth tensor. Shape is model-dependent; Depth
            # Anything V2 Large emits (B, 1, H/14, W/14). Normalize to exactly
            # 4D (B, C, H, W) before F.interpolate, which requires 4D input.
            predicted_depth = outputs.predicted_depth
            if predicted_depth.dim() == 3:
                # (B, H, W) -> (B, 1, H, W)
                predicted_depth = predicted_depth.unsqueeze(1)
            elif predicted_depth.dim() == 4 and predicted_depth.shape[1] != 1:
                # Multi-channel output — average across channels to a single
                # depth channel. Defensive; shouldn't trigger for DA-V2.
                predicted_depth = predicted_depth.mean(dim=1, keepdim=True)

            # Resize the low-res model output back to the ORIGINAL image
            # dimensions using bicubic interpolation. This matches the HF Space
            # pipeline exactly. align_corners=False is the standard choice for
            # continuous-valued tensors like depth maps.
            prediction = F.interpolate(
                predicted_depth,
                size=(src_h, src_w),  # (H, W) ordering
                mode="bicubic",
                align_corners=False,
            )

            # Squeeze batch (B=1) and channel (C=1) dims -> (H, W) float32
            # tensor on CPU as a numpy array. NO clipping, NO quantization.
            depth_arr = (
                prediction.squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            print(f"[infer]   inference took {time.time() - t1:.1f}s  "
                  f"raw depth: min={float(depth_arr.min()):.4f}  "
                  f"max={float(depth_arr.max()):.4f}  shape={depth_arr.shape}")

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
# Only the Large model remains — Small has been purged from the pipeline.
MODEL_LABELS = {
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
        default=MODEL_NAME,
        help=argparse.SUPPRESS,  # Hardcoded to Large; flag kept only for backward compat.
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

    # Force the Large model regardless of what --model was passed. The flag is
    # kept only so older workflow_dispatch payloads that still include `model`
    # don't crash argparse; the value is overwritten here.
    model_name = MODEL_NAME
    model_label = MODEL_LABELS[model_name]
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
        results = run_batch_inference(image_paths, model_name, args.cmap, workdir)
        if not results:
            raise RuntimeError("All inference attempts failed")

        output_paths = [r[0] for r in results]
        depth_ranges = [(r[1], r[2]) for r in results]

        # 4. Send results back to Telegram.
        if is_album and len(output_paths) > 1:
            # Album reply via sendMediaGroup.
            caption = (
                f"✅ *Depth maps ready.*\n"
                f"Model: `{model_name}` ({model_label})\n"
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
                f"Model: `{model_name}` ({model_label})\n"
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
            f"❌ Depth run failed: `{str(exc)[:200]}`\nTry sending fewer photos, "
            f"or retry in a moment.",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
