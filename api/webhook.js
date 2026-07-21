/**
 * api/webhook.js
 *
 * Vercel serverless endpoint that:
 *   1. Receives Telegram webhook payloads.
 *   2. Validates the sender against ALLOWED_TELEGRAM_USER_ID (single-user whitelist).
 *   3. Silently drops anyone else with HTTP 200 (so the bot cannot be probed by attackers).
 *   4. Detects photo messages + optional /fast or /hd flags.
 *   5. Fires a `repository_dispatch` event at GitHub Actions to start heavy compute.
 *   6. Replies instantly with HTTP 200 to avoid Telegram webhook timeouts (the 30s cliff).
 *
 * Security model:
 *   - All secrets live in Vercel env vars; NONE are ever baked into the repo.
 *   - Unauthorized users get an identical 200 OK — no information leakage.
 *   - The repository is public so we get unlimited GitHub Actions minutes, but
 *     the dispatch event requires GH_PAT_TOKEN so a stranger cannot fire it
 *     directly through GitHub's API.
 */

import fetch from 'node-fetch';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TELEGRAM_API = `https://api.telegram.org/bot${process.env.TELEGRAM_BOT_TOKEN}`;

/**
 * Send a short text message to a Telegram chat. Fire-and-forget, errors are
 * swallowed so we never block the 200 OK response to Telegram.
 */
async function notifyTelegram(chatId, text) {
  try {
    await fetch(`${TELEGRAM_API}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        text,
        parse_mode: 'Markdown',
        disable_web_page_preview: true,
      }),
    });
  } catch (e) {
    console.error('notifyTelegram failed:', e.message);
  }
}

/**
 * Trigger the GitHub Actions workflow via the `repository_dispatch` event.
 * The workflow in `.github/workflows/depth_pipeline.yml` listens for the
 * event type `depth_request`.
 */
async function triggerGitHubAction(payload) {
  const owner = process.env.GITHUB_REPO_OWNER;
  const repo = process.env.GITHUB_REPO_NAME;
  const token = process.env.GH_PAT_TOKEN;

  const url = `https://api.github.com/repos/${owner}/${repo}/dispatches`;

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'telegram-depth-anything-bot',
    },
    body: JSON.stringify({
      event_type: 'depth_request',
      client_payload: payload,
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`GitHub dispatch failed (${res.status}): ${body}`);
  }
}

// ---------------------------------------------------------------------------
// Main handler
// ---------------------------------------------------------------------------

export default async function handler(req, res) {
  // Telegram only ever POSTs.
  if (req.method !== 'POST') {
    return res.status(200).end();
  }

  // Always respond 200 ASAP unless we explicitly decide to delay. The bot's
  // user-visible latency is dominated by GitHub Actions (~1-3 min), so an
  // extra 500ms here to await the dispatch call is acceptable and lets us
  // surface dispatch failures to the user immediately rather than silently.
  const body = req.body || {};

  const message = body.message || body.edited_message;
  if (!message) {
    return res.status(200).end();
  }

  const fromId = message.from?.id;
  const chatId = message.chat?.id;
  const allowedUserId = parseInt(process.env.ALLOWED_TELEGRAM_USER_ID, 10);

  // ---- 1. Auth firewall ----------------------------------------------------
  // Unknown users are silently dropped. We return 200 (not 401/403) so the
  // bot looks indistinguishable from a dead endpoint to a scanner.
  if (!fromId || fromId !== allowedUserId) {
    return res.status(200).end();
  }

  // ---- 2. Parse the message ------------------------------------------------
  const text = (message.text || message.caption || '').trim();
  const wantsHd = /\b\/hd\b/i.test(text);
  const wantsFast = /\b\/fast\b/i.test(text);
  const model = wantsHd
    ? 'depth-anything/Depth-Anything-V2-Large-hf'
    : 'depth-anything/Depth-Anything-V2-Small-hf';

  // ---- 3. Reject non-photo messages with a friendly hint -------------------
  const photo = Array.isArray(message.photo) && message.photo.length > 0
    ? message.photo[message.photo.length - 1]  // highest resolution
    : null;

  if (!photo) {
    if (text.startsWith('/start')) {
      await notifyTelegram(
        chatId,
        '*Depthbrobot reporting in.*\n\nSend me a photo. I will reply with a monocular depth map.\n\nFlags:\n• `/fast` — small model (~30s)\n• `/hd` — large model (~90s)\n\nExample: attach a photo, caption `/hd`.'
      );
    } else if (text.startsWith('/help')) {
      await notifyTelegram(
        chatId,
        '*Usage:*\n1. Attach a photo to your message.\n2. Optionally caption it with `/fast` or `/hd`.\n3. Wait ~1–3 minutes for the depth map.\n\nEverything runs on GitHub Actions runners; nothing is stored.'
      );
    } else {
      await notifyTelegram(
        chatId,
        'Send me a photo. Add `/hd` for the large model or `/fast` for the small one.'
      );
    }
    return res.status(200).json({ ok: true, status: 'no_photo' });
  }

  // ---- 4. Fire the GitHub Action ------------------------------------------
  const dispatchPayload = {
    chat_id: chatId,
    message_id: message.message_id,
    file_id: photo.file_id,
    file_unique_id: photo.file_unique_id,
    file_size: photo.file_size || 0,
    model,
    mode: wantsHd ? 'hd' : 'fast',
    sent_at: new Date().toISOString(),
  };

  try {
    await triggerGitHubAction(dispatchPayload);
  } catch (err) {
    console.error('dispatch error:', err);
    await notifyTelegram(
      chatId,
      '⚠️ Could not start the depth job. The action dispatcher failed. Try again in a moment.'
    );
    return res.status(200).json({ ok: false, error: 'dispatch_failed' });
  }

  // ---- 5. Acknowledge to the user -----------------------------------------
  const eta = wantsHd ? '1–3 min (large model)' : '~30–60s (small model)';
  await notifyTelegram(
    chatId,
    `Queued *${wantsHd ? 'HD' : 'fast'}* depth run.\nETA: ${eta}\nI'll reply here with the depth map when it's done.`
  );

  return res.status(200).json({ ok: true, dispatched: true, model });
}
