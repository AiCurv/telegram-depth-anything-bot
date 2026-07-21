/**
 * api/webhook.js
 *
 * Vercel serverless endpoint that:
 *   1. Receives Telegram webhook payloads.
 *   2. Validates the sender against ALLOWED_TELEGRAM_USER_ID (single-user whitelist).
 *   3. Silently drops anyone else with HTTP 200 (so the bot cannot be probed by attackers).
 *   4. Buffers photos that share a `media_group_id` for 1500ms, then fires a
 *      SINGLE GitHub dispatch containing all file_ids — so an album of N photos
 *      triggers ONE Actions run, not N.
 *   5. Single photos (no media_group_id) are dispatched immediately.
 *   6. Replies instantly with HTTP 200 to avoid Telegram webhook timeouts.
 *
 * Command parsing:
 *   Flags can appear in message.text OR message.caption, in any order, with or
 *   without the leading slash. Examples that all work:
 *     `/hd /color`   `/fast /gray`   `hd color`   `/color`   `gray`
 *   Defaults when no flag is given:
 *     model = Depth-Anything-V2-Large-hf  (/hd)
 *     cmap  = gray                        (pure 0-255 grayscale)
 *
 * Album buffering caveat:
 *   Vercel serverless functions do NOT share in-memory state across invocations
 *   — each request can land on a different instance. The 1500ms in-memory
 *   debounce works because Telegram itself serializes album photos within
 *   <500ms in nearly all cases. If a request lands on a fresh instance mid-album,
 *   that batch is dispatched with whatever photos it received; the Python runner
 *   will still send them as a (smaller) album. The user sees a slightly split
 *   album rather than a failure. Acceptable degradation for a zero-infra setup.
 *
 * Security model:
 *   - All secrets live in Vercel env vars; NONE are ever baked into the repo.
 *   - Unauthorized users get an identical 200 OK — no information leakage.
 *   - The repository is public so we get unlimited GitHub Actions minutes, but
 *     the dispatch event requires GH_PAT_TOKEN so a stranger cannot fire it
 *     directly through GitHub's API.
 */

// Node 18+ ships a global `fetch` (and Vercel's Node 18+ runtime exposes it
// too), so we no longer need the node-fetch dependency. Keeping the import
// out of the way also makes local unit testing easier because we can stub
// globalThis.fetch directly.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MODEL_SMALL = 'depth-anything/Depth-Anything-V2-Small-hf';
const MODEL_LARGE = 'depth-anything/Depth-Anything-V2-Large-hf';

const TELEGRAM_API = `https://api.telegram.org/bot${process.env.TELEGRAM_BOT_TOKEN}`;

// How long to wait after the FIRST photo of a media group before dispatching
// the accumulated batch. Telegram sends album photos in quick succession
// (typically <500ms apart), but we leave a healthy margin so slow networks
// still get fully batched. 1500ms is short enough to feel responsive and long
// enough to catch a 10-photo album.
const MEDIA_GROUP_BUFFER_MS = 1500;

// Hard ceiling on the number of photos in a single dispatch. Telegram's
// sendMediaGroup API limits albums to 10 items; if a user somehow uploads
// more (Telegram shouldn't allow it, but defensive), we cap here.
const MAX_PHOTOS_PER_BATCH = 10;

// ---------------------------------------------------------------------------
// Per-instance album buffer
// ---------------------------------------------------------------------------
//
// Map<key, { chatId, messageId, photoIds: string[], captions: string[],
//            model, modelLabel, cmap, cmapLabel, timer: NodeJS.Timeout }>.
//
// `key` is the Telegram media_group_id. Single photos use a synthetic key
// prefixed with 'single:' so they go through the same dispatch path but with
// a different (immediate) timing rule.

const albumBuffer = new Map();

// ---------------------------------------------------------------------------
// Command parser
// ---------------------------------------------------------------------------

/**
 * Parse the user-supplied text (from message.text or message.caption) into a
 * { model, modelLabel, cmap, cmapLabel } tuple.
 *
 * Rules (matches `/hd` or `hd` case-insensitively, with or without the slash):
 *   - Model:   /hd  -> Large (default if neither /hd nor /fast)
 *              /fast-> Small
 *   - Color:   /color, /inferno, or 'color' -> inferno colormap
 *              /gray, /grayscale, or 'gray' -> grayscale (default)
 *
 * The slash is optional so users can type either `/hd color` or `hd color`.
 * Each token must be a whole word — we look-behind for start-of-string or
 * whitespace, and look-ahead for whitespace or end-of-string, so substrings
 * like 'gray' inside 'graymatter' or 'fast' inside 'fastly' do NOT match.
 */
function parseCommandPayload(rawText) {
  // Pad with whitespace so the lookbehind/lookahead regexes can match at both
  // ends of the string uniformly. Lowercase for case-insensitive matching.
  const text = ` ${(rawText || '').toLowerCase()} `;

  // ---- model ----
  const wantsFast = /\s\/?fast\s/.test(text);
  const wantsHd = /\s\/?hd\s/.test(text);
  // If both /hd and /fast are present, /hd wins (HD is the better default and
  // is also the documented default — explicit /hd is treated as an override).
  const model = wantsHd || !wantsFast ? MODEL_LARGE : MODEL_SMALL;
  const modelLabel = model === MODEL_LARGE ? 'HD' : 'fast';

  // ---- color ----
  const wantsColor = /\s\/?(color|inferno)\s/.test(text);
  const wantsGray = /\s\/?(gray|grayscale)\s/.test(text);
  // Truth table:
  //   wantsColor=F, wantsGray=F  -> default gray (neither flag, gray is default)
  //   wantsColor=F, wantsGray=T  -> gray
  //   wantsColor=T, wantsGray=F  -> inferno
  //   wantsColor=T, wantsGray=T  -> inferno (color wins as the deliberate override)
  const cmap = wantsColor ? 'inferno' : 'gray';
  const cmapLabel = cmap === 'inferno' ? 'inferno colormap' : 'grayscale';

  return { model, modelLabel, cmap, cmapLabel };
}

// ---------------------------------------------------------------------------
// Telegram + GitHub glue
// ---------------------------------------------------------------------------

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
// Album buffering
// ---------------------------------------------------------------------------

/**
 * Flush a buffered album: pull the accumulated photo_ids + first non-empty
 * caption out of the buffer, fire a single GitHub dispatch, notify the user,
 * and clean up the buffer entry.
 *
 * This is the only function that actually calls GitHub. It is called either:
 *   - immediately for single-photo messages, OR
 *   - after MEDIA_GROUP_BUFFER_MS for album messages.
 */
async function flushAlbum(key, isAlbum) {
  const entry = albumBuffer.get(key);
  if (!entry) return;
  albumBuffer.delete(key);

  const {
    chatId, messageId, photoIds, captions,
    model, modelLabel, cmap, cmapLabel,
  } = entry;

  // First non-empty caption wins. Telegram normally only puts the caption on
  // the first photo of an album, but defensively we scan all of them.
  const firstCaption = captions.find((c) => c && c.trim().length > 0) || '';

  const count = photoIds.length;
  const label = isAlbum ? `album (${count} photos)` : 'photo';

  console.log(`[flush] dispatching ${label}: model=${modelLabel} cmap=${cmapLabel} photos=${count}`);

  const dispatchPayload = {
    chat_id: chatId,
    message_id: messageId,
    photo_ids: photoIds,           // array of file_ids — single element for non-albums
    photo_count: count,
    is_album: isAlbum,
    model,
    model_label: modelLabel,
    cmap,
    cmap_label: cmapLabel,
    caption: firstCaption,
    sent_at: new Date().toISOString(),
  };

  try {
    await triggerGitHubAction(dispatchPayload);
  } catch (err) {
    console.error(`[flush] dispatch error for ${label}:`, err);
    await notifyTelegram(
      chatId,
      `⚠️ Could not start the depth job for your ${label}. The action dispatcher failed. Try again in a moment.`
    );
    return;
  }

  const eta = model === MODEL_LARGE ? '1–3 min (large model)' : '~30–60s (small model)';
  const perPhoto = model === MODEL_LARGE ? '~15s/photo' : '~2s/photo';
  const albumHint = isAlbum ? `\nBatch processing: ${perPhoto} per photo.` : '';
  await notifyTelegram(
    chatId,
    `Queued *${modelLabel}* depth run with *${cmapLabel}* output for your ${label}.\nETA: ${eta}${albumHint}\nI'll reply here with the depth map${isAlbum ? 's' : ''} when it's done.`
  );
}

/**
 * Add a photo to the album buffer and (re)schedule the flush.
 *
 * For album messages (media_group_id present), the first photo schedules a
 * flush after MEDIA_GROUP_BUFFER_MS. Subsequent photos in the same group
 * just append to the array — they do NOT reset the timer, because we want
 * the dispatch to fire on a fixed cadence regardless of how long the album
 * takes to arrive.
 *
 * For single-photo messages (no media_group_id), we use a synthetic key and
 * flush immediately (no timer).
 */
function bufferMediaGroup({
  mediaGroupId, chatId, messageId, fileId, caption,
  model, modelLabel, cmap, cmapLabel,
}) {
  const isAlbum = !!mediaGroupId;
  const key = isAlbum ? mediaGroupId : `single:${chatId}:${messageId}`;

  let entry = albumBuffer.get(key);
  if (!entry) {
    entry = {
      chatId,
      messageId,
      photoIds: [],
      captions: [],
      model,
      modelLabel,
      cmap,
      cmapLabel,
      timer: null,
    };
    albumBuffer.set(key, entry);
  }

  // Cap at MAX_PHOTOS_PER_BATCH; ignore overflow defensively. Telegram's UI
  // already limits albums to 10, but if a malformed update arrives with more,
  // we just drop the extras.
  if (entry.photoIds.length < MAX_PHOTOS_PER_BATCH) {
    entry.photoIds.push(fileId);
    entry.captions.push(caption || '');
  }

  // Schedule flush only if this is the first photo of the entry.
  if (!entry.timer) {
    if (isAlbum) {
      // Album: wait for siblings to accumulate.
      entry.timer = setTimeout(() => {
        // Don't await — fire-and-forget so the timer callback returns immediately.
        flushAlbum(key, true).catch((err) =>
          console.error(`[flush] unhandled error for album ${key}:`, err)
        );
      }, MEDIA_GROUP_BUFFER_MS);
    } else {
      // Single photo: dispatch immediately (next tick, so the HTTP response
      // goes out first). Using setImmediate keeps the latency floor at 0.
      entry.timer = 'immediate';
      setImmediate(() => {
        flushAlbum(key, false).catch((err) =>
          console.error(`[flush] unhandled error for single ${key}:`, err)
        );
      });
    }
  }

  return { buffered: true, isAlbum, key, count: entry.photoIds.length };
}

// ---------------------------------------------------------------------------
// Main handler
// ---------------------------------------------------------------------------

export default async function handler(req, res) {
  // Telegram only ever POSTs.
  if (req.method !== 'POST') {
    return res.status(200).end();
  }

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
  // Telegram delivers standalone text commands as `message.text`, and photo
  // uploads with a caption as `message.caption`. We normalize to one string so
  // the parser doesn't care which path the user took.
  const rawText = (message.text || message.caption || '').trim();
  const { model, modelLabel, cmap, cmapLabel } = parseCommandPayload(rawText);

  // ---- 3. Reject non-photo messages with a friendly hint -------------------
  const photo = Array.isArray(message.photo) && message.photo.length > 0
    ? message.photo[message.photo.length - 1]  // highest resolution
    : null;

  if (!photo) {
    if (rawText.startsWith('/start')) {
      await notifyTelegram(
        chatId,
        '*Depth bot reporting in.*\n\nSend me a photo (or an album!) and I will reply with monocular depth maps.\n\n*Flags* (combine freely in the caption):\n• `/hd` — large model (default)\n• `/fast` — small model\n• `/gray` — pure grayscale (default)\n• `/color` — inferno colormap\n\nExample: attach a photo, caption `/hd /color`. Or attach multiple photos as an album — they\'ll be processed together in a single run.'
      );
    } else if (rawText.startsWith('/help')) {
      await notifyTelegram(
        chatId,
        '*Usage:*\n1. Attach a photo (or an album of up to 10) to your message.\n2. Optionally caption it with flags: `/hd` `/fast` `/gray` `/color`.\n3. Wait ~1–3 minutes for the depth map.\n\n*Defaults:* HD model + grayscale output.\n\n*Albums:* all photos in a single album are processed in one GitHub Actions run, and you get a single reply album back.\n\nEverything runs on GitHub Actions runners; nothing is stored.'
      );
    } else {
      await notifyTelegram(
        chatId,
        'Send me a photo or album. Flags: `/hd` `/fast` `/gray` `/color`. Defaults to HD + grayscale.'
      );
    }
    return res.status(200).json({ ok: true, status: 'no_photo' });
  }

  // ---- 4. Buffer the photo (album-aware) -----------------------------------
  // If this is part of a media group, it will be queued with a 1500ms debounce
  // so all sibling photos land before we dispatch. Single photos dispatch
  // immediately on the next tick.
  const mediaGroupId = message.media_group_id || null;
  const bufferResult = bufferMediaGroup({
    mediaGroupId,
    chatId,
    messageId: message.message_id,
    fileId: photo.file_id,
    caption: rawText,  // use normalized text (already combines text + caption)
    model,
    modelLabel,
    cmap,
    cmapLabel,
  });

  // ---- 5. Acknowledge immediately -----------------------------------------
  // We return 200 BEFORE the dispatch fires — the buffer's setTimeout / setImmediate
  // ensures the dispatch happens after this response is sent. Telegram is happy
  // because it sees 200 within milliseconds, and the actual GitHub dispatch
  // happens in the background.
  return res.status(200).json({
    ok: true,
    buffered: true,
    is_album: bufferResult.isAlbum,
    photo_count_in_batch: bufferResult.count,
    model,
    cmap,
  });
}
