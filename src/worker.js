/**
 * src/worker.js
 *
 * Cloudflare Worker that:
 *   1. Receives Telegram webhook payloads.
 *   2. Validates the sender against ALLOWED_TELEGRAM_USER_ID (single-user whitelist).
 *   3. Silently drops anyone else with HTTP 200 (so the bot cannot be probed).
 *   4. Buffers photos that share a `media_group_id` into Cloudflare KV with a
 *      1500ms debounce window, then fires a SINGLE GitHub dispatch containing
 *      all file_ids — so an album of N photos triggers ONE Actions run, not N.
 *   5. Single photos (no media_group_id) are dispatched immediately.
 *   6. Replies instantly with HTTP 200 to avoid Telegram webhook timeouts;
 *      the actual GitHub dispatch happens in `ctx.waitUntil` after the
 *      debounce window.
 *
 * Why KV instead of in-memory?
 *   Vercel serverless functions (the previous architecture) do not share
 *   in-memory state across invocations. Album photos that landed on different
 *   Vercel instances were dispatched independently, so the user got N separate
 *   replies instead of one album reply. Cloudflare KV provides durable,
 *   cross-instance state — every Worker invocation, regardless of which colo
 *   it lands on, can read and append to the same album buffer.
 *
 * Why per-photo keys instead of read-modify-write on a single key?
 *   KV's free tier has a soft limit of 1 write per second per key, and KV is
 *   eventually consistent. A read-modify-write loop on a single album key
 *   would race when multiple album photos arrive within ~100ms of each other
 *   (the typical Telegram album cadence). Per-photo keys eliminate the
 *   read-modify-write race: each photo writes to its OWN key, and the flush
 *   path uses `list({ prefix })` to gather all siblings atomically.
 *
 * Last-photo-flushes pattern:
 *   Each album POST schedules `ctx.waitUntil(flush at t+1500ms)`. The flush
 *   reads all photos for this album and checks: "has the album been quiet for
 *   ≥1500ms?" (i.e. is the newest photo's timestamp older than 1500ms?). If
 *   yes, dispatch and delete keys. If no, no-op — a later sibling's flush
 *   will handle it. The LAST photo's flush is the one that actually dispatches.
 *
 *   Edge case: if the last photo's Worker invocation dies before its flush
 *   fires, no dispatch happens. Orphaned KV keys auto-expire after KV_TTL_S
 *   (5 minutes). The user will notice no reply and resend — acceptable for a
 *   single-user bot. To make this bulletproof, use Durable Objects (requires
 *   Workers Paid plan, $5/month).
 *
 * Security model:
 *   - All secrets live in Cloudflare Worker secrets; NONE are in the repo.
 *   - Unauthorized users get an identical 200 OK — no information leakage.
 *   - The repository is public so we get unlimited GitHub Actions minutes, but
 *     the dispatch event requires GH_PAT_TOKEN so a stranger cannot fire it
 *     directly through GitHub's API.
 *
 * Command parsing:
 *   Flags can appear in message.text OR message.caption, in any order, with or
 *   without the leading slash. Examples that all work:
 *     `/hd /color`   `/fast /gray`   `hd color`   `/color`   `gray`
 *   Defaults when no flag is given:
 *     model = Depth-Anything-V2-Large-hf  (/hd)
 *     cmap  = gray                        (pure 0-255 grayscale)
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MODEL_SMALL = 'depth-anything/Depth-Anything-V2-Small-hf';
const MODEL_LARGE = 'depth-anything/Depth-Anything-V2-Large-hf';

// How long to wait after the LATEST photo of a media group before dispatching
// the accumulated batch. Telegram sends album photos in quick succession
// (typically <500ms apart), but we leave a healthy margin so slow networks
// still get fully batched. 2500ms is short enough to feel responsive and long
// enough to catch a 10-photo album.
//
// Note: this is the INITIAL debounce. The flush function also does a 3-try
// re-check loop with 500ms pauses, because Cloudflare KV is eventually
// consistent across colos — a sibling write may not be visible to a flush
// running in a different colo for up to ~1-2 seconds.
const MEDIA_GROUP_BUFFER_MS = 2500;

// How many times the flush should re-check KV for additional photos before
// giving up and dispatching whatever it has. Each retry is 500ms apart.
// Total worst-case wait: MEDIA_GROUP_BUFFER_MS + FLUSH_RETRIES * 500ms.
const FLUSH_RETRIES = 4;
const FLUSH_RETRY_DELAY_MS = 500;

// Hard ceiling on the number of photos in a single dispatch. Telegram's
// sendMediaGroup API limits albums to 10 items; if a user somehow uploads
// more (Telegram shouldn't allow it, but defensive), we cap here.
const MAX_PHOTOS_PER_BATCH = 10;

// KV key TTL — orphaned album buffers auto-expire after this many seconds.
// 5 minutes is long enough that even a slow album delivery completes well
// within the window, and short enough that orphans don't accumulate.
const KV_TTL_S = 300;

// Sentinel key TTL — once an album has been dispatched, we set a short-lived
// sentinel to suppress duplicate dispatches from sibling flush timers that
// fire slightly later. 60s is enough for all sibling timers to have fired.
const SENTINEL_TTL_S = 60;

// ---------------------------------------------------------------------------
// Command parser (identical to the previous Vercel webhook — same behavior,
// same test suite passes)
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
  // Default to gray. If both /gray and /color are present, color wins as the
  // deliberate override.
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
async function notifyTelegram(botToken, chatId, text) {
  try {
    await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
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
async function triggerGitHubAction(env, payload) {
  const owner = env.GITHUB_REPO_OWNER;
  const repo = env.GITHUB_REPO_NAME;
  const token = env.GH_PAT_TOKEN;

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
// KV album buffer
// ---------------------------------------------------------------------------
//
// Storage layout:
//   album:{media_group_id}:{message_id}  -> value="", metadata={photo data}
//   album:{media_group_id}:dispatched    -> value="1", short TTL
//
// Each photo writes its OWN key, so there are no read-modify-write races and
// no per-key write-rate-limit issues. The flush path uses `list({ prefix })`
// which returns metadata inline — no per-key `get()` calls needed.
//
// KV metadata is capped at 1024 bytes per entry. Our photo metadata is
// typically ~500 bytes (file_id ~80 chars, plus a few small fields and a
// truncated caption), well within the limit.

/**
 * Build the KV key for a single photo in an album.
 */
function photoKey(mediaGroupId, messageId) {
  return `album:${mediaGroupId}:${messageId}`;
}

/**
 * Build the KV sentinel key that marks an album as already-dispatched.
 */
function dispatchedSentinelKey(mediaGroupId) {
  return `album:${mediaGroupId}:dispatched`;
}

/**
 * Write a single photo's metadata to KV under its own key.
 *
 * The photo data goes into KV `metadata` (not the value) so that `list()`
 * returns it inline — the flush path doesn't need to call `get()` per photo,
 * saving N KV reads per album.
 *
 * KV metadata is capped at 1024 bytes. We truncate the caption defensively.
 */
async function writePhotoEntry(env, mediaGroupId, photo) {
  // Truncate caption to keep metadata under the 1024-byte KV limit.
  // file_id is typically ~80 chars; we leave ~600 chars for caption.
  const safeCaption = (photo.caption || '').slice(0, 600);

  const metadata = {
    chatId: photo.chatId,
    messageId: photo.messageId,
    fileId: photo.fileId,
    caption: safeCaption,
    model: photo.model,
    modelLabel: photo.modelLabel,
    cmap: photo.cmap,
    cmapLabel: photo.cmapLabel,
    receivedAt: Date.now(),
  };

  const key = photoKey(mediaGroupId, photo.messageId);
  await env.ALBUM_BUFFER.put(key, '', {
    expirationTtl: KV_TTL_S,
    metadata,
  });
}

/**
 * Gather all photos for an album by listing KV keys with the album prefix.
 *
 * Returns an array of { name, metadata } entries, sorted by messageId for
 * stable ordering (Telegram delivers album photos in message_id order, but
 * KV list() does not guarantee any particular order).
 *
 * If the dispatched sentinel exists, callers should treat the album as
 * already-handled (checked separately via isAlreadyDispatched).
 */
async function gatherAlbum(env, mediaGroupId) {
  const prefix = `album:${mediaGroupId}:`;
  const sentinel = dispatchedSentinelKey(mediaGroupId);
  const entries = [];
  let cursor;

  do {
    const result = await env.ALBUM_BUFFER.list({ prefix, cursor, limit: 100 });
    for (const k of result.keys) {
      // Skip the dispatched sentinel — it's not a photo.
      if (k.name === sentinel) continue;
      if (k.metadata) {
        entries.push({ name: k.name, metadata: k.metadata });
      }
    }
    cursor = result.list_complete ? null : result.cursor;
  } while (cursor);

  // Sort by messageId for stable, predictable ordering.
  entries.sort((a, b) => a.metadata.messageId - b.metadata.messageId);
  return entries;
}

/**
 * Delete all photo keys for an album (called after a successful dispatch).
 *
 * Best-effort: if a delete fails, the key will auto-expire after KV_TTL_S.
 */
async function deleteAlbumKeys(env, mediaGroupId) {
  const prefix = `album:${mediaGroupId}:`;
  const sentinel = dispatchedSentinelKey(mediaGroupId);
  let cursor;

  do {
    const result = await env.ALBUM_BUFFER.list({ prefix, cursor, limit: 100 });
    for (const k of result.keys) {
      // Don't delete the sentinel here — it has its own short TTL.
      if (k.name === sentinel) continue;
      await env.ALBUM_BUFFER.delete(k.name);
    }
    cursor = result.list_complete ? null : result.cursor;
  } while (cursor);
}

/**
 * Set the dispatched sentinel so sibling flush timers know to no-op.
 */
async function markDispatched(env, mediaGroupId) {
  await env.ALBUM_BUFFER.put(dispatchedSentinelKey(mediaGroupId), '1', {
    expirationTtl: SENTINEL_TTL_S,
  });
}

/**
 * Check if the dispatched sentinel exists.
 */
async function isAlreadyDispatched(env, mediaGroupId) {
  const val = await env.ALBUM_BUFFER.get(dispatchedSentinelKey(mediaGroupId));
  return val !== null;
}

/**
 * Flush a buffered album: gather all photo entries from KV, fire a single
 * GitHub dispatch, notify the user, and clean up the KV keys.
 *
 * Called either:
 *   - immediately (in `ctx.waitUntil`) for single-photo messages, OR
 *   - after MEDIA_GROUP_BUFFER_MS (in `ctx.waitUntil`) for album messages.
 *
 * Leader election by smallest message_id:
 *   Each album POST schedules its own flush. To avoid N sibling flushes each
 *   dispatching independently, only the photo with the SMALLEST message_id
 *   in the album dispatches. All other siblings no-op.
 *
 *   Telegram delivers album photos in message_id order, so the smallest
 *   message_id is the FIRST photo of the album. Its flush is the "leader".
 *
 * KV eventual consistency handling:
 *   Cloudflare KV is eventually consistent across colos. When 4 album photos
 *   arrive within ~100ms, they may land on different colos, and each colo's
 *   KV read may not see writes from other colos for up to ~1-2 seconds.
 *
 *   The leader's flush does a retry loop: gather, wait, gather again. If the
 *   count is still growing, keep retrying (up to FLUSH_RETRIES). When the
 *   count is stable, dispatch whatever we have.
 *
 *   The sentinel key prevents double-dispatch if the leader crashes mid-flush
 *   and a sibling's retry loop later sees a stable count.
 *
 * `myMessageId` is the message_id of the photo that triggered THIS flush.
 * For single photos, it's the only photo's message_id (always the leader).
 * For albums, the leader check is: am I the smallest message_id in the album?
 */
async function flushAlbum(env, mediaGroupId, isAlbum, myMessageId) {
  // Check sentinel first — a sibling flush may have already dispatched.
  if (await isAlreadyDispatched(env, mediaGroupId)) {
    console.log(`[flush] ${mediaGroupId} already dispatched, skipping`);
    return;
  }

  let entries = await gatherAlbum(env, mediaGroupId);
  if (entries.length === 0) {
    console.log(`[flush] ${mediaGroupId} has no photos (already cleaned up)`);
    return;
  }

  // Leader election: only the photo with the smallest message_id dispatches.
  // This guarantees exactly ONE dispatch per album, regardless of how many
  // sibling flushes fire or which colo they run on.
  if (isAlbum) {
    const minMessageId = Math.min(...entries.map((e) => e.metadata.messageId));
    if (myMessageId !== minMessageId) {
      console.log(
        `[flush] ${mediaGroupId} I am msg ${myMessageId}, leader is msg ${minMessageId}, deferring`
      );
      return;
    }
    console.log(`[flush] ${mediaGroupId} I am the leader (msg ${myMessageId})`);
  }

  // Retry loop: re-gather a few times to let KV propagate sibling writes.
  // This is the critical fix for KV eventual consistency — without it, the
  // leader might dispatch with only 1-2 photos even though 4 were sent,
  // because the sibling writes haven't propagated to the leader's colo yet.
  let lastCount = entries.length;
  for (let attempt = 1; attempt <= FLUSH_RETRIES; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, FLUSH_RETRY_DELAY_MS));

    // Re-check sentinel — a sibling may have dispatched while we waited.
    if (await isAlreadyDispatched(env, mediaGroupId)) {
      console.log(`[flush] ${mediaGroupId} dispatched by sibling during retry ${attempt}`);
      return;
    }

    const reEntries = await gatherAlbum(env, mediaGroupId);
    if (reEntries.length === 0) {
      console.log(`[flush] ${mediaGroupId} cleaned up during retry ${attempt}`);
      return;
    }

    if (reEntries.length > lastCount) {
      console.log(
        `[flush] ${mediaGroupId} retry ${attempt}: ` +
        `count grew ${lastCount} -> ${reEntries.length}, continuing`
      );
      entries = reEntries;
      lastCount = reEntries.length;
      continue;
    }

    // Count is stable — KV has propagated. Use these entries.
    if (reEntries.length !== entries.length) {
      entries = reEntries;
    }
    console.log(
      `[flush] ${mediaGroupId} retry ${attempt}: count stable at ${reEntries.length}, dispatching`
    );
    break;
  }

  // Cap at MAX_PHOTOS_PER_BATCH defensively. Telegram's UI already limits
  // albums to 10, but if a malformed update arrives with more, we drop extras.
  const capped = entries.slice(0, MAX_PHOTOS_PER_BATCH);
  const photoIds = capped.map((e) => e.metadata.fileId);

  // First non-empty caption wins. Telegram normally only puts the caption on
  // the first photo of an album, but defensively we scan all of them.
  const firstCaption =
    capped.map((e) => e.metadata.caption).find((c) => c && c.trim().length > 0) || '';

  // Use the FIRST photo's chat/message/model/cmap — all photos in an album
  // share the same chat, and the first photo's caption determines the flags.
  const first = capped[0].metadata;
  const chatId = first.chatId;
  const messageId = first.messageId;
  const model = first.model;
  const modelLabel = first.modelLabel;
  const cmap = first.cmap;
  const cmapLabel = first.cmapLabel;

  const count = photoIds.length;
  const label = isAlbum ? `album (${count} photos)` : 'photo';

  console.log(
    `[flush] dispatching ${label}: model=${modelLabel} cmap=${cmapLabel} ` +
    `photos=${count} media_group_id=${mediaGroupId}`
  );

  // Set the dispatched sentinel BEFORE firing the GitHub dispatch. This
  // minimizes the race window where a sibling flush could also fire a dispatch.
  await markDispatched(env, mediaGroupId);

  const dispatchPayload = {
    chat_id: chatId,
    message_id: messageId,
    photo_ids: photoIds,
    model,
    cmap,
    is_album: isAlbum,
    photo_count: count,
    caption: firstCaption,
  };

  try {
    await triggerGitHubAction(env, dispatchPayload);
  } catch (err) {
    console.error(`[flush] dispatch error for ${label}:`, err);
    await notifyTelegram(
      env.TELEGRAM_BOT_TOKEN,
      chatId,
      `⚠️ Could not start the depth job for your ${label}. The action dispatcher failed. Try again in a moment.`
    );
    // Clean up the buffer so a retry starts fresh.
    await deleteAlbumKeys(env, mediaGroupId);
    return;
  }

  const eta = model === MODEL_LARGE ? '1–3 min (large model)' : '~30–60s (small model)';
  const perPhoto = model === MODEL_LARGE ? '~15s/photo' : '~2s/photo';
  const albumHint = isAlbum ? `\nBatch processing: ${perPhoto} per photo.` : '';
  await notifyTelegram(
    env.TELEGRAM_BOT_TOKEN,
    chatId,
    `Queued *${modelLabel}* depth run with *${cmapLabel}* output for your ${label}.\n` +
    `ETA: ${eta}${albumHint}\n` +
    `I'll reply here with the depth map${isAlbum ? 's' : ''} when it's done.`
  );

  // Clean up the photo keys. Sentinel auto-expires.
  await deleteAlbumKeys(env, mediaGroupId);
}

// ---------------------------------------------------------------------------
// Main handler
// ---------------------------------------------------------------------------

export default {
  async fetch(request, env, ctx) {
    // Telegram only ever POSTs.
    if (request.method !== 'POST') {
      return new Response(null, { status: 200 });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return new Response(null, { status: 200 });
    }

    const message = body.message || body.edited_message;
    if (!message) {
      return new Response(null, { status: 200 });
    }

    const fromId = message.from?.id;
    const chatId = message.chat?.id;
    const allowedUserId = parseInt(env.ALLOWED_TELEGRAM_USER_ID, 10);

    // ---- 1. Auth firewall ------------------------------------------------
    // Unknown users are silently dropped. We return 200 (not 401/403) so the
    // bot looks indistinguishable from a dead endpoint to a scanner.
    if (!fromId || fromId !== allowedUserId) {
      return new Response(null, { status: 200 });
    }

    // ---- 2. Parse the message --------------------------------------------
    // Telegram delivers standalone text commands as `message.text`, and photo
    // uploads with a caption as `message.caption`. We normalize to one string
    // so the parser doesn't care which path the user took.
    const rawText = (message.text || message.caption || '').trim();
    const { model, modelLabel, cmap, cmapLabel } = parseCommandPayload(rawText);

    // ---- 3. Reject non-photo messages with a friendly hint ---------------
    const photo = Array.isArray(message.photo) && message.photo.length > 0
      ? message.photo[message.photo.length - 1]  // highest resolution
      : null;

    if (!photo) {
      const botToken = env.TELEGRAM_BOT_TOKEN;
      if (rawText.startsWith('/start')) {
        await notifyTelegram(
          botToken,
          chatId,
          '*Depth bot reporting in.*\n\n' +
          'Send me a photo (or an album!) and I will reply with monocular depth maps.\n\n' +
          '*Flags* (combine freely in the caption):\n' +
          '• `/hd` — large model (default)\n' +
          '• `/fast` — small model\n' +
          '• `/gray` — pure grayscale (default)\n' +
          '• `/color` — inferno colormap\n\n' +
          'Example: attach a photo, caption `/hd /color`. ' +
          'Or attach multiple photos as an album — they\'ll be processed together in a single run.'
        );
      } else if (rawText.startsWith('/help')) {
        await notifyTelegram(
          botToken,
          chatId,
          '*Usage:*\n' +
          '1. Attach a photo (or an album of up to 10) to your message.\n' +
          '2. Optionally caption it with flags: `/hd` `/fast` `/gray` `/color`.\n' +
          '3. Wait ~1–3 minutes for the depth map.\n\n' +
          '*Defaults:* HD model + grayscale output.\n\n' +
          '*Albums:* all photos in a single album are buffered for 1.5s and ' +
          'processed in one GitHub Actions run, and you get a single reply album back.\n\n' +
          'Everything runs on GitHub Actions runners; nothing is stored.'
        );
      } else {
        await notifyTelegram(
          botToken,
          chatId,
          'Send me a photo or album. Flags: `/hd` `/fast` `/gray` `/color`. Defaults to HD + grayscale.'
        );
      }
      return new Response(JSON.stringify({ ok: true, status: 'no_photo' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // ---- 4. Buffer the photo (album-aware) -------------------------------
    const mediaGroupId = message.media_group_id || null;
    const isAlbum = !!mediaGroupId;

    if (isAlbum) {
      // Album path: write the photo to KV under its own key, then schedule
      // a debounced flush. The LAST photo's flush (the one that fires after
      // the album has been quiet for MEDIA_GROUP_BUFFER_MS) does the dispatch.
      await writePhotoEntry(env, mediaGroupId, {
        chatId,
        messageId: message.message_id,
        fileId: photo.file_id,
        caption: rawText,
        model,
        modelLabel,
        cmap,
        cmapLabel,
      });

      ctx.waitUntil(
        (async () => {
          // Wait MEDIA_GROUP_BUFFER_MS so siblings can land in KV. The flush
          // function will use leader election (smallest message_id) to ensure
          // exactly ONE dispatch per album, regardless of how many sibling
          // flushes fire or which colo they run on.
          await new Promise((resolve) => setTimeout(resolve, MEDIA_GROUP_BUFFER_MS));
          try {
            await flushAlbum(env, mediaGroupId, true, message.message_id);
          } catch (err) {
            console.error('[waitUntil] album flush threw:', err);
          }
        })()
      );

      return new Response(
        JSON.stringify({
          ok: true,
          buffered: true,
          is_album: true,
          photo_count_in_batch_so_far: 1, // sibling photos may append more
          model,
          cmap,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Single-photo path: no buffer, no debounce. Dispatch immediately.
    // We still use ctx.waitUntil so we can respond 200 to Telegram instantly
    // and let the GitHub dispatch + Telegram notify happen in the background.
    const singleKey = `single:${chatId}:${message.message_id}`;

    // Write a one-photo "album" entry so we can reuse the flush path.
    await writePhotoEntry(env, singleKey, {
      chatId,
      messageId: message.message_id,
      fileId: photo.file_id,
      caption: rawText,
      model,
      modelLabel,
      cmap,
      cmapLabel,
    });

    ctx.waitUntil(
      (async () => {
        try {
          await flushAlbum(env, singleKey, false, message.message_id);
        } catch (err) {
          console.error('[waitUntil] single flush threw:', err);
        }
      })()
    );

    return new Response(
      JSON.stringify({
        ok: true,
        buffered: true,
        is_album: false,
        model,
        cmap,
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );
  },
};
