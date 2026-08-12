// Same-origin model mirror: R2 first, the pinned GitHub release as fallback.
//
// GitHub release assets send no CORS headers, so the browser cannot fetch them
// cross-origin; this Pages Function serves them from the site's own origin. The
// primary source is the MODEL_BUCKET R2 binding (same bytes, digest-verified at
// upload): reads through a binding cost no egress, involve no second HTTP hop,
// and are not subject to GitHub's throttling — which surfaced as intermittent
// 503s on the 32 MiB chunked download the day the playground got real traffic.
// GitHub remains the fallback so a missing/misnamed bucket object degrades to
// the old behavior instead of a dead download. The asset list is a strict
// allowlist — this is a model mirror, not an open proxy.
//
// Every byte served here is still endpoint- and digest-verified by the loader
// against the pinned manifest, so a wrong or stale mirror object is rejected
// client-side rather than trusted.

const RELEASE_BASE =
  "https://github.com/Dicklesworthstone/franken_tts/releases/download/model-qwen3-tts-v1/";

const ALLOWED = new Set([
  "qwen3-tts-12hz-0.6b-base.fttsq",
  "speech_tokenizer_model.safetensors",
  "vocab.json",
  "merges.txt",
  "tokenizer_config.json",
  "config.json",
  "generation_config.json",
]);

const IMMUTABLE = "public, max-age=31536000, immutable";

/// Parses a single-range `Range` header into R2's range option, or null for
/// whole-file / unsupported forms. The loader only sends `bytes=start-end`,
/// but suffix (`bytes=-n`) and open (`bytes=start-`) forms are handled so a
/// hand-typed curl behaves.
function parseRange(header) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header ?? "");
  if (!match || (match[1] === "" && match[2] === "")) return null;
  if (match[1] === "") return { suffix: Number(match[2]) };
  const offset = Number(match[1]);
  if (match[2] === "") return { offset };
  return { offset, length: Number(match[2]) - offset + 1 };
}

function fromR2(object, rangeRequested, totalBytes) {
  const headers = new Headers();
  headers.set("content-type", "application/octet-stream");
  headers.set("accept-ranges", "bytes");
  headers.set("etag", object.httpEtag);
  headers.set("cache-control", IMMUTABLE);
  if (rangeRequested && object.range) {
    const start = object.range.offset ?? 0;
    const length = object.range.length ?? totalBytes - start;
    headers.set("content-length", String(length));
    headers.set(
      "content-range",
      `bytes ${start}-${start + length - 1}/${totalBytes}`,
    );
    return new Response(object.body, { status: 206, headers });
  }
  headers.set("content-length", String(object.size));
  return new Response(object.body, { status: 200, headers });
}

async function fromGitHub(request, asset) {
  const upstream = new Request(RELEASE_BASE + asset, {
    headers: request.headers.has("range")
      ? { range: request.headers.get("range") }
      : {},
    redirect: "follow",
  });
  const response = await fetch(upstream);
  const headers = new Headers();
  for (const name of ["content-type", "content-length", "content-range", "accept-ranges", "etag"]) {
    const value = response.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  // Model files are immutable at this tag; let the CDN and the browser keep them.
  headers.set("cache-control", IMMUTABLE);
  return new Response(response.body, { status: response.status, headers });
}

export async function onRequestGet({ request, params, env }) {
  const asset = Array.isArray(params.path) ? params.path.join("/") : params.path;
  if (!ALLOWED.has(asset)) {
    return new Response("unknown model asset", { status: 404 });
  }

  if (env.MODEL_BUCKET) {
    try {
      const range = parseRange(request.headers.get("range"));
      // One read per request: even a ranged `get` reports the FULL object size
      // on `object.size`, which is all content-range needs — no `head` call.
      const object = range
        ? await env.MODEL_BUCKET.get(asset, { range })
        : await env.MODEL_BUCKET.get(asset);
      if (object) return fromR2(object, range !== null, object.size);
      // Object missing from the bucket: fall through to GitHub.
    } catch {
      // R2 hiccup: fall through to GitHub rather than 5xx the download.
    }
  }

  return fromGitHub(request, asset);
}
