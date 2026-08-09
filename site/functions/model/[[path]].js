// Same-origin proxy to the pinned GitHub model release.
//
// GitHub release assets send no CORS headers, so the browser cannot fetch them
// cross-origin; this Pages Function pipes them through the site's own origin,
// forwarding Range requests so the chunked loader can resume. The asset list is
// a strict allowlist — this is a model mirror, not an open proxy.

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

export async function onRequestGet({ request, params }) {
  const asset = Array.isArray(params.path) ? params.path.join("/") : params.path;
  if (!ALLOWED.has(asset)) {
    return new Response("unknown model asset", { status: 404 });
  }
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
  headers.set("cache-control", "public, max-age=31536000, immutable");
  return new Response(response.body, { status: response.status, headers });
}
