// Tombstone. This route briefly hosted a token-guarded multipart uploader used
// once (2026-08-12) to seed the model mirror bucket past the Cloudflare REST
// layer's 300 MiB single-put cap. The bucket is seeded, the UPLOAD_TOKEN secret
// is deleted, and this handler refuses everything; the file awaits the owner's
// deletion (workspace rule: agents do not delete files).

export function onRequest() {
  return new Response("gone", { status: 410 });
}
