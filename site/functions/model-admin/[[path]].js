// TEMPORARY multipart uploader for the model mirror bucket. Exists because the
// Cloudflare REST layer caps single-object puts at 300 MiB, and the two big
// model files can only enter R2 through a binding's multipart API. Guarded by
// the UPLOAD_TOKEN project secret; allowlisted keys only. DELETE THIS FILE and
// redeploy the moment the bucket holds both objects — it has no other purpose.

const ALLOWED = new Set([
  "qwen3-tts-12hz-0.6b-base.fttsq",
  "speech_tokenizer_model.safetensors",
]);

function denied(env, request) {
  const token = request.headers.get("authorization") ?? "";
  return !env.UPLOAD_TOKEN || token !== `Bearer ${env.UPLOAD_TOKEN}`;
}

export async function onRequest({ request, params, env }) {
  if (denied(env, request)) return new Response("no", { status: 403 });
  if (!env.MODEL_BUCKET) return new Response("no bucket binding", { status: 500 });
  const key = Array.isArray(params.path) ? params.path.join("/") : params.path;
  if (!ALLOWED.has(key)) return new Response("unknown key", { status: 404 });
  const url = new URL(request.url);
  const action = url.searchParams.get("action");

  try {
    if (request.method === "POST" && action === "create") {
      const upload = await env.MODEL_BUCKET.createMultipartUpload(key);
      return Response.json({ uploadId: upload.uploadId });
    }
    if (request.method === "PUT" && action === "part") {
      const uploadId = url.searchParams.get("uploadId");
      const partNumber = Number(url.searchParams.get("partNumber"));
      const upload = env.MODEL_BUCKET.resumeMultipartUpload(key, uploadId);
      const part = await upload.uploadPart(partNumber, request.body);
      return Response.json({ partNumber: part.partNumber, etag: part.etag });
    }
    if (request.method === "POST" && action === "complete") {
      const uploadId = url.searchParams.get("uploadId");
      const parts = await request.json();
      const upload = env.MODEL_BUCKET.resumeMultipartUpload(key, uploadId);
      const object = await upload.complete(parts);
      return Response.json({ size: object.size, etag: object.httpEtag });
    }
    if (request.method === "POST" && action === "abort") {
      const uploadId = url.searchParams.get("uploadId");
      const upload = env.MODEL_BUCKET.resumeMultipartUpload(key, uploadId);
      await upload.abort();
      return Response.json({ aborted: true });
    }
  } catch (error) {
    return new Response(`upload step failed: ${error}`, { status: 500 });
  }
  return new Response("unknown action", { status: 400 });
}
