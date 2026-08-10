// Serves site/ the way Cloudflare Pages does, so a local browser run is not a different product.
//
// Two things must match production or the test is worthless:
//
//   COOP/COEP — these decide `crossOriginIsolated`, which decides WHICH BUILD engine-worker.js
//               loads. Serving without them silently exercises only the serial path, which is how
//               a threaded-path bug reaches users past a green local run. Read from site/_headers
//               rather than restated here, so the file that ships is the file under test.
//   /model/*  — Pages proxies these with byte-Range support and loader.js depends on it for
//               chunked, resumable downloads. Without Range the loader takes a path no user takes.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

/// The global headers from site/_headers, parsed rather than duplicated.
function globalHeaders(siteDir) {
  const file = path.join(siteDir, "_headers");
  const headers = {};
  if (!fs.existsSync(file)) return headers;
  let inGlobal = false;
  for (const raw of fs.readFileSync(file, "utf8").split("\n")) {
    const line = raw.trimEnd();
    if (!line || line.trimStart().startsWith("#")) continue;
    if (!line.startsWith(" ") && !line.startsWith("\t")) {
      inGlobal = line.trim() === "/*";
      continue;
    }
    if (!inGlobal) continue;
    const [name, ...rest] = line.trim().split(":");
    headers[name.trim()] = rest.join(":").trim();
  }
  return headers;
}

export function serve({ siteDir, modelFiles, port = 0 }) {
  const base = globalHeaders(siteDir);
  const server = http.createServer((request, response) => {
    const url = new URL(request.url, "http://localhost");
    const send = (status, headers, body) => {
      response.writeHead(status, { ...base, ...headers });
      if (body) response.end(body);
      else response.end();
    };

    // Model proxy with Range, mirroring the Pages function.
    if (url.pathname.startsWith("/model/")) {
      const name = decodeURIComponent(url.pathname.slice("/model/".length));
      const file = modelFiles[name];
      if (!file || !fs.existsSync(file)) return send(404, {}, "no such model asset");
      const size = fs.statSync(file).size;
      const range = request.headers.range;
      if (range) {
        const match = /bytes=(\d+)-(\d*)/.exec(range);
        const start = Number(match[1]);
        const end = match[2] ? Number(match[2]) : size - 1;
        response.writeHead(206, {
          ...base,
          "Content-Type": "application/octet-stream",
          "Content-Length": end - start + 1,
          "Content-Range": `bytes ${start}-${end}/${size}`,
          // Required under COEP: require-corp, exactly as the deployed proxy must set it.
          "Cross-Origin-Resource-Policy": "same-origin",
        });
        return fs.createReadStream(file, { start, end }).pipe(response);
      }
      response.writeHead(200, {
        ...base,
        "Content-Type": "application/octet-stream",
        "Content-Length": size,
        "Cross-Origin-Resource-Policy": "same-origin",
      });
      return fs.createReadStream(file).pipe(response);
    }

    const relative = url.pathname === "/" ? "index.html" : url.pathname.replace(/^\//, "");
    const file = path.join(siteDir, relative);
    if (!file.startsWith(siteDir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      return send(404, { "Content-Type": "text/plain" }, "not found");
    }
    send(
      200,
      {
        "Content-Type": MIME[path.extname(file)] ?? "application/octet-stream",
        "Cross-Origin-Resource-Policy": "same-origin",
      },
      fs.readFileSync(file),
    );
  });
  return new Promise((resolve) => {
    server.listen(port, "127.0.0.1", () =>
      resolve({ server, port: server.address().port, headers: base }),
    );
  });
}
