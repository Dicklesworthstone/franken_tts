// The pinned model files, mirrored from crates/ftts-cli/pinned/model_manifest.json.
// Sizes and digests are load-bearing: the loader verifies every cached byte against
// them before the engine sees anything.
//
// `sha256` covers the whole file and is what a FRESH download is checked against — that hash is
// computed from the bytes as they stream past, so it costs nothing extra and stays the strongest
// claim we can make about a file we just fetched.
//
// `head`/`tail` are SHA-256 over the first and last ENDPOINT_BYTES, and exist for the far more
// common case: a file already sitting in OPFS from a previous visit. Re-reading 2 GB through a JS
// hash to prove that costs ~10 s on a desktop and much worse on a phone, every single page load.
// Checking exact byte length plus both endpoints reads 20 MB instead of 1,994 MB.
//
// What that does and does not prove is stated honestly in docs/DISCREPANCIES.md (DISC-004): it
// catches truncation, short/failed writes, a partially-overwritten file, a stale manifest, and any
// wrong-file substitution — the failure modes OPFS actually produces. It does NOT prove the middle
// of the file is untouched, so it is a cache-integrity check, not a supply-chain signature. The
// bytes were fully verified when they were downloaded; this re-proves they are still the same
// file, not that a determined attacker with write access to OPFS left the interior alone.
export const ENDPOINT_BYTES = 10 * 1024 * 1024;

export const MODEL_FILES = [
  {
    asset: "qwen3-tts-12hz-0.6b-base.fttsq",
    key: "fttsq",
    bytes: 1312015713,
    sha256: "597f7eb3314a2fe5be74fa10a6a3a28ace9e10e582c641deccd37348a0ccd824",
    head: "b9129421eba7b8a4d4a9b94ddf97afeef87651596f5d57a91ab7d34d985ed8f7",
    tail: "42b1e46ec86720c7f5e93d46810e7522fd1bb0353d06aaaa2077f51cee2a665c",
  },
  {
    asset: "speech_tokenizer_model.safetensors",
    key: "codec",
    bytes: 682293092,
    sha256: "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
    head: "75ea542e2ddc876b3e58c8216e27d30568776fd1bd5fa897113b3b88e191da75",
    tail: "eb396bbf0d4356aeab926bdd64d2c2bb6f3124900bb48fd52d6eed7467bdbdeb",
  },
  {
    asset: "vocab.json",
    key: "vocab",
    text: true,
    bytes: 2776833,
    sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
  },
  {
    asset: "merges.txt",
    key: "merges",
    text: true,
    bytes: 1671839,
    sha256: "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
  },
  {
    asset: "tokenizer_config.json",
    key: "tokenizerConfig",
    text: true,
    bytes: 7344,
    sha256: "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670",
  },
];

export const TOTAL_BYTES = MODEL_FILES.reduce((sum, file) => sum + file.bytes, 0);
export const CHUNK_BYTES = 32 * 1024 * 1024; // 32 MiB ranges: few requests, resumable.
