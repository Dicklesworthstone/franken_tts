// The pinned model files, mirrored from crates/ftts-cli/pinned/model_manifest.json.
// Sizes and digests are load-bearing: the loader verifies every cached byte against
// them before the engine sees anything.
export const MODEL_FILES = [
  {
    asset: "qwen3-tts-12hz-0.6b-base.fttsq",
    key: "fttsq",
    bytes: 1312015713,
    sha256: "597f7eb3314a2fe5be74fa10a6a3a28ace9e10e582c641deccd37348a0ccd824",
  },
  {
    asset: "speech_tokenizer_model.safetensors",
    key: "codec",
    bytes: 682293092,
    sha256: "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
  },
  {
    asset: "vocab.json",
    key: "vocab",
    bytes: 2776833,
    sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
  },
  {
    asset: "merges.txt",
    key: "merges",
    bytes: 1671839,
    sha256: "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
  },
  {
    asset: "tokenizer_config.json",
    key: "tokenizerConfig",
    bytes: 7344,
    sha256: "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670",
  },
];

export const TOTAL_BYTES = MODEL_FILES.reduce((sum, file) => sum + file.bytes, 0);
export const CHUNK_BYTES = 32 * 1024 * 1024; // 32 MiB ranges: few requests, resumable.
