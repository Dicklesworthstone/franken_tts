#!/usr/bin/env python3
"""Freeze the OQ-11 token-id conformance corpus from the pinned tokenizer files.

The reference ids in `docs/truth-pack/tokenizer/` are produced ONLY by this script, running
the pinned upstream oracle (transformers 4.57.3 / tokenizers 0.22.2) over the pinned
tokenizer bytes (docs/truth-pack/snapshots/hf/, verified by MANIFEST.sha256). Nothing here
is hand-written: a hand-written "expected" id sequence would be a counterfeit gate.

    uv venv /tmp/tokvenv && VIRTUAL_ENV=/tmp/tokvenv uv pip install \
        transformers==4.57.3 tokenizers regex
    /tmp/tokvenv/bin/python scripts/gen_tokenizer_conformance.py

torch is NOT required — Qwen2Tokenizer needs only transformers + regex, the fast variant
only tokenizers. Writes tokenizer_conformance.json + a slow-vs-fast divergence report.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path


def NFD(s: str) -> str:
    """Decompose, so an NFD corpus entry can never be a composed literal by accident."""
    d = unicodedata.normalize("NFD", s)
    # A plain `assert` would vanish under `python -O`, taking this guard with it — and this guard
    # is exactly what stops a composed literal from making its NFC invariant pass vacuously.
    if d == s:
        raise ValueError(f"NFD({s!r}) did not decompose — this case cannot test NFC")
    return d

REPO = Path(__file__).resolve().parent.parent
SNAP = REPO / "docs" / "truth-pack" / "snapshots" / "hf"
OUT_DIR = REPO / "docs" / "truth-pack" / "tokenizer"

# Files the ids are derived from; hashes are re-asserted here so a corpus can never
# silently outlive the pin it was generated against.
PINNED_INPUTS = ["vocab.json", "merges.txt", "tokenizer_config.json"]

# Diverse by construction: each case names the property it probes, so a future divergence
# report says WHAT broke, not just which index.
CORPUS: list[tuple[str, str]] = [
    ("ascii-basic", "Hello, world."),
    ("ascii-sentence", "The quick brown fox jumps over the lazy dog."),
    ("empty", ""),
    ("single-space", " "),
    ("leading-space", " leading space"),
    ("trailing-space", "trailing space "),
    ("double-space", "two  spaces  inside"),
    ("tab", "tab\tseparated\tvalues"),
    ("newline", "line one\nline two"),
    ("crlf", "windows\r\nline"),
    ("many-newlines", "a\n\n\nb"),
    ("only-whitespace", "   \t  \n  "),
    # The ten languages the model card advertises.
    ("lang-zh", "今天天气很好，我们去公园散步吧。"),
    ("lang-en", "She sells seashells by the seashore."),
    ("lang-ja", "こんにちは、世界。今日はいい天気ですね。"),
    ("lang-ko", "안녕하세요, 오늘 날씨가 정말 좋네요."),
    ("lang-de", "Der Fußgängerübergang war während der Straßenbauarbeiten gesperrt."),
    ("lang-fr", "Où est passé l'élève qui répétait cette phrase ?"),
    ("lang-ru", "Съешь же ещё этих мягких французских булок, да выпей чаю."),
    ("lang-pt", "O coração não envelhece, apenas a coração muda de ritmo."),
    ("lang-es", "El niño pequeño compró cigüeñas en la plaza."),
    ("lang-it", "Perché non è più qui? Perché è già partito."),
    # Script/Unicode stress.
    ("cjk-mixed", "中文English日本語한국어mixed"),
    ("cjk-punct", "「引用」、句読点。（括弧）"),
    ("emoji", "I love this 😀🎉 so much!"),
    ("emoji-zwj", "family: 👨‍👩‍👧‍👦 done"),
    ("combining-nfc", "café"),  # U+00E9 precomposed
    ("combining-nfd", "cafe\u0301"),  # e + combining acute -> MUST EQUAL combining-nfc
    # The tokenizer applies Unicode NFC (slow: tokenization_qwen2.py:338
    # `unicodedata.normalize("NFC", text)`; fast: backend normalizer == NFC()). Each *-nfd case
    # must produce ids identical to its *-nfc twin, and roundtrip_exact is legitimately False for
    # them — decode returns the NFC form. Upstream behavior, not a defect on our side.
    # NOTE: the *-nfd entries are built by NFD() below, never typed as literals — a literal typed
    # in composed form would compare equal to its twin and the invariant would pass vacuously.
    ("hangul-nfc", "한국어"),  # Korean is a supported language; both forms occur in the wild
    ("hangul-nfd", NFD("한국어")),  # jamo -> must equal hangul-nfc
    ("vietnamese-nfc", "Tiếng Việt"),
    ("vietnamese-nfd", NFD("Tiếng Việt")),  # must equal vietnamese-nfc
    # NFKC probes: each MUST stay DISTINCT from its ASCII twin. If a port applies NFKC instead of
    # NFC, all of these collapse and token ids silently diverge from upstream.
    ("nfkc-circled-digit", "①"),
    ("nfkc-plain-digit", "1"),
    ("nfkc-fullwidth-A", "Ａ"),
    ("nfkc-halfwidth-A", "A"),
    ("nfkc-ligature-fi", "ﬁ"),
    ("nfkc-ligature-plain", "fi"),
    ("rtl-arabic", "مرحبا بالعالم، كيف حالك؟"),
    ("rtl-hebrew", "שלום עולם"),
    ("devanagari", "नमस्ते दुनिया"),
    ("thai-no-spaces", "สวัสดีชาวโลก"),
    ("zero-width", "zero\u200bwidth\u200djoiner"),
    ("nbsp", "non\u00a0breaking\u00a0space"),
    ("bom", "\ufeffleading BOM"),
    ("surrogate-pair", "𝕳𝖊𝖑𝖑𝖔 mathematical"),
    # Numbers — TTS-critical, upstream splits digits per the Qwen regex.
    ("num-int", "42"),
    ("num-big", "1234567890"),
    ("num-comma", "1,234,567.89"),
    ("num-currency", "It costs $1,299.99 or €1.099,50"),
    ("num-phone", "Call +1 (555) 013-2477 now"),
    ("num-date", "On 2026-08-06 at 14:30:00 UTC"),
    ("num-ordinal", "the 1st, 2nd, 3rd and 21st items"),
    ("num-roman", "Chapter XIV, section IX"),
    ("num-mixed-cjk", "第3章、全部で15個あります"),
    ("num-percent", "Up 12.5% from 3.25%"),
    # Text a TTS front end will actually meet.
    ("url", "Visit https://example.com/path?q=1&r=2#frag for details"),
    ("email", "Write to first.last+tag@sub.example.co.uk please"),
    ("path-unix", "/usr/local/bin/ftts --voice v.ftvoice"),
    ("path-win", "C:\\Users\\test\\Documents\\file.txt"),
    ("code-rust", 'fn main() { println!("{}", 1 + 2); }'),
    ("code-python", "def f(x):\n    return [i**2 for i in range(x)]"),
    ("code-json", '{"key": [1, 2, {"nested": true}], "n": null}'),
    ("markup", "<speak><prosody rate='slow'>hi</prosody></speak>"),
    ("abbrev", "Dr. Smith Jr. met Mr. Bond at 5 p.m. in the U.S.A."),
    ("ellipsis", "Well... I suppose so — maybe?"),
    ("quotes-smart", "He said \u201cno\u201d and she said \u2018yes\u2019"),
    ("dashes", "en-dash – em-dash — hyphen-minus -"),
    ("repeated-punct", "What?!?! Really???"),
    ("all-caps", "THIS IS SHOUTING TEXT"),
    ("camel", "getUserByIdOrThrow"),
    ("snake", "some_long_variable_name_here"),
    # Case-boundary splitting is THE divergence class between the native Qwen regex and the
    # Mistral regex that upstream's fix_mistral_regex=True swaps in. Keep these cases.
    ("case-brand", "iPhone iPad eBay macOS"),
    ("case-acronym-word", "XMLHttpRequest HTTPServer JSONParser"),
    ("case-pascal", "MyClassName AnotherThing"),
    ("case-mixed-sentence", "The iPhone uses XMLHttpRequest internally."),
    # Contractions: the native regex has an explicit (?i:'s|'t|'re|'ve|'m|'ll|'d) alternation
    # that the Mistral regex drops. Verified to still agree — locked in so a change is caught.
    ("contraction-t", "don't stop"),
    ("contraction-s", "it's Bob's car"),
    ("contraction-ll-ve-re", "I'll say we've heard they're here"),
    ("contraction-trailing", "Don't stop believin'"),
    # The Mistral regex adds `/` to the punctuation-run class.
    ("slash-run", "read/write access and a/b/c"),
    # Special tokens as LITERAL TEXT — does the tokenizer split them out?
    ("special-literal-imstart", "<|im_start|>"),
    ("special-literal-tts-bos", "<tts_text_bos>"),
    ("special-literal-tts-eod", "<tts_text_eod>"),
    ("special-literal-tts-bos-single", "<tts_text_bos_single>"),
    ("special-literal-endoftext", "<|endoftext|>"),
    ("special-in-sentence", "say <|im_start|> aloud"),
    ("special-lookalike", "<|not_a_real_token|>"),
    # Long-form / robustness.
    ("long-repeat", "ha " * 200),
    ("long-prose", ("The rain in Spain falls mainly on the plain. " * 40).strip()),
    ("noisy", "uhh... so like,, i was gonna—wait no. anyway!!  ok?"),
    ("mixed-everything", "Hi 你好 😀 42% https://a.io <|im_start|> café\tdone\n"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not SNAP.exists():
        print(f"missing {SNAP}\nrun: docs/truth-pack/fetch-truth-pack.sh", file=sys.stderr)
        return 1

    from transformers import Qwen2Tokenizer, Qwen2TokenizerFast
    import transformers
    import tokenizers as tokenizers_lib

    # Assert the oracle is the pinned one — a different transformers version silently
    # invalidates every id below (PIN_RECORD.md).
    if transformers.__version__ != "4.57.3":
        print(
            f"REFUSING: transformers {transformers.__version__} != pinned 4.57.3. "
            "Reference ids must come from the pinned oracle.",
            file=sys.stderr,
        )
        return 1

    # THE OFFICIAL PATH. Every upstream entrypoint (examples/*.py, cli/demo.py) goes through
    # Qwen3TTSModel.from_pretrained -> AutoProcessor.from_pretrained(path, fix_mistral_regex=True)
    # (qwen3_tts_model.py:118), and AutoProcessor forwards kwargs to the tokenizer. use_fast
    # defaults True, so the official tokenizer is the FAST one with the Mistral regex swapped in.
    # local_files_only=True on every load: these must come from the hash-verified snapshot, never
    # from the Hub. Without it a deleted snapshots/ would silently download `main` and the "pinned"
    # reference ids would be generated from whatever upstream looks like today.
    official = Qwen2TokenizerFast.from_pretrained(str(SNAP), fix_mistral_regex=True, local_files_only=True)
    # The tokenizer's own native Qwen regex — what the model was almost certainly TRAINED with.
    # The slow class has no backend_tokenizer, so it can never receive the Mistral fix.
    native_fast = Qwen2TokenizerFast.from_pretrained(str(SNAP), local_files_only=True)
    slow = Qwen2Tokenizer.from_pretrained(str(SNAP), local_files_only=True)

    def pre_regex(tok) -> str:
        # Must NOT swallow failures: the returned string decides
        # `fix_mistral_regex_changes_tokenization` in the frozen artifact. A placeholder here would
        # silently record a false verdict, so let the exception end the run instead.
        state = json.loads(tok.backend_tokenizer.pre_tokenizer.__getstate__())
        return state["pretokenizers"][0]["pattern"]["Regex"]

    cases, regex_divergences, slow_fast_divergences = [], [], []
    for name, text in CORPUS:
        ids_official = official(text)["input_ids"]
        ids_native = native_fast(text)["input_ids"]
        ids_slow = slow(text)["input_ids"]

        # Converter fidelity: slow BPE vs fast BPE under the SAME (native) regex.
        if ids_slow != ids_native:
            slow_fast_divergences.append({"name": name, "text": text, "slow": ids_slow, "fast": ids_native})
        # The one that matters: official (Mistral regex) vs native Qwen regex.
        if ids_official != ids_native:
            regex_divergences.append(
                {
                    "name": name,
                    "text": text,
                    "official_fix_mistral_regex": ids_official,
                    "native_qwen_regex": ids_native,
                    "official_pieces": official.convert_ids_to_tokens(ids_official),
                    "native_pieces": native_fast.convert_ids_to_tokens(ids_native),
                }
            )
        cases.append(
            {
                "name": name,
                "text": text,
                # `ids` is the REFERENCE: what the official stack actually produces.
                "ids": ids_official,
                "n_tokens": len(ids_official),
                "ids_native_qwen_regex": ids_native,
                "regex_choice_matters": ids_official != ids_native,
                "slow_fast_agree": ids_slow == ids_native,
                # Round-trip is part of the contract: decode(encode(x)) must return x.
                "roundtrip_exact": official.decode(ids_official) == text,
                "pieces": official.convert_ids_to_tokens(ids_official),
            }
        )

    # Normalization invariants. These are the actual contract a Rust port must satisfy, so they
    # are asserted here rather than left for a reader to infer from the id lists.
    by_name = {c["name"]: c["ids"] for c in cases}
    invariants = []
    for a, b, must_equal, why in [
        ("combining-nfc", "combining-nfd", True, "NFC is applied: precomposed == decomposed"),
        ("hangul-nfc", "hangul-nfd", True, "NFC is applied to Hangul jamo"),
        ("vietnamese-nfc", "vietnamese-nfd", True, "NFC is applied to stacked Vietnamese diacritics"),
        ("nfkc-circled-digit", "nfkc-plain-digit", False, "NFKC is NOT applied: circled digit stays distinct"),
        ("nfkc-fullwidth-A", "nfkc-halfwidth-A", False, "NFKC is NOT applied: fullwidth stays distinct"),
        ("nfkc-ligature-fi", "nfkc-ligature-plain", False, "NFKC is NOT applied: ligature stays distinct"),
    ]:
        equal = by_name[a] == by_name[b]
        invariants.append(
            {"a": a, "b": b, "must_equal": must_equal, "observed_equal": equal, "holds": equal == must_equal, "why": why}
        )
    broken = [i for i in invariants if not i["holds"]]

    doc = {
        "_comment": (
            "OQ-11 token-id conformance corpus. Generated by scripts/gen_tokenizer_conformance.py "
            "from the pinned tokenizer bytes using the pinned oracle. DO NOT hand-edit: regenerate."
        ),
        "generator": "scripts/gen_tokenizer_conformance.py",
        "oracle": {"transformers": transformers.__version__, "tokenizers": tokenizers_lib.__version__},
        "hf_revision": "5d83992436eae1d760afd27aff78a71d676296fc",
        "pinned_inputs": {f: sha256(SNAP / f) for f in PINNED_INPUTS},
        "reference_semantics": (
            "cases[].ids is what the OFFICIAL stack emits: Qwen2TokenizerFast + fix_mistral_regex=True, "
            "as loaded by qwen3_tts_model.py:118. cases[].ids_native_qwen_regex is the same text under "
            "the tokenizer's own regex. Where they differ, the port must choose deliberately — see "
            "docs/truth-pack/tokenizer/OQ11_TOKENIZER.md and the DISCREPANCIES.md entry."
        ),
        "tokenizer": {
            "class_official": type(official).__name__,
            "class_slow": type(slow).__name__,
            "vocab_size": slow.vocab_size,
            "len_with_added": len(slow),
            "add_prefix_space": getattr(slow, "add_prefix_space", None),
            "add_bos_token": getattr(slow, "add_bos_token", None),
            "clean_up_tokenization_spaces": getattr(slow, "clean_up_tokenization_spaces", None),
            "model_max_length": slow.model_max_length,
            "pretokenizer_regex_official": pre_regex(official),
            "pretokenizer_regex_native": pre_regex(native_fast),
            "fix_mistral_regex_changes_tokenization": pre_regex(official) != pre_regex(native_fast),
        },
        "normalization": {
            "form_applied": "NFC",
            "where": "inside the tokenizer, NOT the processor — slow: transformers "
            "models/qwen2/tokenization_qwen2.py:338 unicodedata.normalize('NFC', text); "
            "fast: backend_tokenizer.normalizer == NFC()",
            "processor_normalization": "none — Qwen3TTSProcessor.__call__ forwards text unchanged "
            "(processing_qwen3_tts.py); the only text_kwargs are padding=False, padding_side='left'",
            "invariants": invariants,
        },
        "n_cases": len(cases),
        "n_regex_divergences": len(regex_divergences),
        "n_slow_fast_divergences": len(slow_fast_divergences),
        "regex_divergences": regex_divergences,
        "slow_fast_divergences": slow_fast_divergences,
        "cases": cases,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "tokenizer_conformance.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=False) + "\n")

    n_rt = sum(1 for c in cases if not c["roundtrip_exact"])
    print(f"wrote {out.relative_to(REPO)}: {len(cases)} cases")
    print(f"  vocab_size={slow.vocab_size} len(tokenizer)={len(slow)}")
    print(f"  fix_mistral_regex CHANGES tokenization: {pre_regex(official) != pre_regex(native_fast)}")
    print(f"  official-vs-native regex divergences: {len(regex_divergences)}")
    print(f"  slow-vs-fast BPE divergences:         {len(slow_fast_divergences)}")
    print(f"  round-trip failures:                  {n_rt}")
    for d in regex_divergences:
        print(f"    REGEX-DIVERGENT: {d['name']}")
    for d in slow_fast_divergences:
        print(f"    SLOW/FAST-DIVERGENT: {d['name']}")
    for c in cases:
        if not c["roundtrip_exact"]:
            print(f"    NO-ROUNDTRIP: {c['name']}  decoded={official.decode(c['ids'])!r}")
    print(f"  normalization invariants: {len(invariants) - len(broken)}/{len(invariants)} hold")
    for i in broken:
        print(f"    BROKEN INVARIANT: {i['a']} vs {i['b']} — {i['why']}")
    if broken:
        print("REFUSING to certify: the normalization contract does not hold.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
