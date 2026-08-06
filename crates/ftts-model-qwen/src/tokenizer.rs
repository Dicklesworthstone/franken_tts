//! Qwen3-TTS's pinned GPT-2-lineage byte-level BPE tokenizer.
//!
//! The official Qwen3-TTS entrypoints use transformers' accidental
//! `fix_mistral_regex=True` path.  `TokenizerRegex::Official` therefore means
//! that Mistral pre-tokenizer, not the Qwen-native expression.  Keep the
//! native expression available only as an explicit, visible experiment.

use std::{cmp::Reverse, collections::HashMap, env, fmt, ops::Range, str::FromStr};

use fancy_regex::Regex;
pub use ftts_core::{
    LanguageSpan, NormalizationChange, NormalizationMode, NormalizationOptions, NormalizationTrace,
    PronunciationEntry,
};
use ftts_core::{PreparedText, TextPreparationError, TextPreparer};
use unicode_normalization::UnicodeNormalization;

const OFFICIAL_PRETOKENIZER: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const NATIVE_PRETOKENIZER: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

type BpeRanks = HashMap<(String, String), usize>;
type AddedTokensById = HashMap<u32, String>;
type AddedTokensByText = Vec<(String, u32)>;

/// Which pre-tokenizer is applied before byte-level BPE.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TokenizerRegex {
    /// The exact regex used by the pinned official inference stack.
    #[default]
    Official,
    /// Qwen's own regex, retained for the explicitly visible listening test.
    Native,
}

impl TokenizerRegex {
    /// Reads the restoration switch, defaulting to the official contract.
    pub fn from_environment() -> Result<Self, TokenizerError> {
        match env::var("FTTS_TOKENIZER_REGEX") {
            Ok(value) => value.parse(),
            Err(env::VarError::NotPresent) => Ok(Self::Official),
            Err(error) => Err(TokenizerError::Environment(error.to_string())),
        }
    }

    fn expression(self) -> &'static str {
        match self {
            Self::Official => OFFICIAL_PRETOKENIZER,
            Self::Native => NATIVE_PRETOKENIZER,
        }
    }
}

impl fmt::Display for TokenizerRegex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Official => "official",
            Self::Native => "native",
        })
    }
}

impl FromStr for TokenizerRegex {
    type Err = TokenizerError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "official" => Ok(Self::Official),
            "native" => Ok(Self::Native),
            _ => Err(TokenizerError::InvalidRegexMode(value.to_owned())),
        }
    }
}

/// Text after requested normalization, plus its explainable trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalizedText {
    pub text: String,
    pub trace: NormalizationTrace,
}

/// The three pinned tokenizer files, supplied by the truth pack or model artifact loader.
#[derive(Clone, Copy, Debug)]
pub struct TokenizerFiles<'a> {
    pub vocab_json: &'a str,
    pub merges_txt: &'a str,
    pub tokenizer_config_json: &'a str,
}

/// Pure-Rust byte-level BPE with Qwen's added-token table.
#[derive(Debug)]
pub struct QwenTokenizer {
    vocab: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    bpe_ranks: BpeRanks,
    added_by_id: AddedTokensById,
    added_by_text: AddedTokensByText,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    regex_mode: TokenizerRegex,
    pretokenizer: Regex,
}

/// A tokenizer construction, tokenization, or normalizer validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizerError {
    Json(String),
    Regex(String),
    Environment(String),
    InvalidRegexMode(String),
    InvalidMerge(String),
    InvalidAddedTokenId(String),
    MissingAddedTokenContent(String),
    MissingVocabPiece(String),
    UnknownTokenId(u32),
    InvalidLanguageSpan {
        range: Range<usize>,
        text_len: usize,
    },
    OverlappingLanguageSpans,
    EmptyPronunciationSurface,
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(message) => write!(formatter, "invalid tokenizer JSON: {message}"),
            Self::Regex(message) => write!(formatter, "invalid tokenizer regex: {message}"),
            Self::Environment(message) => {
                write!(formatter, "cannot read tokenizer environment: {message}")
            }
            Self::InvalidRegexMode(value) => write!(
                formatter,
                "invalid FTTS_TOKENIZER_REGEX={value:?}; expected official or native"
            ),
            Self::InvalidMerge(line) => write!(formatter, "invalid BPE merge line: {line:?}"),
            Self::InvalidAddedTokenId(id) => write!(formatter, "invalid added-token id: {id:?}"),
            Self::MissingAddedTokenContent(id) => {
                write!(formatter, "added token {id:?} has no string content")
            }
            Self::MissingVocabPiece(piece) => {
                write!(formatter, "vocabulary lacks BPE piece {piece:?}")
            }
            Self::UnknownTokenId(id) => write!(formatter, "unknown tokenizer id {id}"),
            Self::InvalidLanguageSpan { range, text_len } => write!(
                formatter,
                "invalid language span {range:?} for normalized text of {text_len} bytes"
            ),
            Self::OverlappingLanguageSpans => formatter.write_str("language spans overlap"),
            Self::EmptyPronunciationSurface => {
                formatter.write_str("pronunciation entry surface is empty")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

impl QwenTokenizer {
    /// Parses the three pinned source files.  This accepts the artifacts at runtime instead of
    /// baking a machine-local truth-pack path into the library binary.
    pub fn from_files(
        files: TokenizerFiles<'_>,
        regex_mode: TokenizerRegex,
    ) -> Result<Self, TokenizerError> {
        let vocab: HashMap<String, u32> = serde_json::from_str(files.vocab_json)
            .map_err(|error| TokenizerError::Json(error.to_string()))?;
        let decoder = vocab
            .iter()
            .map(|(piece, id)| (*id, piece.clone()))
            .collect();
        let bpe_ranks = parse_merges(files.merges_txt)?;
        let (added_by_id, mut added_by_text) = parse_added_tokens(files.tokenizer_config_json)?;
        added_by_text.sort_unstable_by(|left, right| {
            right
                .0
                .len()
                .cmp(&left.0.len())
                .then_with(|| left.0.cmp(&right.0))
        });
        let (byte_encoder, byte_decoder) = byte_maps();
        let pretokenizer = Regex::new(regex_mode.expression())
            .map_err(|error| TokenizerError::Regex(error.to_string()))?;

        Ok(Self {
            vocab,
            decoder,
            bpe_ranks,
            added_by_id,
            added_by_text,
            byte_encoder,
            byte_decoder,
            regex_mode,
            pretokenizer,
        })
    }

    /// Uses the visible environment switch, defaulting to exact official behavior.
    pub fn from_files_using_environment(files: TokenizerFiles<'_>) -> Result<Self, TokenizerError> {
        Self::from_files(files, TokenizerRegex::from_environment()?)
    }

    pub fn regex_mode(&self) -> TokenizerRegex {
        self.regex_mode
    }

    pub fn vocabulary_len(&self) -> usize {
        self.vocab.len()
    }

    pub fn tokenizer_len(&self) -> usize {
        self.vocab.len() + self.added_by_id.len()
    }

    /// Normalizes with the upstream NFC baseline and applies explicit opt-in additions.
    pub fn normalize(
        &self,
        input: &str,
        options: &NormalizationOptions,
    ) -> Result<NormalizedText, TokenizerError> {
        let nfc: String = input.nfc().collect();
        let mut changes = Vec::new();
        if nfc != input {
            changes.push(NormalizationChange {
                rule: "unicode_nfc",
                before: input.to_owned(),
                after: nfc.clone(),
            });
        }

        let text = match options.mode {
            NormalizationMode::Verbatim | NormalizationMode::Conservative => nfc,
            NormalizationMode::LocaleAware => apply_locale_lexicon(
                nfc,
                &options.language_spans,
                &options.pronunciation_lexicon,
                &mut changes,
            )?,
        };

        Ok(NormalizedText {
            text,
            trace: NormalizationTrace {
                mode: options.mode,
                unicode_version: unicode_version(),
                changes,
            },
        })
    }

    /// Encodes under the pinned verbatim normalization contract.
    pub fn encode(&self, input: &str) -> Result<Vec<u32>, TokenizerError> {
        Ok(self
            .encode_with_normalization(input, &NormalizationOptions::default())?
            .0)
    }

    /// Encodes after an explicit normalization policy and returns its trace.
    pub fn encode_with_normalization(
        &self,
        input: &str,
        options: &NormalizationOptions,
    ) -> Result<(Vec<u32>, NormalizationTrace), TokenizerError> {
        let normalized = self.normalize(input, options)?;
        let ids = self.encode_normalized(&normalized.text)?;
        Ok((ids, normalized.trace))
    }

    /// Decodes ids using replacement semantics for an invalid UTF-8 byte sequence, matching the
    /// pinned `errors="replace"` tokenizer setting.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut output = String::new();
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(added) = self.added_by_id.get(id) {
                flush_bytes(&mut output, &mut bytes);
                output.push_str(added);
                continue;
            }
            let piece = self
                .decoder
                .get(id)
                .ok_or(TokenizerError::UnknownTokenId(*id))?;
            for symbol in piece.chars() {
                let byte = self
                    .byte_decoder
                    .get(&symbol)
                    .ok_or_else(|| TokenizerError::MissingVocabPiece(piece.clone()))?;
                bytes.push(*byte);
            }
        }
        flush_bytes(&mut output, &mut bytes);
        Ok(output)
    }

    fn encode_normalized(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut ids = Vec::new();
        let mut ordinary_start = 0;
        let mut offset = 0;

        while offset < text.len() {
            let remaining = &text[offset..];
            if let Some((special, id)) = self
                .added_by_text
                .iter()
                .find(|(special, _)| remaining.starts_with(special))
            {
                self.encode_ordinary(&text[ordinary_start..offset], &mut ids)?;
                ids.push(*id);
                offset += special.len();
                ordinary_start = offset;
            } else {
                let character = remaining
                    .chars()
                    .next()
                    .expect("offset is always a valid non-terminal UTF-8 boundary");
                offset += character.len_utf8();
            }
        }
        self.encode_ordinary(&text[ordinary_start..], &mut ids)?;
        Ok(ids)
    }

    fn encode_ordinary(&self, text: &str, output: &mut Vec<u32>) -> Result<(), TokenizerError> {
        for matched in self.pretokenizer.find_iter(text) {
            let matched = matched.map_err(|error| TokenizerError::Regex(error.to_string()))?;
            let pieces = self.bpe(matched.as_str())?;
            for piece in pieces {
                output.push(
                    *self
                        .vocab
                        .get(&piece)
                        .ok_or(TokenizerError::MissingVocabPiece(piece))?,
                );
            }
        }
        Ok(())
    }

    fn bpe(&self, token: &str) -> Result<Vec<String>, TokenizerError> {
        let mut pieces: Vec<String> = token
            .as_bytes()
            .iter()
            .map(|byte| self.byte_encoder[usize::from(*byte)].to_string())
            .collect();

        while pieces.len() > 1 {
            let Some(merge_index) = lowest_ranked_pair(&pieces, &self.bpe_ranks) else {
                break;
            };
            let merged = format!("{}{}", pieces[merge_index], pieces[merge_index + 1]);
            pieces.splice(merge_index..=merge_index + 1, [merged]);
        }
        Ok(pieces)
    }
}

impl TextPreparer for QwenTokenizer {
    fn prepare(
        &self,
        text: &str,
        options: &NormalizationOptions,
    ) -> Result<PreparedText, TextPreparationError> {
        let (token_ids, normalization_trace) = self
            .encode_with_normalization(text, options)
            .map_err(|error| TextPreparationError::new(error.to_string()))?;
        Ok(PreparedText::new(token_ids, normalization_trace))
    }
}

fn parse_merges(merges: &str) -> Result<BpeRanks, TokenizerError> {
    let mut ranks = HashMap::new();
    for (rank, line) in merges.lines().filter(|line| !line.is_empty()).enumerate() {
        if line.starts_with('#') {
            continue;
        }
        let (left, right) = line
            .split_once(' ')
            .filter(|(_, right)| !right.is_empty())
            .ok_or_else(|| TokenizerError::InvalidMerge(line.to_owned()))?;
        ranks.insert((left.to_owned(), right.to_owned()), rank);
    }
    Ok(ranks)
}

fn parse_added_tokens(
    tokenizer_config_json: &str,
) -> Result<(AddedTokensById, AddedTokensByText), TokenizerError> {
    let config: serde_json::Value = serde_json::from_str(tokenizer_config_json)
        .map_err(|error| TokenizerError::Json(error.to_string()))?;
    let added = config
        .get("added_tokens_decoder")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| TokenizerError::Json("missing added_tokens_decoder object".to_owned()))?;

    let mut by_id = HashMap::new();
    let mut by_text = Vec::new();
    for (id, token) in added {
        let id = id
            .parse::<u32>()
            .map_err(|_| TokenizerError::InvalidAddedTokenId(id.clone()))?;
        let content = token
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| TokenizerError::MissingAddedTokenContent(id.to_string()))?
            .to_owned();
        by_id.insert(id, content.clone());
        by_text.push((content, id));
    }
    Ok((by_id, by_text))
}

fn byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut codepoints: Vec<u32> = bytes.iter().map(|byte| u32::from(*byte)).collect();
    let mut next = 0_u32;
    for byte in 0_u8..=u8::MAX {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            codepoints.push(256 + next);
            next += 1;
        }
    }

    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::new();
    for (byte, codepoint) in bytes.into_iter().zip(codepoints) {
        let symbol = char::from_u32(codepoint).expect("GPT-2 byte mapping is valid Unicode");
        encoder[usize::from(byte)] = symbol;
        decoder.insert(symbol, byte);
    }
    (encoder, decoder)
}

fn lowest_ranked_pair(
    pieces: &[String],
    ranks: &HashMap<(String, String), usize>,
) -> Option<usize> {
    pieces
        .windows(2)
        .enumerate()
        .filter_map(|(index, pair)| {
            ranks
                .get(&(pair[0].clone(), pair[1].clone()))
                .map(|rank| (index, *rank))
        })
        .min_by_key(|(_, rank)| *rank)
        .map(|(index, _)| index)
}

fn flush_bytes(output: &mut String, bytes: &mut Vec<u8>) {
    if !bytes.is_empty() {
        output.push_str(&String::from_utf8_lossy(bytes));
        bytes.clear();
    }
}

fn unicode_version() -> String {
    let (major, minor, patch) = unicode_normalization::UNICODE_VERSION;
    format!("{major}.{minor}.{patch}")
}

fn apply_locale_lexicon(
    text: String,
    spans: &[LanguageSpan],
    lexicon: &[PronunciationEntry],
    changes: &mut Vec<NormalizationChange>,
) -> Result<String, TokenizerError> {
    validate_language_spans(&text, spans)?;
    if lexicon.iter().any(|entry| entry.surface.is_empty()) {
        return Err(TokenizerError::EmptyPronunciationSurface);
    }

    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    for span in spans {
        output.push_str(&replace_entries(
            &text[cursor..span.range.start],
            "und",
            lexicon,
            changes,
        ));
        output.push_str(&replace_entries(
            &text[span.range.clone()],
            &span.language,
            lexicon,
            changes,
        ));
        cursor = span.range.end;
    }
    output.push_str(&replace_entries(&text[cursor..], "und", lexicon, changes));
    Ok(output)
}

fn validate_language_spans(text: &str, spans: &[LanguageSpan]) -> Result<(), TokenizerError> {
    let mut previous_end = 0;
    for span in spans {
        if span.range.start > span.range.end
            || span.range.end > text.len()
            || !text.is_char_boundary(span.range.start)
            || !text.is_char_boundary(span.range.end)
        {
            return Err(TokenizerError::InvalidLanguageSpan {
                range: span.range.clone(),
                text_len: text.len(),
            });
        }
        if span.range.start < previous_end {
            return Err(TokenizerError::OverlappingLanguageSpans);
        }
        previous_end = span.range.end;
    }
    Ok(())
}

fn replace_entries(
    text: &str,
    language: &str,
    lexicon: &[PronunciationEntry],
    changes: &mut Vec<NormalizationChange>,
) -> String {
    let mut output = text.to_owned();
    let mut candidates: Vec<_> = lexicon
        .iter()
        .filter(|entry| entry.language == language || entry.language == "und")
        .collect();
    candidates.sort_unstable_by_key(|entry| Reverse(entry.surface.len()));
    for entry in candidates {
        if output.contains(&entry.surface) {
            let before = output.clone();
            output = output.replace(&entry.surface, &entry.spoken);
            changes.push(NormalizationChange {
                rule: "pronunciation_lexicon",
                before,
                after: output.clone(),
            });
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::*;

    fn fixture_files() -> TokenizerFiles<'static> {
        TokenizerFiles {
            vocab_json: r#"{"a":0,"b":1,"ab":2,"ÿ":3}"#,
            merges_txt: "#version: 0.2\na b\n",
            tokenizer_config_json: r#"{"added_tokens_decoder":{"4":{"content":"<s>"}}}"#,
        }
    }

    #[test]
    fn byte_level_bpe_merges_and_preserves_added_tokens() {
        let tokenizer = QwenTokenizer::from_files(fixture_files(), TokenizerRegex::Official)
            .expect("fixture tokenizer");
        assert_eq!(tokenizer.encode("ab<s>ab").expect("encode"), vec![2, 4, 2]);
        assert_eq!(tokenizer.decode(&[2, 4, 2]).expect("decode"), "ab<s>ab");
        assert_eq!(tokenizer.tokenizer_len(), 5);
    }

    #[test]
    fn decoder_replaces_invalid_utf8() {
        let tokenizer = QwenTokenizer::from_files(fixture_files(), TokenizerRegex::Official)
            .expect("fixture tokenizer");
        assert_eq!(tokenizer.decode(&[3]).expect("decode"), "�");
    }

    #[test]
    fn normalization_trace_matches_the_verbatim_snapshot() {
        let tokenizer = QwenTokenizer::from_files(fixture_files(), TokenizerRegex::Official)
            .expect("fixture tokenizer");
        let result = tokenizer
            .normalize("Cafe\u{301}", &NormalizationOptions::default())
            .expect("normalize");
        assert_eq!(result.text, "Café");
        assert_eq!(
            result.trace,
            NormalizationTrace {
                mode: NormalizationMode::Verbatim,
                unicode_version: unicode_version(),
                changes: vec![NormalizationChange {
                    rule: "unicode_nfc",
                    before: "Cafe\u{301}".to_owned(),
                    after: "Café".to_owned(),
                }],
            }
        );
        assert!(!result.trace.unicode_version.is_empty());
    }

    #[test]
    fn locale_aware_requires_explicit_span_and_lexicon() {
        let tokenizer = QwenTokenizer::from_files(fixture_files(), TokenizerRegex::Official)
            .expect("fixture tokenizer");
        let options = NormalizationOptions {
            mode: NormalizationMode::LocaleAware,
            language_spans: vec![LanguageSpan {
                range: 0..3,
                language: "en".to_owned(),
            }],
            pronunciation_lexicon: vec![PronunciationEntry {
                language: "en".to_owned(),
                surface: "GPU".to_owned(),
                spoken: "gee pee you".to_owned(),
            }],
        };
        let result = tokenizer.normalize("GPU", &options).expect("normalize");
        assert_eq!(result.text, "gee pee you");
        assert_eq!(result.trace.changes[0].rule, "pronunciation_lexicon");
    }

    #[test]
    fn tokenizer_preparer_preserves_the_engine_normalization_options() {
        let tokenizer = QwenTokenizer::from_files(fixture_files(), TokenizerRegex::Official)
            .expect("fixture tokenizer");
        let options = NormalizationOptions {
            mode: NormalizationMode::LocaleAware,
            language_spans: vec![LanguageSpan {
                range: 0..1,
                language: "en".to_owned(),
            }],
            pronunciation_lexicon: vec![PronunciationEntry {
                language: "en".to_owned(),
                surface: "a".to_owned(),
                spoken: "ab".to_owned(),
            }],
        };

        let prepared = tokenizer.prepare("a", &options).expect("prepare text");
        let expected = tokenizer
            .encode_with_normalization("a", &options)
            .expect("encode with the same options");
        assert_eq!(prepared.token_ids, expected.0);
        assert_eq!(prepared.normalization_trace, expected.1);
        assert_eq!(
            prepared.normalization_trace.summary().rules,
            vec!["pronunciation_lexicon"]
        );
    }

    #[test]
    fn invalid_language_spans_do_not_silently_apply() {
        let tokenizer = QwenTokenizer::from_files(fixture_files(), TokenizerRegex::Official)
            .expect("fixture tokenizer");
        let options = NormalizationOptions {
            mode: NormalizationMode::LocaleAware,
            language_spans: vec![LanguageSpan {
                range: 1..2,
                language: "en".to_owned(),
            }],
            pronunciation_lexicon: Vec::new(),
        };
        assert!(matches!(
            tokenizer.normalize("é", &options),
            Err(TokenizerError::InvalidLanguageSpan { .. })
        ));
    }

    #[test]
    fn regex_switch_is_explicit_and_rejects_typos() {
        assert_eq!("official".parse(), Ok(TokenizerRegex::Official));
        assert_eq!("native".parse(), Ok(TokenizerRegex::Native));
        assert!(matches!(
            "mistral".parse::<TokenizerRegex>(),
            Err(TokenizerError::InvalidRegexMode(_))
        ));
    }

    #[test]
    fn pinned_reference_corpus_is_id_exact_for_both_recorded_regexes() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let snapshot = root.join("docs/truth-pack/snapshots/hf");
        assert!(
            snapshot.join("vocab.json").is_file(),
            "pinned tokenizer inputs are required at {}",
            snapshot.display()
        );
        let vocab = fs::read_to_string(snapshot.join("vocab.json")).expect("read pinned vocab");
        let merges = fs::read_to_string(snapshot.join("merges.txt")).expect("read pinned merges");
        let config = fs::read_to_string(snapshot.join("tokenizer_config.json"))
            .expect("read pinned tokenizer config");
        let corpus: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("docs/truth-pack/tokenizer/tokenizer_conformance.json"))
                .expect("read generated conformance corpus"),
        )
        .expect("parse generated conformance corpus");
        let files = TokenizerFiles {
            vocab_json: &vocab,
            merges_txt: &merges,
            tokenizer_config_json: &config,
        };

        let mut recorded_divergences = 0;
        for mode in [TokenizerRegex::Official, TokenizerRegex::Native] {
            let tokenizer = QwenTokenizer::from_files(files, mode).expect("pinned tokenizer");
            assert_eq!(tokenizer.vocabulary_len(), 151_643);
            assert_eq!(tokenizer.tokenizer_len(), 151_676);
            for case in corpus["cases"].as_array().expect("cases array") {
                let text = case["text"].as_str().expect("case text");
                let expected = case[match mode {
                    TokenizerRegex::Official => "ids",
                    TokenizerRegex::Native => "ids_native_qwen_regex",
                }]
                .as_array()
                .expect("case ids")
                .iter()
                .map(|id| id.as_u64().expect("id") as u32)
                .collect::<Vec<_>>();
                if mode == TokenizerRegex::Official
                    && expected
                        != case["ids_native_qwen_regex"]
                            .as_array()
                            .expect("native case ids")
                            .iter()
                            .map(|id| id.as_u64().expect("native id") as u32)
                            .collect::<Vec<_>>()
                {
                    recorded_divergences += 1;
                }
                let actual = tokenizer.encode(text).expect("encode pinned corpus case");
                assert_eq!(actual, expected, "case {} ({mode})", case["name"]);
                let expected_decoded: String = text.nfc().collect();
                assert_eq!(
                    tokenizer
                        .decode(&actual)
                        .expect("decode pinned corpus case"),
                    expected_decoded,
                    "decode case {} ({mode})",
                    case["name"]
                );
            }
        }
        assert_eq!(recorded_divergences, 6, "pinned OQ-11 divergence count");

        let tokenizer = QwenTokenizer::from_files(files, TokenizerRegex::Official)
            .expect("pinned official tokenizer");
        let alphabet: Vec<char> = "aA0 /ไทย🤖\r\n".chars().collect();
        let mut state = 0xD1CE_F00D_u64;
        for case_number in 0..128 {
            let length = (case_number * 17) % 257;
            let mut text = String::with_capacity(length * 4);
            for _ in 0..length {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let alphabet_len = u64::try_from(alphabet.len()).expect("small test alphabet");
                let index = usize::try_from(state % alphabet_len).expect("bounded alphabet index");
                text.push(alphabet[index]);
            }
            let ids = tokenizer
                .encode(&text)
                .expect("deterministic Unicode fuzz encode");
            let expected: String = text.nfc().collect();
            assert_eq!(
                tokenizer
                    .decode(&ids)
                    .expect("deterministic Unicode fuzz decode"),
                expected,
                "deterministic Unicode fuzz case {case_number}"
            );
        }
    }

    #[test]
    fn deterministic_extreme_unicode_inputs_are_total() {
        let tokenizer = QwenTokenizer::from_files(fixture_files(), TokenizerRegex::Official)
            .expect("fixture tokenizer");
        let inputs = [String::new(), "a".repeat(8_192), "a\u{301}".repeat(2_048)];
        for input in &inputs {
            let normalized = tokenizer
                .normalize(input, &NormalizationOptions::default())
                .expect("normalization remains total");
            assert_eq!(normalized.text, input.nfc().collect::<String>());
        }
        let repeated = "a".repeat(8_192);
        assert_eq!(
            tokenizer
                .encode(&repeated)
                .expect("extreme BPE encode")
                .len(),
            repeated.len()
        );
    }
}
