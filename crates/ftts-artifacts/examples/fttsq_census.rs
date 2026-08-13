//! Byte census of a `.fttsq` artifact: exact composition by section, access class, and dtype.
//!
//! The consumer is the artifact-v2 quantization program (`frankentts-fs6z`): every "quantize X
//! next" decision is ordered by measured share of the file, and this is the measurement. NDJSON
//! on stdout, one row per aggregate, `kind` first so a stream consumer can route rows; nothing
//! human-decorated (AGENTS.md agent-ergonomics conventions).
//!
//! ```text
//! cargo run -p ftts-artifacts --example fttsq_census -- ~/.cache/franken_tts/model/qwen3-tts-12hz-0.6b-base.fttsq
//! ```
//!
//! Opening via [`MappedFttsq::open`] means every reported byte is digest-verified before it is
//! counted — a census of a corrupt artifact is refused, not reported.

use ftts_artifacts::fttsq::MappedFttsq;
use std::collections::BTreeMap;

fn row(kind: &str, fields: &[(&str, String)]) {
    let mut line = String::from("{\"kind\":\"");
    line.push_str(kind);
    line.push('"');
    for (key, value) in fields {
        line.push_str(",\"");
        line.push_str(key);
        line.push_str("\":");
        line.push_str(value);
    }
    line.push('}');
    println!("{line}");
}

fn quoted(text: &str) -> String {
    // Section/tensor names are ASCII identifiers by construction; no escaping cases exist.
    format!("\"{text}\"")
}

fn share(bytes: u64, total: u64) -> String {
    format!("{:.4}", bytes as f64 / total as f64)
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: fttsq_census <artifact.fttsq>");
        std::process::exit(2);
    });
    let mapped = match MappedFttsq::open(&path) {
        Ok(mapped) => mapped,
        Err(error) => {
            eprintln!("refusing census: {error}");
            std::process::exit(1);
        }
    };
    let reader = mapped.reader();
    let file_bytes = mapped.len() as u64;

    let mut class_bytes: BTreeMap<&str, u64> = BTreeMap::new();
    let mut sections_total = 0_u64;
    for section in reader.sections() {
        sections_total += section.length;
        *class_bytes
            .entry(section.access_class.as_str())
            .or_default() += section.length;
        row(
            "section",
            &[
                ("name", quoted(&section.name)),
                ("access_class", quoted(section.access_class.as_str())),
                ("bytes", section.length.to_string()),
                ("share_of_file", share(section.length, file_bytes)),
            ],
        );
    }

    // Dtype split inside each section: separates payload from quantization-scale overhead and
    // shows how much of a "quantized" section is still wide.
    let mut dtype_bytes: BTreeMap<(String, &str), u64> = BTreeMap::new();
    let mut scale_bytes: BTreeMap<String, u64> = BTreeMap::new();
    let scale_names: std::collections::BTreeSet<&str> = reader
        .tensors()
        .iter()
        .filter_map(|tensor| tensor.scales.as_deref())
        .collect();
    for tensor in reader.tensors() {
        *dtype_bytes
            .entry((tensor.section.clone(), tensor.dtype.as_str()))
            .or_default() += tensor.length;
        if scale_names.contains(tensor.name.as_str()) {
            *scale_bytes.entry(tensor.section.clone()).or_default() += tensor.length;
        }
    }
    for ((section, dtype), bytes) in &dtype_bytes {
        row(
            "section_dtype",
            &[
                ("section", quoted(section)),
                ("dtype", quoted(dtype)),
                ("bytes", bytes.to_string()),
                ("share_of_file", share(*bytes, file_bytes)),
            ],
        );
    }
    for (section, bytes) in &scale_bytes {
        row(
            "scale_overhead",
            &[
                ("section", quoted(section)),
                ("bytes", bytes.to_string()),
                ("share_of_file", share(*bytes, file_bytes)),
            ],
        );
    }

    for (class, bytes) in &class_bytes {
        row(
            "access_class",
            &[
                ("access_class", quoted(class)),
                ("bytes", bytes.to_string()),
                ("share_of_file", share(*bytes, file_bytes)),
            ],
        );
    }
    row(
        "total",
        &[
            ("file_bytes", file_bytes.to_string()),
            ("section_bytes", sections_total.to_string()),
            (
                "header_directory_bytes",
                (file_bytes - sections_total).to_string(),
            ),
            ("model_family", quoted(reader.model_family())),
        ],
    );
}
