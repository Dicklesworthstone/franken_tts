#![no_main]

use ftts_artifacts::safetensors::SafetensorsIndex;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(index) = SafetensorsIndex::parse(data) else {
        return;
    };

    for name in index.names() {
        let Some(view) = index.view(name, data) else {
            continue;
        };
        let _ = view.get_f32(0);
        let _ = view.get_f32(view.len().saturating_sub(1));
    }
});
