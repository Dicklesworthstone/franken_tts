#![no_main]

use ftts_artifacts::fttsq::FttsqReader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(reader) = FttsqReader::parse_directory(data) else {
        return;
    };

    for tensor in reader.tensors() {
        let _ = reader.tensor_bytes(&tensor.name, data);
    }
    let _ = reader.verify_digests(data);
    let _ = FttsqReader::open(data);
});
