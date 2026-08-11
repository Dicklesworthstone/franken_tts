//! SHA-256 (FIPS 180-4), for `.fttsq` section digests.
//!
//! # Why both implementations exist
//!
//! Streaming writers and file hashing retain the small stateful implementation below. Mapped
//! artifact verification has the complete section in memory and uses RustCrypto's one-shot
//! implementation, which runtime-dispatches to SHA-NI where available.
//!
//! # Scope
//!
//! Digests here are **integrity** checks — they detect truncation, bit-flips, and mismatched
//! sections. They are not a signature scheme and prove nothing about *who* produced an artifact.
//! Verified against the FIPS 180-4 vectors in the tests below.

/// Round constants: the first 32 bits of the fractional parts of the cube roots of the first 64
/// primes (FIPS 180-4 §4.2.2).
const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Initial hash value: the first 32 bits of the fractional parts of the square roots of the first
/// eight primes (FIPS 180-4 §5.3.3).
const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// Streaming SHA-256 state.
///
/// Streaming rather than one-shot because a `.fttsq` section is hundreds of megabytes: the digest
/// must be computable over a borrowed slice in blocks, without a second copy of the payload.
#[derive(Clone, Debug)]
pub struct Sha256 {
    state: [u32; 8],
    /// Partial block awaiting a full 64 bytes.
    buffer: [u8; 64],
    /// Bytes currently held in `buffer`.
    buffered: usize,
    /// Total message length in bytes, for the length suffix.
    length: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Starts a new digest.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: H0,
            buffer: [0; 64],
            buffered: 0,
            length: 0,
        }
    }

    /// Absorbs more message bytes.
    pub fn update(&mut self, mut input: &[u8]) {
        self.length = self.length.wrapping_add(input.len() as u64);

        if self.buffered > 0 {
            let want = 64 - self.buffered;
            let take = want.min(input.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&input[..take]);
            self.buffered += take;
            input = &input[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            } else {
                return;
            }
        }

        let (blocks, tail) = input.as_chunks::<64>();
        for block in blocks {
            self.compress(block);
        }

        self.buffer[..tail.len()].copy_from_slice(tail);
        self.buffered = tail.len();
    }

    /// Finishes the digest and returns the 32 raw bytes.
    #[must_use]
    pub fn finish(mut self) -> [u8; 32] {
        // Padding: 0x80, then zeros, then the bit length as a big-endian u64.
        let bit_length = self.length.wrapping_mul(8);
        self.update_no_count(&[0x80]);
        while self.buffered != 56 {
            self.update_no_count(&[0x00]);
        }
        self.update_no_count(&bit_length.to_be_bytes());

        let mut out = [0_u8; 32];
        for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            *chunk = word.to_be_bytes();
        }
        out
    }

    /// Absorbs padding bytes without advancing the message-length counter.
    fn update_no_count(&mut self, input: &[u8]) {
        for &byte in input {
            self.buffer[self.buffered] = byte;
            self.buffered += 1;
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
    }

    /// One 64-byte block through the compression function (FIPS 180-4 §6.2.2).
    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0_u32; 64];
        for (slot, chunk) in w.iter_mut().zip(block.as_chunks::<4>().0) {
            *slot = u32::from_be_bytes(*chunk);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;

        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        for (slot, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
}

/// Digests a file by streaming fixed-size reads, returning lowercase hex.
///
/// Exists for the same reason [`Sha256`] streams: the files this verifies (downloaded model
/// checkpoints) are hundreds of megabytes to gigabytes, and reading one into memory just to hash
/// it would double the peak footprint of a verification pass.
///
/// # Errors
///
/// Propagates the underlying [`std::io::Error`] from opening or reading the file.
pub fn hex_digest_file(path: &std::path::Path) -> std::io::Result<String> {
    use std::io::Read as _;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    // 1 MiB: large enough that syscall overhead is negligible, small enough to stay cache-polite.
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => hasher.update(&buffer[..read]),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(to_hex(&hasher.finish()))
}

/// Digests a byte slice, returning lowercase hex.
#[must_use]
pub fn hex_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    to_hex(&hasher.finish())
}

/// Digests an in-memory payload through RustCrypto's runtime-dispatched implementation.
#[must_use]
pub fn accelerated_digest(bytes: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    sha2::Sha256::digest(bytes).into()
}

/// Renders raw digest bytes as lowercase hex.
#[must_use]
pub fn to_hex(digest: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest {
        // Hand-rolled rather than `format!("{byte:02x}")` per byte: this runs over every section
        // digest and the formatting machinery dominates otherwise.
        const HEX: &[u8; 16] = b"0123456789abcdef";
        out.push(HEX[usize::from(byte >> 4)] as char);
        out.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS 180-4 / NIST CAVP known-answer vectors.
    ///
    /// A hash implementation that has not been checked against published vectors is an assumption,
    /// and this one gates artifact integrity — a wrong digest either rejects every good artifact or
    /// accepts every corrupted one.
    #[test]
    fn matches_the_published_nist_vectors() {
        assert_eq!(
            hex_digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex_digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex_digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            hex_digest(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            ),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
        // One million 'a' — exercises the length counter past a single block group.
        let mut hasher = Sha256::new();
        for _ in 0..1000 {
            hasher.update(&[b'a'; 1000]);
        }
        assert_eq!(
            to_hex(&hasher.finish()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    /// The streaming path must agree with the one-shot path at every chunk boundary.
    ///
    /// Section digests are fed in whatever chunks the caller has; a boundary bug would produce
    /// digests that depend on read size, which is the worst possible failure — intermittent.
    #[test]
    fn streaming_in_any_chunking_matches_one_shot() {
        let message: Vec<u8> = (0..1000_u32).map(|i| (i % 251) as u8).collect();
        let expected = hex_digest(&message);
        for chunk_size in [1_usize, 2, 7, 31, 63, 64, 65, 127, 128, 999, 1000] {
            let mut hasher = Sha256::new();
            for chunk in message.chunks(chunk_size) {
                hasher.update(chunk);
            }
            assert_eq!(
                to_hex(&hasher.finish()),
                expected,
                "digest changed with chunk size {chunk_size}"
            );
        }
    }

    #[test]
    fn accelerated_digest_matches_the_portable_streaming_implementation() {
        for length in [0_usize, 1, 63, 64, 65, 1_000, 1 << 20] {
            let message: Vec<u8> = (0..length).map(|index| (index % 251) as u8).collect();
            let mut portable = Sha256::new();
            portable.update(&message);
            assert_eq!(accelerated_digest(&message), portable.finish());
        }
    }

    /// The file path must agree with the in-memory path — it is the same algorithm behind a
    /// different read loop, and a divergence would verify downloads against the wrong contract.
    #[test]
    fn file_digest_matches_the_in_memory_digest() {
        let dir = std::env::temp_dir().join(format!("ftts-sha256-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("digest-input.bin");
        // Larger than one read buffer would be unnecessary; larger than one hash block matters.
        let message: Vec<u8> = (0..70_000_u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &message).expect("write temp file");
        assert_eq!(
            hex_digest_file(&path).expect("file digest"),
            hex_digest(&message)
        );
    }

    /// A single flipped bit must change the digest — the property section verification relies on.
    #[test]
    fn a_single_bit_flip_changes_the_digest() {
        let mut message = vec![0_u8; 256];
        let clean = hex_digest(&message);
        message[128] ^= 0x01;
        assert_ne!(hex_digest(&message), clean);
    }
}
