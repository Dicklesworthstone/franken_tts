//! Audited OS-interface island: read-only memory mapping of a checkpoint file.
//!
//! This exists so weights can be *addressed* without being *read*. A 1.7 GB checkpoint loaded with
//! `fs::read` costs 1.7 GB of resident anonymous memory before a single tensor is touched; mapped
//! read-only, the same file costs address space, and only the pages actually dereferenced — the
//! embedding rows a prompt names, the layers a frame walks — are ever faulted in. That difference
//! is the whole point of the `.fttsq` access-class design, and it cannot be expressed in safe Rust.
//!
//! Scope of the island: `mmap`, `munmap`, `fstat`. No kernels, no arithmetic, no parsing. Everything
//! above this file — the safetensors directory, the census, every accessor — is `forbid(unsafe_code)`
//! and operates on the `&[u8]` this hands out.
//!
//! # The truncation hazard, stated plainly
//!
//! A file that is truncated by another process while mapped will fault with `SIGBUS` on access to
//! the vanished pages. Rust cannot prevent this, and neither can any mmap wrapper — it is a property
//! of the syscall. We accept it for the same reason every mmap-based loader does, under a narrow
//! usage contract: the mapped file is a content-addressed model artifact that is written once and
//! read many times, never appended to or truncated in place while an engine holds it. Callers that
//! cannot honour that contract should read the file instead.

use std::fs::File;
use std::io;
use std::ops::Deref;
use std::os::fd::AsRawFd;
use std::path::Path;

/// A read-only, private memory mapping of a whole file.
///
/// Derefs to `&[u8]`, so it drops straight into anything expecting a borrowed buffer — notably the
/// safetensors index, which is a map of byte ranges over exactly such a slice.
#[derive(Debug)]
pub struct MappedFile {
    ptr: *const u8,
    len: usize,
}

// SAFETY: the mapping is `PROT_READ` + `MAP_PRIVATE`, so the pointer addresses immutable memory for
// the lifetime of the value and no interior mutability is reachable through it. `MappedFile` hands
// out only shared slices, and `munmap` happens once in `Drop` on the owning thread. Sharing the
// pointer across threads therefore exposes no data race.
unsafe impl Send for MappedFile {}
// SAFETY: as above — `&MappedFile` yields only `&[u8]` into a read-only mapping.
unsafe impl Sync for MappedFile {}

impl MappedFile {
    /// Map `path` read-only for its entire length.
    ///
    /// An empty file maps to an empty slice without calling `mmap`, because `mmap` rejects a zero
    /// length with `EINVAL` and an empty checkpoint is better rejected by the parser's own
    /// "too short for a header" path than by an opaque errno.
    ///
    /// # Errors
    ///
    /// Propagates the underlying `open`, `fstat` or `mmap` failure.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();

        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                len: 0,
            });
        }

        let len = usize::try_from(len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "checkpoint is larger than this platform's address space",
            )
        })?;

        // SAFETY: `file` is an open, readable descriptor that outlives this call. We request a
        // read-only private mapping of `len` bytes at an address of the kernel's choosing, with
        // offset 0 — `len` came from `fstat` on this same descriptor, so it is a valid extent.
        // `mmap` returns `MAP_FAILED` rather than a null pointer on error, which is checked below;
        // on success the returned range is valid for reads of `len` bytes until `munmap`.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        // The mapping is independent of the descriptor: closing `file` here (by dropping it at the
        // end of scope) does not unmap.
        Ok(Self {
            ptr: ptr.cast::<u8>().cast_const(),
            len,
        })
    }

    /// Advise the kernel that access will be sparse and random.
    ///
    /// Used for the cold text-embedding section, where prefill touches a few hundred scattered rows
    /// out of 151 936: read-ahead around each fault would pull in megabytes we never look at. This
    /// is advisory — a failure changes performance, never correctness, so the errno is discarded
    /// deliberately rather than surfaced as a load failure.
    pub fn advise_random(&self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: `self.ptr`/`self.len` describe our own live mapping, which is exactly the extent
        // `madvise` expects. `MADV_RANDOM` only adjusts kernel read-ahead policy; it cannot
        // invalidate the mapping or change the bytes we observe.
        unsafe {
            libc::madvise(self.ptr as *mut libc::c_void, self.len, libc::MADV_RANDOM);
        }
    }

    /// The mapped bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` addresses `len` initialized, readable bytes for as long as `self` lives
        // (the mapping is only released in `Drop`), and the returned borrow cannot outlive `self`.
        // The mapping is read-only and private, so no other handle can mutate it through us.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Mapped length in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the mapping is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Deref for MappedFile {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for MappedFile {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: `ptr`/`len` are exactly the values returned by our own successful `mmap`, and
        // `Drop` runs once, so the mapping is released exactly once. No slice handed out by
        // `as_slice` can still be alive here: each borrows `self`.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ftts-mmap-{tag}-{}.bin", std::process::id()));
        path
    }

    #[test]
    fn maps_file_contents() {
        let path = temp_path("contents");
        let payload: Vec<u8> = (0u8..=255).cycle().take(9000).collect();
        File::create(&path)
            .and_then(|mut f| f.write_all(&payload))
            .expect("write temp file");

        let mapped = MappedFile::open(&path).expect("maps");
        assert_eq!(mapped.len(), payload.len());
        assert!(!mapped.is_empty());
        assert_eq!(mapped.as_slice(), payload.as_slice());
        // Deref and AsRef expose the same bytes.
        assert_eq!(&mapped[..4], &payload[..4]);
        assert_eq!(AsRef::<[u8]>::as_ref(&mapped).len(), payload.len());
        mapped.advise_random();
        assert_eq!(mapped[8999], payload[8999]);

        drop(mapped);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_file_maps_to_empty_slice() {
        let path = temp_path("empty");
        File::create(&path).expect("create temp file");

        let mapped = MappedFile::open(&path).expect("maps");
        assert!(mapped.is_empty());
        assert_eq!(mapped.len(), 0);
        assert_eq!(mapped.as_slice(), &[] as &[u8]);
        mapped.advise_random();

        drop(mapped);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        let path = temp_path("definitely-absent-xyz");
        let _ = std::fs::remove_file(&path);
        assert!(MappedFile::open(&path).is_err());
    }
}
