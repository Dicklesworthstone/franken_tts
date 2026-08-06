//! Audited OS-interface island: read-only memory mapping of a checkpoint file.
//!
//! This exists so weights can be *addressed* without being *read*. A 1.7 GB checkpoint loaded with
//! `fs::read` costs 1.7 GB of resident anonymous memory before a single tensor is touched; mapped
//! read-only, the same file costs address space, and only the pages actually dereferenced — the
//! embedding rows a prompt names, the layers a frame walks — are ever faulted in. That difference
//! is the whole point of the `.fttsq` access-class design, and it cannot be expressed in safe Rust.
//!
//! Scope of the island: `mmap`, `munmap`, `madvise`, `mincore`, and page-size discovery. No kernels,
//! no arithmetic, no parsing. Everything above this file — the safetensors directory, the census,
//! every accessor — is `forbid(unsafe_code)` and operates on the `&[u8]` this hands out.
//!
//! # The truncation hazard, stated plainly
//!
//! A file that is truncated by another process while mapped will fault with `SIGBUS` on access to
//! the vanished pages. Rust cannot prevent this, and neither can any mmap wrapper — it is a property
//! of the syscall. We accept it for the same reason every mmap-based loader does, under a narrow
//! usage contract: the mapped file is a content-addressed model artifact that is written once and
//! read many times, never appended to or truncated in place while an engine holds it. Callers that
//! cannot honour that contract should read the file instead.

use std::io;
use std::ops::Deref;
use std::path::Path;

#[cfg(all(feature = "native-mmap", unix))]
use std::fs::File;
#[cfg(all(feature = "native-mmap", unix))]
use std::os::fd::AsRawFd;

/// A kernel page-cache hint for a byte range within a mapped artifact.
///
/// This deliberately represents only the two policy actions the `.fttsq` access classes need.
/// Adding another OS-specific hint requires giving it a model-level meaning first; otherwise an
/// advisory syscall becomes an unreviewable collection of performance folklore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAdvice {
    /// The range is recurrent and should be faulted in before steady-state decoding.
    WillNeed,
    /// The range is sparse and row-granular, so read-ahead is counterproductive.
    Random,
}

/// What became of one [`MemoryAdvice`] request.
///
/// Advice cannot change artifact bytes or correctness. Unsupported platforms retain the safe
/// owned-byte fallback and report that they did not make an OS request instead of pretending a
/// Windows `PrefetchVirtualMemory` policy was implemented and tested when it was not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAdviceOutcome {
    /// The native `madvise` request succeeded.
    Applied,
    /// The mapping is empty, so there is no range for the kernel to advise.
    SkippedEmpty,
    /// This build or platform deliberately has no native advisory implementation.
    Unsupported,
}

/// An observation of which pages in one mapped byte range are currently resident.
///
/// This is intentionally an observation rather than an eviction or a residency promise: the OS
/// owns page-cache policy, so callers use it to record an access-class measurement for OQ-18, not
/// to turn a performance hint into a correctness condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryResidency {
    /// `mincore` counted resident pages in the requested range.
    Measured {
        /// Pages the OS currently reports resident.
        resident_pages: usize,
        /// Pages spanned by the requested byte range.
        total_pages: usize,
    },
    /// This build deliberately has no native residency-query implementation.
    Unsupported,
}

/// A read-only, private memory mapping of a whole file.
///
/// Derefs to `&[u8]`, so it drops straight into anything expecting a borrowed buffer — notably the
/// safetensors index, which is a map of byte ranges over exactly such a slice.
#[derive(Debug)]
pub struct MappedFile {
    #[cfg(all(feature = "native-mmap", unix))]
    ptr: *const u8,
    #[cfg(all(feature = "native-mmap", unix))]
    len: usize,
    #[cfg(not(all(feature = "native-mmap", unix)))]
    bytes: Vec<u8>,
}

// SAFETY: the mapping is `PROT_READ` + `MAP_PRIVATE`, so the pointer addresses immutable memory for
// the lifetime of the value and no interior mutability is reachable through it. `MappedFile` hands
// out only shared slices, and `munmap` happens once in `Drop` on the owning thread. Sharing the
// pointer across threads therefore exposes no data race.
#[cfg(all(feature = "native-mmap", unix))]
unsafe impl Send for MappedFile {}
// SAFETY: as above — `&MappedFile` yields only `&[u8]` into a read-only mapping.
#[cfg(all(feature = "native-mmap", unix))]
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
        #[cfg(all(feature = "native-mmap", unix))]
        {
            Self::open_native(path.as_ref())
        }

        #[cfg(not(all(feature = "native-mmap", unix)))]
        {
            // This is the bit-identical scalar fallback for targets where the audited POSIX
            // implementation is unavailable. It is deliberately safe and explicit about its
            // footprint trade-off rather than relying on an untested platform FFI binding.
            Ok(Self {
                bytes: std::fs::read(path)?,
            })
        }
    }

    #[cfg(all(feature = "native-mmap", unix))]
    fn open_native(path: &Path) -> io::Result<Self> {
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

    /// Apply `advice` to one validated byte range.
    ///
    /// The caller supplies artifact-relative offsets. This method bounds-checks them before the
    /// audited native call and aligns only the address down to the host page boundary, as required
    /// by `madvise`; the supplied byte length still limits the advised range.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` for a range outside this mapping, or the native `madvise` error.
    pub fn advise(
        &self,
        offset: u64,
        length: u64,
        advice: MemoryAdvice,
    ) -> io::Result<MemoryAdviceOutcome> {
        let (offset, length) = self.validated_range(offset, length)?;
        if length == 0 {
            return Ok(MemoryAdviceOutcome::SkippedEmpty);
        }

        #[cfg(all(feature = "native-mmap", unix))]
        {
            self.advise_native(offset, length, advice)
        }

        #[cfg(not(all(feature = "native-mmap", unix)))]
        {
            let _ = (offset, advice);
            Ok(MemoryAdviceOutcome::Unsupported)
        }
    }

    /// Counts resident pages in one validated byte range when the platform exposes `mincore`.
    ///
    /// This does not fault pages in or evict them. It is an OQ-18 measurement hook used to make the
    /// cold-embedding policy observable; unsupported targets return [`MemoryResidency::Unsupported`]
    /// instead of claiming equivalent platform behavior without an audited implementation.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` for a range outside this mapping, or the native `mincore` error.
    pub fn resident_pages(&self, offset: u64, length: u64) -> io::Result<MemoryResidency> {
        let (offset, length) = self.validated_range(offset, length)?;
        #[cfg(all(feature = "native-mmap", unix))]
        {
            if length == 0 {
                return Ok(MemoryResidency::Measured {
                    resident_pages: 0,
                    total_pages: 0,
                });
            }
            self.resident_pages_native(offset, length)
        }

        #[cfg(not(all(feature = "native-mmap", unix)))]
        {
            let _ = offset;
            Ok(MemoryResidency::Unsupported)
        }
    }

    fn validated_range(&self, offset: u64, length: u64) -> io::Result<(usize, usize)> {
        let offset = usize::try_from(offset).map_err(|_| invalid_range_error())?;
        let length = usize::try_from(length).map_err(|_| invalid_range_error())?;
        let end = offset.checked_add(length).ok_or_else(invalid_range_error)?;
        if end > self.len() {
            return Err(invalid_range_error());
        }
        Ok((offset, length))
    }

    #[cfg(all(feature = "native-mmap", unix))]
    fn advise_native(
        &self,
        offset: usize,
        length: usize,
        advice: MemoryAdvice,
    ) -> io::Result<MemoryAdviceOutcome> {
        let page_size = page_size()?;
        let aligned_offset = offset - (offset % page_size);
        let advised_length = offset
            .checked_add(length)
            .and_then(|end| end.checked_sub(aligned_offset))
            .ok_or_else(invalid_range_error)?;
        let native_advice = match advice {
            MemoryAdvice::WillNeed => libc::MADV_WILLNEED,
            MemoryAdvice::Random => libc::MADV_RANDOM,
        };

        // SAFETY: `self.ptr`/`self.len` describe our own live mapping, which is exactly the extent
        // `madvise` expects. `aligned_offset` is rounded down to the actual runtime page size and
        // `advised_length` ends no later than `self.len`, so the advised range lies inside the
        // mapping. Both available advice values only influence page-cache behavior; neither can
        // mutate or invalidate the bytes exposed by this read-only private mapping.
        let result = unsafe {
            libc::madvise(
                self.ptr
                    .add(aligned_offset)
                    .cast_mut()
                    .cast::<libc::c_void>(),
                advised_length,
                native_advice,
            )
        };
        if result == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(MemoryAdviceOutcome::Applied)
    }

    #[cfg(all(feature = "native-mmap", unix))]
    fn resident_pages_native(&self, offset: usize, length: usize) -> io::Result<MemoryResidency> {
        let page_size = page_size()?;
        let aligned_offset = offset - (offset % page_size);
        let observed_length = offset
            .checked_add(length)
            .and_then(|end| end.checked_sub(aligned_offset))
            .ok_or_else(invalid_range_error)?;
        let total_pages = observed_length
            .checked_add(page_size - 1)
            .and_then(|bytes| bytes.checked_div(page_size))
            .ok_or_else(invalid_range_error)?;
        let mut residency = vec![0_u8; total_pages];

        // SAFETY: `self.ptr`/`self.len` describe our own live read-only mapping. `aligned_offset`
        // is page-aligned and inside that mapping, `observed_length` ends no later than `self.len`,
        // and `residency` owns exactly one output byte for each page the kernel may report. `mincore`
        // observes page-cache state only; it cannot mutate or invalidate the mapping.
        let result = unsafe {
            libc::mincore(
                self.ptr
                    .add(aligned_offset)
                    .cast_mut()
                    .cast::<libc::c_void>(),
                observed_length,
                residency.as_mut_ptr().cast(),
            )
        };
        if result == -1 {
            return Err(io::Error::last_os_error());
        }

        Ok(MemoryResidency::Measured {
            resident_pages: residency.iter().filter(|state| **state & 1 != 0).count(),
            total_pages,
        })
    }

    /// Advise the kernel that the whole file is accessed sparsely and randomly.
    ///
    /// Retained for the safetensors loader, whose upstream file has no `.fttsq` section directory.
    /// Its best-effort semantics match the historical API: advice failures do not reject a valid
    /// checkpoint because they affect performance only.
    pub fn advise_random(&self) {
        let _ = self.advise(0, self.len() as u64, MemoryAdvice::Random);
    }

    /// The mapped bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        #[cfg(all(feature = "native-mmap", unix))]
        {
            if self.len == 0 {
                return &[];
            }
            // SAFETY: `ptr` addresses `len` initialized, readable bytes for as long as `self` lives
            // (the mapping is only released in `Drop`), and the returned borrow cannot outlive `self`.
            // The mapping is read-only and private, so no other handle can mutate it through us.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }

        #[cfg(not(all(feature = "native-mmap", unix)))]
        {
            &self.bytes
        }
    }

    /// Mapped length in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        #[cfg(all(feature = "native-mmap", unix))]
        {
            self.len
        }

        #[cfg(not(all(feature = "native-mmap", unix)))]
        {
            self.bytes.len()
        }
    }

    /// Whether the mapping is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
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

#[cfg(all(feature = "native-mmap", unix))]
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

fn invalid_range_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "memory-advice range lies outside the mapped artifact",
    )
}

#[cfg(all(feature = "native-mmap", unix))]
fn page_size() -> io::Result<usize> {
    // SAFETY: `sysconf(_SC_PAGESIZE)` has no pointer arguments and does not mutate process state;
    // it returns the runtime page size needed solely to satisfy `madvise`'s address-alignment rule.
    let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if raw <= 0 {
        return Err(io::Error::last_os_error());
    }
    usize::try_from(raw).map_err(|_| io::Error::other("page size does not fit usize"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
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
        assert!(matches!(
            mapped
                .advise(1, 32, MemoryAdvice::Random)
                .expect("in-range advice"),
            MemoryAdviceOutcome::Applied | MemoryAdviceOutcome::Unsupported
        ));
        match mapped
            .resident_pages(1, 32)
            .expect("in-range residency observation")
        {
            MemoryResidency::Measured {
                resident_pages,
                total_pages,
            } => assert!(resident_pages <= total_pages),
            MemoryResidency::Unsupported => {}
        }
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

    #[test]
    fn advice_refuses_a_range_outside_the_mapping() {
        let path = temp_path("range");
        std::fs::write(&path, [1_u8; 32]).expect("write temp file");
        let mapped = MappedFile::open(&path).expect("maps");

        let error = mapped
            .advise(31, 2, MemoryAdvice::WillNeed)
            .expect_err("range crossing EOF must be refused");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        drop(mapped);
        let _ = std::fs::remove_file(&path);
    }
}
