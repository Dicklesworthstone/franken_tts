//! Audited startup-time environment defaulting.
//!
//! `std::env::set_var` is unsafe in edition 2024 because mutating the environment while other
//! threads read it is undefined behavior on some platforms. This island exists so the CLI (a
//! `forbid(unsafe_code)` crate) can install product-default switches at the top of `main`,
//! before anything spawns a thread. It is the same class of tiny OS-interface island as
//! `mmap.rs`.

/// Sets `key` to `value` unless the user already set it.
///
/// # Contract
///
/// Call only during single-threaded process startup — in practice, as the first statements of
/// `cli_main` before any engine, team, or runtime construction. Calling this after threads
/// exist would be the exact hazard `set_var`'s unsafety describes.
pub fn set_default_if_unset(key: &str, value: &str) {
    if std::env::var_os(key).is_some() {
        return;
    }
    // SAFETY: per this function's contract the process is still single-threaded, so no
    // concurrent reader of the environment can exist.
    unsafe { std::env::set_var(key, value) };
}
