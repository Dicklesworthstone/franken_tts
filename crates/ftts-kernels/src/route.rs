//! The one process-wide switch between the optimized product route and the f32 reference.
//!
//! The optimized route (int8 projections, the SLEEF SnakeBeta, the worker team) is the DEFAULT
//! everywhere — library and binary alike. Two things override it:
//!
//! - **Environment**: `FTTS_INT8=0` (or `off`/`false`) selects the reference route for a run;
//!   the finer-grained `FTTS_INT8_CODEC` / `FTTS_FAST_SNAKE` variables tune individual pieces.
//! - **[`pin_reference`]**: a programmatic, first-call-wins pin used by the conformance
//!   harness, so every oracle-parity test measures the reference numerics no matter what the
//!   surrounding environment says. A parity suite that silently tested the optimized route
//!   would be comparing the wrong thing against the oracle.
//!
//! The pin is sticky for the process (OnceLock): parity test binaries pin once, product
//! processes never call it.

use std::sync::OnceLock;

static REFERENCE_PIN: OnceLock<()> = OnceLock::new();

/// Pins this process to the f32 reference route, regardless of environment defaults.
///
/// First call wins and the pin never lifts; call it before any synthesis or decode work so
/// no armed route has cached its decision yet.
pub fn pin_reference() {
    let _ = REFERENCE_PIN.set(());
}

/// Whether [`pin_reference`] has been called.
#[must_use]
pub fn reference_pinned() -> bool {
    REFERENCE_PIN.get().is_some()
}

/// The shared default rule for optimized-route switches.
///
/// Returns `false` (reference) when the process is pinned or `variable` is set to a disabling
/// value; returns `true` otherwise — including when the variable is unset, which is what makes
/// the optimized route the default everywhere.
#[must_use]
pub fn optimized_default(variable: &str) -> bool {
    if reference_pinned() {
        return false;
    }
    !matches!(
        std::env::var(variable).as_deref(),
        Ok("0" | "off" | "false")
    )
}
