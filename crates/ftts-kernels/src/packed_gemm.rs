//! Register-tiled, panel-packed f32 GEMM — the BLAS-shaped dense route for hosts with no BLAS.
//!
//! # Why this exists
//!
//! On macOS the f32 dense path issues the reference's own Accelerate SGEMM and is exact against
//! the oracle. Off that platform — Linux, and above all **wasm, where no BLAS exists at all** —
//! the same call degrades to a dot-product loop: for every output element, walk `k` and reduce.
//! That formulation re-reads the entire activation row once per output column and gets no reuse
//! out of the weights, which is why the browser codec measured 89.1 s of a 97.3 s frame (92%).
//!
//! This is the standard answer, and it is what every serious GEMM does: hold an `MR x NR` tile of
//! the output in registers, stream one packed `k`-panel of the weights past it, and pay for each
//! loaded weight `MR` times instead of once.
//!
//! # Why it is plain scalar Rust with no intrinsics
//!
//! Doctrine #3: hand-rolled wide SIMD over scalar inner loops measured ~5x SLOWER than LLVM
//! autovectorization in the sibling repos. The inner loop below is a fixed-size `[[f32; NR]; MR]`
//! accumulator updated by a broadcast scalar — precisely the shape LLVM turns into `NR/4` v128
//! multiply-adds per row with no help. The structure is the lever; the instruction selection is
//! the compiler's job.
//!
//! Ported from `franken_numpy/crates/fnp-linalg/src/lib.rs` (`packed_gemm_serial_tiled`), f64 to
//! f32, with the packing adapted to this project's `[n, k]` weight layout.
//!
//! # Exactness
//!
//! **Bit-identical to the scalar reference**, and that is a design constraint rather than a happy
//! accident. Each output element accumulates over ascending `k` into its own slot, one `f32` add
//! at a time — the same values in the same order as [`crate::f32ref`]'s scalar dot product. No
//! partial-sum splitting, no reassociation, no fused multiply-add. Blocking and packing change
//! only WHICH element is computed WHEN, never how any single element is summed.
//!
//! That matters here more than speed: the current wasm path uses eight independent partial chains
//! (a different, non-reference reduction order), so adopting this kernel moves the codec CLOSER to
//! the reference while making it faster. `packed_matches_scalar_bit_for_bit` pins the claim.

/// Rows of the portable output tile held in registers.
///
/// Four rows of eight `f32` is 32 accumulators — eight v128 registers on wasm, which fits the
/// 16-register file with room for the operands. Wider tiles spill. Row remainders shorter than
/// `MR` run one row at a time (`accumulate_tile::<1, _>`); there is no intermediate 2-row tile.
const MR: usize = 4;

/// Columns of the portable output tile held in registers: two full v128 lanes of `f32`.
const NR: usize = 8;

/// Column alignment for team stripes: a multiple of every tile width any instantiation uses, so
/// every partition stays on the packed path whichever ISA level is dispatched.
pub(crate) const STRIPE_COLUMNS: usize = 32;

/// Target bytes for one packed weight panel, sized to sit in L2 alongside the activation rows.
const PANEL_BYTES: usize = 256 * 1024;

/// Which compiled instantiation of the packed kernel runs.
///
/// Every level runs the same packing and blocking with a different register tile (`Tile`):
/// the portable tile is plain Rust LLVM vectorizes to the build's baseline width, the x86 tiles
/// are explicit micro-kernels. In all of them each output element accumulates `x * w` over
/// ascending `k` from `0.0`, one separately rounded IEEE multiply and one IEEE add per step (no
/// fused multiply-add anywhere), and is then added once to its bias. So all levels are
/// **bit-identical** to each other and to the scalar reference — this is a speed dispatch, never
/// a numerics one, and needs no kill-switch or DISC (`every_isa_level_is_bit_identical_to_the_
/// scalar_reference` pins it).
///
/// Why it exists: the x86-64 release binaries target the baseline ISA (SSE2), so without runtime
/// dispatch the codec — whose dense ops all reach this kernel off macOS — ran four lanes wide on
/// CPUs with sixteen-lane registers, and the codec became the x86 bottleneck once the int8
/// islands landed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32GemmLevel {
    /// The portable instantiation (`MR`×`NR` = 4×8), whatever the build's baseline target is.
    Portable,
    /// x86-64 AVX2: explicit 6×16 micro-kernel, twelve 256-bit accumulators.
    X86Avx2,
    /// x86-64 AVX-512F: explicit 8×32 micro-kernel, sixteen 512-bit accumulators.
    X86Avx512,
}

impl F32GemmLevel {
    /// Stable machine-readable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            Self::X86Avx2 => "x86-avx2",
            Self::X86Avx512 => "x86-avx512",
        }
    }

    /// Every level this build can execute on the running CPU, portable first.
    #[must_use]
    pub fn available() -> Vec<Self> {
        let mut levels = vec![Self::Portable];
        #[cfg(all(target_arch = "x86_64", feature = "x86-f32"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                levels.push(Self::X86Avx2);
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                levels.push(Self::X86Avx512);
            }
        }
        levels
    }

    /// The level dispatched by default: the widest available, unless `FTTS_F32_GEMM` names
    /// another available level (an A/B override, read once; unknown or unavailable names fall
    /// back to the widest). Being bit-identical, the choice costs speed only.
    #[must_use]
    pub fn dispatch() -> Self {
        static LEVEL: std::sync::OnceLock<F32GemmLevel> = std::sync::OnceLock::new();
        *LEVEL.get_or_init(|| {
            let levels = Self::available();
            let widest = *levels.last().expect("portable is always available");
            std::env::var("FTTS_F32_GEMM")
                .ok()
                .and_then(|name| levels.into_iter().find(|level| level.as_str() == name))
                .unwrap_or(widest)
        })
    }
}

/// `out[m, n] = x[m, k] @ weight[n, k]^T + bias[n]`.
///
/// `weight` is the checkpoint's native `[out_channels, in_channels]` layout — each output row
/// contiguous — so no transpose is ever materialized, matching the project's one GEMM contract.
///
/// # Panics
///
/// If the slice lengths disagree with `m`, `k`, `n`.
pub fn linear_packed(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), m * k, "x must be [m, k]");
    assert_eq!(weight.len(), n * k, "weight must be [n, k]");
    assert_eq!(out.len(), m * n, "out must be [m, n]");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n, "bias must be [n]");
    }
    // SAFETY: `out` is a `&mut [f32]` of exactly `m * n`, and the full column range is requested,
    // so every write below lands inside it. The borrow checker guarantees no other alias.
    unsafe {
        linear_packed_range(x, weight, bias, m, k, n, 0, n, out.as_mut_ptr());
    }
}

/// [`linear_packed`] pinned to one [`F32GemmLevel`] — the A/B seam for benches and tests.
///
/// # Panics
///
/// If the slice lengths disagree with `m`, `k`, `n`, or `level` is not in
/// [`F32GemmLevel::available`].
#[allow(clippy::too_many_arguments)]
pub fn linear_packed_at(
    level: F32GemmLevel,
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), m * k, "x must be [m, k]");
    assert_eq!(weight.len(), n * k, "weight must be [n, k]");
    assert_eq!(out.len(), m * n, "out must be [m, n]");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n, "bias must be [n]");
    }
    assert!(
        F32GemmLevel::available().contains(&level),
        "{} is not executable on this CPU",
        level.as_str()
    );
    // SAFETY: full column range of an exclusively borrowed `m * n` buffer; `level` was just
    // confirmed available.
    unsafe {
        linear_packed_range_at(level, x, weight, bias, m, k, n, 0, n, out.as_mut_ptr());
    }
}

/// Computes only output columns `col_start..col_end`, writing into a `[m, n]` buffer.
///
/// This is the shape the [`crate::team`] needs: each worker owns a disjoint column stripe and
/// writes it in place, so no partition ever touches another's elements and no reduction is split
/// across partitions. The result is bit-identical to the serial whole-matrix call, which is why
/// threading this changes speed only — pinned by `column_partitions_are_bit_identical_to_the_whole`.
///
/// # Safety
///
/// `out` must be valid for writes of `m * n` floats, and no other reference may alias the columns
/// `col_start..col_end` for the duration of the call.
// The argument count is the GEMM contract itself — operands, the (m, k, n) shape, the column
// stripe, and the destination. Bundling them into a struct would add a layer between the caller
// and the hot loop without removing a single value, so the lint is allowed here deliberately,
// matching `f32ref::gqa_attention_head_range_into`.
#[allow(clippy::too_many_arguments)]
// SAFETY: discharged by both callers. `linear_packed` passes the pointer of a `&mut [f32]` it
// holds exclusively, with the full column range. The team passes one worker's disjoint stripe of a
// buffer the dispatcher owns and blocks on until every partition reports done, so the allocation
// outlives all writes and no two stripes address the same element.
pub(crate) unsafe fn linear_packed_range(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    col_start: usize,
    col_end: usize,
    out: *mut f32,
) {
    // SAFETY: forwarded verbatim from this function's own contract; `dispatch()` only ever names
    // a level `available()` confirmed on this CPU.
    unsafe {
        linear_packed_range_at(
            F32GemmLevel::dispatch(),
            x,
            weight,
            bias,
            m,
            k,
            n,
            col_start,
            col_end,
            out,
        );
    }
}

/// [`linear_packed_range`] at an explicit level — the dispatch seam the tests drive directly.
///
/// # Safety
///
/// As [`linear_packed_range`], plus `level` must be one of [`F32GemmLevel::available`].
#[allow(clippy::too_many_arguments)]
// SAFETY: callers pass `dispatch()` (always available) or iterate `available()` in tests.
unsafe fn linear_packed_range_at(
    level: F32GemmLevel,
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    col_start: usize,
    col_end: usize,
    out: *mut f32,
) {
    match level {
        #[cfg(all(target_arch = "x86_64", feature = "x86-f32"))]
        F32GemmLevel::X86Avx512 => {
            assert!(std::arch::is_x86_feature_detected!("avx512f"));
            // SAFETY: AVX-512F confirmed on this CPU just above (the `Avx512Tile` contract); the
            // buffer contract is forwarded verbatim.
            unsafe {
                range_impl::<x86::Avx512Tile>(x, weight, bias, m, k, n, col_start, col_end, out);
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "x86-f32"))]
        F32GemmLevel::X86Avx2 => {
            assert!(std::arch::is_x86_feature_detected!("avx2"));
            // SAFETY: AVX2 confirmed on this CPU just above (the `Avx2Tile` contract); the buffer
            // contract is forwarded verbatim.
            unsafe {
                range_impl::<x86::Avx2Tile>(x, weight, bias, m, k, n, col_start, col_end, out);
            }
        }
        // SAFETY: the buffer contract is forwarded verbatim; the portable tile needs nothing else.
        _ => unsafe {
            range_impl::<PortableTile>(x, weight, bias, m, k, n, col_start, col_end, out);
        },
    }
}

/// One instantiation's register tile: its shape and the micro-kernel that accumulates it.
///
/// The arithmetic contract every implementation honors, which is what makes all of them
/// bit-identical: each output element's accumulator starts at `0.0`, takes `x * w` (one IEEE
/// multiply, rounded) and then one IEEE add per `k` step in ascending order, and is finally added
/// once to the value already in `out` (the bias seed).
trait Tile {
    /// Rows of a full tile.
    const ROWS: usize;
    /// Columns of a tile — the packed panel width.
    const COLS: usize;

    /// Accumulates rows `i0..i0 + rows` (`1 <= rows <= ROWS`) by columns `j0..j0 + COLS` of the
    /// output from a `k * COLS` packed panel.
    ///
    /// # Safety
    ///
    /// `out` must be valid for writes over those rows and columns of an `[m, n]` matrix, `x` must
    /// hold those rows (`x.len() >= (i0 + rows) * k`), and the implementation's CPU-feature
    /// precondition must hold.
    // The GEMM tile contract itself (operands, tile origin and height, shape); see
    // `linear_packed_range` for why it is not bundled into a struct.
    #[allow(clippy::too_many_arguments)]
    unsafe fn accumulate(
        x: &[f32],
        panel: &[f32],
        out: *mut f32,
        i0: usize,
        rows: usize,
        j0: usize,
        k: usize,
        n: usize,
    );
}

/// The portable tile: plain Rust that LLVM vectorizes to the build's baseline width.
struct PortableTile;

impl Tile for PortableTile {
    const ROWS: usize = MR;
    const COLS: usize = NR;

    unsafe fn accumulate(
        x: &[f32],
        panel: &[f32],
        out: *mut f32,
        i0: usize,
        rows: usize,
        j0: usize,
        k: usize,
        n: usize,
    ) {
        // SAFETY (both arms): the trait contract covers exactly these rows and columns.
        if rows == MR {
            unsafe { accumulate_tile::<MR, NR>(x, panel, out, i0, j0, k, n) };
        } else {
            for row in i0..i0 + rows {
                unsafe { accumulate_tile::<1, NR>(x, panel, out, row, j0, k, n) };
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "x86-f32"))]
mod x86 {
    //! The audited x86-64 micro-kernels of the packed GEMM.
    //!
    //! Why intrinsics here, against doctrine #3's default: the doctrine bans hand-rolled SIMD over
    //! *glue*, where LLVM's autovectorizer wins; this is the GEMM micro-kernel, the matmul-class
    //! exception the doctrine itself names. And the autovectorizer measurably fails at it: the
    //! same scalar tile compiled under `avx512f` was SLP-scalarized into `vmulss`/`vinsertps`
    //! chains at 2.6 GFLOP/s (vs 23 portable), and the best autovectorized AVX2 shape reached
    //! ~33 GFLOP/s, where these explicit kernels reach ~64 (AVX2 6×16) and ~75 (AVX-512 8×32) on
    //! the measuring Emerald Rapids host (probe, single thread, PROVISIONAL_LOCAL_WIN).
    //!
    //! Bit-identity: every step is `_mm*_add_ps(acc, _mm*_mul_ps(broadcast(x), w))` — the same
    //! two separately rounded IEEE operations the scalar `*slot += value * weight` performs, per
    //! lane, in the same ascending-`k` order — then one `_mm*_add_ps` into the bias-seeded output.
    //! No fused multiply-add is ever emitted (only `avx2`/`avx512f` are enabled, never `fma`, and
    //! the intrinsics name separate operations), so results equal the portable tile to the bit.
    //!
    //! Loads: panel offsets are bounded by `depth < k` against the `k * COLS` panel, activation
    //! offsets by the caller's row contract; every load and store is unaligned-tolerant.

    use core::arch::x86_64::{
        __m256, __m512, _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_storeu_ps, _mm512_add_ps, _mm512_loadu_ps, _mm512_mul_ps,
        _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };

    /// AVX-512F: 8×32 tile — sixteen zmm accumulators, two per row.
    pub(super) struct Avx512Tile;

    impl super::Tile for Avx512Tile {
        const ROWS: usize = 8;
        const COLS: usize = 32;

        unsafe fn accumulate(
            x: &[f32],
            panel: &[f32],
            out: *mut f32,
            i0: usize,
            rows: usize,
            j0: usize,
            k: usize,
            n: usize,
        ) {
            // SAFETY (every arm): the trait contract — rows/columns in bounds, AVX-512F present.
            unsafe {
                match rows {
                    8 => tile_avx512::<8>(x, panel, out, i0, j0, k, n),
                    7 => tile_avx512::<7>(x, panel, out, i0, j0, k, n),
                    6 => tile_avx512::<6>(x, panel, out, i0, j0, k, n),
                    5 => tile_avx512::<5>(x, panel, out, i0, j0, k, n),
                    4 => tile_avx512::<4>(x, panel, out, i0, j0, k, n),
                    3 => tile_avx512::<3>(x, panel, out, i0, j0, k, n),
                    2 => tile_avx512::<2>(x, panel, out, i0, j0, k, n),
                    _ => tile_avx512::<1>(x, panel, out, i0, j0, k, n),
                }
            }
        }
    }

    /// AVX2: 6×16 tile — twelve ymm accumulators, two per row (the BLIS Haswell shape).
    pub(super) struct Avx2Tile;

    impl super::Tile for Avx2Tile {
        const ROWS: usize = 6;
        const COLS: usize = 16;

        unsafe fn accumulate(
            x: &[f32],
            panel: &[f32],
            out: *mut f32,
            i0: usize,
            rows: usize,
            j0: usize,
            k: usize,
            n: usize,
        ) {
            // SAFETY (every arm): the trait contract — rows/columns in bounds, AVX2 present.
            unsafe {
                match rows {
                    6 => tile_avx2::<6>(x, panel, out, i0, j0, k, n),
                    5 => tile_avx2::<5>(x, panel, out, i0, j0, k, n),
                    4 => tile_avx2::<4>(x, panel, out, i0, j0, k, n),
                    3 => tile_avx2::<3>(x, panel, out, i0, j0, k, n),
                    2 => tile_avx2::<2>(x, panel, out, i0, j0, k, n),
                    _ => tile_avx2::<1>(x, panel, out, i0, j0, k, n),
                }
            }
        }
    }

    /// # Safety
    ///
    /// AVX-512F must be present; `panel.len() >= k * 32`; `x` holds rows `i0..i0 + ROWS` of
    /// length `k`; `out` is writable over those rows and columns `j0..j0 + 32` of `[m, n]`.
    #[target_feature(enable = "avx512f")]
    unsafe fn tile_avx512<const ROWS: usize>(
        x: &[f32],
        panel: &[f32],
        out: *mut f32,
        i0: usize,
        j0: usize,
        k: usize,
        n: usize,
    ) {
        debug_assert!(panel.len() >= k * 32 && x.len() >= (i0 + ROWS) * k);
        let rows: [*const f32; ROWS] = core::array::from_fn(|row| {
            // SAFETY: row `i0 + row` starts inside `x` by the contract.
            unsafe { x.as_ptr().add((i0 + row) * k) }
        });
        let weights = panel.as_ptr();
        let mut low: [__m512; ROWS] = [_mm512_setzero_ps(); ROWS];
        let mut high: [__m512; ROWS] = [_mm512_setzero_ps(); ROWS];
        for depth in 0..k {
            // SAFETY: `depth < k` keeps both 16-lane loads inside the `k * 32` panel.
            let (w_low, w_high) = unsafe {
                (
                    _mm512_loadu_ps(weights.add(depth * 32)),
                    _mm512_loadu_ps(weights.add(depth * 32 + 16)),
                )
            };
            for row in 0..ROWS {
                // SAFETY: `depth < k` stays inside row `row` of `x`.
                let value = _mm512_set1_ps(unsafe { *rows[row].add(depth) });
                low[row] = _mm512_add_ps(low[row], _mm512_mul_ps(value, w_low));
                high[row] = _mm512_add_ps(high[row], _mm512_mul_ps(value, w_high));
            }
        }
        for row in 0..ROWS {
            // SAFETY: the contract makes columns `j0..j0 + 32` of row `i0 + row` writable.
            unsafe {
                let base = out.add((i0 + row) * n + j0);
                _mm512_storeu_ps(base, _mm512_add_ps(_mm512_loadu_ps(base), low[row]));
                let base = base.add(16);
                _mm512_storeu_ps(base, _mm512_add_ps(_mm512_loadu_ps(base), high[row]));
            }
        }
    }

    /// # Safety
    ///
    /// AVX2 must be present; `panel.len() >= k * 16`; `x` holds rows `i0..i0 + ROWS` of length
    /// `k`; `out` is writable over those rows and columns `j0..j0 + 16` of `[m, n]`.
    #[target_feature(enable = "avx2")]
    unsafe fn tile_avx2<const ROWS: usize>(
        x: &[f32],
        panel: &[f32],
        out: *mut f32,
        i0: usize,
        j0: usize,
        k: usize,
        n: usize,
    ) {
        debug_assert!(panel.len() >= k * 16 && x.len() >= (i0 + ROWS) * k);
        let rows: [*const f32; ROWS] = core::array::from_fn(|row| {
            // SAFETY: row `i0 + row` starts inside `x` by the contract.
            unsafe { x.as_ptr().add((i0 + row) * k) }
        });
        let weights = panel.as_ptr();
        let mut low: [__m256; ROWS] = [_mm256_setzero_ps(); ROWS];
        let mut high: [__m256; ROWS] = [_mm256_setzero_ps(); ROWS];
        for depth in 0..k {
            // SAFETY: `depth < k` keeps both 8-lane loads inside the `k * 16` panel.
            let (w_low, w_high) = unsafe {
                (
                    _mm256_loadu_ps(weights.add(depth * 16)),
                    _mm256_loadu_ps(weights.add(depth * 16 + 8)),
                )
            };
            for row in 0..ROWS {
                // SAFETY: `depth < k` stays inside row `row` of `x`.
                let value = _mm256_set1_ps(unsafe { *rows[row].add(depth) });
                low[row] = _mm256_add_ps(low[row], _mm256_mul_ps(value, w_low));
                high[row] = _mm256_add_ps(high[row], _mm256_mul_ps(value, w_high));
            }
        }
        for row in 0..ROWS {
            // SAFETY: the contract makes columns `j0..j0 + 16` of row `i0 + row` writable.
            unsafe {
                let base = out.add((i0 + row) * n + j0);
                _mm256_storeu_ps(base, _mm256_add_ps(_mm256_loadu_ps(base), low[row]));
                let base = base.add(8);
                _mm256_storeu_ps(base, _mm256_add_ps(_mm256_loadu_ps(base), high[row]));
            }
        }
    }
}

thread_local! {
    /// Per-thread packed-panel scratch, shared by every instantiation.
    ///
    /// Thread-local instead of a per-dispatch `vec!`: the kernel runs once per stripe per
    /// dispatch on the steady-state decode path, and the doctrine pins "no allocator activity in
    /// steady-state decode" as load-bearing. Each team worker (and the dispatcher) owns its
    /// thread's buffer, so there is no sharing to reason about; it only ever grows, to the largest
    /// `k * COLS` this thread has seen.
    static PANEL_SCRATCH: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The packed kernel body, generic over the register tile.
///
/// # Safety
///
/// The [`linear_packed_range`] buffer contract, plus `T`'s CPU-feature precondition.
#[allow(clippy::too_many_arguments)]
unsafe fn range_impl<T: Tile>(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    col_start: usize,
    col_end: usize,
    out: *mut f32,
) {
    // Seed this stripe with the bias so the tile accumulates in place.
    for row in 0..m {
        for column in col_start..col_end {
            // SAFETY: `row < m` and `column < n` by the caller's contract.
            unsafe {
                *out.add(row * n + column) = bias.map_or(0.0, |values| values[column]);
            }
        }
    }

    if m == 0 || k == 0 || col_start >= col_end {
        return; // bias-only result, already written
    }

    let columns = col_end - col_start;
    let m_full = m - m % T::ROWS;
    let n_full = col_start + columns - columns % T::COLS;

    // Columns per L2 panel block, sized so one packed panel plus its consumers stay resident; at
    // least one panel always, however large `k` is. Named distinctly from the `columns` above,
    // which is this stripe's WIDTH — reusing that name read as though a panel spanned the stripe.
    let panel_columns = {
        let fitting = PANEL_BYTES / (k.max(1) * size_of::<f32>());
        (fitting / T::COLS).max(1) * T::COLS
    };

    // Taken out of (and later returned to) the thread-local rather than borrowed inside a `with`
    // closure, so a panic mid-kernel merely drops the buffer instead of poisoning the slot.
    let mut panel_buffer = PANEL_SCRATCH.with(|scratch| std::mem::take(&mut *scratch.borrow_mut()));
    if panel_buffer.len() < k * T::COLS {
        panel_buffer.resize(k * T::COLS, 0.0);
    }
    let panel = &mut panel_buffer[..k * T::COLS];

    let mut jc = col_start;
    while jc < n_full {
        let jc_end = (jc + panel_columns).min(n_full);
        let mut j0 = jc;
        while j0 < jc_end {
            // Pack T::COLS weight columns into k-major order.
            //
            // This is the one place the `[n, k]` layout costs something: the reference kernel
            // copies a contiguous run, while here each of the sources is a separate row and the
            // gather has stride `k`. It is paid once per panel and amortized over every one of the
            // `m` rows that consume it, which is the entire point of packing.
            pack_panel(weight, panel, j0, T::COLS, k);

            let mut i0 = 0;
            while i0 < m_full {
                // SAFETY: rows `i0..i0+ROWS` are below `m` and columns `j0..j0+COLS` are inside
                // the caller's stripe, so every write lands within the `m * n` buffer; `T`'s CPU
                // precondition is ours.
                unsafe { T::accumulate(x, panel, out, i0, T::ROWS, j0, k, n) };
                i0 += T::ROWS;
            }
            // Rows below the last full tile still use the packed panel, as one shorter tile.
            if m_full < m {
                // SAFETY: as above, with the `m - m_full < ROWS` remaining rows.
                unsafe { T::accumulate(x, panel, out, m_full, m - m_full, j0, k, n) };
            }
            j0 += T::COLS;
        }
        jc += panel_columns;
    }

    // Remainder columns: fewer than a tile left over, so there is no panel to amortize and the
    // plain ascending-k dot is both simplest and exact.
    for row in 0..m {
        let x_row = &x[row * k..row * k + k];
        for column in n_full..col_end {
            let w_row = &weight[column * k..column * k + k];
            let mut sum = 0.0_f32;
            for depth in 0..k {
                sum += x_row[depth] * w_row[depth];
            }
            // SAFETY: `row < m`, `column < n`, inside the caller's buffer and stripe.
            unsafe { *out.add(row * n + column) += sum };
        }
    }

    PANEL_SCRATCH.with(|scratch| *scratch.borrow_mut() = panel_buffer);
}

/// Packs weight rows `j0..j0 + cols` (each `k` long) into the k-major `panel` (`k * cols`).
///
/// A pure copy — packing order cannot affect any result bit. It walks the transpose in blocks of
/// `DEPTH_BLOCK` depths across all `cols` rows, so the block being written (`DEPTH_BLOCK * cols`
/// floats, 2 KiB at 32 columns) stays in L1 while each source row is read one contiguous chunk at
/// a time. The naive column-at-a-time walk wrote with a `cols * 4`-byte stride across the whole
/// panel, which at 32 columns and k ≥ 512 evicted every line it touched before the next column
/// came back to it.
fn pack_panel(weight: &[f32], panel: &mut [f32], j0: usize, cols: usize, k: usize) {
    const DEPTH_BLOCK: usize = 16;
    let mut depth0 = 0;
    while depth0 < k {
        let depth_end = (depth0 + DEPTH_BLOCK).min(k);
        for column in 0..cols {
            let row = j0 + column;
            let source = &weight[row * k + depth0..row * k + depth_end];
            for (offset, &value) in source.iter().enumerate() {
                panel[(depth0 + offset) * cols + column] = value;
            }
        }
        depth0 = depth_end;
    }
}

/// Accumulates one `ROWS x COLS` output tile from a packed weight panel (the portable tile).
///
/// Generic over the tile so the full-tile and single-row cases share one body and one reduction
/// order; const generics keep the accumulator a fixed-size array, which is what lets LLVM keep it
/// in registers and vectorize the inner update.
///
/// # Safety
///
/// `out` must be valid for writes covering rows `i0..i0+ROWS` and columns `j0..j0+COLS` of an
/// `[m, n]` matrix.
// SAFETY: both call sites sit inside `PortableTile::accumulate`, whose trait contract bounds the
// rows and columns inside the caller's stripe and therefore inside its `m * n` buffer.
#[inline]
unsafe fn accumulate_tile<const ROWS: usize, const COLS: usize>(
    x: &[f32],
    panel: &[f32],
    out: *mut f32,
    i0: usize,
    j0: usize,
    k: usize,
    n: usize,
) {
    let mut acc = [[0.0_f32; COLS]; ROWS];
    for depth in 0..k {
        let weights = &panel[depth * COLS..depth * COLS + COLS];
        for (row, slots) in acc.iter_mut().enumerate() {
            // One activation value, broadcast across COLS weights: the multiply-add LLVM widens.
            let value = x[(i0 + row) * k + depth];
            for (slot, &weight) in slots.iter_mut().zip(weights) {
                *slot += value * weight;
            }
        }
    }
    for (row, slots) in acc.iter().enumerate() {
        let base = (i0 + row) * n + j0;
        for (column, &value) in slots.iter().enumerate() {
            // SAFETY: the caller guarantees this tile lies inside the output matrix.
            unsafe { *out.add(base + column) += value };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference this kernel must reproduce exactly: ascending-k scalar dot per element.
    fn scalar_reference(
        x: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; m * n];
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0_f32;
                for depth in 0..k {
                    sum += x[row * k + depth] * weight[column * k + depth];
                }
                out[row * n + column] = bias.map_or(sum, |b| sum + b[column]);
            }
        }
        out
    }

    fn deterministic(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Spread across a wide exponent range so any reassociation would show up: f32
                // addition is only non-associative when magnitudes differ.
                ((state >> 40) as f32 / 2048.0) - 0.5
            })
            .collect()
    }

    #[test]
    fn packed_matches_scalar_bit_for_bit() {
        // Shapes chosen to exercise every boundary the blocking can get wrong: m and n both above
        // and below the tile, exact multiples, one-off remainders, k = 0 and k = 1, and a k large
        // enough to force more than one column panel.
        let shapes = [
            (1, 1, 1),
            (1, 16, 8),
            (3, 5, 7),
            (4, 8, 8),
            (5, 9, 9),
            (8, 64, 16),
            (7, 128, 13),
            (16, 512, 32),
            (2, 0, 4),
            (4, 1, 8),
            (9, 1024, 24),
        ];
        for (index, &(m, k, n)) in shapes.iter().enumerate() {
            let x = deterministic(m * k, 0x51ED_0000 + index as u64);
            let weight = deterministic(n * k, 0xA113_0000 + index as u64);
            let bias = deterministic(n, 0xB1A5_0000 + index as u64);

            for carry_bias in [None, Some(&bias[..])] {
                let expected = scalar_reference(&x, &weight, carry_bias, m, k, n);
                let mut actual = vec![0.0_f32; m * n];
                linear_packed(&x, &weight, carry_bias, m, k, n, &mut actual);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "m={m} k={k} n={n} bias={}: packed GEMM diverged from the scalar reference",
                    carry_bias.is_some()
                );
            }
        }
    }

    #[test]
    fn every_isa_level_is_bit_identical_to_the_scalar_reference() {
        // The levels differ only in tile shape and register width; each must reproduce the
        // scalar ascending-k order exactly, at toy boundaries and at real codec geometry
        // (block_00's K = 7168, the decoder.1 residual conv's 5376 x 768, a 1-column tail).
        let shapes = [
            (1, 1, 1),
            (3, 5, 7),
            (9, 33, 17),
            (8, 64, 16),
            (17, 96, 40),
            (16, 7168, 48),
            (130, 5376, 32),
            (11, 672, 1),
            (2, 0, 4),
            // Every partial-row tile height of both x86 tiles (m % 8 and m % 6 = 1..7) over
            // full column tiles plus a ragged column remainder.
            (7, 64, 96),
            (13, 40, 72),
            (23, 31, 64),
            (5, 200, 33),
        ];
        for (index, &(m, k, n)) in shapes.iter().enumerate() {
            let x = deterministic(m * k, 0x15A0_0000 + index as u64);
            let weight = deterministic(n * k, 0x15A1_0000 + index as u64);
            let bias = deterministic(n, 0x15A2_0000 + index as u64);
            let expected = scalar_reference(&x, &weight, Some(&bias), m, k, n);
            for level in F32GemmLevel::available() {
                let mut actual = vec![0.0_f32; m * n];
                // SAFETY: full column range of an exclusively held `m * n` buffer, and `level`
                // comes from `available()`.
                unsafe {
                    linear_packed_range_at(
                        level,
                        &x,
                        &weight,
                        Some(&bias),
                        m,
                        k,
                        n,
                        0,
                        n,
                        actual.as_mut_ptr(),
                    );
                }
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "level {} diverged at m={m} k={k} n={n}",
                    level.as_str()
                );
            }
        }
    }

    /// Every partition count reproduces the serial bits at real codec geometry.
    ///
    /// This is the law the team dispatch rests on. It runs the SAME stripe function the workers
    /// run, at the codec's binding worst case (`block_00`, 1024 -> 1536 with kernel 7, so
    /// K = 7168), and at the transformer's shapes — because a partitioning that is exact at toy
    /// sizes and wrong at NR boundaries is exactly the bug that would ship.
    #[test]
    fn every_partition_count_reproduces_the_serial_bits() {
        let shapes = [
            // Batch-regime codec geometry (the original four).
            (32, 7168, 1536),
            (72, 512, 512),
            (48, 512, 1024),
            (17, 96, 40),
            // Single-row GEMV at real codec geometry — the interactive profile's per-frame
            // calls this team route now serves: a transformer projection, the binding
            // worst-case im2col reduction (block_00, kernel 7 x 1024 -> 1536), an upsample
            // pointwise pair, and one below-the-GEMV-floor shape that must stay exact too.
            (1, 512, 1024),
            (1, 7168, 1536),
            (1, 1024, 4096),
            (1, 8, 3),
        ];
        for (index, &(m, k, n)) in shapes.iter().enumerate() {
            let x = deterministic(m * k, 0x9E11_0000 + index as u64);
            let weight = deterministic(n * k, 0x7A31_0000 + index as u64);
            let bias = deterministic(n, 0x1CE5_0000 + index as u64);

            let mut serial = vec![0.0_f32; m * n];
            linear_packed(&x, &weight, Some(&bias), m, k, n, &mut serial);

            for partitions in [1, 2, 3, 5, 6, 8] {
                let mut parallel = vec![0.0_f32; m * n];
                // Exactly the stripe arithmetic in `run_f32_linear_partition`.
                let chunk = n.div_ceil(partitions).next_multiple_of(STRIPE_COLUMNS);
                for worker in 0..partitions {
                    let start = (worker * chunk).min(n);
                    let end = ((worker + 1) * chunk).min(n);
                    if start >= end {
                        continue;
                    }
                    // SAFETY: stripes are disjoint and inside the m*n buffer.
                    unsafe {
                        linear_packed_range(
                            &x,
                            &weight,
                            Some(&bias),
                            m,
                            k,
                            n,
                            start,
                            end,
                            parallel.as_mut_ptr(),
                        );
                    }
                }
                assert_eq!(
                    parallel.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "m={m} k={k} n={n} partitions={partitions}"
                );
            }
        }
    }

    #[test]
    fn column_partitions_are_bit_identical_to_the_whole() {
        // The property the KernelTeam relies on: computing a disjoint column range in isolation
        // yields exactly the bits the full call would have written there. True because no
        // reduction crosses a column.
        let (m, k, n) = (6, 96, 24);
        let x = deterministic(m * k, 0xC0F1);
        let weight = deterministic(n * k, 0xD00D);
        let mut whole = vec![0.0_f32; m * n];
        linear_packed(&x, &weight, None, m, k, n, &mut whole);

        for split in [8, 16] {
            let columns = split;
            let slice: Vec<f32> = weight[..columns * k].to_vec();
            let mut part = vec![0.0_f32; m * columns];
            linear_packed(&x, &slice, None, m, k, columns, &mut part);
            for row in 0..m {
                for column in 0..columns {
                    assert_eq!(
                        part[row * columns + column].to_bits(),
                        whole[row * n + column].to_bits(),
                        "split={split} row={row} column={column}"
                    );
                }
            }
        }
    }
}
