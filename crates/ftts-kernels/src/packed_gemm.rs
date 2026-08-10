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

/// Rows of the output tile held in registers.
///
/// Four rows of eight `f32` is 32 accumulators — eight v128 registers on wasm, which fits the
/// 16-register file with room for the operands. Wider tiles spill. Row remainders shorter than
/// `MR` run one row at a time (`accumulate_tile::<1>`); there is no intermediate 2-row tile.
const MR: usize = 4;

/// Columns of the output tile held in registers: two full v128 lanes of `f32`.
const NR: usize = 8;

/// Target bytes for one packed weight panel, sized to sit in L2 alongside the activation rows.
const PANEL_BYTES: usize = 256 * 1024;

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
    let m_full = m - m % MR;
    let n_full = col_start + columns - columns % NR;

    // Columns per L2 panel block, sized so one packed panel plus its consumers stay resident; at
    // least one panel always, however large `k` is. Named distinctly from the `columns` above,
    // which is this stripe's WIDTH — reusing that name read as though a panel spanned the stripe.
    let panel_columns = {
        let fitting = PANEL_BYTES / (k.max(1) * size_of::<f32>());
        (fitting / NR).max(1) * NR
    };

    // Thread-local scratch instead of a per-dispatch `vec!`: this function runs once per
    // stripe per dispatch on the steady-state decode path, and the doctrine pins "no
    // allocator activity in steady-state decode" as load-bearing. Each team worker (and the
    // dispatcher) owns its thread's buffer, so there is no sharing to reason about; the
    // buffer only ever grows, to the largest `k * NR` this thread has seen.
    thread_local! {
        static PANEL_SCRATCH: std::cell::RefCell<Vec<f32>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    PANEL_SCRATCH.with(|scratch| {
        let mut panel_guard = scratch.borrow_mut();
        if panel_guard.len() < k * NR {
            panel_guard.resize(k * NR, 0.0);
        }
        let panel = &mut panel_guard[..k * NR];

        let mut jc = col_start;
        while jc < n_full {
            let jc_end = (jc + panel_columns).min(n_full);
            let mut j0 = jc;
            while j0 < jc_end {
                // Pack NR weight columns into k-major order.
                //
                // This is the one place the `[n, k]` layout costs something: the reference kernel
                // copies a contiguous run, while here each of the NR sources is a separate row and the
                // gather has stride `k`. It is paid once per panel and amortized over every one of the
                // `m` rows that consume it, which is the entire point of packing.
                for (column, offset) in (j0..j0 + NR).enumerate() {
                    let source = &weight[offset * k..offset * k + k];
                    for (depth, &value) in source.iter().enumerate() {
                        panel[depth * NR + column] = value;
                    }
                }

                let mut i0 = 0;
                while i0 < m_full {
                    // SAFETY: rows `i0..i0+MR` are below `m` and columns `j0..j0+NR` are inside the
                    // caller's stripe, so every write lands within the `m * n` buffer.
                    unsafe { accumulate_tile::<MR>(x, panel, out, i0, j0, k, n) };
                    i0 += MR;
                }
                // Rows below the last full tile still benefit from the packed panel; run them one row
                // at a time rather than dropping to the unpacked tail path.
                for row in m_full..m {
                    // SAFETY: as above, with a single row.
                    unsafe { accumulate_tile::<1>(x, panel, out, row, j0, k, n) };
                }
                j0 += NR;
            }
            jc += panel_columns;
        }

        // Remainder columns: fewer than NR left over, so there is no panel to amortize and the plain
        // ascending-k dot is both simplest and exact.
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
    });
}

/// Accumulates one `ROWS x NR` output tile from a packed weight panel.
///
/// Generic over `ROWS` so the full-tile and single-row cases share one body and one reduction
/// order; a const generic keeps the accumulator a fixed-size array, which is what lets LLVM keep
/// it in registers and vectorize the inner update.
///
/// # Safety
///
/// `out` must be valid for writes covering rows `i0..i0+ROWS` and columns `j0..j0+NR` of an
/// `[m, n]` matrix.
// SAFETY: both call sites sit inside `linear_packed_range`'s loops, where `i0 + ROWS <= m` and
// `j0 + NR <= n_full <= col_end` hold by the loop bounds, so every tile lies inside the caller's
// stripe and therefore inside its `m * n` buffer.
#[inline]
unsafe fn accumulate_tile<const ROWS: usize>(
    x: &[f32],
    panel: &[f32],
    out: *mut f32,
    i0: usize,
    j0: usize,
    k: usize,
    n: usize,
) {
    let mut acc = [[0.0_f32; NR]; ROWS];
    for depth in 0..k {
        let weights = &panel[depth * NR..depth * NR + NR];
        for (row, slots) in acc.iter_mut().enumerate() {
            // One activation value, broadcast across NR weights: the multiply-add LLVM widens.
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

    /// Every partition count reproduces the serial bits at real codec geometry.
    ///
    /// This is the law the team dispatch rests on. It runs the SAME stripe function the workers
    /// run, at the codec's binding worst case (`block_00`, 1024 -> 1536 with kernel 7, so
    /// K = 7168), and at the transformer's shapes — because a partitioning that is exact at toy
    /// sizes and wrong at NR boundaries is exactly the bug that would ship.
    #[test]
    fn every_partition_count_reproduces_the_serial_bits() {
        let shapes = [
            (32, 7168, 1536),
            (72, 512, 512),
            (48, 512, 1024),
            (17, 96, 40),
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
                let chunk = n.div_ceil(partitions).next_multiple_of(NR);
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
