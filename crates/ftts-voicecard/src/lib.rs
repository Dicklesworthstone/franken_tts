//! The voice mosaic: 4 KB of x-vector as a self-locating field of emerald cells.
//!
//! This crate is the Rust twin of the iOS app's `Sources/VoiceCode.swift` (and the
//! chunk layer of `VoicePrintCard.swift`). The two MUST stay in lockstep:
//! [`render_mosaic_pixels`] is bit-identical to the Swift encoder for identical input,
//! and [`decode`] accepts anything the Swift decoder accepts — original PNGs,
//! recompressed JPEGs, screenshots from any size of phone. Change one side only with
//! the other in hand.
//!
//! Design, and why:
//! - 144×144 cells, 2 bits per cell as one of FOUR brightness levels on a constant-hue
//!   emerald ramp. JPEG keeps luminance at full resolution and halves chroma, so the
//!   information rides in the channel compression treats best; four levels per cell is
//!   what makes this denser than a QR code's one bit per module.
//! - Three QR-style finder patterns in the corners. The decoder scans the whole image
//!   for their 1:1:3:1:1 run signature, so registration is automatic at any scale or
//!   offset. Shared images stay axis-aligned (the channel is a file, not a camera).
//! - A calibration row and column teach the decoder the four brightness levels as they
//!   survived compression and pin per-axis scale.
//! - Interleaved Reed-Solomon (255, 223): each of 19 blocks corrects up to 16 wrong
//!   bytes, and the interleave spreads local damage across all of them. A CRC over the
//!   plaintext refuses silent miscorrection.

/// Cells per side of the mosaic grid.
pub const GRID_N: usize = 144;
/// Pixels per side of the rendered mosaic square (quiet margin included).
pub const CARD_SIZE: usize = 1024;
/// Pixels from the card edge to the grid.
pub const REGION_ORIGIN: usize = 80;
/// Pixels per cell side.
pub const CELL_PX: usize = 6;
/// Floats in a speaker vector; the payload is exactly this many.
pub const VECTOR_WIDTH: usize = 1024;

/// Emerald ramp, darkest to brightest. Constant hue family, widely separated luma.
const LEVELS: [[u8; 3]; 4] = [[6, 22, 15], [26, 92, 61], [56, 168, 110], [112, 248, 170]];

const MAGIC: &[u8; 4] = b"FV02";
const DATA_BYTES_PER_BLOCK: usize = 223;
const PARITY_BYTES_PER_BLOCK: usize = 32;
const BLOCKS: usize = 19;
const PLAINTEXT_CAPACITY: usize = DATA_BYTES_PER_BLOCK * BLOCKS; // 4237

const FINDER_SPAN: usize = 7; // finder pattern cells, plus a 1-cell separator
const CALIBRATION_ROW: usize = 8;
const CALIBRATION_COLUMN: usize = 8;

// ---------------------------------------------------------------------------- layout

/// Finder + separator zones: top-left, top-right, bottom-left (8×8 cells each).
fn in_finder_zone(row: usize, column: usize) -> bool {
    let zone = FINDER_SPAN + 1;
    (row < zone && column < zone)
        || (row < zone && column >= GRID_N - zone)
        || (row >= GRID_N - zone && column < zone)
}

/// The calibration column pins the VERTICAL scale the way the row pins the horizontal
/// one; it runs between the top-left and bottom-left zones.
fn in_calibration_column(row: usize, column: usize) -> bool {
    column == CALIBRATION_COLUMN && row > CALIBRATION_ROW && row < GRID_N - FINDER_SPAN - 1
}

fn is_reserved(row: usize, column: usize) -> bool {
    row == CALIBRATION_ROW || in_calibration_column(row, column) || in_finder_zone(row, column)
}

/// Level for a reserved cell: finder rings or the known calibration patterns.
fn reserved_level(row: usize, column: usize) -> usize {
    if row == CALIBRATION_ROW {
        return column % 4;
    }
    if in_calibration_column(row, column) {
        return row % 4;
    }
    // Local coordinates within whichever finder square this is; values outside the
    // 7×7 pattern are the separator ring. Signed arithmetic mirrors the Swift original,
    // where the top-left square keeps raw coordinates and the far squares shift.
    let mut r = row as isize;
    let mut c = column as isize;
    if column >= GRID_N - FINDER_SPAN - 1 {
        c = column as isize - (GRID_N - FINDER_SPAN) as isize;
    }
    if row >= GRID_N - FINDER_SPAN - 1 {
        r = row as isize - (GRID_N - FINDER_SPAN) as isize;
    }
    let span = FINDER_SPAN as isize;
    if !(0..span).contains(&r) || !(0..span).contains(&c) {
        return 0; // separator ring: darkest
    }
    let ring = r.min(c).min(span - 1 - r).min(span - 1 - c);
    if ring == 1 { 0 } else { 3 } // bright border, dark ring, bright 3×3 core
}

// ---------------------------------------------------------------------------- encode

/// Encode name + vector into an RGB24 mosaic image (`CARD_SIZE` × `CARD_SIZE`).
///
/// Bit-identical to the Swift encoder: the name is truncated to its first 64 UTF-8
/// BYTES (even mid-character — parity beats tidiness; the decoder is lossy-tolerant).
///
/// # Panics
///
/// If `vector` is not exactly [`VECTOR_WIDTH`] floats — the payload layout is fixed.
#[must_use]
pub fn render_mosaic_pixels(name: &str, vector: &[f32]) -> Vec<u8> {
    assert_eq!(
        vector.len(),
        VECTOR_WIDTH,
        "the mosaic carries exactly 1,024 floats"
    );
    let mut plaintext = MAGIC.to_vec();
    let name_bytes = &name.as_bytes()[..name.len().min(64)];
    plaintext.push((name_bytes.len() >> 8) as u8);
    plaintext.push((name_bytes.len() & 0xFF) as u8);
    plaintext.extend_from_slice(name_bytes);
    for value in vector {
        plaintext.extend_from_slice(&value.to_le_bytes());
    }
    let crc = crc32(&plaintext);
    plaintext.extend_from_slice(&crc.to_be_bytes());
    assert!(
        plaintext.len() <= PLAINTEXT_CAPACITY,
        "payload exceeds the mosaic"
    );
    plaintext.resize(PLAINTEXT_CAPACITY, 0);

    // Reed-Solomon per block, then byte-interleave across blocks.
    let blocks_out: Vec<Vec<u8>> = (0..BLOCKS)
        .map(|block| {
            let start = block * DATA_BYTES_PER_BLOCK;
            let mut word = plaintext[start..start + DATA_BYTES_PER_BLOCK].to_vec();
            word.extend_from_slice(&rs::parity(&plaintext[start..start + DATA_BYTES_PER_BLOCK]));
            word
        })
        .collect();
    let mut coded = Vec::with_capacity((DATA_BYTES_PER_BLOCK + PARITY_BYTES_PER_BLOCK) * BLOCKS);
    for position in 0..DATA_BYTES_PER_BLOCK + PARITY_BYTES_PER_BLOCK {
        for word in &blocks_out {
            coded.push(word[position]);
        }
    }
    whiten(&mut coded);

    let mut pixels = vec![0_u8; CARD_SIZE * CARD_SIZE * 3];
    let background = [LEVELS[0][0] / 2, LEVELS[0][1] / 2, LEVELS[0][2] / 2];
    for pixel in pixels.chunks_mut(3) {
        pixel.copy_from_slice(&background);
    }
    let mut bit_cursor = 0_usize;
    let total_bits = coded.len() * 8;
    for row in 0..GRID_N {
        for column in 0..GRID_N {
            let level = if is_reserved(row, column) {
                reserved_level(row, column)
            } else if bit_cursor + 2 <= total_bits {
                let byte = coded[bit_cursor >> 3];
                let shift = 6 - (bit_cursor & 7);
                bit_cursor += 2;
                usize::from((byte >> shift) & 0b11)
            } else {
                // Deterministic filler, hashed so it blends with the data field.
                let mut hash = ((row * GRID_N + column) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                hash ^= hash >> 29;
                hash = hash.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                (hash >> 62) as usize
            };
            paint_cell(&mut pixels, row, column, level);
        }
    }
    pixels
}

/// XOR the coded stream with a fixed pseudo-random mask. The floats in the payload
/// repeat byte patterns that would otherwise show as visible stripes, and a
/// pathological payload could produce large flat regions that starve the decoder's
/// threshold estimate — masking keeps the field uniformly mixed.
fn whiten(bytes: &mut [u8]) {
    let mut state: u64 = 0x5DEE_CE66_D1CE_F001;
    for byte in bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte ^= (state >> 32) as u8;
    }
}

fn paint_cell(pixels: &mut [u8], row: usize, column: usize, level: usize) {
    let color = LEVELS[level];
    let x0 = REGION_ORIGIN + column * CELL_PX;
    let y0 = REGION_ORIGIN + row * CELL_PX;
    for y in y0..y0 + CELL_PX {
        let start = (y * CARD_SIZE + x0) * 3;
        for pixel in pixels[start..start + CELL_PX * 3].chunks_mut(3) {
            pixel.copy_from_slice(&color);
        }
    }
}

// ---------------------------------------------------------------------------- decode

#[derive(Clone, Copy, Debug)]
struct FinderCandidate {
    x: f64,
    y: f64,
    module: f64,
    votes: usize,
}

/// Decode a voice from RGB24 pixels of any image containing the mosaic: original
/// PNG, recompressed JPEG, or a screenshot from any size of phone. `None` when no
/// intact mosaic is found.
#[must_use]
pub fn decode(pixels: &[u8], width: usize, height: usize) -> Option<(String, Vec<f32>)> {
    let required_bytes = width.checked_mul(height)?.checked_mul(3)?;
    if width < GRID_N || height < GRID_N || pixels.len() < required_bytes {
        return None;
    }
    let mut luma = vec![0_f32; required_bytes / 3];
    let mut min_luma = 255.0_f64;
    let mut max_luma = 0.0_f64;
    for (index, slot) in luma.iter_mut().enumerate() {
        let at = index * 3;
        let value = 0.299 * f64::from(pixels[at])
            + 0.587 * f64::from(pixels[at + 1])
            + 0.114 * f64::from(pixels[at + 2]);
        *slot = value as f32;
        min_luma = min_luma.min(value);
        max_luma = max_luma.max(value);
    }
    if max_luma - min_luma <= 30.0 {
        return None;
    }
    let threshold = (min_luma + max_luma) / 2.0;

    let finders = find_finder_patterns(&luma, width, height, threshold);
    // Data cells can imitate a finder by chance, so several plausible triples may
    // exist; try them best-first — the calibration fit and the CRC reject impostor
    // grids, so the first one that decodes is the real one.
    for triple in rank_finder_triples(&finders) {
        if let Some(decoded) = decode_grid(triple, &luma, width, height, threshold) {
            return Some(decoded);
        }
    }
    None
}

#[expect(
    clippy::too_many_lines,
    reason = "one grid pass, mirrored from the Swift original"
)]
fn decode_grid(
    triple: (FinderCandidate, FinderCandidate, FinderCandidate),
    luma: &[f32],
    width: usize,
    height: usize,
    threshold: f64,
) -> Option<(String, Vec<f32>)> {
    let bilinear = |x: f64, y: f64| -> f64 {
        let cx = x.clamp(0.0, (width - 1) as f64);
        let cy = y.clamp(0.0, (height - 1) as f64);
        let x0 = (cx as usize).min(width - 2);
        let y0 = (cy as usize).min(height - 2);
        let fx = cx - x0 as f64;
        let fy = cy - y0 as f64;
        let a = f64::from(luma[y0 * width + x0]);
        let b = f64::from(luma[y0 * width + x0 + 1]);
        let c = f64::from(luma[(y0 + 1) * width + x0]);
        let d = f64::from(luma[(y0 + 1) * width + x0 + 1]);
        a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy
    };

    // Sub-pixel refinement of a finder center: brightness-weighted centroid over a
    // symmetric window that covers the finder but stops inside its separator.
    let refined = |finder: FinderCandidate| -> (f64, f64) {
        let reach = finder.module * 3.4;
        let x0 = ((finder.x - reach).round().max(0.0)) as usize;
        let x1 = (((finder.x + reach).round()) as usize).min(width - 1);
        let y0 = ((finder.y - reach).round().max(0.0)) as usize;
        let y1 = (((finder.y + reach).round()) as usize).min(height - 1);
        if x1 <= x0 || y1 <= y0 {
            return (finder.x, finder.y);
        }
        let mut weight_sum = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let weight = (f64::from(luma[y * width + x]) - threshold).max(0.0);
                weight_sum += weight;
                sum_x += weight * x as f64;
                sum_y += weight * y as f64;
            }
        }
        if weight_sum > 0.0 {
            (sum_x / weight_sum, sum_y / weight_sum)
        } else {
            (finder.x, finder.y)
        }
    };

    let top_left = refined(triple.0);
    let top_right = refined(triple.1);
    let bottom_left = refined(triple.2);

    // Grid geometry: cell k's center sits at (k + 0.5) cell-units from the region
    // edge; the finder centers sit at cell coordinate 3.5 and GRID_N − 3.5.
    let center_span = (GRID_N - FINDER_SPAN) as f64;
    let cell_w = (top_right.0 - top_left.0) / center_span;
    let cell_h = (bottom_left.1 - top_left.1) / center_span;
    if cell_w <= 1.2 || cell_h <= 1.2 {
        return None; // near 1 px/cell is gone
    }

    // Matched box sample of one cell, anchored at the top-left finder so a scale
    // tweak grows outward from a fixed point.
    let sample = |row: f64,
                  column: f64,
                  scale_x: f64,
                  scale_y: f64,
                  dx: f64,
                  dy: f64,
                  footprint: f64|
     -> f64 {
        let x = top_left.0 + (column - 3.0) * cell_w * scale_x + dx;
        let y = top_left.1 + (row - 3.0) * cell_h * scale_y + dy;
        if footprint <= 0.0 {
            return bilinear(x, y);
        }
        let rx = cell_w * footprint;
        let ry = cell_h * footprint;
        (bilinear(x, y)
            + bilinear(x - rx, y - ry)
            + bilinear(x + rx, y - ry)
            + bilinear(x - rx, y + ry)
            + bilinear(x + rx, y + ry))
            / 5.0
    };

    // Fit one axis against its known calibration strip: search a small offset and
    // scale range. The score is separation between adjacent levels MINUS the spread
    // within each level — a misaligned grid still averages to clean means (the strip
    // repeats every four flat cells) but bleeds neighboring cells into individual
    // samples, and the variance term is what catches that.
    let fit_axis = |horizontal: bool, footprint: f64| -> Option<(f64, f64, f64, [f64; 4])> {
        let mut best: Option<(f64, f64, f64, [f64; 4])> = None;
        for scale_step in -3_i32..=3 {
            let scale = 1.0 + f64::from(scale_step) * 0.0018;
            for offset_step in -4_i32..=4 {
                let offset =
                    f64::from(offset_step) * (if horizontal { cell_w } else { cell_h }) * 0.1;
                let mut sums = [0.0_f64; 4];
                let mut squares = [0.0_f64; 4];
                let mut counts = [0.0_f64; 4];
                if horizontal {
                    for column in 0..GRID_N {
                        let value = sample(
                            CALIBRATION_ROW as f64,
                            column as f64,
                            scale,
                            1.0,
                            offset,
                            0.0,
                            footprint,
                        );
                        sums[column % 4] += value;
                        squares[column % 4] += value * value;
                        counts[column % 4] += 1.0;
                    }
                } else {
                    for row in CALIBRATION_ROW + 1..GRID_N - FINDER_SPAN - 1 {
                        let value = sample(
                            row as f64,
                            CALIBRATION_COLUMN as f64,
                            1.0,
                            scale,
                            0.0,
                            offset,
                            footprint,
                        );
                        sums[row % 4] += value;
                        squares[row % 4] += value * value;
                        counts[row % 4] += 1.0;
                    }
                }
                let mut means = [0.0_f64; 4];
                let mut spread = 0.0;
                for level in 0..4 {
                    let count = counts[level].max(1.0);
                    means[level] = sums[level] / count;
                    let variance = (squares[level] / count - means[level] * means[level]).max(0.0);
                    spread += variance.sqrt() / 4.0;
                }
                let gaps = [
                    means[1] - means[0],
                    means[2] - means[1],
                    means[3] - means[2],
                ];
                let score = gaps.iter().copied().fold(f64::INFINITY, f64::min) - 2.0 * spread;
                if best.is_none_or(|(s, ..)| score > s) {
                    best = Some((score, scale, offset, means));
                }
            }
        }
        best
    };

    // Cells much bigger than the resampling blur average well over a small footprint;
    // cells near 3 px are only trustworthy at their very center. Try the footprint
    // suited to this scale first, the other as a fallback.
    let footprints: [f64; 2] = if cell_w < 4.5 {
        [0.0, 0.22]
    } else {
        [0.22, 0.0]
    };
    'attempt: for footprint in footprints {
        let Some((score_x, scale_x, dx, means_x)) = fit_axis(true, footprint) else {
            continue;
        };
        let Some((score_y, scale_y, dy, means_y)) = fit_axis(false, footprint) else {
            continue;
        };
        if score_x <= 4.0 || score_y <= 4.0 {
            continue;
        }
        let means: Vec<f64> = means_x
            .iter()
            .zip(&means_y)
            .map(|(a, b)| (a + b) / 2.0)
            .collect();
        let thresholds = [
            (means[0] + means[1]) / 2.0,
            (means[1] + means[2]) / 2.0,
            (means[2] + means[3]) / 2.0,
        ];

        // Sample every data cell in layout order.
        let coded_count = (DATA_BYTES_PER_BLOCK + PARITY_BYTES_PER_BLOCK) * BLOCKS;
        let mut coded = vec![0_u8; coded_count];
        let mut bit_cursor = 0_usize;
        let total_bits = coded_count * 8;
        'grid: for row in 0..GRID_N {
            for column in 0..GRID_N {
                if is_reserved(row, column) {
                    continue;
                }
                if bit_cursor + 2 > total_bits {
                    break 'grid;
                }
                let value = sample(
                    row as f64,
                    column as f64,
                    scale_x,
                    scale_y,
                    dx,
                    dy,
                    footprint,
                );
                let mut level = 0_u8;
                for threshold in thresholds {
                    if value > threshold {
                        level += 1;
                    }
                }
                let shift = 6 - (bit_cursor & 7);
                coded[bit_cursor >> 3] |= level << shift;
                bit_cursor += 2;
            }
        }

        // Unmask, then de-interleave and correct each block.
        whiten(&mut coded);
        let mut plaintext = Vec::with_capacity(PLAINTEXT_CAPACITY);
        for block in 0..BLOCKS {
            let received: Vec<u8> = (0..DATA_BYTES_PER_BLOCK + PARITY_BYTES_PER_BLOCK)
                .map(|position| coded[position * BLOCKS + block])
                .collect();
            let Some(corrected) = rs::correct(&received) else {
                continue 'attempt;
            };
            plaintext.extend_from_slice(&corrected[..DATA_BYTES_PER_BLOCK]);
        }
        if let Some(decoded) = parse(&plaintext) {
            return Some(decoded);
        }
    }
    None
}

/// Scan rows for the bright-dark-bright(3)-dark-bright 1:1:3:1:1 run signature,
/// verify each hit vertically, and cluster the survivors.
fn find_finder_patterns(
    luma: &[f32],
    width: usize,
    height: usize,
    threshold: f64,
) -> Vec<FinderCandidate> {
    let mut candidates: Vec<FinderCandidate> = Vec::new();
    let row_step = (height / 700).max(1);

    let matches_ratio = |runs: &[f64; 5], module: f64| -> bool {
        module > 0.6
            && (runs[0] - module).abs() < module * 0.65
            && (runs[1] - module).abs() < module * 0.65
            && (runs[2] - 3.0 * module).abs() < module * 1.2
            && (runs[3] - module).abs() < module * 0.65
            && (runs[4] - module).abs() < module * 0.65
    };

    // Vertical confirmation at a candidate x: bright core of ~3 modules with a dark
    // ring then a bright ring above and below. Returns center y and module.
    let vertical_check = |x: usize, y: usize, module: f64| -> Option<(f64, f64)> {
        let bright = |yy: usize| f64::from(luma[yy * width + x]) > threshold;
        if !bright(y) {
            return None;
        }
        let limit = (module * 6.0) as usize + 2;
        let mut top = y;
        while top > 0 && bright(top - 1) && y - top < limit {
            top -= 1;
        }
        let mut bottom = y;
        while bottom < height - 1 && bright(bottom + 1) && bottom - y < limit {
            bottom += 1;
        }
        let core = (bottom - top + 1) as f64;
        if (core - 3.0 * module).abs() >= module * 1.6 {
            return None;
        }
        let ring_span = |start: usize, step_down: bool| -> Option<(usize, usize)> {
            let mut yy = if step_down {
                start.checked_add(1)?
            } else {
                start.checked_sub(1)?
            };
            let mut dark = 0_usize;
            loop {
                if yy >= height || bright(yy) || dark >= limit {
                    break;
                }
                dark += 1;
                let next = if step_down {
                    yy.checked_add(1)
                } else {
                    yy.checked_sub(1)
                };
                match next {
                    Some(n) => yy = n,
                    None => break,
                }
            }
            let mut bright_run = 0_usize;
            loop {
                if yy >= height || !bright(yy) || bright_run >= limit {
                    break;
                }
                bright_run += 1;
                let next = if step_down {
                    yy.checked_add(1)
                } else {
                    yy.checked_sub(1)
                };
                match next {
                    Some(n) => yy = n,
                    None => break,
                }
            }
            (dark > 0 && bright_run > 0).then_some((dark, bright_run))
        };
        let above = ring_span(top, false)?;
        let below = ring_span(bottom, true)?;
        let near = |value: usize| (value as f64 - module).abs() < module * 0.9;
        if !(near(above.0) && near(below.0) && near(above.1) && near(below.1)) {
            return None;
        }
        Some(((top + bottom) as f64 / 2.0, core / 3.0))
    };

    let mut y = 0;
    while y < height {
        let row_base = y * width;
        let mut runs: Vec<f64> = Vec::new();
        let mut run_is_bright: Vec<bool> = Vec::new();
        let mut current = f64::from(luma[row_base]) > threshold;
        let mut length = 1.0_f64;
        for x in 1..width {
            let bright = f64::from(luma[row_base + x]) > threshold;
            if bright == current {
                length += 1.0;
            } else {
                runs.push(length);
                run_is_bright.push(current);
                current = bright;
                length = 1.0;
            }
        }
        runs.push(length);
        run_is_bright.push(current);

        let mut position = 0.0_f64;
        for index in 0..runs.len() {
            let advance = runs[index];
            if index + 4 < runs.len() && run_is_bright[index] {
                let window: [f64; 5] = runs[index..index + 5].try_into().expect("five runs");
                let module = window.iter().sum::<f64>() / 7.0;
                if matches_ratio(&window, module) {
                    let center_x = position + window.iter().sum::<f64>() / 2.0;
                    let xi = (center_x.round().max(0.0) as usize).min(width - 1);
                    if let Some((center_y, module_v)) = vertical_check(xi, y, module) {
                        let ratio = module_v / module;
                        if ratio > 0.5 && ratio < 2.0 {
                            let mut merged = false;
                            for candidate in &mut candidates {
                                if (candidate.x - center_x).abs() < module * 4.0
                                    && (candidate.y - center_y).abs() < module * 4.0
                                {
                                    let weight = candidate.votes as f64;
                                    candidate.x =
                                        (candidate.x * weight + center_x) / (weight + 1.0);
                                    candidate.y =
                                        (candidate.y * weight + center_y) / (weight + 1.0);
                                    candidate.module =
                                        (candidate.module * weight + module) / (weight + 1.0);
                                    candidate.votes += 1;
                                    merged = true;
                                    break;
                                }
                            }
                            if !merged {
                                candidates.push(FinderCandidate {
                                    x: center_x,
                                    y: center_y,
                                    module,
                                    votes: 1,
                                });
                            }
                        }
                    }
                }
            }
            position += advance;
        }
        y += row_step;
    }
    candidates.retain(|candidate| candidate.votes >= 2);
    candidates
}

/// Rank plausible (top-left, top-right, bottom-left) triples, best first. The layout
/// itself is the constraint: arms are axis-aligned, near-equal (shared images keep
/// their aspect), and exactly `GRID_N − FINDER_SPAN` cells long, so each arm must be
/// about 137× the finder's own module size.
fn rank_finder_triples(
    candidates: &[FinderCandidate],
) -> Vec<(FinderCandidate, FinderCandidate, FinderCandidate)> {
    if candidates.len() < 3 {
        return Vec::new();
    }
    // The search below is cubic; a busy photo can spawn hundreds of accidental
    // candidates, so rank only the most-voted few dozen.
    let mut pool: Vec<FinderCandidate> = candidates.to_vec();
    pool.sort_by_key(|candidate| std::cmp::Reverse(candidate.votes));
    pool.truncate(48);
    let arm_cells = (GRID_N - FINDER_SPAN) as f64;

    let mut ranked: Vec<(f64, (FinderCandidate, FinderCandidate, FinderCandidate))> = Vec::new();
    for (i, corner) in pool.iter().enumerate() {
        for (j, right) in pool.iter().enumerate() {
            if j == i {
                continue;
            }
            for (k, down) in pool.iter().enumerate() {
                if k == i || k == j {
                    continue;
                }
                let arm_x = (right.x - corner.x, right.y - corner.y);
                let arm_y = (down.x - corner.x, down.y - corner.y);
                if arm_x.0 <= 0.0 || arm_y.1 <= 0.0 {
                    continue; // orientation
                }
                let length_x = arm_x.0.hypot(arm_x.1);
                let length_y = arm_y.0.hypot(arm_y.1);
                if length_x <= 10.0 || length_y <= 10.0 {
                    continue;
                }
                if arm_x.1.abs() >= length_x * 0.1
                    || arm_y.0.abs() >= length_y * 0.1
                    || length_x / length_y <= 0.9
                    || length_x / length_y >= 1.11
                {
                    continue;
                }
                let modules = [corner.module, right.module, down.module];
                let module_max = modules.iter().copied().fold(f64::MIN, f64::max);
                let module_min = modules.iter().copied().fold(f64::MAX, f64::min).max(0.1);
                let module_spread = module_max / module_min;
                if module_spread >= 1.5 {
                    continue;
                }
                let mean_module = modules.iter().sum::<f64>() / 3.0;
                let arm_error = (length_x / (mean_module * arm_cells) - 1.0)
                    .abs()
                    .max((length_y / (mean_module * arm_cells) - 1.0).abs());
                if arm_error >= 0.25 {
                    continue;
                }
                let score = (corner.votes + right.votes + down.votes) as f64
                    - module_spread
                    - arm_error * 10.0;
                ranked.push((score, (*corner, *right, *down)));
            }
        }
    }
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(8);
    ranked.into_iter().map(|(_, triple)| triple).collect()
}

// --------------------------------------------------------------------------- payload

fn parse(plaintext: &[u8]) -> Option<(String, Vec<f32>)> {
    if plaintext.len() < MAGIC.len() + 2 || &plaintext[..MAGIC.len()] != MAGIC {
        return None;
    }
    let mut at = MAGIC.len();
    let name_length = usize::from(plaintext[at]) << 8 | usize::from(plaintext[at + 1]);
    at += 2;
    let vector_bytes = VECTOR_WIDTH * 4;
    if name_length > 64 || plaintext.len() < at + name_length + vector_bytes + 4 {
        return None;
    }
    let name = String::from_utf8_lossy(&plaintext[at..at + name_length]).into_owned();
    at += name_length;
    let mut vector = Vec::with_capacity(VECTOR_WIDTH);
    for index in 0..VECTOR_WIDTH {
        let base = at + index * 4;
        vector.push(f32::from_le_bytes([
            plaintext[base],
            plaintext[base + 1],
            plaintext[base + 2],
            plaintext[base + 3],
        ]));
    }
    at += vector_bytes;
    let stored = u32::from_be_bytes([
        plaintext[at],
        plaintext[at + 1],
        plaintext[at + 2],
        plaintext[at + 3],
    ]);
    if crc32(&plaintext[..at]) != stored || vector.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let name = if name.is_empty() {
        "shared voice".to_owned()
    } else {
        name
    };
    Some((name, vector))
}

/// CRC-32 (polynomial `0xEDB88320`) — the PNG polynomial, shared by the mosaic
/// payload and the lossless chunk.
#[must_use]
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    crc ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------- lossless PNG chunk

/// Private ancillary PNG chunk type carrying the byte-exact payload.
const CHUNK_TYPE: &[u8; 4] = b"ftTS";
const CHUNK_MAGIC: &[u8] = b"FTTSVOICE1";

/// Insert the lossless voice chunk into a well-formed PNG, immediately before IEND.
/// Byte-compatible with the iOS app's fast path. `None` if `png` is not a PNG.
#[must_use]
pub fn embed_chunk(name: &str, vector: &[f32], png: &[u8]) -> Option<Vec<u8>> {
    const SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    // The payload layout is fixed-width, and inserting before an arbitrary final
    // twelve bytes would manufacture a corrupt PNG from a merely signature-shaped
    // input. Validate both boundaries before constructing the ancillary chunk.
    if vector.len() != VECTOR_WIDTH
        || vector.iter().any(|value| !value.is_finite())
        || png.len() < 20
        || png[..8] != SIGNATURE
        || png[png.len() - 12..png.len() - 8] != [0, 0, 0, 0]
        || png[png.len() - 8..png.len() - 4] != *b"IEND"
        || png[png.len() - 4..] != crc32(b"IEND").to_be_bytes()
    {
        return None;
    }
    let mut payload = CHUNK_MAGIC.to_vec();
    let name_bytes = &name.as_bytes()[..name.len().min(64)];
    payload.push((name_bytes.len() >> 8) as u8);
    payload.push((name_bytes.len() & 0xFF) as u8);
    payload.extend_from_slice(name_bytes);
    for value in vector {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let mut chunk = (payload.len() as u32).to_be_bytes().to_vec();
    chunk.extend_from_slice(CHUNK_TYPE);
    chunk.extend_from_slice(&payload);
    let mut crc_input = CHUNK_TYPE.to_vec();
    crc_input.extend_from_slice(&payload);
    chunk.extend_from_slice(&crc32(&crc_input).to_be_bytes());

    // IEND is the last 12 bytes of a well-formed PNG.
    let mut out = png[..png.len() - 12].to_vec();
    out.extend_from_slice(&chunk);
    out.extend_from_slice(&png[png.len() - 12..]);
    Some(out)
}

/// Extract a voice from a PNG's lossless chunk, if present and intact.
#[must_use]
pub fn decode_chunk(data: &[u8]) -> Option<(String, Vec<f32>)> {
    const SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    if data.len() <= 16 || data[..8] != SIGNATURE {
        return None;
    }
    let mut offset = 8_usize;
    while offset + 12 <= data.len() {
        let length = u32::from_be_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        let kind = &data[offset + 4..offset + 8];
        let data_start = offset + 8;
        // Checked: `length` is attacker-controlled and usize is 32-bit on wasm, so
        // unchecked addition could wrap and let a bounds check pass falsely.
        let chunk_end = data_start.checked_add(length)?.checked_add(4)?;
        if chunk_end > data.len() {
            return None;
        }
        if kind == CHUNK_TYPE {
            let payload = &data[data_start..data_start + length];
            let stored = u32::from_be_bytes(
                data[data_start + length..data_start + length + 4]
                    .try_into()
                    .ok()?,
            );
            let mut crc_input = CHUNK_TYPE.to_vec();
            crc_input.extend_from_slice(payload);
            if crc32(&crc_input) != stored {
                return None;
            }
            return parse_chunk(payload);
        }
        offset = chunk_end;
    }
    None
}

fn parse_chunk(payload: &[u8]) -> Option<(String, Vec<f32>)> {
    if payload.len() <= CHUNK_MAGIC.len() + 2 || &payload[..CHUNK_MAGIC.len()] != CHUNK_MAGIC {
        return None;
    }
    let mut at = CHUNK_MAGIC.len();
    let name_length = usize::from(payload[at]) << 8 | usize::from(payload[at + 1]);
    at += 2;
    if name_length > 64 || payload.len() < at + name_length + VECTOR_WIDTH * 4 {
        return None;
    }
    let name = String::from_utf8_lossy(&payload[at..at + name_length]).into_owned();
    at += name_length;
    let mut vector = Vec::with_capacity(VECTOR_WIDTH);
    for index in 0..VECTOR_WIDTH {
        let base = at + index * 4;
        vector.push(f32::from_le_bytes([
            payload[base],
            payload[base + 1],
            payload[base + 2],
            payload[base + 3],
        ]));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let name = if name.is_empty() {
        "shared voice".to_owned()
    } else {
        name
    };
    Some((name, vector))
}

// ----------------------------------------------------------------------- Reed-Solomon

/// Reed-Solomon (255, 223) over GF(2^8), generator polynomial roots α^0..α^31.
mod rs {
    const PARITY: usize = 32;

    fn tables() -> (&'static [u8; 512], &'static [u8; 256]) {
        use std::sync::OnceLock;
        static TABLES: OnceLock<([u8; 512], [u8; 256])> = OnceLock::new();
        let (exp, log) = TABLES.get_or_init(|| {
            let mut exp = [0_u8; 512];
            let mut log = [0_u8; 256];
            let mut x = 1_usize;
            for (power, slot) in exp.iter_mut().enumerate().take(255) {
                *slot = x as u8;
                log[x] = power as u8;
                x <<= 1;
                if x & 0x100 != 0 {
                    x ^= 0x11D;
                }
            }
            for power in 255..512 {
                exp[power] = exp[power - 255];
            }
            (exp, log)
        });
        (exp, log)
    }

    fn multiply(a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            return 0;
        }
        let (exp, log) = tables();
        exp[usize::from(log[usize::from(a)]) + usize::from(log[usize::from(b)])]
    }

    fn inverse(value: u8) -> u8 {
        if value == 0 {
            return 0;
        }
        let (exp, log) = tables();
        exp[255 - usize::from(log[usize::from(value)])]
    }

    fn power(base: u8, exponent: usize) -> u8 {
        if base == 0 {
            return u8::from(exponent == 0);
        }
        let (exp, log) = tables();
        exp[(usize::from(log[usize::from(base)]) * exponent) % 255]
    }

    fn generator() -> &'static [u8; PARITY + 1] {
        use std::sync::OnceLock;
        static GENERATOR: OnceLock<[u8; PARITY + 1]> = OnceLock::new();
        GENERATOR.get_or_init(|| {
            let (exp, _) = tables();
            let mut poly = vec![1_u8];
            for &alpha in exp.iter().take(PARITY) {
                let mut next = vec![0_u8; poly.len() + 1];
                for (index, &coefficient) in poly.iter().enumerate() {
                    next[index] ^= multiply(coefficient, alpha);
                    next[index + 1] ^= coefficient;
                }
                poly = next;
            }
            poly.reverse(); // highest degree first
            poly.try_into()
                .expect("generator has PARITY + 1 coefficients")
        })
    }

    /// Parity bytes for a data block (systematic encoding).
    pub fn parity(data: &[u8]) -> [u8; PARITY] {
        let generator = generator();
        let mut remainder = [0_u8; PARITY];
        for &byte in data {
            let factor = byte ^ remainder[0];
            remainder.copy_within(1.., 0);
            remainder[PARITY - 1] = 0;
            if factor != 0 {
                for index in 0..PARITY {
                    remainder[index] ^= multiply(generator[index + 1], factor);
                }
            }
        }
        remainder
    }

    /// Correct up to 16 byte errors in a 255-byte codeword. `None` when unrecoverable.
    #[expect(
        clippy::too_many_lines,
        reason = "one textbook decoder, mirrored from Swift"
    )]
    pub fn correct(received: &[u8]) -> Option<Vec<u8>> {
        let (exp, _) = tables();
        let n = received.len();
        let mut syndromes = [0_u8; PARITY];
        let mut clean = true;
        for (index, syndrome) in syndromes.iter_mut().enumerate() {
            let mut value = 0_u8;
            for &byte in received {
                value = multiply(value, exp[index]) ^ byte;
            }
            *syndrome = value;
            if value != 0 {
                clean = false;
            }
        }
        if clean {
            return Some(received.to_vec());
        }

        // Berlekamp-Massey for the error locator polynomial.
        let mut sigma = vec![1_u8];
        let mut previous = vec![1_u8];
        let mut discrepancy_last = 1_u8;
        let mut m = 1_usize;
        for step in 0..PARITY {
            let mut discrepancy = syndromes[step];
            for index in 1..sigma.len() {
                if step >= index {
                    discrepancy ^= multiply(sigma[index], syndromes[step - index]);
                }
            }
            if discrepancy == 0 {
                m += 1;
                continue;
            }
            let scale = multiply(discrepancy, inverse(discrepancy_last));
            let mut shifted = vec![0_u8; m];
            shifted.extend_from_slice(&previous);
            for value in &mut shifted {
                *value = multiply(*value, scale);
            }
            if 2 * (sigma.len() - 1) <= step {
                let old = sigma.clone();
                sigma = xor_polynomials(&sigma, &shifted);
                previous = old;
                discrepancy_last = discrepancy;
                m = 1;
            } else {
                sigma = xor_polynomials(&sigma, &shifted);
                m += 1;
            }
        }
        let error_count = sigma.len() - 1;
        if error_count == 0 || error_count > PARITY / 2 {
            return None;
        }

        // Chien search for error positions.
        let mut positions = Vec::new();
        for position in 0..n {
            let x_inverse = exp[(255 - (n - 1 - position)) % 255];
            let mut value = 0_u8;
            for (index, &coefficient) in sigma.iter().enumerate() {
                value ^= multiply(coefficient, power(x_inverse, index));
            }
            if value == 0 {
                positions.push(position);
            }
        }
        if positions.len() != error_count {
            return None;
        }

        // Forney magnitudes: omega = (syndromes · sigma) mod x^PARITY.
        let mut omega = [0_u8; PARITY];
        for (index, slot) in omega.iter_mut().enumerate() {
            let mut value = 0_u8;
            for (j, &coefficient) in sigma.iter().enumerate() {
                if index >= j {
                    value ^= multiply(coefficient, syndromes[index - j]);
                }
            }
            *slot = value;
        }
        let mut corrected = received.to_vec();
        for &position in &positions {
            let x_inverse = exp[(255 - (n - 1 - position)) % 255];
            let mut numerator = 0_u8;
            for (index, &coefficient) in omega.iter().enumerate() {
                numerator ^= multiply(coefficient, power(x_inverse, index));
            }
            let mut denominator = 0_u8;
            let mut index = 1;
            while index < sigma.len() {
                denominator ^= multiply(sigma[index], power(x_inverse, index - 1));
                index += 2;
            }
            if denominator == 0 {
                return None;
            }
            // Forney with first consecutive root α^0 carries an extra factor of X_j.
            let xj = exp[(n - 1 - position) % 255];
            corrected[position] ^= multiply(multiply(numerator, xj), inverse(denominator));
        }
        // Verify: syndromes of the corrected word must vanish.
        for &alpha in exp.iter().take(PARITY) {
            let mut value = 0_u8;
            for &byte in &corrected {
                value = multiply(value, alpha) ^ byte;
            }
            if value != 0 {
                return None;
            }
        }
        Some(corrected)
    }

    fn xor_polynomials(a: &[u8], b: &[u8]) -> Vec<u8> {
        let mut out = vec![0_u8; a.len().max(b.len())];
        for (index, &value) in a.iter().enumerate() {
            out[index] ^= value;
        }
        for (index, &value) in b.iter().enumerate() {
            out[index] ^= value;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vector() -> Vec<f32> {
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        (0..VECTOR_WIDTH)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f64 / f64::from(1 << 24) - 0.5) as f32 * 3.0
            })
            .collect()
    }

    #[test]
    fn pristine_mosaic_round_trips_bit_exact() {
        let vector = test_vector();
        let pixels = render_mosaic_pixels("Jeff's Voice ✓", &vector);
        let (name, decoded) = decode(&pixels, CARD_SIZE, CARD_SIZE).expect("decodes");
        assert_eq!(name, "Jeff's Voice ✓");
        assert_eq!(decoded, vector);
    }

    /// Pinned to the Swift encoder's output, verified byte-identical on 2026-08-10
    /// (`cmp` over the full 3,145,728-byte mosaic for this exact input). If this test
    /// fails, the two implementations have DIVERGED: phones and the CLI would emit
    /// different cards for the same voice. Fix the drift; do not re-pin casually.
    #[test]
    fn the_mosaic_matches_the_swift_encoder_bit_for_bit() {
        let pixels = render_mosaic_pixels("Jeff's Voice ✓", &test_vector());
        assert_eq!(
            crc32(&pixels),
            0x0645_8419,
            "mosaic bytes drifted from the iOS encoder"
        );
    }

    #[test]
    fn an_all_zero_vector_still_works() {
        let vector = vec![0.0_f32; VECTOR_WIDTH];
        let pixels = render_mosaic_pixels("zero", &vector);
        let (name, decoded) = decode(&pixels, CARD_SIZE, CARD_SIZE).expect("decodes");
        assert_eq!(name, "zero");
        assert!(decoded.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn error_correction_survives_130_destroyed_cells() {
        let vector = test_vector();
        let mut pixels = render_mosaic_pixels("damage", &vector);
        let mut rng: u64 = 7;
        for _ in 0..130 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let row = 12 + (rng % 120) as usize;
            let column = 12 + ((rng >> 32) % 120) as usize;
            let wrong = LEVELS[(rng % 4) as usize];
            for y in 0..CELL_PX {
                for x in 0..CELL_PX {
                    let at = ((REGION_ORIGIN + row * CELL_PX + y) * CARD_SIZE
                        + REGION_ORIGIN
                        + column * CELL_PX
                        + x)
                        * 3;
                    pixels[at..at + 3].copy_from_slice(&wrong);
                }
            }
        }
        let (name, decoded) = decode(&pixels, CARD_SIZE, CARD_SIZE).expect("decodes");
        assert_eq!(name, "damage");
        assert_eq!(decoded, vector);
    }

    #[test]
    fn a_downscaled_copy_still_decodes() {
        // 0.6× box downscale approximates a screenshot resample well enough to
        // exercise finder registration and the calibration fit at small cells.
        let vector = test_vector();
        let pixels = render_mosaic_pixels("small", &vector);
        let out = (CARD_SIZE as f64 * 0.6) as usize;
        let mut scaled = vec![0_u8; out * out * 3];
        for y in 0..out {
            for x in 0..out {
                let sy = y * CARD_SIZE / out;
                let sx = x * CARD_SIZE / out;
                let mut sums = [0_u32; 3];
                let mut count = 0_u32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let yy = (sy + dy).min(CARD_SIZE - 1);
                        let xx = (sx + dx).min(CARD_SIZE - 1);
                        let at = (yy * CARD_SIZE + xx) * 3;
                        for c in 0..3 {
                            sums[c] += u32::from(pixels[at + c]);
                        }
                        count += 1;
                    }
                }
                let at = (y * out + x) * 3;
                for c in 0..3 {
                    scaled[at + c] = (sums[c] / count) as u8;
                }
            }
        }
        let (name, decoded) = decode(&scaled, out, out).expect("decodes at 0.6 scale");
        assert_eq!(name, "small");
        assert_eq!(decoded, vector);
    }

    #[test]
    fn random_noise_is_rejected_not_misread() {
        let mut noise = vec![0_u8; 800 * 600 * 3];
        let mut state: u64 = 42;
        for byte in &mut noise {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = (state >> 32) as u8;
        }
        assert!(decode(&noise, 800, 600).is_none());
        assert!(
            decode(&[], usize::MAX, usize::MAX).is_none(),
            "hostile dimensions must be rejected before size arithmetic wraps"
        );
    }

    #[test]
    fn the_lossless_chunk_round_trips_through_a_minimal_png() {
        // A syntactically valid minimal PNG: signature + IHDR + IEND (no image data
        // needed — the chunk walker only cares about chunk framing).
        let mut png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let ihdr_data = [0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0];
        png.extend_from_slice(&(ihdr_data.len() as u32).to_be_bytes());
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&ihdr_data);
        let mut crc_input = b"IHDR".to_vec();
        crc_input.extend_from_slice(&ihdr_data);
        png.extend_from_slice(&crc32(&crc_input).to_be_bytes());
        png.extend_from_slice(&0_u32.to_be_bytes());
        png.extend_from_slice(b"IEND");
        png.extend_from_slice(&crc32(b"IEND").to_be_bytes());

        let vector = test_vector();
        let carrying = embed_chunk("Chunk Voice", &vector, &png).expect("embeds");
        let (name, decoded) = decode_chunk(&carrying).expect("chunk decodes");
        assert_eq!(name, "Chunk Voice");
        assert_eq!(decoded, vector);
        assert!(decode_chunk(&png).is_none(), "plain PNG carries no voice");

        assert!(
            embed_chunk("short", &vector[..VECTOR_WIDTH - 1], &png).is_none(),
            "a malformed-width vector must not create an unreadable card"
        );
        let mut not_iend = png.clone();
        let iend_type = not_iend.len() - 8..not_iend.len() - 4;
        not_iend[iend_type].copy_from_slice(b"JUNK");
        assert!(
            embed_chunk("voice", &vector, &not_iend).is_none(),
            "a signature-shaped buffer without a terminal IEND is not a PNG"
        );
    }
}
