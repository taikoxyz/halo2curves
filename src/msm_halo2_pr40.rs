//! This module implements a fast method for multi-scalar multiplications.
//!
//! Generally it works like pippenger with a couple of tricks to make if faster.
//!
//! - First the coefficients are split into two parts (using the endomorphism). This
//! reduces the number of rounds by half, but doubles the number of points per round.
//! This is faster because half the rounds also means only needing to add all bucket
//! results together half the number of times.
//!
//! - The coefficients are then sorted in buckets. Instead of using
//! the binary representation to do this, a signed digit representation is
//! used instead (WNAF). Unfortunately this doesn't directly reduce the number of additions
//! in a bucket, but it does reduce the number of buckets in half, which halves the
//! work required to accumulate the results of the buckets.
//!
//! - We then need to add all the points in each bucket together. To do this
//! the affine addition formulas are used. If the points are linearly independent the
//! incomplete version of the formula can be used which is quite a bit faster than
//! the full one because some checks can be skipped.
//! The affine formula is only fast if a lot of independent points can be added
//! together. This is because to get the actual result of an addition an inversion is
//! needed which is very expensive, but it's cheap when batched inversion can be used.
//! So the idea is to add a lot of pairs of points together using a single batched inversion.
//! We then have the results of all those additions, and can do a new batch of additions on those
//! results. This process is repeated as many times as needed until all additions for each bucket
//! are done. To do this efficiently we first build up an addition tree that sets everything
//! up correctly per round. We then process each addition tree per round.

use core::slice;
use crate::multicore;
use ark_std::{end_timer, start_timer};
use ff::PrimeField;
use group::Group;
use crate::arithmetic::CurveAffineExt;

fn num_bits(value: usize) -> usize {
    (0usize.leading_zeros() - value.leading_zeros()) as usize
}

fn div_up(a: usize, b: usize) -> usize {
    (a + (b - 1)) / b
}

fn get_wnaf_size_bits(num_bits: usize, w: usize) -> usize {
    div_up(num_bits, w)
}

fn get_wnaf_size<C: CurveAffineExt>(w: usize) -> usize {
    get_wnaf_size_bits(div_up(C::Scalar::NUM_BITS as usize, 2), w)
}

fn get_num_rounds<C: CurveAffineExt>(c: usize) -> usize {
    get_wnaf_size::<C>(c + 1)
}

fn get_num_buckets(c: usize) -> usize {
    (1 << c) + 1
}

fn get_max_tree_size(num_points: usize, c: usize) -> usize {
    num_points * 2 + get_num_buckets(c)
}

fn get_num_tree_levels(num_points: usize) -> usize {
    1 + num_bits(num_points - 1)
}

/// Returns the signed digit representation of value with the specified window size.
/// The result is written to the wnaf slice with the specified stride.
fn get_wnaf(value: u128, w: usize, num_rounds: usize, wnaf: &mut [u32], stride: usize) {
    fn get_bits_at(v: u128, pos: usize, num: usize) -> usize {
        ((v >> pos) & ((1 << num) - 1)) as usize
    }

    let mut borrow = 0;
    let max = 1 << (w - 1);
    for idx in 0..num_rounds {
        let b = get_bits_at(value, idx * w, w) + borrow;
        if b >= max {
            // Set the highest bit to 1 to represent a negative value.
            // This way the lower bits directly represent the bucket index.
            wnaf[idx * stride] = (0x80000000 | ((1 << w) - b)) as u32;
            borrow = 1;
        } else {
            wnaf[idx * stride] = b as u32;
            borrow = 0;
        }
    }
    assert_eq!(borrow, 0);
}

/// Returns the best bucket width for the given number of points.
fn get_best_c(num_points: usize) -> usize {
    if num_points >= 262144 {
        15
    } else if num_points >= 65536 {
        12
    } else if num_points >= 16384 {
        11
    } else if num_points >= 8192 {
        10
    } else if num_points >= 1024 {
        9
    } else {
        7
    }
}

/// MultiExp
#[derive(Clone, Debug, Default)]
pub struct MultiExp<C: CurveAffineExt> {
    /// The bases
    bases: Vec<C>,
}

/// MultiExp context object
#[derive(Clone, Debug, Default)]
pub struct MultiExpContext<C: CurveAffineExt> {
    /// Memory to store the points in the addition tree
    points: Vec<C>,
    /// Memory to store wnafs
    wnafs: Vec<u32>,
    /// Memory split up between rounds
    rounds: SharedRoundData,
}

/// SharedRoundData
#[derive(Clone, Debug, Default)]
struct SharedRoundData {
    /// Memory to store bucket sizes
    bucket_sizes: Vec<usize>,
    /// Memory to store bucket offsets
    bucket_offsets: Vec<usize>,
    /// Memory to store the point data
    point_data: Vec<u32>,
    /// Memory to store the output indices
    output_indices: Vec<u32>,
    /// Memory to store the base positions (on the first level)
    base_positions: Vec<u32>,
    /// Memory to store the scatter maps
    scatter_map: Vec<ScatterData>,
}

/// RoundData
#[derive(Debug, Default)]
struct RoundData<'a> {
    /// Number of levels in the addition tree
    pub num_levels: usize,
    /// The length of each level in the addition tree
    pub level_sizes: Vec<usize>,
    /// The offset to each level in the addition tree
    pub level_offset: Vec<usize>,
    /// The size of each bucket
    pub bucket_sizes: &'a mut [usize],
    /// The offset of each bucket
    pub bucket_offsets: &'a mut [usize],
    /// The point to use for each coefficient
    pub point_data: &'a mut [u32],
    /// The output index in the point array for each pair addition
    pub output_indices: &'a mut [u32],
    /// The point to use on the first level in the addition tree
    pub base_positions: &'a mut [u32],
    /// List of points that are scattered to the addition tree
    pub scatter_map: &'a mut [ScatterData],
    /// The length of scatter_map
    pub scatter_map_len: usize,
}

/// ScatterData
#[derive(Default, Debug, Clone)]
struct ScatterData {
    /// The position in the addition tree to store the point
    pub position: u32,
    /// The point to write
    pub point_data: u32,
}

impl<C: CurveAffineExt> MultiExp<C> {
    /// Create a new MultiExp instance with the specified bases
    pub fn new(bases: &[C]) -> Self {
        let mut endo_bases = vec![C::identity(); bases.len() * 2];

        // Generate the endomorphism bases
        let num_threads = multicore::current_num_threads();
        multicore::scope(|scope| {
            let num_points_per_thread = div_up(bases.len(), num_threads);
            for (endo_bases, bases) in endo_bases
                .chunks_mut(num_points_per_thread * 2)
                .zip(bases.chunks(num_points_per_thread))
            {
                scope.spawn(move |_| {
                    for (idx, base) in bases.iter().enumerate() {
                        endo_bases[idx * 2] = *base;
                        endo_bases[idx * 2 + 1] = C::endo(base);
                    }
                });
            }
        });

        Self { bases: endo_bases }
    }

    /// Performs a multi-exponentiation operation.
    /// Set complete to true if the bases are not guaranteed linearly independent.
    pub fn evaluate(
        &self,
        ctx: &mut MultiExpContext<C>,
        coeffs: &[C::Scalar],
        complete: bool,
    ) -> C::Curve {
        self.evaluate_with(ctx, coeffs, complete, get_best_c(coeffs.len()))
    }

    /// Performs a multi-exponentiation operation with the given bucket width.
    /// Set complete to true if the bases are not guaranteed linearly independent.
    pub fn evaluate_with(
        &self,
        ctx: &mut MultiExpContext<C>,
        coeffs: &[C::Scalar],
        complete: bool,
        c: usize,
    ) -> C::Curve {
        assert!(coeffs.len() * 2 <= self.bases.len());
        assert!(c >= 4);

        // Allocate more memory if required
        ctx.allocate(coeffs.len(), c);

        // Get the data for each round
        let mut rounds = ctx.rounds.get_rounds::<C>(coeffs.len(), c);

        // Get the bases for the coefficients
        let bases = &self.bases[..coeffs.len() * 2];

        let num_threads = multicore::current_num_threads();
        let _timer_desc =  format!("msm {} ({}) ({} threads)", coeffs.len(), c, num_threads);
        let start = start_timer!(|| _timer_desc);
        // if coeffs.len() >= 16 {
        let num_points = coeffs.len() * 2;
        let w = c + 1;
        let num_rounds = get_num_rounds::<C>(c);

        // Prepare WNAFs of all coefficients for all rounds
        calculate_wnafs::<C>(coeffs, &mut ctx.wnafs, c);
        // Sort WNAFs into buckets for all rounds
        sort::<C>(&mut ctx.wnafs[0..num_rounds * num_points], &mut rounds, c);
        // Calculate addition trees for all rounds
        create_addition_trees(&mut rounds);

        // Now process each round individually
        let mut partials = vec![C::Curve::identity(); num_rounds];
        for (round, acc) in rounds.iter().zip(partials.iter_mut()) {
            // Scatter the odd points in the odd length buckets to the addition tree
            do_point_scatter::<C>(round, bases, &mut ctx.points);
            // Do all bucket additions
            do_batch_additions::<C>(round, bases, &mut ctx.points, complete);
            // Get the final result of the round
            *acc = accumulate_buckets::<C>(round, &mut ctx.points, c);
        }

        // Accumulate round results
        let res = partials
            .iter()
            .rev()
            .skip(1)
            .fold(partials[num_rounds - 1], |acc, partial| {
                let mut res = acc;
                for _ in 0..w {
                    res = res.double();
                }
                res + partial
            });
        end_timer!(start);

        res
        // } else {
        //     // Just do a naive msm
        //     let mut acc = C::Curve::identity();
        //     for (idx, coeff) in coeffs.iter().enumerate() {
        //         // Skip over endomorphism bases
        //         acc += bases[idx * 2] * coeff;
        //     }
        //     end_timer!(start);
        //     acc
        // }
    }
}

impl<C: CurveAffineExt> MultiExpContext<C> {
    /// Allocate memory for the evalution
    pub fn allocate(&mut self, num_points: usize, c: usize) {
        let num_points = num_points * 2;
        let num_buckets = get_num_buckets(c);
        let num_rounds = get_num_rounds::<C>(c);
        let tree_size = get_max_tree_size(num_points, c);
        let num_points_total = num_rounds * num_points;
        let num_buckets_total = num_rounds * num_buckets;
        let tree_size_total = num_rounds * tree_size;

        // Allocate memory when necessary
        if self.points.len() < tree_size {
            self.points.resize(tree_size, C::identity());
        }
        if self.wnafs.len() < num_points_total {
            self.wnafs.resize(num_points_total, 0u32);
        }
        if self.rounds.bucket_sizes.len() < num_buckets_total {
            self.rounds.bucket_sizes.resize(num_buckets_total, 0usize);
        }
        if self.rounds.bucket_offsets.len() < num_buckets_total {
            self.rounds.bucket_offsets.resize(num_buckets_total, 0usize);
        }
        if self.rounds.point_data.len() < num_points_total {
            self.rounds.point_data.resize(num_points_total, 0u32);
        }
        if self.rounds.output_indices.len() < tree_size_total / 2 {
            self.rounds.output_indices.resize(tree_size_total / 2, 0u32);
        }
        if self.rounds.base_positions.len() < num_points_total {
            self.rounds.base_positions.resize(num_points_total, 0u32);
        }
        if self.rounds.scatter_map.len() < num_buckets_total {
            self.rounds
                .scatter_map
                .resize(num_buckets_total, ScatterData::default());
        }
    }
}

impl SharedRoundData {
    fn get_rounds<C: CurveAffineExt>(&mut self, num_points: usize, c: usize) -> Vec<RoundData> {
        let num_points = num_points * 2;
        let num_buckets = get_num_buckets(c);
        let num_rounds = get_num_rounds::<C>(c);
        let tree_size = num_points * 2 + num_buckets;

        let mut bucket_sizes_rest = self.bucket_sizes.as_mut_slice();
        let mut bucket_offsets_rest = self.bucket_offsets.as_mut_slice();
        let mut point_data_rest = self.point_data.as_mut_slice();
        let mut output_indices_rest = self.output_indices.as_mut_slice();
        let mut base_positions_rest = self.base_positions.as_mut_slice();
        let mut scatter_map_rest = self.scatter_map.as_mut_slice();

        // Use the allocated memory above to init the memory used for each round.
        // This way the we don't need to reallocate memory for each msm with
        // a different configuration (different number of points or different bucket width)
        let mut rounds: Vec<RoundData> = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let (bucket_sizes, rest) = bucket_sizes_rest.split_at_mut(num_buckets);
            bucket_sizes_rest = rest;
            let (bucket_offsets, rest) = bucket_offsets_rest.split_at_mut(num_buckets);
            bucket_offsets_rest = rest;
            let (point_data, rest) = point_data_rest.split_at_mut(num_points);
            point_data_rest = rest;
            let (output_indices, rest) = output_indices_rest.split_at_mut(tree_size / 2);
            output_indices_rest = rest;
            let (base_positions, rest) = base_positions_rest.split_at_mut(num_points);
            base_positions_rest = rest;
            let (scatter_map, rest) = scatter_map_rest.split_at_mut(num_buckets);
            scatter_map_rest = rest;

            rounds.push(RoundData {
                num_levels: 0,
                level_sizes: vec![],
                level_offset: vec![],
                bucket_sizes,
                bucket_offsets,
                point_data,
                output_indices,
                base_positions,
                scatter_map,
                scatter_map_len: 0,
            });
        }
        rounds
    }
}

#[derive(Clone, Copy)]
struct ThreadBox<T>(*mut T, usize);
#[allow(unsafe_code)]
unsafe impl<T> Send for ThreadBox<T> {}
#[allow(unsafe_code)]
unsafe impl<T> Sync for ThreadBox<T> {}

/// Wraps a mutable slice so it can be passed into a thread without
/// hard to fix borrow checks caused by difficult data access patterns.
impl<T> ThreadBox<T> {
    fn wrap(data: &mut [T]) -> Self {
        Self(data.as_mut_ptr(), data.len())
    }

    fn unwrap(&mut self) -> &mut [T] {
        #[allow(unsafe_code)]
        unsafe {
            slice::from_raw_parts_mut(self.0, self.1)
        }
    }
}

fn calculate_wnafs<C: CurveAffineExt>(coeffs: &[C::Scalar], wnafs: &mut [u32], c: usize) {
    let num_threads = multicore::current_num_threads();
    let num_points = coeffs.len() * 2;
    let num_rounds = get_num_rounds::<C>(c);
    let w = c + 1;

    let start = start_timer!(|| "calculate wnafs".to_string());
    let mut wnafs_box = ThreadBox::wrap(wnafs);
    let chunk_size = div_up(coeffs.len(), num_threads);
    multicore::scope(|scope| {
        for (thread_idx, coeffs) in coeffs.chunks(chunk_size).enumerate() {
            scope.spawn(move |_| {
                let wnafs = &mut wnafs_box.unwrap()[thread_idx * chunk_size * 2..];
                for (idx, coeff) in coeffs.iter().enumerate() {
                    let (p0, _, p1, _) = C::decompose_scalar(coeff);
                    get_wnaf(p0, w, num_rounds, &mut wnafs[idx * 2..], num_points);
                    get_wnaf(p1, w, num_rounds, &mut wnafs[idx * 2 + 1..], num_points);
                }
            });
        }
    });
    end_timer!(start);
}

fn radix_sort(wnafs: &mut [u32], round: &mut RoundData) {
    let bucket_sizes = &mut round.bucket_sizes;
    let bucket_offsets = &mut round.bucket_offsets;

    // Calculate bucket sizes, first resetting all sizes to 0
    bucket_sizes.fill_with(|| 0);
    for wnaf in wnafs.iter() {
        bucket_sizes[(wnaf & 0x7FFFFFFF) as usize] += 1;
    }

    // Calculate bucket offsets
    let mut offset = 0;
    let mut max_bucket_size = 0;
    bucket_offsets[0] = offset;
    offset += bucket_sizes[0];
    for (bucket_offset, bucket_size) in bucket_offsets
        .iter_mut()
        .skip(1)
        .zip(bucket_sizes.iter().skip(1))
    {
        *bucket_offset = offset;
        offset += bucket_size;
        max_bucket_size = max_bucket_size.max(*bucket_size);
    }
    // Number of levels we need in our addition tree
    round.num_levels = get_num_tree_levels(max_bucket_size);

    // Fill in point data grouped in buckets
    let point_data = &mut round.point_data;
    for (idx, wnaf) in wnafs.iter().enumerate() {
        let bucket_idx = (wnaf & 0x7FFFFFFF) as usize;
        point_data[bucket_offsets[bucket_idx]] = (wnaf & 0x80000000) | (idx as u32);
        bucket_offsets[bucket_idx] += 1;
    }
}

/// Sorts the points so they are grouped per bucket
fn sort<C: CurveAffineExt>(wnafs: &mut [u32], rounds: &mut [RoundData], c: usize) {
    let num_rounds = get_num_rounds::<C>(c);
    let num_points = wnafs.len() / num_rounds;

    // Sort per bucket for each round separately
    let start = start_timer!(|| "radix sort".to_string());
    multicore::scope(|scope| {
        for (round, wnafs) in rounds.chunks_mut(1).zip(wnafs.chunks_mut(num_points)) {
            scope.spawn(move |_| {
                radix_sort(wnafs, &mut round[0]);
            });
        }
    });
    end_timer!(start);
}

/// Creates the addition tree.
/// When PREPROCESS is false we just calculate the size of each level.
/// All points in a bucket need to be added to each other. Because the affine formulas
/// are used we need to add points together in pairs. So we have to make sure that
/// on each level we have an even number of points for each level. Odd points are
/// added to lower levels where the length of the addition results is odd (which then
/// makes the length even).
fn process_addition_tree<const PREPROCESS: bool>(round: &mut RoundData) {
    let num_levels = round.num_levels;
    let bucket_sizes = &round.bucket_sizes;
    let point_data = &round.point_data;

    let mut level_sizes = vec![0usize; num_levels];
    let mut level_offset = vec![0usize; num_levels];
    let output_indices = &mut round.output_indices;
    let scatter_map = &mut round.scatter_map;
    let base_positions = &mut round.base_positions;
    let mut point_idx = bucket_sizes[0];

    if !PREPROCESS {
        // Set the offsets to the different levels in the tree
        level_offset[0] = 0;
        for idx in 1..level_offset.len() {
            level_offset[idx] = level_offset[idx - 1] + round.level_sizes[idx - 1];
        }
    }

    // The level where all bucket results will be stored
    let bucket_level = num_levels - 1;

    // Run over all buckets
    for bucket_size in bucket_sizes.iter().skip(1) {
        let mut size = *bucket_size;
        if size == 0 {
            level_sizes[bucket_level] += 1;
        } else if size == 1 {
            if !PREPROCESS {
                scatter_map[round.scatter_map_len] = ScatterData {
                    position: (level_offset[bucket_level] + level_sizes[bucket_level]) as u32,
                    point_data: point_data[point_idx],
                };
                round.scatter_map_len += 1;
                point_idx += 1;
            }
            level_sizes[bucket_level] += 1;
        } else {
            #[derive(Clone, Copy, PartialEq)]
            enum State {
                Even,
                OddPoint(usize),
                OddResult(usize),
            }
            let mut state = State::Even;
            let num_levels_bucket = get_num_tree_levels(size);

            let mut start_level_size = level_sizes[0];
            for level in 0..num_levels_bucket - 1 {
                let is_level_odd = size & 1;
                let first_level = level == 0;
                let last_level = level == num_levels_bucket - 2;

                // If this level has odd size we have to handle it
                if is_level_odd == 1 {
                    // If we already have a point saved from a previous odd level, use it
                    // to make the current level even
                    if state != State::Even {
                        if !PREPROCESS {
                            let pos = (level_offset[level] + level_sizes[level]) as u32;
                            match state {
                                State::OddPoint(point_idx) => {
                                    scatter_map[round.scatter_map_len] = ScatterData {
                                        position: pos,
                                        point_data: point_data[point_idx],
                                    };
                                    round.scatter_map_len += 1;
                                }
                                State::OddResult(output_idx) => {
                                    output_indices[output_idx] = pos;
                                }
                                _ => unreachable!(),
                            };
                        }
                        level_sizes[level] += 1;
                        size += 1;
                        state = State::Even;
                    } else {
                        // Not odd yet, so the state is now odd
                        // Store the point we have to add later
                        if !PREPROCESS {
                            if first_level {
                                state = State::OddPoint(point_idx + size - 1);
                            } else {
                                state = State::OddResult(
                                    (level_offset[level] + level_sizes[level] + size) >> 1,
                                );
                            }
                        } else {
                            // Just mark it as odd, we won't use the actual value anywhere
                            state = State::OddPoint(0);
                        }
                        size -= 1;
                    }
                }

                // Write initial points on the first level
                if first_level {
                    if !PREPROCESS {
                        // Just write all points (except the odd size one)
                        let pos = level_offset[level] + level_sizes[level];
                        base_positions[pos..pos + size]
                            .copy_from_slice(&point_data[point_idx..point_idx + size]);
                        point_idx += size + is_level_odd;
                    }
                    level_sizes[level] += size;
                }

                // Write output indices
                // If the next level would be odd, we have to make it even
                // by writing the last result of this level to the next level that is odd
                // (unless we are writing the final result to the bucket level)
                let next_level_size = size >> 1;
                let next_level_odd = next_level_size & 1 == 1;
                let redirect =
                    if next_level_odd && state == State::Even && level < num_levels_bucket - 2 {
                        1usize
                    } else {
                        0usize
                    };
                // An addition works on two points and has one result, so this takes only half the size
                let sub_level_offset = (level_offset[level] + start_level_size) >> 1;
                // Cache the start position of the next level
                start_level_size = level_sizes[level + 1];
                if !PREPROCESS {
                    // Write the destination positions of the addition results in the tree
                    let dst_pos = level_offset[level + 1] + level_sizes[level + 1];
                    for (idx, output_index) in output_indices
                        [sub_level_offset..sub_level_offset + next_level_size]
                        .iter_mut()
                        .enumerate()
                    {
                        *output_index = (dst_pos + idx) as u32;
                    }
                }
                if last_level {
                    // The result of the last addition for this bucket is written
                    // to the last level (so all bucket results are nicely after each other).
                    // Overwrite the output locations of the last result here.
                    if !PREPROCESS {
                        output_indices[sub_level_offset] =
                            (level_offset[bucket_level] + level_sizes[bucket_level]) as u32;
                    }
                    level_sizes[bucket_level] += 1;
                } else {
                    // Update the sizes
                    level_sizes[level + 1] += next_level_size - redirect;
                    size -= redirect;
                    // We have to redirect the last result to a lower level
                    if redirect == 1 {
                        state = State::OddResult(sub_level_offset + next_level_size - 1);
                    }
                }

                // We added pairs of points together so the next level has half the size
                size >>= 1;
            }
        }
    }

    // Store the tree level data
    round.level_sizes = level_sizes;
    round.level_offset = level_offset;
}

/// The affine formula is only efficient for independent point additions
/// (using the result of the addition requires an inversion which needs to be avoided as much as possible).
/// And so we try to add as many points together on each level of the tree, writing the result of the addition
/// to a lower level. Each level thus contains independent point additions, with only requiring a single inversion
/// per level in the tree.
fn create_addition_trees(rounds: &mut [RoundData]) {
    let start = start_timer!(|| "create addition trees".to_string());
    multicore::scope(|scope| {
        for round in rounds.chunks_mut(1) {
            scope.spawn(move |_| {
                // Collect tree levels sizes
                process_addition_tree::<true>(&mut round[0]);
                // Construct the tree
                process_addition_tree::<false>(&mut round[0]);
            });
        }
    });
    end_timer!(start);
}

/// Here we write the odd points in odd length buckets (the other points are loaded on the fly).
/// This will do random reads AND random writes, which is normally terrible for performance.
/// Luckily this doesn't really matter because we only have to write at most num_buckets points.
fn do_point_scatter<C: CurveAffineExt>(round: &RoundData, bases: &[C], points: &mut [C]) {
    let num_threads = multicore::current_num_threads();
    let scatter_map = &round.scatter_map[..round.scatter_map_len];
    let mut points_box = ThreadBox::wrap(points);
    let start = start_timer!(|| "point scatter".to_string());
    if !scatter_map.is_empty() {
        multicore::scope(|scope| {
            let num_copies_per_thread = div_up(scatter_map.len(), num_threads);
            for scatter_map in scatter_map.chunks(num_copies_per_thread) {
                scope.spawn(move |_| {
                    let points = points_box.unwrap();
                    for scatter_data in scatter_map.iter() {
                        let target_idx = scatter_data.position as usize;
                        let negate = scatter_data.point_data & 0x80000000 != 0;
                        let base_idx = (scatter_data.point_data & 0x7FFFFFFF) as usize;
                        if negate {
                            points[target_idx] = bases[base_idx].neg();
                        } else {
                            points[target_idx] = bases[base_idx];
                        }
                    }
                });
            }
        });
    }
    end_timer!(start);
}

/// Finally do all additions using the addition tree we've setup.
fn do_batch_additions<C: CurveAffineExt>(
    round: &RoundData,
    bases: &[C],
    points: &mut [C],
    complete: bool,
) {
    let num_threads = multicore::current_num_threads();

    let num_levels = round.num_levels;
    let level_counter = &round.level_sizes;
    let level_offset = &round.level_offset;
    let output_indices = &round.output_indices;
    let base_positions = &round.base_positions;
    let mut points_box = ThreadBox::wrap(points);

    let start = start_timer!(|| "batch additions".to_string());
    for i in 0..num_levels - 1 {
        let start = level_offset[i];
        let num_points = level_counter[i];
        multicore::scope(|scope| {
            // We have to make sure we have an even amount here so we don't split within a pair
            let num_points_per_thread = div_up(num_points / 2, num_threads) * 2;
            for thread_idx in 0..num_threads {
                scope.spawn(move |_| {
                    let points = points_box.unwrap();

                    let thread_start = thread_idx * num_points_per_thread;
                    let mut thread_num_points = num_points_per_thread;

                    if thread_start < num_points {
                        if thread_start + thread_num_points > num_points {
                            thread_num_points = num_points - thread_start;
                        }

                        let points = &mut points[(start + thread_start)..];
                        let output_indices = &output_indices[(start + thread_start) / 2..];
                        let offset = start + thread_start;
                        if i == 0 {
                            let base_positions = &base_positions[(start + thread_start)..];
                            if complete {
                                C::batch_add::<true, true>(
                                    points,
                                    output_indices,
                                    thread_num_points,
                                    offset,
                                    bases,
                                    base_positions,
                                );
                            } else {
                                C::batch_add::<false, true>(
                                    points,
                                    output_indices,
                                    thread_num_points,
                                    offset,
                                    bases,
                                    base_positions,
                                );
                            }
                        } else {
                            #[allow(collapsible-else-if)]
                            if complete {
                                C::batch_add::<true, false>(
                                    points,
                                    output_indices,
                                    thread_num_points,
                                    offset,
                                    &[],
                                    &[],
                                );
                            } else {
                                C::batch_add::<false, false>(
                                    points,
                                    output_indices,
                                    thread_num_points,
                                    offset,
                                    &[],
                                    &[],
                                );
                            }
                        }
                    }
                });
            }
        });
    }
    end_timer!(start);
}

/// Accumulate all bucket results to get the result of the round
fn accumulate_buckets<C: CurveAffineExt>(
    round: &RoundData,
    points: &mut [C],
    c: usize,
) -> C::Curve {
    let num_threads = multicore::current_num_threads();
    let num_buckets = get_num_buckets(c);

    let num_levels = round.num_levels;
    let bucket_sizes = &round.bucket_sizes;
    let level_offset = &round.level_offset;

    let start_time = start_timer!(|| "accumulate buckets".to_string());
    let start = level_offset[num_levels - 1];
    let buckets = &mut points[start..(start + num_buckets)];
    let mut results: Vec<C::Curve> = vec![C::Curve::identity(); num_threads];
    multicore::scope(|scope| {
        let chunk_size = num_buckets / num_threads;
        for (thread_idx, ((bucket_sizes, buckets), result)) in bucket_sizes[1..]
            .chunks(chunk_size)
            .zip(buckets[..].chunks_mut(chunk_size))
            .zip(results.chunks_mut(1))
            .enumerate()
        {
            scope.spawn(move |_| {
                // Accumulate all bucket results
                let num_buckets_thread = bucket_sizes.len();
                let mut acc = C::Curve::identity();
                let mut running_sum = C::Curve::identity();
                for b in (0..num_buckets_thread).rev() {
                    if bucket_sizes[b] > 0 {
                        running_sum = running_sum + buckets[b];
                    }
                    acc = acc + &running_sum;
                }

                // Each thread started at a different bucket location
                // so correct for that here
                let bucket_start = thread_idx * chunk_size;
                let num_bits = num_bits(bucket_start);
                let mut accumulator = C::Curve::identity();
                for idx in (0..num_bits).rev() {
                    accumulator = accumulator.double();
                    if (bucket_start >> idx) & 1 != 0 {
                        accumulator += running_sum;
                    }
                }
                acc += accumulator;

                // Store the result
                result[0] = acc;
            });
        }
    });
    end_timer!(start_time);

    // Add the results of all threads together
    results
        .iter()
        .fold(C::Curve::identity(), |acc, result| acc + result)
}
