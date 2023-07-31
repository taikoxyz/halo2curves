use crate::curve_jac_ext::CurveJacExt;
use group::{ff::PrimeField, Group as _};
use rayon::{
    prelude::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

/// Performs a multi-exponentiation operation.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
pub fn best_multiexp<C: CurveJacExt>(
    coeffs: &[C::Scalar],
    bases: &[C],
) -> C::ExtendedJacobianCoordinates {
    let n = coeffs.len();
    assert_eq!(n, bases.len());

    let num_threads = num_cpus::get();
    if n > num_threads && n > 32 {
        let chunk = n / num_threads;
        let results: Vec<C::ExtendedJacobianCoordinates> = coeffs
            .par_chunks(chunk)
            .zip(bases.par_chunks(chunk))
            .map(|(coeffs, bases)| multiexp_serial(coeffs, bases))
            .collect();
        results.iter().fold(C::jac_ext_identity(), |a, b| a + *b)
    } else {
        small_multiexp(coeffs, bases)
    }
}

pub(crate) fn multiexp_serial<C: CurveJacExt>(coeffs: &[C::Scalar], bases: &[C]) -> C::ExtendedJacobianCoordinates {
    let c = if bases.len() < 4 {
        1
    } else if bases.len() < 32 {
        3
    } else {
        (bases.len() as f64).ln().ceil() as usize
    };
    let mut buckets: Vec<Vec<Bucket<C>>> = vec![vec![Bucket::None; (1 << c) - 1]; (256 / c) + 1];

    buckets
        .iter_mut()
        .enumerate()
        .rev()
        .map(|(i, bucket)| {
            for (coeff, base) in coeffs.iter().zip(bases.iter()) {
                let seg = get_at::<C::Scalar>(i, c, &coeff.to_repr());
                if seg != 0 {
                    bucket[seg - 1].add_assign(base);
                }
            }
            bucket
        })
        .fold(C::jac_ext_identity(), |mut sum, bucket| {
            for _ in 0..c {
                sum = sum.double();
            }
            // Summation by parts
            // e.g. 3a + 2b + 1c = a +
            //                    (a) + b +
            //                    ((a) + b) + c
            let mut running_sum = C::jac_ext_identity();
            bucket.iter().rev().for_each(|exp| {
                running_sum = exp.add(running_sum);
                sum += &running_sum;
            });
            sum
        })
}

/// Performs a small multi-exponentiation operation.
/// Uses the double-and-add algorithm with doublings shared across points.
pub(crate) fn small_multiexp<C: CurveJacExt>(coeffs: &[C::Scalar], bases: &[C]) -> C::ExtendedJacobianCoordinates {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();
    let mut acc = C::jac_ext_identity();

    // for byte idx
    for byte_idx in (0..32).rev() {
        // for bit idx
        for bit_idx in (0..8).rev() {
            acc = acc.double();
            // for each coeff
            for coeff_idx in 0..coeffs.len() {
                let byte = coeffs[coeff_idx].as_ref()[byte_idx];
                if ((byte >> bit_idx) & 1) != 0 {
                    acc += bases[coeff_idx];
                }
            }
        }
    }

    acc
}

#[derive(Clone, Copy)]
enum Bucket<C: CurveJacExt> {
    None,
    Affine(C),
    ExtJacobian(C::ExtendedJacobianCoordinates),
}

impl<C: CurveJacExt> Bucket<C> {
    fn add_assign(&mut self, other: &C) {
        *self = match *self {
            Bucket::None => {
                let tmp = *other;
                Bucket::Affine(tmp)},
            Bucket::Affine(a) => {
                let tmp = a + *other;
                Bucket::ExtJacobian(tmp.into())
            }
            Bucket::ExtJacobian(mut a) => {
                a += *other;
                Bucket::ExtJacobian(a)
            }
        }
    }

    fn add(self, mut other: C::ExtendedJacobianCoordinates) -> C::ExtendedJacobianCoordinates {
        match self {
            Bucket::None => C::jac_ext_identity() + other,
            Bucket::Affine(a) => {
                other += a;
                other
            }
            Bucket::ExtJacobian(a) => {
                other + a
            }
        }
    }
}

fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
    let skip_bits = segment * c;
    let skip_bytes = skip_bits / 8;

    if skip_bytes >= 32 {
        return 0;
    }

    let mut v = [0; 8];
    for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
        *v = *o;
    }

    let mut tmp = u64::from_le_bytes(v);
    tmp >>= skip_bits - (skip_bytes * 8);
    (tmp % (1 << c)) as usize
}

#[cfg(test)]
mod test {
    use std::iter::zip;

    use super::{multiexp_serial, small_multiexp};
    use crate::secp256k1::Fq as Scalar;
    use crate::secp256k1::Secp256k1;
    use crate::secp256k1::Secp256k1Affine;
    use crate::secp256k1::Secp256k1JacExt;
    use ff::Field;
    use group::Group;
    use halo2_proofs::poly::commitment::ParamsProver;
    use halo2_proofs::poly::ipa::commitment::ParamsIPA;
    use proptest::prelude::*;
    use rand_core::OsRng;

    fn arb_poly(k: usize, rng: OsRng) -> Vec<Scalar> {
        (0..(1 << k)).map(|_| Scalar::random(rng)).collect::<Vec<_>>()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_multiexp(k in 3usize..10) {
            let coeffs = arb_poly(k, OsRng);
            let params: ParamsIPA<Secp256k1Affine> = ParamsIPA::new(k as u32);
            let g_a = &mut params.get_g();
            let g_b = &mut params.get_g();

            let point_a: Secp256k1 = multiexp_serial(&coeffs, g_a).into();
            let point_b: Secp256k1 = small_multiexp(&coeffs, g_b).into();

            assert_eq!(point_a, point_b);
        }
    }

    /// Test 0 == 0 + 0.
    #[test]
    fn infinity_plus_infinity() {
        use group::Group;

        let infinity = Secp256k1JacExt::identity();
        let proj_inf: Secp256k1 = infinity.into();

        assert_eq!(infinity, infinity + infinity);
        assert_eq!(proj_inf, proj_inf + proj_inf);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        /// Test for all randomly selected a: 0 == a - a.
        #[test]
        fn a_minus_a_is_infinity(k in 3usize..10) {
            use group::Group;

            let params: ParamsIPA<Secp256k1Affine> = ParamsIPA::new(k as u32);
            let g_a = &mut params.get_g();

            let infinity = Secp256k1::identity();
            let points: Vec<Secp256k1JacExt> = g_a.iter().map(|x| x.into()).collect();
            let point_a: Secp256k1JacExt = points[0];
            let a_minus_a = point_a - point_a;

            assert_eq!(infinity, a_minus_a.into());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        /// Test: sum(Vec<JacExt>) == sum(Vec<Affine>).
        #[test]
        fn test_sum(k in 3usize..10) {
            let params: ParamsIPA<Secp256k1Affine> = ParamsIPA::new(k as u32);

            let points = &mut params.get_g();
            let ext_jac_points: Vec<Secp256k1JacExt> = points.iter().map(|x| x.into()).collect();

            let mut expected = Secp256k1::identity();
            for point in points.iter() {
                expected += point;
            }

            let actual: Secp256k1 = ext_jac_points.iter().sum::<Secp256k1JacExt>().into();

            assert_eq!(expected, actual);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        /// Test that for p: Affine and q: JacExt: p+p == q+q.
        #[test]
        fn test_double(k in 3usize..10) {
            let params: ParamsIPA<Secp256k1Affine> = ParamsIPA::new(k as u32);

            let points: &[Secp256k1Affine] = params.get_g();
            let ext_jac_points = points.iter().map(Secp256k1JacExt::from);

            let double_points = points.iter().map(|p| Secp256k1JacExt::from(p + p));
            let double_ext_jac_points = ext_jac_points.map(|p| p + p);

            for (expected, actual) in zip(double_points, double_ext_jac_points) {
                assert_eq!(expected, actual);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        /// Test that for p: JacExt, p + -(p) == 0.
        #[test]
        fn test_opposites(k in 3usize..10) {
            let params: ParamsIPA<Secp256k1Affine> = ParamsIPA::new(k as u32);

            let points: &[Secp256k1Affine] = params.get_g();

            let plus = points.iter().map(Secp256k1JacExt::from);
            let minus = plus.clone().map(std::ops::Neg::neg);

            for (p,m) in zip(plus, minus) {
                assert_eq!(Secp256k1::from(p+m), Secp256k1::identity());
            }
        }
    }
}
