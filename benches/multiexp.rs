#[macro_use]
extern crate criterion;

use criterion::{black_box, BenchmarkId, Criterion};
use group::ff::Field;
use halo2_proofs::{
    arithmetic::best_multiexp,
    halo2curves::pasta::Fp,
    poly::{commitment::ParamsProver, ipa::commitment::ParamsIPA},
};
use pasta_curves::EqAffine;
use rand_core::OsRng;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiexp");
    for k in 8..16 {
        group
            .bench_function(BenchmarkId::new("k", k), |b| {
                let rng = OsRng;
                let params: ParamsIPA<EqAffine> = ParamsIPA::new(k);
                let g = &mut params.get_g().to_vec();
                let len = g.len() / 2;
                let (g_lo, g_hi) = g.split_at_mut(len);
                let coeff_1 = Fp::random(rng);
                let coeff_2 = Fp::random(rng);

                b.iter(|| {
                    for (g_lo, g_hi) in g_lo.iter().zip(g_hi.iter()) {
                        best_multiexp(&[black_box(coeff_1), black_box(coeff_2)], &[*g_lo, *g_hi]);
                    }
                })
            })
            .sample_size(30);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
