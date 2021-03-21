use criterion::{criterion_group, criterion_main, Criterion};
extern crate gomoku;
use gomoku::*;
use std::fs;

fn criterion_benchmark(c: &mut Criterion) {
    let content = fs::read_to_string("tests/data/block_three_two.in").unwrap();
    let mut group = c.benchmark_group("gomoku-solve");
    group.significance_level(0.1).sample_size(10);
    group.bench_function("monte-solve", |b| {
        b.iter(|| algo::gomoku_solve(content.as_str(), "monte"))
    });
    group.bench_function("minimax-solve", |b| {
        b.iter(|| algo::gomoku_solve(content.as_str(), "minimax"))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
