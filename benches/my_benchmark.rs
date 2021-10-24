use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cfg_dataset_gen::{cfg_graph_bench, bison_lr1_check};
use std::time::Duration;

fn lr1_graph_bench(c: &mut Criterion) {
    let gdir = "/home/krish/kv/cfg-dataset-gen/grammars/bench";
    let cfgs: Vec<String> = (1..10).into_iter()
        .map(|i|  format!("{}/{}.y", gdir, i))
        .collect();
    c.bench_function(
        "lr1_bench",
        |b| b.iter(|| cfg_graph_bench(black_box(cfgs.as_slice())))
    );
}

fn bison_lr1_bench(c: &mut Criterion) {
    let gp = "/home/krish/kv/cfg-dataset-gen/grammars/lr1.y";
    let mut grp = c.benchmark_group("lr1-500");
    grp.sample_size(400);
    grp.measurement_time(Duration::new(14, 0));
    grp.bench_function(
        "bison_lr1",
        |b| b.iter(|| bison_lr1_check(black_box(gp)))
    );
    grp.finish();
}

criterion_group!(benches, lr1_graph_bench);
criterion_main!(benches);
