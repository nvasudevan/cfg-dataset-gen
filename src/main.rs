use cfg_dataset_gen::cfg_dataset_as_zip;
use cfg_dataset_gen::DatasetGenInput;
use std::path::Path;

fn main() {
    println!("=> Building cfg dataset ...");
    let cfg_path = "./grammars/lr1.y";
    let data_dir = Path::new("/var/tmp/cfg_ds");
    std::fs::remove_dir_all(&data_dir)
        .expect(&format!("Unable to remove data directory: {}", data_dir.to_str().unwrap()));
    std::fs::create_dir(&data_dir)
        .expect(&format!("Unable to create data directory: {}", data_dir.to_str().unwrap()));
    let ds_label = "CFG";
    let no_samples: usize = 5;
    let max_mutations_per_cfg: usize = 2;
    // 0 - unambiguous (and LR1), 1 - ambiguous, 2 - can't decide
    // we are only interested in 0,1 labels for now
    let allowed_labels: Vec<usize> = vec![0, 1];
    let max_iter_limit = 500;

    let ds_input = DatasetGenInput::new(
        cfg_path,
        data_dir,
        no_samples,
        max_mutations_per_cfg,
        ds_label.to_owned(),
        allowed_labels.as_slice(),
        max_iter_limit
    );

    match cfg_dataset_as_zip(&ds_input) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("error: {}", e);
        }
    }
}
