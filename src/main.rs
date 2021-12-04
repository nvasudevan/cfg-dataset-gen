use cfg_dataset_gen::{cfg_dataset_as_zip, MutType};
use cfg_dataset_gen::DatasetGenInput;
use std::path::Path;
use sinbad_rs::sinbad::SinBADInput;

fn main() {
    println!("=> Building cfg dataset ...");
    let cfg_path = "./grammars/lr1_medium.y";
    let cfg_lex = "./grammars/general.lex";

    let sin = sinbad_rs::sinbad()
        .expect("Unable to create sinbad instance!");
    let sin_backend = "dynamic1";
    let sin_depth = 10;
    let sin_duration: usize = 10;
    let sin_input: SinBADInput = SinBADInput::new(sin_backend, sin_depth, sin_duration);

    let cfg_data_dir = "/var/tmp/cfg_ds";

    let data_dir = Path::new(cfg_data_dir);
    match std::fs::remove_dir_all(&data_dir) {
        Ok(_) => {}
        Err(_) => {
            println!("=> {} - nothing to do!", cfg_data_dir);
        }
    }
    std::fs::create_dir(&data_dir)
        .unwrap_or_else(|_|
            panic!("Unable to create data directory: {}", data_dir.to_str().unwrap())
        );
    let ds_label = "CFG";
    let no_samples: usize = 20;
    let max_mutations_per_cfg: usize = 1;
    let mut_types = vec![
        MutType::InsertTerm,
        MutType::ReplaceTerm,
        MutType::DeleteTerm
    ];

    let max_iterations = 5000;

    let ds_input = DatasetGenInput::new(
        cfg_path,
        cfg_lex,
        &sin,
        &sin_input,
        data_dir,
        no_samples,
        max_mutations_per_cfg,
        mut_types,
        ds_label.to_owned(),
        max_iterations,
    );

    match cfg_dataset_as_zip(&ds_input) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("error: {}", e);
        }
    }
}
