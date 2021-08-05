use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use zip::write::FileOptions;
use zip::ZipWriter;

use crate::cfg::CfgError;
use crate::cfg::dataset::build_dataset;

mod cfg;
mod py3;

/// Defines input parameters for generating a dataset
pub struct DatasetGenInput<'a, 'b, 'c> {
    /// Path to base Cfg
    cfg_path: &'a str,
    data_dir: &'b Path,
    no_samples: usize,
    max_mutations_per_cfg: usize,
    ds_label: String,
    allowed_labels: &'c [usize],
    max_iter_limit: usize,
}

impl<'a, 'b, 'c> DatasetGenInput<'a, 'b, 'c> {
    pub fn new(cfg_path: &'a str,
               data_dir: &'b Path,
               no_samples: usize,
               max_mutations_per_cfg: usize,
               ds_label: String,
               allowed_labels: &'c [usize],
               max_iter_limit: usize) -> Self {
        Self {
            cfg_path,
            data_dir,
            no_samples,
            max_mutations_per_cfg,
            ds_label,
            allowed_labels,
            max_iter_limit,
        }
    }
}

fn prepare_zip_file(zip_p: &Path, ds_label: &str)
                    -> Result<(ZipWriter<File>, String), CfgError> {
    let zip_f = fs::File::create(zip_p)
        .map_err(|e| CfgError::new(e.to_string()))?;

    let mut zip = zip::ZipWriter::new(zip_f);
    let zip_subdir = format!("{}/", ds_label);
    let _ = zip.add_directory(&zip_subdir, Default::default())
        .map_err(|_| CfgError::new(
            format!("Unable to add '{}' directory to zip file", &zip_subdir)
        ))?;

    Ok((zip, zip_subdir))
}

/// Generate dataset based on input params `ds_input` and write to zip file.
pub fn cfg_dataset_as_zip(ds_input: &DatasetGenInput) -> Result<(), CfgError> {
    let ds = build_dataset(&ds_input)?;
    let ds_files = ds.persist(&ds_input.data_dir)?;

    let zip_p = ds_input.data_dir.join(format!("{}.zip", ds_input.ds_label));
    let (mut zip, zip_subdir) = prepare_zip_file(&zip_p, &ds_input.ds_label)?;
    println!("=> Saving dataset to zip file: {}", zip_p.to_str().unwrap());

    for f in ds_files {
        let zip_file_options = FileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .unix_permissions(0o755);
        let ds_p = Path::new(&f);
        let contents = std::fs::read_to_string(ds_p).unwrap();
        let zip_ds_p = format!("{}{}", zip_subdir, ds_p.file_name().unwrap().to_str().unwrap());
        let _ = zip.start_file(zip_ds_p, zip_file_options)
            .map_err(|_| CfgError::new(
                format!("Unable to create file within zip")
            ))?;
        zip.write_all(contents.as_bytes()).unwrap();
    }
    zip.finish().unwrap();

    Ok(())
}
