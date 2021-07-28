use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use zip::write::FileOptions;
use zip::ZipWriter;

use crate::cfg::{
    CfgError,
    dataset::build_dataset,
    mutate::generate,
    parse,
};

mod cfg;
mod py3;


fn prepare_zip_file(zip_p: &Path, cfg_prefix: &str)
    -> Result<(ZipWriter<File>, String), CfgError> {
    let zip_f = fs::File::create(zip_p)
        .map_err(|e| CfgError::new(e.to_string()))?;

    let mut zip = zip::ZipWriter::new(zip_f);
    let zip_subdir = format!("{}/", cfg_prefix);
    let _ = zip.add_directory(&zip_subdir, Default::default())
        .map_err(|_| CfgError::new(
            format!("Unable to add '{}' directory to zip file", &zip_subdir)
        ))?;

    Ok((zip, zip_subdir))
}

/// Generate dataset based on `gf` and write it to a zip file.
pub fn cfg_dataset_as_zip(gf: &str, data_dir: &Path, cfg_prefix: &str)
    -> Result<(), CfgError> {
    let cfg = parse::parse(gf)
        .map_err(|e| CfgError::new(e.msg))?;
    let cfgs = generate(&cfg)?;

    println!("\n=> generated {} cfgs, creating dataset ...", cfgs.len());
    let ds = build_dataset(&cfgs, &data_dir)?;
    let ds_files = ds.persist(&data_dir)?;

    let zip_p = data_dir.join(format!("{}.zip", cfg_prefix));
    let (mut zip, zip_subdir) = prepare_zip_file(&zip_p, cfg_prefix)?;
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
