use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use zip::write::FileOptions;
use zip::ZipWriter;

use crate::cfg::{CfgError, parse};
use crate::cfg::dataset::{build_dataset, CfgDataSet, CfgData};
use sinbad_rs::sinbad::SinBAD;
use crate::cfg::graph::CfgGraph;
use std::rc::Rc;

use cfgz::lr1_check;

mod cfg;

/// Defines the types of mutations possible on a CFG
pub enum MutType {
    /// Inserts a random terminal at a random location
    InsertTerm,
    /// Replaces a terminal with another terminal
    ReplaceTerm,
    /// Deletes a randomly picked terminal
    DeleteTerm,
}

/// Defines input parameters for generating a dataset
pub struct DatasetGenInput<'a, 'b, 'c, 'd, 'e> {
    /// Path to base Cfg
    cfg_path: &'a str,
    /// Path to Cfg lex
    cfg_lex: &'b str,
    /// SinBAD instance to generate target label
    sin: &'c SinBAD,
    /// SinBAD backend to apply
    sin_backend: &'e str,
    /// SinBAD threshold depth
    sin_depth: usize,
    /// SinBAD running time for each CFG
    sin_duration: usize,
    /// Data directory to save CFGs and zip file
    data_dir: &'d Path,
    /// No of CFG samples to generate
    no_samples: usize,
    /// No of mutations allowed per CFG
    max_mutations_per_cfg: usize,
    /// mutation types: insert, replace, delete
    mut_types: Vec<MutType>,
    /// Label used to create zip file and sub directory within it
    ds_label: String,
    /// Maximum no of iterations allowed to generate dataset
    max_iter: usize,
}

impl<'a, 'b, 'c, 'd, 'e> DatasetGenInput<'a, 'b, 'c, 'd, 'e> {
    pub fn new(cfg_path: &'a str,
               cfg_lex: &'b str,
               sin: &'c SinBAD,
               sin_backend: &'e str,
               sin_depth: usize,
               sin_duration: usize,
               data_dir: &'d Path,
               no_samples: usize,
               max_mutations_per_cfg: usize,
               mut_types: Vec<MutType>,
               ds_label: String,
               max_iter: usize) -> Self {
        Self {
            cfg_path,
            cfg_lex,
            sin,
            sin_backend,
            sin_depth,
            sin_duration,
            data_dir,
            no_samples,
            max_mutations_per_cfg,
            mut_types,
            ds_label,
            max_iter,
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
    let mut ds = build_dataset(ds_input)?;
    println!("=> building list of unique nodes+edges ...");
    ds.build_unique_nodes_edges();
    println!("unique nodes: {}, edges; {}", ds.node_ids_map.len(), ds.edge_ids_map.len());

    let ds_files = ds.persist(ds_input.data_dir)?;

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
                "Unable to create file within zip".to_owned()
            ))?;
        zip.write_all(contents.as_bytes()).unwrap();
    }
    zip.finish().unwrap();

    Ok(())
}

/// Benchmark dataset generation for a small collection of CFGs
pub fn cfg_graph_bench(cfgs: &[String]) {
    let mut cfg_data: Vec<CfgData> = Vec::with_capacity(cfgs.len());
    for gp in cfgs {
        let cfg = parse::parse(&gp)
            .expect("Unable to parse the grammar!");
        let g = CfgGraph::new(Rc::new(cfg));
        let g_result = g.instantiate()
            .expect("Unable to process Cfg Graph");
        cfg_data.push(CfgData::new(g_result, 0));
    }
    let mut cfg_ds = CfgDataSet::new(cfg_data);
    cfg_ds.build_unique_nodes_edges();
    let td = tempdir::TempDir::new("cfg-bench")
        .expect("Unable to create a temporary directory");
    let _ = cfg_ds.persist(td.path())
        .expect("Error whilst generating node labels");
}

pub fn bison_lr1_check(cfg_path: &str) {
   let _ = lr1_check(Path::new(cfg_path), false)
       .expect("Bison failed on grammar!");
}