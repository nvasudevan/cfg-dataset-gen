use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::path::Path;

use cfgz::lr1_check;
use rand::{Rng, thread_rng};

use crate::cfg::{Cfg, CfgError, parse};
use crate::cfg::graph::{CfgGraph, GraphResult};
use crate::cfg::mutate::CfgMutation;
use crate::{DatasetGenInput, MutType};
use rand::prelude::SliceRandom;

/// Represents the ML data associated with a Cfg Graph
pub(crate) struct CfgData {
    /// Graph associated with the grammar
    pub(crate) graph: GraphResult,
    /// label: 0 indicates grammar is unambiguous; 1 is ambiguous
    pub(crate) label: usize,
}

impl CfgData {
    pub(crate) fn new(graph: GraphResult, label: usize) -> Self {
        Self {
            graph,
            label,
        }
    }
}

impl fmt::Display for CfgData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = format!(
            "graph: {}, label: {}", self.graph, self.label
        );
        write!(f, "{}", s)
    }
}

pub(crate) struct CfgDataSet {
    pub(crate) cfg_data: Vec<CfgData>,
    pub(crate) node_ids_map: HashMap<String, usize>,
    pub(crate) edge_ids_map: HashMap<String, usize>,
}

impl CfgDataSet {
    pub(crate) fn new(cfg_data: Vec<CfgData>) -> Self {
        Self {
            cfg_data,
            node_ids_map: Default::default(),
            edge_ids_map: Default::default(),
        }
    }

    // /// Have an equal no of CFGs for each label
    // pub(crate) fn curate(&mut self, cfgs_curate: bool) {
    //     if cfgs_curate {
    //         let mut label_0: Vec<&mut CfgData> = self.cfg_data
    //             .iter_mut()
    //             .map(|c| if c.label == 0 {
    //                 c
    //             }).collect();
    //         let mut label_1: Vec<&CfgData> = self.cfg_data
    //             .iter_mut()
    //             .map(|c| if c.label == 1 {
    //                 c
    //             }).collect();
    //
    //         if label_1.len() < label_0.len() {
    //             // get random label_1_cnt from label_0 cfg's
    //             let mut rnd = thread_rng();
    //             let cfgs_0: Vec<&CfgData> = label_0.choose_multiple(
    //                 &mut rnd, label_1.len()
    //             ).collect();
    //         }
    //     }
    // }

    /// Build a map of unique nodes from the CFGs in the dataset
    fn build_unique_nodes_map(&mut self) {
        println!("=> building the list of unique nodes ...");
        let mut node_ids_map: HashMap<String, usize> = HashMap::new();
        let mut node_id_counter: usize = 0;

        for cfg in &self.cfg_data {
            for node in &cfg.graph.nodes {
                let node_s = node.min_item_string();
                if !node_ids_map.contains_key(&node_s) {
                    node_ids_map.insert(node_s, node_id_counter);
                    node_id_counter += 1;
                }
            }
        }
        self.node_ids_map = node_ids_map;
    }

    /// Build a map of unique edges from the CFGs in the dataset
    fn build_unique_edges_map(&mut self) {
        println!("=> building the list of unique edges ...");
        let mut edge_ids_map: HashMap<String, usize> = HashMap::new();
        let mut edge_id_counter: usize = 0;

        for cfg in &self.cfg_data {
            for e in &cfg.graph.edges {
                let e_s = e.edge_label();
                if !edge_ids_map.contains_key(&e_s) {
                    edge_ids_map.insert(e_s, edge_id_counter);
                    edge_id_counter += 1;
                }
            }
        }
        self.edge_ids_map = edge_ids_map;
    }

    /// Using the CFGs generated, build a collection of unique nodes and edges
    pub(crate) fn build_unique_nodes_edges(&mut self) {
        self.build_unique_nodes_map();
        self.build_unique_edges_map();
    }

    /// CFG_node_labels.txt - `i`th line indicates the node label of the `i`th node
    fn write_node_labels(&self, node_labels_file: &Path) -> Result<(), CfgError> {
        println!("=> writing node labels to {}", node_labels_file.to_str().unwrap());
        let mut node_labels: Vec<String> = vec![];
        for cfg in &self.cfg_data {
            for n in &cfg.graph.nodes {
                let n_label = n.min_item_string();
                let v = self.node_ids_map.get(&n_label)
                    .ok_or_else(||
                        CfgError::new(
                            format!("Didn't find node {} in node_ids_map", n))
                    )?;
                // println!("{} -- {}", n, v);
                node_labels.push(v.to_string());
            }
        }

        let node_labels_s = node_labels.join("\n");
        std::fs::write(node_labels_file, node_labels_s)
            .expect("Unable to write node ids' to file");

        Ok(())
    }

    /// Write the class labels for graph
    fn write_graph_labels(&self, graph_labels_file: &Path) -> Result<(), CfgError> {
        println!("=> writing graph labels to {}", graph_labels_file.to_str().unwrap());
        let graph_labels: Vec<String> = self.cfg_data
            .iter()
            .map(|c| c.label.to_string())
            .collect();

        let graph_labels_s = graph_labels.join("\n");
        let _ = std::fs::write(graph_labels_file, graph_labels_s)
            .map_err(|e| CfgError::new(e.to_string()));

        Ok(())
    }

    fn write_graph_indicators(&self, graph_indicator_file: &Path) -> Result<(), CfgError> {
        println!("=> writing graph indicators to {}", graph_indicator_file.to_str().unwrap());
        let mut graph_indicator: Vec<String> = vec![];
        for (i, cfg) in self.cfg_data.iter().enumerate() {
            for _ in &cfg.graph.nodes {
                graph_indicator.push((i + 1).to_string());
            }
        }

        let graphs_indicator_s = graph_indicator.join("\n");
        let _ = std::fs::write(graph_indicator_file, graphs_indicator_s)
            .map_err(|e| CfgError::new(e.to_string()));

        Ok(())
    }

    /// Create `CFG_A.txt` containing the sparse matrix of all edges
    fn write_edge_labels(&self, edge_labels_file: &Path, edges_file: &Path) -> Result<(), CfgError> {
        println!("=> writing edge labels to {}", edge_labels_file.to_str().unwrap());
        println!("=> writing edges (sparse matrix) to {}", edges_file.to_str().unwrap());
        let mut n_i = 1;
        let mut edge_labels: Vec<String> = vec![];
        let mut edges: Vec<String> = vec![];
        for cfg in &self.cfg_data {
            for e in &cfg.graph.edges {
                let edge_label_code = self.edge_ids_map.get(&e.edge_label())
                    .ok_or_else(|| CfgError::new(
                        format!("Unable to retrieve code for edge: {}", e)
                    ))?;
                // edge_labels.push(format!("{}, {}", e.edge_label(), edge_label_code.to_string()));
                edge_labels.push(edge_label_code.to_string());
                let src_node = e.source_node_id() + n_i;
                let tgt_node = e.target_node_id() + n_i;
                // println!("[{} -> {}] - {}", src_node, tgt_node, e);
                edges.push(format!("{}, {}", src_node, tgt_node));
            }
            n_i += cfg.graph.nodes.len();
        }

        let edge_labels_s = edge_labels.join("\n");
        let _ = std::fs::write(edge_labels_file, edge_labels_s)
            .map_err(|e| CfgError::new(e.to_string()));

        let edges_s = edges.join("\n");
        let _ = std::fs::write(edges_file, edges_s)
            .map_err(|e| CfgError::new(e.to_string()));

        Ok(())
    }

    /// Save dataset files; return the list of file names (absolute path)
    ///
    /// - node labels (CFG_node_labels.txt)
    /// - graph labels (CFG_graph_labels.txt)
    /// - node to graph mapping (CFG_graph_indicator.txt)
    /// - edge labels (CFG_edge_labels.txt)
    pub(crate) fn persist(&self, data_dir: &Path) -> Result<Vec<String>, CfgError> {
        let mut ds_files: Vec<String> = vec![];
        let node_labels_file = data_dir.join("CFG_node_labels.txt");
        self.write_node_labels(&node_labels_file)?;
        ds_files.push(node_labels_file.to_str().unwrap().to_owned());

        let graph_labels_file = data_dir.join("CFG_graph_labels.txt");
        self.write_graph_labels(&graph_labels_file)?;
        ds_files.push(graph_labels_file.to_str().unwrap().to_owned());

        let graph_indicator_file = data_dir.join("CFG_graph_indicator.txt");
        self.write_graph_indicators(&graph_indicator_file)?;
        ds_files.push(graph_indicator_file.to_str().unwrap().to_owned());

        let edge_labels_file = data_dir.join("CFG_edge_labels.txt");
        let edges_file = data_dir.join("CFG_A.txt");
        self.write_edge_labels(&edge_labels_file, &edges_file)?;
        ds_files.push(edge_labels_file.to_str().unwrap().to_owned());
        ds_files.push(edges_file.to_str().unwrap().to_owned());

        let readme_file = data_dir.join("README.txt");
        let _ = std::fs::write(&readme_file, "README");
        ds_files.push(readme_file.to_str().unwrap().to_owned());

        Ok(ds_files)
    }
}

// fn weighted_label(labels_cnt: (usize, usize)) -> usize {
//     let label_0 = labels_cnt.0 + 1;
//     let label_1 = labels_cnt.1 + 1;
//     let label_0_r: f32 = (label_0 as f32) / (label_0 + label_1) as f32;
//     let label_1_r: f32 = (label_1 as f32) / (label_0 + label_1) as f32;
//     let label_0_wgt = 1.0 - label_0_r;
//     let label_1_wgt = 1.0 - label_1_r;
//     let labels = [(0, label_0_wgt), (1, label_1_wgt)];
//     let l = labels.choose_weighted(&mut thread_rng(), |lbl| lbl.1).unwrap().0;
//     eprintln!("[{}, {}] ==> {:?} ==> {}", label_0, label_1, labels, l);
//
//     l
// }

/// Calculate the `label` for the given grammar, label:
/// 0 - unambiguous
/// 1 - ambiguous
fn calc_label(cfg: &Cfg, cfg_i: usize, ds_input: &DatasetGenInput) -> Result<usize, CfgError> {
    let cfg_acc = &ds_input.data_dir.join(format!("{}.acc", cfg_i));
    std::fs::write(&cfg_acc, cfg.as_acc())
        .map_err(|e| CfgError::new(
            format!("Error occurred whilst writing cfg in ACCENT format:\n{}",
                    e.to_string())
        ))?;

    let cfg_yacc = &ds_input.data_dir.join(format!("{}.y", cfg_i));
    std::fs::write(&cfg_yacc, cfg.as_yacc())
        .map_err(|e| CfgError::new(
            format!("Error occurred whilst writing cfg in YACC format:\n{}",
                    e.to_string())
        ))?;
    let lr1 = lr1_check(cfg_yacc.as_path())
        .map_err(|e|
            CfgError::new(format!("Error: {}", e.to_string()))
        )?;

    return match lr1 {
        true => { Ok(0) }
        false => {
            let gp = cfg_acc.as_path().to_str().unwrap();
            let res = sinbad_rs::invoke(
                &ds_input.sin,
                ds_input.sin_duration,
                &ds_input.sin_backend,
                ds_input.sin_depth,
                gp,
                &ds_input.cfg_lex,
            )?;
            match res {
                true => { Ok(1) }
                false => { Ok(2) }
            }
        }
    };
}

/// Build dataset from the given input params
pub(crate) fn build_dataset(ds_input: &DatasetGenInput) -> Result<CfgDataSet, CfgError> {
    println!("\n=> generating {} cfgs ...", ds_input.no_samples);
    let base_cfg = parse::parse(ds_input.cfg_path)?;
    let mut cfg_mut = CfgMutation::new(&base_cfg);
    cfg_mut.instantiate();
    let mut generated_cfgs: Vec<Cfg> = vec![];
    let mut i: usize = 0;
    let mut cfg_data: Vec<CfgData> = vec![];
    let mut rnd = rand::thread_rng();
    let mut label0_cfgs: Vec<Cfg> = vec![];
    let mut label1_cfgs: Vec<Cfg> = vec![];
    let label0_samples = ds_input.no_samples/2;
    let label1_samples = ds_input.no_samples/2;
    println!("No samples:: 0: {}, 1:{}", label0_samples, label1_samples);

    loop {
        eprint!(".");
        let no_mutations = rnd.gen_range(1..=ds_input.max_mutations_per_cfg);
        let mut_type = ds_input.mut_types.choose(&mut rnd).unwrap();
        let cfg = match mut_type {
            MutType::InsertTerm => {
                eprint!("[i]");
                cfg_mut.insert(no_mutations)?
            }
            MutType::ReplaceTerm => {
                eprint!("[r]");
                cfg_mut.mutate(no_mutations)?
            }
            MutType::DeleteTerm => {
                eprint!("[d]");
                cfg_mut.delete(no_mutations)?
            }
        };
        if !generated_cfgs.contains(&cfg) {
            let label: usize = calc_label(&cfg, i, &ds_input)?;
            // interested only in 0, 1
            match label {
                0 => {
                    if label0_cfgs.len() < label0_samples {
                        label0_cfgs.push(cfg.clone());
                    }
                }
                1 => {
                    if label1_cfgs.len() < label1_samples {
                        label1_cfgs.push(cfg.clone());
                    }
                }
                _ => {}
            }

            generated_cfgs.push(cfg);
            eprintln!("[{}/{}/{}] - {}", i, label0_cfgs.len(), label1_cfgs.len(), label);
            std::io::stdout().flush().unwrap();
        }
        i += 1;
        if (i >= ds_input.max_iter) ||
            ((label0_cfgs.len() >= label0_samples) &&
                (label1_cfgs.len() >= label1_samples)) {
            break;
        }
    }
    for cfg in label0_cfgs.drain(..) {
        let g = CfgGraph::new(cfg);
        let g_result = g.instantiate().expect("Unable to convert cfg to graph");
        // eprintln!("[{},{}]", g_result.nodes.len(), g_result.edges.len());
        cfg_data.push(CfgData::new(g_result, 0));
    }

    for cfg in label1_cfgs.drain(..) {
        let g = CfgGraph::new(cfg);
        let g_result = g.instantiate().expect("Unable to convert cfg to graph");
        cfg_data.push(CfgData::new(g_result, 1));
    }
    cfg_data.shuffle(&mut rnd);

    Ok(CfgDataSet::new(cfg_data))
}

#[cfg(test)]
mod tests {
    extern crate tempdir;

    use tempdir::TempDir;

    use crate::cfg::dataset::build_dataset;
    use crate::cfg::mutate::generate;
    use crate::cfg::parse;

    #[test]
    fn test_ds_generate() {
        let cfg = parse::parse("./grammars/lr1.y")
            .expect("Unable to parse as a cfg");
        let cfgs = generate(&cfg, 3)
            .expect("Unable to generate mutated CFGs");
        let data_dir = TempDir::new("cfg-ds")
            .expect("Unable to create temp dir");
        let _ = build_dataset(&cfgs, data_dir.path())
            .expect("Unable to build dataset from cfgs");

        data_dir.close()
            .expect("Unable to close the data directory");
    }
}
