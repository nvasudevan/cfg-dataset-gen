use std::{
    collections::HashMap,
    fmt,
    io::Write,
    path::Path,
    rc::Rc,
};

use cfgz::lr1_check;
use rand::{
    Rng,
    prelude::SliceRandom,
};

use crate::cfg::{Cfg, CfgError, parse};
use crate::cfg::graph::{CfgGraph, GraphResult};
use crate::cfg::mutate::CfgMutation;
use crate::{DatasetGenInput, MutType};

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
        let mut node_ids_map: HashMap<String, usize> = HashMap::new();
        let mut node_id_counter: usize = 0;

        for cfg in &self.cfg_data {
            for node in &cfg.graph.nodes {
                let node_s = node.min_item_string();
                if let std::collections::hash_map::Entry::Vacant(e) = node_ids_map.entry(node_s) {
                    e.insert(node_id_counter);
                    node_id_counter += 1;
                }
            }
        }
        self.node_ids_map = node_ids_map;
    }

    /// Build a map of unique edges from the CFGs in the dataset
    fn build_unique_edges_map(&mut self) {
        let mut edge_ids_map: HashMap<String, usize> = HashMap::new();
        let mut edge_id_counter: usize = 0;

        for cfg in &self.cfg_data {
            for e in &cfg.graph.edges {
                let e_s = e.edge_label();
                if let std::collections::hash_map::Entry::Vacant(e) = edge_ids_map.entry(e_s) {
                    e.insert(edge_id_counter);
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
        let mut node_labels: Vec<String> = vec![];
        for cfg in &self.cfg_data {
            for n in &cfg.graph.nodes {
                let n_label = n.min_item_string();
                let v = self.node_ids_map.get(&n_label)
                    .ok_or_else(||
                        CfgError::new(
                            format!("Didn't find node {} in node_ids_map", n))
                    )?;
                node_labels.push((*v).to_string());
            }
        }

        std::fs::write(&node_labels_file, node_labels.join("\n"))
            .expect("Unable to write node ids' to file");

        Ok(())
    }

    /// Write the class labels of graph to `graph_labels_file`.
    fn write_graph_labels(&self, graph_labels_file: &Path) -> Result<(), CfgError> {
        // println!("=> writing graph labels to {}", graph_labels_file.to_str().unwrap());
        let graph_labels: Vec<String> = self.cfg_data
            .iter()
            .map(|c| c.label.to_string())
            .collect();

        let graph_labels_s = graph_labels.join("\n");
        let _ = std::fs::write(graph_labels_file, graph_labels_s)
            .map_err(|e| CfgError::new(e.to_string()));

        Ok(())
    }

    /// Write the (node to graph) mapping to `graph_indicator_file`.
    fn write_graph_indicators(&self, graph_indicator_file: &Path) -> Result<(), CfgError> {
        // println!("=> writing graph indicators to {}", graph_indicator_file.to_str().unwrap());
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
        // println!("=> writing edge labels to {}", edge_labels_file.to_str().unwrap());
        // println!("=> writing edges (sparse matrix) to {}", edges_file.to_str().unwrap());
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
        // println!("=> generating label artifacts ...");
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

/// Calculate the `label` for the given grammar, label:
/// 0 - unambiguous, 1 - ambiguous, 2 - don't know (has conflicts)
fn calc_label(cfg: Rc<Cfg>, cfg_i: usize, ds_input: &DatasetGenInput) -> Result<usize, CfgError> {
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
    let out = false;
    let lr1 = lr1_check(cfg_yacc.as_path(), out)
        .map_err(|e|
            CfgError::new(format!("Error: {}", e.to_string()))
        )?;

    return match lr1 {
        true => { Ok(0) }
        false => {
            let gp = cfg_acc.as_path().to_str().unwrap();
            let res = sinbad_rs::invoke(
                ds_input.sin,
                ds_input.sin_duration,
                ds_input.sin_backend,
                ds_input.sin_depth,
                gp,
                ds_input.cfg_lex,
            )?;
            match res {
                true => { Ok(1) }
                false => { Ok(2) }
            }
        }
    };
}

/// Build dataset from the given input params
/// We are only interested in class labels 0 and 1.
/// Also we aim to generate -- as much as possible -- an equal no of
/// graphs for each label (0/1).
pub(crate) fn build_dataset(ds_input: &DatasetGenInput) -> Result<CfgDataSet, CfgError> {
    let base_cfg = parse::parse(ds_input.cfg_path)?;
    let mut cfg_mut = CfgMutation::new(&base_cfg);
    cfg_mut.instantiate();
    let mut generated_cfgs: Vec<Rc<Cfg>> = Vec::with_capacity(ds_input.no_samples);
    let mut rnd = rand::thread_rng();
    let label0_samples = ds_input.no_samples / 2;
    let label1_samples = ds_input.no_samples / 2;
    let mut label0_cfgs: Vec<Rc<Cfg>> = Vec::with_capacity(label0_samples);
    let mut label1_cfgs: Vec<Rc<Cfg>> = Vec::with_capacity(label1_samples);
    println!("\n=> generating {} cfgs (0: {}, 1: {})",
             ds_input.no_samples,
             label0_samples,
             label1_samples);

    let mut i: usize = 0;
    loop {
        let no_mutations = rnd.gen_range(1..=ds_input.max_mutations_per_cfg);
        let mut_type = ds_input.mut_types.choose(&mut rnd).unwrap();
        let cfg = match mut_type {
            MutType::InsertTerm => {
                eprint!("i/");
                cfg_mut.insert(no_mutations)?
            }
            MutType::ReplaceTerm => {
                eprint!("r/");
                cfg_mut.mutate(no_mutations)?
            }
            MutType::DeleteTerm => {
                eprint!("d/");
                cfg_mut.delete(no_mutations)?
            }
        };
        let cfg_rc = Rc::new(cfg);
        if !generated_cfgs.contains(&cfg_rc) {
            let label: usize = calc_label(Rc::clone(&cfg_rc), i, ds_input)?;
            match label {
                0 => {
                    if label0_cfgs.len() < label0_samples {
                        label0_cfgs.push(Rc::clone(&cfg_rc));
                    }
                }
                1 => {
                    if label1_cfgs.len() < label1_samples {
                        label1_cfgs.push(Rc::clone(&cfg_rc));
                    }
                }
                _ => {}
            }

            generated_cfgs.push(cfg_rc);
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
    // build our cfg dataset
    let cfgs_cnt = label0_cfgs.len() + label1_cfgs.len();
    let mut cfg_data: Vec<CfgData> = Vec::with_capacity(cfgs_cnt);
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

    use crate::MutType;
    use crate::DatasetGenInput;
    use crate::cfg::dataset::build_dataset;

    #[test]
    fn test_build_dataset() {
        let cfg_path = "./grammars/lr1.y";
        let cfg_lex = "/home/krish/kv/sinbad/bin/general.lex";

        std::env::set_var("SINBAD_CMD", "/home/krish/kv/sinbad/src/sinbad");
        std::env::set_var("ACCENT_DIR", "/home/krish/kv/accent");
        std::env::set_var("TIMEOUT_CMD", "/usr/bin/timeout");

        let sin = sinbad_rs::sinbad()
            .expect("Unable to create sinbad instance!");
        let sin_backend = "dynamic1";
        let sin_depth = 10;
        let sin_duration: usize = 10;

        let td = tempdir::TempDir::new("cfg_ds")
            .expect("Unable to create a temporary directory");
        let data_dir = td.path();
        eprintln!("data dir: {}", data_dir.to_str().unwrap());
        let ds_label = "CFG";
        let no_samples: usize = 2;
        let max_mutations_per_cfg: usize = 1;
        let mut_types = vec![
            MutType::InsertTerm,
            MutType::ReplaceTerm,
            MutType::DeleteTerm,
        ];
        let max_iter = 100;

        let ds_input = DatasetGenInput::new(
            cfg_path,
            cfg_lex,
            &sin,
            sin_backend,
            sin_depth,
            sin_duration,
            data_dir,
            no_samples,
            max_mutations_per_cfg,
            mut_types,
            ds_label.to_owned(),
            max_iter,
        );
        let ds = build_dataset(&ds_input)
            .expect("Unable to build dataset from cfgs");

        assert_eq!(ds.cfg_data.len(), no_samples);
    }
}
