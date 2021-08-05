use std::collections::HashMap;

use rand;
use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::cfg::{Cfg, CfgError, LexSymbol, TermSymbol};

pub(crate) struct CfgMutation<'a> {
    cfg: &'a Cfg,
    terminal_indices: HashMap<String, Vec<Vec<usize>>>,
    non_terms_with_terminals: Vec<String>,
    pub(crate) terms: Vec<&'a TermSymbol>,
}

struct TerminalPos {
    rule_name: String,
    alt_i: usize,
    term_j: usize
}

impl TerminalPos {
    fn new(rule_name: String, alt_i: usize, term_j: usize) -> Self {
        Self {
            rule_name,
            alt_i,
            term_j
        }
    }
}

impl PartialEq for TerminalPos {
    fn eq(&self, other: &Self) -> bool {
        if self.rule_name.eq(&other.rule_name) ||
            (self.alt_i == other.alt_i) ||
            (self.term_j == other.term_j) {
            return true;
        }

        false
    }
}


impl<'a> CfgMutation<'a> {
    pub(crate) fn new(cfg: &'a Cfg) -> Self {
        Self {
            cfg,
            terminal_indices: Default::default(),
            non_terms_with_terminals: vec![],
            terms: vec![],
        }
    }

    /// Calculates the indices of alternatives for each rule
    fn terminal_indices(&mut self) {
        let mut term_indices_map: HashMap<String, Vec<Vec<usize>>> = HashMap::new();
        for rule in &self.cfg.rules {
            let mut rule_i: Vec<Vec<usize>> = vec![];
            for alt in &rule.rhs {
                let mut alt_i: Vec<usize> = vec![];
                for (j, sym) in alt.lex_symbols.iter().enumerate() {
                    if let LexSymbol::Term(_) = sym {
                        alt_i.push(j);
                    }
                }
                rule_i.push(alt_i);
            }
            term_indices_map.insert(rule.lhs.to_owned(), rule_i);
        }

        self.terminal_indices = term_indices_map;
    }

    /// Returns a list of non-terminals which have alternatives with terminals in them
    fn nt_alt_with_terminals(&mut self) {
        let mut keys: Vec<String> = vec![];
        for (k, values) in &self.terminal_indices {
            for v in values {
                if v.len() > 0 {
                    keys.push(k.to_owned());
                    break;
                }
            }
        }
        self.non_terms_with_terminals = keys;
    }

    /// Extract terminals and rules containing terminals and their locations.
    pub(crate) fn instantiate(&mut self) {
        self.terminal_indices();
        self.nt_alt_with_terminals();
        self.terms = self.cfg.terminals();
    }

    /// Returns the number of possible mutations where terminals occur
    fn term_pos_cnt(&self) -> usize {
        self.terminal_indices.values()
            .map(|u|
                u.iter().map(|v| v.len()).sum::<usize>()
            )
            .sum::<usize>()
    }

    /// No of single mutations possible:
    /// number of terminal positions X (number of terminals - 1)
    /// (we exclude the current terminal)
    fn single_mutation_cnt(&self) -> usize {
        self.term_pos_cnt() * (self.terms.len() - 1)
    }

    /// No of double mutations possible: nC2
    /// where n is the number of terminal positions and
    /// for each position there are (number of terminals - 1) options
    fn double_mutation_cnt(&self) -> usize {
        let term_cnt = self.term_pos_cnt();
        ((term_cnt * (term_cnt - 1))/2) *
            (self.terms.len() - 1) *
            (self.terms.len() - 1)
    }

    pub(crate) fn total_mutations_cnt(&self) -> usize {
        let single_cnt = self.single_mutation_cnt();
        let double_cnt = self.double_mutation_cnt();
        let cnt = single_cnt + double_cnt;
        println!("=> total mutations: {} ({}, {})", cnt, single_cnt, double_cnt);

        cnt
    }

    /// Returns a tuple of indices of terminal location
    fn alt_with_terminals(&self, nt: &str) -> (usize, usize) {
        let alt_with_terms = self.terminal_indices.get(nt).unwrap();
        let mut alt_indices: Vec<usize> = vec![];
        for (i, v) in alt_with_terms.iter().enumerate() {
            if v.len() > 0 {
                alt_indices.push(i);
            }
        }
        let j = alt_indices.choose(&mut thread_rng()).unwrap();
        let terminal_indices = &alt_with_terms[*j];
        let terminal_i = terminal_indices.choose(&mut thread_rng()).unwrap();

        (*j, *terminal_i)
    }

    /// Create a new cfg from current with `n` mutations
    /// Mutation involves:
    /// - pick a random rule, and a random alternative containing terminals
    /// - from chosen alternative, replace a randomly picked terminal with another one
    pub(crate) fn mutate(&mut self, n: usize) -> Result<Cfg, CfgError> {
        // println!("=> no of mutations: {}", n);
        let mut term_positions: Vec<TerminalPos> = vec![];
        let mut cfg = self.cfg.clone();
        loop {
            let nt = self.non_terms_with_terminals.choose(&mut thread_rng()).unwrap();
            let (alt_i, term_j) = self.alt_with_terminals(&nt);
            // println!("nt: {}, alt_i: {}, term_j: {}", nt, alt_i, term_j);
            let term_pos = TerminalPos::new(nt.to_owned(), alt_i, term_j);
            if ! term_positions.contains(&term_pos) {
                let alt = cfg.get_alt_mut(nt, alt_i)
                    .ok_or_else(||
                        CfgError::new(
                            format!("Failed to get alternative for non-terminal {} (index: {})",
                                    nt,
                                    alt_i
                            )
                        ))?;
                let mut_sym = &alt.lex_symbols[term_j];
                let term_exclude = LexSymbol::to_term(mut_sym)
                    .ok_or_else(|| CfgError::new(
                        format!("Unable to convert LexSymbol {} to TermSymbol", mut_sym)
                    ))?;

                let new_terms: Vec<&TermSymbol> = self.terms
                    .iter()
                    .filter(|t| (**t).ne(term_exclude))
                    .map(|t| *t)
                    .collect();
                let new_term = new_terms.choose(&mut thread_rng()).unwrap();
                alt.lex_symbols[term_j] = LexSymbol::Term((*new_term).clone());
                term_positions.push(term_pos);
            }
            if term_positions.len() >= n {
                break
            }
        }

        Ok(cfg)
    }
}

// /// Generate random CFGs based on `cfg`.
// /// The number of mutations possible is dependent on the terminals in `cfg`.
// /// Start a mutation run until we generate the `cnt` mutated grammars
// /// or hit the `MAX_ITER_LIMIT` threshold.
// pub(crate) fn generate(cfg: &Cfg, cnt: usize) -> Result<Vec<Cfg>, CfgError> {
//     let mut cfg_mut = CfgMutation::new(&cfg);
//     cfg_mut.instantiate();
//     let max_cnt = cfg_mut.mut_cnt() * (cfg_mut.terms.len() - 1);
//     println!("=> max_cnt: {}", max_cnt);
//     let mut mutated_cfgs: Vec<Cfg> = vec![];
//     let mut i: usize = 0;
//     let no_mutations: usize = 1;
//     loop {
//         let cfg = cfg_mut.mutate(no_mutations)?;
//         if !mutated_cfgs.contains(&cfg) {
//             mutated_cfgs.push(cfg);
//             eprint!(".");
//         } else {
//             eprint!("X");
//         }
//         if i % 100 == 0 {
//             println!();
//         }
//         std::io::stdout().flush().unwrap();
//
//         i += 1;
//         if (i >= MAX_ITER_LIMIT) || (mutated_cfgs.len() >= cnt || (cnt >= max_cnt)) {
//             break;
//         }
//     }
//
//     Ok(mutated_cfgs)
// }

#[cfg(test)]
mod tests {
    extern crate tempdir;

    use tempdir::TempDir;

    use crate::cfg::graph::CfgGraph;
    use crate::cfg::mutate::{CfgMutation, generate};
    use crate::cfg::parse;

    #[test]
    fn test_cfg_mutation() {
        let cfg = parse::parse("./grammars/lr1.y")
            .expect("Unable to parse as a cfg");
        let mut cfg_mut = CfgMutation::new(&cfg);
        cfg_mut.instantiate();
        let cfgs = generate(&cfg, 3)
            .expect("Unable to generate a mutated cfg");
        println!("\n=> generated {} cfgs, writing ...", cfgs.len());
        let tempd = TempDir::new("cfg-test")
            .expect("Unable to create temp dir");

        for (i, cfg) in cfgs.iter().enumerate() {
            let cfgp = tempd.path().join(i.to_string());
            std::fs::write(&cfgp, cfg.as_yacc())
                .expect(&format!("Failed to write cfg {}", cfg));
            let g = CfgGraph::new(cfg.clone());
            let g_result = g.instantiate()
                .expect("Unable to convert cfg to graph");
            println!("=>graph: {}", g_result);
        }

        tempd.close().expect("Unable to close temp dir");
    }
}
