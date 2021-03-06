use std::collections::HashMap;


use rand::prelude::SliceRandom;
use rand::{thread_rng, Rng};

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
    term_j: usize,
}

impl TerminalPos {
    fn new(rule_name: String, alt_i: usize, term_j: usize) -> Self {
        Self {
            rule_name,
            alt_i,
            term_j,
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
                if !v.is_empty() {
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

    /// Returns a tuple of indices of terminal location
    fn alt_with_terminals(&self, nt: &str) -> (usize, usize) {
        let alt_with_terms = self.terminal_indices.get(nt).unwrap();
        let mut alt_indices: Vec<usize> = vec![];
        for (i, v) in alt_with_terms.iter().enumerate() {
            if !v.is_empty() {
                alt_indices.push(i);
            }
        }
        let j = alt_indices.choose(&mut thread_rng()).unwrap();
        let terminal_indices = &alt_with_terms[*j];
        let terminal_i = terminal_indices.choose(&mut thread_rng()).unwrap();

        (*j, *terminal_i)
    }

    /// Create a new cfg by inserting a terminal at a randomly selected place
    pub(crate) fn insert(&mut self, n: usize) -> Result<Cfg, CfgError> {
        let mut term_positions: Vec<TerminalPos> = vec![];
        let mut cfg = self.cfg.clone();
        let mut rnd = thread_rng();
        loop {
            let rnd_rule = cfg.rules.choose_mut(&mut rnd).unwrap();
            let alt_i = rnd.gen_range(0..rnd_rule.rhs.len());
            let rnd_alt = &mut rnd_rule.rhs[alt_i];
            let term_j = rnd.gen_range(0..=rnd_alt.lex_symbols.len());
            let term_pos = TerminalPos::new(rnd_rule.lhs.to_owned(), alt_i, term_j);
            if ! term_positions.contains(&term_pos) {
                let rnd_term = self.terms.choose(&mut rnd).unwrap();
                let new_term_sym = LexSymbol::Term((*rnd_term).clone());
                if term_j >= rnd_alt.lex_symbols.len() {
                    rnd_alt.lex_symbols.push(new_term_sym);
                } else {
                    rnd_alt.lex_symbols[term_j] = new_term_sym;
                }
                term_positions.push(term_pos);
            }

            if term_positions.len() >= n {
                break;
            }
        }

        Ok(cfg)
    }

    pub(crate) fn delete(&self, n: usize) -> Result<Cfg, CfgError> {
        let mut term_positions: Vec<TerminalPos> = vec![];
        let mut cfg = self.cfg.clone();
        loop {
            let nt = self.non_terms_with_terminals.choose(&mut thread_rng()).unwrap();
            let (alt_i, term_j) = self.alt_with_terminals(nt);
            // println!("nt: {}, alt_i: {}, term_j: {}", nt, alt_i, term_j);
            let term_pos = TerminalPos::new(nt.to_owned(), alt_i, term_j);
            if !term_positions.contains(&term_pos) {
                let alt = cfg.get_alt_mut(nt, alt_i)
                    .ok_or_else(||
                        CfgError::new(
                            format!("Failed to get alternative for non-terminal {} (index: {})",
                                    nt,
                                    alt_i
                            )
                        ))?;
                alt.lex_symbols.remove(term_j);
                term_positions.push(term_pos);
            } else {
                //
            }
            if term_positions.len() >= n {
                break;
            }
        }

        Ok(cfg)
    }

    /// Create a new cfg from current with `n` mutations
    /// Mutation involves:
    /// - pick a random rule, and a random alternative containing terminals
    /// - from chosen alternative, replace a randomly picked terminal with another one
    pub(crate) fn mutate(&mut self, n: usize) -> Result<Cfg, CfgError> {
        let mut term_positions: Vec<TerminalPos> = vec![];
        let mut cfg = self.cfg.clone();
        loop {
            let nt = self.non_terms_with_terminals.choose(&mut thread_rng()).unwrap();
            let (alt_i, term_j) = self.alt_with_terminals(nt);
            // println!("nt: {}, alt_i: {}, term_j: {}", nt, alt_i, term_j);
            let term_pos = TerminalPos::new(nt.to_owned(), alt_i, term_j);
            if !term_positions.contains(&term_pos) {
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
                    .filter(|t| (**t).ne(term_exclude)).copied()
                    .collect();
                let new_term = new_terms.choose(&mut thread_rng()).unwrap();
                alt.lex_symbols[term_j] = LexSymbol::Term((*new_term).clone());
                term_positions.push(term_pos);
            } else {
                //
            }
            if term_positions.len() >= n {
                break;
            }
        }

        Ok(cfg)
    }
}


#[cfg(test)]
mod tests {
    extern crate tempdir;

    

    
    use crate::cfg::mutate::CfgMutation;
    use crate::cfg::parse;

    #[test]
    fn test_cfg_mutation() {
        let cfg = parse::parse("./grammars/lr1.y")
            .expect("Unable to parse as a cfg");
        let mut cfg_mut = CfgMutation::new(&cfg);
        cfg_mut.instantiate();
        // let cfgs = generate(&cfg, 3)
        //     .expect("Unable to generate a mutated cfg");
        // println!("\n=> generated {} cfgs, writing ...", cfgs.len());
        // let tempd = TempDir::new("cfg-test")
        //     .expect("Unable to create temp dir");
        //
        // for (i, cfg) in cfgs.iter().enumerate() {
        //     let cfgp = tempd.path().join(i.to_string());
        //     std::fs::write(&cfgp, cfg.as_yacc())
        //         .expect(&format!("Failed to write cfg {}", cfg));
        //     let g = CfgGraph::new(cfg.clone());
        //     let g_result = g.instantiate()
        //         .expect("Unable to convert cfg to graph");
        //     println!("=>graph: {}", g_result);
        // }
        //
        // tempd.close().expect("Unable to close temp dir");
    }
}
