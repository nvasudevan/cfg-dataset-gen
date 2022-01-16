use lazy_static::lazy_static;
use regex::Regex;

use crate::cfg::{Cfg, CfgRule, EpsilonSymbol, LexSymbol, NonTermSymbol, RuleAlt, TermSymbol, CfgError};
use std::{fmt, fs};
use std::fmt::{format, Formatter};

const RULE_END_MARKER: char = ';';

pub(crate) struct CfgParser {
    tokens: Vec<char>,
    sym_tokens: Option<Vec<String>>,
    start_symbol: String,
}

impl fmt::Display for CfgParser {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut s = format!("\n--\n%start {}", self.start_symbol);

        if let Some(sym_tokens) = &self.sym_tokens {
            let token_s = sym_tokens.join(" ");
            s = format!("{}\n\n%token {}", s, token_s);
        }

        write!(f, "{}\n--", s)
    }
}

impl CfgParser {
    fn new(tokens: Vec<char>) -> Self {
        Self {
            tokens,
            sym_tokens: None,
            start_symbol: String::new(),
        }
    }

    fn add_start_symbol(&mut self, start_symbol: String) {
        self.start_symbol = start_symbol;
    }

    fn add_sym_tokens(&mut self, sym_tokens: Vec<String>) {
        self.sym_tokens = Some(sym_tokens);
    }

    // TOKENS

    fn term_token_regex(&self, s: &str) -> Option<TermSymbol> {
        lazy_static! {
            static ref RE_TERMINAL: Regex = Regex::new(r"'(?P<tok>[a-zA-Z%=\+\-\*]+)'")
            .expect("Unable to create regex for parsing a terminal token");
        }

        let cap = RE_TERMINAL.captures(s)?;
        let tok = cap.name("tok")?.as_str();

        Some(TermSymbol::new(tok.to_owned()))
    }

    fn non_term_token_regex(&self, s: &str) -> Option<NonTermSymbol> {
        lazy_static! {
            static ref RE_NON_TERMINAL: Regex = Regex::new(r"(?P<tok>[a-zA-Z_]+)")
            .expect("Unable to create regex to parse a non-terminal token");
        }

        let cap = RE_NON_TERMINAL.captures(s)?;
        let tok = cap.name("tok")?.as_str();

        Some(NonTermSymbol::new(tok.to_owned()))
    }

    fn parse_lex_symbols(&self, s: &str) -> Option<LexSymbol> {
        if let Some(sym) = self.term_token_regex(s) {
            return Some(LexSymbol::Term(sym));
        }

        if let Some(sym) = self.non_term_token_regex(s) {
            return Some(LexSymbol::NonTerm(sym));
        }

        None
    }

    /// Parse bison/YACC directives:
    /// - %define - marks the definition part
    /// - %start - marks the start rule part
    /// - %% - has two of these, marks the begin and end of rules section.
    fn header_directives(&mut self, i: usize) -> Result<usize, CfgError> {
        lazy_static! {
            static ref RE_HEADER: Regex = Regex::new(
            r"[\n\r\s]*%define\s+([a-zA-Z\.]+)\s+([a-zA-Z\-]+)[\n\r\s]+%start\s+(?P<start>[a-zA-Z]+)[\n\r\s]+(%token\s+(?P<tokens>[A-Z_\s]+))?[\n\r\s]+%%"
            ).expect("Unable to create RE_HEADER regex");
        }
        let s_chars = self.tokens.as_slice();
        let mut j = i;
        while j < s_chars.len() {
            let c = s_chars.get(j)
                .ok_or_else(|| CfgError::new("Unable to read tokens from input".to_owned()))?;
            if *c == '%' {
                // is the next char '%' too?
                let c_next = s_chars.get(j + 1)
                    .ok_or_else(||
                        CfgError::new("Error whilst parsing the header directives at the start of grammar!".to_owned())
                    )?;
                if *c_next == '%' {
                    // j+2 -- so we read until %%
                    let s: String = s_chars.get(i..j + 2)
                        .ok_or_else(||
                            CfgError::new(
                                format!("Error whilst reading input between {} and {}", i, j + 2)))?
                        .iter()
                        .collect();
                    let cap = RE_HEADER.captures(&s)
                        .ok_or_else(|| CfgError::new(
                            format!("Capture failed for REGEX 'RE_HEADER' for {}", s)
                        ))?;
                    // println!("\n=>cap: {:?}", cap);
                    // process start symbol
                    let start_sym: &str = cap.name("start")
                        .ok_or_else(|| CfgError::new(
                            "Unable to extract 'start' directive from header".to_string()))?
                        .as_str();
                    self.add_start_symbol(start_sym.to_owned());

                    // process token list
                    if let Some(m_tokens) = cap.name("tokens") {
                        let _tokens: &str = m_tokens.as_str();
                        let tokens_v: Vec<String> = _tokens.split_ascii_whitespace()
                            .map(|tk| tk.to_owned())
                            .collect();
                        self.add_sym_tokens(tokens_v);
                    }

                    return Ok(j + 2);
                }
            }
            j += 1;
        }

        Ok(i)
    }

    /// Parses the footer `%%` section, thus marking the end of parsing
    fn footer_tag(&self, i: usize) -> Result<(), CfgError> {
        lazy_static! {
            static ref RE_FOOTER: Regex = Regex::new( r"[\n\r\s]*%%")
            .expect("Unable to create RE_FOOTER regex");
        }
        let s_chars = self.tokens.as_slice();
        let peep = s_chars.get(i..)
            .ok_or_else(|| CfgError::new("Failed to read char from input!".to_owned()))?;
        let s: String = peep.iter().collect();
        RE_FOOTER.captures(&s)
            .ok_or_else(||
                CfgError::new("Failed whilst parsing the footer directive!".to_owned())
            )?;

        Ok(())
    }

    // RULES

    /// read until `;`, otherwise we have reached end of rule section
    fn rule_end_marker(&self, i: usize) -> Option<usize> {
        let s_chars = self.tokens.as_slice();
        let mut j = i;
        while j < s_chars.len() {
            let c = s_chars.get(j)?;
            if *c == RULE_END_MARKER {
                return Some(j);
            }
            j += 1;
        }

        None
    }

    fn parse_alt(&self, s: &str) -> Result<Vec<LexSymbol>, CfgError> {
        let tokens_s: Vec<&str> = s.split_ascii_whitespace().collect();
        let mut syms: Vec<LexSymbol> = vec![];
        for tok in tokens_s {
            println!("tok: {}", tok);
            let sym = self.parse_lex_symbols(tok)
                .ok_or_else(||
                    CfgError::new("Token is neither terminal or non-terminal!".to_owned()))?;
            syms.push(sym);
        }

        Ok(syms)
    }

    /// S: A 'b' C | 'x' | ;
    fn rule_rhs_regex(&self, s: &str) -> Result<Vec<RuleAlt>, CfgError> {
        lazy_static! {
            static ref RE_ALT: Regex = Regex::new(r"(?P<alt>[\w\s%='\+\-\*/]+)")
            .expect("Unable to create regex for parsing rule alternatives");
            static ref RE_EMPTY_ALT: Regex = Regex::new(r"(?P<empty>\s*)")
            .expect("Unable to create regex to parse an empty alternative");
        }
        println!("=> [rhs_regex] {}", s);

        let alts: Vec<&str> = s.split('|').map(|alt| alt.trim()).collect();
        let mut alts_s: Vec<RuleAlt> = vec![];
        for alt in alts {
            println!("=> alt: {}", alt);
            match RE_ALT.captures(alt) {
                Some(cap) => {
                    let alt_s = cap.name("alt")
                        .ok_or_else(||
                            CfgError::new("Failed to create regex capture to parse as an alternative".to_owned())
                        )?.as_str();
                    println!("alt_s: {}", alt_s);
                    let alts_syms = self.parse_alt(alt_s)?;
                    alts_s.push(RuleAlt::new(alts_syms));
                }
                _ => {
                    // try empty alt
                    let empty_alt_cap = RE_EMPTY_ALT.captures(alt)
                        .ok_or_else(||
                            CfgError::new("Failed to create a Regex capture to parse as an empty alternative!".to_owned()))?;
                    let _ = empty_alt_cap.name("empty")
                        .ok_or_else(||
                            CfgError::new("Failed to generate an empty alternative!".to_owned())
                        )?.as_str();
                    let alt_syms: Vec<LexSymbol> = vec![LexSymbol::Epsilon(EpsilonSymbol::new())];
                    alts_s.push(RuleAlt::new(alt_syms));
                }
            }
        }

        Ok(alts_s)
    }

    /// Parse the given string `s` as a rule, returns lhs and rhs of the rule.
    fn rule_regex<'a>(&self, s: &'a str) -> Result<(&'a str, &'a str), CfgError> {
        println!("[rule_regex] s: {}", s);
        lazy_static! {
             static ref RE_RULE: Regex = Regex::new(r"[\n\r\s]+(?P<lhs>[_\-a-zA-Z]+)[\s]*:(?P<rhs>[_a-zA-Z%='\|\s\+\-\*]+)[\s]*;")
            .expect("Unable to create regex");
        }
        let cap = RE_RULE.captures(s)
            .ok_or_else(|| CfgError::new("Failed to capture rule!".to_owned())
            )?;
        let lhs = cap.name("lhs")
            .ok_or_else(|| CfgError::new("Unable to capture LHS of a rule using regex".to_owned()))?
            .as_str();
        println!("lhs: {}", lhs);
        let rhs = cap.name("rhs")
            .ok_or_else(|| CfgError::new("Unable to catpure RHS of a rule using regex".to_owned()))?
            .as_str();
        println!("rhs: {}", rhs);

        Ok((lhs, rhs))
    }

    /// Create a rule
    /// From `i`, tries the rule end marker `;`, if found, returns the rule.
    /// Otherwise, returns `i`.
    fn rule(&self, i: usize) -> Result<(usize, CfgRule), CfgError> {
        let s_chars = self.tokens.as_slice();
        let j = self.rule_end_marker(i)
            .ok_or_else(|| CfgError::new("Unable to find rule end marker".to_owned()))?;
        println!("=> [rule], j: {}", j);
        let rule = s_chars.get(i..j + 1)
            .ok_or_else(||
                CfgError::new("Failed to read from input!".to_owned())
            )?;
        let rule_s: String = rule.iter().collect();
        println!("=> [rule]: *{}*", rule_s);
        let (lhs, rhs) = self.rule_regex(&rule_s)?;
        let alts = self.rule_rhs_regex(rhs)?;
        let cfg_rule = CfgRule::new(lhs.to_owned(), alts);

        Ok((j, cfg_rule))
    }

    /// Parses the rules section between the `%%` markers
    /// First read the root rule and then parse the rest of the rules
    fn parse_rules(&mut self, i: usize) -> Result<(usize, Vec<CfgRule>), CfgError> {
        println!("=> tokens: {:?}", self.tokens);
        println!("parse rules from {}", i);
        let s_chars = self.tokens.as_slice();
        let mut j = i;
        let mut rules: Vec<CfgRule> = vec![];
        // start off with root rule
        let (k, root_rule) = self.rule(j)?;
        rules.push(root_rule);
        j = k + 1;
        println!("j: {} (max={})", j, s_chars.len());
        while j < s_chars.len() {
            match self.rule(j) {
                Ok((k, rule)) => {
                    println!("k: {}, rule: {}", k, rule);
                    rules.push(rule);
                    j = k + 1;
                }
                Err(_) => {
                    // we have reached the end of the rules section
                    break;
                }
            }
        }

        Ok((j + 1, rules))
    }

    fn run(&mut self) -> Result<Cfg, CfgError> {
        let i = self.header_directives(0)?;
        let (j, rules) = self.parse_rules(i)?;
        self.footer_tag(j)?;

        Ok(Cfg::new(rules))
    }
}

pub(crate) fn parse(cfgp: &str) -> Result<Cfg, CfgError> {
    let s = fs::read_to_string(cfgp)
        .map_err(|x| CfgError::new(x.to_string()))?;
    let s_chars: Vec<char> = s.chars().into_iter().collect();
    let mut parser = CfgParser::new(s_chars);
    let cfg = parser.run()?;
    println!("cfg: {}", cfg);

    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use regex::Regex;

    use super::*;

    #[test]
    fn test_cfg() {
        let s = fs::read_to_string("./grammars/test.y")
            .expect("Unable to read grammar file");
        let s_chars: Vec<char> = s.chars().into_iter().collect();
        let mut cfg_parser = CfgParser::new(s_chars);
        let cfg = cfg_parser.run().expect("Parse failed!");
        println!("cfg:\n{}", cfg);
    }

    #[test]
    fn test_header_regex() {
        let s = "%define lr.type canonical-lr\n%start root\n%%";
        // let re = Regex::new(r"(?P<lhs>[a-zA-Z]+):(?P<rhs>[a-zA-z'|\s]+)[\s]*;").expect("Unable to create regex");
        let re = Regex::new(
            r"%[a-zA-Z]+\s+([a-zA-Z\.]+)\s+([a-zA-Z\-]+)[\n\r\s]+%[a-zA-Z]+\s+(?P<startsym>[a-zA-Z]+)[\n\r\s]+%%"
        ).expect("Unable to create regex");
        let cap = re.captures(s)
            .expect("Unable to create capture");
        println!("cap: {:?}", cap);
    }
}