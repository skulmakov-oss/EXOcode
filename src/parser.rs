use crate::{QuadroReg, F, N, S, T};
use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Or,
    Xor,
    And,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr<'a> {
    State(u8),
    Ident(&'a str),
    Unary {
        op: UnaryOp,
        expr: Box<Expr<'a>>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr<'a>>,
        right: Box<Expr<'a>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment<'a> {
    pub target: &'a str,
    pub expr: Expr<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub position: usize,
    pub message: &'static str,
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "parse error at {}: {}", self.position, self.message)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalError<'a> {
    pub ident: &'a str,
    pub message: &'static str,
}

impl<'a> core::fmt::Display for EvalError<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "eval error for '{}': {}", self.ident, self.message)
    }
}

#[cfg(feature = "std")]
impl<'a> std::error::Error for EvalError<'a> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramError {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl core::fmt::Display for ProgramError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "program error at line {}, column {}: {}",
            self.line, self.column, self.message
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ProgramError {}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParserProfile {
    pub aliases: HashMap<String, String>,
}

impl ParserProfile {
    pub fn add_alias(&mut self, raw: impl Into<String>, canonical: impl Into<String>) {
        self.aliases.insert(raw.into(), canonical.into());
    }

    pub fn normalize(&self, input: &str) -> String {
        let tokens = lex_tokens(input);
        if tokens.is_empty() {
            return String::new();
        }

        let mut out = String::new();
        for (i, tok) in tokens.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            if let Some(mapped) = self.aliases.get(*tok) {
                out.push_str(mapped);
            } else {
                out.push_str(tok);
            }
        }
        out
    }

    pub fn to_json(&self) -> Result<String, ProfileIoError> {
        serde_json::to_string_pretty(self).map_err(ProfileIoError::Json)
    }

    pub fn from_json(json: &str) -> Result<Self, ProfileIoError> {
        serde_json::from_str(json).map_err(ProfileIoError::Json)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ProfileIoError> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(ProfileIoError::Io)
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ProfileIoError> {
        let json = std::fs::read_to_string(path).map_err(ProfileIoError::Io)?;
        Self::from_json(&json)
    }
}

#[derive(Debug)]
pub enum ProfileIoError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl core::fmt::Display for ProfileIoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProfileIoError::Io(e) => write!(f, "I/O error: {}", e),
            ProfileIoError::Json(e) => write!(f, "JSON error: {}", e),
        }
    }
}

impl std::error::Error for ProfileIoError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingSample<'a> {
    pub input: &'a str,
    pub target: &'a str,
}

pub fn train_profile(samples: &[TrainingSample<'_>]) -> ParserProfile {
    let mut profile = ParserProfile::default();
    train_profile_in_place(&mut profile, samples);
    profile
}

pub fn train_profile_in_place(profile: &mut ParserProfile, samples: &[TrainingSample<'_>]) {
    for sample in samples {
        let raw_tokens = lex_tokens(sample.input);
        let target_tokens = lex_tokens(sample.target);
        if raw_tokens.len() != target_tokens.len() {
            continue;
        }

        for (raw, canonical) in raw_tokens.iter().zip(target_tokens.iter()) {
            if raw == canonical {
                continue;
            }
            if can_learn_alias(raw, canonical) {
                profile
                    .aliases
                    .entry((*raw).to_string())
                    .or_insert_with(|| (*canonical).to_string());
            }
        }
    }
}

pub fn parse_expression(input: &str) -> Result<Expr<'_>, ParseError> {
    let mut p = Parser::new(input);
    let expr = p.parse_expr()?;
    p.skip_ws();
    if p.is_eof() {
        Ok(expr)
    } else {
        Err(p.err("unexpected trailing input"))
    }
}

pub fn parse_assignment(input: &str) -> Result<Assignment<'_>, ParseError> {
    let mut p = Parser::new(input);
    let target = p.parse_identifier()?;
    p.skip_ws();
    if !p.eat_byte(b'=') {
        return Err(p.err("expected '=' after identifier"));
    }
    let expr = p.parse_expr()?;
    p.skip_ws();
    if !p.is_eof() {
        return Err(p.err("unexpected trailing input"));
    }
    Ok(Assignment { target, expr })
}

pub fn eval_expression<'a, R>(
    expr: &'a Expr<'a>,
    resolve_ident: &R,
) -> Result<QuadroReg, EvalError<'a>>
where
    R: Fn(&'a str) -> Option<QuadroReg>,
{
    match expr {
        Expr::State(state) => Ok(fill_state(*state)),
        Expr::Ident(name) => resolve_ident(name).ok_or(EvalError {
            ident: name,
            message: "unknown identifier",
        }),
        Expr::Unary { op, expr } => {
            let v = eval_expression(expr, resolve_ident)?;
            match op {
                UnaryOp::Not => Ok(v.inverse()),
            }
        }
        Expr::Binary { op, left, right } => {
            let l = eval_expression(left, resolve_ident)?;
            let r = eval_expression(right, resolve_ident)?;
            match op {
                BinaryOp::Or => Ok(l.merge(r)),
                BinaryOp::Xor => Ok(l.raw_delta(r)),
                BinaryOp::And => Ok(l.intersect(r)),
            }
        }
    }
}

pub fn eval_assignment<'a, R>(
    assignment: &'a Assignment<'a>,
    resolve_ident: &R,
) -> Result<(&'a str, QuadroReg), EvalError<'a>>
where
    R: Fn(&'a str) -> Option<QuadroReg>,
{
    let value = eval_expression(&assignment.expr, resolve_ident)?;
    Ok((assignment.target, value))
}

pub fn execute_program(
    program: &str,
    env: &mut HashMap<String, QuadroReg>,
) -> Result<(), ProgramError> {
    for (line_idx, raw_line) in program.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        let assignment = parse_assignment(line).map_err(|e| ProgramError {
            line: line_no,
            column: e.position + 1,
            message: e.message.to_string(),
        })?;

        let (target, value) = eval_assignment(&assignment, &|name| env.get(name).copied())
            .map_err(|e| ProgramError {
                line: line_no,
                column: 1,
                message: format!("{} ('{}')", e.message, e.ident),
            })?;

        env.insert(target.to_string(), value);
    }

    Ok(())
}

pub fn execute_program_with_profile(
    program: &str,
    env: &mut HashMap<String, QuadroReg>,
    profile: &ParserProfile,
) -> Result<(), ProgramError> {
    for (line_idx, raw_line) in program.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        let normalized = profile.normalize(line);
        let assignment = parse_assignment(&normalized).map_err(|e| ProgramError {
            line: line_no,
            column: e.position + 1,
            message: e.message.to_string(),
        })?;

        let (target, value) = eval_assignment(&assignment, &|name| env.get(name).copied())
            .map_err(|e| ProgramError {
                line: line_no,
                column: 1,
                message: format!("{} ('{}')", e.message, e.ident),
            })?;

        env.insert(target.to_string(), value);
    }

    Ok(())
}

#[inline]
fn strip_comment(line: &str) -> &str {
    let slash = line.find("//");
    let hash = line.find('#');
    match (slash, hash) {
        (Some(a), Some(b)) => &line[..a.min(b)],
        (Some(a), None) => &line[..a],
        (None, Some(b)) => &line[..b],
        (None, None) => line,
    }
}

fn can_learn_alias(raw: &str, canonical: &str) -> bool {
    is_alias_input_token(raw) && is_supported_canonical_token(canonical)
}

fn is_alias_input_token(tok: &str) -> bool {
    !tok.is_empty()
        && tok
            .as_bytes()
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || *b == b'_')
}

fn is_supported_canonical_token(tok: &str) -> bool {
    matches!(tok, "!" | "&" | "|" | "^" | "N" | "F" | "T" | "S")
}

fn lex_tokens(input: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let bytes = input.as_bytes();

    while i < bytes.len() {
        let ch = bytes[i];
        if ch.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        if i + 1 < bytes.len() && ch == b'/' && bytes[i + 1] == b'/' {
            break;
        }
        if ch == b'#' {
            break;
        }

        if is_single_char_token(ch) {
            out.push(&input[i..i + 1]);
            i += 1;
            continue;
        }

        let start = i;
        i += 1;
        while i < bytes.len() {
            let c = bytes[i];
            if c.is_ascii_whitespace() || is_single_char_token(c) || c == b'#' {
                break;
            }
            if i + 1 < bytes.len() && c == b'/' && bytes[i + 1] == b'/' {
                break;
            }
            i += 1;
        }
        out.push(&input[start..i]);
    }

    out
}

#[inline]
fn is_single_char_token(ch: u8) -> bool {
    matches!(ch, b'(' | b')' | b'!' | b'&' | b'|' | b'^' | b'=')
}

#[inline]
fn fill_state(state: u8) -> QuadroReg {
    debug_assert!(state <= 0b11);
    match state {
        N => QuadroReg::from_raw(0),
        F => QuadroReg::from_raw(crate::LSB_MASK),
        T => QuadroReg::from_raw(crate::MSB_MASK),
        S => QuadroReg::from_raw(u64::MAX),
        _ => unreachable!("invalid quadit state"),
    }
}

struct Parser<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn err(&self, message: &'static str) -> ParseError {
        ParseError {
            position: self.pos,
            message,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }

    fn eat_byte(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr<'a>, ParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr<'a>, ParseError> {
        let mut left = self.parse_xor()?;
        loop {
            self.skip_ws();
            if !self.eat_byte(b'|') {
                break;
            }
            let right = self.parse_xor()?;
            left = Expr::Binary {
                op: BinaryOp::Or,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_xor(&mut self) -> Result<Expr<'a>, ParseError> {
        let mut left = self.parse_and()?;
        loop {
            self.skip_ws();
            if !self.eat_byte(b'^') {
                break;
            }
            let right = self.parse_and()?;
            left = Expr::Binary {
                op: BinaryOp::Xor,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr<'a>, ParseError> {
        let mut left = self.parse_unary()?;
        loop {
            self.skip_ws();
            if !self.eat_byte(b'&') {
                break;
            }
            let right = self.parse_unary()?;
            left = Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr<'a>, ParseError> {
        self.skip_ws();
        if self.eat_byte(b'!') {
            return Ok(Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(self.parse_unary()?),
            });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr<'a>, ParseError> {
        self.skip_ws();
        match self.peek() {
            Some(b'(') => {
                self.bump();
                let expr = self.parse_expr()?;
                self.skip_ws();
                if !self.eat_byte(b')') {
                    return Err(self.err("expected ')'"));
                }
                Ok(expr)
            }
            Some(ch) if is_ident_start(ch) => {
                let ident = self.parse_identifier()?;
                match ident {
                    "N" => Ok(Expr::State(N)),
                    "F" => Ok(Expr::State(F)),
                    "T" => Ok(Expr::State(T)),
                    "S" => Ok(Expr::State(S)),
                    _ => Ok(Expr::Ident(ident)),
                }
            }
            Some(_) => Err(self.err("unexpected token")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn parse_identifier(&mut self) -> Result<&'a str, ParseError> {
        self.skip_ws();
        let start = self.pos;
        let first = self.peek().ok_or_else(|| self.err("expected identifier"))?;
        if !is_ident_start(first) {
            return Err(self.err("expected identifier"));
        }
        self.pos += 1;
        while let Some(ch) = self.peek() {
            if is_ident_continue(ch) {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.src
            .get(start..self.pos)
            .ok_or_else(|| self.err("invalid utf-8 boundary"))
    }
}

#[inline]
fn is_ident_start(ch: u8) -> bool {
    ch.is_ascii_alphabetic() || ch == b'_'
}

#[inline]
fn is_ident_continue(ch: u8) -> bool {
    ch.is_ascii_alphanumeric() || ch == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_respects_precedence() {
        let expr = parse_expression("T | F & N ^ S").expect("must parse");
        let expected = Expr::Binary {
            op: BinaryOp::Or,
            left: Box::new(Expr::State(T)),
            right: Box::new(Expr::Binary {
                op: BinaryOp::Xor,
                left: Box::new(Expr::Binary {
                    op: BinaryOp::And,
                    left: Box::new(Expr::State(F)),
                    right: Box::new(Expr::State(N)),
                }),
                right: Box::new(Expr::State(S)),
            }),
        };
        assert_eq!(expr, expected);
    }

    #[test]
    fn parse_parentheses_and_unary() {
        let expr = parse_expression("!(a | T) & _x2").expect("must parse");
        let expected = Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(Expr::Binary {
                    op: BinaryOp::Or,
                    left: Box::new(Expr::Ident("a")),
                    right: Box::new(Expr::State(T)),
                }),
            }),
            right: Box::new(Expr::Ident("_x2")),
        };
        assert_eq!(expr, expected);
    }

    #[test]
    fn parse_assignment_ok() {
        let parsed = parse_assignment("out = a & !F").expect("must parse");
        assert_eq!(parsed.target, "out");
        assert_eq!(
            parsed.expr,
            Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(Expr::Ident("a")),
                right: Box::new(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(Expr::State(F)),
                }),
            }
        );
    }

    #[test]
    fn parse_errors_are_reported() {
        let err = parse_expression("(T | F").expect_err("must fail");
        assert_eq!(err.message, "expected ')'");

        let err = parse_assignment("1bad = T").expect_err("must fail");
        assert_eq!(err.message, "expected identifier");
    }

    #[test]
    fn eval_expression_operators_work_on_registers() {
        let mut vars = HashMap::new();
        let mut a = QuadroReg::new();
        a = a.set_by_mask(1_u64 << (0 * 2), T);
        a = a.set_by_mask(1_u64 << (1 * 2), F);
        let mut b = QuadroReg::new();
        b = b.set_by_mask(1_u64 << (0 * 2), F);
        b = b.set_by_mask(1_u64 << (1 * 2), T);
        vars.insert("a", a);
        vars.insert("b", b);

        let expr = parse_expression("!(a & b) | (a ^ b)").expect("must parse");
        let got = eval_expression(&expr, &|name| vars.get(name).copied()).expect("must eval");

        let expected = a.intersect(b).inverse().merge(a.raw_delta(b));
        assert_eq!(got.raw(), expected.raw());
    }

    #[test]
    fn eval_assignment_returns_target_and_value() {
        let mut vars = HashMap::new();
        let x = QuadroReg::from_raw(0x1234_5678_9abc_def0);
        let y = QuadroReg::from_raw(0x0f0f_f0f0_55aa_a55a);
        vars.insert("x", x);
        vars.insert("y", y);

        let stmt = parse_assignment("out = x & y").expect("must parse");
        let (target, value) =
            eval_assignment(&stmt, &|name| vars.get(name).copied()).expect("must eval");

        assert_eq!(target, "out");
        assert_eq!(value.raw(), x.intersect(y).raw());
    }

    #[test]
    fn eval_unknown_identifier_fails() {
        let expr = parse_expression("a | T").expect("must parse");
        let err = eval_expression(&expr, &|_| None).expect_err("must fail");
        assert_eq!(err.ident, "a");
        assert_eq!(err.message, "unknown identifier");
    }

    #[test]
    fn execute_program_updates_context_in_order() {
        let mut env = HashMap::<String, QuadroReg>::new();
        let program = r#"
			a = T
			b = !a
			c = a ^ b
		"#;

        execute_program(program, &mut env).expect("must execute");

        assert_eq!(
            env.get("a").copied(),
            Some(QuadroReg::from_raw(crate::MSB_MASK))
        );
        assert_eq!(
            env.get("b").copied(),
            Some(QuadroReg::from_raw(crate::LSB_MASK))
        );
        assert_eq!(env.get("c").copied(), Some(QuadroReg::from_raw(u64::MAX)));
    }

    #[test]
    fn execute_program_supports_comments_and_blanks() {
        let mut env = HashMap::<String, QuadroReg>::new();
        let program = r#"
			# set initial value
			x = F // inline comment

			y = !x
		"#;

        execute_program(program, &mut env).expect("must execute");
        assert_eq!(
            env.get("x").copied(),
            Some(QuadroReg::from_raw(crate::LSB_MASK))
        );
        assert_eq!(
            env.get("y").copied(),
            Some(QuadroReg::from_raw(crate::MSB_MASK))
        );
    }

    #[test]
    fn execute_program_reports_line_for_eval_error() {
        let mut env = HashMap::<String, QuadroReg>::new();
        let program = "a = T\nb = z & a";
        let err = execute_program(program, &mut env).expect_err("must fail");

        assert_eq!(err.line, 2);
        assert_eq!(err.column, 1);
        assert!(err.message.contains("unknown identifier"));
        assert!(err.message.contains("z"));
    }

    #[test]
    fn execute_program_reports_line_for_parse_error() {
        let mut env = HashMap::<String, QuadroReg>::new();
        let program = "a = T\nbad line";
        let err = execute_program(program, &mut env).expect_err("must fail");

        assert_eq!(err.line, 2);
        assert!(err.column >= 1);
    }

    #[test]
    fn train_profile_learns_aliases_from_examples() {
        let samples = [
            TrainingSample {
                input: "out = a AND b",
                target: "out = a & b",
            },
            TrainingSample {
                input: "q = a OR b",
                target: "q = a | b",
            },
            TrainingSample {
                input: "x = NOT FALSE",
                target: "x = ! F",
            },
            TrainingSample {
                input: "y = TRUE XOR x",
                target: "y = T ^ x",
            },
        ];
        let profile = train_profile(&samples);

        assert_eq!(profile.aliases.get("AND"), Some(&"&".to_string()));
        assert_eq!(profile.aliases.get("OR"), Some(&"|".to_string()));
        assert_eq!(profile.aliases.get("NOT"), Some(&"!".to_string()));
        assert_eq!(profile.aliases.get("TRUE"), Some(&"T".to_string()));
        assert_eq!(profile.aliases.get("FALSE"), Some(&"F".to_string()));
        assert_eq!(profile.aliases.get("XOR"), Some(&"^".to_string()));
        assert_eq!(profile.normalize("z = TRUE AND NOT a"), "z = T & ! a");
    }

    #[test]
    fn execute_program_with_profile_uses_trained_aliases() {
        let samples = [
            TrainingSample {
                input: "out = a AND b",
                target: "out = a & b",
            },
            TrainingSample {
                input: "q = a OR b",
                target: "q = a | b",
            },
            TrainingSample {
                input: "x = NOT FALSE",
                target: "x = ! F",
            },
            TrainingSample {
                input: "y = TRUE XOR x",
                target: "y = T ^ x",
            },
        ];
        let profile = train_profile(&samples);

        let mut env = HashMap::<String, QuadroReg>::new();
        env.insert("a".to_string(), QuadroReg::from_raw(crate::MSB_MASK));
        env.insert("b".to_string(), QuadroReg::from_raw(crate::LSB_MASK));

        let program = r#"
			x = NOT FALSE
			y = TRUE XOR x
			out = y OR (a AND b)
		"#;
        execute_program_with_profile(program, &mut env, &profile).expect("must execute");

        let x = QuadroReg::from_raw(crate::LSB_MASK).inverse();
        let y = QuadroReg::from_raw(crate::MSB_MASK).raw_delta(x);
        let out = y.merge(
            QuadroReg::from_raw(crate::MSB_MASK).intersect(QuadroReg::from_raw(crate::LSB_MASK)),
        );
        assert_eq!(env.get("out").copied(), Some(out));
    }

    #[test]
    fn profile_json_roundtrip() {
        let mut profile = ParserProfile::default();
        profile.add_alias("AND", "&");
        profile.add_alias("TRUE", "T");

        let json = profile.to_json().expect("serialize");
        let restored = ParserProfile::from_json(&json).expect("deserialize");
        assert_eq!(restored, profile);
    }

    #[test]
    fn profile_save_and_load_file() {
        let mut profile = ParserProfile::default();
        profile.add_alias("OR", "|");
        profile.add_alias("NOT", "!");

        let path = std::env::temp_dir().join("exocode_parser_profile_test.json");
        profile.save_to_file(&path).expect("save");
        let loaded = ParserProfile::load_from_file(&path).expect("load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded, profile);
    }
}
