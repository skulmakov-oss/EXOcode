use crate::parser::{parse_assignment, BinaryOp as PBinaryOp, Expr, ParserProfile, UnaryOp};
use crate::{QuadroReg, F, N, S, T};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineProgram {
    pub instructions: Vec<MachineInstr>,
}

impl MachineProgram {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MachineInstr {
    SetState {
        dst: String,
        state: u8,
    },
    Mov {
        dst: String,
        src: String,
    },
    Not {
        dst: String,
        src: String,
    },
    And {
        dst: String,
        lhs: String,
        rhs: String,
    },
    Or {
        dst: String,
        lhs: String,
        rhs: String,
    },
    Xor {
        dst: String,
        lhs: String,
        rhs: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "compile error at line {}, column {}: {}",
            self.line, self.column, self.message
        )
    }
}

impl std::error::Error for CompileError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineParseError {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl core::fmt::Display for MachineParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "machine parse error at line {}, column {}: {}",
            self.line, self.column, self.message
        )
    }
}

impl std::error::Error for MachineParseError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeError {
    pub ip: usize,
    pub message: String,
}

impl core::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "runtime error at instruction {}: {}",
            self.ip, self.message
        )
    }
}

impl std::error::Error for RuntimeError {}

pub fn compile_human_program(
    program: &str,
    profile: Option<&ParserProfile>,
) -> Result<MachineProgram, CompileError> {
    let mut out = MachineProgram::new();
    let mut lower = LoweringCtx::new();

    for (line_idx, raw_line) in program.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        let normalized = if let Some(p) = profile {
            p.normalize(line)
        } else {
            line.to_string()
        };

        let assignment = parse_assignment(&normalized).map_err(|e| CompileError {
            line: line_no,
            column: e.position + 1,
            message: e.message.to_string(),
        })?;

        let src = lower.lower_expr(&assignment.expr);
        let dst = assignment.target.to_string();
        if src != dst {
            lower.instructions.push(MachineInstr::Mov { dst, src });
        }
    }

    out.instructions = optimize_machine_program(&MachineProgram {
        instructions: lower.instructions,
    })
    .instructions;
    Ok(out)
}

pub fn optimize_machine_program(program: &MachineProgram) -> MachineProgram {
    let folded = fold_and_simplify(&program.instructions);
    let dce = eliminate_dead_temps(&folded);
    MachineProgram { instructions: dce }
}

pub fn render_machine_program(program: &MachineProgram) -> String {
    let mut out = String::new();
    for instr in &program.instructions {
        match instr {
            MachineInstr::SetState { dst, state } => {
                out.push_str("SET ");
                out.push_str(dst);
                out.push(' ');
                out.push_str(match *state {
                    N => "N",
                    F => "F",
                    T => "T",
                    S => "S",
                    _ => "N",
                });
            }
            MachineInstr::Mov { dst, src } => {
                out.push_str("MOV ");
                out.push_str(dst);
                out.push(' ');
                out.push_str(src);
            }
            MachineInstr::Not { dst, src } => {
                out.push_str("NOT ");
                out.push_str(dst);
                out.push(' ');
                out.push_str(src);
            }
            MachineInstr::And { dst, lhs, rhs } => {
                out.push_str("AND ");
                out.push_str(dst);
                out.push(' ');
                out.push_str(lhs);
                out.push(' ');
                out.push_str(rhs);
            }
            MachineInstr::Or { dst, lhs, rhs } => {
                out.push_str("OR ");
                out.push_str(dst);
                out.push(' ');
                out.push_str(lhs);
                out.push(' ');
                out.push_str(rhs);
            }
            MachineInstr::Xor { dst, lhs, rhs } => {
                out.push_str("XOR ");
                out.push_str(dst);
                out.push(' ');
                out.push_str(lhs);
                out.push(' ');
                out.push_str(rhs);
            }
        }
        out.push('\n');
    }
    out
}

pub fn parse_machine_program(src: &str) -> Result<MachineProgram, MachineParseError> {
    let mut program = MachineProgram::new();

    for (line_idx, raw_line) in src.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let op = parts[0].to_ascii_uppercase();
        let instr = match op.as_str() {
            "SET" => {
                if parts.len() != 3 {
                    return Err(parse_err(line_no, "SET expects: SET <dst> <N|F|T|S>"));
                }
                let state = parse_state(parts[2])
                    .ok_or_else(|| parse_err(line_no, "invalid SET state, expected N|F|T|S"))?;
                MachineInstr::SetState {
                    dst: parts[1].to_string(),
                    state,
                }
            }
            "MOV" => {
                if parts.len() != 3 {
                    return Err(parse_err(line_no, "MOV expects: MOV <dst> <src>"));
                }
                MachineInstr::Mov {
                    dst: parts[1].to_string(),
                    src: parts[2].to_string(),
                }
            }
            "NOT" => {
                if parts.len() != 3 {
                    return Err(parse_err(line_no, "NOT expects: NOT <dst> <src>"));
                }
                MachineInstr::Not {
                    dst: parts[1].to_string(),
                    src: parts[2].to_string(),
                }
            }
            "AND" | "OR" | "XOR" => {
                if parts.len() != 4 {
                    return Err(parse_err(
                        line_no,
                        "AND/OR/XOR expects: <OP> <dst> <lhs> <rhs>",
                    ));
                }
                match op.as_str() {
                    "AND" => MachineInstr::And {
                        dst: parts[1].to_string(),
                        lhs: parts[2].to_string(),
                        rhs: parts[3].to_string(),
                    },
                    "OR" => MachineInstr::Or {
                        dst: parts[1].to_string(),
                        lhs: parts[2].to_string(),
                        rhs: parts[3].to_string(),
                    },
                    _ => MachineInstr::Xor {
                        dst: parts[1].to_string(),
                        lhs: parts[2].to_string(),
                        rhs: parts[3].to_string(),
                    },
                }
            }
            _ => {
                return Err(parse_err(
                    line_no,
                    "unknown instruction, expected SET|MOV|NOT|AND|OR|XOR",
                ))
            }
        };
        program.instructions.push(instr);
    }

    Ok(program)
}

pub fn execute_machine_program(
    program: &MachineProgram,
    env: &mut HashMap<String, QuadroReg>,
) -> Result<(), RuntimeError> {
    for (ip, instr) in program.instructions.iter().enumerate() {
        match instr {
            MachineInstr::SetState { dst, state } => {
                env.insert(dst.clone(), fill_state(*state));
            }
            MachineInstr::Mov { dst, src } => {
                let v = *env
                    .get(src)
                    .ok_or_else(|| rt_err(ip, format!("unknown variable '{}'", src)))?;
                env.insert(dst.clone(), v);
            }
            MachineInstr::Not { dst, src } => {
                let v = *env
                    .get(src)
                    .ok_or_else(|| rt_err(ip, format!("unknown variable '{}'", src)))?;
                env.insert(dst.clone(), v.inverse());
            }
            MachineInstr::And { dst, lhs, rhs } => {
                let lv = *env
                    .get(lhs)
                    .ok_or_else(|| rt_err(ip, format!("unknown variable '{}'", lhs)))?;
                let rv = *env
                    .get(rhs)
                    .ok_or_else(|| rt_err(ip, format!("unknown variable '{}'", rhs)))?;
                env.insert(dst.clone(), lv.intersect(rv));
            }
            MachineInstr::Or { dst, lhs, rhs } => {
                let lv = *env
                    .get(lhs)
                    .ok_or_else(|| rt_err(ip, format!("unknown variable '{}'", lhs)))?;
                let rv = *env
                    .get(rhs)
                    .ok_or_else(|| rt_err(ip, format!("unknown variable '{}'", rhs)))?;
                env.insert(dst.clone(), lv.merge(rv));
            }
            MachineInstr::Xor { dst, lhs, rhs } => {
                let lv = *env
                    .get(lhs)
                    .ok_or_else(|| rt_err(ip, format!("unknown variable '{}'", lhs)))?;
                let rv = *env
                    .get(rhs)
                    .ok_or_else(|| rt_err(ip, format!("unknown variable '{}'", rhs)))?;
                env.insert(dst.clone(), lv.raw_delta(rv));
            }
        }
    }
    Ok(())
}

struct LoweringCtx {
    instructions: Vec<MachineInstr>,
    next_temp: usize,
}

impl LoweringCtx {
    fn new() -> Self {
        Self {
            instructions: Vec::new(),
            next_temp: 0,
        }
    }

    fn tmp(&mut self) -> String {
        let n = self.next_temp;
        self.next_temp += 1;
        format!("__t{}", n)
    }

    fn lower_expr(&mut self, expr: &Expr<'_>) -> String {
        match expr {
            Expr::State(state) => {
                let dst = self.tmp();
                self.instructions.push(MachineInstr::SetState {
                    dst: dst.clone(),
                    state: *state,
                });
                dst
            }
            Expr::Ident(name) => (*name).to_string(),
            Expr::Unary { op, expr } => {
                let src = self.lower_expr(expr);
                let dst = self.tmp();
                match op {
                    UnaryOp::Not => self.instructions.push(MachineInstr::Not {
                        dst: dst.clone(),
                        src,
                    }),
                }
                dst
            }
            Expr::Binary { op, left, right } => {
                let lhs = self.lower_expr(left);
                let rhs = self.lower_expr(right);
                let dst = self.tmp();
                match op {
                    PBinaryOp::And => self.instructions.push(MachineInstr::And {
                        dst: dst.clone(),
                        lhs,
                        rhs,
                    }),
                    PBinaryOp::Or => self.instructions.push(MachineInstr::Or {
                        dst: dst.clone(),
                        lhs,
                        rhs,
                    }),
                    PBinaryOp::Xor => self.instructions.push(MachineInstr::Xor {
                        dst: dst.clone(),
                        lhs,
                        rhs,
                    }),
                }
                dst
            }
        }
    }
}

#[inline]
fn fill_state(state: u8) -> QuadroReg {
    match state {
        N => QuadroReg::from_raw(0),
        F => QuadroReg::from_raw(crate::LSB_MASK),
        T => QuadroReg::from_raw(crate::MSB_MASK),
        S => QuadroReg::from_raw(u64::MAX),
        _ => QuadroReg::from_raw(0),
    }
}

#[inline]
fn parse_state(s: &str) -> Option<u8> {
    match s {
        "N" => Some(N),
        "F" => Some(F),
        "T" => Some(T),
        "S" => Some(S),
        _ => None,
    }
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

#[inline]
fn parse_err(line: usize, message: &str) -> MachineParseError {
    MachineParseError {
        line,
        column: 1,
        message: message.to_string(),
    }
}

#[inline]
fn rt_err(ip: usize, message: String) -> RuntimeError {
    RuntimeError { ip, message }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    Const(u8),
    Alias(String),
    Unknown,
}

fn fold_and_simplify(input: &[MachineInstr]) -> Vec<MachineInstr> {
    let mut out = Vec::with_capacity(input.len());
    let mut values: HashMap<String, Value> = HashMap::new();

    for instr in input {
        match instr {
            MachineInstr::SetState { dst, state } => {
                values.insert(dst.clone(), Value::Const(*state));
                out.push(MachineInstr::SetState {
                    dst: dst.clone(),
                    state: *state,
                });
            }
            MachineInstr::Mov { dst, src } => {
                let resolved = resolve_alias(src, &values);
                let src_v = read_value(&resolved, &values);
                if dst == &resolved {
                    values.insert(dst.clone(), src_v);
                    continue;
                }
                match src_v {
                    Value::Const(c) => {
                        values.insert(dst.clone(), Value::Const(c));
                        out.push(MachineInstr::SetState {
                            dst: dst.clone(),
                            state: c,
                        });
                    }
                    _ => {
                        values.insert(dst.clone(), Value::Alias(resolved.clone()));
                        out.push(MachineInstr::Mov {
                            dst: dst.clone(),
                            src: resolved,
                        });
                    }
                }
            }
            MachineInstr::Not { dst, src } => {
                let src_r = resolve_alias(src, &values);
                match read_value(&src_r, &values) {
                    Value::Const(c) => {
                        let r = inverse_state(c);
                        values.insert(dst.clone(), Value::Const(r));
                        out.push(MachineInstr::SetState {
                            dst: dst.clone(),
                            state: r,
                        });
                    }
                    _ => {
                        values.insert(dst.clone(), Value::Unknown);
                        out.push(MachineInstr::Not {
                            dst: dst.clone(),
                            src: src_r,
                        });
                    }
                }
            }
            MachineInstr::And { dst, lhs, rhs } => {
                let lhs_r = resolve_alias(lhs, &values);
                let rhs_r = resolve_alias(rhs, &values);
                let lhs_v = read_value(&lhs_r, &values);
                let rhs_v = read_value(&rhs_r, &values);
                let simplified = simplify_binary("AND", dst, &lhs_r, &rhs_r, &lhs_v, &rhs_v);
                track_result_value(&mut values, &simplified);
                out.push(simplified);
            }
            MachineInstr::Or { dst, lhs, rhs } => {
                let lhs_r = resolve_alias(lhs, &values);
                let rhs_r = resolve_alias(rhs, &values);
                let lhs_v = read_value(&lhs_r, &values);
                let rhs_v = read_value(&rhs_r, &values);
                let simplified = simplify_binary("OR", dst, &lhs_r, &rhs_r, &lhs_v, &rhs_v);
                track_result_value(&mut values, &simplified);
                out.push(simplified);
            }
            MachineInstr::Xor { dst, lhs, rhs } => {
                let lhs_r = resolve_alias(lhs, &values);
                let rhs_r = resolve_alias(rhs, &values);
                let lhs_v = read_value(&lhs_r, &values);
                let rhs_v = read_value(&rhs_r, &values);
                let simplified = simplify_binary("XOR", dst, &lhs_r, &rhs_r, &lhs_v, &rhs_v);
                track_result_value(&mut values, &simplified);
                out.push(simplified);
            }
        }
    }

    out
}

fn eliminate_dead_temps(input: &[MachineInstr]) -> Vec<MachineInstr> {
    let mut live: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut keep = vec![false; input.len()];

    for (i, instr) in input.iter().enumerate().rev() {
        let def = defined_var(instr);
        let uses = used_vars(instr);
        let must_keep = def
            .as_ref()
            .map(|d| !is_temp(d) || live.contains(d))
            .unwrap_or(true);
        if must_keep {
            keep[i] = true;
            if let Some(d) = def {
                live.remove(&d);
            }
            for u in uses {
                live.insert(u);
            }
        }
    }

    let mut out = Vec::with_capacity(input.len());
    for (i, instr) in input.iter().enumerate() {
        if keep[i] {
            out.push(instr.clone());
        }
    }
    out
}

fn defined_var(instr: &MachineInstr) -> Option<String> {
    match instr {
        MachineInstr::SetState { dst, .. } => Some(dst.clone()),
        MachineInstr::Mov { dst, .. } => Some(dst.clone()),
        MachineInstr::Not { dst, .. } => Some(dst.clone()),
        MachineInstr::And { dst, .. } => Some(dst.clone()),
        MachineInstr::Or { dst, .. } => Some(dst.clone()),
        MachineInstr::Xor { dst, .. } => Some(dst.clone()),
    }
}

fn used_vars(instr: &MachineInstr) -> Vec<String> {
    match instr {
        MachineInstr::SetState { .. } => Vec::new(),
        MachineInstr::Mov { src, .. } => vec![src.clone()],
        MachineInstr::Not { src, .. } => vec![src.clone()],
        MachineInstr::And { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
        MachineInstr::Or { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
        MachineInstr::Xor { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
    }
}

#[inline]
fn is_temp(name: &str) -> bool {
    name.starts_with("__t")
}

fn resolve_alias(name: &str, values: &HashMap<String, Value>) -> String {
    let mut cur = name.to_string();
    let mut steps = 0usize;
    while steps < 64 {
        match values.get(&cur) {
            Some(Value::Alias(next)) if next != &cur => {
                cur = next.clone();
                steps += 1;
            }
            _ => break,
        }
    }
    cur
}

fn read_value(name: &str, values: &HashMap<String, Value>) -> Value {
    match values.get(name) {
        Some(v) => v.clone(),
        None => Value::Unknown,
    }
}

fn track_result_value(values: &mut HashMap<String, Value>, instr: &MachineInstr) {
    match instr {
        MachineInstr::SetState { dst, state } => {
            values.insert(dst.clone(), Value::Const(*state));
        }
        MachineInstr::Mov { dst, src } => {
            values.insert(dst.clone(), Value::Alias(src.clone()));
        }
        MachineInstr::Not { dst, .. }
        | MachineInstr::And { dst, .. }
        | MachineInstr::Or { dst, .. }
        | MachineInstr::Xor { dst, .. } => {
            values.insert(dst.clone(), Value::Unknown);
        }
    }
}

fn simplify_binary(
    op: &str,
    dst: &str,
    lhs: &str,
    rhs: &str,
    lhs_v: &Value,
    rhs_v: &Value,
) -> MachineInstr {
    match (lhs_v, rhs_v) {
        (Value::Const(a), Value::Const(b)) => {
            return MachineInstr::SetState {
                dst: dst.to_string(),
                state: apply_state_binary(op, *a, *b),
            };
        }
        _ => {}
    }

    if lhs == rhs {
        return match op {
            "XOR" => MachineInstr::SetState {
                dst: dst.to_string(),
                state: N,
            },
            _ => MachineInstr::Mov {
                dst: dst.to_string(),
                src: lhs.to_string(),
            },
        };
    }

    match op {
        "AND" => {
            if is_const_state(lhs_v, N) || is_const_state(rhs_v, N) {
                return MachineInstr::SetState {
                    dst: dst.to_string(),
                    state: N,
                };
            }
            if is_const_state(lhs_v, S) {
                return MachineInstr::Mov {
                    dst: dst.to_string(),
                    src: rhs.to_string(),
                };
            }
            if is_const_state(rhs_v, S) {
                return MachineInstr::Mov {
                    dst: dst.to_string(),
                    src: lhs.to_string(),
                };
            }
            MachineInstr::And {
                dst: dst.to_string(),
                lhs: lhs.to_string(),
                rhs: rhs.to_string(),
            }
        }
        "OR" => {
            if is_const_state(lhs_v, S) || is_const_state(rhs_v, S) {
                return MachineInstr::SetState {
                    dst: dst.to_string(),
                    state: S,
                };
            }
            if is_const_state(lhs_v, N) {
                return MachineInstr::Mov {
                    dst: dst.to_string(),
                    src: rhs.to_string(),
                };
            }
            if is_const_state(rhs_v, N) {
                return MachineInstr::Mov {
                    dst: dst.to_string(),
                    src: lhs.to_string(),
                };
            }
            MachineInstr::Or {
                dst: dst.to_string(),
                lhs: lhs.to_string(),
                rhs: rhs.to_string(),
            }
        }
        "XOR" => {
            if is_const_state(lhs_v, N) {
                return MachineInstr::Mov {
                    dst: dst.to_string(),
                    src: rhs.to_string(),
                };
            }
            if is_const_state(rhs_v, N) {
                return MachineInstr::Mov {
                    dst: dst.to_string(),
                    src: lhs.to_string(),
                };
            }
            MachineInstr::Xor {
                dst: dst.to_string(),
                lhs: lhs.to_string(),
                rhs: rhs.to_string(),
            }
        }
        _ => MachineInstr::Xor {
            dst: dst.to_string(),
            lhs: lhs.to_string(),
            rhs: rhs.to_string(),
        },
    }
}

#[inline]
fn is_const_state(v: &Value, state: u8) -> bool {
    matches!(v, Value::Const(s) if *s == state)
}

#[inline]
fn inverse_state(state: u8) -> u8 {
    ((state & 0b10) >> 1) | ((state & 0b01) << 1)
}

#[inline]
fn apply_state_binary(op: &str, a: u8, b: u8) -> u8 {
    match op {
        "AND" => a & b,
        "OR" => a | b,
        "XOR" => a ^ b,
        _ => a ^ b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{train_profile, TrainingSample};

    #[test]
    fn compile_and_execute_human_program_with_profile() {
        let profile = train_profile(&[
            TrainingSample {
                input: "x = TRUE",
                target: "x = T",
            },
            TrainingSample {
                input: "y = FALSE",
                target: "y = F",
            },
            TrainingSample {
                input: "out = x AND y",
                target: "out = x & y",
            },
            TrainingSample {
                input: "z = NOT out",
                target: "z = ! out",
            },
        ]);

        let human = r#"
			x = TRUE
			y = FALSE
			out = x AND y
			z = NOT out
		"#;
        let program = compile_human_program(human, Some(&profile)).expect("compile");
        let mut env = HashMap::<String, QuadroReg>::new();
        execute_machine_program(&program, &mut env).expect("run");

        let expected_out =
            QuadroReg::from_raw(crate::MSB_MASK).intersect(QuadroReg::from_raw(crate::LSB_MASK));
        let expected_z = expected_out.inverse();
        assert_eq!(env.get("z").copied(), Some(expected_z));
    }

    #[test]
    fn machine_text_parse_and_run() {
        let src = r#"
			SET a T
			SET b F
			AND out a b
			NOT inv out
		"#;
        let program = parse_machine_program(src).expect("parse");
        let mut env = HashMap::<String, QuadroReg>::new();
        execute_machine_program(&program, &mut env).expect("run");
        assert!(env.contains_key("inv"));
    }

    #[test]
    fn machine_parse_rejects_bad_instruction() {
        let err = parse_machine_program("BAD x y").expect_err("must fail");
        assert_eq!(err.line, 1);
        assert!(err.message.contains("unknown instruction"));
    }

    #[test]
    fn optimizer_folds_constants_and_eliminates_temps() {
        let profile = train_profile(&[
            TrainingSample {
                input: "x = TRUE",
                target: "x = T",
            },
            TrainingSample {
                input: "y = FALSE",
                target: "y = F",
            },
            TrainingSample {
                input: "out = x AND y",
                target: "out = x & y",
            },
        ]);
        let human = "x = TRUE\ny = FALSE\nout = x AND y\n";
        let program = compile_human_program(human, Some(&profile)).expect("compile");

        assert!(program
            .instructions
            .iter()
            .all(|i| !format!("{:?}", i).contains("__t")));
        assert!(program.instructions.len() <= 3);
    }

    #[test]
    fn optimizer_removes_redundant_moves() {
        let p = MachineProgram {
            instructions: vec![
                MachineInstr::SetState {
                    dst: "a".to_string(),
                    state: T,
                },
                MachineInstr::Mov {
                    dst: "b".to_string(),
                    src: "a".to_string(),
                },
                MachineInstr::Mov {
                    dst: "c".to_string(),
                    src: "b".to_string(),
                },
            ],
        };
        let o = optimize_machine_program(&p);
        assert_eq!(
            o.instructions,
            vec![
                MachineInstr::SetState {
                    dst: "a".to_string(),
                    state: T,
                },
                MachineInstr::SetState {
                    dst: "b".to_string(),
                    state: T,
                },
                MachineInstr::SetState {
                    dst: "c".to_string(),
                    state: T,
                },
            ]
        );
    }
}
