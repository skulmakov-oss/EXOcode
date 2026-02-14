use crate::exobyte_format::{
    read_f64_le, read_i32_le, read_u16_le, read_u32_le, read_u8, read_utf8, ExobyteFormatError,
    Opcode, MAGIC0, MAGIC1,
};
use crate::frontend::QuadVal;
use std::collections::HashMap;

const MAX_STACK_DEPTH: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Quad(QuadVal),
    Bool(bool),
    I32(i32),
    F64(f64),
    U32(u32),
    Fx(i32),
    Unit,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub pc: usize,
    pub regs: Vec<Value>,
    pub locals: HashMap<String, Value>,
    pub func: String,
    pub return_dst: Option<u16>,
}

#[derive(Debug, Clone)]
pub struct FunctionBytecode {
    pub name: String,
    pub strings: Vec<String>,
    pub code: Vec<u8>,
    pub instr_start: usize,
}

#[derive(Debug, Clone)]
pub struct VM {
    pub functions: HashMap<String, FunctionBytecode>,
    pub callstack: Vec<Frame>,
}

#[derive(Debug, Clone)]
struct ParsedExobyte {
    magic: [u8; 8],
    functions: HashMap<String, FunctionBytecode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    BadHeader,
    BadFormat(String),
    UnknownFunction(String),
    InvalidJumpAddress { func: String, addr: usize },
    TypeMismatchRuntime(String),
    StackUnderflow,
    StackOverflow,
    UnknownVariable(String),
    InvalidStringId(u16),
}

impl core::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RuntimeError::BadHeader => write!(f, "bad EXObyte header"),
            RuntimeError::BadFormat(m) => write!(f, "bad EXObyte format: {}", m),
            RuntimeError::UnknownFunction(n) => write!(f, "unknown function '{}'", n),
            RuntimeError::InvalidJumpAddress { func, addr } => {
                write!(f, "invalid jump address {} in '{}'", addr, func)
            }
            RuntimeError::TypeMismatchRuntime(m) => write!(f, "runtime type mismatch: {}", m),
            RuntimeError::StackUnderflow => write!(f, "stack underflow"),
            RuntimeError::StackOverflow => write!(f, "stack overflow"),
            RuntimeError::UnknownVariable(n) => write!(f, "unknown variable '{}'", n),
            RuntimeError::InvalidStringId(id) => write!(f, "invalid string id {}", id),
        }
    }
}

impl std::error::Error for RuntimeError {}

pub fn run_exobyte(bytes: &[u8]) -> Result<(), RuntimeError> {
    run_exobyte_with_entry(bytes, "main")
}

pub fn run_exobyte_with_entry(bytes: &[u8], entry: &str) -> Result<(), RuntimeError> {
    let parsed = parse_exobyte(bytes)?;
    let mut vm = VM {
        functions: parsed.functions,
        callstack: Vec::new(),
    };
    push_frame(&mut vm, entry, Vec::new(), None)?;
    exec_loop(&mut vm)
}

pub fn disasm_exobyte(bytes: &[u8]) -> Result<String, RuntimeError> {
    let parsed = parse_exobyte(bytes)?;
    let mut out = String::new();
    out.push_str(&format!("{}\n", String::from_utf8_lossy(&parsed.magic)));
    for f in parsed.functions.values() {
        out.push_str(&format!(
            "fn {}: code={} bytes, strings={}\n",
            f.name,
            f.code.len(),
            f.strings.len()
        ));
        let mut pc = 0usize;
        while pc < f.code.len().saturating_sub(f.instr_start) {
            let (line, next) = disasm_one(f, pc)?;
            out.push_str(&format!("  {:04x}: {}\n", pc, line));
            pc = next;
        }
    }
    Ok(out)
}

fn parse_exobyte(bytes: &[u8]) -> Result<ParsedExobyte, RuntimeError> {
    if bytes.len() < MAGIC0.len() {
        return Err(RuntimeError::BadHeader);
    }
    let magic = if &bytes[0..8] == MAGIC0 {
        MAGIC0
    } else if &bytes[0..8] == MAGIC1 {
        MAGIC1
    } else {
        return Err(RuntimeError::BadHeader);
    };
    let mut i = 8usize;
    let mut out = HashMap::new();
    while i < bytes.len() {
        let name_len = read_u16_le(bytes, &mut i).map_err(map_format_err)? as usize;
        let name = read_utf8(bytes, &mut i, name_len).map_err(map_format_err)?;
        let code_len = read_u32_le(bytes, &mut i).map_err(map_format_err)? as usize;
        if i + code_len > bytes.len() {
            return Err(RuntimeError::BadFormat(
                "function code out of bounds".to_string(),
            ));
        }
        let code = bytes[i..i + code_len].to_vec();
        i += code_len;

        let (strings, instr_start) = parse_string_table(&code)?;
        let f = FunctionBytecode {
            name: name.clone(),
            strings,
            code,
            instr_start,
        };
        if out.insert(name.clone(), f).is_some() {
            return Err(RuntimeError::BadFormat(format!(
                "duplicate function '{}'",
                name
            )));
        }
    }
    Ok(ParsedExobyte {
        magic,
        functions: out,
    })
}

fn parse_string_table(code: &[u8]) -> Result<(Vec<String>, usize), RuntimeError> {
    let mut i = 0usize;
    let count = read_u16_le(code, &mut i).map_err(map_format_err)? as usize;
    let mut strings = Vec::with_capacity(count);
    for _ in 0..count {
        let len = read_u16_le(code, &mut i).map_err(map_format_err)? as usize;
        strings.push(read_utf8(code, &mut i, len).map_err(map_format_err)?);
    }
    Ok((strings, i))
}

fn map_format_err(err: ExobyteFormatError) -> RuntimeError {
    match err {
        ExobyteFormatError::UnexpectedEof => RuntimeError::BadFormat("unexpected EOF".to_string()),
        ExobyteFormatError::InvalidUtf8 => RuntimeError::BadFormat("invalid utf8".to_string()),
        ExobyteFormatError::UnknownOpcode(op) => {
            RuntimeError::BadFormat(format!("unknown opcode 0x{:02x}", op))
        }
    }
}

fn exec_loop(vm: &mut VM) -> Result<(), RuntimeError> {
    loop {
        let Some(frame_idx) = vm.callstack.len().checked_sub(1) else {
            return Ok(());
        };
        let func_name = vm.callstack[frame_idx].func.clone();
        let f = vm
            .functions
            .get(&func_name)
            .cloned()
            .ok_or_else(|| RuntimeError::UnknownFunction(func_name.clone()))?;
        let pc = vm.callstack[frame_idx].pc;
        let instr_rel_len = f.code.len().saturating_sub(f.instr_start);
        if pc >= instr_rel_len {
            return Err(RuntimeError::BadFormat(format!(
                "pc out of range in '{}': {}",
                func_name, pc
            )));
        }
        let mut cur = f.instr_start + pc;
        let opcode = Opcode::from_byte(read_u8(&f.code, &mut cur).map_err(map_format_err)?)
            .map_err(map_format_err)?;
        let next_pc: usize;

        match opcode {
            Opcode::LoadQ => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let q = match read_u8(&f.code, &mut cur).map_err(map_format_err)? {
                    0 => QuadVal::N,
                    1 => QuadVal::F,
                    2 => QuadVal::T,
                    3 => QuadVal::S,
                    v => {
                        return Err(RuntimeError::BadFormat(format!(
                            "invalid quad literal {}",
                            v
                        )))
                    }
                };
                set_reg(vm, frame_idx, dst, Value::Quad(q));
                next_pc = cur - f.instr_start;
            }
            Opcode::LoadBool => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let b = read_u8(&f.code, &mut cur).map_err(map_format_err)? != 0;
                set_reg(vm, frame_idx, dst, Value::Bool(b));
                next_pc = cur - f.instr_start;
            }
            Opcode::LoadI32 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let v = read_i32_le(&f.code, &mut cur).map_err(map_format_err)?;
                set_reg(vm, frame_idx, dst, Value::I32(v));
                next_pc = cur - f.instr_start;
            }
            Opcode::LoadF64 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let v = read_f64_le(&f.code, &mut cur).map_err(map_format_err)?;
                set_reg(vm, frame_idx, dst, Value::F64(v));
                next_pc = cur - f.instr_start;
            }
            Opcode::LoadVar => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let name = lookup_str(&f, sid)?;
                let val = vm.callstack[frame_idx]
                    .locals
                    .get(name)
                    .cloned()
                    .ok_or_else(|| RuntimeError::UnknownVariable(name.to_string()))?;
                set_reg(vm, frame_idx, dst, val);
                next_pc = cur - f.instr_start;
            }
            Opcode::StoreVar => {
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let name = lookup_str(&f, sid)?.to_string();
                let val = get_reg(vm, frame_idx, src)?;
                vm.callstack[frame_idx].locals.insert(name, val);
                next_pc = cur - f.instr_start;
            }
            Opcode::QAnd | Opcode::QOr | Opcode::QImpl => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lq = as_quad(get_reg(vm, frame_idx, lhs)?)?;
                let rq = as_quad(get_reg(vm, frame_idx, rhs)?)?;
                let out_q = match opcode {
                    Opcode::QAnd => quad_and(lq, rq),
                    Opcode::QOr => quad_or(lq, rq),
                    Opcode::QImpl => quad_or(quad_not(lq), rq),
                    _ => unreachable!(),
                };
                set_reg(vm, frame_idx, dst, Value::Quad(out_q));
                next_pc = cur - f.instr_start;
            }
            Opcode::QNot => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let q = as_quad(get_reg(vm, frame_idx, src)?)?;
                set_reg(vm, frame_idx, dst, Value::Quad(quad_not(q)));
                next_pc = cur - f.instr_start;
            }
            Opcode::BoolAnd | Opcode::BoolOr => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lb = as_bool(get_reg(vm, frame_idx, lhs)?)?;
                let rb = as_bool(get_reg(vm, frame_idx, rhs)?)?;
                let out_b = if opcode == Opcode::BoolAnd {
                    lb && rb
                } else {
                    lb || rb
                };
                set_reg(vm, frame_idx, dst, Value::Bool(out_b));
                next_pc = cur - f.instr_start;
            }
            Opcode::BoolNot => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let b = as_bool(get_reg(vm, frame_idx, src)?)?;
                set_reg(vm, frame_idx, dst, Value::Bool(!b));
                next_pc = cur - f.instr_start;
            }
            Opcode::CmpEq | Opcode::CmpNe => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lv = get_reg(vm, frame_idx, lhs)?;
                let rv = get_reg(vm, frame_idx, rhs)?;
                let eq = value_eq(&lv, &rv)?;
                let out = if opcode == Opcode::CmpEq { eq } else { !eq };
                set_reg(vm, frame_idx, dst, Value::Bool(out));
                next_pc = cur - f.instr_start;
            }
            Opcode::AddF64 | Opcode::SubF64 | Opcode::MulF64 | Opcode::DivF64 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lv = as_f64(get_reg(vm, frame_idx, lhs)?)?;
                let rv = as_f64(get_reg(vm, frame_idx, rhs)?)?;
                let out_v = match opcode {
                    Opcode::AddF64 => lv + rv,
                    Opcode::SubF64 => lv - rv,
                    Opcode::MulF64 => lv * rv,
                    Opcode::DivF64 => lv / rv,
                    _ => unreachable!(),
                };
                set_reg(vm, frame_idx, dst, Value::F64(out_v));
                next_pc = cur - f.instr_start;
            }
            Opcode::Jmp => {
                let addr = read_u32_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                if addr >= instr_rel_len {
                    return Err(RuntimeError::InvalidJumpAddress {
                        func: func_name,
                        addr,
                    });
                }
                next_pc = addr;
            }
            Opcode::JmpIf => {
                let cond = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let addr = read_u32_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let b = as_bool(get_reg(vm, frame_idx, cond)?)?;
                if b {
                    if addr >= instr_rel_len {
                        return Err(RuntimeError::InvalidJumpAddress {
                            func: func_name,
                            addr,
                        });
                    }
                    next_pc = addr;
                } else {
                    next_pc = cur - f.instr_start;
                }
            }
            Opcode::Call => {
                let has_dst = read_u8(&f.code, &mut cur).map_err(map_format_err)? != 0;
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let callee_sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let argc = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let callee = lookup_str(&f, callee_sid)?.to_string();
                let mut args = Vec::with_capacity(argc);
                for _ in 0..argc {
                    let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                    args.push(get_reg(vm, frame_idx, r)?);
                }
                if let Some(ret) = eval_builtin(&callee, &args)? {
                    if has_dst {
                        set_reg(vm, frame_idx, dst, ret);
                    }
                    next_pc = cur - f.instr_start;
                    vm.callstack[frame_idx].pc = next_pc;
                    continue;
                }
                vm.callstack[frame_idx].pc = cur - f.instr_start;
                push_frame(vm, &callee, args, if has_dst { Some(dst) } else { None })?;
                continue;
            }
            Opcode::Ret => {
                let has_src = read_u8(&f.code, &mut cur).map_err(map_format_err)? != 0;
                let ret_val = if has_src {
                    let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                    get_reg(vm, frame_idx, src)?
                } else {
                    Value::Unit
                };
                let finished = vm.callstack.pop().ok_or(RuntimeError::StackUnderflow)?;
                if let Some(caller) = vm.callstack.last_mut() {
                    if let Some(dst) = finished.return_dst {
                        write_reg(caller, dst as usize, ret_val);
                    }
                } else {
                    return Ok(());
                }
                continue;
            }
        }

        vm.callstack[frame_idx].pc = next_pc;
    }
}

fn push_frame(
    vm: &mut VM,
    func_name: &str,
    args: Vec<Value>,
    return_dst: Option<u16>,
) -> Result<(), RuntimeError> {
    let f = vm
        .functions
        .get(func_name)
        .ok_or_else(|| RuntimeError::UnknownFunction(func_name.to_string()))?;
    if vm.callstack.len() >= MAX_STACK_DEPTH {
        return Err(RuntimeError::StackOverflow);
    }
    let mut regs = vec![Value::Unit; 16];
    if regs.len() < args.len() {
        regs.resize(args.len(), Value::Unit);
    }
    for (i, v) in args.into_iter().enumerate() {
        regs[i] = v;
    }
    let frame = Frame {
        pc: 0,
        regs,
        locals: HashMap::new(),
        func: f.name.clone(),
        return_dst,
    };
    vm.callstack.push(frame);
    Ok(())
}

fn lookup_str<'a>(f: &'a FunctionBytecode, sid: u16) -> Result<&'a str, RuntimeError> {
    f.strings
        .get(sid as usize)
        .map(|s| s.as_str())
        .ok_or(RuntimeError::InvalidStringId(sid))
}

fn get_reg(vm: &VM, frame_idx: usize, r: u16) -> Result<Value, RuntimeError> {
    vm.callstack
        .get(frame_idx)
        .and_then(|fr| fr.regs.get(r as usize))
        .cloned()
        .ok_or_else(|| RuntimeError::BadFormat(format!("read invalid reg r{}", r)))
}

fn set_reg(vm: &mut VM, frame_idx: usize, r: u16, v: Value) {
    if let Some(frame) = vm.callstack.get_mut(frame_idx) {
        write_reg(frame, r as usize, v);
    }
}

fn write_reg(frame: &mut Frame, r: usize, v: Value) {
    if frame.regs.len() <= r {
        frame.regs.resize(r + 1, Value::Unit);
    }
    frame.regs[r] = v;
}

fn as_quad(v: Value) -> Result<QuadVal, RuntimeError> {
    if let Value::Quad(q) = v {
        Ok(q)
    } else {
        Err(RuntimeError::TypeMismatchRuntime(
            "expected quad".to_string(),
        ))
    }
}

fn as_bool(v: Value) -> Result<bool, RuntimeError> {
    if let Value::Bool(b) = v {
        Ok(b)
    } else {
        Err(RuntimeError::TypeMismatchRuntime(
            "expected bool".to_string(),
        ))
    }
}

fn as_f64(v: Value) -> Result<f64, RuntimeError> {
    if let Value::F64(n) = v {
        Ok(n)
    } else {
        Err(RuntimeError::TypeMismatchRuntime(
            "expected f64".to_string(),
        ))
    }
}

fn quad_to_u8(q: QuadVal) -> u8 {
    match q {
        QuadVal::N => 0,
        QuadVal::F => 1,
        QuadVal::T => 2,
        QuadVal::S => 3,
    }
}

fn u8_to_quad(v: u8) -> QuadVal {
    match v & 0b11 {
        0 => QuadVal::N,
        1 => QuadVal::F,
        2 => QuadVal::T,
        _ => QuadVal::S,
    }
}

fn quad_and(a: QuadVal, b: QuadVal) -> QuadVal {
    u8_to_quad(quad_to_u8(a) & quad_to_u8(b))
}

fn quad_or(a: QuadVal, b: QuadVal) -> QuadVal {
    u8_to_quad(quad_to_u8(a) | quad_to_u8(b))
}

fn quad_not(a: QuadVal) -> QuadVal {
    let v = quad_to_u8(a);
    let r = ((v & 0b10) >> 1) | ((v & 0b01) << 1);
    u8_to_quad(r)
}

fn value_eq(a: &Value, b: &Value) -> Result<bool, RuntimeError> {
    match (a, b) {
        (Value::Quad(x), Value::Quad(y)) => Ok(x == y),
        (Value::Bool(x), Value::Bool(y)) => Ok(x == y),
        (Value::I32(x), Value::I32(y)) => Ok(x == y),
        (Value::F64(x), Value::F64(y)) => Ok(x == y),
        (Value::U32(x), Value::U32(y)) => Ok(x == y),
        (Value::Fx(x), Value::Fx(y)) => Ok(x == y),
        (Value::Unit, Value::Unit) => Ok(true),
        _ => Err(RuntimeError::TypeMismatchRuntime(
            "CmpEq/CmpNe operands must have same runtime type".to_string(),
        )),
    }
}

fn eval_builtin(name: &str, args: &[Value]) -> Result<Option<Value>, RuntimeError> {
    let out = match name {
        "sin" => Some(Value::F64(as_f64_arg(name, args, 1, 0)?.sin())),
        "cos" => Some(Value::F64(as_f64_arg(name, args, 1, 0)?.cos())),
        "tan" => Some(Value::F64(as_f64_arg(name, args, 1, 0)?.tan())),
        "sqrt" => Some(Value::F64(as_f64_arg(name, args, 1, 0)?.sqrt())),
        "abs" => Some(Value::F64(as_f64_arg(name, args, 1, 0)?.abs())),
        "pow" => Some(Value::F64(
            as_f64_arg(name, args, 2, 0)?.powf(as_f64_arg(name, args, 2, 1)?),
        )),
        _ => None,
    };
    Ok(out)
}

fn as_f64_arg(
    name: &str,
    args: &[Value],
    expected: usize,
    idx: usize,
) -> Result<f64, RuntimeError> {
    if args.len() != expected {
        return Err(RuntimeError::BadFormat(format!(
            "builtin '{}' expects {} args, got {}",
            name,
            expected,
            args.len()
        )));
    }
    match args.get(idx) {
        Some(Value::F64(n)) => Ok(*n),
        _ => Err(RuntimeError::TypeMismatchRuntime(format!(
            "builtin '{}' expects f64 args",
            name
        ))),
    }
}

fn disasm_one(f: &FunctionBytecode, pc: usize) -> Result<(String, usize), RuntimeError> {
    let mut cur = f.instr_start + pc;
    let opcode = Opcode::from_byte(read_u8(&f.code, &mut cur).map_err(map_format_err)?)
        .map_err(map_format_err)?;
    let text = match opcode {
        Opcode::LoadQ => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let q = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_Q r{}, {}", d, q)
        }
        Opcode::LoadBool => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let b = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_BOOL r{}, {}", d, b)
        }
        Opcode::LoadI32 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let n = read_i32_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_I32 r{}, {}", d, n)
        }
        Opcode::LoadF64 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let n = read_f64_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_F64 r{}, {}", d, n)
        }
        Opcode::LoadVar => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_VAR r{}, s{}", d, s)
        }
        Opcode::StoreVar => {
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("STORE_VAR s{}, r{}", s, r)
        }
        Opcode::QAnd
        | Opcode::QOr
        | Opcode::QImpl
        | Opcode::BoolAnd
        | Opcode::BoolOr
        | Opcode::CmpEq
        | Opcode::CmpNe
        | Opcode::AddF64
        | Opcode::SubF64
        | Opcode::MulF64
        | Opcode::DivF64 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let l = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let op = match opcode {
                Opcode::QAnd => "Q_AND",
                Opcode::QOr => "Q_OR",
                Opcode::QImpl => "Q_IMPL",
                Opcode::BoolAnd => "BOOL_AND",
                Opcode::BoolOr => "BOOL_OR",
                Opcode::CmpEq => "CMP_EQ",
                Opcode::CmpNe => "CMP_NE",
                Opcode::AddF64 => "ADD_F64",
                Opcode::SubF64 => "SUB_F64",
                Opcode::MulF64 => "MUL_F64",
                _ => "DIV_F64",
            };
            format!("{} r{}, r{}, r{}", op, d, l, r)
        }
        Opcode::QNot | Opcode::BoolNot => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let op = if opcode == Opcode::QNot {
                "Q_NOT"
            } else {
                "BOOL_NOT"
            };
            format!("{} r{}, r{}", op, d, s)
        }
        Opcode::Jmp => {
            let a = read_u32_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("JMP {}", a)
        }
        Opcode::JmpIf => {
            let c = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let a = read_u32_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("JMP_IF r{}, {}", c, a)
        }
        Opcode::Call => {
            let has = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let n = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let argc = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
            for _ in 0..argc {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            format!("CALL dst?{} r{} fn#{} argc={}", has, d, n, argc)
        }
        Opcode::Ret => {
            let has = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
            if has != 0 {
                let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                format!("RET r{}", r)
            } else {
                "RET".to_string()
            }
        }
    };
    Ok((text, cur - f.instr_start))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::compile_program_to_exobyte;

    #[test]
    fn vm_runs_empty_main() {
        let src = "fn main() { return; }";
        let bytes = compile_program_to_exobyte(src).expect("compile");
        run_exobyte(&bytes).expect("run");
    }

    #[test]
    fn vm_runs_bool_ops() {
        let src = r#"
			fn main() {
				let a: bool = true;
				let b: bool = false;
				let c = a && b;
				if c == false { return; } else { return; }
			}
		"#;
        let bytes = compile_program_to_exobyte(src).expect("compile");
        run_exobyte(&bytes).expect("run");
    }

    #[test]
    fn vm_runs_quad_ops() {
        let src = r#"
			fn main() {
				let a: quad = T;
				let b: quad = S;
				let c = a && b;
				if c == T { return; } else { return; }
			}
		"#;
        let bytes = compile_program_to_exobyte(src).expect("compile");
        run_exobyte(&bytes).expect("run");
    }

    #[test]
    fn vm_runs_call_ret() {
        let src = r#"
			fn one() -> i32 { return 1; }
			fn main() { let x: i32 = one(); return; }
		"#;
        let bytes = compile_program_to_exobyte(src).expect("compile");
        run_exobyte(&bytes).expect("run");
    }

    #[test]
    fn vm_runs_f64_math_and_builtins() {
        let src = r#"
			fn main() {
				let a: f64 = 1.5 + 2.25 * 2.0;
				let b: f64 = sqrt(9.0);
				let c: f64 = sin(0.0) + cos(0.0) + abs(-1.0) + pow(2.0, 3.0);
				if a == 6.0 { return; } else { return; }
			}
		"#;
        let bytes = compile_program_to_exobyte(src).expect("compile");
        let dis = disasm_exobyte(&bytes).expect("disasm");
        assert!(dis.contains("EXOBYTE1"));
        assert!(dis.contains("ADD_F64"));
        run_exobyte(&bytes).expect("run");
    }
}
