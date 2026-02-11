# EXOcode Toolchain v0
Rust-like frontend + EXObyte emitter + VM runtime.

## What's included
- Parser, type-checker, IR lowering, EXObyte emitter.
- EXObyte v0 VM executor and disassembler.
- CLI tool `exoc` for compile/run/runb/disasm.
- Golden byte-format tests.

## Quickstart
Use these commands as-is from repository root.

```powershell
# 1) Build CLI
cargo build --bin exoc

# 2) Create minimal source program
@'
fn main() {
    return;
}
'@ | Set-Content program.exo

# 3) Compile EXO -> EXObyte
cargo run --bin exoc -- compile program.exo -o program.exb

# 4) Run source directly (compile in memory + execute main)
cargo run --bin exoc -- run program.exo

# 5) Run precompiled EXObyte
cargo run --bin exoc -- runb program.exb

# 6) Disassemble EXObyte
cargo run --bin exoc -- disasm program.exb
```

Example `disasm` output:

```text
EXOBYTE0
fn one: code=13 bytes, strings=0
  0000: LOAD_I32 r0, 1
  0007: RET r0
fn main: code=25 bytes, strings=2
  0000: CALL dst?1 r0 fn#0 argc=0
  0008: STORE_VAR s1, r0
  000d: RET
```

## CLI reference
- `exoc compile <input.exo> -o <out.exb>`
  - Parses, type-checks, lowers, validates IR, emits EXObyte file.
- `exoc run <input.exo>`
  - Compiles source in memory and executes `main` in VM.
- `exoc runb <input.exb>`
  - Executes precompiled EXObyte in VM.
- `exoc disasm <input.exb>`
  - Prints decoded functions/opcodes from EXObyte payload.

## EXObyte v0 format (spec)
- Endianness: little-endian for all integer fields.
- Header: ASCII magic `EXOBYTE0` (8 bytes).
- File body: repeated function records until EOF.
  - `u16 name_len`
  - `name_len` bytes function name (UTF-8)
  - `u32 code_len`
  - `code_len` bytes function code section
- Function `code` layout:
  - string table prefix
  - opcode stream payload
- String table layout:
  - `u16 string_count`
  - for each string: `u16 len`, then `len` bytes UTF-8
  - instructions reference strings by `u16 str_id`
- Jumps:
  - encoded as absolute function-local addresses in opcode stream.
  - `Label` nodes do not serialize into bytecode.
- Opcode and encoding source of truth:
  - `src/exobyte_format.rs` (`MAGIC`, `Opcode`, read/write LE helpers).

## Language constraints (v0)
- `if` condition is `bool` only. `if quad_expr` is forbidden.
- `->` (implies) is `quad`-only.
- `match` is `quad`-only.
- `match` requires explicit default arm `_ => { ... }`.
- Unit-returning call is valid as statement, invalid as value expression.

## Tests
```powershell
cargo fmt --check
cargo test
cargo test --test golden_exobyte
```

## Repository layout
- `src/frontend.rs` - lexer/parser/type-checker/lowering/IR validation/emitter.
- `src/exobyte_format.rs` - EXObyte constants, opcodes, LE read/write helpers.
- `src/exobyte_vm.rs` - EXObyte parser, VM runtime, disassembler.
- `src/bin/exoc.rs` - CLI entrypoint (`compile`, `run`, `runb`, `disasm`).
- `tests/golden/*` - `.exo` and `.exb` golden fixtures.
- `tests/golden_exobyte.rs` - golden byte-for-byte format tests.

## Roadmap
- EXObyte structural validator before VM execution.
- VM step limit / sandbox mode for deterministic safety.
- Extend arithmetic typing and lowering (`u32`, `fx` ops).
- Optimize `match` lowering (jump tables/selective chains).
- Stabilize stdlib ABI and calling convention.
- Debug info mapping (source positions -> bytecode offsets).
- EXObyte versioning policy and compatibility matrix.
