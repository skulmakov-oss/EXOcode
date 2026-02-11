# EXOcode Toolchain v0

Minimal toolchain loop:

`EXO source -> EXObyte v0 -> VM run`

## Quickstart

Build CLI:

```powershell
cargo build --bin exoc
```

Compile source to EXObyte:

```powershell
cargo run --bin exoc -- compile program.exo -o program.exb
```

Run source directly (compile in memory + execute):

```powershell
cargo run --bin exoc -- run program.exo
```

Run precompiled EXObyte:

```powershell
cargo run --bin exoc -- runb program.exb
```

Disassemble EXObyte:

```powershell
cargo run --bin exoc -- disasm program.exb
```

## EXObyte v0 format

- Header: `EXOBYTE0` (8 bytes)
- Then function records:
  - `u16 name_len`, `name bytes`
  - `u32 code_len`, `code bytes`
- LE encoding for all integer fields.
- Opcodes are centralized in `src/exobyte_format.rs`.
- Labels are not stored as opcodes; jumps are patched to absolute function-local addresses during emit.

## Current language constraints

- `if` condition must be `bool` (`if quad_expr` is forbidden; explicit compare is required).
- `->` (implies) is supported only for `quad`.
- `match` is supported only for `quad` scrutinee.
- `match` requires explicit default arm `_ => { ... }`.

## Test commands

```powershell
cargo test
```

Golden format tests:

```powershell
cargo test --test golden_exobyte
```
