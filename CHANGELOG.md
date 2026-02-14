# Changelog

All notable changes to this project are documented in this file.

## v1.0.0 - 2026-02-14

### Added
- `f64` type support in frontend, type checker, AST, and IR.
- Float literals and unary `+` / `-`.
- `f64` arithmetic operators: `+`, `-`, `*`, `/` with precedence parsing.
- EXObyte v1 format marker: `EXOBYTE1`.
- New math opcodes: `LOAD_F64`, `ADD_F64`, `SUB_F64`, `MUL_F64`, `DIV_F64`.
- VM builtin dispatch for `sin`, `cos`, `tan`, `sqrt`, `abs`, `pow`.
- New v1 golden fixture: `tests/golden_v1/calculator.exo` + `.exb`.
- New example program: `examples/calculator.exo`.

### Changed
- EXObyte parser/VM now accept both `EXOBYTE0` and `EXOBYTE1`.
- Bytecode emitter automatically selects:
  - `EXOBYTE0` for v0-compatible programs.
  - `EXOBYTE1` when v1 math opcodes are used.
- Golden tests extended to include v1 fixture.

### Compatibility
- Backward compatibility preserved: existing `EXOBYTE0` binaries remain executable.

## v0.1.0 - 2026-02-14

### Added
- Toolchain v0 baseline: parser, type-checker, IR lowering, EXObyte emitter, VM, CLI.
- EXObyte v0 bytecode format (`EXOBYTE0`) and golden byte-for-byte tests.

### Notes
- v0 is frozen on branch `release/v0` and tag `v0.1.0`.
