# EXOcode Architecture (Profile 2)

EXOcode is a strict Rust-like systems language with native quad logic.

## Positioning
- Not a DSL.
- Not a declarative script format.
- A systems language where `quad` is a built-in primitive, similar to `bool` in Rust.

## Type Core
- `quad` : `N | F | T | S`
- `bool`
- `i32`
- `u32`
- `fx` (Q16.16)

## Quad Operators
- Unary: `!a`
- Binary: `a && b`, `a || b`, `a -> b`
- Equality: `==`, `!=`

Rules:
- `if quad_expr` is forbidden.
- Explicit compare is required: `if state == T`.

## Layers
- **L1 (Human Semantic Layer)**:
  - Rust-like syntax.
  - Strict typing.
  - Minimal constructs: `fn`, `let`, `if/else`, `return`, function calls.
- **L2 (Machine IR)**:
  - Register instructions, explicit control flow, no high-level semantics.
  - Core ops: `LOAD_Q`, `Q_AND`, `Q_OR`, `Q_NOT`, `Q_IMPL`, `CMP_EQ`, `JMP`, `JMP_IF`, `CALL`, `RET`.

## Current frontend module
- File: `src/frontend.rs`
- Contains:
  - EBNF grammar constant.
  - Lexer.
  - AST for strict Rust-like profile.
  - Type checker with explicit `if` bool requirement.
  - Basic expression lowering into IR instruction enum.
