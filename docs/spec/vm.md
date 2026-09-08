# Semantic VM Specification

Status: draft v0
Owner crate: `sm-vm`
Shared runtime vocabulary owner: `sm-runtime-core`

## Purpose

The Semantic VM executes verified SemCode inside a deterministic, isolated
runtime.

The VM is responsible for:

- register-oriented execution
- call-frame management
- deterministic runtime evaluation
- runtime quota enforcement
- safe execution failure reporting

The VM is not responsible for:

- owning the SemCode binary contract
- replacing verifier admission
- owning capability policy semantics
- owning semantic state or rule scheduling

## Standard Execution Rule

The canonical trusted execution route is verifier-first and token-first:
SemCode bytes must pass `sm-verify` admission and produce a verified token
before trusted VM execution.

Current public verified entrypoints include:

- token-first canonical execution:
  - `run_verified_entry_semcode*`
- verified compatibility helpers:
  - `run_verified_semcode`
  - `run_verified_semcode_with_config`
  - `run_verified_semcode_with_entry`
  - `run_verified_semcode_with_entry_and_config`

Lower-level helpers may still exist for tests or narrow integration points, but
they must not redefine the public contract or be described as the canonical
trusted route.

`VerifiedEntrySemCode`'s entry identity is authoritative for execution.
Possession of a token resolved for one entry does not authorize selecting a
different function from the same verified artifact through an independent
identifier. Every public execution API accepting a `VerifiedEntrySemCode`
derives its execution target from `token.entry()` alone.

## Runtime Value Model

Current runtime values:

- `Quad`
- `Bool`
- `I32`
- `F64`
- `U32`
- `Fx`
- `Unit`

Presence in the runtime value model does not by itself imply that every
source-level numeric family is already complete through the full public
language pipeline.

## Symbol Runtime Rule

The VM hot path uses `SymbolId` rather than string-keyed locals.

Current rule:

- function-local string tables are converted into runtime `SymbolId` values
  during SemCode loading
- frame locals are keyed by `SymbolId`
- debug naming remains a separate concern

## Execution Contexts

The VM consumes `ExecutionConfig`, which binds:

- `ExecutionContext`
- `RuntimeQuotas`

Current execution contexts:

- `PureCompute`
- `VerifiedLocal`
- `RuleExecution`
- `KernelBound`

Context rule:

- context selects the runtime quota baseline
- context does not weaken verifier admission or SemCode safety checks

`Steps` is enforced (#1759) at the single shared instruction-dispatch loop:
one successfully decoded opcode charges exactly one `Step`, before that
instruction's own semantic body runs - including a backward jump revisited
on every loop iteration, making `max_steps` genuine execution fuel rather
than a diagnostic count. A malformed opcode that fails to decode charges no
`Step`. Full semantics are frozen in
`docs/roadmap/stable_foundation/ssf08_1759_steps_calls_contract_decision.md`.

## Trap And Error Model

Current public runtime error families include:

- `BadHeader`
- `UnsupportedBytecodeVersion`
- `BadFormat`
- `UnknownFunction`
- `InvalidJumpAddress`
- `TypeMismatchRuntime`
- `StackUnderflow`
- `StackOverflow`
- `QuotaExceeded`
- `VerifierRejected`
- `BorrowWriteConflict`
- `UnknownVariable`
- `InvalidStringId`

Contract rule:

- verifier rejection must be surfaced distinctly from runtime execution failure
- quota exhaustion must preserve the exceeded quota kind, limit, and usage
- malformed or unsupported bytecode must not be treated as a successful run

`MAP_GET` evaluates its default register only on a key miss. Failure to
read the default register is a runtime error. `MAP_GET` must never
convert a register-read failure into `Value::Unit`.

## Determinism Rule

The VM must behave deterministically for the same:

- verified SemCode input
- execution config
- entry function

The VM must not silently reinterpret:

- unsupported header versions
- malformed jump targets
- malformed function layouts

## Function And Frame Model

Current frame model includes:

- program counter
- register vector
- `SymbolId` local map
- frame-local borrowed tuple and direct record-field paths
- function identity
- optional return destination

Register storage capacity and register initialization are independent:
a register slot may physically exist without ever having been defined by
an instruction. `Value::Unit` is an ordinary Semantic value, never a
stand-in for "nothing was written here." A read of an undefined register
deterministically fails; it must never be treated as `Value::Unit` by
default. This runtime defense is independent of and does not replace
verifier dataflow proof of definite assignment.

Current function-bytecode model includes:

- function name
- string table
- runtime symbol ids
- optional debug symbols
- tuple and direct record-field ownership path metadata admitted from `OWN0`
- the decoded canonical callable-signature record admitted from `SIG0`
  (`signature: Option<CallableSignature>`, #1773 / FA-09-005), retained
  unchanged from the decoded envelope; `None` for an artifact whose header
  predates canonical signatures
- instruction stream
- instruction start offset

## Callable Runtime Family Enforcement

`validate_call_arguments` is the one authoritative invocation validator,
fused directly into `push_frame` itself rather than called separately at
each of its callers. `push_frame` is the single choke point every real
frame-entry path already funnels through - `Opcode::Call`, `Opcode::
ClosureCall`, the public invocation helper
(`run_verified_function_semcode_with_args_and_config`), and every other
verified entry route (`run_verified_entry_semcode_with_config` and the
PROMETHEUS/application host routes) - so fusing here closes every one of
them at once, including any future call site, rather than relying on each
caller to remember to call the validator separately. The raw/unverified
execution path (`run_semcode_with_entry_and_config` and friends, which
intentionally bypass `sm-verify` admission) shares `push_frame` too and is
safe to include: it always passes empty args, so this only ever produces
an earlier, cleaner rejection in place of a later, more confusing
uninitialized-register error - never a false rejection of a scenario that
path is trying to exercise.

Given the callee's decoded `signature`:

- `None` (pre-#1773 artifact): no contract to enforce, unchanged behavior
- `Some(contract)`: reject if `args.len() != contract.families.len()`, and
  reject if any supplied argument's runtime family
  (`Value` variant, mapped 1:1 to `CallableValueFamily`) disagrees with the
  contract's declared family at that position - including a parameter the
  callee's body never reads, so no body instruction is needed to expose a
  type mismatch

This is the runtime half of the canonical callable contract; arity alone is
also checked statically by the verifier (see
[`verifier.md`](verifier.md#callable-arity-enforcement)) - the VM cannot
skip its own family check on the assumption the verifier already proved
arity, since the verifier cannot prove runtime family from register
storage alone. Both checks independently agree with the same canonical
signature (see [`semcode.md`](semcode.md#callable-signature-sig0)), never
a caller-derived or independently-reconstructed one.

`push_frame` also enforces the `Calls` quota (#1759), as the last admission
check before the frame is actually constructed - after signature validation
and the `Frames`/`StackDepth`/`Registers` quotas have already succeeded.
Because `push_frame` is the same single choke point described above, the
root/entry frame of an execution (the one `push_frame` call made with an
empty call stack) is structurally exempt: `max_calls = 0` still admits the
selected entry function, and only blocks its first *nested* invocation.
Builtins resolved inline via `try_eval_builtin_call` and effect opcodes
never call `push_frame`, so neither consumes a `Call`. Full semantics
(charge timing, orthogonality with `Steps`, `usize::MAX` overflow
discipline) are frozen in
`docs/roadmap/stable_foundation/ssf08_1759_steps_calls_contract_decision.md`.

## Runtime Ownership Slice

Current supported runtime ownership slice is:

- tuple `AccessPath`
- direct record field `AccessPath`
- frame-local borrow lifetime
- runtime write rejection on overlapping borrowed tuple and direct record field
  paths

Current overlap cases that reject:

- exact path equality
- borrowed parent, written child
- borrowed child, written parent

Current allowed case:

- sibling tuple paths
- sibling direct record fields

Current ownership conflict surface:

- runtime overlap rejection uses `BorrowWriteConflict`

Unsupported ownership behavior remains outside the VM contract here and is
specified explicitly in `runtime_ownership.md`.

## Effect Opcode Boundary

The current VM instruction set includes effect-oriented opcodes such as:

- `GateRead`
- `GateWrite`
- `PulseEmit`

Current core rule:

- effectful instructions are counted against the effect-call quota
- effectful instructions do not define capability policy semantics by
  themselves

Richer ABI and capability binding is outside this core contract PR and must not
be smuggled into the VM execution contract by implication.

## Host ABI Boundary

The host ABI is a trust boundary: a `PrometheusHostAbi` (or
`ApplicationHostAbi`) implementation is arbitrary caller-supplied code, and a
value it returns must be validated before it acquires VM-internal meaning.

The canonical raw quad domain is exactly the four byte values `0`, `1`, `2`,
`3`, mapping respectively to `N`, `F`, `T`, `S`. When a host-returned
`AbiValue::Quad(byte)` crosses into the VM (for example via `GateRead` or
`StateQuery`), the VM must validate `byte` against this domain before
constructing a `Value::Quad`.

Core rule:

- a raw byte inside `0..=3` converts to the corresponding `Value::Quad`
- a raw byte outside `0..=3` is invalid input at the boundary and must
  produce a deterministic `RuntimeError::HostAbi` error
- an out-of-domain byte must never be masked or truncated (e.g. `& 0b11`)
  into a plausible-but-arbitrary valid quad value

## Compatibility Rule

The VM must reject unsupported SemCode versions with a clear migration hint.

It must not:

- silently coerce one supported `SEMCODE*` family into another
- accept malformed bytecode by best effort
- treat verifier rejection as a normal runtime success path
