# SSF-08 Lane 5 / #1763 — Runtime Failure Taxonomy Decision

Status: **CONTRACT DECISION + FALSIFICATION CHECKPOINT ONLY. NO PRODUCTION
BEHAVIOR CHANGE.**
Baseline SHA: `bedec73865f9ae067c480b3ec6dddc6ec3843b0c` (`main`, confirmed
current via `git rev-parse origin/main`).
Target: `#1763` / `FA-08-005`.
Umbrella: SSF-08 umbrella #1579 remains OPEN.
Related standing residual: `#1902` remains OPEN, untouched by this
document.

This document freezes the exact public failure-model contract for
`sm_runtime_core::RuntimeTrap` and `sm_vm::RuntimeError` before any
production Rust is touched. It does not delete, add, or rename anything,
does not touch `#1902`, and does not change any golden snapshot,
diagnostic code, or archive format.

**Headline finding, up front:** `RuntimeTrap` (13 variants) is presented -
in its own doc comment history, in the frozen CTF trap-taxonomy document,
and in the enum's own shape - as shared failure vocabulary. Fresh evidence
proves only **4 of 13** variants are ever constructed anywhere in the
workspace (production or test): `AssertionFailed`, `BorrowWriteConflict`,
`DivisionByZero`, `ArithmeticOverflow`. The other 9 have **zero**
construction sites, and - this is the part that makes the finding load-
bearing rather than cosmetic - **5 of those 9 literally share a name with
a real, richer, actually-constructed top-level `RuntimeError` variant**
(`StackOverflow`, `StackUnderflow`, `VerifierRejected`, `CapabilityDenied`,
`QuotaExceeded`), creating exactly the false-authority hazard the issue
was filed to report: a reader (or a `match` author) can reasonably believe
`RuntimeTrap::StackOverflow` is what fires on stack exhaustion, when the
real, only-ever-constructed value is the unrelated
`RuntimeError::StackOverflow`.

## 1. Baseline and scope

Depends on: `#1759` (Steps/Calls, CLOSED), `#1900` (EffectCalls overflow,
CLOSED), `#1761` (ConstPool, CLOSED), `#1760` (TraceEntries/
`max_debug_symbols_per_function`, CLOSED), `#1762` (execution-envelope
provenance, CLOSED). All five are now closed on `main` - the quota
architecture referenced throughout this document (§13) is fully settled,
not a moving target.

Gathered via two independent, read-only research passes over current
`main` (not the `#1763` issue body, not the old Phase-A audit, not prior
assistant summaries - those are hypotheses and historical evidence only,
per this checkpoint's own governing brief) - one over the Rust-source
construction-site reachability of both enums and every related typed
failure family, one over the frozen CTF trap-taxonomy document, active
specs, CLI/diagnostic consumers, and compatibility posture. Every claim
below traces to an exact file:line citation from one of those passes or
from this document's own direct reads.

## 2. Fresh `RuntimeTrap` definition

`crates/sm-runtime-core/src/lib.rs:152-167`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTrap {
    AssertionFailed,
    BorrowWriteConflict,
    StackOverflow,
    StackUnderflow,
    TypeMismatch,
    InvalidOpcode,
    InvalidJump,
    DivisionByZero,
    ArithmeticOverflow,
    CapabilityDenied,
    AbiViolation,
    VerifierRejected,
    QuotaExceeded(QuotaExceeded),
}
```

13 variants, 12 payload-free, one (`QuotaExceeded`) wrapping
`QuotaExceeded { kind: QuotaKind, limit: usize, used: usize }`
(`sm-runtime-core/src/lib.rs:145-150`). No `impl RuntimeTrap` block exists
anywhere in the workspace, so every construction must be a literal
`RuntimeTrap::Variant` expression - a workspace-wide grep for that literal
is exhaustive over all possible construction sites. Pinned byte-identically
in `tests/golden_snapshots/public_api/sm_runtime_core_lib.txt:59-72`.

## 3. Fresh `RuntimeError` definition

`crates/sm-vm/src/semcode_vm.rs:573-591`:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    BadHeader,
    UnsupportedBytecodeVersion { found: String, supported: String },
    BadFormat(String),
    UnknownFunction(String),
    InvalidJumpAddress { func: String, addr: usize },
    TypeMismatchRuntime(String),
    StackUnderflow,
    StackOverflow,
    QuotaExceeded(QuotaExceeded),
    VerifierRejected(RejectReport),
    UnknownVariable(String),
    InvalidStringId(u16),
    HostAbi(AbiError),
    CapabilityDenied(CapabilityDenied),
    Trap(RuntimeTrap),
}
```

15 variants. `Trap(RuntimeTrap)` is the sole variant wrapping
`sm_runtime_core::RuntimeTrap`. Pinned byte-identically in
`tests/golden_snapshots/public_api/sm_vm_semcode_vm.txt:78-93`.
`Display` (`semcode_vm.rs:593-627`) special-cases two `Trap(...)` inner
variants (`AssertionFailed`, `BorrowWriteConflict`) and falls back to a
generic `"runtime trap: {:?}"` for the other two live ones.

## 4. `RuntimeTrap` production reachability matrix

| Variant | Prod. construction sites | Test-only sites | Reachable via `RuntimeError::Trap` in a real caller path? | Disposition candidate |
|---|---|---|---|---|
| `AssertionFailed` | 1 - `semcode_vm.rs:2836` (`Opcode::Assert`) | 0 (test refs are `matches!` patterns only) | **Yes** | live |
| `BorrowWriteConflict` | 1 - `semcode_vm.rs:3171` (`ensure_write_path_allowed`) | 0 (patterns only) | **Yes** | live |
| `DivisionByZero` | 3 - `semcode_vm.rs:3393,3401,3409` (`fx_div_raw`/`i32_div_raw`/`i32_mod_raw`) | 4 via `assert_traps_under_all_opt_levels` | **Yes** | live |
| `ArithmeticOverflow` | 7 - `semcode_vm.rs:2566,3378,3383,3388,3396,3404,3412` | 2 via the same test helper | **Yes** | live |
| `StackOverflow` | **0** (exhaustive workspace grep) | 0 | No | real channel is `RuntimeError::StackOverflow` (`semcode_vm.rs:3096`) - **name collision** |
| `StackUnderflow` | **0** | 0 | No | real channel is `RuntimeError::StackUnderflow` (`semcode_vm.rs:2911`) - **name collision** |
| `TypeMismatch` | **0** | 0 | No | real channel is `RuntimeError::TypeMismatchRuntime(String)` (56 sites) |
| `InvalidOpcode` | **0** | 0 | No | real channel is `RuntimeError::BadFormat(String)` via `map_format_err` |
| `InvalidJump` | **0** | 0 | No | real channel is `RuntimeError::InvalidJumpAddress{func,addr}` (2 sites) |
| `CapabilityDenied` | **0** | 0 | No | real channel is `RuntimeError::CapabilityDenied(CapabilityDenied)` (8 sites) - **name collision**, and the dead trap variant is payload-free so it could never have carried the real struct |
| `AbiViolation` | **0** | 0 | No | real channel is `RuntimeError::HostAbi(AbiError)` (18 sites) |
| `VerifierRejected` | **0** | 0 | No | real channel is `RuntimeError::VerifierRejected(RejectReport)` (9 sites) - **name collision**, payload-free dead variant could never have carried the real `RejectReport` |
| `QuotaExceeded(QuotaExceeded)` | **0** | 0 | No | real channel is `RuntimeError::QuotaExceeded(QuotaExceeded)` (3 lexical choke points, fans to 6 of 7 `QuotaKind`s) - **name AND payload-type collision** |

`src/bin/smc.rs::vm_trap_message_needle` is an exhaustive `match` over all
13 variants - required by the Rust compiler because the enum has no
wildcard arm. **This exhaustiveness is a compile-time property of the
enum's declared shape, not evidence that the VM ever emits 9 of the 13
values at runtime.** The two facts (compiler-forced exhaustive match vs.
actual runtime reachability) are independent, and the table above answers
the second question empirically, not by inference from the first.

Both `run_verified_entry_semcode_with_config` (canonical verified path)
and `run_semcode_with_entry_and_config` (raw/bypass path) terminate in the
same private function, `run_vm_program_view_with_entry_and_config_with_observation_runtime`
(`semcode_vm.rs:1057`), which calls `push_frame` then `exec_loop` - all
four live variants are reached from there or a direct callee, so they are
reachable from **both** execution paths, not merely the raw one.

## 5. `RuntimeError` production reachability matrix

| Variant | Prod. construction count | Stage | Payload | Same-named `RuntimeTrap`? |
|---|---|---|---|---|
| `BadHeader` | 1 | raw-bytes decode (raw path only) | none | none |
| `UnsupportedBytecodeVersion{found,supported}` | 1 | raw-bytes decode (raw path only) | `{found,supported}` | none |
| `BadFormat(String)` | 29 (7 raw-only, 22 shared) | decode + live dispatch | message | conceptually `InvalidOpcode` (dead) |
| `UnknownFunction(String)` | ~14 | entry resolution + `Call`/`ClosureCall` dispatch | name | none |
| `InvalidJumpAddress{func,addr}` | 2 | live dispatch (`Jmp`/`JmpIf`) | `{func,addr}` | `InvalidJump` (dead) |
| `TypeMismatchRuntime(String)` | 56 | live dispatch, coercion helpers, call-boundary validation | message | `TypeMismatch` (dead) |
| `StackUnderflow` | 1 | live dispatch (`Ret`) | none | `StackUnderflow` - name collision, dead |
| `StackOverflow` | 1 | frame-push admission (`push_frame`, remaps `QuotaExceeded(StackDepth)`) | none (discards the `QuotaExceeded` struct it remaps from) | `StackOverflow` - name collision, dead |
| `QuotaExceeded(QuotaExceeded)` | 3 lexical (`enforce_quota`, `charge_counter` x2), fans to `Frames`/`StackDepth`(→remapped)/`Registers`/`Steps`/`Calls`/`EffectCalls` | frame push, register growth, opcode fuel, call/effect admission | `{kind,limit,used}` | `QuotaExceeded` - name+payload-type collision, dead |
| `VerifierRejected(RejectReport)` | 9 | pre-execution admission, byte-accepting compat shims only | full `RejectReport{diagnostics}` | `VerifierRejected` - name collision, dead |
| `UnknownVariable(String)` | 1 | live dispatch (`LoadVar`) | name | none |
| `InvalidStringId(u16)` | 2 | live dispatch (string/symbol lookup) | id | none |
| `HostAbi(AbiError)` | 18 | host-ABI boundary, post-capability-check | `{call,kind,message}` | `AbiViolation` (dead) |
| `CapabilityDenied(CapabilityDenied)` | 8 | host-ABI boundary, pre-check | full struct | `CapabilityDenied` - name collision, dead |
| `Trap(RuntimeTrap)` | 12 (sum of the 4 live inner variants) | live dispatch | `RuntimeTrap` | - |

Every `RuntimeError` variant has real, current, correctly-staged
production authority. None requires removal, renaming, or restructuring.

## 6. Failure-stage classification

Fresh evidence, not naming, sorts every current failure into exactly one
of these stages - all seven categories the governing brief asked to
evaluate genuinely exist as distinct classes in current code, not merely
as naming accidents:

- **A. Verification/admission rejection** (before trusted execution
  begins): `RuntimeError::VerifierRejected(RejectReport)`, carrying
  `sm-verify`'s own `RejectReport{diagnostics: Vec<VerificationDiagnostic>}`.
- **B. Execution structural/runtime error** (defense-in-depth validation
  during dispatch): `InvalidJumpAddress`, `TypeMismatchRuntime`,
  `StackUnderflow`, `UnknownFunction`, `UnknownVariable`,
  `InvalidStringId`, `BadFormat` (live-dispatch half).
- **C. Resource exhaustion**: `RuntimeError::QuotaExceeded(QuotaExceeded)`,
  settled architecture per `#1759`/`#1900`/`#1761`/`#1760` (§13).
- **D. Semantic program trap** (an otherwise-admitted program's own
  runtime semantics deliberately trap): `AssertionFailed`,
  `BorrowWriteConflict`, `DivisionByZero`, `ArithmeticOverflow` - the
  4 live `RuntimeTrap` variants, and only these four.
- **E. Capability-policy rejection**: `RuntimeError::CapabilityDenied
  (CapabilityDenied)`, carrying `prom-cap`'s own
  `{capability,call,code,manifest,message}`.
- **F. Host/ABI failure**: `RuntimeError::HostAbi(AbiError)`, carrying
  `prom-abi`'s own `{call,kind,message}`.
- **G. Format/version rejection** (pre-execution, raw-decode path only):
  `BadHeader`, `UnsupportedBytecodeVersion`, `BadFormat` (decode half).

These are genuinely separate semantic classes, each with its own owning
crate and its own structured payload type (`RejectReport`,
`QuotaExceeded`, `CapabilityDenied`, `AbiError`) - not implementation
accidents. Category D is the one, and only, category `RuntimeTrap`'s live
variants actually populate.

## 7. CTF/PCC authority cross-check

`docs/roadmap/language_maturity/core_trust_freeze/trap_taxonomy.md`
(status: frozen for PCC core readiness), §3 "Frozen trap classes"
(13 rows). Per-row cited evidence, verbatim, classified:

| Row | Cited authority (verbatim) | Classification |
|---|---|---|
| Malformed SemCode / unsupported header | function name only | neither enum |
| Unknown opcode | function name only | neither enum |
| Invalid jump target | `RuntimeError::InvalidJumpAddress` | RuntimeError, accurate |
| Missing call target | `RuntimeError::UnknownFunction` | RuntimeError, accurate |
| Type mismatch runtime | `RuntimeError::TypeMismatchRuntime` | RuntimeError, accurate |
| Stack underflow | `RuntimeError::StackUnderflow` | RuntimeError, accurate |
| Stack overflow | `RuntimeTrap::StackOverflow` | **RuntimeTrap, INACCURATE** - zero construction sites (§4); real evidence is `RuntimeError::StackOverflow` |
| Quota exceeded | bare `QuotaExceeded`/`RuntimeQuotas` (unprefixed) | ambiguous, resolves in practice to `RuntimeError::QuotaExceeded` |
| Division by zero | no code citation (test-only) | ambiguous |
| Arithmetic overflow | `RuntimeTrap::ArithmeticOverflow` | RuntimeTrap, **accurate** (genuinely constructed) |
| Assertion failure | no code citation (test-only) | ambiguous |
| Borrow-write conflict | no code citation (test-only) | ambiguous |
| Verifier rejected | no code citation (test-only) | ambiguous |

Net: of 13 rows, 4 cite `RuntimeError::X` (all accurate), 2 cite
`RuntimeTrap::X` (one accurate - `ArithmeticOverflow` - one **inaccurate**
- `StackOverflow`), 1 ambiguous/bare, 6 cite no concrete Rust path at all.
**This frozen document does not require `RuntimeTrap` to be the universal
failure channel** - where it does cite something concrete, `RuntimeError`
dominates 4-to-1, and the one place it cites `RuntimeTrap` incorrectly is
itself independent, additional evidence for this checkpoint's headline
finding. Capability/ABI-denial surfaces are explicitly placed **out of
scope** for this taxonomy by the document's own §6 ("owned by the
capability and boundary documentation lanes"), so it does not mis-cite
those two dead trap variants - it simply never claims them.

**Classification of the "Stack overflow" row: requires a frozen-doc
addendum, not a rewrite.** Per this checkpoint's own governing brief, do
not silently edit historical frozen evidence. The row's E1-code citation
is a factual pointer to a specific Rust path that turned out to be wrong
(the real evidence is `RuntimeError::StackOverflow`, not
`RuntimeTrap::StackOverflow`); correcting it is a future implementation-
checkpoint task, executed as an addendum that preserves the original
row's text and appends the correction, matching the append-only pattern
already used throughout SSF-08's own decision documents. Not done here -
this checkpoint does not edit `trap_taxonomy.md`.

## 8. CLI/diagnostic consumer audit

`src/bin/smc.rs::vm_trap_message_needle` (L1451-1469) is an exhaustive
match over `RuntimeTrap`'s 13 variants, called from exactly one place:
`diagnostic_from_vm_error` (L1399-1430), which itself branches on the
**top-level `RuntimeError`** with `Trap(trap) => ...` as one arm out of
two - the `other` catch-all handles every other `RuntimeError` variant
generically via a second exhaustive helper, `vm_error_code` (L1471-1491),
plus `.to_string()`. This confirms structurally what §4/§5 confirm
empirically: `RuntimeError` is the real caller-visible dispatch authority;
`RuntimeTrap` is a nested case handled one level deeper, and its own
exhaustive match narrows trivially once the enum narrows - the compiler
will force every dead-variant arm to be deleted, which is mechanical
fallout, not a design choice.

`crates/smc-cli/src/app.rs` never references `RuntimeTrap` at all (zero
hits); its only two `RuntimeError` match sites (`cmd_run`, L2921-2965)
match `CapabilityDenied`/`HostAbi` specifically for audit-recording
purposes and fall through `_ => {}` for everything else, including any
`Trap(_)`. No JSON/result-envelope classification specific to
`RuntimeTrap` exists anywhere in `app.rs`.

Deleting the 9 dead `RuntimeTrap` variants removes only unreachable
`vm_trap_message_needle` match arms - it does not and cannot change any
real diagnostic behavior, since those arms have never fired.

## 9. Payload-preservation analysis

Hard constraint (governing brief §12): no candidate consolidation may
erase structured failure information. Checked for the selected narrowing:

- The 9 variants proposed for removal are **either already payload-free**
  (`StackOverflow`, `StackUnderflow`, `TypeMismatch`, `InvalidOpcode`,
  `InvalidJump`, `CapabilityDenied`, `AbiViolation`, `VerifierRejected` -
  8 of 9) **or wrap a struct that is never populated because the variant
  is never constructed** (`QuotaExceeded(QuotaExceeded)` - the 9th).
  Since none of the 9 is ever constructed today, deleting them deletes
  zero live information - there is no in-flight payload to lose, because
  there was never a real value of any of these 9 variants to begin with.
- The **real** information for every one of these 9 concepts already
  lives, fully structured, on the corresponding `RuntimeError` variant
  (§5): `RuntimeError::QuotaExceeded` already carries the full
  `{kind,limit,used}`; `RuntimeError::CapabilityDenied` already carries
  the full `prom-cap` struct; `RuntimeError::VerifierRejected` already
  carries the full `RejectReport`; `RuntimeError::HostAbi` already
  carries the full `AbiError`. Removing the dead `RuntimeTrap` duplicates
  does not touch any of these - they are untouched, live `RuntimeError`
  variants, unrelated to `RuntimeTrap`'s own shape.
- **Payload information lost by the selected model: NO.**

## 10. Candidate models evaluated

**Model A - Trap = all execution failures.** Would require converting
`BadFormat`, `UnknownFunction`, `InvalidJumpAddress`, `TypeMismatchRuntime`,
`StackUnderflow`/`StackOverflow`, `QuotaExceeded`, `VerifierRejected`,
`CapabilityDenied`, `HostAbi`, `UnknownVariable`, `InvalidStringId` all
into `RuntimeTrap`-wrapped forms. This would force `RuntimeTrap` to grow
rich payload types matching every one of these - a large, unforced
architecture change with no current requirement (§9's constraint
otherwise forces payload loss). It would also conflate pre-execution
decode-stage rejection (`BadHeader`/`UnsupportedBytecodeVersion`, raw
path only) with execution-stage failure, contradicting the stage
separation independently confirmed real in §6. **Falsified.**

**Model B - Trap = semantic program trap only.** `RuntimeTrap` narrows to
its 4 genuinely-live variants (`AssertionFailed`, `BorrowWriteConflict`,
`DivisionByZero`, `ArithmeticOverflow`) - exactly stage D in §6. Every
other current `RuntimeTrap` concept already has real, richer,
correctly-staged authority on `RuntimeError` or a dedicated typed family
(§5, §9), with zero payload loss. Attempted falsification (per the
governing brief's own required adversarial pass):
- *"Verifier rejection already has richer typed authority"* - confirmed
  true (§5, §14 below); this is evidence **for** Model B, not against it.
- *"Quota exhaustion already has `RuntimeError::QuotaExceeded` with
  payload"* - confirmed true (§5, §13); evidence for.
- *"Capability denial already has its own typed authority"* - confirmed
  true (§5, §15); evidence for.
- *"ABI/host failure already has its own typed authority"* - confirmed
  true (§5, §15); evidence for.
- *"Structural VM failures already exist as `RuntimeError` variants"* -
  confirmed true (§5, §16); evidence for.
- *"Forcing everything through `RuntimeTrap` may throw away payload or
  duplicate existing authorities"* - this is exactly what Model A would
  do and Model B does not; evidence for Model B, against Model A.

None of the adversarial angles defeat Model B - each one, on fresh
evidence, turns out to be a reason Model B is correct rather than a
reason to reject it. **Survives falsification. Selected.**

**Model C - RuntimeError is the only public authority; eliminate
RuntimeTrap.** `RuntimeTrap` is the only channel for the 4 live semantic-
trap variants today, and CTF's own `trap_taxonomy.md` already organizes
around a "trap class" concept matching exactly this 4-variant grouping
(assertion, div-by-zero, arithmetic overflow, borrow-write-conflict are
each individually frozen trap classes in that document). Eliminating a
working, correctly-scoped, actively-constructed 4-variant type with no
current defect would be an unforced architecture change with no
evidenced benefit - the governing brief explicitly forbids "no new error
architecture merely for aesthetic symmetry" and "no speculative
unification." **Falsified** - no evidence found that collapsing the live
4-variant `RuntimeTrap` into flat `RuntimeError` variants fixes anything
Model B doesn't already fix, and doing so would be a materially larger,
unjustified breaking change (removing an actively-used nested type, not
merely deleting dead vocabulary).

**Model D - document the current 13-variant split as intentional,
mark unused variants reserved.** Aggressively challenged per the
governing brief's own instruction, since inert variants are exactly the
false-authority problem #1763 reports. No compatibility authority was
found that requires preserving any of the 9 dead variants: no semver/
stability policy document exists for either enum (checked
`compatibility_statement.md`, `stability_and_compatibility.md`,
`compatibility_policy_stack.md` - none discusses Rust enum-variant
stability); neither type is ever serialized (`RuntimeTrap` derives only
`Debug, Clone, Copy, PartialEq, Eq`; `RuntimeError` derives only `Debug,
Clone, PartialEq, Eq`; neither crate depends on `serde`); no discriminant
is ever cast to a numeric type; `prom-audit`'s own persisted archive
format (`docs/spec/audit.md`) never references either type. The only
actual enforcement mechanism is `tests/public_api_contracts.rs`'s
mechanical drift guard, which exists specifically to require an
**intentional, reviewed** snapshot update on any change - it is a
review-forcing gate, not a stability promise, and it is the same gate
`#1760`/`#1761` already used to narrow `QuotaKind` twice in this exact
program. Five of the 9 candidates for removal additionally **actively
mislead** by name-colliding with a real `RuntimeError` variant doing the
actual job - "reserved" status would leave that specific hazard in place
indefinitely with no justification. **Falsified.**

## 11. Selected definition of "trap"

> **`RuntimeTrap` means: a failure caused by executing an otherwise-
> admitted Semantic program, where the instruction/program semantics
> themselves deliberately trap.** It is not administrative (quota,
> capability, ABI), not structural/defensive VM validation, and not
> pre-execution admission rejection. Its members are exactly:
> `AssertionFailed`, `BorrowWriteConflict`, `DivisionByZero`,
> `ArithmeticOverflow`.
>
> **`RuntimeError` means: the complete, top-level, caller-visible failure
> channel for every stage of the runtime's lifecycle** - decode/format
> rejection, entry resolution, structural/defensive validation, resource
> exhaustion, verifier rejection (via the compatibility shims that admit
> bytes directly), capability denial, host/ABI failure, and semantic
> program traps (wrapped via `Trap(RuntimeTrap)`). It is the sole type a
> real caller ever receives from any `pub fn run_*`/`verify_*` entrypoint.

## 12. Per-`RuntimeTrap`-variant disposition

| Variant | Disposition | Justification |
|---|---|---|
| `AssertionFailed` | **KEEP_ACTIVE** | live, stage D, correct |
| `BorrowWriteConflict` | **KEEP_ACTIVE** | live, stage D, correct |
| `DivisionByZero` | **KEEP_ACTIVE** | live, stage D, correct |
| `ArithmeticOverflow` | **KEEP_ACTIVE** | live, stage D, correct |
| `StackOverflow` | **REMOVE_DUPLICATE** | zero construction; real authority `RuntimeError::StackOverflow` |
| `StackUnderflow` | **REMOVE_DUPLICATE** | zero construction; real authority `RuntimeError::StackUnderflow` |
| `TypeMismatch` | **REMOVE_DUPLICATE** | zero construction; real authority `RuntimeError::TypeMismatchRuntime` |
| `InvalidOpcode` | **REMOVE_DUPLICATE** | zero construction; real authority `RuntimeError::BadFormat` |
| `InvalidJump` | **REMOVE_DUPLICATE** | zero construction; real authority `RuntimeError::InvalidJumpAddress` |
| `CapabilityDenied` | **REMOVE_DUPLICATE** | zero construction; real authority `RuntimeError::CapabilityDenied` |
| `AbiViolation` | **REMOVE_DUPLICATE** | zero construction; real authority `RuntimeError::HostAbi` |
| `VerifierRejected` | **REMOVE_DUPLICATE** | zero construction; real authority `RuntimeError::VerifierRejected` |
| `QuotaExceeded(QuotaExceeded)` | **REMOVE_DUPLICATE** | zero construction; real authority `RuntimeError::QuotaExceeded` |

No variant is dispositioned `RENAME`, `MOVE`,
`DEPRECATE_WITH_LIVE_COMPATIBILITY`, or `RESERVED_WITH_JUSTIFIED_AUTHORITY`
- §10's falsification of Model D found no compatibility authority
justifying reservation for any of the 9, and none of the 9 needs
renaming or moving since its real-named counterpart already exists and
is correctly placed on `RuntimeError`.

`RuntimeError` itself requires **no** variant disposition change - every
one of its 15 variants (§5) already has real, correctly-staged production
authority. `Trap(RuntimeTrap)` is **retained**, narrowed to wrap the
4-variant `RuntimeTrap`.

## 13. Quota failure authority (fresh re-confirmation)

`QuotaKind` (`sm-runtime-core/src/lib.rs:134-143`): exactly 7 variants -
`Steps, Calls, StackDepth, Frames, Registers, SymbolTable, EffectCalls`.
**Confirmed: no `ConstPool`, no `TraceEntries`** (`#1761`/`#1760` both
closed). `RuntimeQuotas` (`sm-runtime-core/src/lib.rs:169-184`): exactly
8 fields, the 8th being `max_debug_symbols_per_function`, whose own doc
comment states verbatim: *"Not a `QuotaKind` member: never charged by
`sm-vm`, never surfaces as `RuntimeError::QuotaExceeded`."* Confirmed
directly in `sm-verify/src/lib.rs:1358-1369`: checked as a plain
`RuntimeQuotas` field read inside `verify_function_code`, with no
`QuotaKind`/`RuntimeTrap` involvement.

`RuntimeError::QuotaExceeded(QuotaExceeded)` is confirmed the active
quota-exhaustion channel (§5), fanning out from 3 lexical choke points
(`enforce_quota`, `charge_counter`) to 6 of 7 `QuotaKind`s (`SymbolTable`
is enforced statically by `sm-verify`, outside `sm-vm`'s charge path
entirely, per the settled `#1761` ownership split - not a gap).

`RuntimeTrap::QuotaExceeded(QuotaExceeded)` is **not retained merely
because it once existed** - it has zero construction sites, and this
checkpoint does not redirect active quota failures through it, since no
authority (frozen or active) requires that change. Disposition:
`REMOVE_DUPLICATE` (§12).

## 14. Verifier-rejection boundary (preserved)

`RuntimeError::VerifierRejected(RejectReport)` is confirmed the
authoritative caller-visible channel for verifier rejection (§5, 9
production sites, all `.map_err(RuntimeError::VerifierRejected)?`
immediately after `verify_semcode_token(_with_quotas)`), occurring before
trusted execution begins - stage A in §6, structurally distinct from
stage D (semantic program trap). This model does **not** collapse
verifier rejection into a payload-free `RuntimeTrap::VerifierRejected` -
that variant is dead and stays dead; the real, full `RejectReport`
authority is untouched and unaffected by this decision.

## 15. Capability/ABI authorities (preserved)

`RuntimeError::CapabilityDenied(CapabilityDenied)` (8 sites, pre-host-call
capability pre-check) and `RuntimeError::HostAbi(AbiError)` (18 sites,
post-capability-check host-boundary failure) both carry their own full,
richer, independently-owned structs (`prom-cap`'s `CapabilityDenied`,
`prom-abi`'s `AbiError`) - confirmed real, current, and unaffected by
this decision. `RuntimeTrap::CapabilityDenied` and
`RuntimeTrap::AbiViolation` are confirmed dead duplicates of these richer
channels (shared name, zero payload, zero authority) - `REMOVE_DUPLICATE`
per §12.

## 16. Structural VM failures (classified separately from semantic traps)

`StackOverflow`, `StackUnderflow`, `TypeMismatchRuntime`,
`InvalidJumpAddress`, and the live-dispatch half of `BadFormat` are
defense-in-depth runtime validation, not semantic program traps - they
fire on internal VM-invariant violations (stack discipline, type
coercion, jump-target bounds, opcode decode), not on a legally-admitted
program's own deliberate semantics. All are reachable from **both** the
canonical verified path and the raw/bypass path (they live inside
`exec_loop_with_profile` or its direct callees, shared by both paths per
§4's call-graph confirmation) - none is removed merely because
verifier-first execution makes it rare in the canonical path; the
raw/bypass path intentionally exercises this defense-in-depth layer, and
these `RuntimeError` variants remain exactly as-is, untouched by this
decision.

## 17. Final target failure taxonomy

| Failure family | Stage | Owning type | Caller-visible channel | Payload | `RuntimeTrap` member? |
|---|---|---|---|---|---|
| Assertion failure | execution, semantic trap | `sm_runtime_core::RuntimeTrap` | `RuntimeError::Trap(RuntimeTrap::AssertionFailed)` | none | **YES** |
| Borrow/write conflict | execution, semantic trap | `sm_runtime_core::RuntimeTrap` | `RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict)` | none | **YES** |
| Division by zero | execution, semantic trap | `sm_runtime_core::RuntimeTrap` | `RuntimeError::Trap(RuntimeTrap::DivisionByZero)` | none | **YES** |
| Arithmetic overflow | execution, semantic trap | `sm_runtime_core::RuntimeTrap` | `RuntimeError::Trap(RuntimeTrap::ArithmeticOverflow)` | none | **YES** |
| Stack overflow | frame-push admission | `sm_vm::RuntimeError` | `RuntimeError::StackOverflow` | none | no |
| Stack underflow | execution (`Ret`) | `sm_vm::RuntimeError` | `RuntimeError::StackUnderflow` | none | no |
| Runtime type mismatch | execution/coercion/call-boundary | `sm_vm::RuntimeError` | `RuntimeError::TypeMismatchRuntime(String)` | message | no |
| Invalid opcode | decode/live dispatch | `sm_vm::RuntimeError` | `RuntimeError::BadFormat(String)` | message | no |
| Invalid jump | execution dispatch | `sm_vm::RuntimeError` | `RuntimeError::InvalidJumpAddress{func,addr}` | func,addr | no |
| Bad header/format | pre-execution decode (raw path) | `sm_vm::RuntimeError` | `RuntimeError::BadHeader`/`BadFormat` | none/message | no |
| Unsupported bytecode version | pre-execution decode (raw path) | `sm_vm::RuntimeError` | `RuntimeError::UnsupportedBytecodeVersion{found,supported}` | found,supported | no |
| Unknown function | entry resolution/call dispatch | `sm_vm::RuntimeError` | `RuntimeError::UnknownFunction(String)` | name | no |
| Quota exhaustion | frame/register/steps/calls/effects | `sm_runtime_core::RuntimeQuotas` | `RuntimeError::QuotaExceeded(QuotaExceeded)` | kind,limit,used | no |
| Verifier rejection | pre-execution admission | `sm_verify::RejectReport` | `RuntimeError::VerifierRejected(RejectReport)` | full diagnostics | no |
| Capability denial | host boundary, pre-check | `prom_cap::CapabilityDenied` | `RuntimeError::CapabilityDenied(CapabilityDenied)` | full struct | no |
| Host/ABI failure | host boundary, post-check | `prom_abi::AbiError` | `RuntimeError::HostAbi(AbiError)` | call,kind,message | no |

No overlapping authority remains unexplained - every row has exactly one
owning type and one caller-visible channel.

## 18. Public API consequence

`RuntimeTrap` narrows from 13 to 4 variants - a source-visible, intentional
public API narrowing. `RuntimeError`'s own shape does **not** change
(still 15 variants, `Trap(RuntimeTrap)` unchanged in position and name -
only the type it wraps narrows). Exact guarded snapshot impact:

- `tests/golden_snapshots/public_api/sm_runtime_core_lib.txt` - the
  `RuntimeTrap` enum body narrows to 4 lines.
- `tests/golden_snapshots/public_api/sm_vm_semcode_vm.txt` - **unaffected**;
  its own pinned text is `Trap(RuntimeTrap),` (a type reference, not an
  inline expansion of `RuntimeTrap`'s variants), which does not change
  when `RuntimeTrap`'s own definition narrows.

This decision checkpoint does **not** touch either snapshot. A future
implementation checkpoint must reproduce the exact `#1760`/`#1761`/`#1762`
evidence discipline: run `cargo test --test public_api_contracts` first
against the stale golden (must fail RED, diff isolated to exactly the 9
removed `RuntimeTrap` lines), then regenerate via
`SM_UPDATE_PUBLIC_API_SNAPSHOTS=1`, review the diff manually, confirm
GREEN.

## 19. Compatibility consequence

No semver/stability policy document was found governing either enum's
variants (§10, Model D falsification). Neither type is ever serialized,
persisted, or cast to a stable numeric discriminant anywhere in the
workspace - `RuntimeTrap` derives only `Debug, Clone, Copy, PartialEq,
Eq`; `RuntimeError` derives only `Debug, Clone, PartialEq, Eq`; neither
`sm-runtime-core` nor `sm-vm` depends on `serde`; `prom-audit`'s own
persisted replay-archive format never references either type. **Removing
9 variants is a breaking Rust source API change** (any external code
matching on them exhaustively without a wildcard arm would fail to
compile), but it is not blocked by any frozen compatibility promise, and
it is acceptable in the current Stable Foundation phase under the same
mechanical drift-guard discipline `#1760`/`#1761` already used twice for
comparable enum narrowings. No serialized/on-disk representation is
affected. **No compatibility blocker.**

## 20. Error precedence

This is a pure vocabulary narrowing - it deletes unreachable enum
variants and corrects documentation, but changes **zero** runtime
behavior, since none of the 9 removed variants was ever constructed in
the first place. Every existing precedence relationship (verifier
rejection vs. runtime execution; `Steps` quota vs. semantic trap; `Calls`
quota vs. frame errors; capability denial vs. `EffectCalls` charge; host
ABI error vs. runtime trap; ownership conflict vs. later instruction
semantics) is governed today by the real `RuntimeError` variants and the
4 live `RuntimeTrap` variants exactly as they already fire - none of that
charge-order/precedence logic is touched, reordered, or reinterpreted by
this decision. The future implementation must not change any execution
order as a side effect of this taxonomy correction.

## 21. Exact future implementation mechanic (frozen)

Binding on the future implementation checkpoint; nothing here is left as
an open choice.

- **`RuntimeTrap`**: remove `StackOverflow`, `StackUnderflow`,
  `TypeMismatch`, `InvalidOpcode`, `InvalidJump`, `CapabilityDenied`,
  `AbiViolation`, `VerifierRejected`, `QuotaExceeded(QuotaExceeded)`.
  Retain `AssertionFailed`, `BorrowWriteConflict`, `DivisionByZero`,
  `ArithmeticOverflow` unchanged, in their current order.
- **`RuntimeError`**: no variant change. `Trap(RuntimeTrap)` retained,
  narrowed automatically by `RuntimeTrap`'s own narrowing.
- **`src/bin/smc.rs::vm_trap_message_needle`**: compiler-forced deletion
  of the 9 now-nonexistent match arms; the 4 remaining arms are
  unchanged text. No new wildcard arm - keep the exhaustive-match
  discipline.
- **`docs/spec/vm.md`'s "Trap And Error Model" section**: correct the
  "Current public runtime error families" list to (a) add `AssertionFailed`,
  `DivisionByZero`, `ArithmeticOverflow` (real, currently omitted), (b)
  add `HostAbi`, `CapabilityDenied` (real, currently omitted), (c)
  represent `Trap(RuntimeTrap)` as the wrapping variant rather than
  flatly listing `BorrowWriteConflict` as if it were top-level.
- **`docs/roadmap/language_maturity/core_trust_freeze/trap_taxonomy.md`**:
  append an addendum (not a rewrite) to the "Stack overflow" row
  correcting its E1-code citation from `RuntimeTrap::StackOverflow` to
  `RuntimeError::StackOverflow`, preserving the original row's text.
- **Public API**: RED→GREEN on `sm_runtime_core_lib.txt` only, per §18.
- **Tests**: add a reachability/taxonomy regression proving the 4 kept
  variants remain constructible exactly as before (no behavior change),
  and confirming compiler-fallout from the 9 removed variants is confined
  to `vm_trap_message_needle`'s match arms (no other production reader
  breaks) - treat any other compile failure as a STOP, not something to
  route around.
- **No** change to `RuntimeTrap::QuotaExceeded`'s redirect, no new
  `RuntimeTrap` variant, no rename of any kept variant, no change to
  charge order or execution precedence (§20).

## 22. AC4.d consequence

**Not satisfied now** - this is a decision, not an implementation.
Becomes **satisfiable** once a future, separately-authorized
implementation checkpoint executes §21's frozen mechanic: verification
rejection, runtime quota exhaustion, semantic trap, capability denial,
host/ABI failure, and structural runtime errors already each have an
explicit, deterministic, correctly-staged channel matching production
behavior (§17) - the only remaining work is deleting `RuntimeTrap`'s dead
vocabulary so the *type* stops overstating what the *behavior* already
correctly does.

## 23. AC4.e consequence

**Not satisfied now.** Becomes **satisfiable** once the future
implementation lands §21's `docs/spec/vm.md` correction and the
`trap_taxonomy.md` addendum - at that point active docs and diagnostics
will describe exactly the channels in §17, without implying any of the 9
removed `RuntimeTrap` variants was ever emitted.

## 24. `#1902` non-dependency

`#1902` (the `snake_learning.sm` `VerifiedLocal` `Steps` overflow) is
independent of this decision and was not used, directly or indirectly, to
justify any part of it - no quota baseline value changes here, no
`RuntimeTrap`/`RuntimeError` disposition depends on `#1902`'s residual,
and nothing in this document touches it. `#1902` remains untouched,
tracked separately.

## 25. Implementation boundary

This document freezes every implementation-significant choice §25 of the
governing brief requires before `CONTRACT FROZEN = YES` /
`READY TO IMPLEMENT = YES` can be claimed:

- exact `RuntimeTrap` variants after implementation (§12, §21)
- exact `RuntimeError` variants after implementation: unchanged (§5, §21)
- whether `RuntimeError::Trap` remains: yes, narrowed (§21)
- whether any variant is renamed: no (§12)
- payload types: unchanged, zero loss (§9)
- diagnostic mappings: `vm_trap_message_needle` narrows mechanically,
  `diagnostic_from_vm_error`/`vm_error_code` unaffected (§8, §21)
- active docs to modify: `docs/spec/vm.md` (§21), `trap_taxonomy.md`
  addendum (§7, §21)
- public API snapshots to update: `sm_runtime_core_lib.txt` only (§18)
- compatibility treatment: mechanical drift-guard RED→GREEN, no blocker
  (§19)
- test matrix: reachability/taxonomy regression + compiler-fallout proof
  (§21)

**Nothing above is left open.** A future implementation checkpoint may
proceed directly from this document without a further design pass.

**Wait for owner GO before implementation.**
