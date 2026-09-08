# Verifier Specification

Status: draft v0
Admission owner: `sm-verify`

## Purpose

This document defines the current SemCode admission contract before standard VM
execution.

The verifier is a public admission layer.
It is not an internal VM detail and it is not an optimizer.

## Public Surface

Current verifier surface is centered on:

- `verify_semcode`
- `VerifiedProgram`
- `VerifiedFunction`
- `RejectReport`
- `VerifiedSemCode`
- `VerifiedEntrySemCode`
- `EntryResolutionError`

Canonical token-producing admission path:

- `verify_semcode_token` — admits against the default `VerifiedLocal` resource
  quota profile.
- `verify_semcode_token_with_quotas` — identical admission logic, but the
  caller supplies the `RuntimeQuotas` profile to admit against (e.g.
  `KernelBound`). `verify_semcode_token` is a thin wrapper over this with
  `RuntimeQuotas::verified_local()`. Rejection diagnostics report the budget
  and usage from whichever profile was supplied, not a fixed profile name —
  callers should not assume a rejection implies `VerifiedLocal` limits.

Compatibility / legacy admission surface:

- `verify_semcode`

## Verification Scope

Current SemCode verification checks include:

- header validity
- supported version validity
- function and section integrity (the number of function definitions in an
  artifact is a static structural bound owned by `sm-format`
  (`MAX_FUNCTIONS`), independent of any runtime resource quota; the verifier
  does not re-derive or narrow it from a quota profile)
- canonical (unambiguous) instruction framing
- opcode validity
- opcode/header-revision consistency
- operand shape validity
- jump-target validity
- reachable control-flow closure (no end-of-stream fallthrough)
- string and debug reference validity
- register-budget validity (against the caller-selected `RuntimeQuotas`
  profile; see Public Surface)
- definite register assignment for functions carrying a canonical `SIG0`
  signature (every reachable register read is defined on every incoming
  path; see [Definite Register Assignment](#definite-register-assignment-1756--fa-07-016))
- program-wide runtime symbol-table budget validity (the number of *distinct*
  strings the VM will intern across every function, deduplicated by exact
  string value program-wide - not a per-function string-table entry count;
  also checked against the caller-selected `RuntimeQuotas` profile)
- call-target validity
- capability consistency with actual opcode usage

## Opcode Admission Matrix

The verifier treats opcode admission as part of SemCode admission.

This matrix documents the current public admission boundary. It uses operation
families rather than binary opcode numbers so that internal encodings can change
without turning this document into a stable bytecode ISA promise.

| Operation family | Admission status | Capability / boundary | Notes |
| --- | --- | --- | --- |
| Ordinary source-derived families: control flow, data movement, literal loading, arithmetic, comparison, calls, and returns | Admitted when produced by supported SemCode and consistent with the emitted contract | No extra capability beyond the artifact contract | Baseline verifier-admitted surface; this document intentionally does not stabilize binary opcode numbers. |
| `SEQUENCE_LEN` | Admitted only when the emitted contract carries `CAP_SEQUENCE_ITERATION` and the header family supports the sequence-iteration contract | Capability-gated | Built-in sequence lowering opcode for the admitted `Sequence(T)` iteration slice. |
| Effect-oriented host-boundary families such as `GateRead`, `GateWrite`, and `PulseEmit` | Admitted only when the emitted contract matches the required capability envelope | Capability-gated / host-boundary | These opcodes do not define capability policy semantics by themselves. |
| Ownership transport payloads admitted through `OWN0` | Admitted structurally only when the ownership transport slice is present and well formed | Header and capability consistency required | Covers tuple-only (`SEMCOD11`), direct record-field (`SEMCOD12`), `SequenceIndexStatic` `Borrow`/`Write` (`SEMCOD21`, `CAP_OWNERSHIP_SEQUENCE_PATHS`), and `AdtPayload` `Borrow`-only (`SEMCOD21`, `CAP_OWNERSHIP_ADT_BORROW_PATHS`) ownership transport. `Write(AdtPayload)` is rejected unconditionally under every header, including `SEMCOD21` - not a capability gap. |
| Unknown, unsupported, or malformed opcode encodings | Rejected | N/A | Rejection must happen before a successful VM execution path. |
| Opcode streams that fail operand, jump-target, reachable-control-flow, call-target, closure-function-target, register-budget, string-reference, or section-integrity checks | Rejected | N/A | Direct calls may resolve to declared functions or admitted builtins; closure targets must resolve to declared functions. These are verifier admission failures, not successful runtime executions. |

Current ownership-specific structural checks for ownership transport include:

- `OWN0` section presence and layout validity when ownership transport is used
- admitted ownership event kind validity
- tuple `AccessPath` payload validity under `SEMCOD11`
- direct record-field `AccessPath` payload validity under `SEMCOD12`
- structural admission for valid `Borrow(Field)` and `Write(Field)` payloads
- `SequenceIndexStatic` `AccessPath` payload validity (`Borrow`/`Write`) under
  `SEMCOD21`, independently re-derived from decoded content (not delegated
  to `sm-format`'s own decode-time gate) via `CAP_OWNERSHIP_SEQUENCE_PATHS`
  capability-consistency accounting (#1718)
- `AdtPayload` `AccessPath` payload validity in `Borrow` events only under
  `SEMCOD21`, via `CAP_OWNERSHIP_ADT_BORROW_PATHS` capability-consistency
  accounting; an `AdtPayload` component in a `Write` event is rejected
  unconditionally, before and independent of capability accounting (#1718)
- header/capability consistency for ownership transport

## Canonical Operand Value Domains

Operand shape validity is not only "a byte is present" — for fields with a
fixed value domain, the byte must also be the canonical encoding for that
domain. The verifier enforces:

- boolean literal operands (`LOAD_BOOL`): must be `0` (false) or `1` (true)
- quad literal operands (`LOAD_Q`): must be `0..=3` (the four-value quad
  domain)
- presence-flag operands (`CALL` and `CLOSURE_CALL` destination-present,
  `RET` source-present): must be `0` or `1`

A byte outside the canonical domain for its field is rejected at admission
(`OperandOutOfBounds`), not normalized downstream. This narrows the
previously admitted surface: byte values outside these domains that an
earlier verifier accepted are no longer admissible.

## Opcode/Header-Revision Consistency

Header identity constrains the executable opcode vocabulary (see
`docs/spec/semcode.md`). For every decoded instruction, the verifier
requires:

```text
artifact_header_revision >= opcode.minimum_semcode_revision()
```

before `VerifiedSemCode` is issued. `Opcode::minimum_semcode_revision` (in
`sm-format`) is the single, format-owned authority for this comparison -
`sm-verify` uses it directly for admission, and `sm-ir`'s header selection
uses the same underlying opcode/feature relationship (through its own IR-
level promotion predicates) to choose a header whose revision actually
covers everything a program emits. There is no separate, independently-
maintained opcode-to-revision table anywhere else.

A violation rejects with `AmbiguousInstructionFraming`'s sibling code,
`VerificationCode::OpcodeRequiresNewerHeader` - deliberately distinct from
`CapabilityViolation`. The two checks address different failure modes:

- `CapabilityViolation`: the artifact's header does not carry a capability
  bit an opcode requires. Because each header revision's capability set is
  fixed and cumulative, this check already transitively enforces a minimum
  header for any capability-gated opcode.
- `OpcodeRequiresNewerHeader`: the opcode requires no missing capability at
  all - it is structurally valid and every capability it needs (if any) is
  present - but the artifact's header revision predates the revision whose
  contract actually admits this opcode's semantics. This is a version-
  identity gap, not a capability gap, and only opcodes with no capability
  bit at all need this check to be independently load-bearing (see #1732 /
  FA-05-002).

`Opcode::minimum_semcode_revision` assigns a minimum revision explicitly to
every `Opcode` variant. Variants established as baseline are explicitly
assigned revision `1` (`SEMCODE0`). An opcode family with a *provable* later
introduction revision - backed by an actual repository decision record, not
by commit date alone - is explicitly assigned that later revision. The match
is intentionally exhaustive and has no wildcard/default revision arm, so a
new `Opcode` variant cannot acquire a revision policy implicitly. This
function must not imply stronger historical knowledge than the repository has
actually established. Currently the only family assigned a non-baseline
minimum revision is `QTruth` (revision `19`, `SEMCOD18`).

## Canonical Instruction Framing

A canonical SemCode function encoding must have exactly one unambiguous
structural interpretation (see `docs/spec/semcode.md`).

The `DBG0` debug-section tag is recognized by sniffing a fixed 4-byte
sequence, and its first byte (`0x44`) is also `TupleGet`'s opcode byte, so a
producer-emitted instruction stream can coincidentally spell the same bytes
as `DBG0` framing. A byte sequence that is simultaneously valid as `DBG0`
metadata and as a complete instruction stream is non-canonical.

The verifier detects this by checking, whenever a `DBG0` section was
recognized, whether the same bytes - read from immediately after the string
table, with no metadata-section recognition at all - would also form a
complete, well-formed instruction stream to the end of the function's code.
If both readings are valid, admission fails closed with
`AmbiguousInstructionFraming` rather than silently keeping the `DBG0`
reading.

This is a purely STRUCTURAL question, deliberately kept separate from
SEMANTIC admission:

- structural framing: opcode recognition, operand byte shape, and
  presence/count-controlled byte lengths, evaluated to determine only
  whether a complete instruction stream exists at all
- semantic admission: canonical literal value domains (`LOAD_Q`,
  `LOAD_BOOL`), canonical presence-flag domains (`CALL`, `CLOSURE_CALL`,
  `RET`), and canonical arity/cardinality domains (`MAKE_TUPLE` arity
  `>= 2`, `MAKE_RECORD` slot count `>= 1`), which remain a separate, later
  concern applied only to whichever single reading admission actually
  accepts

The alternative reading only needs to be structurally complete to count as
a genuine competing interpretation - a non-canonical operand value (for
example an out-of-domain literal byte) does not make an otherwise
shape-complete instruction reading any less structurally real, and does not
exempt an artifact from the ambiguity check. Gating ambiguity detection on
today's semantic admission policy would make the one-canonical-
interpretation invariant depend on that policy instead of being a
decoder-level fact, and would silently keep the `DBG0` reading whenever the
competing instruction reading merely contained a non-canonical literal.

This check reuses the verifier's own operand-shape decoder for the
alternative reading - the same function, same opcode-shape match, that
semantic admission uses, with canonical-domain enforcement turned off -
rather than a second, independently-maintained opcode-shape table.

`OWN0`'s tag byte (`0x4F`) is not a currently valid opcode, so this specific
ambiguity does not apply to ownership-section recognition.

## Reachable Control-Flow Closure

For every function, the verifier reuses the instruction boundaries, decoded
next offsets, and explicit jump targets collected by its normal instruction
walk. Starting at instruction offset zero, it admits only reachable successors
with these VM-derived rules:

- `RET` is terminal and has no successor
- `JMP` has only its explicit jump target
- `JMP_IF` has its explicit jump target and the decoded next instruction
- every other opcode, including `CALL` and `CLOSURE_CALL`, has the decoded next
  instruction as its successor

The two call forms are fallthrough operations structurally because the VM saves
the decoded next PC before entering the callee and resumes there after `RET`.

Every reachable successor must be an instruction start. A successor equal to
the end of the executable instruction stream is rejected with
`VerificationCode::ReachableFunctionFallthrough`; an empty executable stream is
the same defect at entry offset zero. Existing explicit jump-target range and
instruction-boundary checks remain independently enforced as
`InvalidJumpTarget`.

This proves structural control-flow closure, not program termination. A closed
infinite loop is admissible, and an ordinary trailing instruction may fall
through if it is unreachable from function entry. The verifier does not require
the final encoded instruction to be `RET`.

## Debug Reference Validity

A `DBG0` debug symbol's `pc` must reference a decoded instruction boundary,
not merely fall within the executable byte range. The verifier reuses the
same `instr_starts` set collected during its normal instruction walk (the
same authority the jump-target boundary check above already relies on) - it
does not run a second decoder or infer boundaries from opcode-byte scanning.
A `pc` that is numerically less than the instruction stream length but lands
inside a decoded instruction's operand bytes is rejected with
`VerificationCode::InvalidDebugSection`, the same code already used for an
out-of-range `pc`. This proves only that a debug reference denotes a real
instruction boundary; it does not claim source-line fidelity, debugger
correctness, or any stronger metadata semantics than that structural fact.

## Contract Rule

Standard execution uses the chain:

`emit SemCode -> verify_semcode -> (optional) require_entry -> execute`

Important rule:

- VM execution does not replace SemCode admission
- a valid producer path does not waive verifier admission
- entry resolution failure (`EntryResolutionError`) is distinct from verifier rejection (`RejectReport`); missing an entrypoint does not make the artifact itself invalid.

## Separation Rule

`sm-verify` must not become:

- a source parser
- a semantic runtime
- a VM executor
- a general optimizer

It is allowed to reject malformed or contract-inconsistent bytecode only.

Current ownership rule:

- verifier admits ownership payload structurally only
- verifier does not evaluate borrow overlap, release timing, or runtime alias
  policy

## Reject Model

Verifier rejection must preserve:

- the failing verification code
- enough function or offset context to debug the failure
- deterministic diagnostics for the same input artifact

`ReachableFunctionFallthrough` specifically means that control flow reachable
from function entry can advance to the end of the executable instruction
stream without first reaching an admitted terminal instruction.

## Callable Arity Enforcement

For every `Opcode::Call` site whose callee resolves to a known internal
function, the verifier checks the call's `argc` against the callee's decoded
canonical signature (see [`semcode.md`](semcode.md#callable-signature-sig0)),
rejecting a mismatch with `CallArgumentCountMismatch`. This is arity only:

- `sm-verify` cannot prove a `CALL` argument register's runtime family
  statically - registers are untyped storage - so it never attempts to; that
  half of the contract is the VM's responsibility, enforced immediately
  before `push_frame` (see [`vm.md`](vm.md#callable-runtime-family-enforcement))
- caller-supplied `argc` never defines callee arity; only the callee's own
  decoded signature does
- when the callee's artifact predates canonical signatures
  (`signature: None`), there is no contract to check and this enforcement is
  a no-op - unchanged behavior for pre-#1773 artifacts
- a missing `SIG0` section on a header that requires one
  (`SEMCODE_SIGNATURE_MIN_REVISION` or newer) is not a policy check at this
  layer at all: it is a structural decode rejection
  (`InvalidSignatureSection`) before admission ever reaches this check,
  since `SIG0` presence is derived deterministically from the header
  revision, never sniffed

Calls to builtins are out of scope for this check; only calls that resolve
to a known internal function (with a decoded canonical signature to check
against) are covered.

## Definite Register Assignment (#1756 / FA-07-016)

An in-range register identity (see the register-budget check above) is
necessary but not sufficient for admission. The verifier additionally
proves, for every function carrying a canonical `SIG0` signature: **every
reachable register read is definitely defined on every execution path that
can reach it** - not merely that the register number is within budget, and
not merely that *some* path defines it before that read.

This is a forward **MUST** dataflow analysis, not a MAY/reachability
analysis: the property proved is "definitely defined on *every* incoming
path," not "defined on *some* path." A future implementation that only
proves the latter (MAY semantics) does not satisfy this contract and must
not be described as equivalent - MUST is strictly stronger, and is the
property `RegisterSlot::Uninitialized` in `sm-vm` (#1770) already enforces
at runtime; this pass proves the same property statically, ahead of
execution.

The implementation computes this MUST guarantee via its logical dual: not
`DEFINED[n] = intersect(predecessor OUT)` directly, but `MISSING[n] = U \
DEFINED[n] = union(predecessor MISSING_OUT)` (De Morgan's law over the same
equations) - a register is rejected iff it is in `MISSING[n]` at the point
of read, the identical predicate phrased via the complement. This is an
implementation choice, not a semantic relaxation: the MAY-missing/union
formulation computes the exact same MUST fixed point as the MAY-defined/
intersection one it replaced, and gives every (reachable position, register)
pair a work bound that does not depend on control-flow shape (see
"Control-flow reuse" below) - it must not be confused with a MAY/
reachability analysis over *defined* registers, which is a genuinely
different, weaker property this section explicitly rejects above.

**Entry definitions.** For a function whose canonical signature declares `N`
parameters, `ENTRY_DEFS = {r0, r1, ..., r(N-1)}` - exactly the registers
`sm-vm`'s `push_frame` initializes from successfully validated call
arguments (#1770, #1773). No other register is defined at entry merely
because it is numerically in range, because register storage has capacity
for it, or because some call site happens to supply that many arguments;
`Value::Unit` remains an ordinary defined value once genuinely written, not
an "undefined register" sentinel.

**Scope: signature-bearing functions only.** This proof runs only when
`env.signature` is `Some` (a canonical `SIG0` record was decoded - see
[`semcode.md`](semcode.md#callable-signature-sig0)). A function whose header
predates canonical signatures (`signature: None`) has no sound way to
determine `ENTRY_DEFS` - inferring it from caller-supplied `argc`, from
`STORE_VAR` prologue shape, or from any other convention would be an unsound
heuristic (caller `argc` is not even self-consistent across independent call
sites prior to #1773's arity enforcement). This mirrors exactly how the
callable arity check above treats `signature: None`: nothing new to enforce,
unchanged pre-#1756 admission behavior for signature-less artifacts.

**MAP_GET's default register is conservative.** `MAP_GET` reads its default
register lazily at runtime, only on a key miss (#1771). The verifier cannot
soundly prove key presence, so this pass does not attempt a value-sensitive
"only on the miss path" proof: `MAP_GET`'s default register is treated as an
unconditional read, required to be definitely defined regardless of which
runtime path a given key would actually take. Runtime laziness (#1771) and
this static proof (#1756) are deliberately distinct guarantees, not
restatements of each other.

**Control-flow reuse.** This pass reuses the same instruction boundaries and
successor relation the reachable-control-flow check above already computes
(and that same check's reachable-offset result). Missing-register
information is propagated forward *only* along that one successor relation -
a bit-level worklist delivers each (reachable position, register) pair to
that position's successors, at most once ever, with no predecessor
structure built at all. It judges only reachable register reads, matching
the reachable-control-flow check's own reachability boundary; a register
read inside structurally valid but unreachable code is not judged by this
pass, matching the verifier's existing, separate policy on such code.

**Non-entry nodes start at BOTTOM** (no register in the function's register
domain considered missing yet, provisionally), never at the full set. This
is a correctness requirement, not a style preference - the dual of the
MUST-formulation's own non-negotiable rule (non-entry nodes start at TOP,
never empty, under `DEFINED`/intersection): initializing non-entry nodes'
`MISSING` set to the full domain instead computes the wrong converged fixed
point for a register defined once outside a loop and never invalidated
inside it, incorrectly rejecting reads that are actually always defined.

A violation rejects with `VerificationCode::UndefinedRegisterRead` -
deliberately distinct from `InvalidRegisterReference`, which is a numeric
range/quota failure. `UndefinedRegisterRead` means the register number is
in range but at least one reachable incoming execution path has not proven
the register definitely defined before this read - consistent with the
MUST quantification above: a register defined on some paths to a join but
not others is still rejected, since it is not defined on *every* path.

**Resource envelope.** Even fully compressed (see rounds 11-12 of #1756's
own review history: non-branching runs and duplicate-successor artifacts
both collapse to a small leader set), a genuinely branch-dense function
still needs one dense dataflow slot per real decision point - `leader_count
* ceil(register_domain_size / 64)` words of state, and a proportional
amount of propagation work to reach the fixed point. Neither quantity is
bounded by any *other* existing admission check (register-domain size is
capped by the register-budget quota, but `leader_count` - the number of
genuine branch/merge points a function can contain - is not), so this pass
admits only within an explicit `VerificationLimits` envelope; see "Verifier
resource budgets" below.

## Verifier resource budgets

There are three distinct resource domains in this codebase, and they must
never be conflated:

- **Artifact/decode resources** - byte-level limits enforced during
  `sm_format::semcode_decode` (header size, section lengths, and similar),
  reported as `VerificationCode::ResourceLimitExceeded` alongside the
  pre-existing `RuntimeQuotas`-derived checks (register/symbol-table
  budgets) in `verify_function_code`/`verify_semcode_token_with_quotas`.
- **Verifier-analysis resources** - the deterministic working-state and
  work-unit envelope a static analysis pass (currently: definite register
  assignment, #1756 / FA-07-016) needs to reach its proof, expressed as
  `VerificationLimits` (`sm-verify`). Reported as
  `VerificationCode::AnalysisStateLimitExceeded` /
  `AnalysisWorkLimitExceeded`.
- **Runtime-execution resources** - `RuntimeQuotas` (`sm_runtime_core`),
  governing VM execution after admission (`max_steps`, `max_calls`, and so
  on).

**Per-function debug-symbol admission** (#1760, FA-08-002) is a distinct
artifact-admission check inside `verify_function_code`, separate from both
`sm-format`'s own fixed decode-time cap and the `VerificationLimits`
analysis budgets above:

- rule: `env.debug_symbols.len()` (the decoded `DBG0` per-function
  debug-symbol table: pc/line/col) must not exceed
  `RuntimeQuotas::max_debug_symbols_per_function` for the active admission
  profile; violation rejects with
  `VerificationCode::ResourceLimitExceeded`
- profile values: `verified_local = 8192`, `pure_compute = 4096`,
  `kernel_bound = 16384`
- this is independent of `sm_format::semcode_decode::MAX_DEBUG_SYMBOLS_PER_FUNCTION
  = 8192`, a fixed, profile-independent structural cap enforced at decode
  time, before this verifier-layer check ever runs. `verified_local`'s
  8192 exactly matches the decoder cap, and `kernel_bound`'s 16384 cannot
  widen it - the decoder already rejects anything above 8192 for every
  profile - so this check is dead code for both. `pure_compute`'s 4096 is
  strictly tighter than the decoder's 8192, so this check has real,
  distinct admission authority only for `pure_compute`.
- despite living on `RuntimeQuotas` and using
  `VerificationCode::ResourceLimitExceeded` (the same code family as the
  artifact/decode resources above), this field is **not** a `QuotaKind`
  member, is never charged by `sm-vm`, and never surfaces as
  `RuntimeError::QuotaExceeded` - see
  [`quotas.md`](quotas.md#verifier-admission-compatibility-note) for the
  full compatibility note and rationale for why it stays on `RuntimeQuotas`
  rather than moving into `VerificationLimits`

**Scope of the two `VerificationLimits` fields differs, deliberately**
(clarified in round 14 of #1756's own review history, after Codex found
both fields' enforcement did not yet match this intended scope):

- `max_work_units` bounds **one whole artifact verification call**,
  cumulatively across every signature-bearing function it analyzes - not
  a fresh allowance per function. A single counter is created once per
  call and shared, by mutable reference, across every function; the
  moment it is exceeded, verification of that artifact stops immediately
  - no further function is analyzed, and no weaker analysis is attempted
  merely to surface another diagnostic.
- `max_state_words` bounds the **peak LOGICAL verifier-analysis state
  storage simultaneously live for one function's own analysis**, measured
  in canonical 8-byte logical words - not a cumulative total across an
  artifact's functions (functions are verified sequentially and each
  one's analysis state is released before the next function's begins, so
  summing state across functions would overstate what is ever actually
  live at once), and not process RSS or an allocator-exact byte count.
  "Logical" here means the accounting model uses fixed, documented,
  MACHINE-INDEPENDENT sizes for the structures it charges (never a host's
  actual `size_of::<T>()`, which can vary by target) - the same bytes are
  admitted or rejected on every host, deterministically.

  As of round 21 of #1756's own review history, the enforced formula is
  `required_state_words = raw_words + dense_words + fixed_words`, where:

  - `raw_words` (round 16, corrected round 17, extended round 21) is the
    RAW reachable-node representation cost `dataflow_domain_accounting`
    and `compute_leaders` build - scaling with `reachable_count` (a
    `usize`-sized `reachable_indices` entry, a `u32`-sized entry in each
    of the two CSR offset tables, a `bool`-sized `is_leader` entry, and a
    `u8`-sized entry in `compute_leaders`'s own consolidated
    classification-state scratch array (round 21), per reachable node)
    and with `reachable_instruction_bytes` - the TOTAL ENCODED
    BYTES OF REACHABLE INSTRUCTIONS ONLY, deliberately NOT the total
    structural instruction-section length (`instr_len`), which may also
    include structurally valid but unreachable trailing code this
    analysis never decodes (round 17's correction, after round 16's own
    `instr_len`-based proxy overcharged unreachable padding, changing
    this verifier's pre-existing "unreachable code is not judged by this
    pass" admission policy). `reachable_instruction_bytes` bounds `reads_
    flat.len() + writes_flat.len()`, and the transient pre-dedup
    `universe` buffer, via a sound information-theoretic argument: every
    register operand costs exactly 2 bytes to encode, so the reachable
    instructions' own combined byte length bounds how many can ever be
    decoded, independent of how they are distributed across individual
    instructions - see `check_raw_node_state_budget`'s and `reachable_
    instruction_bytes`'s own doc comments for the exact derivation. Both
    `reachable_count` and `reachable_instruction_bytes` are computed BEFORE
    `dataflow_domain_accounting` allocates anything, from already-known
    instruction boundaries (`instr_starts`) and the reachable-offset set
    (`reachable_offsets`) - no CSR table, and no sorted copy of `
    reachable_offsets`, is ever built merely to learn this number - so
    this protection is real, not retroactive. Closes a gap round 15
    explicitly, but incorrectly, excluded: a genuinely branch-FREE
    reachable chain has `leader_count = 1` regardless of length, so
    `dense_words`/`fixed_words` below alone charge almost nothing for it
    no matter how large `reachable_count` grows.
  - `dense_words = (2 * leader_count + 1) * ceil(domain_size / 64)` - the
    dense dataflow lattice payload (round 14: the analysis's `chain_
    killed` and `missing` arrays are simultaneously live, both scaled by
    `leader_count * domain_size`, plus a small constant margin for the
    handful of non-leader-scaled `RegSet`s that can also be transiently
    live).
  - `fixed_words = ceil(fixed_bytes / 8)` for `fixed_bytes = leader_count
    * (per-leader `Vec<RegSet>` header, `chain_targets`, and `leader_
    positions` overhead) + (leader_count + 2 * domain_size) * (worklist
    stack entry size)` - round 15's correction: leader-scaled FIXED
    structural overhead that exists even when `domain_size` is tiny and
    every `RegSet`'s own backing payload is empty, which the dense-only
    formula above does not charge for at all. The worklist stack's own
    peak is `leader_count + 2 * domain_size` entries, proven (not merely
    observed) via a settled-backlog-plus-active-frontier argument.

  `raw_words` and `dense_words + fixed_words` are SUMMED, never maxed:
  the raw-node structures above are never dropped before the leader-state
  phase begins (several are read again by the final violation scan), so
  every structure this envelope charges for is simultaneously live for
  this analysis's entire remaining duration once allocated. See `check_
  raw_node_state_budget`'s and `check_analysis_state_budget`'s own doc
  comments in `sm-verify` for the exact structural inventory, the named
  logical-size constants, and the full derivation.

`VerificationLimits` is never inferred from `RuntimeQuotas::max_steps`,
`max_calls`, or any other execution quota, and is never attached to
`RuntimeQuotas` for API convenience - #1751 established that conflating a
dynamic execution resource with a static verifier bound is unsound, and
that ruling remains authoritative. Exact (non-approximating) static
verification is performed only within the selected, explicitly-bound
`VerificationLimits` envelope; the convenience entry points
(`verify_semcode_token`, `verify_semcode_token_with_quotas`,
`verify_semcode`) admit against `VerificationLimits::default_profile()`,
and `verify_semcode_token_with_quotas_and_limits` binds an explicit
profile instead. Exceeding either bound fails closed, deterministically,
before the corresponding unsafe allocation or work begins - never via a
timeout, an OS/RSS memory observation, or any other non-portable,
machine-dependent signal. Resource exhaustion means the analysis could not
be proven within the selected envelope; it does not mean the program's
semantics are actually invalid, and canonical admission still fails closed
either way - there is no fallback to a weaker analysis, no "assume
defined," and no partial-proof acceptance.

## Verified Execution Rule

The standard `.smc` execution route must require `sm-verify` admission.

Helpers that bypass verification may exist for lower-level testing, but they
must not redefine the public execution contract.

`verify_semcode_token` is the canonical admission boundary for token-first
execution flows. `verify_semcode` remains a compatibility / legacy admission
API and should not be described as the preferred canonical boundary for new
trusted public routes unless a document explicitly distinguishes the two.

## Review Rule

Changes to the verifier require review if they alter:

- what SemCode is considered admissible
- the meaning of an existing verification code
- the structure or deterministic order of reject diagnostics

Required follow-up:

1. update this specification
2. update verifier tests
3. update compatibility or golden tests if public behavior changed
