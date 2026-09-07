# SSF-08 Lane 5 / #1759 — Steps & Calls Bounded-Execution Contract Decision

Status: **CONTRACT DECISION ONLY. NO PRODUCTION BEHAVIOR CHANGE.**
Baseline SHA: `a2543bc3a4f0b37fea3e4208c531cd9c6eb99a1b` (`main`, confirmed
current; `origin/main` unmoved).
Umbrella: SSF-08 umbrella #1579 remains OPEN.

This document freezes the exact semantic contract for `RuntimeQuotas::max_steps`
and `RuntimeQuotas::max_calls` before any counter is implemented. It does not
add a VM counter, restructure `RuntimeError`/`RuntimeTrap`, or touch
`#1760`/`#1761`/`#1762`/`#1763`. #1759 remains OPEN after this document.

## 1. Current architecture (re-inspected fresh, not assumed)

`crates/sm-vm/src/semcode_vm.rs::exec_loop_with_profile` is the **single**
instruction-dispatch loop for the entire VM - one shared `loop { ... }`
spanning every frame of an execution, not a per-frame or per-function loop.
Each iteration: reads `frame_idx = vm.callstack.len() - 1` (the *current
top* frame, whichever function that is), decodes exactly one opcode byte
(`Opcode::from_byte`), records it for profiling, runs
`check_write_execution_site`, then dispatches through one large `match
opcode { ... }`. A decode failure returns `Err` via `?` **before** reaching
any of the code that would charge a quota - it never enters the match body.

`push_frame` is confirmed, by exhaustive grep, to be **the only
`vm.callstack.push` call site in the entire file**. Every root/entry
invocation (`run_verified_entry_semcode*`, `run_verified_function_semcode_with_args_and_config`,
the raw/diagnostic entrypoints, and every test harness) calls it once with
an *empty* callstack; every nested `Opcode::Call`/`Opcode::ClosureCall`
calls it with a *non-empty* callstack. `push_frame`'s current order is:
resolve function → `validate_call_arguments` (signature/arity) →
`enforce_quota(Frames, next_depth)` → `enforce_quota(StackDepth, next_depth)`
(remapped to `RuntimeError::StackOverflow` on failure, per the existing
documented compatibility note) → `enforce_quota(Registers, initial_reg_count)`
→ construct frame and ownership state → `vm.callstack.push(frame)`.

`Opcode::Call`'s own handling resolves the callee three ways: (a) a real
internal Semantic function → `push_frame` then `continue` the loop; (b) a
builtin (`try_eval_builtin_call`) → evaluated **inline, without ever
calling `push_frame`**, `next_pc` advances normally; (c) truly unknown →
routed through `push_frame` anyway so it fails with the existing
`UnknownFunction` error. Effect opcodes (`GateRead`/`GateWrite`/etc.) are
ordinary arms in the same `match`, each calling `bump_effect_calls` (which
itself calls `enforce_quota(EffectCalls, ...)`) as part of their own body -
they do not call `push_frame` and are not routed through it.

**No prior commit, test, or doc anywhere in this repository defines what
"one Step" or "one Call" means** - `docs/spec/quotas.md` and `docs/spec/vm.md`
both state the *existence* of the quota kinds and baseline numbers, never
the charging semantics. This decision is genuinely being made fresh, not
recovered from an existing but undocumented convention.

## 2. Decision A — Step definition

**Frozen: Step = one successfully decoded VM opcode dispatch attempt,
charged before that opcode's own semantic body executes.**

- Charge point: immediately after `Opcode::from_byte` succeeds, before
  `check_write_execution_site` and before the `match opcode { ... }` body -
  i.e. before any semantic effect of that instruction can occur.
- `CALL`/`ClosureCall` consume one Step each, for the `CALL`/`ClosureCall`
  instruction itself, exactly like any other opcode - **YES**.
- Callee opcodes consume their own Steps once execution resumes inside the
  new frame - **YES** (this falls out automatically: the shared loop
  charges every iteration uniformly, regardless of which frame is on top).
- `RET` consumes one Step - **YES** (an ordinary opcode in the same match).
- `JMP` consumes one Step on every visit, including every iteration of a
  backward branch - **YES**. This is the property that makes the quota real
  fuel rather than a statistic: a closed loop cannot iterate without
  spending fuel.
- A runtime-failing opcode (e.g. a trap fires partway through the match
  arm) still counts its Step - **YES, no refund**. The Step was already
  charged before the match began; nothing after that point can un-charge
  it. This matches the existing, unrelated `RuntimeQuotas::exceed`
  convention of never reversing a charge.
- A malformed/undecodable opcode byte does **not** count as a Step - **NO**,
  by construction: `Opcode::from_byte`'s failure returns via `?` before the
  proposed charge point is ever reached. This requires no special-casing;
  it is a structural consequence of where the charge sits.
- Host/effect operations consume a Step **in addition to** `EffectCalls` -
  **YES**. `GateRead`/`GateWrite`/etc. are ordinary opcodes in the same
  dispatch loop; the generic per-instruction Step charge and the
  opcode-specific `EffectCalls` charge are two orthogonal resource domains
  measuring different things (raw instruction fuel vs. a specific
  operation-class budget), exactly the same relationship item 7 asks to be
  frozen for Steps vs. Calls, generalized.

**Falsification attempted (candidate B, "successfully completed opcode"):**
rejected. Charging only on successful completion would mean an opcode that
traps partway through (say, a division that fails) consumes *zero* Steps -
an attacker (or a buggy program) could then loop `DivisionByZero`-triggering
attempts... except that path terminates in an error, not a loop, so the
real counterexample is subtler: a `JMP` whose *target* is out of range would
be rejected at verification, not runtime, so this specific case cannot
arise for verified code - but the general principle (fuel must be spent on
*attempt*, not *success*, or a program that reliably fails a semantic check
without terminating could starve the counter) is the deciding argument, and
it generalizes to any future opcode with a fallible-but-repeatable failure
mode. Candidate A ("opcode fetch attempt," charging even on a decode
failure) was also rejected: it would make quota exhaustion indistinguishable
from - and interleaved with - `BadFormat`, contaminating a resource-budget
signal with what is actually a decode-admission bug category, since decode
failures should never occur for admitted (verified) SemCode in the first
place.

## 3. Step charge timing and error precedence

**Frozen ordering:** decode opcode → charge one Step → if exhausted, return
`RuntimeError::QuotaExceeded(Steps, limit, used)` immediately, before any
opcode-specific check or effect (including `check_write_execution_site` and
the `match` body) → otherwise proceed normally.

**Consequence, explicit:** with `max_steps = N`, exactly `N` opcode
dispatches may proceed. Attempted dispatch `N+1` reports
`QuotaExceeded { kind: Steps, limit: N, used: N+1 }` and performs **no**
opcode-specific mutation or effect for that attempted instruction - this
uses the existing `RuntimeQuotas::exceed`'s own `used > limit` convention
verbatim (`(used > limit).then_some(...)`), not a new one; #1759 does not
change that shared convention.

**Precedence, frozen:**
- Steps exhaustion vs. `BorrowWriteConflict`: **Steps wins.** If there is no
  fuel for this instruction, it does not semantically execute at all, so
  `check_write_execution_site` (which runs after the proposed charge point)
  never gets a chance to fire in the same dispatch.
- Steps exhaustion vs. `DivisionByZero`/other semantic traps: **Steps
  wins**, for the identical reason - the arithmetic never runs.
- Steps exhaustion vs. a host effect: **Steps wins** - `bump_effect_calls`
  and the host call both live inside the match body, downstream of the
  proposed charge point.
- Steps exhaustion vs. Calls quota (on a `CALL`/`ClosureCall` instruction
  specifically): **Steps wins.** The `CALL` opcode's own Step charge fires
  at dispatch time, before the match arm that would eventually call
  `push_frame` (and therefore the Calls charge, §5) ever runs. A `CALL`
  instruction that cannot even be dispatched for lack of Step fuel never
  reaches the point where a Calls charge would apply.

This ordering is the direct, mechanical consequence of the loop structure
in §1 - it is not an arbitrary policy layered on top of it.

## 4. Decision B — Call definition

**Frozen: Call = one admitted, non-root Semantic function invocation - a
`push_frame` call that occurs while `vm.callstack` is already non-empty.**

- `Opcode::Call` targeting a real internal function → counts, via
  `push_frame`.
- `Opcode::ClosureCall` → counts, via `push_frame`.
- Recursive invocation (direct or mutual) → counts identically to any other
  nested call; the counter does not distinguish recursion from ordinary
  nesting.
- **Root/entry invocation does NOT count.** This is not a design choice
  layered on top of `push_frame` - it is a falsification finding from §1:
  `push_frame` is the *only* place any frame is ever constructed, including
  the very first one, so a Calls charge placed unconditionally inside it
  would also charge the entry frame, directly contradicting the intended
  "`max_calls = 0` still runs the entry" semantic (§10). The correct,
  already-available discriminator is the same `next_depth`/
  `vm.callstack.len()` value `push_frame` already computes for the Frames
  quota: charge Calls **only when `vm.callstack.len() > 0` at the moment
  `push_frame` is invoked** (equivalently, `next_depth > 1`) - the root
  invocation is uniquely the one `push_frame` call where the stack was
  empty beforehand, confirmed exhaustively against every call site in §1,
  not merely plausible by construction.
- Host ABI operations and effect opcodes (`GateRead`/`GateWrite`/`PulseEmit`/etc.)
  do **not** count as Calls - confirmed structurally: none of them call
  `push_frame` (§1); they are charged against `EffectCalls` instead, an
  already-separate, already-enforced resource.
- A builtin call resolved via `try_eval_builtin_call` does **not** count -
  confirmed structurally: this path is explicitly `push_frame`-free (§1).
- A failed invocation attempt (signature/arity mismatch, or a Frames/
  StackDepth/Registers quota failure) does **not** count as a Call - see §6
  for the exact ordering and rationale.

**Falsification attempted:**
- *Root-inclusive* (root counts): rejected per the structural finding
  above - it collapses "no nested calls allowed" and "no code at all may
  run" into the same `max_calls = 0` behavior, which contradicts
  `docs/spec/quotas.md`'s own framing of quotas as bounding *execution*,
  not *admission of the selected entry itself* (entry selection is a
  verifier/token concern, per `docs/spec/vm.md`'s "Standard Execution
  Rule," not a runtime quota concern).
- *Attempt-based* (every `push_frame` invocation counts, success or
  failure): rejected - see §6's own falsification for why charging before
  validation would make the Calls counter measure something other than
  "invocations that actually happened."
- *Successful-entry-based, but counting root* was not considered separately
  from root-inclusive, since it is the same claim under a different name.

## 5. Calls vs. Steps are orthogonal

**Frozen:** a `CALL`/`ClosureCall` instruction legitimately charges **both**
domains on a successful nested invocation - one Step (§2, the instruction
itself was dispatched) and one Call (§4, a new admitted invocation was
created). This is not double-counting: Steps measures raw instruction fuel
(would exhaust even on a program with zero calls, via a pure loop); Calls
measures invocation-nesting budget (would exhaust even on a program with a
single, cheap, repeatedly-invoked helper that costs almost no Steps per
call). A program can exhaust one without approaching the other in either
direction, which is the proof that they are measuring genuinely different
physical quantities, not the same one twice.

**Falsification attempted:** could Calls be defined as a subset/count
derived from Steps (e.g. "the number of Step-charging `CALL` opcodes seen
so far") rather than its own independent counter? Rejected - this would
conflate a `CALL` opcode that resolves to a *builtin* (no `push_frame`, no
real invocation, §4) with one that resolves to a real function, since both
charge a Step identically at dispatch time. Only counting at the
`push_frame` choke point correctly excludes builtin/host calls.

## 6. Call admission timing

**Frozen ordering, preserving every existing check's own relative order and
adding Calls as the last gate before the frame is actually constructed:**

```
1. resolve function name (existing - UnknownFunction on failure)
2. validate_call_arguments (existing - signature/arity)
3. enforce_quota(Frames, next_depth)          (existing, unchanged)
4. enforce_quota(StackDepth, next_depth)       (existing, unchanged;
                                                 remapped to StackOverflow)
5. enforce_quota(Registers, initial_reg_count) (existing, unchanged)
6. enforce_quota(Calls, next_calls) IF vm.callstack.len() > 0 before
   this push (new - §4's root-exemption test)
7. construct frame, push onto vm.callstack     (existing, unchanged)
```

**Required final statement: does a failed call attempt consume Calls? NO.**
"Failed" means any of steps 1-5 above rejecting the invocation. Calls is
charged only once every other admission check has already succeeded,
immediately before the frame is actually constructed - meaning the Calls
counter's `used` value always equals the number of frames that *were
actually pushed* as non-root invocations, never an attempt count inflated
by rejected calls for unrelated reasons.

**Falsification attempted (charge Calls first, before Frames/StackDepth/
Registers):** rejected. If Calls were charged before the structural
resource checks, a caller near the Frames/StackDepth ceiling could be
charged a Call for an invocation that never actually produced a frame (it
failed StackOverflow immediately after) - meaning `used` for Calls would
count invocations that structurally never happened, contradicting the
"admitted... invocation" framing in §4's own definition. Charging last
avoids this by construction: every increment to the Calls counter
corresponds 1:1 to a frame that genuinely exists on `vm.callstack` after
the operation completes.

**CALL_ATTEMPT vs. SUCCESSFULLY-ADMITTED-FRAME-ENTRY (item 6's own explicit
question):** the frozen semantic is SUCCESSFULLY-ADMITTED-FRAME-ENTRY,
non-root - not raw attempt count, per the falsification immediately above.

## 7. Counter scope

**Frozen: both counters are execution-wide, not per-frame or per-function.**
`vm.steps`/`vm.calls` (or equivalent field names, left to the implementation
checkpoint) live on the shared `VM` struct itself, alongside the existing
`vm.effect_calls` field this decision explicitly mirrors (`bump_effect_calls`
already reads and writes a single execution-wide counter, not a per-frame
one - the same pattern generalizes directly). Required invariants, all
falling out of "one counter per `VM` instance, incremented in the one shared
dispatch loop / the one shared `push_frame`":
- Entering a callee does not reset Steps - true by construction, since the
  charge point lives in the single shared loop, not inside any per-function
  state.
- Returning from a callee does not reset Calls - true by construction, same
  reasoning applied to `push_frame`.
- Recursion shares the same counters as any other nesting - true, since the
  counter has no per-function identity at all.
- Sibling calls share the same counters - true, for the identical reason.

This matches `docs/spec/quotas.md`'s own framing precisely: the quotas
belong to the *execution envelope* (one `ExecutionConfig`, one `VM`
instance, one call to a `run_verified_*` entrypoint), not to any smaller
unit inside it.

## 8. Trusted vs. raw execution paths

**Frozen: YES - the counters live in the shared VM engine (`exec_loop_with_profile`
and `push_frame`), so every execution path that constructs a `VM` with an
`ExecutionConfig` is mechanically quota-bound, including raw/diagnostic
helpers that intentionally bypass `sm-verify` admission.**

**Why:** `docs/spec/vm.md`'s own "Compatibility Rule" and the existing
precedent for every other quota kind (`Frames`/`StackDepth`/`Registers`/
`EffectCalls` are already enforced identically regardless of which
`run_*`/`run_verified_*` entrypoint constructed the `VM`) already establish
that quota enforcement is a property of the shared engine, not of the trust
route. A raw/test helper deliberately bypassing verifier admission must not
also silently acquire a *second*, weaker fuel model - that would mean a
test harness could "prove" a program terminates under the raw path while
the verified path (the actual public trust contract) enforces different
timing, an inconsistency with no current precedent anywhere else in the
quota system. This does not expand the public trust claim for raw bytecode
execution itself (§9's own scope: `docs/spec/vm.md`'s "Standard Execution
Rule" is unchanged by this decision) - it only means the *fuel* is uniform
across both routes, which is a resource-safety property, not an admission
one.

## 9. Zero-limit semantics

**Frozen, matching the mechanisms in §3 and §6 exactly, not as a separate
special case:**

- `max_steps = 0`: the entry function is selected and its frame is
  constructed (root `push_frame` is unaffected by Steps at all - Steps is
  charged per *instruction dispatch*, not per frame construction), but its
  very first opcode cannot be dispatched: `QuotaExceeded { kind: Steps,
  limit: 0, used: 1 }` fires at the first iteration of the shared loop.
- `max_calls = 0`: the root entry executes normally (root is exempt, §4),
  and its own instructions run under whatever Step budget applies; the
  first nested `Opcode::Call`/`Opcode::ClosureCall` that would produce a
  non-root `push_frame` fails with `QuotaExceeded { kind: Calls, limit: 0,
  used: 1 }` at step 6 of §6's ordering, after the callee's signature and
  the caller's Frames/StackDepth/Registers have already been validated for
  that specific attempted frame (§6 - Calls is charged last, so those
  checks still run and could theoretically fail first for an unrelated
  reason, which is consistent, not a special case).

Both are direct applications of the already-frozen general rules, not
additional policy - this is the conceptual-consistency check item 10 asks
for, and it passes: no new mechanism is invoked for the boundary value 0.

## 10. Counter overflow / `usize::MAX` audit

**Finding (in-scope contract rule, not deferred):** `RuntimeQuotas::exceed`
itself (`(used > limit).then_some(...)`) never overflows - it only compares
two already-computed `usize` values. The risk lives entirely in how a
*caller* computes the incremented `used` value before passing it in - the
existing precedent, `bump_effect_calls`'s `let next = vm.effect_calls + 1;`,
uses a bare `+`, which **wraps silently in a release build** (Rust's
default, non-panicking release behavior for integer overflow) rather than
erroring. A custom `ExecutionConfig` supplying `max_steps = usize::MAX` (or
any caller-controlled quota field near that ceiling) combined with a
sufficiently long-running program would eventually drive the counter's own
`+ 1` increment past `usize::MAX`, wrapping to `0` - at which point `next
(0) > limit (usize::MAX)` is false, and the quota silently stops
functioning as fuel, defeating the entire bounded-termination guarantee
this checkpoint exists to establish. This is a real, currently-latent
pattern already present for `EffectCalls` today, not a new concern this
decision invents, but #1759 must not repeat it for Steps/Calls given that
Steps is specifically the bounded-*termination* promise.

**Frozen rule for the implementation checkpoint (not implemented here):**
Steps/Calls counters must increment via `checked_add` (or equivalent),
returning a deterministic error if the increment itself would overflow -
never a silent wraparound, and never a `wrapping_add`. This mirrors the
exact discipline `docs/spec/semcode.md`'s own "Offset Arithmetic Must Stay
Inside The Result Model" section already mandates for decode-time
cursor/length arithmetic, generalized to runtime counters for the identical
reason: an attacker- or misconfiguration-controlled value must never be
able to wrap a safety-relevant counter back into an apparently-valid range.
This audit does not extend the same requirement to the pre-existing
`EffectCalls` counter's own `+ 1` - that is a `#1760`-or-later-adjacent
finding outside #1759's own scope, recorded here for visibility only.

## 11. Failure taxonomy boundary

**Frozen: #1759 uses `RuntimeError::QuotaExceeded(QuotaExceeded { kind,
limit, used })` exclusively - the existing, already-live, top-level
channel every other quota kind already uses.** #1759 does **not** route
through `RuntimeTrap::QuotaExceeded(QuotaExceeded)`, which is never
constructed today (confirmed fresh in the Lane 5 audit) and whose fate is
explicitly `#1763`'s own, later question. #1759 must not pre-empt that
decision by choosing a channel for it as a side effect.

## 12. Reproduction matrix (evidence design only, not implemented)

**Steps:** a flat, verified, real-source backward loop (a `Stmt::Loop` or
equivalent lowering to a backward `JMP`) with no calls, no effect opcodes,
no register growth, and no frame growth - isolates Steps from every other
quota so only it can be the failing one. Boundary cases: a program whose
natural instruction count is exactly `limit - 1` (must complete normally),
exactly `limit` (must complete normally - `used == limit` is not
exhaustion, only `used > limit` is), `limit + 1` (must fail with
`QuotaExceeded { kind: Steps, limit, used: limit + 1 }`), and `limit = 0`
(must fail on the very first opcode, `used = 1`).

**Calls:** an *iterative* (not recursive) caller that repeatedly invokes a
tiny helper and returns after each call, deliberately avoiding growing
stack depth so that `Frames`/`StackDepth` cannot become the first quota to
fire and mask the Calls result - matching item 13's own explicit warning
that a recursive-calls test is a poor primary proof for this reason.
Boundary cases mirror Steps exactly: `limit - 1`, `limit`, `limit + 1`,
and `limit = 0` invocations, with the same `used > limit` exhaustion
convention and the same expected `QuotaExceeded { kind: Calls, ... }`
shape. A root-entry-only program (zero nested calls) run under
`max_calls = 0` must succeed, proving the root-exemption from §4/§9
empirically, not just by code reading.

## 13. Strongest falsification attempt overall

The single strongest challenge to this whole contract is the one already
surfaced in §1/§4: **`push_frame` is architecturally a single, undifferentiated
choke point for both root and nested invocations**, which could have led to
a naive, convenient-but-wrong implementation charging Calls unconditionally
inside it. This decision survives that challenge only because `push_frame`
already computes `next_depth` (equivalently, `vm.callstack.len()` before the
push) for the Frames quota - a signal that already, for free, distinguishes
"this is the first frame" from "this is a nested one." Had that signal not
already existed for an unrelated reason, root-exemption would have required
either a new boolean parameter threaded through every `push_frame` call site
(a larger, more invasive change than this decision currently implies) or a
weaker, less-precise heuristic. This is recorded explicitly because the
*next* (implementation) checkpoint must not silently reach for a different,
newly-invented signal instead of the one already justified here.

## 14. Rejected alternatives summary

| Question | Rejected alternative | Why rejected |
|---|---|---|
| Step charge point | successful-completion-based | starves the counter on any program that reliably fails a semantic check without terminating (§2) |
| Step charge point | fetch-attempt-based (charges even on decode failure) | conflates quota exhaustion with decode/verification-admission bugs, which cannot occur for genuinely verified code (§2) |
| Call definition | root-inclusive | collapses "no nested calls" and "no execution at all" into one `max_calls = 0` behavior, contradicting quotas.md's execution-bounding framing (§4) |
| Call definition | attempt-based (charge on every `push_frame` call regardless of outcome) | conflates real invocations with ones rejected for unrelated resource reasons (§6) |
| Call charge order | before Frames/StackDepth/Registers | counts invocations that never actually produced a frame (§6) |
| Call charge order | as a derived count from Step-charged `CALL` opcodes | conflates builtin-resolved `CALL`s (no real invocation) with real ones (§5) |
| Counter overflow | silent wraparound (bare `+`, matching existing `EffectCalls` precedent) | defeats the bounded-termination guarantee itself for a caller-controlled near-`usize::MAX` limit (§10) |
| Failure channel | `RuntimeTrap::QuotaExceeded` | pre-empts `#1763`'s own, later, independent taxonomy decision (§11) |

## 15. Effect on Lane 5 / AC4

`#1759` remains **OPEN** after this document - a contract was decided, not
implemented. AC4.a is **NOT SATISFIED**: no counter exists yet, so "every
quota advertised as an active runtime bound has an authoritative resource"
still fails for `Steps`/`Calls`. The next checkpoint is implementation,
gated on a separate, explicit GO, covering: the two counters on `VM`, the
charge points and root-exemption test frozen in §2-§9, `checked_add`
overflow discipline per §10, the `RuntimeError::QuotaExceeded` channel per
§11, and the reproduction matrix in §12 as permanent regression tests.

**Wait for explicit GO before implementation.**
