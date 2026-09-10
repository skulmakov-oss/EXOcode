# SSF-08 Lane 5 — Resource / Quota / Failure-Taxonomy Closure Audit & Execution-Order Reconciliation

Status: **audit only, no repair authorized by this document**
Audit SHA: `01aa1a079a7cd9cac82e8ad33dd557289761f5bd` (exact `main`, confirmed via
`git fetch origin && git checkout main && git pull --ff-only && git rev-parse HEAD`
immediately before this audit; `origin/main` had not moved past this SHA)
Umbrella: SSF-08 umbrella #1579 remains OPEN
Purpose: determine, with fresh current-`main` evidence (not the 2026-08
Phase-A audit's own text taken as fact), what in Lane 5 (#1759-#1763, AC4)
is still genuinely live, what depends on what, and in what order the
remaining work should execute. No production Rust changed while producing
this document.

**Process note (safety invariant for this whole checkpoint and every future
one touching this file):** a prior docs-only commit's message contained the
sentence "still READY TO CLOSE #1579: NO" and GitHub's issue-linking
keyword scanner matched "close #1579" as a literal substring on merge,
auto-closing #1579 despite the negation. #1579 was reopened immediately.
This document and every commit/PR produced for this checkpoint deliberately
avoid the words close/closes/closed/fix/fixes/fixed/resolve/resolves/
resolved immediately followed by `#1579` anywhere in commit messages, PR
titles, or PR bodies.

## 1. Baseline and scope

`#1579`'s current AC status, independently re-confirmed at this SHA (see
`ssf08_closure_audit.md` addendum #3, itself independently re-checked, not
just cited): AC1/AC2/AC3/AC5/AC6/AC7 SATISFIED, AC4 NOT SATISFIED. The
residual candidate set is exactly `#1759`-`#1763`, all `OPEN`, filed under
umbrella `#1617` (a much older, broader, 18-module "platform-wide readiness
self-deception" audit, Phase A only, no repair performed at filing time),
module 08 (`sm-runtime-core`), at a much older audit SHA
(`4ac4d1a169902777e760e23eaee92aa233edc04d`). Every finding below is
re-derived from current `main`, not inherited from that audit's own text.

## 2. Fresh-check summary (all five issues)

| Issue | Title (short) | GitHub state | Comments | Linked PR | Later commits touching claimed authority |
|---|---|---|---|---|---|
| #1759 | `max_steps`/`max_calls` published, never enforced | OPEN | 0 | none | none found (see §5) |
| #1760 | `trace_enabled`/`max_trace_entries` inert | OPEN | 0 | none | none found (see §6) |
| #1761 | `ConstPool` quota inert | OPEN | 0 | none | none found (see §7) |
| #1762 | `ExecutionContext`/quota identity can diverge | OPEN | 0 | none | none found (see §8) |
| #1763 | `RuntimeTrap` mostly unconstructed, parallel `RuntimeError` used | OPEN | 0 | none | none found (see §9) |

`git log 4ac4d1a1..HEAD -- crates/sm-runtime-core/src/lib.rs crates/sm-vm/src/semcode_vm.rs`
shows real, substantial activity in this window (#1725/#1726/#1756/#1773/
#1820/#1821/#1891/#1718 and others), but every one of those commits is
about ownership-path identity/timing, register/symbol-table verifier
quotas, callable signatures, or the #1718 path-family authority - **none**
touch `Steps`/`Calls`/`ConstPool`/`TraceEntries`/`trace_enabled`/
`ExecutionContext` validation/`RuntimeTrap` construction. GitHub state alone
(`OPEN`) is not being used as the sole basis for classification anywhere
below - this is the independent falsification pass required by item 4.

## 3. #1759 — Steps / Calls bounded execution

**Fresh trace.** `crates/sm-runtime-core/src/lib.rs`: `QuotaKind::Steps` and
`QuotaKind::Calls` exist (lines 136-137); every baseline profile
(`verified_local`, `pure_compute`, `kernel_bound`) assigns finite values
(100k/250k steps, 16384/32768 calls). `RuntimeQuotas::exceed` has a match
arm for both. In `crates/sm-vm/src/semcode_vm.rs`, every `enforce_quota`
call site was enumerated directly (not inferred): `QuotaKind::EffectCalls`
(×2), `QuotaKind::Frames`, `QuotaKind::StackDepth`, `QuotaKind::Registers`
(×2). **`QuotaKind::Steps` and `QuotaKind::Calls` appear in zero
`enforce_quota` call sites.** A repository-wide search for any
alternately-named counter (`steps_executed`, `step_count`, `instr_count`,
`instructions_executed`, `call_count`, `calls_executed`, `fuel`) in
`semcode_vm.rs` returned **zero matches**.

**Falsification attempted:** searched for a hidden fuel/step counter under
any other name (§ above, zero hits); searched git history since the audit
SHA for any commit touching `enforce_quota`/`QuotaKind` (§2, zero hits).
The finding is not obsolete.

**Contract questions this would need frozen before implementation** (not
answered here - these are exactly the open design decisions item 5 asks to
surface, not resolve):
- What counts as one step: every opcode fetched, every successfully
  executed opcode, does a failing opcode still count, does `CALL` itself
  count separately from the callee's own instructions?
- What counts as one call: every `CALL`/`ClosureCall` attempt, only
  successful `push_frame` completions, do host calls count, does the entry
  function itself count as call #1?
- Exhaustion timing: does the quota fire when `used == limit`, or when the
  *next* charge would exceed the limit (`RuntimeQuotas::exceed`'s existing
  convention, per `(used > limit).then_some(...)`, fires strictly *after*
  the limit is exceeded, not at equality - any Steps/Calls implementation
  should note this existing convention rather than inventing a new one).

No normative doc (`docs/spec/quotas.md`, `docs/spec/vm.md`) answers these
three questions today; they are genuinely open, not merely undocumented.

**Reproduction (identified, not fixed):** a finite backward-loop program
(e.g. an unconditional `JMP` back to a lower PC with no other quota
tripped first) run through `run_verified_entry_semcode*` on the
`verified_local` profile would execute indefinitely today - `StackDepth`/
`Frames`/`Registers`/`EffectCalls` never fire for a flat backward jump with
no calls, no growing register set, and no effect opcodes. A repeated-call
reproduction (a function that calls itself or a helper `max_calls + 1`
times without growing stack depth, e.g. via tail position or an iterative
driver) would likewise never trip `Calls`, though it may incidentally trip
`Frames` first depending on call shape - the two reproductions are
independent and should both be constructed explicitly in the eventual
implementation checkpoint's own test matrix, not assumed interchangeable.

**Disposition: REQUIRED — IMPLEMENT.**
**Why:** the published contract (`docs/spec/quotas.md`: "Runtime quotas
define the bounded execution contract") is a bounded-*termination* promise
for `max_steps` specifically - a verified, otherwise-admissible program can
run forever today, which is the strongest possible violation among the five
findings (P1, matches its own filed severity).
**Authority:** `docs/spec/quotas.md` (self), `stable_foundation_target_contract.md`
line 96 ("Execution remains verifier-first, deterministic, quota-bounded").
**Dependencies:** none on the other four findings; #1763 depends on this
one's outcome (see §10), not the reverse.
**Smallest valid next checkpoint:** freeze the three contract questions
above, then implement `Steps`/`Calls` charging at the two natural choke
points (`push_frame` for calls - already the confirmed universal choke
point per `docs/spec/vm.md`'s "Callable Runtime Family Enforcement" section;
the instruction-dispatch loop for steps), with both a finite-backward-loop
and a repeated-call regression test.

**Decision update (contract frozen, no implementation):** the three
contract questions above have been frozen in
`docs/roadmap/stable_foundation/ssf08_1759_steps_calls_contract_decision.md` -
Step = one decoded-opcode dispatch attempt, charged before semantic
execution; Call = one admitted, non-root `push_frame` invocation (charged
last, after signature/Frames/StackDepth/Registers admission, using the
already-existing `vm.callstack.len() > 0` signal `push_frame` computes for
its own Frames check to exempt the root/entry frame); exhaustion keeps the
existing `used > limit` convention unchanged; overflow at the `usize::MAX`
ceiling is fully frozen as a `checked_add`-based fail-closed rule with
saturated-`usize::MAX` reporting on the single unrepresentable case, no new
error channel; failure channel is `RuntimeError::QuotaExceeded`, explicitly
not `RuntimeTrap::QuotaExceeded` (that remains #1763's own, independent
question - this decision does not resolve it). This same pass also
surfaced a new residual finding outside #1759's own scope - `EffectCalls`
shares the pre-existing unchecked-overflow defect this decision closes for
`Steps`/`Calls` - recorded in §8, finding 4. **This is a contract decision
only. #1759 remains OPEN. No counter, no `enforce_quota` call site, and no
test were added by this update - AC4.a remains not satisfied for `Steps`/
`Calls` until a separately
authorized implementation checkpoint lands.**

**Implementation update:** the frozen contract above has since been
implemented - `VM.steps`/`VM.calls` execution-wide counters, the `Steps`
charge in `exec_loop_with_profile`, the `Calls` charge (root-exempt) in
`push_frame`, and the shared `checked_add`-based `charge_counter` primitive
are now live on `main`, backed by direct helper-level `usize::MAX` tests
and a full end-to-end VM regression suite (zero-limit, exact-limit,
one-under, backward-loop, root-exemption, `ClosureCall`, builtin-call and
effect-opcode exclusion, Steps-before-Calls precedence, and
Frames-rejection non-accounting), each verified by a mutation proof.
`Steps`/`Calls` move from **INERT** to **ACTIVE** in §10's inventory. **This
does not satisfy AC4.a globally**: the `EffectCalls` overflow residual
(§8 finding 4 / #1900) and the remaining Lane 5 findings (#1760-#1763)
are unaffected and unresolved by this checkpoint. Activating real
enforcement also surfaced one genuine, previously-invisible quota
violation - `examples/benchmarks/snake_learning.sm` exceeds the published
`max_steps = 100000` `VerifiedLocal` baseline by one opcode (§8 finding 5) -
left as a documented, `#[ignore]`d known-issue rather than fixed as part of
this checkpoint.

## 4. #1760 — `trace_enabled` / `max_trace_entries`

**Fresh trace.** Repository-wide (`grep -rn "trace_enabled"`, all `.rs`
files, `target/` excluded): exactly two matches, both inside
`ExecutionConfig`'s own definition in `crates/sm-runtime-core/src/lib.rs`
(the field declaration and its `false` default in `ExecutionConfig::new`).
**Zero production reads anywhere in the repository.** `QuotaKind::TraceEntries`/
`max_trace_entries`: defined and given baseline values (4096/8192/16384
across the three profiles); zero `enforce_quota` call sites (confirmed in
the same enumeration as §3).

**Distinguishing profiler/diagnostic/audit/trace** (required by item 6):
this repository has a real, active, separate execution-trace-adjacent
surface - the "7hell" diagnostics layer (`src/bin/smc.rs`,
`SevenHellDiagnostic`) and `vm_opcode_profile`/`vm_opcode_profile_workloads`
test harnesses (`crates/sm-vm/tests/`) - but neither reads
`ExecutionConfig::trace_enabled` or charges `QuotaKind::TraceEntries`; they
are independent, unrelated mechanisms (diagnostic reporting after a
failure, and opcode-frequency profiling for benchmarking, not a bounded
runtime trace-entry resource gated by this specific config field). This
confirms `trace_enabled`/`TraceEntries` is not merely undocumented - it is
architecturally disconnected from every trace-like mechanism that does
exist.

**Falsification attempted:** searched for any later trace resource under a
renamed type or field (zero hits beyond the two definitional occurrences);
searched git history (§2, zero hits).

**Disposition: NEW DECISION REQUIRED BEFORE REPAIR** (not a default
"implement it" - per item 6's explicit instruction not to build a feature
merely because a field exists).
**Why:** there is no evidence in `stable_foundation_target_contract.md` or
any current-facing doc that a bounded, quota-governed execution trace is
part of Semantic's Stable Foundation promise at all - unlike `#1759`
(`max_steps`/`max_calls`), which is directly required by the already-stated
"quota-bounded execution" contour, no equivalent statement commits Stable
Foundation to a trace *resource*. The two live candidates are genuinely
open:
- **IMPLEMENT**: give `trace_enabled`/`max_trace_entries` a real production
  consumer (own a trace buffer, charge per recorded entry, reject on
  exhaustion) - only justified if Stable Foundation actually wants a
  bounded execution trace as a runtime-visible resource, which is not
  established today.
- **NARROW/REMOVE**: `trace_enabled` and the `TraceEntries` quota kind are
  currently inert public API surface with baseline numbers that look like
  policy but govern nothing; the honest repair may be to remove them from
  the *active* Stable Foundation runtime-quota contract (deprecate/delete
  the field and quota kind, or explicitly re-label them "reserved,
  unenforced" in `docs/spec/quotas.md`) rather than build a mechanism
  solely to retroactively justify an existing field.
**Authority:** no existing doc commits either way - this is exactly why it
is a new decision, not an implementation default.
**Dependencies:** informs #1763 (§10) - if `TraceEntries` is removed from
`QuotaKind`, there is one fewer quota-exhaustion family for the taxonomy
question to account for.
**Smallest valid next checkpoint:** a short, dedicated decision checkpoint
(no implementation) choosing IMPLEMENT vs NARROW/REMOVE for this one field
pair, citing Stable Foundation's actual execution-trace requirements (or
lack thereof).

**Decision update (contract frozen, no implementation):** the IMPLEMENT vs
NARROW/REMOVE question above has been decided in
`docs/roadmap/stable_foundation/ssf08_1760_trace_contract_decision.md` -
**SPLIT**, not a uniform disposition across all three symbols. Fresh
re-derivation confirms `trace_enabled` and `QuotaKind::TraceEntries` are
exactly as inert as `ConstPool` was, with the same REMOVE disposition. But
this pass corrects a gap in this section's own original evidence above:
"zero `enforce_quota` call sites" is true for `TraceEntries`, but
`max_trace_entries`'s raw numeric value has a **real, live, non-`enforce_quota`
reader** in `crates/sm-verify/src/lib.rs::verify_function_code`, which
compares a decoded function's `debug_symbols.len()` against
`quotas.max_trace_entries` as a per-function structural cap - dead code
for `verified_local`/`kernel_bound` (redundant with `sm-format`'s own
fixed `MAX_DEBUG_SYMBOLS_PER_FUNCTION = 8192`), but genuinely live for
`pure_compute` (`max_trace_entries = 4096`, stricter than 8192). This
field is therefore not inert vocabulary the way `ConstPool` was; it is a
real, mislabeled, profile-inconsistent debug-symbol bound. The decision
document freezes, with the exact rename mechanic itself now decided (not
left open): `trace_enabled` → REMOVE; `QuotaKind::TraceEntries` (the
execution-trace taxonomy member) → REMOVE; `RuntimeQuotas::max_trace_entries`
→ RENAME to `RuntimeQuotas::max_debug_symbols_per_function`, preserving
its exact current values (8192/4096/16384) and the exact existing
`sm-verify` check byte-for-byte in behavior - explicitly **not** an
execution-trace quota, not represented in `QuotaKind`, not charged by
`sm-vm`, never `RuntimeError::QuotaExceeded`. The alternative (delete the
check outright, relying solely on `sm-format`'s fixed
`MAX_DEBUG_SYMBOLS_PER_FUNCTION = 8192`) is **explicitly rejected** by the
decision document, not merely deferred - it would loosen `pure_compute`'s
admission from 4096 to 8192 debug symbols per function, an undisclosed
admission-policy change this checkpoint has no authority to make. **This
is a contract decision only. `#1760` remains OPEN. No field, variant,
baseline value, or golden snapshot was touched by this update - AC4.b
remains not satisfied for `TraceEntries`/`trace_enabled` until a
separately authorized implementation checkpoint executes this
now-complete mechanic.**

**Implementation update (SPLIT disposition implemented; issue closure is
merge-gated):** `ExecutionConfig::trace_enabled` and
`QuotaKind::TraceEntries` are removed; `RuntimeQuotas::max_trace_entries`
is renamed to `RuntimeQuotas::max_debug_symbols_per_function` with its
three profile values (8192/4096/16384) and its `sm-verify` admission
check preserved exactly, only the diagnostic's "trace budget" wording
corrected. A repository-wide residue audit found zero remaining active
references to the retired names in `crates/**`, `docs/spec/**`, or the
public API golden snapshot. The public API guard was proven RED (drift
isolated to exactly these three surfaces) before being regenerated GREEN.
Two adversarial mutations (loosening `pure_compute`'s configured value to
8192; bypassing the configured value with a hardcoded 8192 comparison)
each turned a dedicated 4097-debug-symbol/`pure_compute` regression RED
and were fully reverted, proving the rename did not silently widen
`pure_compute`'s admission. Full detail:
`docs/roadmap/stable_foundation/ssf08_1760_trace_contract_decision.md`
§14.

## 5. #1761 — `ConstPool` quota

**Fresh trace.** `QuotaKind::ConstPool`/`max_const_pool`: defined,
`65_536` in every baseline profile. Repository-wide search: only appears in
`RuntimeQuotas`'s own field/constructor/`exceed`-match-arm definitions
(`crates/sm-runtime-core/src/lib.rs`). Zero `enforce_quota` call sites.

**Does a real "constant pool" resource exist under another name?**
Inspected: instruction immediates (inline operands, not pooled), per-function
string tables (`sm-format`'s function-local string table, capped by
`MAX_STRINGS_PER_FUNCTION`/`MAX_STRING_LEN` in `sm-format`, a *decode-time
structural* bound, not a `RuntimeQuotas`-governed one), the program-wide
`RuntimeSymbolTable` (governed by `max_symbol_table`/`QuotaKind::SymbolTable`,
a *different*, already-enforced quota kind - see §12), the decoded `SIG0`
signature table (per-function, capped by
`MAX_SIGNATURE_PARAMETERS_PER_FUNCTION` in `sm-format`, again a decode-time
structural bound). **None of these is charged against, or even conceptually
named, `max_const_pool`/`QuotaKind::ConstPool`.** There is no shared,
runtime-resident "constant pool" object in this codebase that this quota
kind could describe.

**Falsification attempted:** searched for a real constant-pool
representation under any other name (checked every candidate value-storage
mechanism above; none matches); searched git history (§2, zero hits).

**Disposition: REQUIRED — CONTRACT NARROWING.** The narrowing *direction*
is frozen by this audit - unlike `#1760`, this is not left as an open
implement-or-remove question. Only the exact *mechanism* remains a short
decision: REMOVE `max_const_pool`/`QuotaKind::ConstPool` outright, or
explicitly re-scope the field under a newly-authorized contract meaning
(which `docs/spec/quotas.md`'s own "Version Review Rule" requires treating
as a genuine version-reviewed decision, not a free relabeling).
**Why:** unlike `#1760`'s trace surface (where a bounded-trace *concept*
could plausibly belong in Stable Foundation even if unimplemented),
`ConstPool` describes a resource that has **no current architectural
referent at all** - SemCode has no pooled-constant representation; values
are either inline operands or per-function string-table entries governed
by *different*, already-enforced bounds. This is a stronger case than
#1760's: there is nothing to "finish implementing," because the resource
this quota kind names does not exist as a distinct concept in the current
design - the only open question is how the narrowing is executed, not
whether it happens.
**Authority:** none - no current-facing doc claims a runtime constant pool
exists as a named resource; `docs/spec/quotas.md` merely lists it alongside
genuinely-enforced kinds, which is the overstatement itself.
**Dependencies:** informs #1763 (§10), same as #1760.
**Smallest valid next checkpoint:** decide whether `max_const_pool`/
`QuotaKind::ConstPool` is removed outright or re-scoped to name an actual
existing resource (e.g. redefining it to mean the per-function string table
would be a *version-reviewed* meaning change per `docs/spec/quotas.md`'s
own "Version Review Rule," not a free relabeling) - this is a genuine
contract decision, not a mechanical deletion, so it should not be done as a
side effect of another checkpoint.

**Decision update (contract frozen, no implementation):** the mechanism
question above has been decided in
`docs/roadmap/stable_foundation/ssf08_1761_constpool_contract_decision.md`
- **REMOVE**, falsified against RE-SCOPE (every candidate referent either
duplicates an existing, correctly-named quota kind or is not a resource at
all) and against a RESERVED/deprecated compatibility surface (no
compatibility authority requires one, and the codebase's own dominant
struct-literal-with-`..Default` construction pattern would make a
`#[deprecated]` attribute silently ineffective at most call sites). Fresh
re-derivation confirms no real ConstPool-shaped resource exists anywhere
in the current codebase - `RuntimeSymbolTable`/`QuotaKind::SymbolTable`
already owns the one genuine pooling mechanism in this design (string
interning), and it is not this quota kind under another name.
Removal is confirmed source-visible (the golden snapshot
`tests/golden_snapshots/public_api/sm_runtime_core_lib.txt` locks in both
`ConstPool` and `max_const_pool`); the decision document also establishes
that this surface carries no stable label or binding deprecation
commitment found by any authority - it is unclassified, not proven stable
- and that `compatibility_policy_stack.md`'s own "boundary/runtime
contract changes" review trigger is satisfied by this decision checkpoint
itself acting as that explicit review. **This is a contract decision only.
`#1761`
remains OPEN. No field, variant, baseline value, or golden snapshot was
touched by this update - AC4.b remains not satisfied for `ConstPool` until
a separately authorized implementation checkpoint lands.**

**Implementation update: REMOVE implemented.** `QuotaKind::ConstPool` and
`RuntimeQuotas::max_const_pool` are removed from
`crates/sm-runtime-core/src/lib.rs` (enum variant, struct field, all three
baseline-profile assignments, the `exceed()` match arm), from
`tests/golden_snapshots/public_api/sm_runtime_core_lib.txt` (via the
explicit `SM_UPDATE_PUBLIC_API_SNAPSHOTS=1` mechanism, reviewed manually -
exactly those two lines, nothing else), and from `docs/spec/quotas.md`'s
active taxonomy, descriptor-field list, and all three baseline-profile
sections. `cargo check --workspace --all-targets` immediately after the
Rust removal produced zero compile errors anywhere - empirical
confirmation, not inference, that `ConstPool` had no live downstream
consumer. `ConstPool` is no longer a current `RuntimeQuotas`/`QuotaKind`
member; §10's inventory below reflects this as a resolved, historical
entry rather than a live row. **`#1761` remains OPEN pending closure by
its own implementation PR** (this repository update does not itself close
issues). **AC4.b is no longer blocked by `ConstPool`** - it remains **NOT
SATISFIED globally**, since `TraceEntries`/`#1760` still presents an
inert/deferred resource as an enforced quota kind.

## 6. #1762 — `ExecutionContext` / quota identity

**Fresh trace.** `ExecutionConfig::new(context, quotas)` (line 257) performs
no validation that `quotas` matches the baseline `ExecutionContext::for_context(context)`
would have produced - confirmed by direct reading, not inference.
`crates/sm-vm/src/semcode_vm.rs` was searched for any `.context`/`config.context`
read: **none found** - the VM enforces `vm.config.quotas` exclusively (every
`enforce_quota` call site in §3 reads `&vm.config.quotas` or `quotas`
parameters derived from it, never `config.context`). `crates/prom-runtime/src/lib.rs`:
`RuntimeSessionDescriptor` (line 18) has exactly three fields - `context`,
`capability_manifest`, `gate_registry_bound` - **no `quotas` field at all**;
every construction site (lines 99, 242, 407) copies `context: config.context`
verbatim, never the quota envelope. `crates/prom-audit/src/lib.rs`:
`AuditSessionMetadata` (line 20) likewise has `context: ExecutionContext`
and no quota field. `tests/ssf04_effect_quota.rs` confirms
`ExecutionConfig::new(ExecutionContext::VerifiedLocal, <custom quotas>)` is
an actively-used, legitimate test pattern today - not a hypothetical misuse
this audit invented.

**Falsification attempted:** searched for later validation logic in
`ExecutionConfig::new`/`for_context` (none added since the audit SHA per
§2's git-log check) and for any full-quota-envelope provenance recording in
`prom-runtime`/`prom-audit` (none found - both structs are unchanged in
shape from what the finding describes).

**What does `ExecutionContext` normatively mean today?** Per fresh
evidence, it is closest to **(C) a descriptive execution class only** -
`for_context()` provides a *default* baseline mapping, but nothing in the
type system, the VM, or the audit/provenance layer treats `context` as
proof of which quotas actually governed execution, and a legitimate,
tested seam (`ExecutionConfig::new`) exists specifically to decouple them.
It is not (A) exact canonical identity (nothing enforces that), and
calling it (B) "a baseline that may be tightened" would itself be a new,
undecided policy statement, not a description of current behavior.

**Disposition: NEW DECISION REQUIRED BEFORE REPAIR.**
**Why:** this is a provenance/identity question, not a `used > limit`
enforcement gap - the three candidate repairs listed in the governing brief
(VALIDATE / RECORD / NARROW CLAIM) each imply a different, real policy
choice:
- VALIDATE would break the currently-legitimate `ssf04_effect_quota.rs`
  custom-quota pattern unless it is explicitly re-scoped as an
  intentionally-privileged internal test seam, not a general public API
  guarantee.
- RECORD (add `quotas: RuntimeQuotas` to `RuntimeSessionDescriptor`/
  `AuditSessionMetadata`) preserves the existing flexibility but makes
  provenance honest - this is the repair that most directly addresses the
  filed gap ("audit records only the context label") without touching
  execution semantics at all.
- NARROW CLAIM (stop claiming `ExecutionContext` selects/proves the actual
  quota envelope, document it as a descriptive label only) requires
  updating `docs/spec/quotas.md`'s "context selects the runtime quota
  baseline" language, which currently overstates the guarantee.
**Authority:** `docs/spec/quotas.md`'s own "context selection is explicit"
and "must not silently weaken the core safety contract" language is in
tension with the current, legitimate custom-quota test seam - the tension
itself is the thing to resolve, not a bug to patch mechanically.
**Dependency on #1759 (explicit, as the brief requests):** confirmed real -
if `Steps`/`Calls` remain inert, then a `RuntimeSessionDescriptor`/
`AuditSessionMetadata` recording (say) `KernelBound`'s `max_steps = 250_000`
would misleadingly imply that figure actually governed the run, when in
current reality it governs nothing. **Any RECORD-style repair for #1762
should land only after #1759's own disposition is settled**, or it risks
recording quota values that are honest about *configuration* but dishonest
by omission about *enforcement*.
**Smallest valid next checkpoint:** a decision checkpoint (no
implementation) choosing among VALIDATE/RECORD/NARROW CLAIM/a documented
combination, informed by #1759's outcome.

**Decision update (contract frozen, no implementation):** `#1759` (and
`#1900`/`#1761`/`#1760`, all closed since this section was first written)
have settled the dependency this section itself named as a precondition -
`RuntimeQuotas` is now a fully truthful, fully enforced quota/profile
authority (scoped exactly to the dimensions it represents - `sm-vm`
runtime-quota charging plus the `sm-verify` checks that explicitly consume
it; it does not subsume `VerificationLimits`, `sm-format`'s structural
caps, unrelated verifier rules, or capability policy - see the decision
document's own §5), not aspirational vocabulary, so a RECORD-style repair
can proceed without risk of recording configuration that governs nothing.
The VALIDATE/RECORD/NARROW-CLAIM question is now decided in
`docs/roadmap/stable_foundation/ssf08_1762_execution_envelope_provenance_decision.md`:
**RECORD_EFFECTIVE** (the RECORD option named above), not VALIDATE and not
NARROW CLAIM. `ExecutionContext` is frozen as a baseline-selector/
audit-class label (confirming this section's own "(C) descriptive
execution class only" reading); `RuntimeQuotas` is frozen as the
effective authority for that same, narrower quota/profile scope; the
existing `ssf04_effect_quota.rs` custom-envelope seam remains fully
legitimate and unrestricted, per the decision's own
falsification of a "stricter than baseline only" restriction as
unauthorized, ill-posed new scope. `docs/spec/quotas.md`'s "must not
weaken the core safety contract silently" is frozen to mean: custom
envelopes are permitted, but must be explicitly visible in provenance -
never hidden. The exact future mechanic (fields, copy path, archive wire
format, version bump discipline, public API impact) is fully frozen in
that document's own §15-§24 - a fully-specified mechanic, not an open
choice. **This is a contract decision only. `#1762` remains OPEN. No
field, struct, archive format constant, or golden snapshot was touched by
this update - AC4.c remains not satisfied until a separately authorized
implementation checkpoint executes this now-complete mechanic.**

**Implementation update (RECORD_EFFECTIVE implemented by PR #1911; issue
closure is merge-gated):** `RuntimeSessionDescriptor` and
`AuditSessionMetadata` both gained `quotas: RuntimeQuotas`, copied
one-way from the same `ExecutionConfig` used for execution (never
re-derived from `context`); `AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION` bumped
`1` → `2` exactly as frozen, `MULTI_SESSION_REPLAY_ARCHIVE_FORMAT_VERSION`
left at `1`. A distinct custom envelope survived the full canonical
pipeline value-for-value, and two sessions sharing one `ExecutionContext`
with different `RuntimeQuotas` were proven distinguishable after a
multi-session round-trip - the central #1762 trust invariant, proven not
merely asserted.

In `crates/smc-cli/src/app.rs`, `collect_controlled_observation_envelope`
now constructs exactly **one** `ExecutionConfig` binding and threads it
through every consumer: its `quotas` feed quota-aware verifier admission
(`verify_semcode_token_with_quotas`), the same binding feeds a new
config-aware VM observation-execution helper
(`run_semcode_collecting_hello_observations_with_config`, added to
`sm-vm` for this purpose - the legacy no-config helper now delegates to
it with the canonical default), and the same binding populates the audit
metadata. A first attempt at this site constructed a second,
independently-hardcoded canonical config purely for provenance while
execution still hardcoded its own separate config internally - itself an
instance of the reconstruction pattern this checkpoint exists to
eliminate - caught in review and corrected before merge.

Four mutation proofs (context-derived descriptor reconstruction, parser
context-derived reconstruction, missed version bump, and - added for the
single-authority CLI fix - the config-aware VM helper silently ignoring
its caller-supplied config and falling back to canonical `VerifiedLocal`
internally) each turned RED and were reverted. A pre-existing, unrelated
gap was discovered and resolved with explicit owner input: `prom-audit`
was missing from the public-API golden-snapshot guard's tracked file list
entirely, leaving `AuditSessionMetadata`'s own public API unguarded; the
owner chose to restore it to the tracked list within this same PR, which
surfaced (and required reviewing) unrelated accumulated drift alongside
the two intended changes. Every production construction site of both
structs was manually classified; zero discard an available effective
envelope in favor of a context-derived reconstruction. Full detail:
`docs/roadmap/stable_foundation/ssf08_1762_execution_envelope_provenance_decision.md`
§25.

## 7. #1763 — `RuntimeTrap` / `RuntimeError` taxonomy

**Fresh trace, exhaustive, not sampled.** `RuntimeTrap` (13 variants):
`AssertionFailed`, `BorrowWriteConflict`, `StackOverflow`, `StackUnderflow`,
`TypeMismatch`, `InvalidOpcode`, `InvalidJump`, `DivisionByZero`,
`ArithmeticOverflow`, `CapabilityDenied`, `AbiViolation`, `VerifierRejected`,
`QuotaExceeded(QuotaExceeded)`. A repository-wide search for
`RuntimeTrap::<Variant>` construction in `crates/sm-vm/src/semcode_vm.rs`
found constructions for **exactly four**: `AssertionFailed`,
`BorrowWriteConflict`, `DivisionByZero`, `ArithmeticOverflow` - all reached
only via `RuntimeError::Trap(RuntimeTrap::X)`. The remaining nine
(`StackOverflow`, `StackUnderflow`, `TypeMismatch`, `InvalidOpcode`,
`InvalidJump`, `CapabilityDenied`, `AbiViolation`, `VerifierRejected`,
`QuotaExceeded`) have **zero construction sites anywhere in production
code** - confirmed by direct grep, not sampling. Each of these nine has a
same-named or same-shaped **top-level** `RuntimeError` variant that *is*
constructed instead: `RuntimeError::StackOverflow`, `::StackUnderflow`,
`::TypeMismatchRuntime`, `::InvalidJumpAddress` (`InvalidJump`'s
counterpart), `::CapabilityDenied`, `::HostAbi` (`AbiViolation`'s
counterpart), `::VerifierRejected`, `::QuotaExceeded` - confirmed
constructed at real production call sites (`enforce_quota`, capability
checks, `verify_semcode_token` wrapping, etc.). `InvalidOpcode` has no
directly-named top-level counterpart but the equivalent failure surfaces as
`RuntimeError::BadFormat`/`UnknownFunction`-family errors at load time,
before a trap would even apply.

`src/bin/smc.rs::vm_trap_message_needle` is a Rust-exhaustive match over
all 13 `RuntimeTrap` variants (required by the compiler, since the enum has
no wildcard arm) - this function's mere existence is exactly the kind of
"exhaustive text mapping reinforcing the appearance that every variant is
part of one authoritative channel" the issue itself names as a contributing
cause, confirmed still true and still present verbatim. Separately, the CLI
diagnostic dispatch at `src/bin/smc.rs` (~line 1400-1427) already correctly
routes only the four real `RuntimeError::Trap(_)` cases through this
function, and every other `RuntimeError` variant (including the nine
"orphaned" `RuntimeTrap` names' actual top-level counterparts) through a
separate `vm_error_code`/`VmError` path - **the CLI's own behavior already
discriminates correctly; only the documentation and the enum shape
overstate a unified channel.**

**A new, current-SHA-specific finding beyond the original filing:**
`docs/spec/vm.md`'s own "Trap And Error Model" section (its normative,
current-facing list of "public runtime error families") **omits
`AssertionFailed`, `DivisionByZero`, and `ArithmeticOverflow` entirely** -
three families that *are* real, tested, production-constructed failure
paths (reachable via `RuntimeError::Trap(RuntimeTrap::X)`). This means the
documentation drift found here is not one-directional: `vm.md` both fails
to mention real, live failure families and (via `trap_taxonomy.md`'s
separate, cross-referenced evidence table) cites `RuntimeTrap::StackOverflow`
as if it were the constructed value for the "Stack overflow" frozen trap
class, when the actual production evidence is the top-level
`RuntimeError::StackOverflow` variant, never the `RuntimeTrap` one.

**Cross-reference to an independent, already-frozen authority
(`docs/roadmap/language_maturity/core_trust_freeze/trap_taxonomy.md`,
owned by a *different* governance track, CTF-2/PCC, not SSF-08):** this
document's own "frozen trap classes" table (§3) already cites, for its
"Quota exceeded" row, evidence of `crates/sm-runtime-core/src/lib.rs::QuotaExceeded`
and `::RuntimeQuotas` directly - **not** `RuntimeTrap::QuotaExceeded` -
which is the historically correct citation (this is exactly the top-level
variant that is actually constructed). Its "Stack overflow" row, by
contrast, cites `RuntimeTrap::StackOverflow` as evidence, which - per the
fresh grep above - is imprecise: the constructed value is the top-level
`RuntimeError::StackOverflow`. This means CTF-2's own frozen taxonomy
already, implicitly, treats "one authoritative failure vocabulary spanning
both `RuntimeError`'s top-level variants and `RuntimeTrap`'s four live
variants" as the real contract - it is only imprecise in one evidence
citation, not wrong about the underlying design. **This is significant for
disposition**: it suggests SSF-08's own repair should not consolidate
everything into `RuntimeTrap` (that would contradict an already-frozen,
independently-owned CTF-2 authority that already accepts top-level
`RuntimeError` variants as legitimate evidence for several frozen classes),
but should instead **narrow `RuntimeTrap` to the four variants it actually
has*, or explicitly document the split as intentional*.

**Falsification attempted:** searched for later taxonomy consolidation
work (§2's git log, zero hits) and for any changed normative doc language
narrowing or reconciling the split (none found - `vm.md`'s list is
unchanged in shape from what the issue describes, and still has its own,
newly-identified omissions above).

**Disposition: REQUIRED — CONTRACT NARROWING, SPLITTABLE.** The
non-quota-related eight variants (`StackOverflow`, `StackUnderflow`,
`TypeMismatch`, `InvalidOpcode`, `InvalidJump`, `CapabilityDenied`,
`AbiViolation`, `VerifierRejected`) and the `vm.md` omission fix are
independent of every other Lane 5 finding and may proceed on their own.
Only `RuntimeTrap::QuotaExceeded(QuotaExceeded)`'s specific fate is blocked
- and only on `#1760`/`#1761` (see §10 and the correction below), not on
`#1759`.
**Why not SPLIT REPAIR (mass conversion) or IMPLEMENT (construct the
missing nine):** constructing the nine orphaned `RuntimeTrap` variants
would create genuine duplicate failure channels for conditions already
correctly handled by top-level `RuntimeError` variants, and would
contradict CTF-2's own already-frozen, evidence-cited taxonomy for several
of those classes. The evidence points toward `RuntimeTrap` being narrowed
to its four genuinely-live variants (`AssertionFailed`, `BorrowWriteConflict`,
`DivisionByZero`, `ArithmeticOverflow` - the semantic-execution-trap
subset), with the other nine variants either removed as dead code or
explicitly redocumented as intentionally-unused/reserved, and `vm.md`
corrected to list the real, complete current failure vocabulary (including
the three currently-omitted live trap families).
**Authority:** `trap_taxonomy.md` (independent, CTF-2-owned, already
frozen, cross-checked above) plus the fresh code evidence in this section.
**Dependencies:** `RuntimeTrap::QuotaExceeded(QuotaExceeded)`'s own
disposition cannot be finalized until `#1760`/`#1761` settle which
`QuotaKind` variants remain part of the active contract - narrowing
`QuotaKind` first, then revisiting whether `RuntimeTrap::QuotaExceeded`
should exist at all (given the top-level `RuntimeError::QuotaExceeded`
already serves this role in production), avoids repairing the taxonomy
twice. **`#1759` is explicitly not part of this dependency**: it decides
whether `Steps`/`Calls` charging becomes *real*, not whether `QuotaKind`'s
own variant set changes shape - `Steps` and `Calls` remain declared
`QuotaKind` variants either way, and the failure channel they would use on
exhaustion (`RuntimeError::QuotaExceeded`, already live) is unaffected by
`#1759`'s own disposition.
**Smallest valid next checkpoint:** last in Lane 5's own execution order
(see §10) - a decision-plus-doc checkpoint correcting `vm.md`'s omissions,
freezing which `RuntimeTrap` variants remain, and reconciling with
`trap_taxonomy.md`'s own evidence citations, informed by the other four
findings' final shape.

**Decision update (contract frozen, no implementation):** `#1759`/`#1900`/
`#1761`/`#1760`/`#1762`, all closed since this section was first written,
have settled every dependency this section itself named - `QuotaKind`'s
active shape is final (7 members, no `ConstPool`, no `TraceEntries`), and
`RuntimeError::QuotaExceeded` is confirmed the live channel for 5 of 7
`QuotaKind`s (`Steps`/`Calls`/`Frames`/`Registers`/`EffectCalls`) - **not**
the sole quota-exhaustion outcome: `StackDepth` exhaustion is deliberately
remapped to `RuntimeError::StackOverflow` by an existing compatibility
mapping, and `SymbolTable` is a static `sm-verify` admission rejection,
outside this runtime channel entirely. The disposition question is now
decided in
`docs/roadmap/stable_foundation/ssf08_1763_runtime_failure_taxonomy_decision.md`:
**Model B (semantic program traps only)**. `RuntimeTrap` is frozen as
meaning exactly "an execution-semantic failure produced by the semantics
of an executing Semantic instruction/program operation, independent of
whether that execution was admitted through `sm-verify` first" - its four
live members (`AssertionFailed`, `BorrowWriteConflict`,
`DivisionByZero`, `ArithmeticOverflow`) are the whole of that class; the
other nine are frozen `REMOVE_DUPLICATE`, each with its real authority
already living, fully structured, on a same-conceptual `RuntimeError`
variant (`StackOverflow`, `StackUnderflow`, `TypeMismatchRuntime`,
`BadFormat`, `InvalidJumpAddress`, `CapabilityDenied`, `HostAbi`,
`VerifierRejected`, `QuotaExceeded` respectively) - confirmed zero payload
information is lost by the narrowing, since none of the nine was ever
constructed to begin with. `RuntimeError` itself needs no variant change;
`Trap(RuntimeTrap)` is retained, narrowed automatically. Models A
(trap = all execution failures) and C (eliminate `RuntimeTrap` entirely)
were falsified as unforced architecture changes with no evidenced
benefit; Model D (reserve the dead variants) was falsified for the same
reason `ConstPool`/`TraceEntries` were - no compatibility authority
requires preserving inert, misleading vocabulary, and five of the nine
dead variants actively name-collide with the real `RuntimeError` channel
doing the actual job. The exact future mechanic (which variants are
removed, the `vm.md` correction, the `trap_taxonomy.md` addendum, the
public-API RED→GREEN scope, the test matrix) is fully frozen in that
document's own §21-§25 - a fully-specified mechanic, not an open choice.
**This is a contract decision only. `#1763` remains OPEN. No field,
variant, diagnostic, or golden snapshot was touched by this update -
AC4.d/AC4.e remain not satisfied until a separately authorized
implementation checkpoint executes this now-complete mechanic.**

## 8. New residual findings discovered during this audit

Per item 11's explicit instruction, this section records findings outside
the five filed issues rather than silently expanding implementation scope:

1. **`docs/spec/vm.md`'s "Trap And Error Model" list omits three live
   failure families** (`AssertionFailed`, `DivisionByZero`,
   `ArithmeticOverflow`) — see §7. This is a documentation gap adjacent to,
   but distinct from, #1763's own filed claim (which is about *unconstructed*
   variants, not *unlisted* ones). Recorded here as a new residual finding,
   not folded into #1763's disposition text as if it were the same defect.
2. **`docs/roadmap/language_maturity/core_trust_freeze/trap_taxonomy.md`'s
   own evidence citation for "Stack overflow" is imprecise** (cites
   `RuntimeTrap::StackOverflow`; the actual constructed value is the
   top-level `RuntimeError::StackOverflow`) — see §7. This document is owned
   by a different governance track (CTF-2/PCC), not SSF-08; SSF-08 cannot
   correct it unilaterally. Recorded here as a cross-track finding to
   surface, not to repair.
3. **`SymbolTable` is a `RuntimeQuotas`-sourced quota kind enforced at the
   *verifier* layer, not the VM**, contradicting `docs/spec/quotas.md`'s own
   blanket "Enforcement owner: `sm-vm`" ownership statement — see §12. This
   was not a safety gap (the quota genuinely is enforced, just not by the
   crate the spec named), but it was a documentation-precision gap.
   **Corrected** in #1761's implementation PR: `quotas.md`'s header,
   "Quota model rule," and "Ownership Rule" sections now name `sm-verify`'s
   static, pre-execution enforcement of `SymbolTable` explicitly, since
   that PR was already the authoritative edit point for this exact file.
4. **`EffectCalls` — an already-ACTIVE, already-enforced runtime quota — uses
   unchecked `vm.effect_calls + 1` in `bump_effect_calls`, which wraps
   silently (rather than erroring) in a release build if the counter is ever
   driven near `usize::MAX`.** Discovered during the #1759 Steps/Calls
   contract-decision pass while auditing overflow discipline for the two new
   counters (see
   `docs/roadmap/stable_foundation/ssf08_1759_steps_calls_contract_decision.md`
   §10.1) - not discovered by, and not part of, this document's own original
   fresh-check pass. **Classification: FAIL-OPEN EDGE / QUALIFICATION GAP.**
   **AC impact: AC4.a** - an active, enforced quota's counter is not
   qualified to be fail-closed at its own numeric ceiling, which the #1759
   decision now requires of `Steps`/`Calls` but does not itself apply to
   `EffectCalls`. **This is not `#1760`** (`#1760` is `TraceEntries`/
   `trace_enabled`) **and is not repaired by `#1759`** (`Steps`/`Calls` are
   new counters; `EffectCalls` is a separate, pre-existing one) - `#1759`'s
   implementation scope is explicitly not expanded to fix it. **Tracking
   issue: #1900 (FA-08-011)**, filed after the contract PR landed.
   **Fully repaired in this checkpoint** - the free-function
   `bump_effect_calls(vm: &mut VM)` site (Gate/Pulse/State/Event/Clock
   opcodes) now reuses the shared `charge_counter` primitive #1759 landed
   (generalized beyond `Steps`/`Calls` to cover all three execution
   counters), fails closed at the `usize::MAX` ceiling with saturated-`used`
   reporting only, and was verified by release-mode mutation proofs (a
   restored bare `+ 1` wraps silently in `--release` and fails the ceiling
   test; a `saturating_add` substitute lets the ceiling case through as
   `Ok`, masking exhaustion). Call-site ordering relative to capability
   checks and host dispatch is unchanged. A second, independent site,
   `ApplicationVmHost::bump_effect_calls`, was found during this same
   checkpoint to share the identical defect; owner-authorized scope
   expansion repaired it too, before merge, with its own release-mode
   mutation proof - see §8 finding 6. AC4.a is now satisfied for
   `EffectCalls` as a whole quota kind; AC4 remains not satisfied globally
   - see §12.
5. **`examples/benchmarks/snake_learning.sm` exceeds the already-published
   `VerifiedLocal` `max_steps = 100000` baseline by exactly one opcode
   (`QuotaExceeded { kind: Steps, limit: 100000, used: 100001 }`).**
   Discovered running the full workspace test suite during #1759's own
   implementation checkpoint - `tests/snake_learning_benchmark.rs`'s `smc
   run` invocation hits `cmd_run_controlled_observation`, hardcoded to
   `ExecutionConfig::for_context(ExecutionContext::VerifiedLocal)` with no
   CLI override available. **Classification: FIRST REAL QUOTA VIOLATION
   SURFACED BY #1759's ACTIVATION** - not a defect in #1759's
   implementation; `Steps` was always meant to bound this execution, and
   this benchmark's own step usage always nominally exceeded it, invisibly,
   until enforcement existed. **AC impact: none directly** (AC4.a concerns
   the quota mechanism's own correctness, not every consumer's compliance
   with it). Explicitly not fixed here - editing this benchmark's own
   parameters, adding a CLI quota-override flag to `smc run`, and raising
   the published `verified_local` baseline are each a legitimate but
   separate decision outside #1759's narrow contract-implementation scope.
   Per explicit user decision, `snake_learning_passes_check_run_compile_verify`
   is left `#[ignore]`d with a citation back to this finding rather than
   silently patched. **Tracking issue: #1902 (FA-08-012)** - independent of
   #1759, so the ignore annotation does not cite an issue that will itself
   close once #1759's implementation lands.
6. **`EffectCalls` had two architecturally independent enforcement sites
   sharing one `QuotaKind`, and both are now repaired.**
   `bump_effect_calls(vm: &mut VM)` (the free function, called from
   `Opcode::GateRead`/`GateWrite`/`PulseEmit`/`StateQuery`/`StateUpdate`/
   `EventPost`/`ClockRead` - the PROMETHEUS/Gate-opcode host boundary) and
   `ApplicationVmHost::bump_effect_calls(&mut self)` (a separate
   `effect_calls: usize` field on a separate host-bridge type introduced by
   #1600 specifically so application builtins would also consume
   `max_effect_calls`, called from the `args_read`/`fs_read_text`/
   `fs_write_text`/etc. boundary and exercised by
   `tests/ssf04_effect_quota.rs`) both had the identical unchecked
   `+ 1` increment. Fresh implementation inspection during #1900 discovered
   the sibling only after the free-function site was already repaired;
   since both are production charge paths for the one `EffectCalls`
   resource - not two separate features - the owner authorized expanding
   #1900's own PR rather than filing a second issue. Both sites now reuse
   the shared `charge_counter` primitive, and both were independently
   proven load-bearing by their own release-mode mutation proof (the
   sibling's own bare `+ 1` wraps silently under `--release`, exactly like
   the free-function's did). **Classification: SIBLING FAIL-OPEN EDGE,
   DISCOVERED AND REPAIRED WITHIN #1900's OWN IMPLEMENTATION CHECKPOINT,
   BEFORE MERGE.** **AC impact: AC4.a** - now satisfied for `EffectCalls`
   as a whole quota kind (both real charge sites are fail-closed and
   tested); AC4 remains not satisfied globally, since AC4.b-e and
   #1760-#1763 are unaffected. **Tracking issue: none allocated** - by
   owner decision, this was resolved within #1900 rather than split into a
   separate issue.

None of these six is added to Lane 5's implementation scope by this audit
beyond #1900's own now-completed repair.

## 9. Cross-issue interaction audit

The governing brief's own candidate graph was tested against the fresh
evidence above and required one correction: the initial pass in this audit
drew `#1759` feeding into `#1763`'s own dependency, which does not survive
a second, more careful falsification pass (below) - `#1759` decides
whether `Steps`/`Calls` charging becomes *real*, not whether `QuotaKind`'s
own variant set changes shape, and `RuntimeTrap::QuotaExceeded`'s fate
depends only on the latter. Corrected graph (with the `SymbolTable`/verifier
distinction in §12 noted as an addition that does not change the graph's
shape, only explains why it is not itself a sixth finding):

```
#1759  Steps/Calls (bounded execution, P1)
       — independent track, does not feed #1763

#1760  TraceEntries ──┐
                       ├──→ final quota-related #1763 reconciliation
#1761  ConstPool ──────┘    (QuotaKind shape feeds RuntimeTrap::QuotaExceeded's
                             own disposition)

#1762  context/provenance decision — independent, may start immediately;
                            RECORD-style *implementation* (not the decision
                            itself) should follow #1759's own outcome - see §6

#1763  non-quota taxonomy work — independent, may start immediately
#1763  QuotaExceeded-variant question — waits on #1760/#1761 only
```

**Attempted falsification of this graph:** could `#1763` be resolved
*before* `#1760`/`#1761`? Only partially - the eight genuinely-live and
dead-code-adjacent non-quota `RuntimeTrap` variants (`AssertionFailed`/
`BorrowWriteConflict`/`DivisionByZero`/`ArithmeticOverflow` plus the five
never-constructed ones with no quota relationship) and the `vm.md` omission
fix (§8, finding 1) do not depend on the quota findings at all and *could*
move independently. Only the `QuotaExceeded` variant's specific fate is
genuinely blocked, and only on `#1760`/`#1761` - **not on `#1759`**, which
was the initial drafting error this correction fixes: `#1759` governs
whether `Steps`/`Calls` are actually charged at runtime, a question that
does not touch which `QuotaKind` variants exist or which `RuntimeError`
channel their exhaustion already uses (`RuntimeError::QuotaExceeded`,
already live regardless of `#1759`'s outcome). This means #1763 is not a
strict single blocking dependency - it is **splittable**: the non-quota-related
taxonomy corrections could proceed in parallel with every other Lane 5
track, while only the `QuotaExceeded`-variant question waits, and only on
`#1760`/`#1761`. This corrects, rather than merely refines, the original
hypothesis that #1763 belongs last as a whole - only one narrow slice of
it does; the rest is independent from the start.
Could `#1762` depend on `#1760`/`#1761` too, not just `#1759`? Checked: no
- `#1762`'s provenance-recording question is orthogonal to which specific
`QuotaKind` variants are active; it would apply equally whether `Steps`
alone or all four inert kinds end up implemented/removed. The dependency is
specifically and only on `#1759`, as stated in §6.

## 10. RuntimeQuotas / QuotaKind full inventory

| Quota kind | Public field | Finite baseline | Actual runtime resource | Enforcement site | Charge point | Exhaustion error | Positive test | Negative/exhaustion test | Docs | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `Frames` | `max_frames` | 256/256/256 | Call-stack frame count | `sm-vm` (`push_frame`) | frame push | `RuntimeError::QuotaExceeded` | yes | yes (existing suite) | `quotas.md` | **ACTIVE** |
| `StackDepth` | `max_stack_depth` | 256/256/256 | Effective stack depth | `sm-vm` (`push_frame`) | frame push | `RuntimeError::StackOverflow` (compat: surfaced as `StackOverflow`, not `QuotaExceeded`, per `quotas.md`'s own documented compatibility note) | yes | yes (existing suite) | `quotas.md` | **ACTIVE** |
| `Registers` | `max_registers` | 4096/4096/8192 | Register vector growth | `sm-vm` (frame init + growth) | register write | `RuntimeError::QuotaExceeded` | yes | yes (existing suite) | `quotas.md` | **ACTIVE** |
| `EffectCalls` | `max_effect_calls` | 1024/0/4096 | Effect-opcode invocation count | `sm-vm`, two independent sites: `bump_effect_calls(vm)` (Gate/Pulse/State/Event/Clock) and `ApplicationVmHost::bump_effect_calls` (application builtins) | effect opcode / builtin dispatch | `RuntimeError::QuotaExceeded` | yes | yes (existing suite + #1900 numeric-ceiling suite, both sites) | `quotas.md` | **ACTIVE; both production charge paths overflow-safe, numeric-ceiling fail-closed, qualified** (§8 findings 4 and 6 - both repaired by #1900) |
| `SymbolTable` | `max_symbol_table` | 16384 (all profiles) | Program-wide unique runtime symbol count | **`sm-verify`** (not `sm-vm` - see §12) | pre-execution, at admission | verifier `RejectReport` | yes | yes (#1820 suite) | `quotas.md` (ownership line corrected by #1761's implementation PR - see below) | **ACTIVE, correctly documented** |
| `Steps` | `max_steps` | 100000/100000/250000 | Opcode-dispatch fuel counter | `sm-vm` (`exec_loop_with_profile`) | opcode decode succeeds | `RuntimeError::QuotaExceeded` | yes | yes (incl. backward-loop regression) | `quotas.md`, `vm.md` | **ACTIVE** (#1759, implemented, checked-add overflow discipline) |
| `Calls` | `max_calls` | 16384/16384/32768 | Admitted non-root Semantic invocation count | `sm-vm` (`push_frame`) | frame push, root-exempt | `RuntimeError::QuotaExceeded` | yes | yes (incl. root-exemption + boundary suite) | `quotas.md`, `vm.md` | **ACTIVE** (#1759, implemented, checked-add overflow discipline) |

`ConstPool`/`max_const_pool` and `TraceEntries`/`max_trace_entries` are
intentionally **no longer rows above** - neither is a current
`RuntimeQuotas`/`QuotaKind` member. #1761's REMOVE disposition (frozen by
`docs/roadmap/stable_foundation/ssf08_1761_constpool_contract_decision.md`)
has been implemented: the enum variant, the struct field, all three
baseline-profile assignments, the `exceed()` match arm, the public API
golden snapshot entries, and the active `docs/spec/quotas.md` references
are all removed. #1760's SPLIT disposition (frozen by
`docs/roadmap/stable_foundation/ssf08_1760_trace_contract_decision.md`)
has also been implemented: `QuotaKind::TraceEntries` and
`ExecutionConfig::trace_enabled` are removed the same way; unlike
`ConstPool`, `max_trace_entries` itself was not inert - its raw numeric
value renamed in place to `RuntimeQuotas::max_debug_symbols_per_function`,
values and `sm-verify` admission behavior preserved exactly, and is
therefore intentionally still present on `RuntimeQuotas` under its new
name and new, non-quota classification (see the "Verifier admission
policy, not a runtime quota" row-equivalent note below). Both removed
kinds remain solely as historical record in their own decision documents
and in §4/§5 above - resolved entries, not live quotas.
`docs/spec/quotas.md`'s own enforcement-owner text (previously overstating
that all quota enforcement belongs to `sm-vm`) was also narrowed, in
#1761's implementation PR, to name `sm-verify`'s static enforcement of
`SymbolTable` explicitly, since that PR was already the authoritative edit
point for this exact file and the discrepancy (§8 finding 3) was already
proven.

**Verifier admission policy, not a runtime quota:**
`max_debug_symbols_per_function` (formerly `max_trace_entries`) | 8192/
4096/16384 | per-function debug-symbol-table admission limit | `sm-verify`
(`verify_function_code`, direct field read, not via `QuotaKind`/
`enforce_quota`) | pre-execution, at admission | `VerificationCode::
ResourceLimitExceeded` | yes (4096-accepted, 4097-rejected-under-
`pure_compute`, 4097-accepted-under-`verified_local`) | same suite | `quotas.md`,
`verifier.md` | **ACTIVE, correctly classified as verifier-admission
policy rather than a `QuotaKind` member - live specifically for
`pure_compute` (4096 < `sm-format`'s fixed 8192 decode-time cap), dead
code for `verified_local`/`kernel_bound`** (#1760, implemented).

Every `QuotaKind` variant is accounted for above; at audit time four
inert kinds were found (`Steps`, `Calls`, `ConstPool`, `TraceEntries`) -
all four are now resolved: `Steps`/`Calls` are `ACTIVE` (#1759),
`ConstPool` is removed entirely (#1761), and `TraceEntries`/
`trace_enabled` are removed entirely with `max_trace_entries`'s real,
mislabeled, profile-inconsistent consumer preserved under its correct
name and classification (#1760). The seven remaining `QuotaKind`
variants (`Frames`, `StackDepth`, `Registers`, `EffectCalls`,
`SymbolTable`, `Steps`, `Calls`) each have a real, current enforcement
owner, freshly re-confirmed against the `#1760` implementation branch
(not assumed from this table's own prior text) - see §12.

## 11. Verifier limits vs runtime quotas (architecture check)

Confirmed the architecture keeps these two resource domains genuinely
separate, per `docs/spec/verifier.md`'s own "Verifier resource budgets"
section (re-confirmed fresh, unchanged since the #1718 checkpoint's own
reading of this file):

- **Bounded before execution** (`VerificationLimits`, `sm-verify` alone):
  `max_work_units` (whole-artifact static-analysis work), `max_state_words`
  (peak static-analysis memory) - governs the definite-register-assignment
  pass (#1756) only, reported as `AnalysisStateLimitExceeded`/
  `AnalysisWorkLimitExceeded`.
- **Bounded during execution** (`RuntimeQuotas`, `sm-runtime-core`/`sm-vm`):
  the nine kinds in §10's table.
- **A third, distinct domain**: artifact/decode-time byte-level limits
  (`sm-format`, `MAX_FUNCTIONS`/`MAX_STRINGS_PER_FUNCTION`/etc.), reported
  as `ResourceLimitExceeded`, structurally separate from both of the above.

`#1751`'s own ruling (cited in `verifier.md`) that conflating a dynamic
execution resource with a static verifier bound is unsound remains the
correct authority: **`max_steps`'s missing enforcement must not be "fixed"
by moving step-counting into the verifier.** A verified backward loop must
still be bounded at *runtime*, by the VM, if bounded execution is the
published promise - static analysis cannot prove termination of an
admissible unbounded loop, so this is not an available shortcut, and no
evidence in this repository suggests anyone has attempted that shortcut.

The `SymbolTable` case (§10) is the one quota kind that crosses this
boundary in an interesting way: it is a `RuntimeQuotas`-sourced value
(shared taxonomy, `sm-runtime-core`-owned) but is actually charged at
*verifier admission time*, before execution - this is architecturally
sound (the count is knowable statically from the decoded program, so
proving it before execution is strictly stronger than checking it at
runtime), but it means `quotas.md`'s blanket "Enforcement owner: `sm-vm`"
statement is imprecise for this one kind (§8, finding 3).

## 12. AC4 decomposition into testable subcriteria

Rewriting AC4 ("Resource quotas and failure taxonomy are explicit") into
explicit, testable subcriteria, checked against Position A and the current
target contract for over-strengthening:

- **AC4.a** — Every quota advertised as an active runtime bound has an
  authoritative resource, charge point, deterministic exhaustion behavior,
  and test. *(Satisfied for `Frames`/`StackDepth`/`Registers`/`SymbolTable`,
  as of #1759's implementation `Steps`/`Calls`, and as of #1900's
  implementation `EffectCalls` - both of its independent charge sites,
  `bump_effect_calls(vm)` (Gate-opcode) and
  `ApplicationVmHost::bump_effect_calls` (application builtins), are now
  fail-closed at the numeric ceiling - §8 findings 4 and 6.)*
- **AC4.b** — Inert/deferred resource concepts are not presented as
  enforced runtime quotas. *(Satisfied, freshly re-audited against the
  current implementation branch rather than assumed from this document's
  own prior text: `ConstPool`'s REMOVE disposition (see
  `docs/roadmap/stable_foundation/ssf08_1761_constpool_contract_decision.md`)
  and `TraceEntries`/`trace_enabled`'s REMOVE disposition (see
  `docs/roadmap/stable_foundation/ssf08_1760_trace_contract_decision.md`)
  have both landed - neither is presented as a quota kind anywhere in
  production code, the public API, or `docs/spec/quotas.md`.
  `max_trace_entries`'s real but mislabeled consumer was RE-SCOPED, not
  deleted: it survives as `RuntimeQuotas::max_debug_symbols_per_function`,
  explicitly documented as verifier-admission policy rather than a
  `QuotaKind` member (zero `QuotaKind` membership, zero `sm-vm` charge
  sites, zero `RuntimeError::QuotaExceeded` paths). Every remaining
  `QuotaKind` variant (`Frames`, `StackDepth`, `Registers`, `EffectCalls`,
  `SymbolTable`, `Steps`, `Calls`) was re-enumerated against current code
  and has a real, current enforcement owner - see §10. This is a
  point-in-time confirmation on the current branch, not a standing
  guarantee against a future addition reopening the same failure mode.)*
- **AC4.c** — `ExecutionContext`/provenance does not overstate the quota
  envelope that actually governed execution. *(Satisfied, freshly
  re-audited against the implementation branch: `RuntimeSessionDescriptor`
  and `AuditSessionMetadata` both now record `quotas: RuntimeQuotas`,
  copied one-way from the same `ExecutionConfig` used for execution -
  never re-derived from `context`. Every production construction site of
  both structs was manually classified (SAME AUTHORITY or explicit
  synthetic/audit-only fixture); zero discard an available effective
  envelope in favor of a context-derived reconstruction. A distinct
  custom envelope was proven to survive the full canonical archive
  pipeline value-for-value, and two sessions sharing one `ExecutionContext`
  with different `RuntimeQuotas` were proven distinguishable after a
  multi-session round-trip. See
  `docs/roadmap/stable_foundation/ssf08_1762_execution_envelope_provenance_decision.md`
  §25. This is a point-in-time confirmation on the current branch, not a
  standing guarantee against a future construction site reintroducing the
  same failure mode.)*
- **AC4.d** — Verification rejection, runtime quota exhaustion, semantic
  trap, capability denial, and host/ABI failure have an explicit,
  deterministic taxonomy that matches what production code actually
  constructs. *(Not satisfied: nine of thirteen `RuntimeTrap` variants are
  never constructed, and `vm.md`'s own list is incomplete in the other
  direction. Disposition now frozen - Model B (semantic program traps
  only), see
  `docs/roadmap/stable_foundation/ssf08_1763_runtime_failure_taxonomy_decision.md` -
  but not yet implemented; the exact mechanic (which variants are
  removed, diagnostic/doc updates, public-API scope, test matrix) is
  fully specified, not an open choice.)*
- **AC4.e** — Public docs (`docs/spec/quotas.md`, `docs/spec/vm.md`,
  `trap_taxonomy.md` insofar as SSF-08 can influence a cross-track
  document) match the actual failure/resource authority present in code.
  *(Not satisfied per §8's three residual findings. `vm.md`'s correction
  and a `trap_taxonomy.md` addendum are both frozen as required future
  implementation steps in the same decision document referenced above.)*

These subcriteria refine, but do not strengthen, #1579's existing AC4 text
- they decompose one broad acceptance line into checkable parts already
implied by "explicit," without adding any new promise Position A did not
already make (in particular, AC4.b explicitly permits narrowing/removal as
a valid closure path, not just implementation, matching Position A's own
bounded, deterministic-VM framing rather than an aspirational one).

## 13. Minimal Lane 5 DAG

```
Track A — bounded execution (safety-priority, P1)
  #1759

Track B — inert quota/config vocabulary (decision-then-repair)
  #1760  (open decision: implement vs narrow/remove)
  #1761  (narrowing direction frozen - REQUIRED, CONTRACT NARROWING; only
          the exact mechanism, remove vs re-scope, needs a short decision)

Track C — configuration/provenance identity (decision-then-repair)
  #1762  (depends on #1759's outcome for its RECORD option specifically;
          the decision itself can start in parallel)

Track D — failure taxonomy (partially blocked)
  #1763  non-quota-related corrections (vm.md omissions, RuntimeTrap
          narrowing for the 8 non-quota orphaned variants) — independent,
          can run in parallel with A/B/C
  #1763  QuotaExceeded-variant disposition — blocked on B (Tracks #1760/#1761)
          settling which QuotaKind variants remain active
```

**Parallelizable:** Tracks A, B, and C's own decision passes may all start
independently and in parallel - none of their *investigative/decision*
work depends on another track's completion. Track D's non-quota half is
likewise independent. **Sequencing constraint:** Track D's quota-related
half (the `RuntimeTrap::QuotaExceeded` question) must wait for Track B's
final disposition; Track C's RECORD-style repair option (not its decision
pass) is strongest once Track A's disposition is known.

**Exact next issue/checkpoint (not started by this audit):** `#1759`
(Track A) - the sole P1, the only finding that breaks a stated safety
promise outright (bounded execution), and the only one with no
architectural ambiguity about whether repair is even wanted (unlike
`#1760`, which is still genuinely open between implement and remove, and
`#1762`, which needs a policy choice among VALIDATE/RECORD/NARROW CLAIM;
`#1761`'s *direction* is already frozen by this audit, with only its exact
mechanism left open). Its own first step is a small, docs-scoped decision
pass freezing the three contract questions in §3 (step/call counting
semantics, exhaustion timing) - not code - before any counter is added.

## 14. Verdicts

**Lane 5 READY TO IMPLEMENT: NO** — `#1760` and `#1762` still require an
open contract decision before any implementation-shaped repair is
well-defined; `#1761`'s narrowing direction is frozen but its exact
mechanism (remove vs. re-scope) still needs a short decision; `#1763` is
partially blocked, on `#1760`/`#1761` only, for its `QuotaExceeded`-variant
slice alone. Only `#1759` has an unambiguous implementation-required
disposition, and even it needs its own three contract questions (§3) frozen
first.

**AC4 SATISFIED: NO.**

**SSF-08 umbrella #1579 remains OPEN: YES.**

No production Rust changed while producing this document.
