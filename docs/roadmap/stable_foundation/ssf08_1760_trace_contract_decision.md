# SSF-08 Lane 5 / #1760 — Trace Contract Decision

Status: **CONTRACT DECISION ONLY. NO PRODUCTION BEHAVIOR CHANGE.**
Baseline SHA: `8a12f1aa7e81a8c07e6cf8c27810b71a2b236684` (`main`, confirmed
current).
Target: `#1760` / `FA-08-002`.
Umbrella: SSF-08 umbrella #1579 remains OPEN.

This document freezes the disposition of `ExecutionConfig::trace_enabled`,
`QuotaKind::TraceEntries`, and `RuntimeQuotas::max_trace_entries`. It does
not delete anything, does not touch `#1762`/`#1763`/`#1902`, and does not
change any golden snapshot, baseline value, or production Rust.

**Headline finding, up front:** falsification did not confirm the working
hypothesis of a clean three-symbol REMOVE. `trace_enabled` and
`QuotaKind::TraceEntries` are exactly as inert as `ConstPool` was. But
`RuntimeQuotas::max_trace_entries` - the numeric field specifically - has
**one real, live, per-profile-varying production reader** in `sm-verify`,
using it (under a misleading name) as a per-function debug-symbol-count
structural cap, not an execution-trace budget. This is the "unexpected
architectural consumer" this decision's own governing brief asked to be
caught and reported rather than papered over. The selected disposition is
therefore **SPLIT**, not a uniform REMOVE across all three symbols.

## 1. Fresh evidence

Repository-wide search for `trace_enabled`, `TraceEntries`,
`max_trace_entries`, `trace`, `tracing`, `profiler`/`profile`/
`record_opcode`, and related terms across `crates/sm-runtime-core/`,
`crates/sm-vm/`, `crates/sm-verify/`, `crates/prom-runtime/`,
`crates/prom-audit/`, `src/bin/smc.rs`, `docs/spec/vm.md`,
`docs/spec/quotas.md`.

### 1.1 `ExecutionConfig::trace_enabled`

Three hits total, all in `crates/sm-runtime-core/src/lib.rs`: the field
declaration, its `false` default inside `ExecutionConfig::new`, and the
golden-snapshot mirror. **Zero reads anywhere else in the workspace.**
No code path branches on this value. **Classification: NO TRACE
RESOURCE.**

### 1.2 `QuotaKind::TraceEntries`

`crates/sm-vm/src/semcode_vm.rs` is the only place `enforce_quota`/
`charge_counter` are ever called, and every call site uses `Steps`,
`Calls`, `Frames`, `StackDepth`, `Registers`, or `EffectCalls` - never
`TraceEntries`. As an *execution-trace quota-enforcement mechanism*, this
variant is exactly as inert as `ConstPool` was before #1761. **Classification:
NO TRACE RESOURCE** (as tracing; see §1.3 for the field's own separate
fate).

### 1.3 `RuntimeQuotas::max_trace_entries` — the actual finding

`crates/sm-verify/src/lib.rs`'s `verify_function_code` (called once per
function, inside the program's function-verification loop) contains:

```rust
let debug_symbol_count = env.debug_symbols.len();
if debug_symbol_count > quotas.max_trace_entries {
    return Err(reject_one(
        name,
        VerificationCode::ResourceLimitExceeded,
        0,
        format!(
            "debug section uses {} entries, exceeding the trace budget of {}",
            debug_symbol_count, quotas.max_trace_entries
        ),
    ));
}
```

`env.debug_symbols` is the decoded `DBG0` per-function debug-symbol table
(pc/line/col source-location records) - **not** an execution trace by any
definition. This is a real, live, per-function structural size check that
happens to reuse the `max_trace_entries` field and quota-taxonomy slot
under a name that describes something else entirely.

**Is it dead code?** Partially. `sm-format`'s own decoder already enforces
an independent, hardcoded `MAX_DEBUG_SYMBOLS_PER_FUNCTION = 8192` at
decode time (`crates/sm-format/src/semcode_decode.rs`), which runs
*before* `sm-verify` ever sees the artifact. Comparing against the three
baseline profiles:

| Profile | `max_trace_entries` | vs. `sm-format`'s fixed 8192 cap | Effect of this check |
|---|---|---|---|
| `verified_local` | 8192 | equal | **dead code** - decode already guarantees ≤8192, so this comparison can never fire |
| `kernel_bound` | 16384 | looser | **dead code** - same reason, even more so |
| `pure_compute` | 4096 | **stricter** | **live** - a function with 4097-8192 debug symbols passes `sm-format`'s decode-time cap but is rejected here, specifically and only under `pure_compute` |

So this is not fully inert vocabulary the way `ConstPool` was: it is a
**real, narrow, profile-inconsistent structural limit**, mislabeled as a
trace budget, redundant with a separate decode-time cap for two of three
profiles, and quietly load-bearing for the third. Deleting the field
outright (the `ConstPool` playbook) would either fail to compile (a real
caller exists) or, if the caller is deleted too, silently loosen
`pure_compute`'s debug-symbol bound back to `sm-format`'s blanket 8192 -
an actual behavior change, not a removal of dead vocabulary, and not one
this decision-only checkpoint is authorized to wave through as a side
effect of closing `#1760`.

This finding was not caught by the original Lane 5 audit's own `#1760`
section (`ssf08_lane5_resource_failure_closure_audit.md` §4), which
correctly found zero `enforce_quota` call sites for `TraceEntries` but did
not trace the raw numeric field `max_trace_entries` to this separate,
non-`enforce_quota` read site in `sm-verify`. This document supersedes
that section's evidence on this specific point without erasing it (see
the Lane 5 audit update below).

### 1.4 Trace-adjacent mechanisms, classified (none of them connect to `trace_enabled`/`TraceEntries`)

| Mechanism | Classification | Evidence |
|---|---|---|
| `VmOpcodeProfile` (`vm-profile` Cargo feature) | **PROFILING ONLY** | A fixed-size `[u64; N]` per-opcode dispatch-count histogram plus a `total_instructions` counter (`record_opcode_slot`). No ordered/retained sequence of individual events, no timestamps, no per-instance data - a counter, not a trace. Zero references to `trace_enabled`/`TraceEntries`. |
| `SevenHellDiagnostic` (`src/bin/smc.rs`) | **DEBUG METADATA / DIAGNOSTIC-REPORTING ONLY** | A one-shot post-failure diagnostic struct (syntax/verifier/VM-trap categories). Zero occurrences of the word "trace" anywhere in the file. |
| `HelloObservationRuntime`/`HelloObservationEvent` | Ordered and sequence-indexed, but records only "controlled text observation" (print-like host-output calls) - not opcode-level execution. Unrelated to `trace_enabled`/`TraceEntries`. |
| `prom-audit`'s `AuditTrail`/`HelloObservation` audit records | **AUDIT/PROVENANCE ONLY** | Capability/host-call observation records with Record/Redact/NoStore/Deny policy; the module's own header comment describes it as a skeleton "without wiring into production audit storage." Not an opcode-level execution trace. |
| Rust `tracing` crate | **N/A** | Not a workspace dependency at all (zero `Cargo.toml` hits, zero `use tracing::`). |
| Debug symbol table (`DebugNameMap`, `RuntimeSymbolTable`, `DBG0` decode) | **DEBUG METADATA ONLY** | Static pc/line/col table baked in at compile time - the very resource `max_trace_entries` accidentally bounds (§1.3), but conceptually and structurally distinct from an execution trace. |
| "Golden trace" (`docs/roadmap/language_maturity/core_trust_freeze/golden_trace_policy.md`) | **UNRELATED CONCEPT** | "Trace" here means a recorded test-fixture/golden-snapshot category (Syntax/Type/IR/SemCode/Verifier/VM trace = test-stability artifacts), explicitly distinguished in that same document from "VM runtime traces." Not a runtime feature. |

**No real, ordered, retained execution-trace consumer exists anywhere in
the codebase.** Profiling counters, diagnostic reports, audit records,
debug symbols, and test-fixture "traces" are each independently built,
independently gated, and structurally incapable of being silently
relabeled as `TraceEntries` (per this decision's own falsification
discipline: a counter is not a trace, audit metadata is not a trace,
debug symbols are not a trace, logging is not a trace).

## 2. Three questions, answered independently

**A. Does Stable Foundation need a runtime trace feature?**
No normative authority found. `docs/roadmap/stable_foundation/stable_foundation_target_contract.md`
contains zero mentions of "trace." `docs/DNA.md`'s "Local auditability and
reproducible execution traces as part of the system model" is a
vision/differentiation bullet under "What Remains Unique to Semantic" -
descriptive aspiration, not a `must`-statement - and the same document's
own "Architectural Invariants" section (which does use `must` throughout)
contains no trace requirement. No PCC/`core_trust_freeze` document, no
determinism-guarantee document, and no debugging-contract document
requires one either. **Answer: no requirement found; future usefulness is
not the same as a current promise.**

**B. If yes, must trace enablement belong in `ExecutionConfig`?**
Moot given (A) - no current requirement exists to place. Recorded for
completeness: even if a future trace feature is built, nothing found
requires it to be a boolean flag on the shared execution-config struct
rather than a separate, purpose-built configuration surface.

**C. If yes, is trace storage a quota-governed runtime resource requiring
`TraceEntries`/`max_trace_entries`?**
Moot given (A). Recorded for completeness: even if a future trace feature
is built, nothing found requires its storage bound to be expressed as a
`RuntimeQuotas`/`QuotaKind` member rather than its own dedicated
configuration.

It is valid, and is the answer given here, for future tracing to remain
plausible while current Stable Foundation makes no such promise.

## 3. Decision candidates evaluated

**A. IMPLEMENT** - build a real VM execution-trace facility: `trace_enabled`
controls recording, `TraceEntries` owns a precise entry resource,
`max_trace_entries` is enforced against it, exhaustion semantics are
deterministic.

**B. REMOVE** - remove `ExecutionConfig::trace_enabled`,
`QuotaKind::TraceEntries`, and `RuntimeQuotas::max_trace_entries` in their
entirety. Future tracing, if ever required, returns under a separately
specified observability/debugging contract.

**C. SPLIT** - keep some trace notion but remove `TraceEntries` quota
vocabulary, or vice versa.

**D. RESERVED/DEPRECATED** - keep the current fields but explicitly label
them inert/reserved.

## 4. REMOVE (uniform, all three symbols) — falsification

Actively searched for authority requiring runtime tracing in: the Stable
Foundation target contract, `docs/spec/vm.md`, debugging contracts,
audit/provenance contracts, determinism guarantees, PCC/verifier
qualification requirements, CLI behavior, tests, and release/compatibility
documents. No document requires anything materially equivalent to "VM
execution must expose/retain a bounded ordered runtime trace." The
strongest text found (`docs/DNA.md`'s "reproducible execution traces")
is aspirational marketing/vision language, not a contractual `must`.

**REMOVE survives falsification for `trace_enabled` and
`QuotaKind::TraceEntries` as an execution-trace concept.** It does **not**
survive uniformly for `max_trace_entries` as a bare field deletion,
because §1.3 establishes a real, compiling, behavior-affecting consumer -
which `ConstPool` never had at any point in its own falsification. A
disposition that ignores this and freezes "delete all three, identically"
would be forcing the original hypothesis through past contrary evidence,
exactly what this checkpoint's own falsification discipline exists to
prevent.

## 5. SPLIT — the selected disposition (FROZEN, mechanic included)

- **`ExecutionConfig::trace_enabled` → REMOVE.** Zero readers, zero
  authority, no compatibility bridge needed.
- **`QuotaKind::TraceEntries` (the execution-trace taxonomy member) →
  REMOVE.** Zero enforcement, zero authority, exactly the `ConstPool`
  pattern.
- **`RuntimeQuotas::max_trace_entries` → RENAME/RE-SCOPE to
  `RuntimeQuotas::max_debug_symbols_per_function`. This mechanic is
  FROZEN, not left open.** The field's *current, real, already-shipped*
  behavior (a per-function debug-symbol-count structural cap, live
  specifically for `pure_compute`) is preserved exactly - same three
  values, same verifier check, same error path - with only the misleading
  name and `QuotaKind` taxonomy membership corrected.

**Owner decision, closing what an earlier draft of this document left as
two undecided mechanics:** Option 1 (rename in place) is selected.
Option 2 (delete entirely, relying solely on `sm-format`'s fixed
`MAX_DEBUG_SYMBOLS_PER_FUNCTION = 8192`) is **explicitly rejected for this
checkpoint** - not merely deprioritized. Deleting the check would loosen
`pure_compute`'s admission from 4096 to 8192 debug symbols per function,
a real, disclosed admission-policy change for which no positive
authorization exists today. The safe, decision-scope-respecting side is to
preserve existing semantics exactly; a future, separate, explicitly-scoped
decision may revisit whether `pure_compute`'s extra tightening is worth
keeping, but this checkpoint does not pre-empt that question by silently
loosening admission as a side effect of a vocabulary cleanup.

**Exact frozen mechanic:**

```
RuntimeQuotas::max_trace_entries
    → RuntimeQuotas::max_debug_symbols_per_function

Preserve exact existing values:
    verified_local = 8192
    pure_compute   = 4096
    kernel_bound   = 16384

Preserve exact verifier admission behavior:
    sm-verify::verify_function_code's existing check
    (debug_symbol_count > configured limit
     -> VerificationCode::ResourceLimitExceeded)
    continues unchanged, reading the renamed field.
```

**`max_debug_symbols_per_function` is explicitly NOT an execution-trace
quota.** It is:

- NOT represented in `QuotaKind`
- NOT charged by `sm-vm`
- NOT reported via `RuntimeError::QuotaExceeded`
- a **verifier-side artifact-admission / debug-metadata limit**, enforced
  by `sm-verify`, whose per-profile values are carried in `RuntimeQuotas`
  for compatibility with the existing admission-profile plumbing - not
  because it is architecturally a runtime execution quota.

**Placement is intentionally minimal for `#1760`.** This checkpoint does
**not** move the field into `VerificationLimits` (`sm-verify`'s own
static-analysis-budget struct, used today for `max_work_units`/
`max_state_words`) even though that would be a more architecturally
consistent home for a verifier-owned, decode-adjacent limit. Doing so
would require redesigning the profile-mapping plumbing between
`ExecutionContext`, `RuntimeQuotas`, and verifier-owned limits - a broader
redesign this trace-contract closure does not need to force. That ownership
cleanup, if ever wanted, is a separate, later, explicitly-scoped decision,
not a precondition for closing `#1760`'s false trace-contract claim.

This resolves the apparent tension a reviewer correctly caught: a contract
cannot be simultaneously "frozen" and "implementation-ready" while a real
mechanic remains open. It no longer does - the mechanic above is the
complete, exact specification a future implementation checkpoint executes
without further design choices.

## 6. RESERVED/DEPRECATED — rejected

Same reasoning as `#1761`'s own decision: an inert public field/variant
with baseline numbers is exactly the false-ready condition this checkpoint
exists to remove, and this codebase's dominant struct-literal-with-
`..Default` construction pattern would make any `#[deprecated]` attribute
silently ineffective at most call sites (identical analysis to
`ssf08_1761_constpool_contract_decision.md` §6, not repeated in full here).
**Rejected** for `trace_enabled` and `QuotaKind::TraceEntries`. Not
applicable to `max_trace_entries`, which is not being reserved but
re-scoped to its own real, current meaning.

## 7. Why IMPLEMENT is not a small repair

Enumerated per this checkpoint's own governing brief, to record explicitly
why "just implement it" is an architecture project rather than a Lane 5
closure:

- What is one `TraceEntry`? No existing authority defines this.
- Which opcodes/events create entries - every dispatch, only calls, only
  effects? No existing authority answers this.
- Charged before or after opcode semantics? Undefined.
- Are failing opcodes traced? Undefined.
- Are `CALL`/`RET` traced, and how does that interact with `Steps`/`Calls`
  (#1759)? Undefined - a real design question, not a default.
- Are host effects traced, and do traced effect arguments risk recording
  sensitive host/capability data? Undefined - a real security question a
  bolt-on trace feature must not skip.
- What data does an entry contain - opcode only, operands, register
  values? Undefined.
- Is tracing semantically observable - does `trace_enabled` change
  execution's deterministic result or timing in an observable way? This
  interacts directly with `docs/spec/vm.md`'s Determinism Rule and must
  not be answered by accident.
- Is trace exhaustion fatal, and does it fire before or after the traced
  action - the exact ordering question `#1759` had to freeze carefully for
  `Steps`/`Calls`?
- Who owns/returns the trace buffer, and what is the public retrieval API?
- What is the per-entry memory cost, and how does this behave under
  `no_std` (`sm-runtime-core` supports `no_std`; an unbounded or
  heap-allocating trace buffer is not automatically available there)?

None of these has existing authority to answer. Choosing IMPLEMENT here
would mean this "Lane 5 closure checkpoint" silently became a new feature
design project with no frozen requirements - exactly the shape of scope
creep this whole SSF-08 program has repeatedly guarded against.

## 8. Public API / compatibility impact

`tests/golden_snapshots/public_api/sm_runtime_core_lib.txt` locks in all
three surfaces verbatim: `TraceEntries,`, `pub max_trace_entries: usize,`,
`pub trace_enabled: bool,`. Removing `trace_enabled` and `TraceEntries`,
and renaming/re-scoping `max_trace_entries`, is therefore an intentional,
source-visible narrowing - governed by the same compatibility analysis
already established in `ssf08_1761_constpool_contract_decision.md` §2
(this surface carries no stable label or binding deprecation commitment;
`compatibility_policy_stack.md`'s "boundary/runtime contract changes"
review trigger is satisfied by this decision checkpoint itself). That
analysis is not repeated in full here; it applies identically. **This
decision checkpoint does not touch the golden snapshot.** A future
implementation checkpoint must update it via the same explicit
`SM_UPDATE_PUBLIC_API_SNAPSHOTS=1` mechanism used for `#1761`, and must
expect `public_api_contracts` to fail RED first as the falsification proof
that the guard is load-bearing, exactly as `#1761`'s implementation
checkpoint did.

## 9. Baseline-profile consequence

If `trace_enabled`/`QuotaKind::TraceEntries` are removed, no profile value
changes at all for those two symbols (they have no per-field numeric
representation beyond the boolean default and the taxonomy slot). For
`max_trace_entries`/`max_debug_symbols_per_function`, **the numbers
currently bound something real** (§1.3) - unlike `ConstPool`'s baseline
values, these are not inert vocabulary being deleted; they are the actual
current behavior of a real, if previously mislabeled, check, and §5 froze
their preservation exactly. **The future implementation checkpoint must
not change `verified_local = 8192`, `pure_compute = 4096`, or
`kernel_bound = 16384`.** Option 2 (delete-and-loosen) is rejected by this
document (§5) - it is not an implementation choice left open, and
executing it would itself be a scope violation of this decision.

## 10. `ExecutionConfig` consequence

If `trace_enabled` is removed, `ExecutionConfig` retains exactly `context`
and `quotas`. `docs/spec/vm.md`'s "Execution Contexts" section (currently:
*"The VM consumes `ExecutionConfig`, which binds: `ExecutionContext`,
`RuntimeQuotas`, trace enablement"*) must later stop claiming `ExecutionConfig`
binds trace enablement - a future implementation-checkpoint documentation
update, not made here. Per this checkpoint's own scope guard, no
replacement field (`diagnostics_enabled`, `debug_mode`, `trace_mode`, or
similar) is authorized in this track. A future trace feature, should one
ever be justified, must earn its own contract from scratch, informed by
§7's own enumeration of the real design questions it would need to answer.

## 11. `#1763` consequence

This decision is the last quota-shape input the quota-related slice of
`#1763` needs. If the future implementation checkpoint executes this
decision as specified: `QuotaKind::TraceEntries` disappears from the
active taxonomy. Combined with `#1761` (already merged, `ConstPool`
removed): **neither `ConstPool` nor `TraceEntries` remains in the active
quota taxonomy** after `#1760`'s own implementation lands. `#1763`'s
`QuotaExceeded`-variant question therefore no longer needs to reason about
either inert kind once both implementations land. This document does not
modify `RuntimeTrap` or `RuntimeError`.

## 12. AC4.b consequence

**Decision frozen ≠ AC4.b satisfied.** AC4.b ("inert/deferred resource
concepts are not presented as enforced runtime quotas") remains **NOT
SATISFIED** until a future, separately-authorized implementation
checkpoint actually removes `trace_enabled`/`TraceEntries` and executes
the frozen `max_trace_entries` → `max_debug_symbols_per_function` rename
per §5 - a fully-specified mechanic at this point, not an open choice.
Once that implementation lands - and assuming no further inert quota kind
is discovered in the process (this document makes no such claim; the
`max_trace_entries` finding in §1.3 is itself proof that "assumed inert"
requires verification, not assumption) - AC4.b becomes satisfiable,
pending confirmation at that time.

## 13. Effect on Lane 5

`#1760` remains **OPEN** after this document - a contract was decided, not
implemented. This document does not delete any field, variant, or golden
snapshot entry, and does not touch `#1762`, `#1763`, or `#1902`.

**Wait for explicit owner GO before implementation.**
