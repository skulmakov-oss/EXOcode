# SSF-08 Lane 5 / #1762 — Execution Envelope Provenance Decision

Status: **CONTRACT DECISION ONLY. NO PRODUCTION BEHAVIOR CHANGE. NO AUDIT
FORMAT CHANGE.**
Baseline SHA: `f72a497cb1fcd5498345fa5032c3b362ecebd487` (`main`, confirmed
current via `git rev-parse origin/main`).
Target: `#1762` / `FA-08-004`.
Umbrella: SSF-08 umbrella #1579 remains OPEN.

This document freezes the contract for the relationship between
`ExecutionContext`, `RuntimeQuotas`, `ExecutionConfig`,
`RuntimeSessionDescriptor`, and `AuditSessionMetadata` — specifically, how
an execution/audit record proves which effective quota/admission envelope
actually governed a run. It does not delete, add, or rename anything, does
not touch `#1763`/`#1902`, and does not change any golden snapshot,
archive format constant, or production Rust.

## 1. Baseline / scope

Depends on: `#1759` (Steps/Calls, CLOSED), `#1900` (EffectCalls overflow,
CLOSED), `#1761` (ConstPool, CLOSED), `#1760` (TraceEntries/
`max_debug_symbols_per_function`, CLOSED). All four are now closed on
`main`, which matters directly: this decision's central finding is that
`RuntimeQuotas` is the one real, fully-enforced execution/admission
authority in the system — a claim that would have been only partially true
before those four checkpoints landed (e.g. `Steps`/`Calls` were unenforced
before `#1759`, and `max_debug_symbols_per_function`'s real consumer was
mislabeled before `#1760`). The Lane 5 audit's own pre-existing `#1762`
section (quoted in full in §7 below) explicitly gated a RECORD-style repair
on `#1759`'s disposition being settled first; it now is.

## 2. Fresh implementation evidence

Gathered via two independent, read-only research passes over current
`main` (not the old Lane 5 audit text, not memory from earlier checkpoints
in this program) — one over the execution/enforcement side
(`sm-runtime-core`, `sm-vm`, `sm-verify`, `smc-cli`), one over the
provenance/audit side (`prom-runtime`, `prom-audit`, archive
serialization, public API snapshots). Every claim below traces to an exact
file:line citation from one of those passes or from this document's own
direct reads.

### 2.1 `ExecutionConfig`'s current shape

`crates/sm-runtime-core/src/lib.rs:244-267`, unchanged in shape by this
decision:

```rust
pub struct ExecutionConfig {
    pub context: ExecutionContext,
    pub quotas: RuntimeQuotas,
}

impl ExecutionConfig {
    pub const fn new(context: ExecutionContext, quotas: RuntimeQuotas) -> Self {
        Self { context, quotas }
    }

    pub const fn for_context(context: ExecutionContext) -> Self {
        let quotas = match context {
            ExecutionContext::PureCompute => RuntimeQuotas::pure_compute(),
            ExecutionContext::VerifiedLocal | ExecutionContext::RuleExecution => {
                RuntimeQuotas::verified_local()
            }
            ExecutionContext::KernelBound => RuntimeQuotas::kernel_bound(),
        };
        Self::new(context, quotas)
    }
}
```

Confirms (post-`#1760`) exactly two fields, no `trace_enabled` residue.
`ExecutionConfig::new` performs **no validation** that `quotas` matches
`for_context(context)`'s canonical mapping — it accepts the two
independently, unconditionally.

### 2.2 `sm-vm` never reads `config.context`

A field-access sweep of `crates/sm-vm/src/semcode_vm.rs` (9393 lines) for
`.context`, `config.context`, `vm.context`, `self.context` returns **zero
matches** outside `use` imports and `ExecutionConfig::for_context(...)`
constructor call sites. Every quota enforcement/charge site reads
`vm.config.quotas.max_X` (a concrete, already-resolved numeric value):

```
vm.steps = charge_counter(vm.steps, vm.config.quotas.max_steps, QuotaKind::Steps)?;   // line 2031
enforce_quota(&vm.config.quotas, QuotaKind::Frames, next_depth)?;                      // line 3074
enforce_quota(&vm.config.quotas, QuotaKind::StackDepth, next_depth)?;                  // line 3075
enforce_quota(&vm.config.quotas, QuotaKind::Registers, initial_reg_count)?;            // line 3080
vm.calls = charge_counter(vm.calls, vm.config.quotas.max_calls, QuotaKind::Calls)?;    // line 3095
let limit = vm.config.quotas.max_effect_calls; ... charge_counter(...)                 // lines 3275-3276
```

`ApplicationVmHost`'s own independent `EffectCalls` charge path is
populated at construction as `quotas: vm.config.quotas` (line 886) — a
plain value copy, again with no `ExecutionContext` involved.

**`ExecutionContext` has zero runtime effect on `sm-vm`'s behavior.** It is
carried on `VM.config` but never read to branch, gate, or select anything
during execution. `RuntimeQuotas` is the sole enforcement authority.

### 2.3 `sm-verify` never sees `ExecutionContext` at all

A workspace grep for `ExecutionContext` in `crates/sm-verify/src/lib.rs`
(12356 lines) returns **zero matches** — the type is never imported.
Every public verify entrypoint takes `RuntimeQuotas` (and, for the
`_and_limits` variant, `VerificationLimits`) directly, never a context:

```rust
pub fn verify_semcode_token(bytes: &[u8]) -> Result<VerifiedSemCode<'_>, RejectReport>              // line 641, hardcodes RuntimeQuotas::verified_local()
pub fn verify_semcode_token_with_quotas(bytes: &[u8], quotas: RuntimeQuotas) -> ...                  // line 663
pub fn verify_semcode_token_with_quotas_and_limits(bytes: &[u8], quotas: RuntimeQuotas, limits: VerificationLimits) -> ...   // line 686
```

The `SymbolTable` admission check (`quotas.max_symbol_table`, line 835)
and the `#1760` debug-symbol admission check
(`quotas.max_debug_symbols_per_function`, line 1359) both read directly
from the `quotas` parameter passed at the call boundary — again, no
`ExecutionContext` anywhere in this crate.

### 2.4 `smc-cli` only ever constructs canonical configs

`crates/smc-cli/src/app.rs` (3410 lines): `cmd_run` (line 2933) and
`cmd_test` (line 3107) both call `ExecutionConfig::for_context(ExecutionContext::VerifiedLocal)`
— canonical. `cmd_verify` (line 3057) calls `sm_verify::verify_semcode`,
which never constructs an `ExecutionConfig` at all (it hardcodes
`RuntimeQuotas::verified_local()` internally, per §2.3). `cmd_check` has
no quota involvement whatsoever. A grep for any CLI flag resembling
`--quota`/`--limit`/`max_steps` etc. across `app.rs` returns **zero
matches**. **No production, non-test call site anywhere in the workspace
constructs an `ExecutionConfig` with quotas that diverge from
`for_context`'s canonical mapping.**

### 2.5 The one, legitimate, load-bearing custom-envelope consumer

Exactly three call sites workspace-wide construct `ExecutionConfig::new`
with non-canonical quotas, all in `tests/ssf04_effect_quota.rs` (lines
121, 195, 221), all identical in shape:

```rust
let quotas = RuntimeQuotas { max_effect_calls, ..RuntimeQuotas::verified_local() };
...
ExecutionConfig::new(ExecutionContext::VerifiedLocal, quotas)
```

Every override touches only `max_effect_calls` (values as low as `1`),
every other field stays at `verified_local()`'s canonical value via
struct-update syntax. This is the SSF-04 effect-quota enforcement
qualification seam — the exact test infrastructure `#1900`'s own mutation
proofs exercised to make `EffectCalls` exhaustion reachable in a handful
of calls instead of the real baseline's 1024 (the same "shrink the natural
bound so the mutation fails fast instead of hanging" lesson `#1759`'s own
C2 correction already established for `Steps`). A second, structurally
identical pattern exists in `crates/sm-vm/src/semcode_vm.rs`'s own
`#[cfg(test)] mod tests` (~15 sites, e.g. lines 4511, 6191, 6368...),
mutating `config.quotas.max_X` after `for_context` construction for the
same reason — testing boundary/exhaustion behavior at a reachable limit.

**Both patterns are test-only, both are legitimate, and both would need to
be broken or specially exempted by any disposition that forbids
non-canonical `context`/`quotas` pairings outright.**

### 2.6 `RuntimeSessionDescriptor` and `AuditSessionMetadata` currently drop the envelope on the floor

`crates/prom-runtime/src/lib.rs:17-22`:

```rust
pub struct RuntimeSessionDescriptor {
    pub context: ExecutionContext,
    pub capability_manifest: CapabilityManifestMetadata,
    pub gate_registry_bound: bool,
}
```

`crates/prom-audit/src/lib.rs:19-24` — an identical three-field shape,
and `prom-audit` does not even import `RuntimeQuotas` (confirmed: no such
`use` anywhere in the crate — it is structurally impossible for
`AuditSessionMetadata` to carry a quota value today).

Every constructor of `RuntimeSessionDescriptor`
(`ExecutionSession::new`, `GateExecutionSession::new`, both in
`crates/prom-runtime/src/lib.rs`) receives the **full** `ExecutionConfig`
(context + quotas) as a parameter — the quota half is used internally for
execution (`self.config.quotas`, e.g. line 372) but is never copied into
`self.descriptor`. `build_audit_session` (`prom-runtime`, lines 97-103)
then copies `context`/`capability_manifest`/`gate_registry_bound` from the
descriptor into `AuditSessionMetadata` — quotas were already gone by that
point, so none can flow through.

### 2.7 The canonical audit archive text carries zero quota data

`AuditReplayArchive::to_canonical_text` (`crates/prom-audit/src/lib.rs:178-222`)
emits, on the `session` line, exactly four values after the `"session"`
tag:

```rust
out.push_str("session\t");
out.push_str(display_execution_context(self.session.context));
out.push('\t');
out.push_str(&escape_archive_field(&self.session.capability_manifest.schema));
out.push('\t');
out.push_str(display_manifest_version(self.session.capability_manifest.version));
out.push('\t');
out.push_str(if self.session.gate_registry_bound { "true" } else { "false" });
out.push('\n');
```

`from_canonical_text`'s mirror-image parse (lines 233-255) requires
exactly 5 tokens (`"session"` + the 4 values above) and rejects anything
else. No `AuditEventKind` variant carries quota data either. **The
canonical archive text structurally cannot record, and today does not
record, which `RuntimeQuotas` values actually governed a session** — only
the `ExecutionContext` display string.

### 2.8 Format version discipline (existing, to be preserved)

```rust
pub const AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION: u32 = 1;              // line 83
pub const MULTI_SESSION_REPLAY_ARCHIVE_FORMAT_VERSION: u32 = 1;      // line 84
```

Both are enforced by a strict equality-or-reject check with a typed error
(`AuditReplayArchiveFormatError` / `MultiSessionReplayArchiveFormatError`,
lines 233-239 and 393-398) — no forward/backward-compat branch exists
anywhere in the format today. `MultiSessionReplayArchive::from_canonical_text`
genuinely delegates each embedded session's parse to
`AuditReplayArchive::from_canonical_text` (confirmed by direct read,
`crates/prom-audit/src/lib.rs:464`: `let archive = AuditReplayArchive::from_canonical_text(&archive_text)?;`)
rather than duplicating the session-line parsing inline — the outer
multi-session wire shape (magic, outer version, session count,
`session\t<ordinal>\t<line-count>` framing, `archive\t`-prefixed embedded
lines) is structurally independent of the inner per-session `session`
line's own token shape.

### 2.9 No test anywhere asserts on quota provenance

`tests/prometheus_audit.rs` and `prom-audit`'s own `#[cfg(test)]` module
assert only on `context`, `capability_manifest`, `gate_registry_bound`,
and event contents/counts — never on a quota value. There is no existing
test this decision's selected disposition needs to preserve compatibility
with beyond the shapes already quoted above.

## 3. Existing public contract

`tests/golden_snapshots/public_api/`:

- `sm_runtime_core_lib.txt:87-92` — `ExecutionConfig { context, quotas }`,
  confirming no `trace_enabled` residue.
- `prom_runtime_lib.txt:2-6` — `RuntimeSessionDescriptor { context,
  capability_manifest, gate_registry_bound }`.
- `prom_audit_lib.txt:5-9, 16-20, 23-28, 33-36` — `AuditSessionMetadata`,
  `ReplayMetadata`, `AuditReplayArchive`, `MultiSessionReplayArchive`, all
  matching the struct shapes quoted in §2.6/§2.7 exactly.

None currently expose any quota-derived field. This is the exact surface a
RECORD-style disposition would need to widen (see §11).

## 4. ExecutionContext meaning — frozen

`ExecutionContext` is, today, **neither** an exact/enforced policy
identity **nor** something read at execution time. §2.2/§2.3 prove neither
`sm-vm` nor `sm-verify` ever inspects it. §2.4/§2.5 prove production code
only ever selects it through the canonical `for_context` mapping, while a
legitimate test seam explicitly decouples it. It is therefore frozen as:

> **`ExecutionContext` is a construction-time baseline/default selector
> (via `for_context`) and an audit/provenance execution-class label. It is
> not proof, by itself, of which `RuntimeQuotas` values actually governed
> a given execution.**

This is a description of current, evidenced behavior, not a new promise.

## 5. RuntimeQuotas authority — frozen

§2.2 and §2.3 together prove `RuntimeQuotas` — specifically, the resolved
numeric values sitting in `config.quotas` at the moment `sm-vm`/
`sm-verify` consume it — is the **sole real effective execution/admission
envelope** in the system today. Frozen:

> **`RuntimeQuotas` (as carried in `ExecutionConfig.quotas`) is the actual
> effective execution/admission authority. `ExecutionContext` is not.**

## 6. Custom-envelope policy — frozen

`ExecutionConfig::new(context, quotas)` remains a supported, public
constructor accepting `context` and `quotas` independently, with **no
new validation added by this decision**. Reasoning:

- §2.5 shows the only current consumer of non-canonical pairings is a
  legitimate, load-bearing test seam (SSF-04 effect-quota qualification,
  plus `sm-vm`'s own unit-test quota-mutation pattern) — both would break
  under any disposition that rejects mismatched pairings outright.
- No cited authority (§3 of `docs/spec/quotas.md`, quoted in full in §8
  below) forbids custom envelopes; it requires only that they "not
  weaken the core safety contract *silently*" — an explicit disclosure
  requirement, not a prohibition (see §8's precise reading).
- Restricting custom envelopes to "stricter than baseline only" was
  considered and rejected as an *additional*, currently-unrequired
  invariant: no existing consumer has ever needed a looser-than-baseline
  envelope (100% of current custom usage tightens `max_effect_calls`),
  but defining "stricter" as a total order across all eight
  `RuntimeQuotas` fields is genuinely ill-posed — fields can move in
  different directions with no single field-by-field comparison giving a
  clean answer, and `max_debug_symbols_per_function` is verifier-admission
  policy, not a runtime quota (frozen by `#1760`), so folding it into a
  "safety-contract strictness" ordering would misclassify it right back
  into the taxonomy `#1760` deliberately took it out of. Inventing this
  ordering is unauthorized new scope this checkpoint has no mandate for.
- Implementing any such restriction would itself require production
  validation code, which this decision-only checkpoint is explicitly
  forbidden from adding (§20 scope guard of the governing brief).

## 7. Current provenance loss — the actual defect

The Lane 5 audit's own pre-existing `#1762` section (`ssf08_lane5_resource_failure_closure_audit.md`,
its "## 6. #1762" heading, quoted here in full since a fresh read is
required rather than trusting an old summary) already reached this
conclusion independently, before this document existed:

> Fresh trace. `ExecutionConfig::new(context, quotas)` (line 257) performs
> no validation that `quotas` matches the baseline
> `ExecutionContext::for_context(context)` would have produced - confirmed
> by direct reading, not inference. [...] `RuntimeSessionDescriptor`
> (line 18) has exactly three fields [...] no `quotas` field at all [...]
> `AuditSessionMetadata` likewise has `context: ExecutionContext` and no
> quota field. `tests/ssf04_effect_quota.rs` confirms
> `ExecutionConfig::new(ExecutionContext::VerifiedLocal, <custom quotas>)`
> is an actively-used, legitimate test pattern today - not a hypothetical
> misuse this audit invented.
>
> [...] **Disposition: NEW DECISION REQUIRED BEFORE REPAIR.**
> [...] Any RECORD-style repair for #1762 should land only after #1759's
> own disposition is settled, or it risks recording quota values that are
> honest about *configuration* but dishonest by omission about
> *enforcement*.

`#1759` is now CLOSED (`Steps`/`Calls` are genuinely enforced), and
`#1900`/`#1761`/`#1760` have since closed the remaining enforcement gaps
(`EffectCalls` both sites, `ConstPool` removed, `TraceEntries` removed/
`max_debug_symbols_per_function` correctly re-scoped). The gate the Lane 5
audit itself set on a RECORD-style repair is now satisfied: recording the
effective `RuntimeQuotas` today means recording values that genuinely,
fully govern (or, for `max_debug_symbols_per_function`, genuinely admit)
execution — not aspirational vocabulary.

**The defect is precisely this document's headline claim**: two sessions
under the identical `ExecutionContext::VerifiedLocal` label — one running
under the canonical baseline, one running with `max_effect_calls` reduced
to `1` — produce byte-identical `session` lines in the canonical audit
archive. A reader of the archive alone cannot determine which envelope
actually governed either run.

## 8. "Must not weaken silently" — precise meaning, frozen

`docs/spec/quotas.md`'s "Quota model rule" (lines 12-18), quoted verbatim:

```
- `sm-runtime-core` defines quota taxonomy and baseline profiles
- `sm-vm` enforces quotas during execution, except `SymbolTable`, which
  `sm-verify` enforces statically, pre-execution, at admission
- higher integration layers may choose context-specific quota envelopes, but
  must not weaken the core safety contract silently
```

Of the three meanings the governing brief poses:

- (A) weaker envelopes are forbidden outright
- (B) weaker envelopes are permitted only when represented explicitly in
  provenance/audit
- (C) baselines are defaults, not safety minima; the only invariant is
  deterministic boundedness

**Frozen: (B).** The operative word in the existing text is *silently* —
the sentence already contrasts "may choose context-specific quota
envelopes" (permission) against "must not... silently" (a qualifier on
*how*, not *whether*). Reading it as (A) would contradict the same
document's own explicit permission clause in the same sentence. Reading it
as (C) drops the "must not weaken silently" clause entirely, leaving it
without content. (B) is the only reading that gives every word in the
existing sentence work to do, and it is exactly what §6's frozen
custom-envelope policy, combined with this document's selected disposition
(§10), operationalizes: envelopes may diverge from baseline; they may not
diverge *invisibly*.

No per-field "stricter than baseline" ordering is frozen (§6) — "not
silently" is satisfied by truthful recording of whatever envelope was
actually used, not by restricting which envelopes are permitted.

## 9. Candidate A — exact canonical context

`config.quotas == RuntimeQuotas::<profile>()` for `config.context` must
hold exactly for any auditable/production execution; custom envelopes
rejected or isolated to a non-authoritative path.

**Falsified.** §2.5 shows exactly three production/test call sites
(effect-quota qualification) and ~15 `sm-vm` unit-test sites depend on
non-canonical pairings today, all legitimate, none hypothetical. Forcing
exactness would either delete real test coverage (the effect-quota
exhaustion tests cannot use the real 1024-call baseline without becoming
impractically slow, mirroring the exact lesson `#1759`'s C2 correction
already learned for `Steps`) or require inventing an unauthorized
"privileged bypass" concept with no current basis. It would also require
new validation/rejection production code — forbidden in this
decision-only checkpoint, and more fundamentally not evidenced as
necessary: nothing in the current codebase has ever exploited the
existing flexibility to *hide* a weaker-than-claimed envelope; the actual
defect (§7) is provenance honesty, not envelope restriction. **Rejected.**

## 10. Candidate B — record effective envelope

`ExecutionContext` remains a baseline/default/class label (§4).
`RuntimeQuotas` remains the actual effective policy (§5). Custom envelopes
remain first-class (§6). `RuntimeSessionDescriptor` and
`AuditSessionMetadata` must preserve the effective `RuntimeQuotas` that
actually governed verification/execution; audit/replay serialization must
carry it.

**Selected.** This is the disposition the governing brief's own §7 named
as the hypothesis to falsify. §9 (Candidate A) and §12 (Candidate D) below
record the falsification attempts against it; neither defeats it. The
strongest argument found against it (§14, Case 4) shows it does not by
itself prevent semantically confusing pairings (e.g. `PureCompute` paired
with a looser-than-`pure_compute` envelope) — but it does not need to:
this checkpoint's mandate is truthful provenance, not envelope
restriction (§6), and Candidate B makes every pairing, confusing or not,
fully visible rather than inventing a new restriction with no current
authority behind it.

## 11. Candidate C — baseline + explicit override

Preserve `ExecutionContext` plus a distinct explicit override/custom
identity (e.g. `origin: Baseline | Override`).

**Falsified as unnecessary, not as wrong.** Once the full effective
`RuntimeQuotas` values are recorded (Candidate B), "was this envelope
customized" is a *derived* fact — any reader can compare the recorded
values against `RuntimeQuotas::verified_local()`/`pure_compute()`/
`kernel_bound()` (the same public, stable constructors used everywhere
else in the codebase) for the recorded `context`. An explicit `origin` tag
would duplicate information the raw values already carry, adding a second
representation of the same fact that could itself drift out of sync with
the values it labels. Per the governing brief's own instruction to "judge
whether the extra identity is actually needed" — it is not. **Rejected as
redundant.**

## 12. Candidate D — documentation-only narrowing

Keep implementation unchanged; state in prose that `context` is
descriptive only; audit records no effective quotas.

**Falsified.** §2.6/§2.7 prove `RuntimeSessionDescriptor`/
`AuditSessionMetadata`/`AuditReplayArchive` are *structurally* incapable
of distinguishing the two sessions in §7's headline example — this is not
a prose-overclaim problem that a documentation fix resolves, it is a
representational gap. `AC4.c` ("`ExecutionContext`/provenance does not
overstate the quota envelope that actually governed execution") requires
the archive to be able to *establish* what governed a run, not merely to
stop *claiming* it can. Narrowing the docs alone would leave replay/audit
permanently blind to the actual envelope even after `#1759`/`#1900`/
`#1761`/`#1760` finally made `RuntimeQuotas` a fully truthful, enforced
authority — a regression in what the audit trail is capable of attesting
to, at the exact moment the underlying enforcement became trustworthy
enough to be worth attesting to. **Rejected**, per the governing brief's
own instruction not to select D unless evidence shows AC4.c requires
less — evidence shows the opposite.

## 13. RuleExecution — frozen classification

`RuleExecution` maps to the *same* `RuntimeQuotas::verified_local()` call
as `VerifiedLocal`, in the same match arm (§2.1). It also has its own
independent audit-string round-trip (`display_execution_context`/
`parse_execution_context`, `crates/prom-audit/src/lib.rs:756-775`,
`RuleExecution => "rule-execution"`), used purely for rendering — no
branching on the value affects execution or quotas anywhere.

Frozen: **`RuleExecution` is a distinct execution-class/audit label that
currently shares `VerifiedLocal`'s default quota envelope — it is not an
alias to be collapsed, and shared *values* do not imply shared
*identity*.** `PureCompute` and `verified_local` already share several
field values (e.g. `max_steps = 100_000` in both) without that making
`PureCompute` an alias of `VerifiedLocal`. Do not claim `RuntimeQuotas`
values alone uniquely determine `ExecutionContext` (Case 5, §14) —
Candidate B's design deliberately keeps `context` and `quotas` as two
separate recorded fields precisely so a reader is never forced to infer
one from the other.

## 14. Falsification matrix

| Case | Config | Admitted? | Effective authority | Recorded provenance (post-implementation) | Context alone truthful? | Archive consequence |
|---|---|---|---|---|---|---|
| 1 | `VerifiedLocal` + `verified_local` baseline | Yes | `verified_local()` values | `context: VerifiedLocal, quotas: {verified_local() values}` | Yes (matches) | v2 session line carries values equal to baseline |
| 2 | `VerifiedLocal` + stricter `max_effect_calls` (e.g. 1) | Yes (today's `ssf04_effect_quota.rs` pattern) | The custom quotas — `sm-vm` charges against 1, not 1024 | `context: VerifiedLocal, quotas: {..verified_local(), max_effect_calls: 1}` | **No** — label alone implies 1024 | v2 carries the true value 1; exactly the gap this decision closes |
| 3 | `VerifiedLocal` + looser `max_steps` (e.g. 500 000) | Yes, per §6's frozen policy | The looser quotas genuinely govern `Steps` charging | `context: VerifiedLocal, quotas: {..verified_local(), max_steps: 500000}` | **No**, resolved the same way as Case 2 | v2 carries the true, looser value — visibly, not silently |
| 4 | `PureCompute` + `verified_local` envelope (`max_effect_calls: 1024`) | Yes — nothing in the type system forbids it | `verified_local()` values, including `max_effect_calls: 1024` under a context conventionally meaning "no host effects" | `context: PureCompute, quotas: {verified_local() values}` | **No** | v2 makes this confusing-but-real pairing fully visible; Candidate B does not forbid it (§6), only hides it worse than today if left unrecorded — see §10's own acknowledgment of this as the strongest counter-argument |
| 5 | `RuleExecution` + `verified_local` baseline | Yes (canonical) | `verified_local()` values | `context: RuleExecution, quotas: {verified_local() values}` | Yes for enforcement; `context` (not `quotas`) is what distinguishes this from Case 1 | v2 session line is value-identical to Case 1's except for the `context` field — expected, per §13 |
| 6 | `KernelBound` + custom stricter envelope | Yes | The custom, stricter values | `context: KernelBound, quotas: {..kernel_bound(), <tightened fields>}` | **No** | v2 carries the true, stricter values |

## 15. Audit/replay serialization consequence — exact future mechanic (FROZEN)

This section is binding on the future implementation checkpoint; nothing
here is left as an open choice.

**Fields to add**, mirroring `ExecutionConfig`'s own naming exactly (no
new type introduced):

```
RuntimeSessionDescriptor.quotas: RuntimeQuotas   (crates/prom-runtime/src/lib.rs)
AuditSessionMetadata.quotas:     RuntimeQuotas   (crates/prom-audit/src/lib.rs)
```

**Ownership and copy path** — both structs gain the field; both are
populated directly from the same immutable `ExecutionConfig` already
threaded through construction, never re-derived and never reconstructed
after the fact (binding trust rule, see §16):

- `ExecutionSession::new`/`GateExecutionSession::new`
  (`crates/prom-runtime/src/lib.rs`) already receive `config:
  ExecutionConfig` as a parameter — add `quotas: config.quotas` to the
  `RuntimeSessionDescriptor` literal alongside the existing `context:
  config.context`.
- `build_audit_session` (`crates/prom-runtime/src/lib.rs:97-103`) already
  copies `context`/`capability_manifest`/`gate_registry_bound` from
  `descriptor` — add `quotas: descriptor.quotas` to the
  `AuditSessionMetadata` literal, propagating the same value one hop
  further, never re-reading `ExecutionConfig` a second time.

**Representation: full raw values, not a digest, delta, or boolean.** A
`custom: bool` flag would tell a reader *that* something diverged without
saying *what* (insufficient per the governing brief's own §12). A digest/
hash would let a reader confirm identity against a known-good preimage but
not reconstruct or display the envelope on its own, and no current
architecture need justifies adding cryptographic machinery for this. Full
`RuntimeQuotas` values are the simplest representation that preserves
deterministic truth and require no new machinery — reject both
alternatives.

**Canonical text wire change** — append the eight `RuntimeQuotas` fields,
in the same order as the struct's own field declaration, as additional
tab-separated tokens on the existing `session` line, after the four
existing values:

```
session\t<context>\t<schema>\t<manifest-version>\t<gate_registry_bound>\t
    <max_steps>\t<max_calls>\t<max_stack_depth>\t<max_frames>\t
    <max_registers>\t<max_symbol_table>\t<max_effect_calls>\t
    <max_debug_symbols_per_function>
```

13 tokens total (the `"session"` tag + 12 values), up from today's 5.
`from_canonical_text`'s existing `session_parts.len() != 5` guard becomes
`!= 13`. Numeric encoding: plain decimal `usize::to_string()`, matching
the existing convention used for `event_count`/`last_event_id` elsewhere
in the same function — no hex, no padding, no new encoding scheme.

## 16. Compatibility/versioning consequence — exact future mechanic (FROZEN)

**`AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION` must bump, `1` → `2`.** The
`session` line's wire shape changes (5 tokens → 13); the archive's own
existing discipline (§2.8) is a strict equality-or-reject check with no
forward/backward-compat branch, so a shape change requires a version bump
under that same discipline — not a new policy invented for this decision.

**`MULTI_SESSION_REPLAY_ARCHIVE_FORMAT_VERSION` does NOT need to bump.**
Verified by direct read (§2.8, `crates/prom-audit/src/lib.rs:464`):
`MultiSessionReplayArchive::from_canonical_text` genuinely delegates each
embedded session's parse to `AuditReplayArchive::from_canonical_text`,
which independently enforces `AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION`. The
outer multi-session wire shape (magic, outer version, session count,
per-session ordinal/line-count framing, `archive\t`-prefixed embedding)
does not change under this decision — only the inner per-session
`session` line does, and the inner parser already independently gates on
its own version. A multi-session archive built from new-shape inner
sessions still passes the unchanged outer version check; an *old* (pre-
this-change) multi-session reader encountering such an archive will still
correctly fail closed, because its own bundled `AuditReplayArchive::from_canonical_text`
will reject the new inner version 2 against its compiled-in expectation
of 1 — the existing discipline extends correctly to this case without any
further design work.

**Legacy v1 archives: new readers reject them; old readers keep working
on them.** No forward/backward-compat parsing is introduced. This mirrors
100% of existing precedent in this format (no version has ever supported
multi-version reading) and avoids inventing new machinery (e.g.
`Option<RuntimeQuotas>` "legacy, envelope unknown" semantics) beyond what
the codebase's own established convention already provides. A v2-aware
reader given a v1 archive gets the same `"unsupported archive format
version 1; expected 2"` rejection any version mismatch already produces
today — it never silently reinterprets a legacy archive as carrying quota
information it never stored (binding per §17's trust rule). Old (pre-#1762-
implementation) reader code is untouched and continues to correctly read
old v1 archives, since nothing about v1 parsing changes.

**New writers emit only the new version.** Once implemented, every new
archive write uses `format_version = 2`; no dual-format writer is
authorized or needed.

## 17. Public API consequence

`RuntimeSessionDescriptor` gains `pub quotas: RuntimeQuotas`;
`AuditSessionMetadata` gains `pub quotas: RuntimeQuotas`. This is an
intentional, source-visible widening of `prom_runtime_lib.txt` and
`prom_audit_lib.txt`. `sm_runtime_core_lib.txt` is untouched — `ExecutionConfig`
itself does not change shape. **This decision does not touch any golden
snapshot.** A future implementation checkpoint must reproduce the exact
`#1760`/`#1761` evidence discipline: run `cargo test --test
public_api_contracts` first against the stale golden (must fail RED,
diff isolated to exactly the two added fields), then regenerate via
`SM_UPDATE_PUBLIC_API_SNAPSHOTS=1`, review the diff manually, confirm
GREEN.

## 18. Trust rule — binding on implementation

Record the authority actually used; never reconstruct a nicer-looking
policy after the fact. Concretely:

- metadata is captured from the same immutable `ExecutionConfig` instance
  already passed to `sm-verify`/`sm-vm` for the session in question — not
  a freshly re-derived `RuntimeQuotas::<profile>()` call keyed off
  `context`
- no fallback to a context-derived baseline when the actual `quotas` value
  is available (which it always is, per §15's copy path)
- no independently reconstructed "expected quotas" anywhere in the
  recording path

## 19. Selected disposition

**RECORD_EFFECTIVE** (Candidate B, §10), with the meaning of "must not
weaken silently" frozen as (B) (§8): explicit-in-provenance, not
prohibited. `ExecutionContext` remains a baseline-selector/audit-class
label (§4); `RuntimeQuotas` remains the sole effective authority (§5);
custom envelopes remain fully permitted, unrestricted by any new
stricter-than-baseline rule (§6); `RuntimeSessionDescriptor`/
`AuditSessionMetadata` must record the full effective `RuntimeQuotas`
(§15); the archive format version must bump accordingly (§16).

## 20. Rejected alternatives

- **Candidate A (exact canonical context)** — §9. Would break the
  legitimate `ssf04_effect_quota.rs` seam and `sm-vm`'s own unit-test
  quota-mutation pattern for no evidenced safety benefit; the actual
  defect is provenance honesty, not envelope restriction.
- **Candidate C (baseline + explicit override identity)** — §11.
  Redundant once full effective values are recorded; a derived fact does
  not need its own duplicated, driftable representation.
- **Candidate D (documentation-only narrowing)** — §12. Leaves
  replay/audit structurally unable to establish the actual execution
  envelope, which is exactly what AC4.c requires it to be able to do.
- **Stricter-than-baseline-only custom-envelope restriction** — §6.
  Unauthorized new scope; no current evidence requires it; ill-posed
  across heterogeneous fields including the non-quota
  `max_debug_symbols_per_function`.

## 21. AC4.c consequence

**Not satisfied by this document alone** — this is a decision, not an
implementation. `AC4.c` ("`ExecutionContext`/provenance does not overstate
the quota envelope that actually governed execution") becomes
**satisfiable** once a future, separately-authorized implementation
checkpoint executes §15-§17's frozen mechanic: adds `quotas:
RuntimeQuotas` to both provenance structs, copies it from the same
`ExecutionConfig` already used for execution (never re-derived), bumps
`AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION` to 2, updates the two golden
snapshots, and updates `docs/spec/quotas.md`'s "Contract rule"/"must not
weaken silently" language to state the (B) reading explicitly (§8) rather
than leave it implicit. None of that is done here.

## 22. #1763 consequence

Not implemented here; `RuntimeTrap`/`RuntimeError` untouched. Recorded
per the governing brief: once `#1762`'s context/envelope provenance
contract is implemented, `AC4.c` can be evaluated independently, and
`#1763` can then reconcile `RuntimeTrap`/`RuntimeError`/verifier
rejection/quota exhaustion against a final, truthful resource/provenance
model rather than one still missing effective-envelope evidence.

## 23. #1902 non-dependency

`#1902` (the `snake_learning.sm` `VerifiedLocal` `Steps` overflow) is
independent of this decision and was not used, directly or indirectly, to
justify any part of it — no `VerifiedLocal` baseline value changes here,
no CLI quota-override flag is introduced or implied by §6's custom-envelope
policy (which describes the existing, already-public `ExecutionConfig::new`
seam, not a new CLI-facing mechanism), and nothing in this document
touches `#1762` policy on the basis of `#1902`'s residual. `#1902` remains
untouched, tracked separately.

## 24. Implementation boundary

This document freezes every implementation-significant choice for a
RECORD-style disposition per the governing brief's own §19 checklist:

- exact metadata fields: `quotas: RuntimeQuotas` on both
  `RuntimeSessionDescriptor` and `AuditSessionMetadata` (§15)
- which object owns them: both, propagated `ExecutionConfig` →
  `RuntimeSessionDescriptor` → `AuditSessionMetadata` (§15, §18)
- where values are copied from: the same immutable `ExecutionConfig`
  already passed to session construction, never re-derived (§18)
- whether full effective quotas are recorded: yes, all eight fields, not
  a digest/delta/boolean (§15)
- archive wire/version mechanic: exact token order and count frozen
  (§15); `AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION` 1→2,
  `MULTI_SESSION_REPLAY_ARCHIVE_FORMAT_VERSION` unchanged, both with
  reasoning traced to the actual delegation code (§16)
- legacy archive behavior: v1 archives rejected by v2 readers, remain
  readable only by v1 reader code, no silent backfill (§16)
- public API change: both golden snapshots widen; RED→GREEN discipline
  specified (§17)
- docs requiring update: `docs/spec/quotas.md`'s "Contract rule" language
  (§8, §21)

**Nothing above is left open.** A future implementation checkpoint may
proceed directly from this document without a further design pass.

**Wait for owner GO before implementation.**
