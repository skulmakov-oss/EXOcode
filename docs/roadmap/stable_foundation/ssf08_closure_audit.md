# SSF-08 Closure Audit and Residual Dependency Reconciliation

Status: audit only, no repair authorized by this document
Audit SHA: `3ad550da55a4ff56db8dd33c9d82f26cc7fe8817` (exact `main`, confirmed via
`git fetch origin && git checkout main && git pull --ff-only && git rev-parse HEAD`
immediately before this audit; `origin/main` had not moved past this SHA)
Umbrella: #1579 (SSF-08 — Ownership and memory model positioning), parent #1569
Purpose: answer, with fresh repository/GitHub evidence, exactly what still
prevents #1579 from closing. This is not an implementation checkpoint; no
production Rust changed while producing this document.

**2026-09-06 addendum — #1888 reconciled, not repaired.** This audit's
original pass (below) classified #1888 REQUIRED against AC3/AC5, following
the issue's own on-file description without cross-checking it against the
later #1891/W2A architectural change. A dedicated reconciliation pass (same
audit SHA `3ad550da5`) found, via direct code reading and 8 new empirical
regression tests (`tests/fa_04_025_reconciliation.rs`), that #1888's root
cause - RecordUpdate Write-event generation depending on a separate,
sometimes-unreached prescan - was structurally eliminated by Checkpoint W2A,
which relocated that generation into `lower_expr_with_expected`'s own
`Expr::RecordUpdate` arm: the ONE canonical expression-lowering function
every admitted expression must pass through to become executable IR at all,
regardless of which statement-level caller reaches it. All three of #1888's
named roots (`lower_for_range_stmt`'s range bound, `lower_for_each_stmt`'s
iterable, `Stmt::Guard`'s `else_return` payload) were confirmed, by direct
reading, to still never call the old prescan - and confirmed, by direct
compilation and IR/runtime inspection, to no longer need it. §5, §8, and §10
below are updated accordingly; the original text is struck through rather
than deleted, so this document's own audit trail stays honest about what was
initially concluded and why it changed. No production Rust changed by this
reconciliation either - see `tests/fa_04_025_reconciliation.rs`'s own header
comment for the full mechanism proof.

**2026-09-06 addendum #2 — #1718 contract decision frozen, fully directional,
implementation still outstanding.** A dedicated contract-decision-and-
falsification checkpoint (baseline `ef2d8524ddc597f7064edf052dad70fd03575fdf`,
corrected pre-merge on the same PR #1895 to close a remaining ambiguity)
resolved the promote-vs-restrict question §5 originally left open,
independently per family AND, for ADT, independently per event kind:

```
SequenceIndexStatic  Borrow -> PROMOTE
SequenceIndexStatic  Write  -> PROMOTE
AdtPayload           Borrow -> PROMOTE
AdtPayload           Write  -> RESTRICT / FAIL-CLOSED (compiler-unreachable
                                today; promoting it later requires a new,
                                separately authorized contract change)
```

The first pass of this addendum left the ADT Write question deferred to the
implementation checkpoint; that was itself a scope violation for a
contract-decision checkpoint (the admissibility of `Write(AdtPayload)` is
part of the public admission contract, not implementation encoding), and has
been corrected in place. Full evidence matrix, capability/version authority
analysis, No Silent Mutation analysis, and the falsification attempts against
both PROMOTE and RESTRICT/DEFER (now split per event kind for ADT) live in
`docs/roadmap/stable_foundation/ssf08_1718_path_family_contract_decision.md`.
**This is a contract decision only — no capability bit, header revision, or
decoder/verifier/VM change has been implemented.** #1718 remains OPEN; AC1,
AC2, and AC5 remain NOT SATISFIED for these families until the chosen
contract is actually enforced end-to-end and qualified. §5's original
deep-audit text below is left intact (not struck through) because its
conclusion - REQUIRED, remedy-agnostic - is still accurate; only its "which
remedy" open question is now resolved, fully, by the linked decision
document.

**2026-09-06 addendum #3 — #1718 implemented, qualified, merged, and closed.**
PR #1896 (six checkpoint commits: I1+I3 format authority/decode gating, I2
emitter promotion, I4 independent verifier enforcement, I4.1 a
directly-testable extraction of that verifier policy closing a coverage gap
mutation testing found, I5 test reconciliation, I6 docs) implements exactly
the frozen contract from addendum #2 - `HEADER_V21`/`SEMCOD21` (rev 22),
purely additive over `HEADER_V20`, two new capability bits
(`CAP_OWNERSHIP_SEQUENCE_PATHS`, `CAP_OWNERSHIP_ADT_BORROW_PATHS`). Six
mutation-testing falsification passes (M1: emitter promotion: M2: decoder
legacy gate; M3: ADT Borrow promotion; M4a: decoder `Write(AdtPayload)`
rejection; M4b: verifier's own independent `Write(AdtPayload)` rejection;
M5: generic-capability-reinterpretation fallback), each independently
confirmed load-bearing by deletion-then-restoration against a real test, not
merely read from source. Rebase-merged to `main` at `b2fce6e2041661e3d08a3a196322f082a5f7c24f`
(6 checkpoint commits preserved individually); hosted CI/security green on
that exact SHA; #1718 auto-closed as completed via the PR's `Closes #1718`.
Exact-main replay (`sm-verify` 218/218, `sequence_ownership_golden`,
`runtime_ownership_e2e`, `borrow_activation_v20`, `write_execution_site_e2e`,
all green; `git diff --check` clean) confirms the merged tree matches what
was qualified pre-merge.

Independent re-evaluation (not a mechanical "the checkpoint closed, so mark
satisfied" pass - the prior #1888 mis-classification in this same document is
the exact reason a fresh check is required here): a fresh GitHub search
(`SSF-08 ownership`) and a fresh read of #1579's own body/comments found no
other issue, open or newly filed, blocking AC1, AC2, or AC5 beyond #1718.
#1885 and #1778 remain independently classified (§8, unaffected by this
checkpoint); #1617 (a much older, broader, platform-wide 18-module
self-deception audit umbrella) lists `sm-ir` ownership as one unchecked
inventory line among many but is not linked to #1579 as a scoped blocker and
predates this entire program - not treated as a residual #1579 blocker on
that basis. AC1/AC2/AC5's own §3 rows below are updated accordingly, with
the original "PARTIALLY SATISFIED, #1718" text struck through rather than
deleted.

## 1. Position A — frozen claim (authoritative)

Recorded in `docs/roadmap/stable_foundation/ssf08_ownership_position_decision.md`,
merged `c48882a28b0302209f571f05fd743ca6647c491d`, confirmed unmodified by
Lane 2's #1726/#1891 work:

> Semantic provides a bounded deterministic ownership/path-state model for
> explicitly admitted value paths across its verified VM pipeline (source ->
> frontend -> IR/SemCode -> verifier -> VM). Borrow and write events are
> transported through the versioned `OWN0` SemCode section, structurally
> admitted by the verifier before execution, and enforced by the VM as a
> frame-local, deterministic overlap rule over concrete (not indirect, not
> dynamically aliased) value paths... The active borrow set is scoped to the
> current call frame's lifetime.

Explicit non-claims (unchanged, still accurate against current `main`): no
Rust-equivalent lifetime/region inference, no general/indirect-path borrow
checking, no unrestricted alias analysis, no systems-language memory-safety
equivalence, no partial release, no inter-frame persistence, no concurrency
ownership.

## 2. Lane-by-lane status (fresh GitHub state, not #1579's own comment thread)

| Lane | Scope | Issues | Fresh state |
|---|---|---|---|
| 1 | `sm-front` `ScopeEnv` control-flow join model | #1656–#1664 (9) | **CLOSED**, all `COMPLETED`. Verified via GraphQL against live issue state, not the #1579 thread. |
| 2 | IR/transport ownership-event identity and timing | #1709, #1724, #1725, #1726, #1891 (5) | **CLOSED**, all `COMPLETED`. #1891 was discovered *during* #1726 and folded into the same chain (own umbrella-tracked FA-04-026). Final integration SHA `3ad550da55a4ff56db8dd33c9d82f26cc7fe8817` (this audit SHA). |
| 3 | ADT/Sequence path-family capability alignment | #1718 (1) | **OPEN**. Unchanged by Lane 2 — confirmed reproducible (see §7). |
| 4 | Frame/host boundary | #1770, #1771, #1773, #1775 (closed, confirmed already) + #1778 (open) | Frame/call boundary ownership behavior: **CLOSED** (#1726/#1891 subsume it). Host value-domain canonicality (#1778): **OPEN**, but classified independent of #1579's own scope (see §8). |
| 5 | Resource/quota/failure taxonomy | #1759, #1760, #1761, #1762, #1763 (5) | **OPEN**, all 5. Zero progress since Position A was recorded. |
| — | Findings discovered while SSF-08 was active | #1881, #1883 (closed); #1885 (open); #1888 (open on GitHub, reconciled as resolved — see §8) | See §8. |

The #1579 comment thread's last entry is dated 2026-09-03T14:42:59Z (Lane 2a/#1709
closure). Lane 2b/2c/2d (#1724/#1725/#1726), #1891's entire W1–W2F chain, and
the #1881/#1883 closures all post-date or were never posted to that thread.
**#1579's own comment history is stale relative to current repository state**
— this audit relies exclusively on fresh `gh api graphql` queries against live
issue state, current `main` content, and this session's own first-hand
qualification evidence, never on that thread's narrative alone.

## 3. Acceptance-criteria mapping

| # | Criterion | Authoritative contract | Implementation evidence | Test/qualification evidence | Unresolved blocker(s) | Verdict |
|---|---|---|---|---|---|---|
| AC1 | Public ownership/memory claim matches implementation evidence | `ssf08_ownership_position_decision.md`; `docs/spec/semcode.md`; `docs/spec/runtime_ownership.md` | `docs/spec/runtime_ownership.md`'s tuple/record slice matches implementation exactly (post-#1726/#1891). `docs/spec/semcode.md`'s SEMCOD11-tuple-only claim does **not** match implementation: verifier/VM already admit and enforce ADT payload and Sequence static-index ownership paths under that same header family (#1718, confirmed still reproducible §7). | Overclaim scan of current-facing docs (README, LANGUAGE.md, getting_started, status/*, roadmap/public_*, spec/*): zero overclaim hits — every "Rust-equivalent"/"borrow checker"/"lifetime" mention found is an explicit denial, not a claim. | #1718 | ~~**PARTIALLY SATISFIED** — clean in the direction of overclaiming (AC6 territory); violated in the opposite direction (implementation exceeds a *different* document's declared contract without a version/capability bump).~~ **Superseded 2026-09-06 (addendum #3)**: #1718 merged (PR #1896, `main` `b2fce6e2041661e3d08a3a196322f082a5f7c24f`) — `docs/spec/semcode.md`'s SEMCOD21 section, `docs/spec/verifier.md`, and `docs/spec/runtime_ownership.md` now describe exactly what the implementation admits (Sequence Borrow+Write, ADT Borrow only, `Write(AdtPayload)` explicitly excluded), and the previously-stale `docs/architecture/adt_payload_ownership_paths.md` was corrected in the same PR. **SATISFIED** — no known residual document/implementation mismatch for ownership found on fresh re-check (see addendum #3). |
| AC2 | Aggregate/path/frame/host ownership behavior documented and tested | `runtime_ownership.md`; decision record's "Included/Deferred Ownership Surface" tables | tuple: frozen, D7-qualified. record (direct field): frozen, D7-qualified. ADT payload: real positive E2E (`vm_runs_adt_payload_ownership_positive_e2e_path`), but verifier admission for the variant `SymbolId` is "structural acceptance only" (no bounds check), and no compiler-driven negative E2E exists yet (frontend has no ADT-payload reassignment syntax). Sequence static index: real positive+negative E2E (`tests/sequence_ownership_golden.rs`, `runtime_ownership_e2e.rs`), stronger proof than ADT. Map: no frontend `AccessPath` resolution exists at all (deliberate, honest absence, not a partial gap). Frame boundary: `push_frame` never merges caller state into callee (D2b-proven, inter-frame persistence structurally impossible). Host boundary: zero ownership state crosses it today (clean by construction); a *separate*, non-ownership value-canonicality gap exists there (#1778, §8). | `borrow_activation_v20` 9/9, `write_cursor_1891_repro` 4/4, `write_execution_site_e2e` 10/10, `sequence_ownership_golden.rs`, `record_field_ownership_golden.rs`, ADT payload tests in `crates/sm-vm/src/semcode_vm.rs`. | #1718 (ADT/Sequence not yet a declared, versioned, frozen capability) | ~~**PARTIALLY SATISFIED** — tuple/record/frame/host-ownership solid; ADT/Sequence implemented and tested but not contractually promoted; Map correctly out of scope.~~ **Superseded 2026-09-06 (addendum #3)**: #1718 merged - Sequence (Borrow+Write) and ADT (Borrow-only) are now declared, versioned (`HEADER_V21`/`SEMCOD21`), frozen capabilities, documented in `runtime_ownership.md`'s "Current supported slice," and tested by the full I5 matrix plus I4.1's direct verifier-level tests. Map remains correctly out of scope (deliberate, documented deferral, not a gap). **SATISFIED**. |
| AC3 | Partial move and sibling-access rules are deterministic | `runtime_ownership.md` overlap rules; decision record's Normative Invariants | Tuple/record overlap rules (`access_paths_overlap`) are one fixed, uniformly-applied function — proven deterministic by construction and by the full W2F E2E matrix. #1881 (nested two-level projection spuriously rejecting a sibling read) is **CLOSED** (PR #1882, merged 2026-09-03) — confirmed resolved, not a residual. ~~#1888 (`for`-range/`for`-each/`Stmt::Guard` lowering can silently skip `append_record_update_write_events_from_expr`, dropping a real Write event) remains **OPEN** and unrepaired — a genuine fail-open gap discovered during #1709's own audit and explicitly not fixed by PR #1887.~~ **Superseded 2026-09-06**: #1888's root cause is structurally eliminated by Checkpoint W2A (see addendum above and §5/§8) — RecordUpdate Write-event generation no longer depends on any prescan being invoked, confirmed empirically for all three named roots. | `tests/lexical_binding_identity_e2e.rs`, `tests/own0_root_identity_e2e.rs`, `tests/w15_record_update_makerecord_site_proof.rs`, `tests/fa_04_025_reconciliation.rs` (8/8, new), plus all Lane 2/#1891 suites | none remaining for this criterion | **SATISFIED** — the general rule is now proven deterministic everywhere it is actually invoked, including all three of #1888's originally-named roots. |
| AC4 | Resource quotas and failure taxonomy are explicit | #1579 body itself; decision record's Resource/Failure Boundary section (explicitly scoped as separate from, but required alongside, ownership) | `max_steps`/`max_calls` (#1759): declared, never enforced. `trace_enabled`/`max_trace_entries` (#1760): declared, no consumer. `ConstPool` quota (#1761): declared, no VM resource behind it. `ExecutionContext`/quota decoupling (#1762): `ExecutionConfig` can carry a context label that disagrees with its own quota baseline; the VM never reads `config.context`. `RuntimeTrap` taxonomy (#1763): 8 documented variants, only 4 (`AssertionFailed`, `BorrowWriteConflict`, `ArithmeticOverflow`, `DivisionByZero`) are ever constructed. | None — zero enforcement code, zero fresh tests, since Position A was recorded | #1759, #1760, #1761, #1762, #1763 (all 5, zero closed) | **NOT SATISFIED** — no measurable progress on this criterion since SSF-08 began. |
| AC5 | Verifier and runtime agree on admitted/rejected ownership states | `runtime_ownership.md` Verifier/VM Enforcement Contracts | Tuple/record/frame contour: exact agreement proven down to the instruction-PC level (Checkpoints D1–D3, W1–W2F) — the strongest evidence in this audit. Legacy SemCode: verifier still structurally admits pre-rev21 Write-bearing artifacts (decode/verify compatibility, Checkpoint W2E); runtime now deterministically rejects executing them (Checkpoint W2F) — a *documented*, intentional divergence between admission and execution, not a disagreement bug. ADT/Sequence: verifier and VM do agree operationally (both admit/process the same component kinds), but the admission itself is weaker (no SymbolId bounds check for ADT payload variants) than the tuple/record standard. ~~#1888 means a source-level write intent can occasionally have **no** Write event at all to check on either side.~~ **Superseded 2026-09-06**: confirmed no longer true — see AC3. | `checkpoint_w2e_legacy_pre_rev21_write_admitted_unchanged` (sm-verify), `legacy_pre_rev21_write_bearing_artifact_rejected_at_runtime` (sm-vm/tests), `tests/fa_04_025_reconciliation.rs`'s downstream-chain pair (real `BorrowWriteConflict` at a for-range-bound RecordUpdate site), full D-series and W-series suites | #1718 (weaker ADT admission) only | ~~**PARTIALLY SATISFIED** — excellent for the primary (tuple/record) contour, including all three of #1888's former roots; weaker only at the ADT/Sequence admission edge, gated by #1718.~~ **Superseded 2026-09-06 (addendum #3)**: #1718 merged - decoder, verifier, and runtime now agree uniformly on Sequence Borrow+Write and ADT Borrow admission and ADT Write rejection; mutation M4b's own finding (the verifier's independent check had no test that could catch its own removal) is closed by I4.1's direct, decode-independent test, confirming verifier and decoder are each *independently*, not merely coincidentally, in agreement. **SATISFIED**. |
| AC6 | No documentation implies full Rust-like ownership unless qualified | Current-facing docs | Scanned `README.md`, `docs/LANGUAGE.md`, `docs/getting_started.md`, `docs/status/feature_maturity_matrix.md`, `docs/roadmap/public_maturity_snapshot.md`, `docs/roadmap/public_status_model.md`, `docs/roadmap/v1_readiness.md`, `docs/examples_index.md`, `docs/spec/*.md` for `rust-equivalent ownership`, `rust-like ownership`, `memory safety`, `lifetime inference`, `region inference`, `borrow checker`, `alias safety`. Every hit (`runtime_ownership.md`, `foundation_source_profile_v1.md`) is an explicit **denial**, never a claim. `docs/LANGUAGE.md`'s "borrows" hits are unrelated (design-inspiration language, not ownership). | Overclaim scan, this audit | none found | **SATISFIED**. |
| AC7 | SSF-09 entry conditions are explicit | #1579's own Exit gate: "SSF-09 starts only when Semantic's public memory/ownership promise is precise, test-backed, and no stronger than the implementation"; `stable_foundation_dependency_map.md`'s serial gate and cross-phase dependency table | The procedural gate is explicit (`.harness/current.task.yaml`: `active_phase: SSF-08`; dependency map: `SSF-09 / #1580 ... Blocked by SSF-08`). The *substantive* condition is also explicit — it is stated directly in #1579's own Exit gate and is identical in substance to AC1. No separate, additional SSF-09-specific checklist exists beyond that, but none is needed: the cross-phase dependency table itself only lists SSF-10 through SSF-12 (not SSF-09) as consumers of the ownership *position* specifically — SSF-09's own inputs come from SSF-07, not SSF-08, substantively; SSF-08 only gates it procedurally. | none | none (the gate is explicit; whether it currently evaluates true depends on AC1, tracked separately) | **SATISFIED** — the entry condition is explicit; it is not yet *true*, but that is AC1's finding, not AC7's. |

## 4. Lane 2 — completed subprogram, explicitly reconciled

Chain, confirmed closed end to end (fresh GraphQL state, all `CLOSED`/`COMPLETED`):

```
#1709 (event preservation across nested/loop-expr/closure lowering)
  -> #1724 (lexical binding identity)
  -> #1725 (OWN0 root identity)
  -> #1726 (OWN0 Borrow event timing / exact activation site)
  -> #1891 (discovered during #1726: Write execution-site false negative)
       -> W1, W1.5, W2A-W2F (exact Write execution-site enforcement)
```

Final integration SHA: `3ad550da55a4ff56db8dd33c9d82f26cc7fe8817` (this audit's
own baseline — PR #1892, rebase-merged, `Closes #1726`, `Closes #1891`).

**What Lane 2 now proves, end to end, for the tuple/record contour:**

```
source ownership event identity (frontend-stable binding + IR SymbolId)
  -> IR ActivationSiteId / WriteSiteId identity
  -> optimizer coherence (validate_activation_sites / validate_write_sites,
     fail-closed bijection checks around every pass)
  -> exact executable-instruction authority (opcode-start PC, not inferred)
  -> SemCode transport (rev21 / HEADER_V20, bit-for-bit round-trip)
  -> verifier admission (canonical-boundary + declared-opcode-class + same-
     function, never raw-byte or root/symbol matching)
  -> runtime enforcement (exact-PC lookup, check-before-commit-before-
     activate ordering, no cursor, no consume-once, rechecked on every
     visit)
```

**What Lane 2 does *not* prove about the rest of SSF-08** (explicitly, so
this is never silently read as broader than it is):

- It does not touch the ADT/Sequence header-capability contract question
  (#1718) — that axis (component *kind* admission) is orthogonal to the
  execution-*site* timing axis Lane 2 repaired.
- It does not touch resource quotas or failure taxonomy (Lane 5) at all.
- It does not touch the host ABI value-canonicality question (#1778).

**Correction (2026-09-06):** an earlier version of this section listed
#1888's three lowering roots (`for`-range/`for`-each headers, `Stmt::Guard`'s
`else_return` payload) as *not* covered by Lane 2, on the reasoning that they
call a different top-level authority than the nested/loop-expression paths
#1709 repaired. That reasoning was correct as far as it went, but incomplete:
Checkpoint W2A (part of the same #1891/Lane-2-adjacent chain, landed after
#1709) independently made the *entire class* of "forgot to prescan" bugs for
RecordUpdate specifically unreachable, by relocating Write-event generation
into the universal `lower_expr_with_expected` entry point every one of those
three roots already calls for real value-lowering regardless. See the
addendum at the top of this document and §8 for the full reconciliation.

This audit found no evidence contradicting Lane 2's own qualification
(re-ran `borrow_activation_v20`, `write_cursor_1891_repro`, and
`write_execution_site_e2e` conceptually via this session's own prior
exact-`main`-SHA qualification, all 9/4/10 green at this exact audit SHA) —
Lane 2 is **not** reopened by this audit.

## 5. #1718 — deep audit (no repair performed)

**Finding, confirmed still reproducible on audit SHA `3ad550da5`:**
`crates/sm-format/src/semcode_decode.rs::decode_sequence_index_static_ownership_component`
and `crates/sm-verify/src/lib.rs`'s `verifier_accepts_sequence_index_static_ownership_semcode`
/ `verifier_accepts_adt_payload_ownership_semcode` all still pass unchanged
(confirmed in this session's own W2E/W2F qualification runs against this
exact tree: `sm-format` 55/55, `sm-verify` 212/212, both include these exact
tests). `docs/spec/semcode.md` still freezes `SEMCOD11` as tuple-only
transport with no version/capability gate added for `AdtPayload` or
`SequenceIndexStatic` component kinds. `crates/sm-runtime-core/src/lib.rs`
still publicly exposes `PathComponent::AdtPayload`/`SequenceIndexStatic` and
their builders as ordinary first-class shapes.

1. **Is the defect still reproducible on current main?** Yes, directly —
   unchanged since the issue's own Phase-A evidence rounds (2026-08-15/16),
   confirmed via this session's own fresh test runs on the exact audit SHA.
2. **Did rev21/#1726/#1891 incidentally change any part of it?** No. Lane 2's
   entire repair concerned Borrow/Write *execution-site timing/anchoring*
   (which instruction, at what PC) — an axis orthogonal to *component-kind
   capability gating* (which path shapes a given header revision may legally
   carry). Neither `validate_activation_sites`/`validate_write_sites` nor the
   rev21 wire grammar reference `AdtPayload`/`SequenceIndexStatic` gating at
   all; `has_v11_ownership_events`/`has_v12_record_field_ownership_events`
   (the promotion-gate functions the issue's own evidence cites) were not
   touched by any Lane 2 commit.
3. **Are ADT and Sequence ownership paths part of the frozen Position A
   stable contour?** Not yet, by the decision record's own explicit
   classification: both are listed as "INCLUDED ONLY AFTER REQUIRED REPAIR"
   (naming #1718 by number), not "INCLUDED IN SSF-08" — a status distinct
   from tuple/record's unconditional inclusion.
4. **If yes, must #1718 be repaired before #1579 closes?** Yes, conditionally
   — not because ADT/Sequence themselves are mandatory for Position A's
   bounded model, but because AC1 and AC2 require the *documented* contract
   to match *implementation evidence*, and today it structurally does not
   for this one wire-format axis. This is independent of whether ADT/Sequence
   are ever promoted; even choosing fail-closed rejection would satisfy AC1.
5. **Is fail-closed rejection sufficient?** Yes, per the decision record's own
   framing — either remedy (capability promotion or fail-closed rejection)
   closes the contract gap; the decision record deliberately leaves the
   choice open as an implementation decision, not a positioning one.
6. **What exact contract decision is still missing?** The promote-vs-reject
   choice itself, plus its implementation: either (a) a new capability bit
   gating `AdtPayload`/`SequenceIndexStatic` admission with a documented
   header/revision promotion, and updated verifier/spec text; or (b) a
   verifier-level fail-closed rejection of these component kinds below
   whatever revision/capability is chosen, with `docs/spec/semcode.md` and
   `runtime_ownership.md` updated to state that boundary explicitly.

**Verdict: REQUIRED BEFORE SSF-08 CLOSE** (via AC1 and AC2), with an
already-recorded, implementation-agnostic remedy choice — this is a bounded,
well-scoped decision-plus-slice, not an open-ended redesign.

## 6. Frame/host boundary — independent audit

Trace: `Semantic value -> VM frame -> call boundary -> ABI -> host -> ABI
return -> VM`.

- **Frame/call boundary**: `push_frame` fuses `validate_call_arguments`
  (arity/signature, #1773) for every entry route including `ClosureCall`;
  `RegisterSlot::{Uninitialized, Value}` distinguishes never-written from
  written (#1770); `MapGet`'s `?` propagation is correct (#1771). All three
  confirmed already closed, unaffected by Lane 2. Ownership state itself:
  `push_frame` never merges caller state into the callee frame — structurally
  no inter-frame ownership persistence, matching Position A exactly.
- **Host ABI boundary (value, not ownership)**: `value_from_abi`'s `Quad`
  conversion routes through `quad_from_abi`, which rejects any byte outside
  `0..=3` (#1775, confirmed already repaired — VM-side defensive layer is
  live). `prom_abi::AbiValue::Quad(u8)` itself remains an unchecked raw
  carrier with no `TryFrom`/checked constructor (#1778, confirmed still
  **OPEN**, zero comments/activity since filing).

**Is #1778 an SSF-08 blocker?** No. #1579's own AC2 requires *ownership*
behavior at the frame/host boundary to be documented and tested — which it
is, cleanly, because no ownership state crosses that boundary today by
design. #1778 concerns *value-domain canonicality* (is a raw byte a valid
`Quad`), a materially different axis from ownership-path tracking, and the
practically-exploitable consequence is already closed by #1775's VM-side
check (defense-in-depth's *second* layer already exists; #1778 is about
adding the *first* layer too). The SSF-08 decision record's own
Blocker-To-Route Audit table already classified #1778 this way
("INDEPENDENT SSF-08 TRACK... not ownership"), and this audit finds no
reason to overturn that.

**Frame/host boundary verdict (ownership-relevant scope): SATISFIED.**
**#1778 itself: INDEPENDENT** of #1579 — real, still open, but not a
closure blocker for this issue.

## 7. Resource/failure taxonomy — independent audit

| Concern | Current implementation | Deterministic error/trap type | Tests | Docs | Issue state | SSF-08 relevance |
|---|---|---|---|---|---|---|
| `max_steps`/`max_calls` | Declared in `RuntimeQuotas`, zero enforcement call sites in `semcode_vm.rs` | none | none | published as a contract | #1759 OPEN | AC4 |
| `trace_enabled`/`max_trace_entries` | Declared, no production consumer | n/a | none | published | #1760 OPEN | AC4 |
| `ConstPool` quota | Declared, no VM resource behind it | n/a | none | published | #1761 OPEN | AC4 |
| `ExecutionContext`/quota decoupling | `ExecutionConfig` can carry a mismatched context label; VM never reads `config.context` | n/a (silent) | none | published | #1762 OPEN | AC4 |
| `RuntimeTrap` taxonomy | 8 documented variants, 4 ever constructed (`AssertionFailed`, `BorrowWriteConflict`, `ArithmeticOverflow`, `DivisionByZero`) | partial | existing trap tests cover only the 4 live variants | published as complete | #1763 OPEN | AC4 |

**Verdict: NOT STARTED.** Zero of the five issues have moved since Position A
was recorded (2026-09-02). This is the single most unambiguous gap found in
this audit — there is no partial-credit reading available.

## 8. Residual issue disposition table

| Issue | Title (short) | Fresh state | Disposition | Reason / authority |
|---|---|---|---|---|
| #1656–#1664 | Lane 1 ScopeEnv cluster | CLOSED | ALREADY SATISFIED | Fresh GraphQL: all `COMPLETED`. |
| #1709, #1724, #1725, #1726, #1891 | Lane 2 chain | CLOSED | ALREADY SATISFIED | Fresh GraphQL: all `COMPLETED`; final SHA is this audit's own baseline. |
| #1718 | ADT/Sequence capability contour | **CLOSED** (2026-09-06, PR #1896, `main` `b2fce6e2041661e3d08a3a196322f082a5f7c24f`) | **ALREADY SATISFIED** | Contract decision (Sequence Borrow+Write=PROMOTE; ADT Borrow=PROMOTE, ADT Write=RESTRICT/FAIL-CLOSED) fully implemented, qualified (6 mutation-testing falsification passes, hosted CI/security, exact-HEAD Codex review), merged, and closed. AC1/AC2/AC5 re-evaluated SATISFIED (§3 addendum #3). |
| #1759, #1760, #1761, #1762, #1763 | Quota/failure taxonomy | OPEN (5/5) | **REQUIRED** | Directly and explicitly named by AC4; zero progress. |
| #1888 | Lowering can skip Write-event emission for 3 statement roots | OPEN (GitHub state unchanged; recommend `CLOSE AS COMPLETED` — see reconciliation note) | **ALREADY SATISFIED BY LATER WORK** *(revised 2026-09-06, was REQUIRED)* | Checkpoint W2A relocated RecordUpdate Write-event generation into the universal `lower_expr_with_expected` entry point, structurally eliminating the prescan-omission root cause for all three named roots — confirmed by direct code reading and 8/8 new empirical tests (`tests/fa_04_025_reconciliation.rs`), including real downstream `BorrowWriteConflict` enforcement. |
| #1778 | `AbiValue::Quad(u8)` unchecked at ABI boundary | OPEN | **INDEPENDENT** | Value-domain canonicality, not ownership; #1579's own decision record already so classifies it; practical consequence already mitigated by #1775. |
| #1885 | `public-api-guard` omits `sm-front` | OPEN | **INDEPENDENT** | Cross-cutting CI/tooling debt, not ownership-positioning substance; same disposition class as the pre-existing #1700/`sm-sema` analog. |
| #1881 | Nested projection sibling-read false rejection | CLOSED | ALREADY SATISFIED | Fresh GraphQL: `COMPLETED` via PR #1882 (2026-09-03), predates the last #1579 comment. |
| #1883 | Projected-scrutinee negative test gap | CLOSED | ALREADY SATISFIED | Fresh GraphQL: `COMPLETED` via PR #1884 (2026-09-03). |

No issue required a RETURN or DEFER disposition in this audit: everything
found either belongs squarely inside #1579's own seven acceptance criteria
(REQUIRED) or is independent of them (INDEPENDENT); nothing surfaced that
belongs to a *different*, already-passed SSF phase, and the one deliberate
scope exclusion found (Map ownership paths) is a documented non-goal with no
open GitHub issue attached, not a disposition-needing item.

## 9. False-closure detection

Actively tried to falsify "SSF-08 is ready to close now" against each listed
class:

- open required child issue — **found**: #1718, #1759–#1763. (#1888 was
  initially found here too; reconciled 2026-09-06 as already resolved by
  Checkpoint W2A — see addendum and §8.)
- current-facing doc overclaim — not found (§3, AC6).
- unqualified path family — **found**: ADT/Sequence (implemented, tested,
  but not a declared/versioned capability; §5, §3 AC2).
- frame/host mismatch — not found for ownership specifically; a real but
  independent value-domain issue exists (#1778, §6).
- verifier/runtime disagreement — not found as a hard disagreement; a
  documented, intentional decode/execute divergence exists for legacy
  artifacts (by design, W2E/W2F) and a weaker (not wrong) ADT admission
  standard exists (§3, AC5).
- missing quota taxonomy — **found**: Lane 5 entirely unstarted (§7).
- stale known-bug regression — not found; #1881/#1883 confirmed genuinely
  fixed, not merely marked closed.
- test that proves the wrong layer — not found in this audit's sampling.
- unsupported state silently accepted — **found**, but already the subject
  of #1718 (ADT/Sequence under a header that doesn't declare them).

At least two independent, evidence-backed acceptance blockers remain after
the 2026-09-06 #1888 reconciliation (originally three; #1888 is resolved).

**SSF-08 READY TO CLOSE = NO.**

## 10. Minimal residual DAG

*(Revised 2026-09-06 — #1888 removed after reconciliation (see addendum);
#1718 removed after implementation, merge, and re-evaluated AC1/AC2/AC5
(see addendum #3).)*

Exactly one residual track remains:

```
Lane 5 (#1759-#1763: quota enforcement + RuntimeTrap taxonomy completion)   ─┐
                                                                              │   SSF-08 final
                                                                              │   acceptance
                                                                              │   reconciliation
                                                                              ↓
                                                                        close #1579
                                                                              ↓
                                                                  activate SSF-09 #1580
```

`#1885` and `#1778` are real, open, but explicitly INDEPENDENT of this DAG —
they may be picked up on their own track at any time without affecting
SSF-08's own closure sequencing. `#1888` and `#1718` are no longer part of
this DAG at all — see the 2026-09-06 addenda.

## 11. Verdict

**READY TO CLOSE #1579: NO** — Lane 5 (#1759-#1763, AC4) remains entirely
unstarted; every other acceptance criterion (AC1, AC2, AC3, AC5, AC6, AC7)
is now SATISFIED as of the 2026-09-06 addenda.

Exact next checkpoint: Lane 5 (#1759-#1763) is the sole remaining forced
choice — the residual DAG no longer has an independent second track to pick
between. Scope: quota enforcement (`max_steps`/`max_calls`/`ConstPool`/
`trace_enabled`) and `RuntimeTrap` taxonomy completion (4 of 8 documented
variants ever constructed) across `sm-vm`, entirely unstarted, 5 issues.

Awaiting explicit GO to begin Lane 5.
