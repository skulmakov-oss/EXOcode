# SSF-08 Lane 5 / #1761 — ConstPool Quota Contract Decision

Status: **CONTRACT DECISION ONLY. NO PRODUCTION BEHAVIOR CHANGE.**
Baseline SHA: `4cbe46ca0d47eb53d06034679a01e8d8e20010ae` (`main`, confirmed
current).
Target: `#1761` / `FA-08-003`.
Umbrella: SSF-08 umbrella #1579 remains OPEN.

This document freezes the disposition of `RuntimeQuotas::max_const_pool`
and `QuotaKind::ConstPool`. It does not delete anything, does not touch
`#1760`/`#1762`/`#1763`, and does not change any golden snapshot, baseline
value, or production Rust. No dedicated `#1761` decision authority existed
before this document (checked: no file matching
`*1761*constpool*` under `docs/roadmap/stable_foundation/` prior to this
one) - the only prior authority is the Lane 5 closure audit's own §5,
which this document builds on and formalizes into a standalone,
implementation-ready decision.

## 1. Fresh evidence (re-derived, not assumed from the prior audit)

Repository-wide search for `const_pool`/`ConstPool`/`constant_pool` across
`crates/sm-runtime-core/`, `crates/sm-vm/`, `crates/sm-format/`,
`crates/sm-verify/`, `crates/sm-emit/`, and `docs/spec/quotas.md` returns
matches **only** in `crates/sm-runtime-core/src/lib.rs`:

- the `QuotaKind::ConstPool` enum variant (line 141)
- the `RuntimeQuotas::max_const_pool: usize` field (line 178)
- three baseline-profile assignments, `65_536` in `verified_local`,
  `pure_compute`, and `kernel_bound` alike (lines 192/206/220 - identical
  across every profile, itself a signal no profile-specific tuning was
  ever done because nothing consumes the value)
- one match arm in `RuntimeQuotas::exceed()` (line 234)

Zero matches in `sm-vm`, `sm-format`, `sm-verify`, or `sm-emit`. Zero
`enforce_quota`/`charge_counter` call sites anywhere pass
`QuotaKind::ConstPool`. `docs/spec/quotas.md` lists `ConstPool` in its
quota-kind taxonomy (line 22) and its baseline-profile tables, but its own
"Enforcement Rule" section's "Current enforced areas" list (opcode-dispatch
count, admitted-call count, frame count, stack depth, register growth,
effect-call budget) **omits `ConstPool` entirely** - the specification is
internally honest that this kind is not enforced, even while still listing
it as though it were live vocabulary.

### 1.1 Candidate-resource classification

Every plausible referent for "a pooled/deduplicated constant storage
mechanism" was inspected and classified:

| Candidate | Real ConstPool resource? | Evidence |
|---|---|---|
| Per-function string table (`FunctionBytecode.strings`, decoded via `sm-format`) | **DIFFERENT RESOURCE** | Decode-time structural cap only (`MAX_STRINGS_PER_FUNCTION`, `MAX_STRING_LEN` in `sm-format`), not a `RuntimeQuotas`-governed bound. The *producer* side (`sm-ir`'s `StringInterner`) does dedup per function being lowered, but that interner is rebuilt fresh per function - it is not a program-wide pool. |
| `RuntimeSymbolTable` (`crates/sm-runtime-core/src/lib.rs`) | **REAL pooling mechanism, but it is `SymbolTable`, not `ConstPool`** | Genuine dedup-by-value (`.intern()` returns an existing id if present), built once per program by interning every function's string table (`crates/sm-vm/src/semcode_vm.rs`), and already bounded by its own dedicated, already-enforced quota: `QuotaKind::SymbolTable`/`max_symbol_table`, checked program-wide in `sm-verify` (`crates/sm-verify/src/lib.rs`, `unique_runtime_symbol_count > quotas.max_symbol_table` → `ResourceLimitExceeded`). This is the closest thing in the codebase to what "ConstPool" sounds like it should name - and it already has its own, correctly-named, correctly-enforced quota kind. `ConstPool` is not an alias, a stale rename, or a duplicate of this; it is simply disconnected from it. |
| `SIG0` / `CallableSignature` (parameter-family records) | **DIFFERENT RESOURCE / STRUCTURAL LIMIT** | Per-function type metadata (arity + family per parameter), not a value/constant at all. Decode-time cap only (`MAX_SIGNATURE_PARAMETERS_PER_FUNCTION`), unrelated to any `RuntimeQuotas` field. |
| Inline instruction immediates (`LoadI32`/`LoadU32`/`LoadF64`/`LoadFx`/`LoadQ`/`LoadBool`) | **NO POOLING MECHANISM EXISTS** | Every one of these writes its literal value as raw bytes directly into the instruction stream at lowering time (`crates/sm-ir/src/legacy_lowering.rs`) - no table, no index, no dedup. Two identical `LoadI32` literals in the same function each get their own 4 raw bytes. This is the one category a real "constant pool" would classically exist to deduplicate, and nothing in this codebase does so. |

**None of these candidates is charged against, or even conceptually named,
`max_const_pool`/`QuotaKind::ConstPool`.** There is no shared,
runtime-resident "constant pool" object anywhere in this codebase that
this quota kind could describe under any name.

**Falsification attempted:** searched for a real constant-pool
representation under any other name across every crate above (none
found); searched git log for any commit ever touching `ConstPool`/
`const_pool` outside its own original definition (none found beyond the
Lane 5 audit's own prior pass, which reached the identical conclusion
independently). The finding is not stale and does not depend on the prior
audit's own authority - it was re-derived fresh against current `main`.

## 2. Public compatibility surface (not merely a private cleanup)

`tests/golden_snapshots/public_api/sm_runtime_core_lib.txt` locks in both:

```
ConstPool,
pub max_const_pool: usize,
```

`tests/public_api_contracts.rs`'s own `public_api_inventory_matches_checked_in_contract_snapshots`
test fails on **any** drift from this snapshot, additive or subtractive,
and its own failure message reads: *"public API inventory drifted for
{source}; update snapshot only for intentional contract changes"* - the
guard is a drift detector requiring explicit human intent
(`SM_UPDATE_PUBLIC_API_SNAPSHOTS=1`), not a policy statement that the
snapshotted surface must never shrink. This is evidence of what currently
exists, not authority that the inert surface must survive.

`docs/spec/quotas.md`'s own "Version Review Rule" (the only quota-specific
governance text) requires review for: changing a quota kind's meaning,
adding a new quota kind, changing a baseline value in a user-visible
execution path, or changing quota-exhaustion error-reporting semantics.
**It does not explicitly enumerate removal as a listed case.** This
document does not infer platform policy from that silence - a rule not
naming a case is not itself evidence about how that case should be
treated. Independently,
`docs/roadmap/language_maturity/compatibility_policy_stack.md` (Status:
proposed v0) names "boundary/runtime contract changes" as one of its own
explicit "Review Triggers" (its own §"Review Triggers", listing source
syntax, CLI, stdlib, manifest/lockfile, SemCode/Profile meaning, and
boundary/runtime contract changes alike) requiring explicit compatibility
review. **This #1761 decision checkpoint is that explicit review** for the
`ConstPool` surface specifically - not a policy inferred from what the
narrower quota-specific rule happens not to mention.

`docs/roadmap/language_maturity/stability_and_compatibility.md` (Status:
**proposed v0**) is the only platform-wide compatibility-policy document,
and it states plainly that the platform does **not yet have** a complete
compatibility policy stack, stability labels, or a deprecation/migration
process - "no public surface should be left unlabeled *once it becomes
part of the published platform story*" is an aspiration this workstream
has not yet delivered, not a currently-binding constraint. No `sm-runtime-
core` surface is currently labeled `stable`/`beta`/`draft`/`experimental`
under any authority, because that labeling system does not exist yet.
`docs/roadmap/stable_foundation/stable_foundation_target_contract.md`
attributes ownership of "final minimal APIs" to a different phase
(SSF-03), not to this Lane 5 checkpoint or to SSF-08 generally.

**Conclusion: no stable label or binding deprecation commitment for this
specific `sm-runtime-core` surface was found.** The current surface is
therefore **unclassified** rather than proven stable or proven unstable -
`stability_and_compatibility.md` itself says draft/experimental surfaces
"may change faster, but must remain labeled as such," which presumes a
labeling act this platform has not yet performed for any `sm-runtime-core`
item, `ConstPool` included. This document does not conclude from that gap
that no deprecation process could ever apply here; it concludes only that
none currently does, because none has been established for this surface
by any authority found.

Stable Foundation may therefore perform the intentionally reviewed
narrowing now - via this decision checkpoint as the explicit compatibility
review (§2 above) - but the future implementation checkpoint must
document the change as source-visible and update the approved public-API
snapshot explicitly rather than treat it as an internal refactor. The
required qualification is: (1) an explicit, human-approved regeneration of
the golden snapshot via `SM_UPDATE_PUBLIC_API_SNAPSHOTS=1` - never
automatic, never silently absorbed (the same mechanism already used,
correctly, for #1759's additive `VM.steps`/`VM.calls` fields); (2) a
`docs/spec/quotas.md` update; (3) qualification tests updated to stop
asserting the removed field/variant exists. A future, more mature
compatibility regime that does classify this surface and does define a
deprecation process is not foreclosed by this document - it simply does
not exist yet, so it cannot be the authority this decision defers to.

## 3. Decision candidates evaluated

**A. REMOVE** - delete `QuotaKind::ConstPool`, delete
`RuntimeQuotas::max_const_pool`, remove the three baseline-profile values,
remove the `docs/spec/quotas.md` references.

**B. RE-SCOPE** - redefine `ConstPool` to mean an actually-existing
resource.

**C. RESERVED / DEPRECATED COMPATIBILITY SURFACE** - keep the type shape
temporarily, but explicitly remove it from the active enforced-quota
contract (e.g. document it as inert-by-design, or gate it behind a
`#[deprecated]` attribute without removing it).

## 4. REMOVE — falsification

**Preferred candidate, stated plainly:** ConstPool does not exist as a
runtime resource. Therefore the Stable Foundation contract should remove
`QuotaKind::ConstPool` and `RuntimeQuotas::max_const_pool` rather than
invent a resource solely to justify old vocabulary.

**Why this is the preferred candidate going in:** no runtime object
exists (§1); no charge point exists (§1); no enforcement exists (§1); no
current language feature requires such a pool (§1.1 - every candidate
value-storage mechanism is either unpooled inline data or already owned by
a different, correctly-named, already-enforced quota kind); other
candidate resources already have different names/owners/limits
(`SymbolTable` owns string dedup; decode-time structural caps in
`sm-format` own per-function table sizes).

**Actively attempted to disprove REMOVE using each of the following
authorities - none succeeds:**

- **Compatibility commitments:** none exist for this surface today (§2).
  No user-facing doc, release note, or stability label promises
  `QuotaKind::ConstPool`'s continued existence.
- **External API requirements:** no repository-governed external
  compatibility commitment for `QuotaKind::ConstPool`/`max_const_pool` was
  found - no changelog entry, migration note, or external-facing doc
  references it. Possible arbitrary out-of-repo source consumers of
  `sm-runtime-core` cannot be exhaustively disproven and are **not** used
  as a premise of this decision; the REMOVE disposition rests instead on
  the affirmative findings already established above - no architectural
  referent (§1), no enforcement (§1), no serialization/wire dependency
  (below), no stable-surface commitment found (§2) - combined with the
  explicit public-API review and golden-snapshot update this same
  checkpoint performs (§2, §10).
- **Future-format commitments:** no SemCode header revision
  (`HEADER_V21`/`SEMCOD21` etc.) or wire-format document reserves a
  `ConstPool`-shaped section. The wire format has no concept of a constant
  pool section at all - this is purely an in-memory `RuntimeQuotas` struct
  field, never serialized to or decoded from SemCode bytes.
- **Verifier/decoder invariants:** `sm-verify` and `sm-format` never read
  `QuotaKind::ConstPool` or `max_const_pool` (§1) - no invariant depends on
  its presence.
- **`no_std` constraints:** `sm-runtime-core` supports a `no_std` build
  (confirmed via the workspace's own `check-no-std` CI gate); removing a
  `usize` field and an enum variant strictly reduces surface and cannot
  introduce a `no_std` incompatibility.
- **Serialization/ABI dependencies:** `RuntimeQuotas` and `QuotaKind` are
  plain in-memory Rust types with no `Serialize`/`Deserialize` derive and
  no wire encoding (confirmed by their derive lists in
  `crates/sm-runtime-core/src/lib.rs` - `Debug, Clone, Copy, PartialEq,
  Eq` only). Removing a field changes the type's shape but crosses no
  serialized-ABI boundary.

**No authority requires retaining it. REMOVE is frozen** as the selected
disposition, pending only the exact removal mechanics in a future,
separately-authorized implementation checkpoint.

## 5. RE-SCOPE — falsification

Re-scoping is not a rename; `docs/spec/quotas.md`'s own "Version Review
Rule" requires treating any quota-kind meaning change as a genuine,
reviewed decision. For each candidate referent from §1.1:

| Referent | Runtime-resident? | Already bounded? | Bound owner | Structural or dynamic? | Would "ConstPool" be accurate? | Duplicates an existing quota? |
|---|---|---|---|---|---|---|
| Per-function string table | Yes, as decoded bytes | Yes | `sm-format` (`MAX_STRINGS_PER_FUNCTION`) | Structural (decode-time, fixed) | No - it is text-literal storage, not a general constant pool, and it is per-function, not program-wide | Would duplicate the *decode-time* structural cap already owned by `sm-format` |
| `RuntimeSymbolTable` | Yes | Yes | `sm-verify` (`max_symbol_table`) | Dynamic (value-dependent dedup count), checked statically pre-execution | No - it already has an accurate, specific name (`SymbolTable`) that this checkpoint must not blur | **Yes** - would exactly duplicate `QuotaKind::SymbolTable`, the precise outcome §7's "no compatibility ghost" rule forbids |
| `SIG0`/signature table | Yes, as decoded metadata | Yes | `sm-format` (`MAX_SIGNATURE_PARAMETERS_PER_FUNCTION`) | Structural | No - type metadata is not a constant value store | Would duplicate a `sm-format` decode-time cap with no `RuntimeQuotas` counterpart today |
| Inline immediates | Yes, as raw bytes | No | none | N/A - no pool exists to bound | No - re-scoping to "the literal bytes already inline in the instruction stream" is not a resource, it is the absence of one | N/A |

No candidate produces a strong positive answer on **both** "would this
name be semantically accurate" and "would this avoid duplicating an
existing quota." The `RuntimeSymbolTable` candidate is the closest
conceptual fit and is exactly the one case where re-scoping would create
an outright duplicate quota kind for an already-correctly-named,
already-enforced resource - the single worst outcome for RE-SCOPE to
produce. **RE-SCOPE is rejected.**

## 6. RESERVED / DEPRECATED COMPATIBILITY SURFACE — not accepted casually

Candidate C was not dismissed by default; it was tested against the "no
compatibility ghost" question: can an inert public enum variant/field
remain without continuing the false-ready signal this whole Lane 5 audit
exists to close?

**No.** The entire premise of SSF-08's Lane 5 audit (see the closure audit
doc's own §2/§14) is that a resource being *listed* in the quota
taxonomy and baseline profiles - regardless of whether any caller ever
reads its exhaustion result - constitutes a false-ready signal by itself:
a caller inspecting `RuntimeQuotas::verified_local().max_const_pool` and
seeing `65_536` has no way to discover, from the type alone, that this
number governs nothing. Keeping the field "reserved" (even with a
`#[deprecated]` attribute) preserves exactly this discoverability failure
- a `#[deprecated]` warning fires only at call sites that name the item,
and `RuntimeQuotas`'s own struct-literal-with-default-remaining-fields
construction pattern (used throughout this codebase, e.g. `RuntimeQuotas {
max_effect_calls, ..RuntimeQuotas::verified_local() }`) means most
existing and future call sites would never trigger the warning at all.
**Candidate C is rejected**: no concrete compatibility authority (§2)
requires such a bridge, and false compatibility is still false-ready.

## 7. No compatibility ghost (explicit statement)

Per the frozen REMOVE disposition, the future implementation checkpoint
must **not**:

- keep `max_const_pool` but ignore it
- keep `ConstPool` forever as a reserved/deprecated variant
- silently map it to `SymbolTable`
- silently map it to string-table limits
- return a fake `QuotaExceeded(ConstPool)` from any code path

None of these is authorized by any compatibility authority identified in
§2, and each would reproduce the exact false-ready pattern this decision
exists to close.

## 8. Failure-taxonomy consequence for #1763

**If REMOVE lands:** `QuotaKind::ConstPool` disappears from the active
quota taxonomy entirely. `RuntimeError::QuotaExceeded` can therefore never
legitimately carry `kind: QuotaKind::ConstPool` after that implementation
lands (it already never does today - this makes the impossibility
structural rather than merely coincidental). This checkpoint does **not**
modify `RuntimeTrap` or `RuntimeError` - it only freezes one input fact
`#1763`'s own future taxonomy-reconciliation work must account for: the
`QuotaExceeded`-variant question `#1763` still owes (per the Lane 5 DAG,
gated on `#1760`/`#1761` together) can now assume `ConstPool` is not, and
after implementation will never be, one of the `QuotaKind` values that
question needs to reason about.

## 9. Baseline-profile consequence

If REMOVE lands, the future implementation must remove `max_const_pool =
65_536` from `verified_local`, `pure_compute`, and `kernel_bound` alike.
**This is explicitly not a numeric-baseline change to a real resource** -
`docs/spec/quotas.md`'s "Version Review Rule" language about "changing a
baseline profile value in a user-visible execution path" describes
*tuning* an existing, meaningful limit (e.g. raising `max_steps`), which
is a different kind of decision than removing a vocabulary entry that
never bounded anything. This document records that distinction explicitly
so a future reader does not conflate "we changed a safety-relevant number"
with "we deleted dead vocabulary."

## 10. Public API implications (stated plainly)

**Removing `ConstPool` is source-visible.** It is not an internal
refactor. If REMOVE is selected (frozen above), the future implementation
checkpoint must deliberately update:

- `sm-runtime-core`'s public API (the enum variant and struct field
  themselves)
- `tests/golden_snapshots/public_api/sm_runtime_core_lib.txt` (via
  `SM_UPDATE_PUBLIC_API_SNAPSHOTS=1`, explicit human action)
- `docs/spec/quotas.md` (taxonomy list, descriptor-field list, three
  baseline-profile tables)
- any quota-profile tests asserting the field/variant's existence or
  value
- any exhaustive `match`/`matches!` over `QuotaKind` that will fail to
  compile once a variant is removed (Rust's own exhaustiveness check will
  surface every such site as a compile error, not a silent gap)

**The public-api-guard is expected to go RED until the approved golden is
updated in that later implementation checkpoint.** That RED is not
evidence removal is wrong; it is evidence the approved contract changed,
exactly as the guard's own failure message already describes ("update
snapshot only for intentional contract changes").

## 11. Rejected alternatives summary

| Question | Rejected alternative | Why rejected |
|---|---|---|
| Disposition | RE-SCOPE to `RuntimeSymbolTable` | Would create an exact duplicate of the already-correctly-named, already-enforced `QuotaKind::SymbolTable` (§5) |
| Disposition | RE-SCOPE to per-function string table or `SIG0` | Both are decode-time structural bounds owned by `sm-format`, not `RuntimeQuotas`-governed dynamic resources; re-scoping would duplicate an existing cap under a new, less-specific name (§5) |
| Disposition | RESERVED/deprecated compatibility surface | Struct-literal-with-`..Default`-style construction (already the dominant pattern in this codebase) means most call sites would never trigger a `#[deprecated]` warning, preserving the exact discoverability failure this decision exists to close (§6) |
| Disposition | Silent compatibility bridge (map to `SymbolTable`, fake `QuotaExceeded`, etc.) | Not authorized by any identified compatibility authority; reproduces false-readiness under a different name (§7) |

## 12. Effect on Lane 5 / AC4.b

`#1761` remains **OPEN** after this document - a contract was decided,
not implemented. **AC4.b** ("inert/deferred resource concepts are not
presented as enforced runtime quotas") remains **NOT SATISFIED** for
`ConstPool` until the REMOVE mechanics above actually land in a
separately-authorized implementation checkpoint. This document does not
mark `#1761` repaired or closed, does not delete `max_const_pool` or
`QuotaKind::ConstPool`, does not touch any golden snapshot, and does not
touch `#1760`/`#1762`/`#1763`.

**Wait for explicit owner GO before implementation.**

---

## Implementation update

REMOVE landed. `QuotaKind::ConstPool` and `RuntimeQuotas::max_const_pool`
were removed from `crates/sm-runtime-core/src/lib.rs` (the enum variant,
the struct field, all three baseline-profile assignments, and the
`RuntimeQuotas::exceed()` match arm), from the public API golden snapshot
(`tests/golden_snapshots/public_api/sm_runtime_core_lib.txt`), and from
the active specification (`docs/spec/quotas.md`'s taxonomy, descriptor
field list, and all three baseline-profile sections).

`cargo check --workspace --all-targets` immediately after the Rust removal
produced **zero** compile errors anywhere in the workspace - empirically
confirming, not merely inferring, that `ConstPool` had no live downstream
consumer of any kind (§1's fresh sweep already predicted this; this is the
falsification the "STOP if an unexpected consumer appears" rule in the
implementation brief existed to catch, and it did not fire). The public
API guard's own falsification proof was run as specified: with the Rust
surface already removed but the golden snapshot still stale, the guard
failed RED, and its own diff showed the change was precisely `ConstPool,`
and `pub max_const_pool: usize,` disappearing - nothing else. The golden
was then regenerated via the repository's existing
`SM_UPDATE_PUBLIC_API_SNAPSHOTS=1` mechanism, and the resulting diff was
reviewed manually: exactly those same two lines removed, nothing else. A
temporary reinsertion of the enum variant (reverted immediately after)
confirmed the same guard also catches `ConstPool`'s return, not just its
departure.

No compatibility ghost was introduced: no replacement field, no re-scope
to `SymbolTable`/string-table limits/`SIG0`, no deprecated alias.

This addendum records the implementation outcome. The evidence,
falsification analysis, and rejected-alternatives record above remain
unedited as the historical record of why REMOVE was selected.
