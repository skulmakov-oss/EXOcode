# Runtime Quotas Specification

Status: draft v0
Model owner: `sm-runtime-core`
Enforcement owner: `sm-vm` (runtime-resident quotas); `sm-verify`
(`SymbolTable`, checked statically at admission - see "Ownership Rule")

## Purpose

Runtime quotas define the bounded execution contract for Semantic programs.

Quota model rule:

- `sm-runtime-core` defines quota taxonomy and baseline profiles
- `sm-vm` enforces quotas during execution, except `SymbolTable`, which
  `sm-verify` enforces statically, pre-execution, at admission
- higher integration layers may choose context-specific quota envelopes, but
  must not weaken the core safety contract silently - meaning divergence
  from the baseline must be explicit in provenance (recorded, not hidden),
  not that a weaker envelope is itself forbidden; see "Execution Envelope
  Provenance Rule" below (#1762, FA-08-004)

## Quota Taxonomy

Current quota kinds:

- `Steps`
- `Calls`
- `StackDepth`
- `Frames`
- `Registers`
- `SymbolTable`
- `EffectCalls`

Current quota descriptor fields:

- `max_steps`
- `max_calls`
- `max_stack_depth`
- `max_frames`
- `max_registers`
- `max_symbol_table`
- `max_effect_calls`

`RuntimeQuotas` also carries a field that is not part of this taxonomy -
`max_debug_symbols_per_function` - see "Verifier Admission Compatibility
Note" below.

## Current Baseline Profiles

### `verified_local`

- `max_steps = 100000`
- `max_calls = 16384`
- `max_stack_depth = 256`
- `max_frames = 256`
- `max_registers = 4096`
- `max_symbol_table = 16384`
- `max_effect_calls = 1024`

### `pure_compute`

- `max_steps = 100000`
- `max_calls = 16384`
- `max_stack_depth = 256`
- `max_frames = 256`
- `max_registers = 4096`
- `max_symbol_table = 16384`
- `max_effect_calls = 0`

### `kernel_bound`

- `max_steps = 250000`
- `max_calls = 32768`
- `max_stack_depth = 256`
- `max_frames = 256`
- `max_registers = 8192`
- `max_symbol_table = 16384`
- `max_effect_calls = 4096`

## Verifier Admission Compatibility Note

`RuntimeQuotas` also carries `max_debug_symbols_per_function`:

- `verified_local = 8192`
- `pure_compute = 4096`
- `kernel_bound = 16384`

This field:

- is a verifier-side per-function debug-metadata admission limit, enforced
  by `sm-verify::verify_function_code` against a decoded function's
  `debug_symbols.len()`
- is **not** a `QuotaKind` member and is not listed in the quota taxonomy
  above
- is **not** charged by `sm-vm` and has no runtime charge point
- is **not** surfaced as `RuntimeError::QuotaExceeded` - a rejection under
  this limit is a static admission failure
  (`VerificationCode::ResourceLimitExceeded`), not a runtime quota
  exhaustion
- remains a field on `RuntimeQuotas` only to preserve the existing
  per-profile admission plumbing (#1760, FA-08-002); it is intentionally
  kept distinct from the runtime quota taxonomy above rather than merged
  into it

`sm-format` independently enforces a fixed, profile-independent decode-time
structural cap, `MAX_DEBUG_SYMBOLS_PER_FUNCTION = 8192`, before this
verifier-layer check ever runs. `verified_local` (8192) and `kernel_bound`
(16384, which cannot widen the decoder's own fixed 8192 cap) make this
verifier-layer check dead code in practice; `pure_compute` (4096) is
strictly tighter than the decoder's 8192 cap, so this check has real,
distinct authority only for `pure_compute`.

## Context Mapping

Current `ExecutionContext -> RuntimeQuotas` mapping:

- `PureCompute -> pure_compute`
- `VerifiedLocal -> verified_local`
- `RuleExecution -> verified_local`
- `KernelBound -> kernel_bound`

Contract rule:

- context selection is explicit through `ExecutionConfig`
- default execution for standard verified runs is `VerifiedLocal`

## Execution Envelope Provenance Rule

Frozen and implemented (#1762, FA-08-004,
`docs/roadmap/stable_foundation/ssf08_1762_execution_envelope_provenance_decision.md`):

- `ExecutionContext` is a construction-time baseline/default selector (via
  `ExecutionConfig::for_context`) and an audit-class label. It is **not**
  proof, by itself, of which `RuntimeQuotas` values actually governed a
  given execution.
- The `RuntimeQuotas` values carried in `ExecutionConfig.quotas` are the
  actual effective authority for the quota/profile dimensions
  `RuntimeQuotas` represents. This does **not** extend to
  `VerificationLimits` (an orthogonal verifier envelope),
  `sm-format`'s structural decode-time caps, verifier rules unrelated to
  `RuntimeQuotas`, capability policy, or any other independent admission
  authority.
- `ExecutionConfig::new(context, quotas)` remains supported: custom
  envelopes (a `context` paired with `quotas` that diverge from
  `for_context`'s canonical mapping) are permitted, with no
  equality-with-baseline validation and no stricter-than-baseline
  restriction.
- Because custom envelopes are permitted, `prom_runtime::RuntimeSessionDescriptor`
  and `prom_audit::AuditSessionMetadata` record the **effective**
  `RuntimeQuotas` actually used - copied from the same `ExecutionConfig`
  passed to session construction, never re-derived from `context` - so
  that two sessions sharing the same `ExecutionContext` but running under
  different `RuntimeQuotas` remain distinguishable in provenance. See
  `docs/spec/audit.md`'s "Effective Quota Provenance Rule" for the archive
  wire-format consequence.

## Enforcement Rule

Quota enforcement happens at runtime for the resources the VM cannot prove
statically.

Current enforced areas include:

- opcode-dispatch count (`Steps`)
- admitted non-root call count (`Calls`)
- frame count
- effective stack depth
- register growth
- effect-call budget

Quota exhaustion produces `QuotaExceeded { kind, limit, used }`.

Current compatibility note:

- stack-depth quota overflow is still surfaced to callers as `StackOverflow` on
  the VM path
- the stack limit is nonetheless governed by the shared runtime quota contract

`Steps`/`Calls` semantics (charge point, root-frame exemption, exhaustion
timing, and `usize::MAX` overflow discipline) are frozen in
`docs/roadmap/stable_foundation/ssf08_1759_steps_calls_contract_decision.md`.
`Steps` / `Calls` / `EffectCalls` execution counters use overflow-safe
fail-closed charging (a shared `checked_add`-based primitive): ordinary
usage reports the exact attempted value, and the single unrepresentable
case (the increment itself would overflow `usize::MAX`) fails closed with
`used` SATURATED at `usize::MAX` for reporting only - never a silent wrap,
never continued execution.

## Determinism Rule

Quota behavior must be deterministic for the same:

- verified bytecode
- execution config
- runtime entry path

The VM must not:

- continue execution after quota exhaustion
- downgrade quota failure to a warning
- hide which quota kind was exceeded

## Ownership Rule

The following ownership split is mandatory:

- quota taxonomy: `sm-runtime-core`
- quota enforcement: `sm-vm`, except `SymbolTable`, which `sm-verify`
  enforces statically at admission, before execution begins
- session-level quota selection: higher orchestration layers

## Version Review Rule

The following changes require quota contract review:

- changing a quota kind meaning
- adding a new quota kind
- changing a baseline profile value in a user-visible execution path
- changing error-reporting semantics for quota exhaustion

Required follow-up:

1. update this specification
2. update quota-related tests
3. update user-facing reporting if the surfaced contract changed
