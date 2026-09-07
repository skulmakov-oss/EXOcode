# Runtime Quotas Specification

Status: draft v0
Model owner: `sm-runtime-core`
Enforcement owner: `sm-vm`

## Purpose

Runtime quotas define the bounded execution contract for Semantic programs.

Quota model rule:

- `sm-runtime-core` defines quota taxonomy and baseline profiles
- `sm-vm` enforces quotas during execution
- higher integration layers may choose context-specific quota envelopes, but
  must not weaken the core safety contract silently

## Quota Taxonomy

Current quota kinds:

- `Steps`
- `Calls`
- `StackDepth`
- `Frames`
- `Registers`
- `ConstPool`
- `SymbolTable`
- `EffectCalls`
- `TraceEntries`

Current quota descriptor fields:

- `max_steps`
- `max_calls`
- `max_stack_depth`
- `max_frames`
- `max_registers`
- `max_const_pool`
- `max_symbol_table`
- `max_effect_calls`
- `max_trace_entries`

## Current Baseline Profiles

### `verified_local`

- `max_steps = 100000`
- `max_calls = 16384`
- `max_stack_depth = 256`
- `max_frames = 256`
- `max_registers = 4096`
- `max_const_pool = 65536`
- `max_symbol_table = 16384`
- `max_effect_calls = 1024`
- `max_trace_entries = 8192`

### `pure_compute`

- `max_steps = 100000`
- `max_calls = 16384`
- `max_stack_depth = 256`
- `max_frames = 256`
- `max_registers = 4096`
- `max_const_pool = 65536`
- `max_symbol_table = 16384`
- `max_effect_calls = 0`
- `max_trace_entries = 4096`

### `kernel_bound`

- `max_steps = 250000`
- `max_calls = 32768`
- `max_stack_depth = 256`
- `max_frames = 256`
- `max_registers = 8192`
- `max_const_pool = 65536`
- `max_symbol_table = 16384`
- `max_effect_calls = 4096`
- `max_trace_entries = 16384`

## Context Mapping

Current `ExecutionContext -> RuntimeQuotas` mapping:

- `PureCompute -> pure_compute`
- `VerifiedLocal -> verified_local`
- `RuleExecution -> verified_local`
- `KernelBound -> kernel_bound`

Contract rule:

- context selection is explicit through `ExecutionConfig`
- default execution for standard verified runs is `VerifiedLocal`

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
- quota enforcement: `sm-vm`
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
