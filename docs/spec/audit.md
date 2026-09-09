# PROMETHEUS Audit Specification

Status: draft v0
Owner crate: `prom-audit`

This document defines the current canonical audit and replay metadata contract for Semantic runtime execution.

## Current Audit Surface

Current canonical audit types:

- `AuditSessionMetadata`
- `AuditEventId`
- `AuditEventKind`
- `AuditEvent`
- `AuditTrail`
- `ReplayMetadata`
- `AuditReplayArchive`
- `MultiSessionReplayArchiveSession`
- `MultiSessionReplayArchive`
- `MultiSessionReplayArchiveFormatError`

## Ownership Rule

`prom-audit` owns:

- audit event schema
- centralized audit trail structure
- replay metadata schema
- replay archive envelope shape
- capability denial and host-effect event representation

`prom-audit` does not own:

- execution session orchestration
- state storage invariants
- rule scheduling semantics
- ABI descriptor semantics

## Event Rule

Current event families:

- session start and finish
- capability denial
- gate read and gate write
- pulse emit
- rule activation
- state transition
- free-form notes

Current event invariant:

- audit event ids are assigned monotonically within one audit trail

## Replay Rule

Current replay metadata must include:

- execution context
- effective `RuntimeQuotas` (#1762, FA-08-004) - the actual quota/profile
  envelope that governed the session, not a baseline re-derived from
  execution context; see "Effective Quota Provenance Rule" below
- capability manifest metadata
- whether a gate registry was bound
- event count
- last event id

Current persisted replay rule:

- `AuditReplayArchive` wraps session metadata, recorded events, and replay
  metadata under one explicit archive envelope
- archive metadata is explicit through `format_version`, currently `2`
  (bumped from `1` by #1762 to carry the effective quota envelope - see
  below)
- archive materialization/loading uses one canonical deterministic text envelope
- persisted replay ownership does not widen orchestration or runtime recovery
  semantics by implication

## Effective Quota Provenance Rule

`AuditSessionMetadata` records:

- `ExecutionContext` - a baseline/default selector and audit-class label
- `RuntimeQuotas` - the actual effective quota/profile envelope that
  governed the session, copied from the same `ExecutionConfig` used for
  execution, never re-derived from `context`
- capability manifest metadata
- whether a gate registry was bound

This exists because `ExecutionContext` alone does not prove which
`RuntimeQuotas` values governed a session (`ExecutionConfig::new` permits
custom envelopes; see `docs/spec/quotas.md`). Recording the effective
envelope directly means two sessions sharing the same `ExecutionContext`
but running under different `RuntimeQuotas` are distinguishable in the
audit trail - see `docs/roadmap/stable_foundation/ssf08_1762_execution_envelope_provenance_decision.md`
for the full contract.

Current archive wire rule:

- the canonical `session` line carries all eight `RuntimeQuotas` fields
  (`max_steps`, `max_calls`, `max_stack_depth`, `max_frames`,
  `max_registers`, `max_symbol_table`, `max_effect_calls`,
  `max_debug_symbols_per_function`), in that order, as plain decimal
  tokens, after the pre-existing context/capability-manifest/
  gate-registry-bound fields
- `AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION = 2` archives carry this shape;
  version `1` archives (recorded before this rule existed) never carried
  quota data at all
- a `v2` reader rejects a `v1` archive outright, through the existing
  version-mismatch channel - it never backfills a `v1` archive's missing
  quota data with a context-derived baseline, since the archive itself
  never recorded what actually governed that session
- `v1` archives remain readable only by `v1`-era reader code; this rule
  does not add cross-version parsing
- `MULTI_SESSION_REPLAY_ARCHIVE_FORMAT_VERSION` remains `1` - the outer
  multi-session envelope's own wire shape is unchanged by this rule; it
  embeds independently-versioned `AuditReplayArchive` records, each
  version-checked on its own by the same `v2`/`v1` rule above

This does not imply `RuntimeQuotas` captures every independent
verifier/admission authority - `VerificationLimits`, `sm-format`'s
structural caps, and capability policy remain outside what this recorded
envelope attests to (see `docs/spec/quotas.md`'s own scope note).

Current post-stable owner-layer widening on `main`:

- `prom-audit` now also owns explicit multi-session replay bundle types:
  - `MultiSessionReplayArchiveSession`
  - `MultiSessionReplayArchive`
- canonical text materialization/loading for multi-session replay is now
  admitted through one explicit deterministic envelope
- session bundles remain ordered by `session_ordinal`, which must be monotonic
  from zero
- this widening still does not imply rollback, recovery, or runtime replay
  orchestration

## Boundary Rule

Current architectural rule:

- `prom-runtime` may provide session metadata to initialize an audit trail
- `prom-audit` owns the shape of audit records and replay metadata
- future runtime hooks may emit into this schema, but must not redefine it
