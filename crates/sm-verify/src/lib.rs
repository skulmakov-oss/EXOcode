#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use sm_format::semcode_format::{
    read_f64_le, read_i32_le, read_u16_le, read_u32_le, read_u8, Opcode, SemcodeFormatError,
    SemcodeHeaderSpec, CAP_ARGS_READ, CAP_CLOCK_READ, CAP_CLOSURE_VALUES, CAP_DEBUG_SYMBOLS,
    CAP_EVENT_POST, CAP_F64_MATH, CAP_FS_READ, CAP_FS_WRITE, CAP_FX_MATH, CAP_FX_VALUES,
    CAP_GATE_SURFACE, CAP_MAP_VALUES, CAP_OWNERSHIP_ADT_BORROW_PATHS, CAP_OWNERSHIP_FIELD_PATHS,
    CAP_OWNERSHIP_PATHS, CAP_OWNERSHIP_SEQUENCE_PATHS, CAP_PATH_INSPECT, CAP_PRNG,
    CAP_SEQUENCE_ITERATION, CAP_SEQUENCE_VALUES, CAP_STATE_QUERY, CAP_STATE_UPDATE,
    CAP_STDERR_WRITE, CAP_STDIN_READ_TEXT, CAP_STDOUT, CAP_STDOUT_WRITE, CAP_TEXT_VALUES,
    CAP_TIME_DURATION,
};
use sm_runtime_core::RuntimeQuotas;
use std::collections::HashSet;

#[cfg(feature = "std")]
pub mod hello_pending_admission;
#[cfg(feature = "std")]
pub mod hello_real_semcode_admission;

/// #1756 Codex review round 18: a `System`-delegating global allocator
/// that also counts allocation calls per THREAD, used only by `cfg(test)`
/// to give `c1756_structural_decode_never_allocates_operand_storage_for_
/// unreachable_padding` genuine allocator-instrumentation evidence (per
/// the review's own preference for "instrument the operand collector/sink
/// in tests" over wall-clock timing alone) rather than a wall-clock-only
/// proxy. `thread_local!` counting - not a shared atomic - is what makes
/// this safe under `cargo test`'s default parallel test execution: each
/// test thread only ever observes allocations its OWN code caused, never
/// another concurrently-running test's. Behaves identically to `System`
/// in every other respect, so it does not change any other test's
/// observable behavior - only adds a per-thread counter alongside every
/// `alloc`/`realloc` call. Never active outside test builds: ordinary
/// library consumers of this crate see no allocator override at all.
// Kept flat at crate-root scope, with no `pub` modifier at all (rather than
// a nested `mod` with `pub(crate)` items) - Rust's default privacy already
// makes a crate-root item visible to every descendant module, including
// `mod tests` below, and the repository's own public-API-surface snapshot
// test (`tests/public_api_contracts.rs`) flags any line starting with
// `pub`, regardless of restricted visibility like `pub(crate)`, as contract
// drift - these items are genuinely never part of this crate's public API.
#[cfg(all(test, feature = "std"))]
use std::alloc::{GlobalAlloc, Layout, System};
#[cfg(all(test, feature = "std"))]
use std::cell::Cell;

#[cfg(all(test, feature = "std"))]
thread_local! {
    static ALLOC_CALLS: Cell<u64> = const { Cell::new(0) };
}

#[cfg(all(test, feature = "std"))]
struct CountingAllocator;

// SAFETY: every method delegates directly to `System`, which already
// upholds `GlobalAlloc`'s contract; the only addition is a same-thread,
// non-reentrant counter increment around each call.
#[cfg(all(test, feature = "std"))]
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_CALLS.with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[cfg(all(test, feature = "std"))]
fn alloc_tracking_calls() -> u64 {
    ALLOC_CALLS.with(|c| c.get())
}

#[cfg(all(test, feature = "std"))]
#[global_allocator]
static COUNTING_ALLOCATOR: CountingAllocator = CountingAllocator;

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationCode {
    BadHeader,
    UnsupportedVersion,
    TruncatedFunction,
    InvalidFunctionName,
    DuplicateFunction,
    InvalidStringTable,
    InvalidDebugSection,
    InvalidOwnershipSection,
    UnknownOpcode,
    OperandOutOfBounds,
    InvalidJumpTarget,
    InvalidStringReference,
    InvalidRegisterReference,
    UnknownCallTarget,
    ResourceLimitExceeded,
    CapabilityViolation,
    /// The byte range immediately after the string table has more than one
    /// valid canonical structural reading: it can be interpreted as an
    /// empty/populated `DBG0` debug section AND, independently, as a
    /// complete, well-formed instruction stream. Verified SemCode must have
    /// exactly one canonical structural interpretation (see #1731); when
    /// both readings are structurally valid, admission fails closed rather
    /// than silently choosing one.
    AmbiguousInstructionFraming,
    /// The decoded opcode's minimum SemCode header revision
    /// (`Opcode::minimum_semcode_revision`) exceeds the artifact's actual
    /// header revision (see #1732 / FA-05-002). This is a version-identity
    /// gap, not a missing-capability gap: the opcode is structurally valid
    /// and every capability it needs (if any) is present, but the artifact
    /// header predates the revision whose contract actually admits this
    /// opcode's semantics.
    OpcodeRequiresNewerHeader,
    /// A control-flow successor reachable from the function entry falls off
    /// the executable instruction stream instead of reaching another
    /// instruction boundary or an admitted terminal instruction (`RET`).
    /// Closed loops remain admissible; this code does not imply termination.
    ReachableFunctionFallthrough,
    /// The function envelope's `SIG0` callable-signature section is absent,
    /// truncated, or carries an unrecognized parameter-family tag (#1773 /
    /// FA-09-005). Structurally distinct from `UnsupportedVersion`: the
    /// header itself is recognized, but this specific function's signature
    /// record could not be decoded.
    InvalidSignatureSection,
    /// An `Opcode::Call` site's `argc` disagrees with its resolved callee's
    /// canonical parameter count (#1773 / FA-09-005). This is arity
    /// enforcement only - sm-verify cannot prove a CALL argument register's
    /// runtime family statically (registers are untyped storage), so family
    /// mismatches are rejected by the VM before `push_frame` instead.
    CallArgumentCountMismatch,
    /// A register read is reachable from function entry on some execution
    /// path where that register has not been definitely written (#1756 /
    /// FA-07-016). Distinct from `InvalidRegisterReference`: the register
    /// number is in range and within the configured quota - what's missing
    /// is a proof that every incoming path defines it before this read.
    /// Only checked for a function carrying a canonical `SIG0` signature
    /// (`SEMCODE_SIGNATURE_MIN_REVISION`+); a signature-less artifact has no
    /// sound `IN[entry]` to prove against and is unaffected by this code.
    UndefinedRegisterRead,
    /// The definite-register-assignment analysis's (#1756 / FA-07-016)
    /// peak logical analysis-state requirement for one function exceeded
    /// `VerificationLimits::max_state_words` - a deterministic, verifier-
    /// OWNED resource bound checked via `checked` arithmetic BEFORE the
    /// state it accounts for is ever allocated (overflow itself counts
    /// as exceeding it). That requirement is the SUM of several
    /// independently-scaling components - raw reachable-node
    /// representation, leader-compressed dense dataflow state, and fixed
    /// leader/structural/worklist overhead - so any one of them can
    /// trigger this code on its own (for example, a genuinely branch-
    /// free reachable chain can reject on raw-node cost alone while its
    /// leader-compressed state stays negligible). See `VerificationLimits
    /// ::max_state_words`'s own doc comment, `check_raw_node_state_
    /// budget`'s and `check_analysis_state_budget`'s doc comments, and
    /// `docs/spec/verifier.md` for the exact enforced formula - not
    /// duplicated here to avoid a second copy that can drift from the
    /// enforced one. This is a genuinely different resource domain than
    /// `RuntimeQuotas` (#1751: execution resources and static-
    /// verification resources are distinct and must not be conflated)
    /// and different from `ResourceLimitExceeded` above (which reports
    /// pre-existing decode/symbol-table budgets, not this analysis's own
    /// working state). Exceeding this means the artifact could not be
    /// PROVEN within the selected verification resource envelope - it
    /// does not mean the program's semantics are actually invalid, and
    /// admission still fails closed regardless.
    AnalysisStateLimitExceeded,
    /// #1756 Codex review round 13 (owner decision): the same analysis
    /// pass's own deterministic, machine-independent work-unit count
    /// (see `compute_missing_sets`'s work-unit accounting - one unit per
    /// register checked at entry seeding, or per missing-bit delivery
    /// dequeued from the worklist) exceeded
    /// `VerificationLimits::max_work_units` before the dataflow fixed
    /// point converged. Distinct from `AnalysisStateLimitExceeded`: this
    /// can fire even when the state budget was never in danger, if
    /// reaching convergence itself takes more propagation steps than the
    /// configured envelope allows (a branch-dense CFG with a slowly-
    /// narrowing register domain - see the round-13 finding). Never
    /// inferred from wall-clock time, allocator behavior, or CPU cycles.
    AnalysisWorkLimitExceeded,
    /// #1726 Checkpoint D2b: a `StoreVarSite(anchor)` Borrow activation
    /// (SEMCOD20/rev21+) failed structural admission against this same
    /// function's own decoded instruction stream. Exactly one of: the
    /// anchor is not one of this function's own canonical instruction-start
    /// offsets (this same verification walk's `instr_starts`, the identical
    /// authority `InvalidDebugSection`'s own `sym.pc` check already reuses -
    /// never a raw `code[offset] == StoreVar` byte compare, which an operand
    /// byte could coincidentally satisfy); the canonical instruction at that
    /// offset is not `Opcode::StoreVar`; or two distinct Borrow events in
    /// the same function declare the identical anchor (the frozen
    /// producers' own emission contract, including dynamic-root dedup,
    /// never legitimately produces this - see the design note on
    /// `validate_storevar_site_anchors` for the audited proof).
    ///
    /// Failure-model boundary, stated explicitly so it is never read as
    /// stronger than it is: this proves the anchor is a genuine, in-range,
    /// correctly-typed executable instruction boundary in the declaring
    /// function - an artifact-level structural fact. It does NOT prove that
    /// instruction is the compiler's original source-level introduction
    /// site rather than some other admissible StoreVar a hostile producer
    /// chose to point at instead; the current rev21 wire format carries no
    /// additional provenance for that stronger claim, and D2b does not
    /// pretend otherwise. Reconstructing source provenance from bytecode
    /// alone is out of scope for this code path.
    ///
    /// #1891 Checkpoint W2E extends this exact same code and the exact same
    /// trust boundary to rev21+ Write execution sites
    /// (`DecodedWriteExecution::StoreVarSite`/`MakeRecordSite`, see
    /// `validate_write_execution_site_anchors`): a rev21 Write event names
    /// an exact executable instruction of the compiler-declared class
    /// (StoreVar or MakeRecord) within the same function. It does not
    /// reconstruct or prove source-level ownership provenance between the
    /// AccessPath and that instruction - for example, that a MakeRecord's
    /// declared field-override paths correspond to fields actually
    /// overridden in source, or that a Write path's root/path matches the
    /// original source assignment. That is the same trust boundary already
    /// accepted for Borrow above, now shared by both domains without either
    /// one gaining authority over the other's anchors.
    InvalidOwnershipAnchor,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationDiagnostic {
    pub code: VerificationCode,
    pub function: Option<String>,
    pub offset: Option<usize>,
    pub message: String,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectReport {
    pub diagnostics: Vec<VerificationDiagnostic>,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedFunction {
    pub name: String,
    pub code_len: usize,
    pub string_count: usize,
    pub debug_symbol_count: usize,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedProgram {
    pub header: SemcodeHeaderSpec,
    pub functions: Vec<VerifiedFunction>,
}

/// #1756 Codex review round 13 (owner decision): a deterministic,
/// verifier-OWNED resource envelope for static analysis passes such as
/// definite-register-assignment (#1756 / FA-07-016) - entirely distinct
/// from `RuntimeQuotas` (`sm_runtime_core` - execution resources only,
/// see #1751) and from the pre-existing decode/symbol-table budgets
/// reported via `VerificationCode::ResourceLimitExceeded`. There are
/// three distinct resource domains - artifact/decode resources,
/// verifier-analysis resources, runtime-execution resources - and they
/// must never be conflated (see `docs/spec/verifier.md`).
///
/// `verify_semcode_token`/`verify_semcode_token_with_quotas`/
/// `verify_semcode` (the convenience entrypoints) admit against
/// `VerificationLimits::default_profile()`; use
/// `verify_semcode_token_with_quotas_and_limits` to admit against an
/// explicit profile instead. No verifier limit here is ever inferred
/// from `RuntimeQuotas::max_steps`/`max_calls` or any other execution
/// quota.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationLimits {
    /// Upper bound, in canonical 8-byte logical words, on the definite-
    /// register-assignment analysis's PEAK simultaneously-live verifier-
    /// analysis state for one function - not process RSS or an
    /// allocator-exact byte count, and not cumulative across an
    /// artifact's functions (see `max_work_units` below for the
    /// genuinely cumulative counterpart).
    ///
    /// As of round 21 of #1756's own review history, this covers BOTH:
    /// raw reachable-node representation (`reachable_indices`, `reads_
    /// flat`/`writes_flat` and their offset tables, `is_leader`,
    /// `compute_leaders`'s own classification-state scratch array - all
    /// scaling with `reachable_count` and the TOTAL ENCODED BYTES OF
    /// REACHABLE INSTRUCTIONS specifically (`reachable_instruction_
    /// bytes`) - deliberately NOT the total structural instruction-
    /// section length, which may also include structurally valid but
    /// unreachable trailing code this analysis never decodes (round 17's
    /// correction) - checked by `check_raw_node_state_budget` BEFORE any
    /// of them is allocated), and leader-compressed
    /// dense dataflow state (`chain_killed`/`missing` and their headers,
    /// `chain_targets`, `leader_positions`, the worklist stack's proven
    /// peak - scaling with `leader_count` and `ceil(register_domain_
    /// size / 64)`, checked by `check_analysis_state_budget`). Every
    /// logical size is a canonical, MACHINE-INDEPENDENT constant, never
    /// a host's actual `size_of::<T>()`, so admission is deterministic
    /// across hosts regardless of target pointer width. See `check_raw_
    /// node_state_budget`'s and `check_analysis_state_budget`'s own doc
    /// comments for the full structural inventory and exact formula.
    /// Every accounting step uses `checked` arithmetic; overflow anywhere
    /// counts as exceeding this limit, since there is no finite word
    /// count to compare against in that case.
    pub max_state_words: usize,
    /// Upper bound on the analysis's own deterministic work-unit count
    /// (see `compute_missing_sets`'s work-unit accounting), never
    /// inferred from wall-clock time, allocator behavior, or CPU cycles:
    /// a purely combinatorial count of meaningful lattice-propagation
    /// steps, machine-independent by construction.
    pub max_work_units: usize,
}

#[cfg(feature = "std")]
impl VerificationLimits {
    /// #1756 Codex review round 13: CANDIDATE default profile pending
    /// explicit owner sign-off - not a frozen release value. Selected
    /// from the benchmark table in round 13's PR reply/commit message
    /// (ordinary compiler-generated programs, the largest existing
    /// golden fixtures, the #1756 adversarial stress fixtures, and
    /// synthetic leader-count sweeps at register domains of 64, 4,096,
    /// and 8,192 - `max_registers`'s own ceiling, so `domain_size` can
    /// never legitimately exceed it).
    ///
    /// **Re-validated in round 14** after two accounting corrections
    /// changed what these same numbers actually admit (see the round-14
    /// PR reply/commit message for the updated table): `max_state_words`
    /// now bounds `(2*leader_count + 1) * word_count(domain_size)` (P2 -
    /// `chain_killed` and `missing` are simultaneously live, not just
    /// one array), roughly HALVING the genuine-leader count this same
    /// number admits at any given `domain_size` versus round 13's
    /// formula - at `domain_size = 8,192` (the largest legal domain),
    /// down from ~65,536 to ~32,767 genuine leaders, still several
    /// orders of magnitude above every ordinary/golden/already-committed
    /// -adversarial shape measured (at most tens of thousands of state
    /// words) and comfortably above the round-13 stress fixture's own
    /// ~2.56M words at that same domain size. `max_work_units` now
    /// bounds the WHOLE ARTIFACT (P1), not one function - a many-
    /// function program sharing this one budget is a materially
    /// stricter (more protective) constraint than round 13's per-
    /// function reset ever was, and remains comfortably above realistic
    /// multi-function programs (thousands of ordinary functions cost
    /// only millions of units combined) while now correctly bounding the
    /// "many expensive functions in one artifact" shape P1 exists to
    /// close. Kept UNCHANGED numerically pending owner confirmation - see
    /// the round-14 reply for the honest re-validation against both
    /// corrected formulas rather than a silent bump to keep old stress
    /// tests passing.
    ///
    /// **Re-validated again in round 15** after `max_state_words` gained
    /// a third accounting term: fixed per-leader/stack structural
    /// overhead (`check_analysis_state_budget`'s own doc comment has the
    /// full inventory), closing a gap that specifically hurt LOW-
    /// `domain_size`, HIGH-`leader_count` artifacts - round 14's dense-
    /// payload-only formula scaled with `domain_size`, so a `domain_size
    /// = 1` artifact could drive `leader_count` into the millions before
    /// that term alone objected, even though each leader still costs
    /// real, fixed bytes regardless of how few registers are live.
    /// `max_state_words = 8_388_608` (UNCHANGED numerically) now admits,
    /// exactly (`required_words <= max_state_words` at the reported
    /// `leader_count`, `Err` one leader past it):
    ///
    /// | `domain_size` | max genuine `leader_count` |
    /// |---|---|
    /// | 1 | 559,240 |
    /// | 64 | 559,223 |
    /// | 4,096 | 59,377 |
    /// | 8,192 (largest legal) | 31,062 |
    ///
    /// Still several orders of magnitude above every ordinary/golden/
    /// already-committed-adversarial shape measured (at most tens of
    /// thousands of state words), and the `domain_size = 1` ceiling in
    /// particular is now a REAL, meaningful bound rather than an
    /// effectively-unbounded one - round 14's formula alone would have
    /// admitted `leader_count` up to ~4.19 million at `domain_size = 1`,
    /// about 7.5x more than round 15's corrected 559,240, matching the
    /// severity of the gap this round closes. Kept numerically UNCHANGED
    /// again - see the round-15 reply for the full re-derivation.
    ///
    /// **Re-validated again in round 16** after `max_state_words` gained
    /// a fourth accounting term: raw reachable-node representation
    /// (`check_raw_node_state_budget`'s own doc comment has the full
    /// inventory), closing a gap round 15 explicitly, but incorrectly,
    /// excluded - a genuinely branch-FREE reachable chain has `leader_
    /// count = 1` regardless of length, so rounds 13-15's leader/domain-
    /// scaled formula alone charged almost nothing for it no matter how
    /// large `reachable_count` grew. `max_state_words = 8_388_608`
    /// (UNCHANGED numerically) now admits, exactly:
    ///
    /// | shape | max genuine size |
    /// |---|---|
    /// | long-linear (no branches), `domain_size = 1` | 2,684,347 reachable nodes |
    /// | genuine diamonds, `domain_size = 1` | 453,437 leaders |
    /// | genuine diamonds, `domain_size = 64` | 453,423 leaders |
    /// | genuine diamonds, `domain_size = 4,096` | 57,937 leaders |
    /// | genuine diamonds, `domain_size = 8,192` (largest legal) | 30,663 leaders |
    ///
    /// (the long-linear row has no meaningful "leader_count" - it stays 1
    /// regardless of length, so its own ceiling is reported in reachable
    /// nodes instead, the dimension raw-node accounting actually bounds).
    /// At `max_state_words * 8 = 67,108,864` logical bytes (64 MiB) worst
    /// case. Every genuine-diamond ceiling above is LOWER than round 15's
    /// corresponding number (e.g. `domain_size = 8,192`: 30,663 versus
    /// round 15's 31,062) since raw-node cost now competes for the same
    /// budget as leader-state cost, for the SAME reason `domain_size = 1`
    /// dropped from round 14's ~4.19 million to round 15's 559,240 - this
    /// is the intended, honest effect of closing a real gap, not
    /// regression. All ceilings remain several orders of magnitude above
    /// every ordinary/golden/already-committed-adversarial shape measured
    /// (at most tens of thousands of state words). Kept numerically
    /// UNCHANGED again - see the round-16 reply for the full re-
    /// derivation and the exact relationships used to compute this table.
    ///
    /// **Re-checked in round 17** after `raw_words` was corrected to
    /// scale with `reachable_instruction_bytes` (total encoded bytes of
    /// REACHABLE instructions only) rather than `instr_len` (the total
    /// structural instruction-section length, unreachable padding
    /// included) - every construction in the table above is fully
    /// reachable, with zero unreachable padding, so `reachable_
    /// instruction_bytes == instr_len` exactly for each of them; this
    /// table's numbers are therefore UNCHANGED by round 17, confirmed by
    /// the full c1756 suite continuing to pass with the SAME hardcoded
    /// exact numbers after the round-17 fix, with no test needing
    /// recomputation.
    ///
    /// **Re-validated again in round 21** after `raw_words` gained a
    /// fifth accounting term: `compute_leaders`'s own consolidated
    /// classification-state scratch array (`check_raw_node_state_
    /// budget`'s own doc comment has the full inventory), one `u8` per
    /// reachable node, simultaneously live with `is_leader` while
    /// `is_leader` is being built. `max_state_words = 8_388_608`
    /// (UNCHANGED numerically) now admits, exactly (measured directly
    /// against `check_raw_node_state_budget`/`check_analysis_state_
    /// budget`, not hand-derived - see the round-21 PR reply for the
    /// calibration method):
    ///
    /// | shape | max genuine size |
    /// |---|---|
    /// | long-linear (no branches), `domain_size = 1` | 2,581,103 reachable nodes |
    /// | genuine diamonds, `domain_size = 1` | 450,393 leaders |
    /// | genuine diamonds, `domain_size = 64` | 450,379 leaders |
    /// | genuine diamonds, `domain_size = 4,096` | 57,887 leaders |
    /// | genuine diamonds, `domain_size = 8,192` (largest legal) | 30,649 leaders |
    ///
    /// Every ceiling above is LOWER than round 16/17's corresponding
    /// number (e.g. `domain_size = 8,192`: 30,649 versus 30,663) since
    /// the new classifier-scratch term now also competes for the same
    /// budget - the same kind of honest, intended tightening round 16
    /// itself produced over round 15. Still several orders of magnitude
    /// above every ordinary/golden/already-committed-adversarial shape
    /// measured (at most tens of thousands of state words). Kept
    /// numerically UNCHANGED again - see the round-21 reply for the full
    /// re-derivation.
    pub const fn default_profile() -> Self {
        Self {
            max_state_words: 8_388_608, // 2^23 - logical verifier-analysis words, worst case
            max_work_units: 536_870_912, // 2^29 - ~537M lattice-propagation steps, whole artifact
        }
    }
}

#[cfg(feature = "std")]
impl Default for VerificationLimits {
    fn default() -> Self {
        Self::default_profile()
    }
}

#[cfg(feature = "std")]
impl core::fmt::Display for RejectReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (idx, diag) in self.diagnostics.iter().enumerate() {
            if idx > 0 {
                writeln!(f)?;
            }
            write!(f, "verify error [{:?}]", diag.code)?;
            if let Some(function) = &diag.function {
                write!(f, " in '{}'", function)?;
            }
            if let Some(offset) = diag.offset {
                write!(f, " @0x{:04x}", offset)?;
            }
            write!(f, ": {}", diag.message)?;
        }
        Ok(())
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RejectReport {}

/// Canonical admission token for a verified SemCode artifact.
///
/// Instances of this type represent SemCode bytes that have successfully
/// passed the `verify_semcode_token` admission gate. This forms the first
/// boundary in the canonical verified execution path.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct VerifiedSemCode<'a> {
    bytes: &'a [u8],
    program: VerifiedProgram,
    decoded: Vec<sm_format::semcode_decode::DecodedFunctionEnvelope<'a>>,
}

/// Error type for canonical entry resolution.
///
/// This error indicates that a `VerifiedSemCode` artifact does not contain
/// the required entry function, preventing it from proceeding to the
/// `VerifiedEntrySemCode` boundary.
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryResolutionError {
    MissingEntry { entry: String },
}

#[cfg(feature = "std")]
impl core::fmt::Display for EntryResolutionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EntryResolutionError::MissingEntry { entry } => {
                write!(f, "entry function '{}' not found in SemCode", entry)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EntryResolutionError {}

/// Canonical entry-resolved token for a verified SemCode artifact.
///
/// Instances of this type represent a verified artifact that has also
/// been confirmed to contain a specific entry point function. This forms
/// the second boundary in the canonical verified execution path, directly
/// preceding execution in the `sm-vm` layer.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct VerifiedEntrySemCode<'token, 'bytes> {
    artifact: &'token VerifiedSemCode<'bytes>,
    entry: String,
}

#[cfg(feature = "std")]
impl<'token, 'bytes> VerifiedEntrySemCode<'token, 'bytes> {
    pub fn artifact(&self) -> &'token VerifiedSemCode<'bytes> {
        self.artifact
    }

    pub fn entry(&self) -> &str {
        &self.entry
    }

    pub fn bytes(&self) -> &'bytes [u8] {
        self.artifact.bytes()
    }

    pub fn program(&self) -> &VerifiedProgram {
        self.artifact.program()
    }
}

#[cfg(feature = "std")]
impl<'a> VerifiedSemCode<'a> {
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn program(&self) -> &VerifiedProgram {
        &self.program
    }

    pub fn function_names(&self) -> impl Iterator<Item = &str> {
        self.decoded.iter().map(|env| env.name.as_str())
    }

    pub fn has_entry(&self, entry: &str) -> bool {
        self.decoded.iter().any(|env| env.name == entry)
    }

    /// Canonical entry resolution gate.
    ///
    /// Upgrades this admission token to a `VerifiedEntrySemCode` token
    /// if the specified entry function is present in the verified artifact.
    pub fn require_entry<'token>(
        &'token self,
        entry: &str,
    ) -> Result<VerifiedEntrySemCode<'token, 'a>, EntryResolutionError> {
        if self.has_entry(entry) {
            Ok(VerifiedEntrySemCode {
                artifact: self,
                entry: entry.to_string(),
            })
        } else {
            Err(EntryResolutionError::MissingEntry {
                entry: entry.to_string(),
            })
        }
    }

    /// Intentionally public, hidden implementation seam for VM preparation.
    ///
    /// `sm-vm` (a separate workspace crate) calls this to bridge a verified
    /// token into VM-executable state (`prepare_verified_execution` in
    /// `crates/sm-vm/src/semcode_vm.rs`). Rust has no visibility level
    /// narrower than `pub` that still permits a separate crate to call a
    /// method, so this is genuinely part of `sm-verify`'s externally
    /// callable Rust surface -- `#[doc(hidden)]` only suppresses it from
    /// generated documentation, it does not restrict downstream crates from
    /// naming and calling it. It is not recommended for use outside this
    /// workspace's own crates. The closure-based shape only prevents a
    /// caller from retaining the borrowed `&'scope` slice itself past this
    /// call; it is not a containment guarantee -- `DecodedFunctionEnvelope`
    /// derives `Clone`, so a closure can return owned copies of the decoded
    /// data, and `sm_format::semcode_decode::decode_semcode_envelope` is
    /// itself public, so any caller with the same bytes can independently
    /// reconstruct equivalent decoded envelopes without this method at all.
    #[cfg(feature = "std")]
    #[doc(hidden)]
    pub fn with_decoded_envelopes<R, F>(&self, f: F) -> R
    where
        F: for<'scope> FnOnce(
            &'scope sm_format::semcode_format::SemcodeHeaderSpec,
            &'scope [sm_format::semcode_decode::DecodedFunctionEnvelope<'a>],
        ) -> R,
    {
        f(&self.program.header, &self.decoded)
    }
}

/// Canonical admission gate for SemCode bytes.
///
/// This is the primary entry point for the canonical verified execution path.
/// It validates raw SemCode bytes against admission policies and returns a
/// `VerifiedSemCode` token if successful. Admits against the `VerifiedLocal`
/// quota profile; use `verify_semcode_token_with_quotas` to admit against a
/// different (e.g. `KernelBound`) profile.
#[cfg(feature = "std")]
pub fn verify_semcode_token(bytes: &[u8]) -> Result<VerifiedSemCode<'_>, RejectReport> {
    verify_semcode_token_with_quotas(bytes, RuntimeQuotas::verified_local())
}

/// Admission gate for SemCode bytes against an explicit quota profile.
///
/// Identical to `verify_semcode_token` except the caller supplies the
/// `RuntimeQuotas` admission is checked against, instead of the hardcoded
/// `VerifiedLocal` profile. This lets a context (e.g. `KernelBound`) admit
/// tokens whose register/symbol usage exceeds `VerifiedLocal`'s budget but
/// stays within its own. The returned token's admission proof reflects only
/// the quotas passed here - it says nothing about any other profile.
///
/// #1756 Codex review round 13 (owner decision): admits against
/// `VerificationLimits::default_profile()` - use
/// `verify_semcode_token_with_quotas_and_limits` to admit against an
/// explicit verification-resource profile instead. `RuntimeQuotas` and
/// `VerificationLimits` are deliberately separate parameters, not
/// combined into one profile: they govern different resource domains
/// (execution vs. static-verification, see `VerificationLimits`'s own
/// doc comment) and must not be conflated.
#[cfg(feature = "std")]
pub fn verify_semcode_token_with_quotas(
    bytes: &[u8],
    quotas: RuntimeQuotas,
) -> Result<VerifiedSemCode<'_>, RejectReport> {
    verify_semcode_token_with_quotas_and_limits(
        bytes,
        quotas,
        VerificationLimits::default_profile(),
    )
}

/// Admission gate for SemCode bytes against an explicit quota profile AND
/// an explicit verification-resource profile.
///
/// #1756 Codex review round 13 (owner decision): the quota-aware entry
/// point that binds the exact `VerificationLimits` admission is checked
/// against, for callers that need control over the definite-register-
/// assignment analysis's own resource envelope (`max_state_words`,
/// `max_work_units`) rather than the default profile
/// `verify_semcode_token_with_quotas` uses. `quotas` and `limits` remain
/// two separate parameters on purpose - see `VerificationLimits`'s own
/// doc comment for why they must not be merged into one profile.
#[cfg(feature = "std")]
pub fn verify_semcode_token_with_quotas_and_limits(
    bytes: &[u8],
    quotas: RuntimeQuotas,
    limits: VerificationLimits,
) -> Result<VerifiedSemCode<'_>, RejectReport> {
    let mut diagnostics = Vec::new();
    // #1756 Codex review round 14 (owner clarification): `max_work_units`
    // is the budget for this ENTIRE artifact verification call, not a
    // fresh allowance per function - created ONCE here and threaded by
    // `&mut` reference into every signature-bearing function's analysis
    // below, never recreated or reset. See `AnalysisWorkMeter`.
    let mut work_meter = AnalysisWorkMeter::new(limits.max_work_units);

    let (header, decoded_functions) =
        match sm_format::semcode_decode::decode_semcode_envelope(bytes) {
            Ok(v) => v,
            Err(err) => {
                let diag = match err {
                    sm_format::semcode_decode::DecodeError::BadHeader => diag(
                        VerificationCode::BadHeader,
                        None,
                        None,
                        "SemCode file is shorter than the 8-byte header",
                    ),
                    sm_format::semcode_decode::DecodeError::UnsupportedVersion {
                        found, ..
                    } => diag(
                        VerificationCode::UnsupportedVersion,
                        None,
                        Some(0),
                        format!("unsupported SemCode header '{}'", found),
                    ),
                    sm_format::semcode_decode::DecodeError::TruncatedFunction { offset, msg } => {
                        diag(VerificationCode::TruncatedFunction, None, Some(offset), msg)
                    }
                    sm_format::semcode_decode::DecodeError::InvalidFunctionName { offset, msg } => {
                        diag(
                            VerificationCode::InvalidFunctionName,
                            None,
                            Some(offset),
                            msg,
                        )
                    }
                    sm_format::semcode_decode::DecodeError::InvalidStringTable { offset, msg } => {
                        diag(
                            VerificationCode::InvalidStringTable,
                            None,
                            Some(offset),
                            msg,
                        )
                    }
                    sm_format::semcode_decode::DecodeError::InvalidDebugSection { offset, msg } => {
                        diag(
                            VerificationCode::InvalidDebugSection,
                            None,
                            Some(offset),
                            msg,
                        )
                    }
                    sm_format::semcode_decode::DecodeError::InvalidOwnershipSection {
                        offset,
                        msg,
                    } => diag(
                        VerificationCode::InvalidOwnershipSection,
                        None,
                        Some(offset),
                        msg,
                    ),
                    sm_format::semcode_decode::DecodeError::InvalidSignatureSection {
                        offset,
                        msg,
                    } => diag(
                        VerificationCode::InvalidSignatureSection,
                        None,
                        Some(offset),
                        msg,
                    ),
                    sm_format::semcode_decode::DecodeError::ResourceLimit { offset, msg } => diag(
                        VerificationCode::ResourceLimitExceeded,
                        None,
                        Some(offset),
                        msg,
                    ),
                };
                diagnostics.push(diag);
                return Err(RejectReport { diagnostics });
            }
        };

    let mut functions = Vec::new();
    let mut pending_functions = Vec::new();
    let mut seen_names = HashSet::new();

    for env in &decoded_functions {
        let function_start = env.name_offset;
        let name = env.name.clone();

        if !seen_names.insert(name.clone()) {
            diagnostics.push(diag(
                VerificationCode::DuplicateFunction,
                Some(name.clone()),
                Some(function_start),
                format!("duplicate function '{}'", name),
            ));
            break;
        }

        match verify_function_code(env, &header, &quotas, &limits, &mut work_meter) {
            Ok(function) => {
                functions.push(function.verified.clone());
                pending_functions.push(function);
            }
            Err(report) => {
                // #1756 Codex review round 14 (owner decision): resource
                // exhaustion means the proof cannot continue inside the
                // selected verifier resource envelope, so admission
                // terminates fail-closed immediately here - it must NOT
                // continue verifying further functions merely to surface
                // more diagnostics (every additional function would only
                // spend MORE of an already-exhausted shared budget). Every
                // OTHER per-function rejection reason keeps the existing,
                // deliberate "collect every function's diagnostics in one
                // pass" behavior unchanged.
                let work_exhausted = report
                    .diagnostics
                    .iter()
                    .any(|d| d.code == VerificationCode::AnalysisWorkLimitExceeded);
                diagnostics.extend(report.diagnostics);
                if work_exhausted {
                    return Err(RejectReport { diagnostics });
                }
            }
        }
    }

    // #1754 (FA-07-014): the VM builds ONE program-wide `RuntimeSymbolTable`
    // shared across every function (see `build_vm_program_view_from_decoded`
    // in crates/sm-vm/src/semcode_vm.rs), interning each function's
    // `env.strings` into it and deduplicating by exact string value. A
    // per-function check against `max_symbol_table` cannot catch a program
    // that stays under the budget in every individual function but exceeds
    // it once all functions' distinct strings are unioned program-wide, so
    // the check has to be made at this granularity to match the real
    // runtime resource.
    let unique_runtime_symbols: HashSet<&str> = decoded_functions
        .iter()
        .flat_map(|env| env.strings.iter().map(|s| s.as_str()))
        .collect();
    let unique_runtime_symbol_count = unique_runtime_symbols.len();
    if unique_runtime_symbol_count > quotas.max_symbol_table {
        diagnostics.push(diag(
            VerificationCode::ResourceLimitExceeded,
            None,
            None,
            format!(
                "program-wide runtime symbol table uses {} distinct entries, exceeding the symbol budget of {}",
                unique_runtime_symbol_count, quotas.max_symbol_table
            ),
        ));
    }

    if header.capabilities & CAP_OWNERSHIP_PATHS != 0
        && !pending_functions
            .iter()
            .any(|function| function.has_ownership_section)
    {
        diagnostics.push(diag(
            VerificationCode::InvalidOwnershipSection,
            None,
            None,
            "header advertises ownership-path metadata but no OWN0 section is present",
        ));
    }

    let known_functions = pending_functions
        .iter()
        .map(|function| function.verified.name.as_str())
        .collect::<HashSet<_>>();
    // #1773 (FA-09-005): callee name -> its decoded canonical signature, so
    // the cross-function pass below can enforce arity. `None` for a known
    // callee means the artifact's header predates canonical signatures
    // (`SEMCODE_SIGNATURE_MIN_REVISION`) - nothing to check, matching how
    // pre-#1773 artifacts already behaved.
    let signatures_by_name: std::collections::HashMap<
        &str,
        &sm_format::semcode_format::CallableSignature,
    > = decoded_functions
        .iter()
        .filter_map(|env| env.signature.as_ref().map(|sig| (env.name.as_str(), sig)))
        .collect();
    for function in &pending_functions {
        for (offset, callee, allows_builtin, argc) in &function.call_targets {
            if known_functions.contains(callee.as_str()) {
                // #1773 (FA-09-005): arity enforcement. Family/runtime-type
                // enforcement is the VM's responsibility before `push_frame`
                // (see `validate_call_arguments` in sm-vm) - sm-verify
                // cannot know a CALL argument register's runtime family
                // (registers are untyped storage, see the #1773 architecture
                // checkpoint), so it enforces only what it can prove
                // statically: parameter count.
                if let (Some(argc), Some(signature)) =
                    (argc, signatures_by_name.get(callee.as_str()))
                {
                    if *argc != signature.families.len() {
                        diagnostics.push(diag(
                            VerificationCode::CallArgumentCountMismatch,
                            Some(function.verified.name.clone()),
                            Some(*offset),
                            format!(
                                "call to '{}' passes {} argument(s), but its canonical signature declares {} parameter(s)",
                                callee,
                                argc,
                                signature.families.len()
                            ),
                        ));
                    }
                }
                continue;
            }

            if *allows_builtin {
                if let Some(required_capabilities) = builtin_call_required_capabilities(callee) {
                    if header.capabilities & required_capabilities != required_capabilities {
                        diagnostics.push(diag(
                            VerificationCode::CapabilityViolation,
                            Some(function.verified.name.clone()),
                            Some(*offset),
                            format!(
                                "builtin call target '{}' requires capability bits 0x{required_capabilities:08x}",
                                callee
                            ),
                        ));
                    }
                    continue;
                }
            }

            diagnostics.push(diag(
                VerificationCode::UnknownCallTarget,
                Some(function.verified.name.clone()),
                Some(*offset),
                format!(
                    "call target '{}' does not resolve to any function in this SemCode program",
                    callee
                ),
            ));
        }
    }

    if diagnostics.is_empty() {
        Ok(VerifiedSemCode {
            bytes,
            program: VerifiedProgram { header, functions },
            decoded: decoded_functions,
        })
    } else {
        Err(RejectReport { diagnostics })
    }
}

/// Legacy admission gate (returns verified program model directly).
///
/// Prefer `verify_semcode_token` for canonical execution flows that require
/// the token-based boundary pattern.
#[cfg(feature = "std")]
pub fn verify_semcode(bytes: &[u8]) -> Result<VerifiedProgram, RejectReport> {
    verify_semcode_token(bytes).map(|token| token.program.clone())
}

/// Returns true if `code[start..]` decodes as a complete, well-formed
/// instruction stream: every opcode byte is valid, every operand's byte
/// shape is fully present (including count/flag-controlled variable-length
/// operands), and the walk consumes exactly to `code.len()` with nothing
/// left over.
///
/// This is a STRUCTURAL question only - it must not depend on whether the
/// resulting operand *values* happen to be semantically canonical (e.g. a
/// bool literal byte of 2, or a presence flag of 5). Ambiguous framing is a
/// fact about the bytes' shape: a decoder that doesn't apply the verifier's
/// current canonical-domain policy (an older tool version, a disassembler,
/// a different implementation) could still read this range as instructions.
/// Coupling "was this ambiguous" to "is sm-verify's semantic policy today
/// willing to admit the alternative reading" would make the one-canonical-
/// interpretation invariant depend on semantic policy instead of being a
/// decoder-level fact, and would silently keep the DBG0 reading whenever
/// the competing instruction reading merely contains a non-canonical
/// literal - exactly the kind of hidden-content risk #1731 exists to close.
///
/// Reuses `decode_operands` - the same function, same opcode-shape match,
/// that the normal admission walk uses - with canonical-domain enforcement
/// turned off via its `enforce_canonical_domains` parameter, rather than a
/// second, independently-maintained opcode-shape table. Every byte read and
/// every count/flag-controlled branch is identical to the semantic-
/// admission walk; only the four inline out-of-domain rejections (LOAD_Q,
/// LOAD_BOOL, CALL/CLOSURE_CALL dst-present flag, RET src-present flag) are
/// skipped here.
#[cfg(feature = "std")]
fn instruction_stream_parses_fully(name: &str, code: &[u8], start: usize) -> bool {
    let mut cursor = start;
    while cursor < code.len() {
        let offset = cursor - start;
        let Ok(raw_opcode) = read_u8(code, &mut cursor) else {
            return false;
        };
        let Ok(opcode) = Opcode::from_byte(raw_opcode) else {
            return false;
        };
        if decode_operands(
            name,
            code,
            &mut cursor,
            offset,
            opcode,
            false,
            &mut NoCollect,
            &mut NoCollect,
        )
        .is_err()
        {
            return false;
        }
    }
    cursor == code.len()
}

/// #1726 Checkpoint D2b: same-function structural admission for every
/// rev21+ `StoreVarSite(anchor)` Borrow activation. Mirrors the `#1746`
/// `DebugSymbol.pc` check in `verify_function_code`: `store_var_starts` is
/// exactly the subset of this function's own `instr_starts` whose decoded
/// opcode is `StoreVar` (built in the same decode pass, see
/// `verify_function_code`), so a single `binary_search` against it proves
/// both "genuine canonical instruction boundary" and "opcode is StoreVar"
/// at once - never a raw `code[anchor] == StoreVar` byte compare, which an
/// operand byte could coincidentally satisfy. Anchors are function-relative
/// by construction: `store_var_starts` was built only from this function's
/// own `code`/`instr_start` slice, so a cross-function or absolute-artifact
/// offset is rejected simply by never matching an entry in it.
///
/// Duplicate anchors across two distinct Borrow events in the same function
/// are rejected unconditionally. This is not a heuristic:
/// `emit_semcode_function` resolves every Borrow event's anchor from
/// `activation_anchors`, a `HashMap<ActivationSiteId, ExecutableAnchor>`
/// populated under `crate::passes::validate_activation_sites`'s proven
/// bijection (exactly one `ActivationSiteId` per Borrow event, exactly one
/// per StoreVar instruction) plus the emission loop's own insert-once check
/// — and instruction start offsets are strictly increasing byte positions in
/// one instruction stream, so two distinct StoreVar instructions can never
/// share a start offset. No frozen producer, including dynamic-root dedup,
/// can legitimately emit two Borrow events sharing one anchor; this check
/// exists to fail closed on a hostile or corrupted artifact that declares
/// one anyway, not to guard against a real compiler output shape.
///
/// Failure-model boundary, restated from
/// `VerificationCode::InvalidOwnershipAnchor`: this proves the anchor is a
/// genuine, in-range StoreVar instruction boundary belonging to this
/// function - an artifact-level structural fact. It does NOT prove that
/// instruction is the compiler's original source-level introduction site
/// rather than some other admissible StoreVar a hostile producer chose to
/// point at instead; the current rev21 wire format carries no additional
/// provenance for that stronger claim, and this check does not pretend
/// otherwise.
#[cfg(feature = "std")]
fn validate_storevar_site_anchors(
    name: &str,
    borrowed_paths: &[sm_format::semcode_decode::DecodedAccessPath],
    store_var_starts: &[usize],
) -> Result<(), RejectReport> {
    let mut claimed = Vec::new();
    for path in borrowed_paths {
        let Some(sm_format::semcode_decode::DecodedBorrowActivation::StoreVarSite(anchor)) =
            path.activation
        else {
            continue;
        };
        let anchor = anchor as usize;
        if store_var_starts.binary_search(&anchor).is_err() {
            return Err(reject_one(
                name,
                VerificationCode::InvalidOwnershipAnchor,
                anchor,
                "borrow activation anchor does not land on a StoreVar instruction boundary in this function",
            ));
        }
        if claimed.contains(&anchor) {
            return Err(reject_one(
                name,
                VerificationCode::InvalidOwnershipAnchor,
                anchor,
                "two borrow events in the same function declare the identical activation anchor",
            ));
        }
        claimed.push(anchor);
    }
    Ok(())
}

/// #1891 Checkpoint W2E: same-function structural admission for every
/// rev21+ Write execution site, mirroring `validate_storevar_site_anchors`'
/// D2b design exactly but for the independent Write domain (never sharing
/// its `claimed` list or any other state with the Borrow check above - a
/// valid artifact may declare `Borrow StoreVarSite(A)` and
/// `Write StoreVarSite(A)` for the identical numeric PC, since the two are
/// independent semantic domains with no cross-domain anchor-uniqueness
/// constraint; see item 8). `store_var_starts` and `make_record_starts` are
/// each the subset of this function's own `instr_starts` whose decoded
/// opcode matches, built in the same decode pass as `verify_function_code`'s
/// own `instr_starts` - so a single `binary_search` proves both "genuine
/// canonical instruction boundary" and "opcode matches the compiler-declared
/// execution-mode class" at once, never a raw `code[anchor] == opcode` byte
/// compare an operand byte could coincidentally satisfy. This is also why a
/// cross-kind swap (`StoreVarSite` pointing at a real `MakeRecord`, or vice
/// versa) is rejected the same way an out-of-range anchor is: the anchor
/// simply never appears in the list being searched.
///
/// Cardinality (item 6, audited against the current W2A/W2B/W2C/W2D chain,
/// not assumed): `validate_write_sites` (sm-ir, Checkpoint W2A) already
/// proves, at the IR level and before this artifact could ever have been
/// emitted, that a StoreVar `WriteSiteId` (producers A/`assign_tuple_items`
/// and B/`Stmt::Assign`) pairs with exactly one Write event, while a
/// MakeRecord `WriteSiteId` (producer C/`Expr::RecordUpdate`) legitimately
/// pairs with 1..N. Two Write events in the same function declaring the
/// identical `StoreVarSite` anchor are therefore rejected unconditionally -
/// no frozen producer can legitimately emit that shape, mirroring Borrow's
/// own duplicate-anchor rule exactly. Multiple Write events declaring the
/// identical `MakeRecordSite` anchor are never rejected as duplicates -
/// that is producer C's normal, required multi-field-RecordUpdate shape
/// (importing Borrow's "duplicate anchor forbidden" rule here would reject
/// ordinary, correct output).
///
/// Legacy (pre-rev21) Write events decode with `write_execution: None` and
/// are never validated here at all - the header revision, not this
/// function, is the sole grammar/semantic boundary (item 4); this function
/// is never even reached for them in a meaningful way since `None` is
/// skipped outright.
///
/// Failure-model boundary, mirroring `VerificationCode::InvalidOwnershipAnchor`'s
/// existing Borrow-side note exactly: this proves the anchor is a genuine,
/// in-range instruction boundary of the compiler-declared opcode class,
/// belonging to this function - an artifact-level structural fact. It does
/// NOT prove the AccessPath corresponds to an actual source-level
/// overridden field, that a MakeRecord's N declared events match its true
/// source-level override count, or any other source-level ownership
/// provenance between the path and that instruction; the current rev21 wire
/// format carries no additional information for that stronger claim, and
/// this check does not pretend otherwise (item 1, item 7, item 14).
#[cfg(feature = "std")]
fn validate_write_execution_site_anchors(
    name: &str,
    write_paths: &[sm_format::semcode_decode::DecodedAccessPath],
    store_var_starts: &[usize],
    make_record_starts: &[usize],
) -> Result<(), RejectReport> {
    let mut claimed_store_var = Vec::new();
    for path in write_paths {
        match path.write_execution {
            None => continue,
            Some(sm_format::semcode_decode::DecodedWriteExecution::StoreVarSite(anchor)) => {
                let anchor = anchor as usize;
                if store_var_starts.binary_search(&anchor).is_err() {
                    return Err(reject_one(
                        name,
                        VerificationCode::InvalidOwnershipAnchor,
                        anchor,
                        "write execution anchor declares StoreVarSite but does not land on a StoreVar instruction boundary in this function",
                    ));
                }
                if claimed_store_var.contains(&anchor) {
                    return Err(reject_one(
                        name,
                        VerificationCode::InvalidOwnershipAnchor,
                        anchor,
                        "two write events in the same function declare the identical StoreVarSite anchor",
                    ));
                }
                claimed_store_var.push(anchor);
            }
            Some(sm_format::semcode_decode::DecodedWriteExecution::MakeRecordSite(anchor)) => {
                let anchor = anchor as usize;
                if make_record_starts.binary_search(&anchor).is_err() {
                    return Err(reject_one(
                        name,
                        VerificationCode::InvalidOwnershipAnchor,
                        anchor,
                        "write execution anchor declares MakeRecordSite but does not land on a MakeRecord instruction boundary in this function",
                    ));
                }
                // Producer C's 1..N cardinality (W1.5/W2A): multiple Write
                // events legitimately share one MakeRecord's anchor - never
                // rejected as a duplicate here.
            }
        }
    }
    Ok(())
}

/// #1718 (I4.1): the verifier's independent ownership path-family capability
/// contract, extracted into its own function so it is directly testable at
/// the Rust level - given a header and the already-decoded `borrowed_paths`/
/// `write_paths`, with no byte-level decode involved at all.
///
/// This extraction exists because of a real gap mutation M4b found during
/// the #1718 implementation checkpoint: `sm-format`'s decode-time gate
/// already rejects a `Write(AdtPayload)` artifact (or a `Sequence`/
/// `AdtPayload`-Borrow artifact under too old a header) before
/// `verify_function_code`'s body is ever reached through the public,
/// byte-oriented API (`verify_semcode`/`verify_semcode_token` both decode
/// first) - so removing *this* function's own rejection, while leaving
/// `sm-format`'s decode-time gate intact, changed nothing any end-to-end
/// test could observe. The frozen contract requires the verifier to
/// independently enforce this policy, not merely inherit decode's
/// enforcement by accident of call order; this function is the seam that
/// makes that independent enforcement exercisable on its own terms (see
/// `verifier_rejects_adt_payload_write_independently_of_decoder` below,
/// and its own mutation-M4b proof).
///
/// Owns exactly: `Field`/`Sequence`/ADT-Borrow capability accounting, and
/// the unconditional ADT-Write rejection. Does **not** own
/// `CAP_OWNERSHIP_PATHS`/`has_ownership_section` accounting, which predates
/// #1718 and is unaffected by it - that stays in `verify_function_code`
/// alongside debug-symbol and opcode-driven capability accounting, all of
/// which feed one shared, combined `missing_caps` diagnostic there.
///
/// Returns the ownership-family capability bits this function's content
/// actually requires (to be folded into the caller's own `used_caps`) on
/// success, or a `RejectReport` if `Write(AdtPayload)` is present (always,
/// regardless of header) or if the header lacks a capability this content
/// requires.
#[cfg(feature = "std")]
fn verify_ownership_path_family_contract(
    name: &str,
    header: &SemcodeHeaderSpec,
    borrowed_paths: &[sm_format::semcode_decode::DecodedAccessPath],
    write_paths: &[sm_format::semcode_decode::DecodedAccessPath],
) -> Result<u32, RejectReport> {
    let mut has_record_field_ownership = false;
    // #1718: `has_sequence_ownership` and `has_adt_borrow_ownership` feed the
    // same capability-consistency mechanism `has_record_field_ownership`
    // already uses below - this is the verifier's OWN independent authority
    // over the frozen path-family contract, not a check that trusts
    // `sm-format`'s decode-time gate to have already done this work.
    // `has_adt_write_ownership` is tracked separately from `borrowed_paths`/
    // `write_paths` (not chained together) specifically so a `Write` event's
    // `AdtPayload` component can never be silently folded into
    // `has_adt_borrow_ownership` - the two are independently-decided
    // admission domains (see
    // `docs/roadmap/stable_foundation/ssf08_1718_path_family_contract_decision.md`)
    // and must never share one accounting flag.
    let mut has_sequence_ownership = false;
    let mut has_adt_borrow_ownership = false;
    let mut has_adt_write_ownership = false;
    for p in borrowed_paths {
        for c in &p.components {
            match c {
                sm_format::semcode_decode::DecodedAccessPathComponent::FieldSymbol(_) => {
                    has_record_field_ownership = true;
                }
                sm_format::semcode_decode::DecodedAccessPathComponent::AdtPayload { .. } => {
                    has_adt_borrow_ownership = true;
                    // Variant is a global SymbolId; it cannot be bounds-checked
                    // against the local string table. Structural acceptance only.
                }
                sm_format::semcode_decode::DecodedAccessPathComponent::SequenceIndexStatic(_) => {
                    has_sequence_ownership = true;
                }
                _ => {}
            }
        }
    }
    for p in write_paths {
        for c in &p.components {
            match c {
                sm_format::semcode_decode::DecodedAccessPathComponent::FieldSymbol(_) => {
                    has_record_field_ownership = true;
                }
                sm_format::semcode_decode::DecodedAccessPathComponent::AdtPayload { .. } => {
                    has_adt_write_ownership = true;
                }
                sm_format::semcode_decode::DecodedAccessPathComponent::SequenceIndexStatic(_) => {
                    has_sequence_ownership = true;
                }
                _ => {}
            }
        }
    }

    // #1718: unconditional rejection, independent of capability. No header
    // revision or capability bit - present, future, or hypothetically
    // reinterpreted - authorizes `Write(AdtPayload)`; this is not a missing-
    // capability situation (`CapabilityViolation`) that a wider header could
    // ever cure, so it is checked and rejected before, and separately from,
    // the capability-consistency check below.
    if has_adt_write_ownership {
        return Err(reject_one(
            name,
            VerificationCode::InvalidOwnershipSection,
            0,
            "Write(AdtPayload) is not an admitted SemCode ownership path under any header \
             revision (#1718)",
        ));
    }

    let mut used_caps = 0u32;
    if has_record_field_ownership {
        used_caps |= CAP_OWNERSHIP_FIELD_PATHS;
    }
    if has_sequence_ownership {
        used_caps |= CAP_OWNERSHIP_SEQUENCE_PATHS;
    }
    if has_adt_borrow_ownership {
        used_caps |= CAP_OWNERSHIP_ADT_BORROW_PATHS;
    }

    let missing_caps = used_caps & !header.capabilities;
    if missing_caps != 0 {
        return Err(reject_one(
            name,
            VerificationCode::CapabilityViolation,
            0,
            format!(
                "function requires capability bits 0x{missing_caps:08x}, but header '{}' \
                 provides only 0x{:08x}",
                String::from_utf8_lossy(&header.magic),
                header.capabilities
            ),
        ));
    }

    Ok(used_caps)
}

#[cfg(feature = "std")]
fn verify_function_code(
    env: &sm_format::semcode_decode::DecodedFunctionEnvelope,
    header: &SemcodeHeaderSpec,
    quotas: &RuntimeQuotas,
    limits: &VerificationLimits,
    work_meter: &mut AnalysisWorkMeter,
) -> Result<PendingVerifiedFunction, RejectReport> {
    let name = env.name.as_str();

    // #1731 (FA-05-001): the DBG0 sentinel collides with TupleGet's opcode
    // byte (0x44 = 'D'). If the decoder recognized a DBG0 section, check
    // whether the same bytes, read from string_table_end_offset with no
    // metadata-section recognition at all, would ALSO form a complete,
    // well-formed instruction stream all the way to the end of this
    // function's code. If both readings are valid, the artifact has more
    // than one canonical structural interpretation and admission must fail
    // closed rather than silently keep the DBG0 reading.
    if env.has_debug_section
        && instruction_stream_parses_fully(name, env.code_slice, env.string_table_end_offset)
    {
        return Err(reject_one(
            name,
            VerificationCode::AmbiguousInstructionFraming,
            env.string_table_end_offset,
            "byte range is both a valid DBG0 debug section and a valid instruction stream",
        ));
    }

    // #1754 (FA-07-014): a per-function string-table check against
    // `max_symbol_table` used to live here, but sm-format already caps each
    // function's local string table at `MAX_STRINGS_PER_FUNCTION` (256, see
    // semcode_decode.rs), which is far below `max_symbol_table` (16_384) -
    // making that check dead code. The real program-wide resource is the VM's
    // `RuntimeSymbolTable` (crates/sm-runtime-core/src/lib.rs), which is
    // built ONCE per program and shared across every function (see
    // `build_vm_program_view_from_decoded` in crates/sm-vm/src/semcode_vm.rs,
    // which interns each function's `env.strings` into a single
    // program-wide, dedup-by-value table). That budget is enforced
    // program-wide in `verify_semcode_token` instead, see the check there.
    let string_count = env.strings.len();

    let debug_symbol_count = env.debug_symbols.len();
    if debug_symbol_count > quotas.max_debug_symbols_per_function {
        return Err(reject_one(
            name,
            VerificationCode::ResourceLimitExceeded,
            0,
            format!(
                "debug section uses {} entries, exceeding the per-function debug-symbol limit of {}",
                debug_symbol_count, quotas.max_debug_symbols_per_function
            ),
        ));
    }

    let has_ownership_section = env.has_ownership_section;
    // #1718 (I4.1): extracted into its own function so it is directly,
    // independently testable at the Rust level - see
    // `verify_ownership_path_family_contract`'s own doc comment for why
    // this extraction exists (mutation M4b's finding).
    let ownership_family_caps =
        verify_ownership_path_family_contract(name, header, &env.borrowed_paths, &env.write_paths)?;

    let code = env.code_slice;
    let mut cursor = env.instr_start_offset;
    let instr_start = cursor;
    let instr_len = code.len().saturating_sub(instr_start);
    let mut instr_starts = Vec::new();
    // #1726 Checkpoint D2b: the subset of `instr_starts` whose decoded
    // opcode is `StoreVar`, built in the same pass and therefore inheriting
    // the same "genuine canonical instruction boundary" proof `instr_starts`
    // itself carries - never a raw `code[offset] == StoreVar` byte compare,
    // which an operand byte could coincidentally satisfy.
    let mut store_var_starts = Vec::new();
    // #1891 Checkpoint W2E: the subset of `instr_starts` whose decoded
    // opcode is `MakeRecord`, built in the same pass for the identical
    // reason `store_var_starts` is - see `validate_write_execution_site_anchors`.
    let mut make_record_starts = Vec::new();
    let mut instruction_successors = Vec::new();
    let mut jump_targets = Vec::new();
    let mut string_refs = Vec::new();
    let mut call_argcs = Vec::new();
    let mut max_register: Option<usize> = None;
    let mut used_caps = 0u32;
    while cursor < code.len() {
        let offset = cursor - instr_start;
        instr_starts.push(offset);
        let opcode = read_u8(code, &mut cursor).map_err(|_| {
            reject_one(
                name,
                VerificationCode::UnknownOpcode,
                offset,
                "missing opcode byte",
            )
        })?;
        let opcode = Opcode::from_byte(opcode).map_err(|err| match err {
            SemcodeFormatError::UnknownOpcode(_) => reject_one(
                name,
                VerificationCode::UnknownOpcode,
                offset,
                err.to_string(),
            ),
            _ => reject_one(
                name,
                VerificationCode::OperandOutOfBounds,
                offset,
                err.to_string(),
            ),
        })?;
        if opcode == Opcode::StoreVar {
            store_var_starts.push(offset);
        }
        if opcode == Opcode::MakeRecord {
            make_record_starts.push(offset);
        }
        // #1756 Codex review round 21: `MetadataCollector` appends
        // directly into these outer `jump_targets`/`string_refs`/
        // `call_argcs` accumulators - no per-instruction `Vec` is built
        // and then `.extend()`-ed in and dropped, mirroring `FlatSink`'s
        // own "write straight into the final destination" design.
        // `jump_targets_before` lets this loop recover "this specific
        // instruction's own jump target" (at most one is ever pushed per
        // instruction - only `Jmp`/`JmpIf` ever call `jump_target`, each
        // exactly once) without `OperandRefs` needing to carry it.
        let jump_targets_before = jump_targets.len();
        let mut metadata_collector = MetadataCollector {
            jump_targets: &mut jump_targets,
            string_refs: &mut string_refs,
            call_argcs: &mut call_argcs,
        };
        let refs = decode_operands(
            name,
            code,
            &mut cursor,
            offset,
            opcode,
            true,
            &mut NoCollect,
            &mut metadata_collector,
        )?;
        let next_offset = cursor - instr_start;
        let jump_target = jump_targets.get(jump_targets_before).copied();
        let successors = match (opcode, jump_target) {
            (Opcode::Ret, _) => InstructionSuccessors::None,
            (Opcode::Jmp, Some(target)) => InstructionSuccessors::One(target),
            (Opcode::JmpIf, Some(target)) => InstructionSuccessors::Two(target, next_offset),
            (Opcode::Jmp | Opcode::JmpIf, None) => {
                return Err(reject_one(
                    name,
                    VerificationCode::InvalidJumpTarget,
                    offset,
                    "jump instruction has no decoded target",
                ));
            }
            _ => InstructionSuccessors::One(next_offset),
        };
        instruction_successors.push(successors);
        let min_rev = opcode.minimum_semcode_revision();
        if header.rev < min_rev {
            return Err(reject_one(
                name,
                VerificationCode::OpcodeRequiresNewerHeader,
                offset,
                format!(
                    "opcode {opcode:?} requires SemCode header revision >= {min_rev}, but artifact header '{}' is revision {}",
                    String::from_utf8_lossy(&header.magic),
                    header.rev
                ),
            ));
        }
        used_caps |= refs.required_capabilities;
        max_register = match (max_register, refs.max_register) {
            (Some(lhs), Some(rhs)) => Some(lhs.max(rhs)),
            (None, some) => some,
            (some, None) => some,
        };
    }

    for sym in &env.debug_symbols {
        if sym.pc >= instr_len {
            return Err(reject_one(
                name,
                VerificationCode::InvalidDebugSection,
                sym.pc,
                "debug symbol pc points past the instruction stream",
            ));
        }
        // #1746 (FA-07-006): range alone does not prove `sym.pc` denotes an
        // actual instruction boundary - it could land inside a decoded
        // operand's bytes. Reuse the instruction-start set this same
        // verification walk already built, exactly like the jump-target
        // check below does for `InvalidJumpTarget`. `instr_starts` is
        // pushed in strictly increasing order (each iteration's `offset` is
        // read from `cursor` before `cursor` only ever advances further),
        // so `binary_search` is valid and avoids an O(symbols x
        // instructions) linear scan - the format permits up to 8,192 debug
        // symbols per function, so a debug-heavy artifact could otherwise
        // force a disproportionate amount of verifier CPU per function
        // (review finding on this PR).
        if instr_starts.binary_search(&sym.pc).is_err() {
            return Err(reject_one(
                name,
                VerificationCode::InvalidDebugSection,
                sym.pc,
                "debug symbol pc does not land on an instruction boundary",
            ));
        }
    }

    // #1726 Checkpoint D2b: same-function structural admission of every
    // rev21+ `StoreVarSite(anchor)` Borrow activation - see
    // `validate_storevar_site_anchors` for the full design note.
    validate_storevar_site_anchors(name, &env.borrowed_paths, &store_var_starts)?;
    // #1891 Checkpoint W2E: same-function structural admission of every
    // rev21+ Write execution site - see
    // `validate_write_execution_site_anchors` for the full design note.
    validate_write_execution_site_anchors(
        name,
        &env.write_paths,
        &store_var_starts,
        &make_record_starts,
    )?;

    if debug_symbol_count > 0 {
        used_caps |= CAP_DEBUG_SYMBOLS;
    }
    if has_ownership_section {
        used_caps |= CAP_OWNERSHIP_PATHS;
    }
    // #1718 (I4.1): Field/Sequence/ADT-Borrow capability accounting now
    // lives in `verify_ownership_path_family_contract`, which already
    // independently checked these bits against `header.capabilities` on its
    // own terms (see that function). Folding its result into this function's
    // own `used_caps`/`missing_caps` pass keeps a single combined diagnostic
    // for callers that also care about opcode-driven/debug-symbol
    // capabilities, without re-deriving or duplicating the ownership-family
    // policy itself.
    used_caps |= ownership_family_caps;

    let missing_caps = used_caps & !header.capabilities;
    if missing_caps != 0 {
        return Err(reject_one(
            name,
            VerificationCode::CapabilityViolation,
            0,
            format!(
                "function requires capability bits 0x{missing_caps:08x}, but header '{}' provides only 0x{:08x}",
                String::from_utf8_lossy(&header.magic),
                header.capabilities
            ),
        ));
    }

    for target in jump_targets {
        if target >= instr_len {
            return Err(reject_one(
                name,
                VerificationCode::InvalidJumpTarget,
                target,
                "jump target points past the instruction stream",
            ));
        }
        if !instr_starts.contains(&target) {
            return Err(reject_one(
                name,
                VerificationCode::InvalidJumpTarget,
                target,
                "jump target does not land on an instruction boundary",
            ));
        }
    }

    let reachable_offsets =
        verify_reachable_control_flow(name, instr_len, &instr_starts, &instruction_successors)?;

    // #1773 (FA-09-005): argc, keyed by the Call instruction's own offset -
    // only "call target" entries (real `Opcode::Call` sites) ever have an
    // entry here; "closure function name" entries come from `MakeClosure`
    // (a static reference check, not a call site with a known argc) and are
    // deliberately left with `argc: None` below.
    let call_argcs: std::collections::HashMap<usize, usize> = call_argcs.into_iter().collect();

    let mut call_targets = Vec::new();
    for (offset, sid, usage) in string_refs {
        if sid >= string_count {
            return Err(reject_one(
                name,
                VerificationCode::InvalidStringReference,
                offset,
                format!("{usage} uses missing string id s{sid}"),
            ));
        }
        match usage {
            "call target" => call_targets.push((
                offset,
                env.strings[sid].clone(),
                true,
                call_argcs.get(&offset).copied(),
            )),
            "closure function name" => {
                call_targets.push((offset, env.strings[sid].clone(), false, None));
            }
            _ => {}
        }
    }

    if let Some(max_register) = max_register {
        let used = max_register + 1;
        if used > quotas.max_registers {
            return Err(reject_one(
                name,
                VerificationCode::InvalidRegisterReference,
                used - 1,
                format!(
                    "function references register r{}, exceeding the register budget of {}",
                    max_register, quotas.max_registers
                ),
            ));
        }
    }

    // #1756 (FA-07-016): forward MUST definite-assignment proof - every
    // reachable register read must be definitely defined on every incoming
    // execution path, not merely in range. Runs only when this function has
    // a canonical `SIG0` signature (`env.signature.is_some()`): a pre-#1773
    // artifact's header predates canonical per-function arity metadata, so
    // `IN[entry]` cannot be soundly determined for it - caller-inferred or
    // convention-inferred entry definitions are exactly the unsound
    // heuristic #1756's own research explicitly rejected (see the issue's
    // Phase B checkpoint). This preserves pre-#1756 admission behavior for
    // signature-less artifacts exactly, mirroring how #1773's own
    // `validate_call_arguments`/arity check already treats `signature: None`
    // as "nothing new to enforce" - not a new decision, the same one
    // continued.
    if let Some(signature) = env.signature.as_ref() {
        let entry_param_count = signature.families.len();
        // #1756: the domain must be bounded by the already-established
        // register quota before any allocation sized by it - an
        // attacker-controlled SIG0 parameter count (bounded only by the
        // decoder's own MAX_SIGNATURE_PARAMETERS_PER_FUNCTION, independent
        // of this verification call's actual `quotas.max_registers`) must
        // not be allowed to size a per-node bitset on its own.
        if entry_param_count > quotas.max_registers {
            return Err(reject_one(
                name,
                VerificationCode::InvalidRegisterReference,
                0,
                format!(
                    "canonical signature declares {} parameter register(s), exceeding the register budget of {}",
                    entry_param_count, quotas.max_registers
                ),
            ));
        }
        prove_definite_register_assignment(
            name,
            code,
            instr_start,
            &instr_starts,
            &instruction_successors,
            &reachable_offsets,
            entry_param_count,
            limits,
            work_meter,
        )?;
    }

    Ok(PendingVerifiedFunction {
        verified: VerifiedFunction {
            name: name.to_string(),
            code_len: code.len(),
            string_count,
            debug_symbol_count,
        },
        call_targets,
        has_ownership_section,
    })
}

#[cfg(feature = "std")]
enum InstructionSuccessors {
    None,
    One(usize),
    Two(usize, usize),
}

/// Returns the set of reachable instruction offsets on success. #1756
/// (FA-07-016) reuses this exact return value as the authoritative
/// reachable-node set for its definite-assignment pass, rather than
/// re-deriving reachability with a second traversal.
#[cfg(feature = "std")]
fn verify_reachable_control_flow(
    function: &str,
    instr_len: usize,
    instr_starts: &[usize],
    instruction_successors: &[InstructionSuccessors],
) -> Result<HashSet<usize>, RejectReport> {
    let mut pending = vec![0usize];
    let mut reachable = HashSet::new();

    while let Some(offset) = pending.pop() {
        if !reachable.insert(offset) {
            continue;
        }
        if offset == instr_len {
            return Err(reject_one(
                function,
                VerificationCode::ReachableFunctionFallthrough,
                offset,
                "reachable control flow falls off the end of the instruction stream",
            ));
        }
        let index = instr_starts.binary_search(&offset).map_err(|_| {
            reject_one(
                function,
                VerificationCode::ReachableFunctionFallthrough,
                offset,
                "reachable control-flow successor is not an instruction boundary",
            )
        })?;
        match &instruction_successors[index] {
            InstructionSuccessors::None => {}
            InstructionSuccessors::One(successor) => pending.push(*successor),
            InstructionSuccessors::Two(first, second) => {
                pending.push(*first);
                pending.push(*second);
            }
        }
    }

    Ok(reachable)
}

/// #1756 (FA-07-016): a compact fixed-size bitset over register numbers
/// `0..domain_size`. `domain_size` is always bounded by the register-range
/// quota already enforced before this pass runs (see the caller), so this
/// never allocates from an attacker-controlled unbounded register id.
///
/// #1756 Codex review round 8: this now represents the MAY-MISSING dataflow
/// lattice element (a register set under UNION meet), not the original
/// MUST-DEFINED one (intersection meet) - see `compute_missing_sets`'s doc
/// comment for the full duality derivation. `RegSet` itself is an
/// unopinionated bitset either way; only the meet operation used on it (now
/// plain per-bit `insert`, driving the worklist in `compute_missing_sets`,
/// rather than a whole-set `intersect_with`) and its initialization
/// (BOTTOM=empty for non-entry nodes, the dual of the old TOP=full rule)
/// changed.
///
/// #1756 Codex review round 10: `empty()` no longer eagerly allocates
/// `word_count(domain_size)` words up front - it starts with zero words,
/// and `insert` grows the backing `Vec<u64>` only as far as the highest
/// bit index this PARTICULAR set has ever actually held (never past
/// `word_count(domain_size)`, since every inserted `reg` is already
/// guaranteed `< domain_size` by the invariant above). A set that never
/// receives a single bit costs nothing beyond its own header; a set whose
/// content genuinely spans the full domain still costs the same as before.
/// This is what lets `compute_missing_sets` allocate one `RegSet` per
/// reachable position (needed regardless - see its own doc comment)
/// without that array's per-entry cost being `domain_size`-wide
/// unconditionally, for positions whose actual missing set is small or
/// empty.
#[cfg(feature = "std")]
#[derive(Clone)]
struct RegSet {
    words: Vec<u64>,
}

#[cfg(feature = "std")]
impl RegSet {
    fn word_count(domain_size: usize) -> usize {
        domain_size.saturating_add(63) / 64
    }

    /// Every register in the domain is set. Used only to compute `entry`'s
    /// fixed MISSING set via complement (`full() `then removing
    /// `ENTRY_DEFS` bits) - no node is ever initialized to this directly
    /// under the round-8 MAY-missing formulation (see `compute_missing_
    /// sets`'s doc comment for why BOTTOM/empty, not this, is every
    /// non-entry node's correct starting point).
    fn full(domain_size: usize) -> Self {
        let mut words = vec![u64::MAX; Self::word_count(domain_size)];
        let rem = domain_size % 64;
        if rem != 0 {
            if let Some(last) = words.last_mut() {
                *last &= (1u64 << rem) - 1;
            }
        }
        RegSet { words }
    }

    /// #1756 Codex review round 10: starts genuinely empty (zero words,
    /// zero heap allocation) rather than eagerly pre-sized to
    /// `word_count(domain_size)` - `insert` grows the backing storage on
    /// demand instead, so callers no longer need `domain_size` here.
    fn empty() -> Self {
        RegSet { words: Vec::new() }
    }

    fn contains(&self, reg: usize) -> bool {
        let word = reg / 64;
        let bit = reg % 64;
        self.words.get(word).is_some_and(|w| (w >> bit) & 1 != 0)
    }

    /// Grows the backing storage to cover `reg` if this set has never held
    /// a bit that high before (#1756 Codex review round 10) - safe because
    /// every `reg` this pass ever inserts is already guaranteed
    /// `< domain_size` by the struct-level invariant, so this can never
    /// grow past `word_count(domain_size)` regardless of call pattern.
    fn insert(&mut self, reg: usize) {
        let word = reg / 64;
        let bit = reg % 64;
        if self.words.len() <= word {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1u64 << bit;
    }

    fn remove(&mut self, reg: usize) {
        let word = reg / 64;
        let bit = reg % 64;
        if let Some(w) = self.words.get_mut(word) {
            *w &= !(1u64 << bit);
        }
    }
}

/// #1756 (FA-07-016): forward MUST dataflow proving every reachable register
/// read is definitely defined on every incoming execution path (meet =
/// intersection). Reuses the verifier's own successor relation
/// (`instruction_successors`) to build predecessors, and its own computed
/// reachable-offset set (`reachable_offsets`, from
/// `verify_reachable_control_flow`) to decide which nodes this proof
/// actually applies to - no second CFG/branch-target engine, no
/// re-derivation of reachability.
///
/// `entry_param_count` registers `0..entry_param_count` are `IN[entry]`
/// (`ENTRY_DEFS`, from the caller's canonical `SIG0` signature) - the only
/// registers defined at function entry, regardless of numeric range or
/// register-storage capacity. Every other node initializes to TOP
/// (`RegSet::full`), per the non-negotiable loop-correctness rule.
///
/// Unreachable nodes are never allocated per-node dataflow state at all (see
/// the Codex-review note below): an unreachable predecessor's contribution
/// to any meet would always be TOP (the identity element for intersection),
/// so an edge from an unreachable node is simply never recorded rather than
/// being represented and then ignored. This preserves the verifier's
/// existing, separate policy on structurally valid-but-unreachable code
/// untouched - this pass only judges reachable reads.
///
/// Codex review round 1 on this PR (#1840) found two independent
/// amplification sources in an earlier revision, fixed together here:
///
/// 1. Allocating one `RegSet` per node for every structurally decoded
///    instruction, reachable or not, let a structurally valid artifact (an
///    entry `RET` followed by megabytes of never-executed trailing
///    instructions, still legal per the verifier's reachable-control-flow
///    policy) force memory proportional to total instruction count, with no
///    existing quota bounding that count. Fixed: dataflow state below is
///    allocated and indexed only for reachable nodes (by position within
///    `reachable_indices`, not by raw instruction index) - the same
///    reachable set the existing verifier already walks for its own
///    reachability BFS, not a new, larger one.
/// 2. Sizing the register domain from the raw numeric span
///    (`0..=max_register_id`) rather than the registers actually in use
///    meant a single reference to a high register number (still legal and
///    in-budget) forced a full-width bitset even when only one or two
///    registers mattered - a program referencing only `{r0, r4095}` needs a
///    domain of size 2, not 4096. Fixed: the register universe `U` below is
///    built densely, from the registers `ENTRY_DEFS` and reachable
///    instructions actually reference, each remapped to a compact index;
///    diagnostics still report the original raw register identity.
///
/// Both fixes only change how densely this pass's own bounded state is
/// packed - `U`'s size can never exceed the pre-existing register-budget
/// quota (every register `U` collects was already proven in-range by that
/// check before this function runs), so neither fix relaxes or replaces
/// that quota; it remains the only source of truth for the upper bound.
///
/// Computes the two quantities that bound `prove_definite_register_
/// assignment`'s own memory use: the reachable node index set (ascending,
/// position 0 = function entry), the reachable nodes' own reads/writes (also
/// position-indexed), and the dense register universe `U` (ascending,
/// deduplicated). Factored out from that function so tests can assert
/// directly on the returned lengths - the exact per-node and per-register-
/// domain multipliers - rather than relying only on timing to demonstrate
/// the Codex-review fixes above.
///
/// `reads_writes_of` is called exactly once per REACHABLE raw instruction
/// index, never for an unreachable one (Codex review round 3 on this PR,
/// #1840: the prior revision built dense `instr_reads`/`instr_writes: Vec<
/// Vec<u16>>` for every structurally decoded instruction in the caller,
/// before reachability was even known - a `Vec<u16>` costs 24 bytes even
/// empty, so millions of unreachable instructions cost hundreds of MB for
/// these two arrays alone. The real caller's closure re-decodes just that
/// one reachable instruction's operands on demand via `decode_operands`;
/// tests can supply a trivial closure over a small, hand-built map instead).
///
/// Codex review round 4 on this PR: even after round 3 removed the
/// unreachable-instruction cost, storing each reachable node's reads/writes
/// as its OWN `Vec<u16>` (one 24-byte header plus a separate heap
/// allocation, per node, per array) still meant a large fully REACHABLE
/// straight-line stream (e.g. millions of `LOAD_BOOL r0` in a row, no
/// branching, no unreachable padding at all) cost dozens of bytes of
/// heap-container overhead per instruction. Fixed: reads and writes are now
/// returned CSR-style - one flat `Vec<u16>` holding every reachable
/// instruction's registers back to back, plus an offset table
/// (`reads_offsets`/`writes_offsets`, `u32` per node, `reachable_count + 1`
/// entries) marking each node's slice boundaries - a single allocation per
/// array instead of one per node. Use `csr_slice` to read a given
/// position's slice back out.
///
/// Codex review round 18: `reads_writes_of` no longer RETURNS a `(Vec<u16>,
/// Vec<u16>)` pair per reachable instruction - it now takes an `&mut dyn
/// OperandSink` and emits directly into it. `FlatSink` (below) is that
/// sink here, appending straight into this function's own `reads_flat`/
/// `writes_flat` - the temporary per-instruction `Vec<u16>` pair round 4
/// already eliminated from the FLAT ARRAYS' own representation existed one
/// layer further out, in the closure boundary itself (build a pair, hand
/// it back, `extend_from_slice` it in, drop it) - this removes that too,
/// so a reachable instruction's registers go straight from `decode_
/// operands`'s match arms into `reads_flat`/`writes_flat` with no
/// intermediate collection at all.
///
/// Returns `(reachable_indices, reads_flat, reads_offsets, writes_flat,
/// writes_offsets, universe)`.
#[cfg(feature = "std")]
type DataflowDomainAccounting = (Vec<usize>, Vec<u16>, Vec<u32>, Vec<u16>, Vec<u32>, Vec<u16>);

/// #1756 Codex review round 18: `dataflow_domain_accounting`'s own sink -
/// appends each emitted register straight into the CALLER's `reads_flat`/
/// `writes_flat` CSR buffers, so a reachable instruction's operands never
/// pass through any intermediate `Vec<u16>` on their way from `decode_
/// operands` into the flat arrays this pass ultimately needs.
#[cfg(feature = "std")]
struct FlatSink<'a> {
    reads: &'a mut Vec<u16>,
    writes: &'a mut Vec<u16>,
}

#[cfg(feature = "std")]
impl OperandSink for FlatSink<'_> {
    fn read(&mut self, reg: u16) {
        self.reads.push(reg);
    }
    fn write(&mut self, reg: u16) {
        self.writes.push(reg);
    }
}

#[cfg(feature = "std")]
fn dataflow_domain_accounting(
    instr_starts: &[usize],
    reachable_offsets: &HashSet<usize>,
    entry_param_count: usize,
    mut reads_writes_of: impl FnMut(usize, &mut dyn OperandSink),
) -> DataflowDomainAccounting {
    let mut reachable_indices: Vec<usize> = reachable_offsets
        .iter()
        .filter_map(|offset| instr_starts.binary_search(offset).ok())
        .collect();
    reachable_indices.sort_unstable();

    let mut reads_flat: Vec<u16> = Vec::new();
    let mut reads_offsets: Vec<u32> = Vec::with_capacity(reachable_indices.len() + 1);
    let mut writes_flat: Vec<u16> = Vec::new();
    let mut writes_offsets: Vec<u32> = Vec::with_capacity(reachable_indices.len() + 1);
    let mut universe: Vec<u16> = (0..entry_param_count).map(|r| r as u16).collect();
    for &idx in &reachable_indices {
        let reads_start = reads_flat.len();
        let writes_start = writes_flat.len();
        reads_offsets.push(reads_start as u32);
        writes_offsets.push(writes_start as u32);
        {
            let mut sink = FlatSink {
                reads: &mut reads_flat,
                writes: &mut writes_flat,
            };
            reads_writes_of(idx, &mut sink);
        }
        universe.extend_from_slice(&reads_flat[reads_start..]);
        universe.extend_from_slice(&writes_flat[writes_start..]);
    }
    reads_offsets.push(reads_flat.len() as u32);
    writes_offsets.push(writes_flat.len() as u32);
    universe.sort_unstable();
    universe.dedup();

    (
        reachable_indices,
        reads_flat,
        reads_offsets,
        writes_flat,
        writes_offsets,
        universe,
    )
}

/// Reads position `pos`'s slice back out of a CSR-encoded (flat, offsets)
/// pair - built by `dataflow_domain_accounting` for register lists, and (as
/// of #1756 Codex review round 7) by the SCC-membership table in
/// `prove_definite_register_assignment` for reachable positions - generic
/// over the element type since both callers share the identical
/// flat-Vec-plus-offset-table shape.
#[cfg(feature = "std")]
fn csr_slice<'a, T>(flat: &'a [T], offsets: &[u32], pos: usize) -> &'a [T] {
    &flat[offsets[pos] as usize..offsets[pos + 1] as usize]
}

/// Invokes `visit` once per REACHABLE successor position of raw instruction
/// `idx`, resolved from the verifier's own `instruction_successors` (no
/// second CFG) via `instr_starts`/`reachable_indices` binary search. A
/// target that isn't a real instruction boundary can only belong to an
/// unreachable fallthrough off the end of the stream - every jump target
/// was already fully validated above regardless of reachability, so this is
/// never a real, reachable edge silently dropped.
#[cfg(feature = "std")]
fn for_each_reachable_successor(
    idx: usize,
    instruction_successors: &[InstructionSuccessors],
    instr_starts: &[usize],
    reachable_indices: &[usize],
    mut visit: impl FnMut(usize),
) {
    let mut link = |target: usize| {
        if let Ok(target_idx) = instr_starts.binary_search(&target) {
            if let Ok(target_pos) = reachable_indices.binary_search(&target_idx) {
                visit(target_pos);
            }
        }
    };
    match &instruction_successors[idx] {
        InstructionSuccessors::None => {}
        InstructionSuccessors::One(target) => link(*target),
        InstructionSuccessors::Two(first, second) => {
            link(*first);
            link(*second);
        }
    }
}

/// Delivers `bit` to `target`'s MISSING set, in place - `target`'s `RegSet`
/// is mutated directly (`insert`), never cloned or replaced, which is what
/// eliminates the round-8 "duplicate live RegSet states" amplification (see
/// `prove_definite_register_assignment`'s doc comment): there is exactly
/// one `RegSet` per reachable position, period, so no number of branches
/// independently computing the same result can ever duplicate it. Returns
/// `true` iff this was new information (the bit wasn't already present) -
/// the caller only enqueues `(target, bit)` for further propagation when
/// this is `true`, which is what guarantees each (position, bit) pair is
/// ever enqueued at most once (MISSING only ever grows, so once a bit is
/// present it can never need re-delivery). Position 0 (entry) is never
/// delivered to - its MISSING set is fixed, see the sibling doc comment for
/// the soundness argument.
#[cfg(feature = "std")]
fn deliver(missing: &mut [RegSet], target: usize, bit: usize) -> bool {
    if target != 0 && !missing[target].contains(bit) {
        missing[target].insert(bit);
        true
    } else {
        false
    }
}

/// Core bit-level MAY-MISSING worklist - factored out from `prove_definite_
/// register_assignment` so tests can assert directly on its `event_count`/
/// `peak_queue_len` outputs (Codex review round 8's explicit request for
/// allocation/relaxation counters, not timing alone) using small, hand-
/// constructed `successors_of`/`writes_contains` closures rather than real
/// SemCode bytes.
///
/// Returns `(missing, event_count, peak_queue_len)`: the converged MISSING
/// set for every reachable position (position 0 = entry, whose value is
/// `entry_missing` unchanged), the total number of (position, bit) events
/// the worklist processed after seeding, and the largest the worklist ever
/// grew to at any one moment.
///
/// **Total work bound.** Every `(position, bit)` pair is enqueued at MOST
/// once, ever, across the whole computation - `deliver` only returns `true`
/// (the caller's only enqueue trigger) the one time a bit transitions from
/// absent to present in some position's `MISSING` set, and `MISSING` only
/// ever grows (union, never shrinks) - so the total number of *distinct*
/// (position, bit) pairs that can ever be enqueued is bounded by
/// `reachable_count * domain_size`, REGARDLESS of the reachable subgraph's
/// structure: cyclic, irreducible, arbitrary fan-in through a shared loop
/// header, all of it is irrelevant, because a bit that is already present
/// at a node can never need re-delivery to it. Each dequeued event costs
/// O(1) amortized: a single `writes_contains` check (a linear scan over
/// that position's own reads/writes list, which is small - at most a
/// handful of registers per instruction, by construction, not tied to
/// `domain_size`) plus O(out_degree) <= O(2) successor deliveries. This
/// gives a provable `O(reachable_count * domain_size)` TOTAL bit-level
/// work bound, independent of ordering - #1756 Codex review round 8's
/// second finding (`local_reverse_postorder_ranks`, round 7's fix, does not
/// bound convergence inside one large strongly-connected component against
/// an adversarial backedge-fan-in shape) is closed by this property: there
/// is no "large SCC" concept left to reprocess, because nothing at the bit
/// level is ever reprocessed. `successors_of`/`writes_contains` need no
/// predecessor structure, no reverse-postorder ranking, and no strongly-
/// connected-component decomposition - `tarjan_scc`, `TarjanFrame`,
/// `nth_reachable_successor`, `SccInfo`, `scc_membership_csr`, and
/// `local_reverse_postorder_ranks` (rounds 6-7's SCC/ordering machinery)
/// are removed entirely as of this round: they existed only to bound
/// whole-`RegSet` reprocessing, which this formulation does not do in the
/// first place.
///
/// **Peak live-memory bound** (#1756 Codex review round 9): the total-work
/// bound above says nothing about how large the worklist can grow at any
/// one moment - a wide fan-out (e.g. a balanced binary dispatch tree with
/// many leaves) can, under FIFO order, accumulate an entire breadth level
/// (up to `domain_size` bits at EACH of that level's nodes) before any of
/// it drains, peaking near the full `reachable_count * domain_size` bound.
/// This worklist processes LIFO (a stack, popping the MOST recently pushed
/// item) instead: popping `(pos, bit)` pushes at most `out_degree(pos) <=
/// 2` new items for that SAME bit, so once a bit's delivery begins it runs
/// to completion - reaching nodes with no further successors, or nodes
/// that already have it - before any sibling branch (for that same bit, or
/// any other) is touched, exactly the standard DFS depth bound. Peak
/// worklist size is therefore bounded by the current bit's own delivery
/// depth (`O(reachable_count)`, the same order `missing` itself already
/// costs) PLUS the not-yet-started backlog waiting beneath it on the stack
/// (bounded by how many `(pos, bit)` pairs any single pop can contribute -
/// `O(out_degree) <= O(2)` - times how many pops are pending, not by
/// `domain_size` multiplied across an entire wide level at once). This
/// changes only which end of the worklist is popped from - `deliver`'s
/// "already delivered, skip" guard (hence the total-work bound above) is
/// completely unaffected by pop order, so this is a peak-memory
/// representation choice, not a new correctness-load-bearing ordering
/// heuristic of the kind rounds 5-7 tried and round 8 replaced.
///
/// **Per-node memory bound** (#1756 Codex review round 10): the `missing`
/// array itself needs one `RegSet` slot per reachable position regardless,
/// with no way around that minimum - it is exactly the same
/// `O(reachable_count)` array `event_count`/`peak_queue_len` already cost
/// nothing extra to track. What round 10 fixes is each slot's OWN backing
/// storage: before this round, `RegSet::empty` eagerly allocated
/// `word_count(domain_size)` words for every non-entry position, whether
/// or not that position's missing set ever held a single bit, so a long,
/// fully-reachable chain that finishes defining every register early and
/// spends the rest of its length on cheap re-writes (see
/// `c1756_reachable_full_domain_then_long_redefine_chain_stays_correct`)
/// still forced `reachable_count * domain_size` bits of storage even
/// though almost every position's real content, after convergence, is
/// empty. `RegSet::insert` (see its own doc comment) now grows each set's
/// storage only as far as the highest bit it has ever actually held, so
/// an empty position costs nothing beyond its own array slot, and only
/// positions whose missing set genuinely spans much of the domain cost
/// anywhere close to the old unconditional `domain_size` width.
///
/// #1756 Codex review round 14 (owner decision): the deterministic work
/// budget for a WHOLE artifact verification call - `VerificationLimits.
/// max_work_units` is "maximum deterministic analysis work for one
/// artifact verification call," never per function. A single instance is
/// created ONCE by `verify_semcode_token_with_quotas_and_limits` and a
/// `&mut` reference is threaded through every signature-bearing
/// function's analysis (`verify_function_code` ->
/// `prove_definite_register_assignment` -> `compute_missing_sets`),
/// never recreated or reset per function - `used` therefore always
/// reflects the CUMULATIVE total across every function analyzed so far
/// in this call, and comparing it against `limit` directly gives
/// artifact-wide (not per-function) accounting for free, including in
/// the diagnostic message when it is exceeded.
#[cfg(feature = "std")]
struct AnalysisWorkMeter {
    used: usize,
    limit: usize,
}

#[cfg(feature = "std")]
impl AnalysisWorkMeter {
    fn new(limit: usize) -> Self {
        Self { used: 0, limit }
    }

    /// #1756 Codex review round 19 (owner decision): the ONLY sanctioned
    /// way to increment `used` - every charge site in this crate must
    /// call this, never mutate `self.used` directly, so there is no
    /// bypass of the checked-arithmetic guarantee below. `used` is
    /// artifact-CUMULATIVE (see this struct's own doc comment) and
    /// `max_work_units` is a caller-supplied `usize`, so on a 32-bit
    /// target a caller can legally configure `max_work_units =
    /// usize::MAX` and enough individually state-bounded functions can
    /// still cumulatively drive `used` to `usize::MAX` before this call -
    /// an unchecked `self.used += 1` at that point is build-dependent
    /// (`debug`: panic; `release`: silently wrap to `0`, after which the
    /// verifier would resume accepting further work UNDER the configured
    /// limit) rather than deterministic, violating the whole point of
    /// this being a deterministic, machine-independent resource budget
    /// (see `VerificationLimits`'s own doc comment on why this
    /// distinction matters at all).
    ///
    /// Returns `Ok(used)` (the new cumulative total, exactly as before,
    /// for the caller's own diagnostic reporting) when charging one unit
    /// keeps `used <= limit`. Returns `Err(used)` - the ordinary, exact
    /// attempted usage - when charging one more unit would exceed
    /// `limit` but is still representable. Returns `Err(usize::MAX)` -
    /// the same overflow sentinel `check_analysis_state_budget` and
    /// `check_raw_node_state_budget` already use for their own `checked_
    /// add` failures - when `used + 1` itself cannot be represented as a
    /// `usize` at all: the true attempted usage would be `usize::MAX +
    /// 1`, which has no finite representation, so it is treated as
    /// certainly exceeding any finite `limit`, exactly like every other
    /// overflow path in this verifier's resource envelope. Never panics,
    /// never wraps, never saturates-and-continues - overflow fails
    /// closed, immediately, as an ordinary `AnalysisWorkLimitExceeded`
    /// rejection at the caller, not a host-dependent crash or silent
    /// reset.
    fn charge_one(&mut self) -> Result<usize, usize> {
        let Some(used) = self.used.checked_add(1) else {
            return Err(usize::MAX);
        };
        self.used = used;
        if used > self.limit {
            return Err(used);
        }
        Ok(used)
    }
}

/// **Work budget** (#1756 Codex review round 13, owner decision; scope
/// corrected to be artifact-wide in round 14 - see `AnalysisWorkMeter`):
/// a deterministic `work_units` counter - one unit per register checked
/// while seeding from entry, plus one per `(position, bit)` pair
/// dequeued from the worklist (`event_count`'s own definition) - is
/// charged against the CALLER-OWNED, POSSIBLY-ALREADY-PARTIALLY-SPENT
/// `work_meter` after every increment, in BOTH loops below, aborting
/// with `Err(work_meter.used)` (the cumulative artifact-wide total, not
/// just this call's own contribution) the instant `work_meter.limit` is
/// exceeded. This is deliberately machine-independent: it counts
/// discrete lattice-propagation steps, never wall-clock time, allocator
/// behavior, or CPU cycles, and it aborts mid-computation rather than
/// only checking a final total - a genuinely unbounded input (e.g. the
/// round-13 branch-dense construction with no budget) must never be
/// allowed to run this loop to completion before being rejected. The
/// bound this enforces per call is exactly the same one the doc comments
/// above already prove: `work_units <= domain_size + reachable_count *
/// domain_size` (with `reachable_count` here being the CALLER's
/// compressed `leader_count`) - summed across however many functions
/// share this `work_meter` before it is exhausted.
#[cfg(feature = "std")]
fn compute_missing_sets(
    reachable_count: usize,
    domain_size: usize,
    entry_missing: RegSet,
    writes_contains: impl Fn(usize, usize) -> bool,
    mut successors_of: impl FnMut(usize) -> [Option<usize>; 2],
    work_meter: &mut AnalysisWorkMeter,
) -> Result<(Vec<RegSet>, usize, usize), usize> {
    let mut missing: Vec<RegSet> = (0..reachable_count)
        .map(|pos| {
            if pos == 0 {
                entry_missing.clone()
            } else {
                RegSet::empty()
            }
        })
        .collect();
    if reachable_count == 0 {
        return Ok((missing, 0, 0));
    }

    // A stack (LIFO), not a FIFO queue - see the peak-live-memory doc
    // comment above for why this bounds peak size without weakening the
    // total-work bound at all.
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut peak_queue_len: usize = 0;

    // Seed from entry's own MISSING_OUT (post-write) - entry's own MISSING
    // is fixed and never re-delivered to (see `deliver`), so this is the
    // one place entry's contribution enters the worklist.
    let entry_successors = successors_of(0);
    for bit in 0..domain_size {
        work_meter.charge_one()?;
        if missing[0].contains(bit) && !writes_contains(0, bit) {
            for succ in entry_successors.into_iter().flatten() {
                if deliver(&mut missing, succ, bit) {
                    stack.push((succ, bit));
                }
            }
        }
    }
    peak_queue_len = peak_queue_len.max(stack.len());

    let mut event_count: usize = 0;
    while let Some((pos, bit)) = stack.pop() {
        work_meter.charge_one()?;
        event_count += 1;
        if writes_contains(pos, bit) {
            continue; // killed here - defined by this instruction's own write
        }
        for succ in successors_of(pos).into_iter().flatten() {
            if deliver(&mut missing, succ, bit) {
                stack.push((succ, bit));
            }
        }
        peak_queue_len = peak_queue_len.max(stack.len());
    }

    Ok((missing, event_count, peak_queue_len))
}

/// #1756 Codex review round 12: `JMP_IF` unconditionally records
/// `InstructionSuccessors::Two(target, next_offset)` - if its taken
/// branch happens to target the very next instruction (a legal, if
/// pointless, no-op conditional branch), `target == next_offset` and
/// `for_each_reachable_successor` reports the SAME reachable position
/// twice. Left as-is, every consumer of `[Option<usize>; 2]` successors
/// below (`compute_leaders`, `chain_step`, `build_leader_chains`,
/// `find_first_violation`) would see that as two edges rather than one -
/// `compute_leaders` in particular would double-count the shared
/// target's in-degree AND treat it as a branch target, forcing it (and,
/// for a whole chain of such instructions, every one of their targets)
/// to become its own leader, defeating round 11's compression entirely
/// for that chain. Collapsing it back to a single logical edge here,
/// once, is what lets every downstream consumer keep treating the two
/// slots as "distinct successors, if both present" as a precondition
/// rather than a fact they each have to re-derive.
#[cfg(feature = "std")]
fn dedup_successors(succs: [Option<usize>; 2]) -> [Option<usize>; 2] {
    match succs {
        [Some(a), Some(b)] if a == b => [Some(a), None],
        other => other,
    }
}

/// #1756 Codex review round 15: canonical, MACHINE-INDEPENDENT logical
/// byte sizes for the fixed (non-`RegSet`-payload) structural overhead
/// `check_analysis_state_budget` charges per leader / per stack entry -
/// deliberately NOT `size_of::<T>()` (a host-dependent Rust ABI detail
/// that could vary by target), so admission is deterministic across
/// hosts. Each ties to the exact accounted structure, per the review's
/// own request to avoid one opaque magic constant:
///
/// - `REGSET_HEADER_LOGICAL_BYTES`: one `RegSet { words: Vec<u64> }`'s
///   OWN header (a `Vec`'s canonical ptr+len+cap triple, 3 words) - NOT
///   its backing payload, which `word_count(domain_size)` already
///   charges separately. `chain_killed` and `missing` each hold
///   `leader_count` of these.
/// - `CHAIN_TARGET_LOGICAL_BYTES`: one `[Option<usize>; 2]` entry in
///   `chain_targets` - two canonical `Option<usize>` slots (no niche
///   optimization for a bare `usize` payload, so 2 words each: a
///   discriminant plus the value).
/// - `LEADER_INDEX_LOGICAL_BYTES`: one `usize` entry in
///   `leader_positions`.
/// - `STACK_ENTRY_LOGICAL_BYTES`: one `(usize, usize)` entry in
///   `compute_missing_sets`'s worklist stack.
/// - `REACHABLE_INDEX_LOGICAL_BYTES`: one `usize` entry in
///   `reachable_indices` (round 16).
/// - `CSR_OFFSET_LOGICAL_BYTES`: one `u32` entry in `reads_offsets` or
///   `writes_offsets` (round 16).
/// - `REGISTER_OPERAND_LOGICAL_BYTES`: one `u16` entry in `reads_flat`,
///   `writes_flat`, or the transient `universe` buffer `dataflow_domain_
///   accounting` builds before its own final dedup (round 16).
/// - `LEADER_FLAG_LOGICAL_BYTES`: one `bool` entry in `is_leader` -
///   Rust's `Vec<bool>` is not bit-packed, so this is a full byte per
///   entry, not one bit (round 16).
/// - `LEADER_CLASSIFICATION_STATE_LOGICAL_BYTES`: one `u8` entry in
///   `compute_leaders`'s own `state` scratch array - simultaneously
///   live with `is_leader` while `is_leader` is being built (round 21;
///   see `compute_leaders`'s own doc comment for why one consolidated
///   array replaces the two, `in_degree`/`branch_target`, an earlier
///   revision used).
#[cfg(feature = "std")]
const WORD_BYTES: usize = 8;
#[cfg(feature = "std")]
const REGSET_HEADER_LOGICAL_BYTES: usize = 3 * WORD_BYTES;
#[cfg(feature = "std")]
const CHAIN_TARGET_LOGICAL_BYTES: usize = 2 * (2 * WORD_BYTES);
#[cfg(feature = "std")]
const LEADER_INDEX_LOGICAL_BYTES: usize = WORD_BYTES;
#[cfg(feature = "std")]
const STACK_ENTRY_LOGICAL_BYTES: usize = 2 * WORD_BYTES;
#[cfg(feature = "std")]
const REACHABLE_INDEX_LOGICAL_BYTES: usize = WORD_BYTES;
#[cfg(feature = "std")]
const CSR_OFFSET_LOGICAL_BYTES: usize = WORD_BYTES / 2;
#[cfg(feature = "std")]
const REGISTER_OPERAND_LOGICAL_BYTES: usize = WORD_BYTES / 4;
#[cfg(feature = "std")]
const LEADER_FLAG_LOGICAL_BYTES: usize = 1;
#[cfg(feature = "std")]
const LEADER_CLASSIFICATION_STATE_LOGICAL_BYTES: usize = 1;

/// #1756 Codex review round 17 (owner decision): the exact sum of
/// encoded byte lengths of every REACHABLE structural instruction,
/// computed from already-known instruction boundaries (`instr_starts`),
/// requiring no second decoder, no opcode-specific length heuristics,
/// and no new large allocation (in particular, never a sorted copy of
/// `reachable_offsets`, and never `dataflow_domain_accounting`'s CSR
/// tables, which this value exists to bound BEFORE they're built).
///
/// Round 16's `check_raw_node_state_budget` bounded `reads_flat.len() +
/// writes_flat.len()` using the TOTAL structural instruction-section
/// length (`instr_len = code.len() - instr_start`) as a proxy for "bytes
/// available to encode reachable operands" - a SOUND upper bound, but
/// one that overcharges whenever structurally valid but UNREACHABLE
/// trailing code exists. `dataflow_domain_accounting` only ever decodes
/// and stores operands for REACHABLE instructions (Codex review round
/// 3's own fix), so unreachable bytes contribute nothing to the real
/// `reads_flat`/`writes_flat` cost - yet round 16's `instr_len` proxy
/// charged the WHOLE section regardless, letting a construction like
/// "entry `RET` followed by megabytes of valid unreachable `RET`
/// padding" (the exact shape `c1756_rejects_large_unreachable_ret_
/// padding_without_dense_metadata` already exercises for a DIFFERENT
/// reason) drive `AnalysisStateLimitExceeded` from padding alone -
/// Codex review round 17's finding, changing this verifier's pre-
/// existing, deliberate "unreachable code is not judged by this pass"
/// policy for the FIRST time since #1756 began.
///
/// For each reachable instruction start offset, its encoded length is
/// the distance to the NEXT structural instruction start, or to the end
/// of the instruction section for the last structural instruction - the
/// same canonical boundary every other structural check in this
/// verifier already trusts (jump-target validation, debug-symbol
/// placement), never re-derived via opcode-specific arithmetic. Every
/// reachable offset is looked up in `instr_starts` via `binary_search`
/// (the identical lookup `dataflow_domain_accounting` itself performs,
/// via `filter_map`, to build `reachable_indices`), so this costs
/// O(reachable_count * log(total_instruction_count)) time and O(1)
/// additional space.
///
/// Sound invariant, not merely assumed: every reachable instruction's
/// byte span is a disjoint sub-range of the single `[0, instr_len)`
/// structural instruction section, so this sum can never exceed
/// `instr_len` itself. `checked_add`/`checked_sub` are used throughout
/// regardless, consistent with this codebase's established practice,
/// even though overflow is not reachable given that invariant.
#[cfg(feature = "std")]
fn reachable_instruction_bytes(
    instr_starts: &[usize],
    reachable_offsets: &HashSet<usize>,
    instr_len: usize,
) -> Option<usize> {
    reachable_offsets
        .iter()
        .filter_map(|offset| instr_starts.binary_search(offset).ok())
        .try_fold(0usize, |total, idx| {
            let start = instr_starts[idx];
            let end = instr_starts.get(idx + 1).copied().unwrap_or(instr_len);
            let len = end.checked_sub(start)?;
            total.checked_add(len)
        })
}

/// #1756 Codex review round 16 (owner decision), corrected in round 17:
/// the deterministic, verifier-OWNED pre-allocation admission check for
/// the RAW reachable-node representation `dataflow_domain_accounting`
/// and `compute_leaders` build, BEFORE either one allocates - closing
/// the gap round 15 explicitly, but incorrectly, excluded (see `check_
/// analysis_state_budget`'s own doc comment for the corrected
/// inventory).
///
/// A genuinely branch-FREE reachable stream (a long linear run of
/// single-register instructions, no merges, no branch targets at all)
/// has `leader_count = 1` REGARDLESS of how long it is - round 15's
/// leader/domain-scaled formula alone charges almost nothing for it,
/// even as `reachable_count` (and every structure sized by it) grows
/// without bound. This is Codex's round-16 P1 finding: raw reachable-
/// node state was never bounded by anything, independent of leader or
/// domain structure.
///
/// **Why this check can run BEFORE `dataflow_domain_accounting`**:
/// `reachable_count` is already known for free - `reachable_offsets`
/// (from `verify_reachable_control_flow`, `HashSet<usize>`) reports its
/// own `len()` in O(1), no CSR allocation required to learn it. The
/// caller computes `reachable_instruction_bytes` (see that function's
/// own doc comment - round 17's correction) from the SAME already-known
/// `instr_starts`/`reachable_offsets` inputs, likewise before any of
/// this analysis's own allocations exist. Both inputs exist strictly
/// BEFORE `dataflow_domain_accounting` or `compute_leaders` ever
/// allocate anything, so this check can - and does - run first.
///
/// **Deriving a sound per-instruction-operand-count-independent bound**:
/// rather than enumerate a "maximum registers per opcode" table (some
/// opcodes, e.g. `Call`'s argument list and `MakeClosure`'s capture
/// list, are genuinely variadic, bounded only by how many 2-byte `u16`
/// register slots remain in the code buffer), this uses a strictly
/// information-theoretic argument instead: every register operand this
/// pass ever records - read or write, on any instruction - costs
/// EXACTLY 2 bytes to encode (`read_u16_le`), and every such byte comes
/// from the reachable instructions' own encoded byte spans (Codex review
/// round 3's own `decode_operands` re-decode, see `dataflow_domain_
/// accounting`'s doc comment) - NOT the total structural instruction
/// section, which may also contain unreachable bytes this analysis never
/// decodes (round 17's correction; see `reachable_instruction_bytes`'s
/// own doc comment for why `instr_len` alone overcharged). Therefore
/// `reads_flat.len() + writes_flat.len() <= reachable_instruction_bytes
/// / 2` - a SOUND, provable, information-theoretic ceiling that holds no
/// matter how those operands are distributed (one huge variadic
/// instruction or many small fixed-arity ones cost the same either way),
/// computed with zero per-opcode knowledge and zero CSR allocation. This
/// is `operand_count_bound` below.
///
/// **What this bounds** (every raw-node structure `prove_definite_
/// register_assignment` and its callees allocate whose size scales with
/// `reachable_count` or `reachable_instruction_bytes`, genuinely owned
/// by THIS analysis - see `check_analysis_state_budget`'s own doc
/// comment for the full, corrected inventory including what remains
/// deliberately excluded): `reachable_indices`, `reads_offsets` +
/// `writes_offsets`, `reads_flat` + `writes_flat` (bounded by `operand_
/// count_bound` above), the transient `universe` buffer `dataflow_
/// domain_accounting` builds before its own final dedup (also bounded
/// by `operand_count_bound`, plus `entry_param_count`), and `is_leader`.
///
/// **Formula**: `raw_words = ceil(raw_bytes / WORD_BYTES)` for
/// `raw_bytes = reachable_count * REACHABLE_INDEX_LOGICAL_BYTES +
/// (reachable_count + 1) * 2 * CSR_OFFSET_LOGICAL_BYTES + reachable_
/// count * LEADER_FLAG_LOGICAL_BYTES + (2 * operand_count_bound +
/// entry_param_count) * REGISTER_OPERAND_LOGICAL_BYTES`, where `operand_
/// count_bound = reachable_instruction_bytes / 2` (round 17 - NOT total
/// `instr_len / 2`) - every multiplication and addition `checked`;
/// overflow anywhere is the same `usize::MAX` sentinel used throughout
/// this envelope.
///
/// Returns `Ok(raw_words)` when within budget, `Err(reported_words)`
/// otherwise - the caller (`prove_definite_register_assignment`) rejects
/// immediately on `Err`, before `dataflow_domain_accounting` or `compute_
/// leaders` ever run, and reuses the `Ok` value as an input to the later,
/// COMBINED `check_analysis_state_budget` call once `leader_count` and
/// `domain_size` are also known.
#[cfg(feature = "std")]
fn check_raw_node_state_budget(
    reachable_count: usize,
    reachable_instruction_bytes: usize,
    entry_param_count: usize,
    max_state_words: usize,
) -> Result<usize, usize> {
    let overflow = || Err(usize::MAX);

    let operand_count_bound = reachable_instruction_bytes / 2;

    let reachable_index_bytes = reachable_count.checked_mul(REACHABLE_INDEX_LOGICAL_BYTES);
    let Some(reachable_index_bytes) = reachable_index_bytes else {
        return overflow();
    };

    let offset_entries = reachable_count.checked_add(1);
    let Some(offset_entries) = offset_entries else {
        return overflow();
    };
    let offset_bytes = offset_entries
        .checked_mul(2) // reads_offsets + writes_offsets
        .and_then(|doubled| doubled.checked_mul(CSR_OFFSET_LOGICAL_BYTES));
    let Some(offset_bytes) = offset_bytes else {
        return overflow();
    };

    let leader_flag_bytes = reachable_count.checked_mul(LEADER_FLAG_LOGICAL_BYTES);
    let Some(leader_flag_bytes) = leader_flag_bytes else {
        return overflow();
    };

    // #1756 Codex review round 21 (owner decision), P2: `compute_
    // leaders`'s own `state` scratch array (see its doc comment) is
    // simultaneously live with `is_leader` while `is_leader` is being
    // built - the genuine peak moment for the "raw" phase of this
    // analysis - so its cost is summed in here too, not omitted.
    let leader_classification_state_bytes =
        reachable_count.checked_mul(LEADER_CLASSIFICATION_STATE_LOGICAL_BYTES);
    let Some(leader_classification_state_bytes) = leader_classification_state_bytes else {
        return overflow();
    };

    // reads_flat + writes_flat (operand_count_bound entries combined)
    // plus the transient pre-dedup `universe` buffer (entry_param_count
    // + operand_count_bound entries) - see the doc comment above.
    let operand_bytes = operand_count_bound
        .checked_mul(2)
        .and_then(|doubled| doubled.checked_add(entry_param_count))
        .and_then(|total_entries| total_entries.checked_mul(REGISTER_OPERAND_LOGICAL_BYTES));
    let Some(operand_bytes) = operand_bytes else {
        return overflow();
    };

    let raw_bytes = reachable_index_bytes
        .checked_add(offset_bytes)
        .and_then(|sum| sum.checked_add(leader_flag_bytes))
        .and_then(|sum| sum.checked_add(leader_classification_state_bytes))
        .and_then(|sum| sum.checked_add(operand_bytes));
    let Some(raw_bytes) = raw_bytes else {
        return overflow();
    };

    let raw_words = raw_bytes
        .checked_add(WORD_BYTES - 1)
        .map(|rounded| rounded / WORD_BYTES);
    let Some(raw_words) = raw_words else {
        return overflow();
    };

    match raw_words {
        w if w <= max_state_words => Ok(w),
        w => Err(w),
    }
}

/// #1756 Codex review round 14 (owner decision), corrected in round 15,
/// extended in round 16, corrected again in round 17: the deterministic,
/// verifier-OWNED pre-allocation admission check for the definite-
/// register-assignment analysis's PEAK simultaneously-live verifier-
/// analysis state - now the FULL logical storage this analysis needs,
/// raw reachable-node representation included, not merely dense
/// `RegSet` backing words.
///
/// **Full lifetime inventory** (every allocation `prove_definite_
/// register_assignment` and its callees own whose size scales with
/// `reachable_count`, `leader_count`, `domain_size`, the total encoded
/// bytes of REACHABLE instructions (`reachable_instruction_bytes` -
/// round 17: NOT the total structural instruction-section length,
/// which may also include unreachable bytes this analysis never
/// decodes), or worklist depth - round 15's own request, extended in
/// round 16 after Codex demonstrated the round-15 raw-node exclusion
/// below was itself exploitable, corrected in round 17 after Codex
/// demonstrated round 16's own total-byte proxy overcharged unreachable
/// padding):
///
/// - `chain_killed` / `missing` (`Vec<RegSet>`, `leader_count` entries
///   each): dense payload already charged by round 14's `(2 *
///   leader_count + 1) * word_count(domain_size)` term; round 15 added
///   their own `Vec<RegSet>` HEADERS - `REGSET_HEADER_LOGICAL_BYTES`
///   per entry, times 2 arrays, times `leader_count`. Simultaneously
///   live for `compute_missing_sets`'s entire call (round 14's finding).
/// - `chain_targets` (`Vec<[Option<usize>; 2]>`, `leader_count`
///   entries): `CHAIN_TARGET_LOGICAL_BYTES` per entry. Built before
///   `compute_missing_sets` runs and read throughout it via
///   `compressed_successors_of`.
/// - `leader_positions` (`Vec<usize>`, `leader_count` entries):
///   `LEADER_INDEX_LOGICAL_BYTES` per entry. Round 20 (Codex P2)
///   corrected an earlier belief (rounds 15-19: "necessarily built
///   BEFORE this very check can run at all... the one structure this
///   check cannot gate ahead of itself by construction") - that was
///   never actually true. `leader_count` is derived by counting `is_
///   leader`'s true bits (`is_leader.iter().filter(..).count()`, zero
///   additional allocation - `is_leader` itself is already `raw_words`-
///   authorized, see `check_raw_node_state_budget`'s own doc comment),
///   so this check CAN and DOES gate ahead of `leader_positions`'s own
///   materialization: the check runs first, using only the count, and
///   `leader_positions` is collected (via the identical `.filter(..).
///   collect()` expression as before) only once this check has already
///   authorized its cost. No exception remains in this envelope.
/// - `compute_missing_sets`'s worklist stack (`Vec<(usize, usize)>`):
///   proven (see `c1756_reversed_order_diamond_chain_stack_peak_
///   scales_with_leader_count` for the adversarial construction and
///   the argument below) to peak at `leader_count + 2 * domain_size`
///   entries, NOT the domain_size-only "generous ceiling" round 9's own
///   test happened to observe (that test never stressed successor
///   ORDER, which is what actually controls this). Proof sketch: at any
///   moment the stack is `settled_backlog` (items from the seed phase
///   or earlier bits, bounded by the seed loop's own `2 * domain_size`
///   push ceiling, and never regrown once shrunk since only the
///   CURRENTLY active bit's traversal pushes new items) plus
///   `active_frontier` (the one currently-being-traced bit's own unresolved
///   siblings, bounded by `leader_count` via `deliver`'s "each position
///   visited at most once per bit" guarantee) - `STACK_ENTRY_LOGICAL_
///   BYTES` per entry.
/// - `entry_missing` / `build_leader_chains`'s in-progress `killed` /
///   `find_first_violation`'s per-leader `local` clone: O(1) each,
///   never overlapping in time (established in round 14) - covered by
///   the existing `+1` leader in the dense-payload term, unchanged.
/// - `reachable_indices` (`Vec<usize>`, `reachable_count` entries),
///   `reads_offsets` / `writes_offsets` (`Vec<u32>`, `reachable_count +
///   1` entries each), `reads_flat` / `writes_flat` (`Vec<u16>`,
///   combined length bounded by `reachable_instruction_bytes / 2` -
///   round 17: NOT total `instr_len / 2`, see `reachable_instruction_
///   bytes`'s and `check_raw_node_state_budget`'s doc comments for the
///   exact information-theoretic argument and why the distinction
///   matters), and the transient pre-dedup `universe` buffer (same
///   bound, plus `entry_param_count`): all built by `dataflow_domain_
///   accounting`, exclusively for THIS analysis - never allocated for a
///   signature-less function, unlike the genuinely shared structures
///   below. Round 16 (Codex P1): charged via the new, EARLIER `check_
///   raw_node_state_budget`, which runs strictly before `dataflow_
///   domain_accounting` itself, so this protection is real, not
///   retroactive.
/// - `is_leader` (`Vec<bool>`, `reachable_count` entries): `LEADER_FLAG_
///   LOGICAL_BYTES` per entry. Round 15 grouped this with the CSR
///   tables above under "excluded, reachable_count-scaled, pre-existing
///   gap" - that grouping was IMPRECISE: `compute_leaders` (which
///   allocates it) is called ONLY from within this analysis, exclusively
///   for signature-bearing functions, exactly like the CSR tables, not
///   shared with any check that runs regardless of signature. Corrected
///   in round 16: charged by `check_raw_node_state_budget`, not
///   excluded.
/// - `compute_leaders`'s own `state` scratch array (`Vec<u8>`,
///   `reachable_count` entries): `LEADER_CLASSIFICATION_STATE_LOGICAL_
///   BYTES` per entry. Round 21 (Codex P2) found this consolidated
///   classification-state array (which replaced two separate `in_
///   degree`/`branch_target` scratch arrays of the same `reachable_
///   count` size - see `compute_leaders`'s own doc comment) simultaneously
///   live with `is_leader` while `is_leader` is being built, the genuine
///   peak moment for the raw phase, yet uncharged by any check. Charged
///   in `check_raw_node_state_budget` alongside `is_leader`, not
///   excluded - same reasoning as `is_leader` above: `compute_leaders`
///   is called only from within this analysis, exclusively for
///   signature-bearing functions.
/// - Genuinely still EXCLUDED, and NOT silently: `instr_starts` (`Vec<
///   usize>`) and `instruction_successors` (`Vec<InstructionSuccessors>`),
///   both sized to the function's TOTAL instruction count (not merely
///   `reachable_count` - unreachable code is decoded once too), and
///   `reachable_offsets` (`HashSet<usize>`, from `verify_reachable_
///   control_flow`). All three are built, with their own independent
///   verification work performed, for EVERY function `verify_function_
///   code` processes - signature-bearing or not (debug-symbol placement,
///   jump-target validity, and general reachable-control-flow validity
///   all depend on them regardless of whether a canonical `SIG0`
///   signature exists at all). #1756's analysis borrows all three BY
///   REFERENCE rather than allocating its own copies, and they cost
///   exactly the same whether or not this analysis ever runs on a given
///   function. Bounding raw total-instruction-count (independent of
///   reachability, leader, or domain structure) remains the SAME
///   explicitly acknowledged, separate, out-of-scope gap since round 1
///   of this PR's review (no code-size/instruction-count quota exists
///   anywhere in this verifier) - this round does not change that
///   boundary, and does not claim to.
///
/// **Formula**: `required_state_words = raw_words + dense_words +
/// fixed_words`, where `raw_words` comes from the EARLIER `check_raw_
/// node_state_budget` call (see its own doc comment), and `dense_words`/
/// `fixed_words` are round 15's unchanged terms: `dense_words = (2 *
/// leader_count + 1) * word_count(domain_size)`, `fixed_words =
/// ceil(fixed_bytes / WORD_BYTES)` for `fixed_bytes = leader_count *
/// (2 * REGSET_HEADER_LOGICAL_BYTES + CHAIN_TARGET_LOGICAL_BYTES +
/// LEADER_INDEX_LOGICAL_BYTES) + (leader_count + 2 * domain_size) *
/// STACK_ENTRY_LOGICAL_BYTES`. Every multiplication and addition is
/// `checked`, with the byte-to-word rounding also checked even though
/// the chosen logical sizes happen to divide evenly. Overflow anywhere
/// in the chain is treated as certainly exceeding any finite
/// `max_state_words`, the same honest `usize::MAX` sentinel established
/// in round 13, never undefined or silently-wrapping behavior.
///
/// `raw_words` and `dense_words + fixed_words` are SUMMED, never
/// maxed: the raw-node structures above are never dropped before the
/// leader-state phase begins - `reachable_indices`, `reads_flat`,
/// `writes_flat`, their offset tables, and `is_leader` are all still
/// read by `find_first_violation` at the very end of this analysis, so
/// every structure this envelope charges for is simultaneously live for
/// the analysis's ENTIRE remaining duration once allocated. Per the
/// owner's own "peak, not double-counted lifetimes" rule: overlapping
/// phases sum; only genuinely mutually-exclusive phases would take a
/// max, and no phase here is ever dropped before the next begins.
///
/// This remains a PEAK-simultaneous-state envelope for ONE function's
/// analysis, not cumulative across an artifact's functions (functions
/// are verified sequentially and each one's analysis state is released
/// before the next begins - see `AnalysisWorkMeter` for the DIFFERENT,
/// genuinely cumulative-across-the-artifact accounting `max_work_units`
/// uses instead). Returns `Ok(required_words)` (the exact peak count,
/// for accounting) when within budget, `Err(reported_words)` otherwise.
#[cfg(feature = "std")]
fn check_analysis_state_budget(
    raw_words: usize,
    leader_count: usize,
    domain_size: usize,
    max_state_words: usize,
) -> Result<usize, usize> {
    let overflow = || Err(usize::MAX);

    let words_per_leader = RegSet::word_count(domain_size);
    let dense_words = leader_count
        .checked_mul(2)
        .and_then(|doubled| doubled.checked_add(1))
        .and_then(|factor| factor.checked_mul(words_per_leader));
    let Some(dense_words) = dense_words else {
        return overflow();
    };

    let per_leader_fixed_bytes = REGSET_HEADER_LOGICAL_BYTES
        .checked_mul(2)
        .and_then(|regset_headers| regset_headers.checked_add(CHAIN_TARGET_LOGICAL_BYTES))
        .and_then(|sum| sum.checked_add(LEADER_INDEX_LOGICAL_BYTES));
    let Some(per_leader_fixed_bytes) = per_leader_fixed_bytes else {
        return overflow();
    };
    let leader_fixed_bytes = leader_count.checked_mul(per_leader_fixed_bytes);
    let Some(leader_fixed_bytes) = leader_fixed_bytes else {
        return overflow();
    };

    // Stack peak: `leader_count + 2 * domain_size` entries (see the doc
    // comment's proof sketch), each `STACK_ENTRY_LOGICAL_BYTES`.
    let stack_peak_entries = domain_size
        .checked_mul(2)
        .and_then(|doubled| doubled.checked_add(leader_count));
    let Some(stack_peak_entries) = stack_peak_entries else {
        return overflow();
    };
    let stack_bytes = stack_peak_entries.checked_mul(STACK_ENTRY_LOGICAL_BYTES);
    let Some(stack_bytes) = stack_bytes else {
        return overflow();
    };

    let fixed_bytes = leader_fixed_bytes.checked_add(stack_bytes);
    let Some(fixed_bytes) = fixed_bytes else {
        return overflow();
    };
    let fixed_words = fixed_bytes
        .checked_add(WORD_BYTES - 1)
        .map(|rounded| rounded / WORD_BYTES);
    let Some(fixed_words) = fixed_words else {
        return overflow();
    };

    let leader_words = dense_words.checked_add(fixed_words);
    let Some(leader_words) = leader_words else {
        return overflow();
    };

    let required_words = raw_words.checked_add(leader_words);
    match required_words {
        Some(required_words) if required_words <= max_state_words => Ok(required_words),
        Some(required_words) => Err(required_words),
        None => Err(usize::MAX),
    }
}

/// #1756 Codex review round 11: identifies which reachable positions are
/// "leaders" - the only positions that need their own entry in the
/// compressed dataflow graph `compute_missing_sets` runs over (see
/// `build_leader_chains` and `prove_definite_register_assignment`'s doc
/// comment for the finding this closes). A position is a leader iff it
/// is function entry (position 0), or it has more than one predecessor
/// (a merge point), or it is one of two successors of some other
/// position (a branch target). Every other position has exactly one
/// predecessor, and that predecessor has exactly one successor - a
/// genuine 1-to-1 edge - so its MISSING value is always identical to its
/// predecessor's MISSING_OUT and never needs independent storage or
/// worklist participation; `chain_step` walks straight past it instead.
///
/// **Precondition** (#1756 Codex review round 12): `successors_of`'s two
/// slots, when both `Some`, must already be DISTINCT positions - a
/// source with only one true logical successor (e.g. a self-targeting
/// `JMP_IF`, see `dedup_successors`) must present that as `[Some(x),
/// None]`, never `[Some(x), Some(x)]`. This function trusts that
/// precondition rather than re-deriving it, so every caller must run
/// raw successors through `dedup_successors` first.
///
/// This always terminates when traced backward from any non-leader
/// position: a cycle reachable from entry must contain a node entered
/// from outside the cycle too (giving that node in-degree >= 2, making
/// it a leader by the first rule) unless the cycle passes through entry
/// itself, which is unconditionally a leader regardless of its own
/// in-degree - so no run of non-leader positions can loop back on itself
/// without first hitting a leader.
///
/// #1756 Codex review round 21 (owner decision), P2: classification uses
/// ONE `reachable_count`-sized `Vec<u8>` (`state`), not the two separate
/// arrays (`in_degree: Vec<u8>`, `branch_target: Vec<bool>`) an earlier
/// revision used - a genuine reduction in peak scratch storage, not
/// merely a relabeling. This is sound because `branch_target[pos]` only
/// ever changes the final `is_leader` verdict when `in_degree[pos] ==
/// 1` (once in-degree reaches 2, `in_degree[pos] != 1` alone already
/// makes `pos` a leader, so a separate branch-target bit adds no new
/// information for that position ever again) - the two pieces of
/// per-position information are never independently needed at the SAME
/// time, so they fit in one small enum-like state instead of two full
/// arrays. Four states: `0` = never yet visited (in-degree 0), `1` =
/// visited exactly once via a non-branching edge (in-degree 1, not a
/// branch target), `2` = visited exactly once via a branching edge
/// (in-degree 1, IS a branch target), `3` = visited two or more times
/// (in-degree >= 2, already a leader regardless of any branch-target
/// bit). `is_leader[pos] = (pos == 0) || (state[pos] != STATE_IN_DEGREE
/// _ONE)` - state `1` is the ONLY "not yet proven a leader by these
/// rules" value; every other state already implies leader status,
/// collapsing the original two-condition check (`in_degree[pos] != 1 ||
/// branch_target[pos]`) into one comparison.
#[cfg(feature = "std")]
const STATE_UNVISITED: u8 = 0;
#[cfg(feature = "std")]
const STATE_IN_DEGREE_ONE: u8 = 1;
#[cfg(feature = "std")]
const STATE_IN_DEGREE_ONE_BRANCH_TARGET: u8 = 2;
#[cfg(feature = "std")]
const STATE_IN_DEGREE_MANY: u8 = 3;

#[cfg(feature = "std")]
fn compute_leaders(
    reachable_count: usize,
    mut successors_of: impl FnMut(usize) -> [Option<usize>; 2],
) -> Vec<bool> {
    let mut state: Vec<u8> = vec![STATE_UNVISITED; reachable_count];
    for pos in 0..reachable_count {
        let succs = successors_of(pos);
        let out_degree = succs.iter().filter(|s| s.is_some()).count();
        for succ in succs.into_iter().flatten() {
            state[succ] = match state[succ] {
                STATE_UNVISITED if out_degree > 1 => STATE_IN_DEGREE_ONE_BRANCH_TARGET,
                STATE_UNVISITED => STATE_IN_DEGREE_ONE,
                STATE_IN_DEGREE_ONE | STATE_IN_DEGREE_ONE_BRANCH_TARGET => STATE_IN_DEGREE_MANY,
                STATE_IN_DEGREE_MANY => STATE_IN_DEGREE_MANY,
                _ => unreachable!("state is always one of the four named constants above"),
            };
        }
    }
    (0..reachable_count)
        .map(|pos| pos == 0 || state[pos] != STATE_IN_DEGREE_ONE)
        .collect()
}

/// One step of a leader's compressed run (#1756 Codex review round 11):
/// `Continue` while the current position has exactly one successor and
/// that successor is not itself a leader (see `compute_leaders`);
/// `End` once the run reaches a leader, a branch, or a dead end,
/// carrying whichever successors it actually has (0, 1, or 2 - each
/// `Some` one is guaranteed to be a leader by construction of
/// `compute_leaders`). Shared by chain-summary building and final read
/// validation so both agree, by construction, on exactly which
/// positions belong to which leader's run.
#[cfg(feature = "std")]
enum ChainStep {
    Continue(usize),
    End([Option<usize>; 2]),
}

#[cfg(feature = "std")]
fn chain_step(succs: [Option<usize>; 2], is_leader: &[bool]) -> ChainStep {
    match succs {
        [Some(s), None] if !is_leader[s] => ChainStep::Continue(s),
        other => ChainStep::End(other),
    }
}

/// #1756 Codex review round 11: walks each leader's compressed run once,
/// folding every visited position's writes into ONE cumulative kill set
/// per leader and recording which leader(s) (by compact leader-index,
/// via `leader_index_of`) the run eventually reaches. This is what lets
/// `compute_missing_sets` run over `leader_positions.len()` nodes instead
/// of `reachable_count` - see `prove_definite_register_assignment`'s doc
/// comment for the finding this closes and the accounting it restores.
/// `writes_of` enumerates a position's own writes (dense bit indices) via
/// callback, mirroring `for_each_reachable_successor`'s style - no
/// per-position allocation regardless of how many positions a run visits.
#[cfg(feature = "std")]
fn build_leader_chains(
    leader_positions: &[usize],
    is_leader: &[bool],
    mut successors_of: impl FnMut(usize) -> [Option<usize>; 2],
    mut writes_of: impl FnMut(usize, &mut dyn FnMut(usize)),
    mut leader_index_of: impl FnMut(usize) -> usize,
) -> (Vec<RegSet>, Vec<[Option<usize>; 2]>) {
    let mut chain_killed: Vec<RegSet> = Vec::with_capacity(leader_positions.len());
    let mut chain_targets: Vec<[Option<usize>; 2]> = Vec::with_capacity(leader_positions.len());
    for &leader_pos in leader_positions {
        let mut killed = RegSet::empty();
        let mut pos = leader_pos;
        let targets = loop {
            writes_of(pos, &mut |bit| killed.insert(bit));
            match chain_step(successors_of(pos), is_leader) {
                ChainStep::Continue(next) => pos = next,
                ChainStep::End(succs) => {
                    let mut out = [None, None];
                    for (i, s) in succs.into_iter().flatten().enumerate() {
                        out[i] = Some(leader_index_of(s));
                    }
                    break out;
                }
            }
        };
        chain_killed.push(killed);
        chain_targets.push(targets);
    }
    (chain_killed, chain_targets)
}

/// #1756 Codex review round 11: finds the lowest-*position* violating
/// read across every leader's compressed run, given each leader's own
/// converged MISSING value (`missing`, indexed the same way
/// `leader_positions` is). Each leader's run is walked in EXECUTION
/// order (following successors), which is NOT guaranteed to match
/// ascending position order (see
/// `c1756_backward_propagating_chain_still_rejects_and_converges_promptly`
/// for a genuine, pre-existing case where they're exact opposites) - so
/// this tracks the minimum-position candidate across every run instead
/// of returning on the first one any single walk happens to encounter,
/// preserving the deterministic-first-diagnostic contract
/// (`c1756_case25`/`c1756_case26`) regardless of which leader or which
/// run reaches it first. Returns `(position, dense register bit)`.
#[cfg(feature = "std")]
fn find_first_violation(
    leader_positions: &[usize],
    is_leader: &[bool],
    missing: &[RegSet],
    mut successors_of: impl FnMut(usize) -> [Option<usize>; 2],
    mut reads_of: impl FnMut(usize, &mut dyn FnMut(usize)),
    mut writes_of: impl FnMut(usize, &mut dyn FnMut(usize)),
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (leader_idx, &leader_pos) in leader_positions.iter().enumerate() {
        let mut local = missing[leader_idx].clone();
        let mut pos = leader_pos;
        loop {
            let mut found_here: Option<usize> = None;
            reads_of(pos, &mut |bit| {
                if found_here.is_none() && local.contains(bit) {
                    found_here = Some(bit);
                }
            });
            if let Some(bit) = found_here {
                let better = match best {
                    None => true,
                    Some((best_pos, _)) => pos < best_pos,
                };
                if better {
                    best = Some((pos, bit));
                }
            }
            writes_of(pos, &mut |bit| local.remove(bit));
            match chain_step(successors_of(pos), is_leader) {
                ChainStep::Continue(next) => pos = next,
                ChainStep::End(_) => break,
            }
        }
    }
    best
}

/// #1756 (FA-07-016) Codex review round 8: the dual, MAY-MISSING
/// formulation of definite-register-assignment, replacing the round 1-7
/// MUST/DEFINED/intersection formulation entirely.
///
/// **Derivation** (De Morgan's law over the equations this replaces):
///
///   DEFINED[entry] = ENTRY_DEFS;  DEFINED[n] = intersect(OUT[p] for p in preds(n))
///   OUT[n] = DEFINED[n] union WRITES[n]
///
/// Let MISSING[n] = U minus DEFINED[n]. Then:
///
///   MISSING[entry] = U minus ENTRY_DEFS   (fixed - see below)
///   MISSING[n] = U minus intersect(OUT[p]) = union(U minus OUT[p]) = union(MISSING_OUT[p])
///   MISSING_OUT[n] = U minus OUT[n] = U minus (DEFINED[n] union WRITES[n])
///                  = (U minus DEFINED[n]) intersect (U minus WRITES[n])
///                  = MISSING[n] minus WRITES[n]
///
/// A register `r` is read-safe at `n` iff `r` is not in MISSING[n] - the
/// identical predicate as before (`r` in DEFINED[n]), just phrased via the
/// complement; `decode_operands`'s per-opcode read/write classification
/// (including MAP_GET's conservative `default_val` read, CALL/CLOSURE_CALL's
/// has-dst-flag asymmetry, and same-instruction read-before-write - reads
/// are validated against `MISSING[n]`, the PRE-write state, exactly as the
/// original formulation validated against `DEFINED[n]` pre-write) is
/// entirely unchanged; only the dataflow equations computing `MISSING`
/// itself are new.
///
/// **Initialization**: non-entry nodes start at BOTTOM = empty (the dual of
/// the non-negotiable TOP=U rule the original formulation required - union's
/// identity element is the empty set, as intersection's was U). This is
/// load-bearing for the identical reason `c1756_case12` proves for the
/// original rule: under the wrong rule (non-entry nodes starting at U, the
/// naive-looking mirror of the WRONG empty-init the original formulation
/// explicitly rejected), a loop's first, still-unconverged pass through its
/// own back edge would union with U (already everything, unable to add
/// anything) and get permanently stuck reporting every register missing,
/// even ones a real forward path (unrelated to the loop) already proves
/// defined - the dual of case 12's own wrongly-empty-forever failure mode,
/// under the DEFINED<->MISSING, intersect<->union, TOP<->BOTTOM
/// substitution throughout. `c1756_case12_accepts_loop_read_requires_top_
/// initialization` (kept, unmodified, as a regression) still exercises this
/// exact property under the new formulation.
///
/// Entry (position 0) is never re-unioned by any predecessor, including a
/// back edge that targets instruction 0 directly (structurally legal - a
/// jump to offset 0 is a valid instruction boundary): by the dual of the
/// round-6 argument, every reachable MISSING_OUT is provably a SUBSET of
/// MISSING[entry] (writes only ever REMOVE bits from MISSING, monotonically,
/// so by induction no reachable node's missing set can ever grow past what
/// entry itself starts with) - unioning entry's fixed MISSING with any such
/// subset changes nothing. `compute_missing_sets` encodes this as an
/// explicit `target != 0` guard in `deliver`, rather than relying on the
/// argument alone.
///
/// **Work bound**: see `compute_missing_sets`'s doc comment - this
/// formulation's bit-level worklist is what makes reverse-postorder ranking
/// and strongly-connected-component decomposition (rounds 5-7's successive
/// fixes) unnecessary. Both existed solely to bound how many times a WHOLE
/// `RegSet` could be re-narrowed and re-compared (an O(|U|/64)-per-event
/// cost with no bound on event count without careful ordering); at the bit
/// level, each (position, register) event is inherently processed at most
/// once, so no adversarial ordering or fan-in can force "reprocessing" -
/// there is nothing to reprocess. This also directly closes Codex review
/// round 8's first finding (duplicate live `RegSet` states across
/// branches): every reachable position owns exactly one `RegSet`, allocated
/// once and mutated in place via `insert` - there is no clone-on-write, no
/// structural-sharing bookkeeping, and therefore no way for independently-
/// computed-but-equal branch states to duplicate memory, regardless of how
/// many branches compute the identical result.
///
/// **Leader compression** (#1756 Codex review round 11): round 10 fixed
/// per-position storage that stayed accidentally wide (a chain converging
/// to a small or empty missing set still paid full width per position
/// until it converged); round 11's finding is the complementary case - a
/// long, NON-branching run whose missing set stays wide THROUGHOUT,
/// because nothing along it ever narrows enough (e.g. millions of
/// redundant re-writes to an already-defined register, interleaved among
/// many still-missing ones). Lazy growth doesn't help there: every
/// position genuinely holds close to `domain_size` live bits, so each of
/// `reachable_count` positions still costs close to the full width, and -
/// independently - the bit-level worklist still delivers every one of
/// those live bits at every one of those positions, one hop at a time,
/// achieving close to the full `reachable_count * domain_size` EVENT
/// bound too (not just the memory bound). Neither symptom is fixable by
/// changing any ONE position's own representation, because the actual
/// waste is redundant STORAGE AND PROPAGATION across MANY positions that
/// all hold the identical value.
///
/// The fix: before running `compute_missing_sets` at all, collapse the
/// reachable graph down to its "leaders" (`compute_leaders`) - a position
/// needs its own slot only if it is entry, a merge point, or a branch
/// target; every other position has exactly one predecessor with exactly
/// one successor, so its value is always identical to that predecessor's
/// MISSING_OUT and is derived on demand instead (`chain_step`,
/// `build_leader_chains`, `find_first_violation`). `compute_missing_sets`
/// itself is unchanged - it still runs the identical bit-level worklist,
/// just over `leader_positions.len()` nodes instead of `reachable_count`.
/// For a run with no internal branching at all, `leader_positions.len()`
/// is 1, regardless of how long the run is - collapsing both the storage
/// and the event count for Codex's adversarial shape to `O(domain_size)`,
/// with the run's own length only costing a plain O(1)-per-instruction
/// local walk (no `domain_size` factor) to fold its writes into one
/// cumulative kill set (`build_leader_chains`) and, once the compressed
/// worklist converges, to validate its reads (`find_first_violation`).
#[cfg(feature = "std")]
#[allow(clippy::too_many_arguments)] // private helper; `limits`/`work_meter` (rounds 13-14) have no natural grouping
fn prove_definite_register_assignment(
    function: &str,
    code: &[u8],
    instr_start: usize,
    instr_starts: &[usize],
    instruction_successors: &[InstructionSuccessors],
    reachable_offsets: &HashSet<usize>,
    entry_param_count: usize,
    limits: &VerificationLimits,
    work_meter: &mut AnalysisWorkMeter,
) -> Result<(), RejectReport> {
    if instr_starts.is_empty() {
        return Ok(());
    }

    // #1756 Codex review round 16 (owner decision), corrected in round
    // 17: the raw-node state budget check runs FIRST, before `dataflow_
    // domain_accounting` allocates any reachable-node CSR table -
    // `reachable_count` and `reachable_instruction_bytes` (round 17: the
    // exact reachable-only byte sum, NOT the total structural
    // instruction-section length - see `reachable_instruction_bytes`'s
    // own doc comment for why that distinction matters) are both already
    // known for free at this point (see `check_raw_node_state_budget`'s
    // own doc comment), so this check never needs to build the very
    // structures it exists to bound.
    let reachable_count = reachable_offsets.len();
    let instr_len = code.len().saturating_sub(instr_start);
    let Some(reachable_bytes) =
        reachable_instruction_bytes(instr_starts, reachable_offsets, instr_len)
    else {
        return Err(reject_one(
            function,
            VerificationCode::AnalysisStateLimitExceeded,
            0,
            "definite-register-assignment analysis could not determine the reachable \
             instruction stream's byte length without overflow, exceeding any finite \
             verification state budget"
                .to_string(),
        ));
    };
    let raw_words = match check_raw_node_state_budget(
        reachable_count,
        reachable_bytes,
        entry_param_count,
        limits.max_state_words,
    ) {
        Ok(raw_words) => raw_words,
        Err(raw_words) => {
            return Err(reject_one(
                function,
                VerificationCode::AnalysisStateLimitExceeded,
                0,
                format!(
                    "definite-register-assignment analysis needs at least {} logical verifier-analysis word(s) for raw reachable-node representation alone ({reachable_count} reachable node(s) totaling {reachable_bytes} encoded byte(s) of reachable instructions), exceeding the verification state budget of {} before any leader-compressed state is even computed",
                    if raw_words == usize::MAX { "more than usize::MAX".to_string() } else { raw_words.to_string() },
                    limits.max_state_words
                ),
            ));
        }
    };

    // Codex review round 3 on this PR (#1840): re-decodes exactly one
    // reachable instruction's operands on demand, via the same
    // `decode_operands` the main structural walk already used - never a
    // second decoder, and never materialized for an unreachable
    // instruction. Every byte here already passed full structural
    // validation earlier in `verify_function_code` (opcode recognition,
    // operand shape, canonical domains), so a decode failure here would be
    // an internal invariant violation, not an attacker-reachable outcome.
    //
    // #1756 Codex review round 18 (owner decision): forwards whichever
    // sink `dataflow_domain_accounting` supplies (a `FlatSink` writing
    // straight into its own `reads_flat`/`writes_flat`) rather than
    // collecting into a temporary `Vec<u16>` pair per reachable
    // instruction and copying that into the flat arrays afterward.
    let reads_writes_of = |idx: usize, sink: &mut dyn OperandSink| {
        let offset = instr_starts[idx];
        let mut cursor = instr_start + offset;
        let opcode_byte =
            read_u8(code, &mut cursor).expect("previously-decoded instruction must re-decode");
        let opcode =
            Opcode::from_byte(opcode_byte).expect("previously-decoded instruction must re-decode");
        decode_operands(
            function,
            code,
            &mut cursor,
            offset,
            opcode,
            true,
            sink,
            &mut NoCollect,
        )
        .expect("previously-decoded instruction must re-decode");
    };
    let (reachable_indices, reads_flat, reads_offsets, writes_flat, writes_offsets, universe) =
        dataflow_domain_accounting(
            instr_starts,
            reachable_offsets,
            entry_param_count,
            reads_writes_of,
        );
    // `reachable_count` was already computed for free above, before
    // `dataflow_domain_accounting` ran - confirm the two agree, rather
    // than silently trusting it (round 16).
    debug_assert_eq!(reachable_indices.len(), reachable_count);
    debug_assert_eq!(reachable_indices.first(), Some(&0));
    let domain_size = universe.len();
    let dense = |raw: u16| -> usize {
        universe
            .binary_search(&raw)
            .expect("register must be in U by construction")
    };

    let mut entry_missing = RegSet::full(domain_size);
    for r in 0..entry_param_count {
        entry_missing.remove(dense(r as u16));
    }

    let successors_of = |pos: usize| -> [Option<usize>; 2] {
        let mut out = [None, None];
        let mut i = 0usize;
        for_each_reachable_successor(
            reachable_indices[pos],
            instruction_successors,
            instr_starts,
            &reachable_indices,
            |succ_pos| {
                if i < 2 {
                    out[i] = Some(succ_pos);
                    i += 1;
                }
            },
        );
        // #1756 Codex review round 12: a self-targeting JMP_IF reports
        // its one true successor twice - collapse it to a single edge
        // here, upholding the precondition `compute_leaders` (and every
        // other leader-compression consumer) documents and trusts.
        dedup_successors(out)
    };

    // #1756 Codex review round 11: see this function's own doc comment
    // ("Leader compression") for the finding this closes.
    let is_leader = compute_leaders(reachable_count, successors_of);

    // #1756 Codex review round 20 (owner decision): `leader_count` is
    // derived by counting `is_leader`'s true bits - a plain iterator
    // `.filter(..).count()`, zero additional allocation - rather than by
    // first materializing `leader_positions` and taking its `.len()`.
    // `is_leader` itself (`reachable_count`-sized, `Vec<bool>`) was
    // already allocated under the EARLIER raw-node check's own
    // authorization (`LEADER_FLAG_LOGICAL_BYTES * reachable_count`, part
    // of `raw_words` above); counting its bits reads that already-
    // authorized memory, allocating nothing new. This is what lets the
    // combined check below run, and potentially reject, BEFORE
    // `leader_positions` - a SEPARATE, `leader_count`-sized `Vec<usize>`
    // allocation this check's own `fixed_bytes` term also charges for -
    // is ever constructed. Closes the last remaining pre-allocation
    // timing hole in this envelope: round 15 charged `leader_positions`
    // honestly because building it BEFORE the check was, at the time,
    // believed unavoidable ("the one structure this check cannot gate
    // ahead of itself by construction") - round 20's finding is that
    // this was never actually true, since `is_leader` already contains
    // everything needed to compute `leader_count` without collecting
    // positions at all.
    let leader_count = is_leader.iter().filter(|&&leader| leader).count();

    // #1756 Codex review round 13 (owner decision), round 16, round 20:
    // the deterministic verifier-resource admission check, BEFORE
    // `leader_positions`, `build_leader_chains`, or `compute_missing_
    // sets` allocates any leader_count-sized state - see `check_
    // analysis_state_budget`'s own doc comment and `VerificationLimits`.
    // A genuinely branch-dense reachable graph (round 13's finding) is
    // not a bug in leader compression - it is the correct, minimal
    // leader set an EXACT MUST-dataflow proof requires - so this is a
    // resource-envelope decision, not a correctness fix. Round 16: now
    // COMBINES the raw-node cost already proven safe above with this
    // phase's own leader/domain-scaled cost, since both remain
    // simultaneously live for the rest of this analysis.
    if let Err(state_words) =
        check_analysis_state_budget(raw_words, leader_count, domain_size, limits.max_state_words)
    {
        return Err(reject_one(
            function,
            VerificationCode::AnalysisStateLimitExceeded,
            0,
            format!(
                "definite-register-assignment analysis needs {} logical verifier-analysis word(s) ({reachable_count} reachable node(s), {leader_count} leader(s) at a {domain_size}-register domain: raw reachable-node representation plus dense lattice payload plus fixed per-leader/stack structural overhead), exceeding the verification state budget of {}",
                if state_words == usize::MAX { "more than usize::MAX".to_string() } else { state_words.to_string() },
                limits.max_state_words
            ),
        ));
    }

    // #1756 Codex review round 20: only NOW, with the combined check
    // above having authorized `leader_count`'s own storage cost, is
    // `leader_positions` actually materialized.
    let leader_positions: Vec<usize> = (0..reachable_count).filter(|&p| is_leader[p]).collect();

    let leader_index_of = |pos: usize| -> usize {
        leader_positions
            .binary_search(&pos)
            .expect("every chain-end successor is a leader by construction of compute_leaders")
    };
    let (chain_killed, chain_targets) = build_leader_chains(
        &leader_positions,
        &is_leader,
        successors_of,
        |pos, visit| {
            for &w in csr_slice(&writes_flat, &writes_offsets, pos) {
                visit(dense(w));
            }
        },
        leader_index_of,
    );
    let compressed_writes_contains =
        |leader_idx: usize, bit: usize| -> bool { chain_killed[leader_idx].contains(bit) };
    let compressed_successors_of =
        |leader_idx: usize| -> [Option<usize>; 2] { chain_targets[leader_idx] };

    let (missing, _event_count, _peak_queue_len) = match compute_missing_sets(
        leader_count,
        domain_size,
        entry_missing,
        compressed_writes_contains,
        compressed_successors_of,
        work_meter,
    ) {
        Ok(result) => result,
        Err(used) => {
            // #1756 Codex review round 14 (owner decision): `used` is the
            // shared `work_meter`'s CUMULATIVE total across every
            // function analyzed so far in this artifact verification
            // call, not just this function's own contribution - see
            // `AnalysisWorkMeter`'s doc comment. Round 19: `used ==
            // usize::MAX` specifically means `AnalysisWorkMeter::charge_
            // one`'s own `checked_add` overflowed - the true attempted
            // usage (`usize::MAX + 1`) has no finite representation, so
            // it is reported with the same "more than usize::MAX"
            // wording the state-budget overflow paths already use,
            // rather than printing the sentinel value as if it were a
            // real, representable work-unit count.
            return Err(reject_one(
                function,
                VerificationCode::AnalysisWorkLimitExceeded,
                0,
                format!(
                    "definite-register-assignment analysis performed {} lattice-propagation work unit(s) across this verification call, exceeding the verification work budget of {}",
                    if used == usize::MAX { "more than usize::MAX".to_string() } else { used.to_string() },
                    limits.max_work_units
                ),
            ));
        }
    };

    // Validate reads against the converged fixed point only - never during
    // an unstable iteration, so the reported diagnostic always describes the
    // actual least/greatest fixed point, not transient worklist state. See
    // `find_first_violation`'s own doc comment for why this can no longer
    // be a simple single ascending scan with an early return: each
    // leader's run is walked in execution order, not position order.
    let violation = find_first_violation(
        &leader_positions,
        &is_leader,
        &missing,
        successors_of,
        |pos, visit| {
            for &r in csr_slice(&reads_flat, &reads_offsets, pos) {
                visit(dense(r));
            }
        },
        |pos, visit| {
            for &w in csr_slice(&writes_flat, &writes_offsets, pos) {
                visit(dense(w));
            }
        },
    );
    if let Some((pos, bit)) = violation {
        let idx = reachable_indices[pos];
        let raw_reg = universe[bit];
        return Err(reject_one(
            function,
            VerificationCode::UndefinedRegisterRead,
            instr_starts[idx],
            format!(
                "register r{raw_reg} is read but not definitely defined on every execution path reaching this instruction"
            ),
        ));
    }

    Ok(())
}

/// Decodes one instruction's operands, advancing `cursor` past its full byte
/// shape (including count/flag-controlled variable-length fields).
///
/// `enforce_canonical_domains` separates two distinct concerns that this
/// function used to conflate:
///
/// - STRUCTURAL SHAPE: opcode recognition, operand byte widths, and
///   presence/count-controlled byte lengths - always applied, regardless of
///   the flag. This is what determines whether a byte range decodes as a
///   complete instruction stream at all.
/// - SEMANTIC ADMISSION: canonical literal value domains (`LOAD_Q`,
///   `LOAD_BOOL`), canonical presence-flag domains (`CALL`,
///   `CLOSURE_CALL`, `RET`), and canonical arity/cardinality domains
///   (`MAKE_TUPLE` arity `>= 2`, `MAKE_RECORD` slot count `>= 1`) - applied
///   only when `enforce_canonical_domains` is `true`. In every one of these
///   cases the count or flag byte itself is read unconditionally and always
///   determines how many further operand bytes follow (there is no
///   width-affecting difference between a canonical and non-canonical
///   value), so disabling this enforcement never changes byte-shape
///   consumption - only whether an out-of-domain value is rejected.
///
/// Every other rejection in this function's match arms is a truncation /
/// missing-bytes error from a failed `read_*` call - i.e. genuinely
/// structural (unknown opcode is rejected by the caller before this
/// function is even entered; every remaining path here fails only on
/// missing/truncated operand bytes).
///
/// Normal verifier admission (the real instruction-stream walk in
/// `verify_function_code`) must enforce both, so it passes `true`. The
/// `#1731` ambiguity probe (`instruction_stream_parses_fully`) must decide
/// only whether an alternative byte-level *framing* exists at all - a
/// non-canonical operand value doesn't make an otherwise complete
/// instruction-shaped reading any less structurally real - so it passes
/// `false`. This keeps both concerns on one shared opcode-shape match
/// rather than duplicating it.
// #1756 Codex review round 21: gained an 8th parameter (`metadata_sink`)
// when `jump_targets`/`string_refs`/`call_argcs` moved from `OperandRefs`
// to the same caller-directed sink pattern round 18 already applied to
// `reads`/`writes` (see `MetadataSink`'s own doc comment) - each parameter
// is independently meaningful (the decode target, the canonical-domain
// policy switch, and the two orthogonal sink roles), and bundling them
// into a struct purely to satisfy this lint would add indirection without
// reducing real complexity.
#[cfg(feature = "std")]
#[allow(clippy::too_many_arguments)]
fn decode_operands(
    function: &str,
    code: &[u8],
    cursor: &mut usize,
    offset: usize,
    opcode: Opcode,
    enforce_canonical_domains: bool,
    sink: &mut dyn OperandSink,
    metadata_sink: &mut dyn MetadataSink,
) -> Result<OperandRefs, RejectReport> {
    let invalid =
        |msg: &str| reject_one(function, VerificationCode::OperandOutOfBounds, offset, msg);
    let mut refs = OperandRefs::default();
    let mut mark_reg = |reg: u16| {
        let reg = reg as usize;
        refs.max_register = Some(refs.max_register.map_or(reg, |current| current.max(reg)));
    };

    match opcode {
        Opcode::LoadQ => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            let literal = read_u8(code, cursor).map_err(|_| invalid("truncated quad literal"))?;
            if enforce_canonical_domains && literal > 3 {
                return Err(invalid("non-canonical quad literal: must be 0..=3"));
            }
        }
        Opcode::LoadBool => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            let literal = read_u8(code, cursor).map_err(|_| invalid("truncated bool literal"))?;
            if enforce_canonical_domains && literal > 1 {
                return Err(invalid("non-canonical bool literal: must be 0 or 1"));
            }
        }
        Opcode::LoadI32 => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            read_i32_le(code, cursor).map_err(|_| invalid("truncated i32 literal"))?;
        }
        Opcode::AddI32 => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            let lhs = read_u16_le(code, cursor).map_err(|_| invalid("truncated lhs register"))?;
            let rhs = read_u16_le(code, cursor).map_err(|_| invalid("truncated rhs register"))?;
            mark_reg(dst);
            mark_reg(lhs);
            mark_reg(rhs);
            sink.read(lhs);
            sink.read(rhs);
            sink.write(dst);
        }
        Opcode::SubI32 | Opcode::MulI32 | Opcode::DivI32 | Opcode::ModI32 => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            let lhs = read_u16_le(code, cursor).map_err(|_| invalid("truncated lhs register"))?;
            let rhs = read_u16_le(code, cursor).map_err(|_| invalid("truncated rhs register"))?;
            mark_reg(dst);
            mark_reg(lhs);
            mark_reg(rhs);
            sink.read(lhs);
            sink.read(rhs);
            sink.write(dst);
        }
        Opcode::LoadU32 => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            read_u32_le(code, cursor).map_err(|_| invalid("truncated u32 literal"))?;
        }
        Opcode::LoadF64 => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            refs.required_capabilities |= CAP_F64_MATH;
            read_f64_le(code, cursor).map_err(|_| invalid("truncated f64 literal"))?;
        }
        Opcode::LoadFx => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            refs.required_capabilities |= CAP_FX_VALUES;
            read_i32_le(code, cursor).map_err(|_| invalid("truncated fx literal"))?;
        }
        Opcode::LoadText => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            refs.required_capabilities |= CAP_TEXT_VALUES;
            let sid = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated text literal string id"))?;
            metadata_sink.string_ref(offset, sid as usize, "text literal");
        }
        Opcode::ConcatText => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            let lhs = read_u16_le(code, cursor).map_err(|_| invalid("truncated lhs register"))?;
            let rhs = read_u16_le(code, cursor).map_err(|_| invalid("truncated rhs register"))?;
            mark_reg(dst);
            mark_reg(lhs);
            mark_reg(rhs);
            sink.read(lhs);
            sink.read(rhs);
            sink.write(dst);
            refs.required_capabilities |= CAP_TEXT_VALUES;
        }
        Opcode::MakeSequence => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            refs.required_capabilities |= CAP_SEQUENCE_VALUES;
            let count = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence arity"))? as usize;
            for _ in 0..count {
                let src = read_u16_le(code, cursor)
                    .map_err(|_| invalid("truncated sequence item register"))?;
                mark_reg(src);
                sink.read(src);
            }
        }
        Opcode::MakeTuple => {
            let dst =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated tuple dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            let count =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated tuple arity"))? as usize;
            if enforce_canonical_domains && count < 2 {
                return Err(invalid("tuple literal arity must be at least 2"));
            }
            for _ in 0..count {
                let src = read_u16_le(code, cursor)
                    .map_err(|_| invalid("truncated tuple item register"))?;
                mark_reg(src);
                sink.read(src);
            }
        }
        Opcode::MakeRecord => {
            let dst =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated record dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            let sid = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated record type string id"))?;
            metadata_sink.string_ref(offset, sid as usize, "record type name");
            let count = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated record slot count"))?
                as usize;
            if enforce_canonical_domains && count == 0 {
                return Err(invalid("record literal must encode at least one slot"));
            }
            for _ in 0..count {
                let src = read_u16_le(code, cursor)
                    .map_err(|_| invalid("truncated record slot register"))?;
                mark_reg(src);
                sink.read(src);
            }
        }
        Opcode::MakeAdt => {
            let dst =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated enum dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            let sid =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated enum type string id"))?;
            metadata_sink.string_ref(offset, sid as usize, "enum type name");
            let variant_sid = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated enum variant string id"))?;
            metadata_sink.string_ref(offset, variant_sid as usize, "enum variant name");
            read_u16_le(code, cursor).map_err(|_| invalid("truncated enum tag"))?;
            let count = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated enum payload count"))?
                as usize;
            for _ in 0..count {
                let src = read_u16_le(code, cursor)
                    .map_err(|_| invalid("truncated enum payload register"))?;
                mark_reg(src);
                sink.read(src);
            }
        }
        Opcode::AdtTag => {
            let dst =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated adt-tag dst register"))?;
            let src =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated adt-tag src register"))?;
            let sid = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated adt-tag type string id"))?;
            mark_reg(dst);
            mark_reg(src);
            sink.read(src);
            sink.write(dst);
            metadata_sink.string_ref(offset, sid as usize, "enum type name");
        }
        Opcode::AdtGet => {
            let dst =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated adt-get dst register"))?;
            let src =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated adt-get src register"))?;
            let sid = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated adt-get type string id"))?;
            read_u16_le(code, cursor).map_err(|_| invalid("truncated adt-get payload index"))?;
            mark_reg(dst);
            mark_reg(src);
            sink.read(src);
            sink.write(dst);
            metadata_sink.string_ref(offset, sid as usize, "enum type name");
        }
        Opcode::RecordGet => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated record-get dst register"))?;
            let src = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated record-get src register"))?;
            let sid = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated record-get type string id"))?;
            read_u16_le(code, cursor).map_err(|_| invalid("truncated record-get slot index"))?;
            mark_reg(dst);
            mark_reg(src);
            sink.read(src);
            sink.write(dst);
            metadata_sink.string_ref(offset, sid as usize, "record type name");
        }
        Opcode::TupleGet => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated tuple-get dst register"))?;
            let src = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated tuple-get src register"))?;
            read_u16_le(code, cursor).map_err(|_| invalid("truncated tuple-get index"))?;
            mark_reg(dst);
            mark_reg(src);
            sink.read(src);
            sink.write(dst);
        }
        Opcode::SequenceGet => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-get dst register"))?;
            let src = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-get src register"))?;
            let index = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-get index register"))?;
            mark_reg(dst);
            mark_reg(src);
            mark_reg(index);
            sink.read(src);
            sink.read(index);
            sink.write(dst);
            refs.required_capabilities |= CAP_SEQUENCE_VALUES;
        }
        Opcode::SequenceLen => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-len dst register"))?;
            let src = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-len src register"))?;
            mark_reg(dst);
            mark_reg(src);
            sink.read(src);
            sink.write(dst);
            refs.required_capabilities |= CAP_SEQUENCE_VALUES;
            refs.required_capabilities |= CAP_SEQUENCE_ITERATION;
        }
        Opcode::SequenceIsEmpty => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-is-empty dst register"))?;
            let src = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-is-empty src register"))?;
            mark_reg(dst);
            mark_reg(src);
            sink.read(src);
            sink.write(dst);
            refs.required_capabilities |= CAP_SEQUENCE_VALUES;
            refs.required_capabilities |= CAP_SEQUENCE_ITERATION;
        }
        Opcode::SequenceContains => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-contains dst register"))?;
            let seq = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-contains seq register"))?;
            let val = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-contains val register"))?;
            mark_reg(dst);
            mark_reg(seq);
            mark_reg(val);
            sink.read(seq);
            sink.read(val);
            sink.write(dst);
            refs.required_capabilities |= CAP_SEQUENCE_VALUES;
            refs.required_capabilities |= CAP_SEQUENCE_ITERATION;
        }
        Opcode::SequencePush => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-push dst register"))?;
            let seq = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-push seq register"))?;
            let val = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-push val register"))?;
            mark_reg(dst);
            mark_reg(seq);
            mark_reg(val);
            sink.read(seq);
            sink.read(val);
            sink.write(dst);
            refs.required_capabilities |= CAP_SEQUENCE_VALUES;
            refs.required_capabilities |= CAP_SEQUENCE_ITERATION;
        }
        Opcode::SequencePrepend => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-prepend dst register"))?;
            let seq = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-prepend seq register"))?;
            let val = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-prepend val register"))?;
            mark_reg(dst);
            mark_reg(seq);
            mark_reg(val);
            sink.read(seq);
            sink.read(val);
            sink.write(dst);
            refs.required_capabilities |= CAP_SEQUENCE_VALUES;
            refs.required_capabilities |= CAP_SEQUENCE_ITERATION;
        }
        Opcode::SequencePop => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-pop dst register"))?;
            let src = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated sequence-pop src register"))?;
            mark_reg(dst);
            mark_reg(src);
            sink.read(src);
            sink.write(dst);
            refs.required_capabilities |= CAP_SEQUENCE_VALUES;
            refs.required_capabilities |= CAP_SEQUENCE_ITERATION;
        }
        Opcode::MapEmpty => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated map-empty dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            refs.required_capabilities |= CAP_MAP_VALUES;
        }
        Opcode::MapContains => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated map-contains dst register"))?;
            let map = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated map-contains map register"))?;
            let key = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated map-contains key register"))?;
            mark_reg(dst);
            mark_reg(map);
            mark_reg(key);
            sink.read(map);
            sink.read(key);
            sink.write(dst);
            refs.required_capabilities |= CAP_MAP_VALUES;
        }
        Opcode::MapGet => {
            let dst =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated map-get dst register"))?;
            let map =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated map-get map register"))?;
            let key =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated map-get key register"))?;
            let default_val = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated map-get default register"))?;
            mark_reg(dst);
            mark_reg(map);
            mark_reg(key);
            mark_reg(default_val);
            sink.read(map);
            sink.read(key);
            // #1756 (FA-07-016): MAP_GET reads `default_val` lazily at
            // runtime, only on a key miss (see #1771). The verifier cannot
            // soundly prove key presence, so this pass conservatively
            // requires `default_val` definitely defined unconditionally -
            // never a value-sensitive "only on the miss path" proof.
            sink.read(default_val);
            sink.write(dst);
            refs.required_capabilities |= CAP_MAP_VALUES;
        }
        Opcode::MapSet => {
            let dst =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated map-set dst register"))?;
            let map =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated map-set map register"))?;
            let key =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated map-set key register"))?;
            let val =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated map-set val register"))?;
            mark_reg(dst);
            mark_reg(map);
            mark_reg(key);
            mark_reg(val);
            sink.read(map);
            sink.read(key);
            sink.read(val);
            sink.write(dst);
            refs.required_capabilities |= CAP_MAP_VALUES;
        }
        Opcode::RngSeed => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated rng-seed dst register"))?;
            let seed = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated rng-seed seed register"))?;
            mark_reg(dst);
            mark_reg(seed);
            sink.read(seed);
            sink.write(dst);
            refs.required_capabilities |= CAP_PRNG;
        }
        Opcode::RngNextI32 => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated rng-next-i32 dst register"))?;
            let lo = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated rng-next-i32 lo register"))?;
            let hi = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated rng-next-i32 hi register"))?;
            mark_reg(dst);
            mark_reg(lo);
            mark_reg(hi);
            sink.read(lo);
            sink.read(hi);
            sink.write(dst);
            refs.required_capabilities |= CAP_PRNG;
        }
        Opcode::MakeClosure => {
            let dst =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated closure dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            refs.required_capabilities |= CAP_CLOSURE_VALUES;
            let sid = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated closure function string id"))?;
            metadata_sink.string_ref(offset, sid as usize, "closure function name");
            let count = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated closure capture arity"))?
                as usize;
            for _ in 0..count {
                let src = read_u16_le(code, cursor)
                    .map_err(|_| invalid("truncated closure capture register"))?;
                mark_reg(src);
                sink.read(src);
            }
        }
        Opcode::ClosureCall => {
            let has_dst_flag =
                read_u8(code, cursor).map_err(|_| invalid("truncated closure-call dst flag"))?;
            if enforce_canonical_domains && has_dst_flag > 1 {
                return Err(invalid(
                    "non-canonical closure-call dst flag: must be 0 or 1",
                ));
            }
            let has_dst = has_dst_flag != 0;
            if has_dst {
                let dst = read_u16_le(code, cursor)
                    .map_err(|_| invalid("truncated closure-call dst register"))?;
                mark_reg(dst);
                sink.write(dst);
            } else {
                // #1756 (FA-07-016): matches the pre-existing `mark_reg`
                // asymmetry below - the encoded dummy dst register bytes are
                // still consumed from the stream, but this opcode form
                // defines no register at all, so the dummy dst is neither a
                // read nor a write.
                let _ = read_u16_le(code, cursor)
                    .map_err(|_| invalid("truncated closure-call dst register"))?;
            }
            let closure = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated closure-call source register"))?;
            let arg = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated closure-call arg register"))?;
            mark_reg(closure);
            mark_reg(arg);
            sink.read(closure);
            sink.read(arg);
            refs.required_capabilities |= CAP_CLOSURE_VALUES;
        }
        Opcode::LoadVar => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            let sid =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated variable string id"))?;
            metadata_sink.string_ref(offset, sid as usize, "variable reference");
        }
        Opcode::StoreVar => {
            let sid =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated variable string id"))?;
            metadata_sink.string_ref(offset, sid as usize, "variable reference");
            let src = read_u16_le(code, cursor).map_err(|_| invalid("truncated src register"))?;
            mark_reg(src);
            // #1756 (FA-07-016): reads the source register into the named
            // local-variable slot. Register dataflow only - the variable
            // store itself is outside this pass's domain (see LOAD_VAR,
            // which writes only its destination register, not a register
            // form of the variable it names).
            sink.read(src);
        }
        Opcode::QNot | Opcode::BoolNot | Opcode::QTruthNot => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            let src = read_u16_le(code, cursor).map_err(|_| invalid("truncated src register"))?;
            mark_reg(dst);
            mark_reg(src);
            sink.read(src);
            sink.write(dst);
        }
        Opcode::QAnd
        | Opcode::QOr
        | Opcode::QImpl
        | Opcode::BoolAnd
        | Opcode::BoolOr
        | Opcode::CmpEq
        | Opcode::QTruthAnd
        | Opcode::QTruthOr
        | Opcode::QTruthImpl
        | Opcode::CmpNe
        | Opcode::CmpI32Lt
        | Opcode::CmpI32Le
        | Opcode::AddF64
        | Opcode::SubF64
        | Opcode::MulF64
        | Opcode::DivF64
        | Opcode::AddFx
        | Opcode::SubFx
        | Opcode::MulFx
        | Opcode::DivFx => {
            let dst = read_u16_le(code, cursor).map_err(|_| invalid("truncated dst register"))?;
            let lhs = read_u16_le(code, cursor).map_err(|_| invalid("truncated lhs register"))?;
            let rhs = read_u16_le(code, cursor).map_err(|_| invalid("truncated rhs register"))?;
            mark_reg(dst);
            mark_reg(lhs);
            mark_reg(rhs);
            sink.read(lhs);
            sink.read(rhs);
            sink.write(dst);
            if matches!(
                opcode,
                Opcode::AddF64 | Opcode::SubF64 | Opcode::MulF64 | Opcode::DivF64
            ) {
                refs.required_capabilities |= CAP_F64_MATH;
            }
            if matches!(
                opcode,
                Opcode::AddFx | Opcode::SubFx | Opcode::MulFx | Opcode::DivFx
            ) {
                refs.required_capabilities |= CAP_FX_MATH;
            }
        }
        Opcode::Jmp => {
            let target = read_u32_le(code, cursor).map_err(|_| invalid("truncated jump target"))?;
            metadata_sink.jump_target(target as usize);
        }
        Opcode::JmpIf => {
            let cond =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated condition register"))?;
            mark_reg(cond);
            sink.read(cond);
            let target = read_u32_le(code, cursor).map_err(|_| invalid("truncated jump target"))?;
            metadata_sink.jump_target(target as usize);
        }
        Opcode::Call => {
            let has_dst_flag =
                read_u8(code, cursor).map_err(|_| invalid("truncated call destination flag"))?;
            if enforce_canonical_domains && has_dst_flag > 1 {
                return Err(invalid(
                    "non-canonical call destination flag: must be 0 or 1",
                ));
            }
            let has_dst = has_dst_flag != 0;
            if has_dst {
                let dst = read_u16_le(code, cursor)
                    .map_err(|_| invalid("truncated call dst register"))?;
                mark_reg(dst);
                sink.write(dst);
            } else {
                // #1756 (FA-07-016): same asymmetry as `ClosureCall` above -
                // the dummy dst register bytes are consumed but define
                // nothing; this form's `CALL` defines no register.
                let _ = read_u16_le(code, cursor)
                    .map_err(|_| invalid("truncated call dst register"))?;
            }
            let sid =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated callee string id"))?;
            metadata_sink.string_ref(offset, sid as usize, "call target");
            let argc = read_u16_le(code, cursor).map_err(|_| invalid("truncated argc"))? as usize;
            // #1773 (FA-09-005): recorded alongside the "call target" string
            // ref above, keyed by the same `offset`, so the cross-function
            // pass can enforce arity against the callee's canonical
            // signature without re-decoding operands.
            metadata_sink.call_argc(offset, argc);
            for _ in 0..argc {
                let arg = read_u16_le(code, cursor)
                    .map_err(|_| invalid("truncated call arg register"))?;
                mark_reg(arg);
                sink.read(arg);
            }
        }
        Opcode::Assert => {
            let cond = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated assert condition register"))?;
            mark_reg(cond);
            sink.read(cond);
        }
        Opcode::GateRead => {
            let dst =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated gate dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            refs.required_capabilities |= CAP_GATE_SURFACE;
            read_u16_le(code, cursor).map_err(|_| invalid("truncated gate device id"))?;
            read_u16_le(code, cursor).map_err(|_| invalid("truncated gate port"))?;
        }
        Opcode::GateWrite => {
            refs.required_capabilities |= CAP_GATE_SURFACE;
            read_u16_le(code, cursor).map_err(|_| invalid("truncated gate device id"))?;
            read_u16_le(code, cursor).map_err(|_| invalid("truncated gate port"))?;
            let src =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated gate src register"))?;
            mark_reg(src);
            sink.read(src);
        }
        Opcode::PulseEmit => {
            refs.required_capabilities |= CAP_GATE_SURFACE;
            let sid =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated signal string id"))?;
            metadata_sink.string_ref(offset, sid as usize, "pulse signal");
        }
        Opcode::StateQuery => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated state-query dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            refs.required_capabilities |= CAP_STATE_QUERY;
            let sid =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated state query key id"))?;
            metadata_sink.string_ref(offset, sid as usize, "state query key");
        }
        Opcode::StateUpdate => {
            refs.required_capabilities |= CAP_STATE_UPDATE;
            let sid =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated state update key id"))?;
            metadata_sink.string_ref(offset, sid as usize, "state update key");
            let src = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated state-update src register"))?;
            mark_reg(src);
            sink.read(src);
        }
        Opcode::EventPost => {
            refs.required_capabilities |= CAP_EVENT_POST;
            let sid =
                read_u16_le(code, cursor).map_err(|_| invalid("truncated event-post signal id"))?;
            metadata_sink.string_ref(offset, sid as usize, "event post signal");
        }
        Opcode::ClockRead => {
            let dst = read_u16_le(code, cursor)
                .map_err(|_| invalid("truncated clock-read dst register"))?;
            mark_reg(dst);
            sink.write(dst);
            refs.required_capabilities |= CAP_CLOCK_READ;
        }
        Opcode::Ret => {
            let has_src = read_u8(code, cursor).map_err(|_| invalid("truncated return flag"))?;
            if enforce_canonical_domains && has_src > 1 {
                return Err(invalid("non-canonical return flag: must be 0 or 1"));
            }
            if has_src != 0 {
                let src = read_u16_le(code, cursor)
                    .map_err(|_| invalid("truncated return src register"))?;
                mark_reg(src);
                sink.read(src);
            }
        }
    }

    Ok(refs)
}

#[cfg(feature = "std")]
fn builtin_call_required_capabilities(name: &str) -> Option<u32> {
    match name {
        "sin" | "cos" | "tan" | "sqrt" | "abs" | "pow" => Some(CAP_F64_MATH),
        "to_text" => Some(CAP_TEXT_VALUES),
        "print" => Some(CAP_STDOUT),
        "args_read" => Some(CAP_ARGS_READ),
        "stdin_read_text" => Some(CAP_STDIN_READ_TEXT),
        "stdout_write" => Some(CAP_STDOUT_WRITE),
        "stderr_write" => Some(CAP_STDERR_WRITE),
        "path_inspect" => Some(CAP_PATH_INSPECT),
        "fs_read_text" => Some(CAP_FS_READ),
        "fs_write_text" => Some(CAP_FS_WRITE),
        "time_duration_ms" => Some(CAP_TIME_DURATION),
        _ => None,
    }
}

/// #1756 Codex review round 21: no longer holds `jump_targets`/`string_
/// refs`/`call_argcs` - those are now emitted through `MetadataSink`
/// (below), exactly like `reads`/`writes` moved to `OperandSink` in
/// round 18, for the identical reason: the reachable-only re-decode pass
/// never reads them, so building and dropping fresh `Vec`s for them on
/// every re-decoded instruction was pure waste (Codex's round-21 P1
/// finding). What remains here (`max_register`, `required_capabilities`)
/// is cheap regardless of mode - neither is heap-backed - so both stay
/// direct fields, always populated, never routed through a sink.
#[cfg(feature = "std")]
#[derive(Default)]
struct OperandRefs {
    max_register: Option<usize>,
    required_capabilities: u32,
}

/// #1756 Codex review round 18 (owner decision): every register an
/// instruction reads or writes, in the order fixed by `decode_operands`'s
/// exhaustive `Opcode` match (the same match that already classifies every
/// operand for the register-range quota), is now emitted through this
/// sink rather than collected into an `OperandRefs`-owned `Vec<u16>` - the
/// main structural decode pass (`verify_function_code`'s first walk)
/// never reads a decoded instruction's reads/writes at all (it only
/// consumes `jump_targets`/`string_refs`/`call_argcs`/`max_register`/
/// `required_capabilities`), so building and immediately dropping a
/// `Vec<u16>` pair for EVERY structurally decoded instruction - reachable
/// or not - cost one to two heap allocation/deallocation cycles per
/// instruction for no reason: an artifact shaped as a reachable entry
/// `RET` followed by millions of structurally valid but unreachable
/// `LOAD_BOOL` instructions paid millions of allocator cycles that
/// `max_state_words`/`max_work_units` (both #1756-analysis-only budgets,
/// scoped to REACHABLE state) could never see or bound, since this cost
/// happens earlier, in the shared structural walk every function pays
/// regardless of signature. `read`/`write` are independent - a register
/// can appear in both (e.g. an opcode reading and writing the same
/// numeric register would report it to both) - never merged, so the
/// definite-assignment pass can still validate reads against `IN[n]`
/// before folding writes into `OUT[n]`.
///
/// Two implementations: `NoCollect` (below) does nothing at all - the
/// structural pass's sink, zero heap allocation, zero storage, a plain
/// virtual-call-and-return per register mention. `FlatSink` (in
/// `dataflow_domain_accounting`) appends straight into the CALLER's own
/// `reads_flat`/`writes_flat` CSR buffers - the reachable-only re-decode's
/// sink, avoiding even a temporary per-instruction `Vec<u16>` that would
/// otherwise be built and then copied into the flat arrays and dropped.
/// `&mut dyn OperandSink` (dynamic dispatch, not a generic parameter) lets
/// ONE closure type (`reads_writes_of` in `prove_definite_register_
/// assignment`) forward whichever sink its own caller supplies, without
/// needing to be generic itself - the indirection costs one vtable call
/// per register mention, never a heap allocation, so it does not
/// reintroduce the cost this round removes.
#[cfg(feature = "std")]
trait OperandSink {
    fn read(&mut self, reg: u16);
    fn write(&mut self, reg: u16);
}

/// The main structural decode pass's sink (`verify_function_code`'s first
/// walk over every instruction, reachable or not) - this pass never reads
/// a decoded instruction's register reads/writes, only `jump_targets`/
/// `string_refs`/`call_argcs`/`max_register`/`required_capabilities`, so
/// recording reads/writes at all would be pure waste. Both methods are
/// empty; with `&mut dyn OperandSink` this is a vtable call to a no-op,
/// never a heap allocation - the property this round exists to establish.
#[cfg(feature = "std")]
struct NoCollect;

#[cfg(feature = "std")]
impl OperandSink for NoCollect {
    fn read(&mut self, _reg: u16) {}
    fn write(&mut self, _reg: u16) {}
}

/// #1756 Codex review round 21 (owner decision): the SAME sink pattern
/// `OperandSink` established in round 18, extended to `decode_operands`'s
/// three remaining heap-backed fields - `jump_targets`, `string_refs`,
/// `call_argcs` - which round 18 left untouched (it addressed only
/// `reads`/`writes`). The reachable-only re-decode pass (`reads_writes_
/// of` in `prove_definite_register_assignment`) never reads ANY of these
/// three either - it discards the whole returned `OperandRefs` via
/// `.expect(...)`, keeping only what its own `FlatSink` captured - so a
/// zero-argument, no-destination `CALL` (which still populates `string_
/// refs` with its callee name and `call_argcs` with `0`) paid a fresh
/// heap allocation for each, per reachable instruction, for metadata
/// nobody ever reads (Codex's round-21 P1 finding: roughly two allocator
/// cycles per reachable no-op `CALL`, unbounded by `max_state_words`/
/// `max_work_units` for the identical reason round 18's `reads`/`writes`
/// gap was: this cost happens in the shared structural re-decode, not in
/// anything either budget scopes).
///
/// `NoCollect` (same zero-field marker type as `OperandSink`'s no-op
/// impl) also implements this trait with empty bodies - one universal
/// "collect nothing" sink for BOTH operand and metadata suppression, so
/// callers that need neither (like the reachable re-decode) pass exactly
/// one shared no-op value for both parameters. `MetadataCollector`
/// (below) is the structural pass's real collector - it appends directly
/// into the CALLER's own outer accumulator `Vec`s (mirroring `FlatSink`'s
/// own "write straight into the final destination, never a throwaway
/// per-call `Vec`" design), rather than collecting into `OperandRefs`
/// fields that the caller would then have to `.extend()` into its own
/// accumulators and drop - eliminating that whole intermediate round-trip
/// for the structural pass too, not just suppressing it for re-decode.
#[cfg(feature = "std")]
trait MetadataSink {
    fn jump_target(&mut self, target: usize);
    fn string_ref(&mut self, offset: usize, sid: usize, usage: &'static str);
    fn call_argc(&mut self, offset: usize, argc: usize);
}

#[cfg(feature = "std")]
impl MetadataSink for NoCollect {
    fn jump_target(&mut self, _target: usize) {}
    fn string_ref(&mut self, _offset: usize, _sid: usize, _usage: &'static str) {}
    fn call_argc(&mut self, _offset: usize, _argc: usize) {}
}

/// #1756 Codex review round 21: the main structural decode pass's real
/// metadata sink - appends directly into the CALLER's (`verify_function_
/// code`'s) own outer `jump_targets`/`string_refs`/`call_argcs`
/// accumulators, borrowed for the duration of one `decode_operands` call.
#[cfg(feature = "std")]
struct MetadataCollector<'a> {
    jump_targets: &'a mut Vec<usize>,
    string_refs: &'a mut Vec<(usize, usize, &'static str)>,
    call_argcs: &'a mut Vec<(usize, usize)>,
}

#[cfg(feature = "std")]
impl MetadataSink for MetadataCollector<'_> {
    fn jump_target(&mut self, target: usize) {
        self.jump_targets.push(target);
    }
    fn string_ref(&mut self, offset: usize, sid: usize, usage: &'static str) {
        self.string_refs.push((offset, sid, usage));
    }
    fn call_argc(&mut self, offset: usize, argc: usize) {
        self.call_argcs.push((offset, argc));
    }
}

#[cfg(feature = "std")]
struct PendingVerifiedFunction {
    verified: VerifiedFunction,
    // (offset, callee name, allows_builtin, argc for real Call sites)
    call_targets: Vec<(usize, String, bool, Option<usize>)>,
    has_ownership_section: bool,
}

#[cfg(feature = "std")]
fn reject_one(
    function: &str,
    code: VerificationCode,
    offset: usize,
    message: impl Into<String>,
) -> RejectReport {
    RejectReport {
        diagnostics: vec![diag(
            code,
            Some(function.to_string()),
            Some(offset),
            message,
        )],
    }
}

#[cfg(feature = "std")]
fn diag(
    code: VerificationCode,
    function: Option<String>,
    offset: Option<usize>,
    message: impl Into<String>,
) -> VerificationDiagnostic {
    VerificationDiagnostic {
        code,
        function,
        offset,
        message: message.into(),
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use sm_format::semcode_format::{
        read_u16_le, read_u32_le, CallableValueFamily, MAGIC0, MAGIC10, MAGIC11, MAGIC18, MAGIC20,
        MAGIC21, MAGIC3, MAGIC4, MAGIC5, MAGIC6, MAGIC7, OWNERSHIP_SECTION_TAG,
    };
    use sm_ir::{
        compile_program_to_semcode, compile_program_to_semcode_with_options_debug,
        emit_ir_to_semcode, CompileProfile, IrFunction, IrInstr, OptLevel,
    };

    fn emit_test_function(instrs: Vec<IrInstr>) -> Vec<u8> {
        emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs,
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit test function")
    }

    /// #1756 Codex review round 18: like `emit_test_function`, but with a
    /// second, callable "helper" function (`RET`, no operands) - needed to
    /// build a real `CALL`/`MAKE_CLOSURE` instruction that references a
    /// resolvable callee name.
    fn emit_test_function_with_helper(main_instrs: Vec<IrInstr>) -> Vec<u8> {
        emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "helper".to_string(),
                    instrs: vec![IrInstr::Ret { src: None }],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: main_instrs,
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit test function with helper")
    }

    /// #1756 Codex review round 18: a test-only `OperandSink` that
    /// collects every emitted register into its own `Vec<u16>`, exactly
    /// matching the shape `decode_operands` used to return directly (via
    /// `OperandRefs.reads`/`.writes`) before this round's refactor - used
    /// ONLY to assert the sink-based path still reports the identical
    /// read/write identities `decode_operands`'s match arms always
    /// produced, never in the structural or reachable-dataflow production
    /// paths themselves.
    struct RecordingSink {
        reads: Vec<u16>,
        writes: Vec<u16>,
    }

    impl OperandSink for RecordingSink {
        fn read(&mut self, reg: u16) {
            self.reads.push(reg);
        }
        fn write(&mut self, reg: u16) {
            self.writes.push(reg);
        }
    }

    /// #1756 Codex review round 18: decodes real SemCode bytes' FIRST
    /// structural instruction directly via `decode_operands` and a
    /// `RecordingSink`, returning `(reads, writes)` exactly as `OperandRefs`
    /// used to expose them - direct design-test evidence that the round-18
    /// sink refactor produces IDENTICAL register identities to the pre-
    /// refactor `Vec`-returning shape, for opcode categories ranging from
    /// zero-operand to genuinely variadic.
    fn decode_first_instruction_operands(
        bytes: &[u8],
        function_name: &str,
    ) -> (Vec<u16>, Vec<u16>) {
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(bytes).expect("decode semcode");
        let env = functions
            .iter()
            .find(|f| f.name == function_name)
            .unwrap_or_else(|| panic!("function '{function_name}' not found"));
        let code = env.code_slice;
        let mut cursor = env.instr_start_offset;
        let opcode_byte = read_u8(code, &mut cursor).expect("read opcode byte");
        let opcode = Opcode::from_byte(opcode_byte).expect("valid opcode");
        let mut sink = RecordingSink {
            reads: Vec::new(),
            writes: Vec::new(),
        };
        decode_operands(
            function_name,
            code,
            &mut cursor,
            0,
            opcode,
            true,
            &mut sink,
            &mut NoCollect,
        )
        .expect("decode operands");
        (sink.reads, sink.writes)
    }

    /// #1756 Codex review round 21: a test-only `MetadataSink` that
    /// collects every emitted item into its own `Vec` - the metadata
    /// analogue of `RecordingSink` (round 18), used ONLY to assert the
    /// sink-based path still reports the identical structural metadata
    /// `decode_operands`'s match arms always produced, for opcodes that
    /// populate `jump_targets`/`string_refs`/`call_argcs`.
    struct RecordingMetadataSink {
        jump_targets: Vec<usize>,
        string_refs: Vec<(usize, usize, &'static str)>,
        call_argcs: Vec<(usize, usize)>,
    }

    impl MetadataSink for RecordingMetadataSink {
        fn jump_target(&mut self, target: usize) {
            self.jump_targets.push(target);
        }
        fn string_ref(&mut self, offset: usize, sid: usize, usage: &'static str) {
            self.string_refs.push((offset, sid, usage));
        }
        fn call_argc(&mut self, offset: usize, argc: usize) {
            self.call_argcs.push((offset, argc));
        }
    }

    /// #1756 Codex review round 21: decodes real SemCode bytes' FIRST
    /// structural instruction directly via `decode_operands` and a
    /// `RecordingMetadataSink`, returning `(jump_targets, string_refs,
    /// call_argcs)` - direct design-test evidence that the round-21
    /// metadata-sink refactor produces IDENTICAL structural metadata to
    /// the pre-refactor `OperandRefs`-field shape.
    #[allow(clippy::type_complexity)]
    fn decode_first_instruction_metadata(
        bytes: &[u8],
        function_name: &str,
    ) -> (
        Vec<usize>,
        Vec<(usize, usize, &'static str)>,
        Vec<(usize, usize)>,
    ) {
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(bytes).expect("decode semcode");
        let env = functions
            .iter()
            .find(|f| f.name == function_name)
            .unwrap_or_else(|| panic!("function '{function_name}' not found"));
        let code = env.code_slice;
        let mut cursor = env.instr_start_offset;
        let opcode_byte = read_u8(code, &mut cursor).expect("read opcode byte");
        let opcode = Opcode::from_byte(opcode_byte).expect("valid opcode");
        let mut metadata_sink = RecordingMetadataSink {
            jump_targets: Vec::new(),
            string_refs: Vec::new(),
            call_argcs: Vec::new(),
        };
        decode_operands(
            function_name,
            code,
            &mut cursor,
            0,
            opcode,
            true,
            &mut NoCollect,
            &mut metadata_sink,
        )
        .expect("decode operands");
        (
            metadata_sink.jump_targets,
            metadata_sink.string_refs,
            metadata_sink.call_argcs,
        )
    }

    /// #1756 Codex review round 21 (owner decision): "prove structural
    /// decoding still sees identical metadata" - the metadata analogue
    /// of round 18's read/write identity test, covering every opcode
    /// category that populates `jump_targets`/`string_refs`/`call_argcs`:
    /// `JMP` (one jump target, no string/argc metadata), variadic `CALL`
    /// (one "call target" string ref AND one argc entry together - the
    /// exact shape Codex's P1 finding cites), and `MAKE_ADT` (TWO
    /// distinct string refs from a single instruction - "enum type name"
    /// and "enum variant name" - proving multi-metadata opcodes are not
    /// truncated to one entry).
    #[test]
    fn c1756_decode_operands_metadata_sink_matches_pre_refactor_identities_across_opcode_shapes() {
        // JMP: one jump target, no string_refs/call_argcs at all.
        let bytes = emit_test_function(vec![
            IrInstr::Jmp {
                label: "target".to_string(),
            },
            IrInstr::Label {
                name: "target".to_string(),
            },
            IrInstr::Ret { src: None },
        ]);
        let (jump_targets, string_refs, call_argcs) =
            decode_first_instruction_metadata(&bytes, "main");
        assert_eq!(jump_targets.len(), 1, "JMP: exactly one jump target");
        assert!(string_refs.is_empty(), "JMP: no string refs");
        assert!(call_argcs.is_empty(), "JMP: no call argcs");

        // Variadic CALL with 3 arguments: exactly one "call target"
        // string ref AND one call_argc entry, together, from one
        // instruction - the exact shape Codex's round-21 P1 finding
        // cites (a no-destination, zero-argument CALL is the same shape
        // with argc = 0).
        let bytes = emit_test_function_with_helper(vec![IrInstr::Call {
            dst: None,
            name: "helper".to_string(),
            args: vec![1, 2, 3],
        }]);
        let (jump_targets, string_refs, call_argcs) =
            decode_first_instruction_metadata(&bytes, "main");
        assert!(jump_targets.is_empty(), "CALL: no jump targets");
        assert_eq!(
            string_refs
                .iter()
                .map(|(_, _, usage)| *usage)
                .collect::<Vec<_>>(),
            vec!["call target"],
            "CALL: exactly one call-target string ref"
        );
        assert_eq!(
            call_argcs.iter().map(|(_, argc)| *argc).collect::<Vec<_>>(),
            vec![3],
            "CALL: exactly one call_argc entry with the true argument count"
        );

        // MAKE_ADT: TWO distinct string refs from ONE instruction (enum
        // type name, enum variant name) - proves a multi-metadata opcode
        // is not truncated to a single entry.
        let bytes = emit_test_function(vec![IrInstr::MakeAdt {
            dst: 0,
            adt_name: "MyEnum".to_string(),
            variant_name: "Variant".to_string(),
            tag: 0,
            items: vec![],
        }]);
        let (_, string_refs, _) = decode_first_instruction_metadata(&bytes, "main");
        assert_eq!(
            string_refs
                .iter()
                .map(|(_, _, usage)| *usage)
                .collect::<Vec<_>>(),
            vec!["enum type name", "enum variant name"],
            "MAKE_ADT: both string refs present, in order, neither dropped"
        );
    }

    /// #1756 Codex review round 18 (owner decision): "direct design test" -
    /// proves the sink-based `decode_operands` produces the exact same
    /// read/write register identities, in the exact same order, as the
    /// pre-round-18 `Vec`-returning shape did, across four representative
    /// opcode categories: fixed-arity (`ADD_I32`), zero-operand (`RET`),
    /// and both genuinely VARIADIC forms (`CALL`'s argument list, `MAKE_
    /// CLOSURE`'s capture list) - the variadic cases are the ones most at
    /// risk of truncation or loss from an allocation-avoidance refactor,
    /// since they are exactly why a small fixed-capacity inline buffer was
    /// rejected as a design (see `OperandSink`'s own doc comment).
    #[test]
    fn c1756_decode_operands_sink_matches_pre_refactor_identities_across_opcode_shapes() {
        // Fixed-arity: ADD_I32 dst=2, lhs=10, rhs=11.
        let bytes = emit_test_function(vec![IrInstr::AddI32 {
            dst: 2,
            lhs: 10,
            rhs: 11,
        }]);
        let (reads, writes) = decode_first_instruction_operands(&bytes, "main");
        assert_eq!(reads, vec![10, 11], "fixed-arity: reads in lhs, rhs order");
        assert_eq!(writes, vec![2], "fixed-arity: writes dst only");

        // Zero-operand: RET with no source register at all.
        let bytes = emit_test_function(vec![IrInstr::Ret { src: None }]);
        let (reads, writes) = decode_first_instruction_operands(&bytes, "main");
        assert_eq!(reads, Vec::<u16>::new(), "zero-operand: no reads");
        assert_eq!(writes, Vec::<u16>::new(), "zero-operand: no writes");

        // Variadic CALL: 5 argument registers, one destination register.
        let bytes = emit_test_function_with_helper(vec![IrInstr::Call {
            dst: Some(9),
            name: "helper".to_string(),
            args: vec![1, 2, 3, 4, 5],
        }]);
        let (reads, writes) = decode_first_instruction_operands(&bytes, "main");
        assert_eq!(
            reads,
            vec![1, 2, 3, 4, 5],
            "variadic CALL: every argument register, in order, none truncated"
        );
        assert_eq!(writes, vec![9], "variadic CALL: the dst register");

        // Variadic MAKE_CLOSURE: 5 capture registers, one destination
        // register.
        let bytes = emit_test_function_with_helper(vec![IrInstr::MakeClosure {
            dst: 0,
            name: "helper".to_string(),
            captures: vec![1, 2, 3, 4, 5],
        }]);
        let (reads, writes) = decode_first_instruction_operands(&bytes, "main");
        assert_eq!(
            reads,
            vec![1, 2, 3, 4, 5],
            "variadic MAKE_CLOSURE: every capture register, in order, none truncated"
        );
        assert_eq!(writes, vec![0], "variadic MAKE_CLOSURE: the dst register");
    }

    // #1756 (FA-07-016): like `emit_test_function`, but with a caller-chosen
    // canonical SIG0 parameter list, for regressions that need a non-empty
    // `ENTRY_DEFS`.
    fn emit_test_function_with_params(
        params: Vec<CallableValueFamily>,
        instrs: Vec<IrInstr>,
    ) -> Vec<u8> {
        emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs,
                ownership_events: Vec::new(),
                params,
            }],
            false,
        )
        .expect("emit test function")
    }

    #[test]
    fn verifier_rejects_reachable_ordinary_instruction_fallthrough() {
        let bytes = emit_test_function(vec![IrInstr::LoadBool { dst: 0, val: true }]);
        let report =
            verify_semcode(&bytes).expect_err("reachable LOAD_BOOL followed by EOF must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::ReachableFunctionFallthrough
        );
        assert_eq!(report.diagnostics[0].offset, Some(4));
    }

    #[test]
    fn verifier_rejects_empty_instruction_stream() {
        let bytes = emit_test_function(Vec::new());
        let report =
            verify_semcode(&bytes).expect_err("empty executable instruction stream must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::ReachableFunctionFallthrough
        );
        assert_eq!(report.diagnostics[0].offset, Some(0));
    }

    #[test]
    fn verifier_rejects_conditional_branch_with_eof_fallthrough() {
        let bytes = emit_test_function(vec![
            IrInstr::LoadBool { dst: 0, val: true },
            IrInstr::Label {
                name: "branch".to_string(),
            },
            IrInstr::JmpIf {
                cond: 0,
                label: "branch".to_string(),
            },
        ]);
        let report = verify_semcode(&bytes)
            .expect_err("conditional branch with a reachable EOF successor must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::ReachableFunctionFallthrough
        );
    }

    #[test]
    fn verifier_accepts_function_ending_in_ret() {
        let bytes = emit_test_function(vec![IrInstr::Ret { src: None }]);
        verify_semcode(&bytes).expect("RET is a terminal instruction");
    }

    #[test]
    fn verifier_accepts_unconditional_jump_to_valid_return_boundary() {
        let bytes = emit_test_function(vec![
            IrInstr::Jmp {
                label: "return".to_string(),
            },
            IrInstr::LoadBool { dst: 0, val: true },
            IrInstr::Label {
                name: "return".to_string(),
            },
            IrInstr::Ret { src: None },
        ]);
        verify_semcode(&bytes).expect("JMP has only its valid explicit target as successor");
    }

    #[test]
    fn verifier_accepts_structurally_closed_infinite_loop() {
        let bytes = emit_test_function(vec![
            IrInstr::Label {
                name: "loop".to_string(),
            },
            IrInstr::Jmp {
                label: "loop".to_string(),
            },
            IrInstr::LoadBool { dst: 0, val: true },
        ]);
        verify_semcode(&bytes)
            .expect("closed reachable control flow may leave trailing fallthrough unreachable");
    }

    #[test]
    fn verifier_preserves_call_and_closure_call_fallthrough() {
        let direct_call = emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "helper".to_string(),
                    instrs: vec![IrInstr::Ret { src: None }],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: vec![
                        IrInstr::Call {
                            dst: None,
                            name: "helper".to_string(),
                            args: Vec::new(),
                        },
                        IrInstr::Ret { src: None },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit direct call");
        verify_semcode(&direct_call).expect("CALL returns to the decoded next instruction");

        let closure_call = emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "helper".to_string(),
                    instrs: vec![IrInstr::Ret { src: None }],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: vec![
                        IrInstr::MakeClosure {
                            dst: 0,
                            name: "helper".to_string(),
                            captures: Vec::new(),
                        },
                        IrInstr::LoadBool { dst: 1, val: true },
                        IrInstr::ClosureCall {
                            dst: None,
                            closure: 0,
                            arg: 1,
                        },
                        IrInstr::Ret { src: None },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit closure call");
        verify_semcode(&closure_call)
            .expect("CLOSURE_CALL returns to the decoded next instruction");
    }

    #[test]
    fn verifier_accepts_valid_semcode() {
        let bytes = compile_program_to_semcode("fn main() { return; }").expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_fx_semcode() {
        let src = r#"
            fn id(x: fx) -> fx { return x; }
            fn main() {
                let x: fx = 1.25;
                let y: fx = id(-2.0);
                if x == x { return; } else { return; }
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        // #1773 (FA-09-005): SEMCOD19/rev20 is now the floor for every
        // compiled artifact - see the analogous sm-ir comment.
        assert_eq!(verified.header.rev, 20);
    }

    #[test]
    fn verifier_accepts_cli_o0_f64_arithmetic_storevar_layout() {
        let src = r#"
            fn main() {
                let y: f64 = 1.0 + 2.0;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode_with_options_debug(
            src,
            CompileProfile::Auto,
            OptLevel::O0,
            false,
        )
        .expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.header.rev, 20);
    }

    #[test]
    fn verifier_accepts_builtin_f64_call_targets() {
        let src = r#"
            fn main() {
                let y: f64 = sqrt(16.0);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode_with_options_debug(
            src,
            CompileProfile::Auto,
            OptLevel::O0,
            false,
        )
        .expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.header.rev, 20);
    }

    #[test]
    fn verifier_accepts_assert_opcode() {
        let src = r#"
            fn main() {
                assert(true);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_state_query_semcode() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::StateQuery {
                        dst: 0,
                        key: "decision.mode".to_string(),
                    },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.header.rev, 20);
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_state_update_semcode() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadBool { dst: 0, val: true },
                    IrInstr::StateUpdate {
                        key: "decision.mode".to_string(),
                        src: 0,
                    },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.header.rev, 20);
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_event_post_semcode() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::EventPost {
                        signal: "alert.raised".to_string(),
                    },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.header.rev, 20);
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_clock_read_semcode() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![IrInstr::ClockRead { dst: 0 }, IrInstr::Ret { src: None }],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.header.rev, 20);
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_text_semcode() {
        let src = r#"
            fn main() {
                let left: text = "alpha";
                let right: text = "alpha";
                assert(left == right);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.header.rev, 20);
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_u32_numeric_literal_semcode() {
        let src = r#"
            fn main() {
                let left: u32 = 1_000u32;
                let right: u32 = 0x3e8u32;
                assert(left == right);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_stage1_record_make_record_semcode() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext { quality: 0.75, camera: T };
                let _ = ctx;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_stage1_record_get_semcode() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext { quality: 0.75, camera: T };
                let seen: quad = ctx.camera;
                assert(seen == T);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_record_pass_return_and_safe_equality_semcode() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn echo(ctx: DecisionContext) -> DecisionContext {
                return ctx;
            }

            fn main() {
                let left: DecisionContext = DecisionContext { quality: 0.75, camera: T };
                let right: DecisionContext = echo(left);
                assert(right == right);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 2);
    }

    #[test]
    fn verifier_accepts_record_access_policy_scenario() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                badge: quad,
                override_state: quad,
                tamper: quad,
                quality: f64,
            }

            fn allow(ctx: DecisionContext) -> quad {
                if ctx.tamper == T || ctx.tamper == S {
                    return S;
                }
                if ctx.override_state == T {
                    return T;
                }
                if ctx.camera == T && ctx.badge == T {
                    return T;
                }
                return N;
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext {
                    quality: 0.50,
                    tamper: F,
                    override_state: N,
                    badge: T,
                    camera: T,
                };
                let decision: quad = allow(ctx);
                assert(decision == T);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 2);
    }

    #[test]
    fn verifier_accepts_record_runtime_config_scenario() {
        let src = r#"
            record RuntimeConfig {
                max_steps: u32,
                debug_mode: bool,
                fallback_state: quad,
            }

            fn fallback(cfg: RuntimeConfig) -> quad {
                if cfg.debug_mode == true {
                    return cfg.fallback_state;
                }
                return N;
            }

            fn main() {
                let cfg: RuntimeConfig = RuntimeConfig {
                    fallback_state: S,
                    debug_mode: true,
                    max_steps: 16u32,
                };
                let state: quad = fallback(cfg);
                assert(state == S);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 2);
    }

    #[test]
    fn verifier_accepts_record_destructuring_bind_semcode() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn main() {
                let DecisionContext { camera: seen_camera, quality: _ } =
                    DecisionContext { quality: 0.75, camera: T };
                assert(seen_camera == T);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_record_let_else_semcode() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn main() {
                let DecisionContext { camera: T, quality: score } =
                    DecisionContext { quality: 0.75, camera: T } else return;
                assert(score == 0.75);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_record_copy_with_semcode() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext { camera: T, quality: 0.75 };
                let patched: DecisionContext = ctx with { quality: 1.0 };
                assert(patched.camera == T);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_record_stage2_ergonomics_scenario() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                override_state: quad,
                quality: f64,
            }

            fn main() {
                let camera: quad = T;
                let override_state: quad = N;
                let quality: f64 = 0.75;
                let ctx: DecisionContext = DecisionContext { camera, override_state, quality };
                let patched: DecisionContext = ctx with { camera };
                let DecisionContext { camera: T, override_state, quality } =
                    patched else return;
                assert(camera == T);
                assert(override_state == N);
                assert(quality == 0.75);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_ownership_semcode() {
        let bytes = ownership_semcode_bytes();
        let verified = verify_semcode(&bytes).expect("verify");
        // #1726 Checkpoint D2a: this program's Borrow event now always
        // carries a resolved ActivationSiteId, promoting it to SEMCOD20/rev21
        // (was rev20/SIG0's floor before D2a).
        assert_eq!(verified.header.rev, 21);
        assert_eq!(verified.functions.len(), 2);
    }

    #[test]
    fn verifier_accepts_record_field_borrow_ownership_semcode() {
        let bytes = record_field_borrow_semcode_bytes();
        assert_eq!(&bytes[..MAGIC20.len()], &MAGIC20);
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.header.rev, 21);
        assert_eq!(verified.functions.len(), 1);
    }

    // #1726 Checkpoint D2b, item 11 positive tests. Items 1/2 (plain Tuple
    // and Record Borrow StoreVarSite) are already exercised by
    // `verifier_accepts_ownership_semcode` /
    // `verifier_accepts_record_field_borrow_ownership_semcode` above; items
    // "let-else tuple", "let-else record", and "introduction+reassignment
    // (anchor still at introduction)" are already exercised end-to-end
    // (compile -> `verify_semcode_token` -> `sm_vm::run_verified_entry_semcode`)
    // by `tests/borrow_target_identity_proof.rs`'s
    // `frozen_borrow_targets_do_not_uniquely_identify_store_sites` - not
    // duplicated here. The remaining items (mixed StoreVarSite+FrameEntry,
    // shadowed distinct anchors, rev20 legacy unchanged) are new below.

    #[test]
    fn checkpoint_d2b_mixed_store_var_site_and_frame_entry_admitted() {
        let src = r#"
            fn main() {
                let opt: Option(i32) = Option::Some(1);
                let result: i32 = match opt {
                    Option::Some(ref value) => { value }
                    Option::None => { 0 }
                };
                let pair: (i32, i32) = (1, 2);
                let (ref left, _): (i32, i32) = pair;
                let _ = result;
                let _ = left;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        // #1718: this program's `Option::Some(ref value)` arm is a real
        // `Borrow(AdtPayload)` event (`Option` is ADT-represented), which
        // now requires `CAP_OWNERSHIP_ADT_BORROW_PATHS` and promotes one
        // revision past `HEADER_V20`/rev21 (this test's original D2b
        // subject), to `HEADER_V21`/rev22 - purely additive, same OWN0
        // grammar. Before #1718, ADT Borrow had no dedicated capability, so
        // this program only ever reached `MAGIC20`/rev21.
        assert_eq!(&bytes[..MAGIC21.len()], &MAGIC21);
        let verified = verify_semcode(&bytes)
            .expect("D2b must admit a mixed FrameEntry+StoreVarSite artifact");
        assert_eq!(verified.header.rev, 22);
    }

    #[test]
    fn checkpoint_d2b_shadowed_bindings_distinct_anchors_admitted() {
        let src = r#"
            fn main() {
                let p1: (i32, i32) = (1, 2);
                let p2: (i32, i32) = (3, 4);
                if true {
                    let (ref x, _): (i32, i32) = p1;
                    let _ = x;
                }
                if true {
                    let (ref x, _): (i32, i32) = p2;
                    let _ = x;
                }
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        verify_semcode(&bytes)
            .expect("D2b must admit distinct anchors for shadowed same-spelling bindings");
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = decoded.iter().find(|f| f.name == "main").expect("main");
        assert_eq!(main.borrowed_paths.len(), 2);
        let anchors: Vec<u32> = main
            .borrowed_paths
            .iter()
            .map(|p| match p.activation {
                Some(sm_format::semcode_decode::DecodedBorrowActivation::StoreVarSite(a)) => a,
                other => panic!("expected StoreVarSite, got {other:?}"),
            })
            .collect();
        assert_ne!(
            anchors[0], anchors[1],
            "the verifier's own decoded view must show two distinct anchors, not a shared one"
        );
    }

    // #1718 update: this test's original claim - a FrameEntry-only ADT
    // Borrow (no StoreVarSite anchor anywhere) never gets promoted to the
    // rev21 anchor grammar - is no longer true in general, and cannot be:
    // #1718 requires `CAP_OWNERSHIP_ADT_BORROW_PATHS` for ANY `AdtPayload`
    // Borrow event regardless of activation mode, and that capability first
    // exists at `HEADER_V21`/rev22, which is unconditionally >=
    // `SEMCODE_OWNERSHIP_ANCHOR_MIN_REVISION` (21). What the original D2b
    // checkpoint actually cared about - that the anchor *mechanism itself*
    // (`has_v20_ownership_execution_anchor`) is not what's driving
    // promotion here, since there is no StoreVarSite anywhere - remains true
    // and is what this renamed test now asserts directly: the artifact is
    // admitted, promoted by the *ADT-Borrow* capability requirement, and its
    // Borrow event still faithfully encodes `FrameEntry`, not a spuriously
    // invented anchor.
    #[test]
    fn checkpoint_1718_frame_entry_only_adt_borrow_admitted_via_adt_capability_not_anchor() {
        let bytes = adt_payload_ownership_semcode_bytes();
        let verified =
            verify_semcode(&bytes).expect("FrameEntry-only ADT Borrow artifact must admit");
        assert_eq!(
            verified.header.rev,
            sm_format::semcode_format::SEMCODE_ADT_BORROW_OWNERSHIP_MIN_REVISION,
            "promotion here must land exactly on the ADT-Borrow capability floor, not higher"
        );
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let read_payload = decoded
            .iter()
            .find(|f| f.name == "read_payload")
            .expect("read_payload");
        assert_eq!(read_payload.borrowed_paths.len(), 1);
        assert_eq!(
            read_payload.borrowed_paths[0].activation,
            Some(sm_format::semcode_decode::DecodedBorrowActivation::FrameEntry),
            "the ADT producer's Borrow event must stay FrameEntry - promotion here is driven \
             by the #1718 ADT-Borrow capability requirement, not by the anchor mechanism"
        );
    }

    // #1718 (I4.1): direct, Rust-level tests of
    // `verify_ownership_path_family_contract`, with no byte-level decode
    // involved at all - constructing `DecodedAccessPath` values directly and
    // calling the verifier's own policy function. These exist because
    // mutation M4b found that removing this function's `Write(AdtPayload)`
    // rejection, while leaving `sm-format`'s decode-time gate intact, broke
    // no existing test: `verify_semcode`/`verify_semcode_token` both decode
    // first, so a `Write(AdtPayload)` artifact never reaches this function's
    // body through the public API. `verifier_rejects_adt_payload_write_independently_of_decoder`
    // below is the one that closes that gap - it must not obtain its input
    // by decoding bytes, or it would prove the same thing the decoder-level
    // tests already prove, not this function's own independent authority.
    mod ownership_path_family_contract_tests {
        use super::*;
        use sm_format::semcode_decode::{
            DecodedAccessPath, DecodedAccessPathComponent, DecodedBorrowActivation,
        };
        use sm_format::semcode_format::{HEADER_V20, HEADER_V21};

        fn path(components: Vec<DecodedAccessPathComponent>) -> DecodedAccessPath {
            DecodedAccessPath {
                root_symbol_id: 0,
                components,
                activation: Some(DecodedBorrowActivation::FrameEntry),
                write_execution: None,
            }
        }

        #[test]
        fn v21_sequence_borrow_admitted() {
            let borrowed = vec![path(vec![DecodedAccessPathComponent::SequenceIndexStatic(
                0,
            )])];
            verify_ownership_path_family_contract("f", &HEADER_V21, &borrowed, &[])
                .expect("V21 must admit Sequence Borrow");
        }

        #[test]
        fn v21_sequence_write_admitted() {
            let write = vec![path(vec![DecodedAccessPathComponent::SequenceIndexStatic(
                0,
            )])];
            verify_ownership_path_family_contract("f", &HEADER_V21, &[], &write)
                .expect("V21 must admit Sequence Write");
        }

        #[test]
        fn v21_adt_borrow_admitted() {
            let borrowed = vec![path(vec![DecodedAccessPathComponent::AdtPayload {
                variant: 9,
                index: 0,
            }])];
            verify_ownership_path_family_contract("f", &HEADER_V21, &borrowed, &[])
                .expect("V21 must admit ADT Borrow");
        }

        // The critical test (I4.1's acceptance criterion): proves the
        // verifier's OWN rejection, independent of sm-format's decode-time
        // gate - the input here was never decoded from bytes at all, so
        // decoder cannot have "already caught this" the way it does for
        // every other test in this file that goes through
        // decode_semcode_envelope/verify_semcode/verify_semcode_token.
        #[test]
        fn verifier_rejects_adt_payload_write_independently_of_decoder() {
            let write = vec![path(vec![DecodedAccessPathComponent::AdtPayload {
                variant: 9,
                index: 0,
            }])];
            let report = verify_ownership_path_family_contract("f", &HEADER_V21, &[], &write)
                .expect_err("verifier must independently reject Write(AdtPayload)");
            assert_eq!(
                report.diagnostics[0].code,
                VerificationCode::InvalidOwnershipSection
            );
        }

        #[test]
        fn v20_sequence_rejected_for_missing_capability() {
            let borrowed = vec![path(vec![DecodedAccessPathComponent::SequenceIndexStatic(
                0,
            )])];
            let report = verify_ownership_path_family_contract("f", &HEADER_V20, &borrowed, &[])
                .expect_err("V20 lacks CAP_OWNERSHIP_SEQUENCE_PATHS");
            assert_eq!(
                report.diagnostics[0].code,
                VerificationCode::CapabilityViolation
            );
        }

        #[test]
        fn v20_adt_borrow_rejected_for_missing_capability() {
            let borrowed = vec![path(vec![DecodedAccessPathComponent::AdtPayload {
                variant: 9,
                index: 0,
            }])];
            let report = verify_ownership_path_family_contract("f", &HEADER_V20, &borrowed, &[])
                .expect_err("V20 lacks CAP_OWNERSHIP_ADT_BORROW_PATHS");
            assert_eq!(
                report.diagnostics[0].code,
                VerificationCode::CapabilityViolation
            );
        }
    }

    // #1726 Checkpoint D2b, item 8 fail-closed / adversarial tests, all
    // built by taking a real, compiled, admitted rev21 artifact
    // (`ownership_semcode_bytes()`) and corrupting exactly one anchor field
    // via `set_store_var_site_anchor` - which itself re-decodes and asserts
    // the intended field (and only that field) changed, so a wrong offset
    // fails the test setup loudly rather than silently testing nothing.
    // "Cross-function anchor" (item 8) is not a separate byte-level test:
    // `store_var_starts` is a local built fresh per call to
    // `verify_function_code` from that one function's own decoded
    // instructions, with no shared/global table a wrong-function anchor
    // could coincidentally hit - proven by construction (the variable's
    // scope), not by a runtime probe.

    #[test]
    fn checkpoint_d2b_rejects_out_of_range_anchor() {
        let mut bytes = ownership_semcode_bytes();
        set_store_var_site_anchor(&mut bytes, "main", 0, u32::MAX);
        let report = verify_semcode(&bytes).expect_err("out-of-range anchor must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_d2b_rejects_anchor_at_function_code_end() {
        let mut bytes = ownership_semcode_bytes();
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = functions.iter().find(|f| f.name == "main").expect("main");
        let instr_len = u32::try_from(main.code_slice.len() - main.instr_start_offset)
            .expect("instr_len fits u32");
        set_store_var_site_anchor(&mut bytes, "main", 0, instr_len);
        let report =
            verify_semcode(&bytes).expect_err("anchor exactly at the code end must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_d2b_rejects_anchor_mid_instruction() {
        let mut bytes = ownership_semcode_bytes();
        let real_anchor = anchor_of(&bytes, "main", Opcode::StoreVar, 0);
        set_store_var_site_anchor(&mut bytes, "main", 0, real_anchor + 1);
        let report = verify_semcode(&bytes)
            .expect_err("an offset one byte into a real StoreVar instruction must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_d2b_rejects_anchor_at_valid_non_store_var_instruction() {
        let mut bytes = ownership_semcode_bytes();
        let ret_anchor = anchor_of(&bytes, "main", Opcode::Ret, 0);
        set_store_var_site_anchor(&mut bytes, "main", 0, ret_anchor);
        let report = verify_semcode(&bytes)
            .expect_err("a genuine instruction boundary of the wrong opcode must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    // Item 12's required mutation test: retarget a real, working anchor from
    // its true StoreVar instruction start to an *operand* byte deliberately
    // set to the exact numeric value of `Opcode::StoreVar`'s own opcode
    // byte. This is precisely the failure mode the D2b design exists to
    // rule out: `code[offset] == StoreVar` is not sufficient, `offset` must
    // also be a genuine canonical instruction boundary. No shared/global
    // state is mutated (everything here is a local `Vec<u8>` freshly built
    // by `ownership_semcode_bytes()` at the top of the test), so there is
    // nothing to restore afterward.
    #[test]
    fn checkpoint_d2b_mutation_rejects_operand_byte_coincidentally_equal_to_store_var_opcode() {
        let mut bytes = ownership_semcode_bytes();
        let load_f64_pos = find_instruction(&bytes, "main", Opcode::LoadF64, 0);
        // LoadF64 layout: opcode(1) + dst register (u16) + f64 literal (8
        // raw bytes, never range- or value-checked by the verifier) - the
        // first literal byte is a safe place to plant an arbitrary value
        // without tripping an unrelated structural check first.
        let operand_byte_pos = load_f64_pos + 1 + 2;
        assert_ne!(
            bytes[operand_byte_pos],
            Opcode::StoreVar.byte(),
            "test setup should be introducing this collision, not observing a pre-existing one"
        );
        bytes[operand_byte_pos] = Opcode::StoreVar.byte();
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = functions.iter().find(|f| f.name == "main").expect("main");
        let planted_anchor =
            u32::try_from(operand_byte_pos - main.code_offset - main.instr_start_offset)
                .expect("anchor fits u32");
        set_store_var_site_anchor(&mut bytes, "main", 0, planted_anchor);
        let report = verify_semcode(&bytes).expect_err(
            "a byte that numerically equals the StoreVar opcode, but is not a canonical \
             instruction start, must still be rejected",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    fn two_tuple_borrows_semcode_bytes() -> Vec<u8> {
        let src = r#"
            fn main() {
                let pair: (i32, i32) = (1, 2);
                let (ref a, ref b): (i32, i32) = pair;
                let _ = a;
                let _ = b;
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    #[test]
    fn checkpoint_d2b_rejects_duplicate_anchor_between_two_borrow_events() {
        let bytes = two_tuple_borrows_semcode_bytes();
        verify_semcode(&bytes).expect("the un-corrupted two-anchor artifact must admit");
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = decoded.iter().find(|f| f.name == "main").expect("main");
        assert_eq!(
            main.borrowed_paths.len(),
            2,
            "fixture must carry two Borrow events"
        );
        let first_anchor = match main.borrowed_paths[0].activation {
            Some(sm_format::semcode_decode::DecodedBorrowActivation::StoreVarSite(a)) => a,
            other => panic!("expected StoreVarSite, got {other:?}"),
        };
        let mut bytes = bytes;
        set_store_var_site_anchor(&mut bytes, "main", 1, first_anchor);
        let report = verify_semcode(&bytes).expect_err(
            "two Borrow events sharing one anchor in the same function must be rejected",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    // #1891 Checkpoint W2E, item 9 positive tests.

    #[test]
    fn checkpoint_w2e_store_var_site_admitted() {
        let bytes = ownership_semcode_bytes();
        verify_semcode(&bytes).expect("a genuine StoreVarSite Write anchor must admit");
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = decoded.iter().find(|f| f.name == "main").expect("main");
        assert!(matches!(
            main.write_paths[0].write_execution,
            Some(sm_format::semcode_decode::DecodedWriteExecution::StoreVarSite(_))
        ));
    }

    #[test]
    fn checkpoint_w2e_make_record_site_admitted() {
        let bytes = record_field_write_semcode_bytes();
        verify_semcode(&bytes).expect("a genuine MakeRecordSite Write anchor must admit");
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = decoded.iter().find(|f| f.name == "main").expect("main");
        assert!(matches!(
            main.write_paths[0].write_execution,
            Some(sm_format::semcode_decode::DecodedWriteExecution::MakeRecordSite(_))
        ));
    }

    #[test]
    fn checkpoint_w2e_multi_field_record_update_shared_anchor_admitted() {
        let bytes = multi_field_record_update_write_semcode_bytes();
        verify_semcode(&bytes)
            .expect("two Write events sharing one MakeRecordSite anchor must admit");
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = decoded.iter().find(|f| f.name == "main").expect("main");
        assert_eq!(main.write_paths.len(), 2);
        assert_eq!(
            main.write_paths[0].write_execution, main.write_paths[1].write_execution,
            "both field-override Write events must share the identical MakeRecordSite anchor"
        );
    }

    #[test]
    fn checkpoint_w2e_repeated_assignment_distinct_anchors_admitted() {
        let bytes = repeated_assignment_write_semcode_bytes();
        verify_semcode(&bytes)
            .expect("two reassignments with distinct StoreVarSite anchors must admit");
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = decoded.iter().find(|f| f.name == "main").expect("main");
        assert_eq!(main.write_paths.len(), 2);
        assert_ne!(
            main.write_paths[0].write_execution, main.write_paths[1].write_execution,
            "distinct reassignments must never share a StoreVarSite anchor"
        );
    }

    #[test]
    fn checkpoint_w2e_mixed_store_var_and_make_record_admitted() {
        let bytes = mixed_store_var_and_make_record_write_semcode_bytes();
        verify_semcode(&bytes)
            .expect("a StoreVarSite and a MakeRecordSite Write in one function must both admit");
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = decoded.iter().find(|f| f.name == "main").expect("main");
        assert_eq!(main.write_paths.len(), 2);
        assert!(matches!(
            main.write_paths[0].write_execution,
            Some(sm_format::semcode_decode::DecodedWriteExecution::StoreVarSite(_))
        ));
        assert!(matches!(
            main.write_paths[1].write_execution,
            Some(sm_format::semcode_decode::DecodedWriteExecution::MakeRecordSite(_))
        ));
    }

    // Item 8: Borrow and Write are independent semantic domains with no
    // cross-domain anchor-uniqueness constraint - a valid artifact may
    // declare `Borrow StoreVarSite(A)` and `Write StoreVarSite(A)` for the
    // identical numeric PC. Constructed by retargeting a real Write event's
    // anchor onto the same real StoreVar instruction the fixture's own
    // Borrow event already anchors to (via `set_write_execution_site_anchor`,
    // which itself proves only the intended field changed).
    #[test]
    fn checkpoint_w2e_borrow_and_write_share_same_store_var_anchor_admitted() {
        let mut bytes = ownership_semcode_bytes();
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = decoded.iter().find(|f| f.name == "main").expect("main");
        let borrow_anchor = match main.borrowed_paths[0].activation {
            Some(sm_format::semcode_decode::DecodedBorrowActivation::StoreVarSite(a)) => a,
            other => panic!("expected StoreVarSite, got {other:?}"),
        };
        set_write_execution_site_anchor(&mut bytes, "main", 0, borrow_anchor);
        verify_semcode(&bytes).expect(
            "a Borrow and a Write declaring the identical StoreVarSite anchor must both admit - \
             independent domains, no cross-domain uniqueness constraint",
        );
    }

    // Item 9.G: with today's producers, every real Write event always
    // carries a resolved WriteSiteId (W2A), so a genuinely legacy
    // (write_execution: None) Write event can no longer occur in real
    // compiler output at any revision - it is reachable only via an
    // artifact predating that grammar. `downgrade_header_stripping_signature`
    // (already exercised for exactly this purpose elsewhere in this file)
    // reconstructs precisely that: a real Write event's root/path, decoded
    // then re-encoded in the legacy (pre-rev21) shape under a header below
    // `SEMCODE_OWNERSHIP_ANCHOR_MIN_REVISION`.
    #[test]
    fn checkpoint_w2e_legacy_pre_rev21_write_admitted_unchanged() {
        let bytes = downgrade_to_legacy_write_preserving_code(&ownership_semcode_bytes(), MAGIC11);
        let verified =
            verify_semcode(&bytes).expect("legacy pre-rev21 Write artifact must still admit");
        assert!(
            verified.header.rev < sm_format::semcode_format::SEMCODE_OWNERSHIP_ANCHOR_MIN_REVISION
        );
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = decoded.iter().find(|f| f.name == "main").expect("main");
        assert_eq!(
            main.write_paths[0].write_execution, None,
            "a legacy Write path must decode with no execution-site metadata at all"
        );
    }

    // #1891 Checkpoint W2E, item 10 fail-closed tests, mirroring D2b's
    // pattern exactly: take a real, admitted rev21 artifact and corrupt
    // exactly one anchor field via `set_write_execution_site_anchor`.

    #[test]
    fn checkpoint_w2e_rejects_store_var_site_out_of_range_anchor() {
        let mut bytes = ownership_semcode_bytes();
        set_write_execution_site_anchor(&mut bytes, "main", 0, u32::MAX);
        let report = verify_semcode(&bytes).expect_err("out-of-range anchor must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_w2e_rejects_store_var_site_anchor_at_function_code_end() {
        let mut bytes = ownership_semcode_bytes();
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = functions.iter().find(|f| f.name == "main").expect("main");
        let instr_len = u32::try_from(main.code_slice.len() - main.instr_start_offset)
            .expect("instr_len fits u32");
        set_write_execution_site_anchor(&mut bytes, "main", 0, instr_len);
        let report =
            verify_semcode(&bytes).expect_err("anchor exactly at the code end must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_w2e_rejects_store_var_site_anchor_mid_instruction() {
        let mut bytes = ownership_semcode_bytes();
        let real_anchor = anchor_of(&bytes, "main", Opcode::StoreVar, 0);
        set_write_execution_site_anchor(&mut bytes, "main", 0, real_anchor + 1);
        let report = verify_semcode(&bytes)
            .expect_err("an offset one byte into a real StoreVar instruction must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_w2e_rejects_store_var_site_at_valid_non_store_var_instruction() {
        let mut bytes = ownership_semcode_bytes();
        let ret_anchor = anchor_of(&bytes, "main", Opcode::Ret, 0);
        set_write_execution_site_anchor(&mut bytes, "main", 0, ret_anchor);
        let report = verify_semcode(&bytes)
            .expect_err("a genuine instruction boundary of the wrong opcode must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_w2e_rejects_make_record_site_out_of_range_anchor() {
        let mut bytes = record_field_write_semcode_bytes();
        set_write_execution_site_anchor(&mut bytes, "main", 0, u32::MAX);
        let report = verify_semcode(&bytes).expect_err("out-of-range anchor must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_w2e_rejects_make_record_site_anchor_at_function_code_end() {
        let mut bytes = record_field_write_semcode_bytes();
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = functions.iter().find(|f| f.name == "main").expect("main");
        let instr_len = u32::try_from(main.code_slice.len() - main.instr_start_offset)
            .expect("instr_len fits u32");
        set_write_execution_site_anchor(&mut bytes, "main", 0, instr_len);
        let report =
            verify_semcode(&bytes).expect_err("anchor exactly at the code end must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_w2e_rejects_make_record_site_anchor_mid_instruction() {
        let mut bytes = record_field_write_semcode_bytes();
        let real_anchor = anchor_of(&bytes, "main", Opcode::MakeRecord, 0);
        set_write_execution_site_anchor(&mut bytes, "main", 0, real_anchor + 1);
        let report = verify_semcode(&bytes)
            .expect_err("an offset one byte into a real MakeRecord instruction must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_w2e_rejects_make_record_site_at_valid_non_make_record_instruction() {
        let mut bytes = record_field_write_semcode_bytes();
        let ret_anchor = anchor_of(&bytes, "main", Opcode::Ret, 0);
        set_write_execution_site_anchor(&mut bytes, "main", 0, ret_anchor);
        let report = verify_semcode(&bytes)
            .expect_err("a genuine instruction boundary of the wrong opcode must be rejected");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    // Item 10's cross-kind swap - especially important since W2D exists
    // specifically to transport this discriminator: a declared StoreVarSite
    // pointing at a real, canonical MakeRecord (and vice versa) must be
    // rejected exactly like any other class mismatch, never accepted merely
    // because the anchor is a genuine instruction boundary of *some* kind.
    #[test]
    fn checkpoint_w2e_rejects_store_var_site_pointing_to_real_make_record() {
        let mut bytes = mixed_store_var_and_make_record_write_semcode_bytes();
        let make_record_anchor = anchor_of(&bytes, "main", Opcode::MakeRecord, 0);
        // write_paths[0] is this fixture's StoreVarSite event (see
        // `checkpoint_w2e_mixed_store_var_and_make_record_admitted`).
        set_write_execution_site_anchor(&mut bytes, "main", 0, make_record_anchor);
        let report = verify_semcode(&bytes).expect_err(
            "a StoreVarSite declaring a real MakeRecord's anchor must be rejected, not admitted \
             just because it is some genuine instruction boundary",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_w2e_rejects_make_record_site_pointing_to_real_store_var() {
        let mut bytes = mixed_store_var_and_make_record_write_semcode_bytes();
        let store_var_anchor = anchor_of(&bytes, "main", Opcode::StoreVar, 0);
        // write_paths[1] is this fixture's MakeRecordSite event.
        set_write_execution_site_anchor(&mut bytes, "main", 1, store_var_anchor);
        let report = verify_semcode(&bytes).expect_err(
            "a MakeRecordSite declaring a real StoreVar's anchor must be rejected, not admitted \
             just because it is some genuine instruction boundary",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    // Item 11's required mutation tests: retarget a real, working anchor
    // from its true instruction start to an *operand* byte deliberately set
    // to the exact numeric value of the target opcode's own opcode byte -
    // the precise failure mode canonical-decode-boundary admission exists to
    // rule out. No shared/global state is mutated.

    #[test]
    fn checkpoint_w2e_mutation_rejects_operand_byte_coincidentally_equal_to_store_var_opcode() {
        let mut bytes = ownership_semcode_bytes();
        let load_f64_pos = find_instruction(&bytes, "main", Opcode::LoadF64, 0);
        let operand_byte_pos = load_f64_pos + 1 + 2;
        assert_ne!(
            bytes[operand_byte_pos],
            Opcode::StoreVar.byte(),
            "test setup should be introducing this collision, not observing a pre-existing one"
        );
        bytes[operand_byte_pos] = Opcode::StoreVar.byte();
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = functions.iter().find(|f| f.name == "main").expect("main");
        let planted_anchor =
            u32::try_from(operand_byte_pos - main.code_offset - main.instr_start_offset)
                .expect("anchor fits u32");
        set_write_execution_site_anchor(&mut bytes, "main", 0, planted_anchor);
        let report = verify_semcode(&bytes).expect_err(
            "a byte that numerically equals the StoreVar opcode, but is not a canonical \
             instruction start, must still be rejected",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn checkpoint_w2e_mutation_rejects_operand_byte_coincidentally_equal_to_make_record_opcode() {
        let mut bytes = record_field_write_semcode_bytes();
        // `DecisionContext { camera: T, quality: 0.75 }`'s own MakeRecord
        // (occurrence 0) constructs `ctx`; occurrence 1 is the RecordUpdate
        // under test. A LoadF64 (the `0.75`/`1.0` literals) provides the
        // same safe, never-range-checked operand byte D2b's own mutation
        // test uses.
        let load_f64_pos = find_instruction(&bytes, "main", Opcode::LoadF64, 0);
        let operand_byte_pos = load_f64_pos + 1 + 2;
        assert_ne!(
            bytes[operand_byte_pos],
            Opcode::MakeRecord.byte(),
            "test setup should be introducing this collision, not observing a pre-existing one"
        );
        bytes[operand_byte_pos] = Opcode::MakeRecord.byte();
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = functions.iter().find(|f| f.name == "main").expect("main");
        let planted_anchor =
            u32::try_from(operand_byte_pos - main.code_offset - main.instr_start_offset)
                .expect("anchor fits u32");
        set_write_execution_site_anchor(&mut bytes, "main", 0, planted_anchor);
        let report = verify_semcode(&bytes).expect_err(
            "a byte that numerically equals the MakeRecord opcode, but is not a canonical \
             instruction start, must still be rejected",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    // Item 12: cross-function isolation. `store_var_starts`/
    // `make_record_starts` are locals freshly built per call to
    // `verify_function_code` from that one function's own decoded
    // instructions (mirroring D2b's own documented reasoning for Borrow
    // exactly) - there is no shared/global table a wrong-function anchor
    // could coincidentally hit. Demonstrated concretely: two functions where
    // the identical relative anchor is a genuine StoreVar boundary in one
    // but not the other; a Write event in the *other* function declaring
    // that anchor must reject, proving validity elsewhere confers no
    // authority here.
    #[test]
    fn checkpoint_w2e_cross_function_anchor_confers_no_authority() {
        let src = r#"
            fn helper() {
                let mut y: i32 = 0;
                y = 7;
                return;
            }

            fn main() {
                let mut x: i32 = 0;
                x = 5;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let helper_anchor = anchor_of(&bytes, "helper", Opcode::StoreVar, 0);
        let main_anchor = anchor_of(&bytes, "main", Opcode::StoreVar, 0);
        assert_eq!(
            helper_anchor, main_anchor,
            "test setup requires the identical relative anchor to be a genuine StoreVar \
             boundary in both functions, so retargeting proves same-function isolation \
             specifically, not merely an out-of-range anchor"
        );
        verify_semcode(&bytes).expect("the unmodified two-function artifact must admit");

        // Corrupt `helper`'s own Write event to point at a position that is
        // NOT a StoreVar boundary in `helper` (one byte into its own real
        // StoreVar) while remaining numerically identical to `main`'s valid
        // anchor is already guaranteed equal above - so this single
        // retarget exercises both directions: `helper`'s own local table
        // must reject it even though the identical relative offset is a
        // genuine StoreVar boundary in `main`.
        let mut bytes = bytes;
        set_write_execution_site_anchor(&mut bytes, "helper", 0, helper_anchor + 1);
        let report = verify_semcode(&bytes).expect_err(
            "helper's own Write event must be judged only against helper's own instruction \
             table, never admitted because the identical relative offset happens to be valid \
             in main",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    fn two_store_var_writes_semcode_bytes() -> Vec<u8> {
        let src = r#"
            fn main() {
                let mut x: i32 = 0;
                let mut y: i32 = 0;
                x = 1;
                y = 2;
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    // #1891 Checkpoint W2E, item 13: duplicate-anchor audit. `validate_write_sites`
    // (sm-ir, W2A) already proves a StoreVar WriteSiteId pairs with exactly
    // one Write event, so no frozen producer can legitimately emit two Write
    // events sharing one StoreVarSite anchor - reject unconditionally,
    // mirroring Borrow's own duplicate rule.
    #[test]
    fn checkpoint_w2e_rejects_duplicate_store_var_site_anchor_between_two_write_events() {
        let bytes = two_store_var_writes_semcode_bytes();
        verify_semcode(&bytes).expect("the un-corrupted two-anchor artifact must admit");
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = decoded.iter().find(|f| f.name == "main").expect("main");
        assert_eq!(
            main.write_paths.len(),
            2,
            "fixture must carry two Write events"
        );
        let first_anchor = match main.write_paths[0].write_execution {
            Some(sm_format::semcode_decode::DecodedWriteExecution::StoreVarSite(a)) => a,
            other => panic!("expected StoreVarSite, got {other:?}"),
        };
        let mut bytes = bytes;
        set_write_execution_site_anchor(&mut bytes, "main", 1, first_anchor);
        let report = verify_semcode(&bytes).expect_err(
            "two Write events sharing one StoreVarSite anchor in the same function must be rejected",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipAnchor
        );
    }

    #[test]
    fn verifier_accepts_record_field_write_ownership_semcode() {
        let bytes = record_field_write_semcode_bytes();
        // #1891 Checkpoint W2D: this RecordUpdate's Write event now always
        // carries a resolved WriteSiteId (Checkpoint W2C), promoting this
        // artifact to SEMCOD20/rev21 - was SEMCOD19/rev20 before this
        // checkpoint.
        assert_eq!(&bytes[..MAGIC20.len()], &MAGIC20);
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.header.rev, 21);
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_accepts_v13_program_without_record_field_ownership_payload() {
        let src = r#"
            fn saw_retry(values: Sequence(i32)) -> bool {
                let found: bool = false;
                for value in values {
                    if value == 2 {
                        found ||= true;
                    }
                }
                return found;
            }

            fn main() {
                let values: Sequence(i32) = [2, 9, 4];
                let found: bool = saw_retry(values);
                assert(found == true);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        // #1891 Checkpoint W2D: `found ||= true;` is a plain reassignment
        // (producer B), which now always carries a resolved WriteSiteId
        // (Checkpoint W2C), promoting this artifact to SEMCOD20/rev21 - was
        // SEMCOD19/rev20 (the SIG0 floor) before this checkpoint.
        assert_eq!(&bytes[..MAGIC20.len()], b"SEMCOD20");
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.header.rev, 21);
        assert_eq!(verified.functions.len(), 2);
    }

    #[test]
    fn verifier_rejects_short_header() {
        let report = verify_semcode(b"SEMC").expect_err("must reject");
        assert_eq!(report.diagnostics[0].code, VerificationCode::BadHeader);
    }

    #[test]
    fn verifier_rejects_unknown_opcode() {
        let mut bytes = compile_program_to_semcode("fn main() { return; }").expect("compile");
        // #1773 (FA-09-005): the first opcode byte's absolute offset now
        // depends on whether a SIG0 section (and, at this revision, OWN0)
        // precede the instruction stream - located via the real decode
        // rather than a hand-counted literal, which silently drifted out of
        // the instruction stream once SIG0 became mandatory.
        let (_, code_start, _) = function_code_span(&bytes, "main");
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let main = functions
            .iter()
            .find(|f| f.name == "main")
            .expect("main function");
        let opcode_pos = code_start + main.instr_start_offset;
        bytes[opcode_pos] = 0xff;
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(report.diagnostics[0].code, VerificationCode::UnknownOpcode);
    }

    #[test]
    fn verifier_rejects_truncated_function_body() {
        let mut bytes = compile_program_to_semcode("fn main() { return; }").expect("compile");
        bytes.truncate(bytes.len() - 1);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::TruncatedFunction
        );
    }

    #[test]
    fn verifier_rejects_truncated_string_table() {
        let mut bytes = compile_program_to_semcode("fn main() { return; }").expect("compile");
        let code_len_pos = 8 + 2 + 4;
        bytes[code_len_pos..code_len_pos + 4].copy_from_slice(&1u32.to_le_bytes());
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidStringTable
        );
    }

    #[test]
    fn verifier_rejects_jump_past_instruction_stream() {
        let mut bytes = compile_program_to_semcode("fn main() { if true { return; } return; }")
            .expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::JmpIf, 0);
        let target_pos = opcode_pos + 1 + 2;
        bytes[target_pos..target_pos + 4].copy_from_slice(&999u32.to_le_bytes());
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidJumpTarget
        );
    }

    #[test]
    fn verifier_rejects_bad_string_reference() {
        let mut bytes =
            compile_program_to_semcode("fn helper() { return; } fn main() { helper(); return; }")
                .expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::Call, 0);
        let sid_pos = opcode_pos + 1 + 1 + 2;
        bytes[sid_pos..sid_pos + 2].copy_from_slice(&99u16.to_le_bytes());
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidStringReference
        );
    }

    #[test]
    fn verifier_rejects_register_past_verified_local_budget() {
        let mut bytes = compile_program_to_semcode("fn main() { let a: bool = true; return; }")
            .expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::LoadBool, 0);
        let dst_pos = opcode_pos + 1;
        bytes[dst_pos..dst_pos + 2].copy_from_slice(&5000u16.to_le_bytes());
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidRegisterReference
        );
    }

    // #1757 (FA-07-017): a register above VerifiedLocal's 4096 budget but
    // within KernelBound's 8192 budget must be rejected under the default
    // (VerifiedLocal) admission gate.
    #[test]
    fn verify_semcode_token_default_still_rejects_register_past_verified_local_budget() {
        let mut bytes = compile_program_to_semcode("fn main() { let a: bool = true; return; }")
            .expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::LoadBool, 0);
        let dst_pos = opcode_pos + 1;
        bytes[dst_pos..dst_pos + 2].copy_from_slice(&5000u16.to_le_bytes());
        let report = verify_semcode_token(&bytes).expect_err("must reject under default quotas");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidRegisterReference
        );
    }

    // The same register must be ACCEPTED when the caller explicitly admits
    // against KernelBound quotas via the new quota-aware API.
    #[test]
    fn verify_semcode_token_with_quotas_accepts_register_within_kernel_bound_budget() {
        let mut bytes = compile_program_to_semcode("fn main() { let a: bool = true; return; }")
            .expect("compile");
        let load_opcode_pos = find_instruction(&bytes, "main", Opcode::LoadBool, 0);
        let load_dst_pos = load_opcode_pos + 1;
        bytes[load_dst_pos..load_dst_pos + 2].copy_from_slice(&5000u16.to_le_bytes());
        // #1756 (FA-07-016): the compiled body also has `StoreVar "a" src=r0`
        // reading whatever register `LoadBool` originally wrote. Patching
        // only `LoadBool`'s dst would leave that `StoreVar` reading a
        // register nothing ever defines - repoint it at the same patched
        // register so this fixture keeps proving "r5000 is within budget",
        // not "an undefined read happens to still be in range".
        let store_opcode_pos = find_instruction(&bytes, "main", Opcode::StoreVar, 0);
        let store_src_pos = store_opcode_pos + 1 + 2;
        bytes[store_src_pos..store_src_pos + 2].copy_from_slice(&5000u16.to_le_bytes());
        verify_semcode_token_with_quotas(&bytes, RuntimeQuotas::kernel_bound())
            .expect("r5000 must be within KernelBound's register budget");
    }

    // KernelBound is not infinite: a register above its own 8192 budget must
    // still be rejected even when explicitly admitting against KernelBound.
    #[test]
    fn verify_semcode_token_with_quotas_rejects_register_past_kernel_bound_budget() {
        let mut bytes = compile_program_to_semcode("fn main() { let a: bool = true; return; }")
            .expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::LoadBool, 0);
        let dst_pos = opcode_pos + 1;
        bytes[dst_pos..dst_pos + 2].copy_from_slice(&8200u16.to_le_bytes());
        let report = verify_semcode_token_with_quotas(&bytes, RuntimeQuotas::kernel_bound())
            .expect_err("must reject past KernelBound's own register budget");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidRegisterReference
        );
    }

    // No regression to the common path: an ordinary small artifact still
    // admits cleanly under the unchanged default quota profile.
    #[test]
    fn verify_semcode_token_default_still_accepts_ordinary_artifact() {
        let bytes = compile_program_to_semcode("fn main() { let a: bool = true; return; }")
            .expect("compile");
        verify_semcode_token(&bytes).expect("ordinary artifact must still admit");
    }

    // #1757 review follow-up: since quotas are now caller-supplied, a
    // rejection diagnostic must not hardcode "verified-local" - that text
    // would misidentify the active policy for any other profile (e.g. a
    // KernelBound rejection reporting its 8192-register budget as if it
    // were verified-local's, whose real budget is 4096).
    #[test]
    fn verify_semcode_token_with_quotas_rejection_message_does_not_hardcode_verified_local() {
        let mut bytes = compile_program_to_semcode("fn main() { let a: bool = true; return; }")
            .expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::LoadBool, 0);
        let dst_pos = opcode_pos + 1;
        bytes[dst_pos..dst_pos + 2].copy_from_slice(&8200u16.to_le_bytes());
        let report = verify_semcode_token_with_quotas(&bytes, RuntimeQuotas::kernel_bound())
            .expect_err("must reject past KernelBound's own register budget");
        assert!(
            !report.diagnostics[0].message.contains("verified-local"),
            "diagnostic must not claim a KernelBound rejection is a verified-local budget: {}",
            report.diagnostics[0].message
        );
    }

    // FA-07-001 (#1741): LOAD_BOOL literal byte must be canonical 0/1. A byte
    // outside that domain must be rejected at admission, not silently
    // normalized to `true` by the VM's `!= 0` check.
    #[test]
    fn verifier_rejects_non_canonical_bool_literal() {
        let mut bytes = compile_program_to_semcode("fn main() { let a: bool = true; return; }")
            .expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::LoadBool, 0);
        let literal_pos = opcode_pos + 1 + 2;
        bytes[literal_pos] = 0xff;
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::OperandOutOfBounds
        );
    }

    // FA-07-002 (#1742): LOAD_Q literal byte must be canonical 0..=3. A byte
    // outside that domain must be rejected at admission instead of receiving
    // a Verified token and only failing later in the VM as BadFormat.
    #[test]
    fn verifier_rejects_non_canonical_quad_literal() {
        // This exact fixture interns "a" as a string; the string table's
        // length byte for a 1-character string equals LoadQ's opcode byte
        // (0x01), which used to make a naive whole-buffer byte scan land on
        // the string table instead of the real LOAD_Q instruction (see
        // #1791). find_instruction locates it via the decoded instruction
        // stream instead, so it is unaffected by that coincidence.
        let mut bytes =
            compile_program_to_semcode("fn main() { let a: quad = N; return; }").expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::LoadQ, 0);
        let literal_pos = opcode_pos + 1 + 2;
        bytes[literal_pos] = 0xff;
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::OperandOutOfBounds
        );
    }

    // FA-07-003 (#1743): CALL's destination-present flag must be canonical
    // 0/1. The verifier currently reads and discards this byte entirely.
    #[test]
    fn verifier_rejects_non_canonical_call_dst_flag() {
        let mut bytes =
            compile_program_to_semcode("fn helper() { return; } fn main() { helper(); return; }")
                .expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::Call, 0);
        bytes[opcode_pos + 1] = 0xff;
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::OperandOutOfBounds
        );
    }

    // FA-07-013 (#1753): a no-destination CALL (has_dst=0) must not count its
    // ignored dst placeholder as a live register reference. The emitter
    // writes a dummy dst field even when has_dst=0; that dummy value must
    // never reach mark_reg, so an oversized placeholder must not spuriously
    // blow the register budget.
    #[test]
    fn verifier_accepts_call_no_dst_with_out_of_budget_placeholder() {
        let mut bytes =
            compile_program_to_semcode("fn helper() { return; } fn main() { helper(); return; }")
                .expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::Call, 0);
        assert_eq!(
            bytes[opcode_pos + 1],
            0u8,
            "expected has_dst=0 for a no-assignment call"
        );
        let dst_pos = opcode_pos + 2;
        bytes[dst_pos..dst_pos + 2].copy_from_slice(&5000u16.to_le_bytes());
        verify_semcode(&bytes).expect("no-dst call must ignore its dummy dst placeholder");
    }

    // FA-07-013 (#1753) control: a CALL that genuinely has a destination
    // (has_dst=1) must still have that real dst register validated against
    // the register budget - the fix must not blanket-skip mark_reg.
    #[test]
    fn verifier_rejects_call_real_dst_past_register_budget() {
        let mut bytes = compile_program_to_semcode(
            "fn helper() -> bool { return true; } fn main() { let a: bool = helper(); return; }",
        )
        .expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::Call, 0);
        assert_eq!(
            bytes[opcode_pos + 1],
            1u8,
            "expected has_dst=1 for an assigned call"
        );
        let dst_pos = opcode_pos + 2;
        bytes[dst_pos..dst_pos + 2].copy_from_slice(&5000u16.to_le_bytes());
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidRegisterReference
        );
    }

    // FA-07-013 (#1753) control: CLOSURE_CALL already ignores its dummy dst
    // placeholder when has_dst=0 - this locks in that reference behavior
    // that Opcode::Call's decode_operands branch was made to mirror.
    #[test]
    fn verifier_accepts_closure_call_no_dst_with_out_of_budget_placeholder() {
        let mut bytes = emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "helper".to_string(),
                    instrs: vec![IrInstr::Ret { src: None }],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: vec![
                        IrInstr::MakeClosure {
                            dst: 0,
                            name: "helper".to_string(),
                            captures: Vec::new(),
                        },
                        IrInstr::ClosureCall {
                            dst: None,
                            closure: 0,
                            arg: 0,
                        },
                        IrInstr::Ret { src: None },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::ClosureCall, 0);
        assert_eq!(bytes[opcode_pos + 1], 0u8, "expected has_dst=0");
        let dst_pos = opcode_pos + 2;
        bytes[dst_pos..dst_pos + 2].copy_from_slice(&5000u16.to_le_bytes());
        verify_semcode(&bytes).expect("no-dst closure-call must ignore its dummy dst placeholder");
    }

    // FA-07-003 (#1743): CLOSURE_CALL's destination-present flag must be
    // canonical 0/1, not "any nonzero byte is true".
    #[test]
    fn verifier_rejects_non_canonical_closure_call_dst_flag() {
        let mut bytes = emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "helper".to_string(),
                    instrs: vec![IrInstr::Ret { src: None }],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: vec![
                        IrInstr::MakeClosure {
                            dst: 0,
                            name: "helper".to_string(),
                            captures: Vec::new(),
                        },
                        IrInstr::ClosureCall {
                            dst: None,
                            closure: 0,
                            arg: 0,
                        },
                        IrInstr::Ret { src: None },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::ClosureCall, 0);
        bytes[opcode_pos + 1] = 0xff;
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::OperandOutOfBounds
        );
    }

    // FA-07-003 (#1743): RET's source-present flag must be canonical 0/1.
    #[test]
    fn verifier_rejects_non_canonical_ret_src_flag() {
        let mut bytes = compile_program_to_semcode("fn main() { return; }").expect("compile");
        let opcode_pos = find_instruction(&bytes, "main", Opcode::Ret, 0);
        bytes[opcode_pos + 1] = 0xff;
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::OperandOutOfBounds
        );
    }

    // FA-05-001 (#1731): TupleGet's opcode byte (0x44 = 'D') can begin the
    // exact same six bytes as the DBG0 debug-section sentinel + an empty
    // debug_symbol_count (`44 42 47 30 00 00` == "DBG0" + 0u16). This exact
    // IR sequence matches the confirmed Phase A fixture: it makes the
    // decoder currently reclassify the producer's real first instruction
    // (TupleGet dst=0x4742) as an empty debug section, hiding a destination
    // register far outside the verified-local register budget from every
    // downstream admission check. Both readings of these six bytes are
    // structurally valid (empty DBG0 metadata, or the start of a real
    // instruction stream), so admission must fail closed rather than
    // silently choosing one.
    #[test]
    fn verifier_rejects_ambiguous_dbg0_tupleget_collision() {
        let emitted = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::TupleGet {
                        dst: 0x4742,
                        src: 0x0030,
                        index: 0x6600,
                    },
                    IrInstr::Ret { src: None },
                    IrInstr::ClockRead { dst: 0 },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");

        // #1773 (FA-09-005): `emit_ir_to_semcode` now unconditionally
        // targets SEMCOD19+ (mandatory OWN0 + SIG0 sections), which places
        // real, correctly-tagged sections between the string table and the
        // instruction stream - closing off exactly this collision site for
        // any artifact the current emitter can produce (the DBG0 sniff runs
        // once, immediately after the string table, and now finds a real
        // "OWN0" tag there instead of the TupleGet's colliding bytes). The
        // #1731 vulnerability this test proves closed remains real for
        // pre-#1732 header revisions (V0-V10, no mandatory OWN0), so this
        // rebuilds the bare `[string table][instruction stream]` envelope
        // shape those headers produce - string table and instruction-stream
        // bytes taken verbatim from the real emission above, only the
        // now-mandatory OWN0/SIG0 sections excised - to keep proving the
        // fix holds for every artifact shape the verifier must still admit.
        let (_, emitted_functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&emitted).expect("decode");
        let emitted_env = &emitted_functions[0];
        let string_table = &emitted_env.code_slice[..2];
        let instr_stream = &emitted_env.code_slice[emitted_env.instr_start_offset..];
        let mut code = Vec::new();
        code.extend_from_slice(string_table);
        code.extend_from_slice(instr_stream);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC0);
        bytes.extend_from_slice(&4u16.to_le_bytes());
        bytes.extend_from_slice(b"main");
        bytes.extend_from_slice(&(code.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&code);

        // B: byte identity - the reconstructed instruction stream literally
        // spells DBG0 where the real TupleGet begins (string table is
        // empty: a bare 2-byte count=0, so the collision starts at
        // code_slice[2]).
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let env = &functions[0];
        assert_eq!(&env.code_slice[2..6], b"DBG0");

        // C: decoder reinterpretation - instr_start_offset must currently
        // land past the fake sentinel + fake empty count (2 + 4 + 2 = 8),
        // not at the true instruction start (2, right after the empty
        // string table) - i.e. this is not merely byte-identical, the
        // shared decoder actually consumes it as metadata today.
        assert_eq!(env.instr_start_offset, 8);

        // D+E: admission consequence - despite the hidden TupleGet
        // referencing register 0x4742 (18242), far outside any verified-
        // local register budget, admission must fail closed rather than
        // silently accept it.
        let report = verify_semcode(&bytes).expect_err("must reject ambiguous framing");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AmbiguousInstructionFraming
        );
    }

    // #1731 regression matrix (2/3): a minimal genuine DBG0 section (debug
    // symbols enabled, one traced instruction) must remain accepted
    // unchanged - it is not byte-identical to any instruction reading, so
    // it is not ambiguous. Built via emit_ir_to_semcode (IR-level, not the
    // source front-end) so this test doesn't depend on the `debug-symbols`
    // cargo feature being enabled for every invocation this repo uses to
    // run sm-verify's tests (e.g. 7hell Hell 4 enables sm-ir/profile-rust
    // only).
    #[test]
    fn verifier_accepts_minimal_genuine_debug_section() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![IrInstr::Ret { src: None }],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            true,
        )
        .expect("emit");
        let verified = verify_semcode(&bytes).expect("genuine debug section must verify");
        assert_eq!(verified.functions.len(), 1);
        assert!(verified.functions[0].debug_symbol_count > 0);
    }

    // #1731 regression matrix (4): an ordinary, non-empty DBG0 section
    // (several traced instructions) must remain accepted unchanged.
    #[test]
    fn verifier_accepts_ordinary_debug_section_with_multiple_entries() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadI32 { dst: 0, val: 1 },
                    IrInstr::LoadI32 { dst: 1, val: 2 },
                    IrInstr::AddI32 {
                        dst: 2,
                        lhs: 0,
                        rhs: 1,
                    },
                    IrInstr::Ret { src: Some(2) },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            true,
        )
        .expect("emit");
        let verified = verify_semcode(&bytes).expect("ordinary debug section must verify");
        assert_eq!(verified.functions.len(), 1);
        assert!(
            verified.functions[0].debug_symbol_count > 1,
            "fixture must exercise a genuinely non-empty debug section"
        );
    }

    /// #1746 (FA-07-006) test helper: overwrites the `pc` field of the
    /// `symbol_index`-th `DBG0` debug-symbol entry in `bytes` in place, via
    /// the decoded format offsets (`code_offset`, `string_table_end_offset`)
    /// rather than a raw-byte search. Entry layout after the `DBG0` tag (4
    /// bytes) and symbol count (2 bytes) is `pc: u32 LE, line: u32,
    /// col: u16` (10 bytes per entry), matching
    /// `semcode_decode::parse_string_table_debug_and_ownership`.
    fn overwrite_debug_symbol_pc(bytes: &mut [u8], symbol_index: usize, new_pc: u32) {
        let entry_offset = {
            let (_, functions) = sm_format::semcode_decode::decode_semcode_envelope(bytes)
                .expect("decode for mutation");
            let f = &functions[0];
            f.code_offset + f.string_table_end_offset + 4 + 2 + symbol_index * 10
        };
        bytes[entry_offset..entry_offset + 4].copy_from_slice(&new_pc.to_le_bytes());
    }

    // #1746 (FA-07-006) regression matrix (1/7 and 2/7): a debug-symbol pc
    // that is numerically in range but lands inside a decoded instruction's
    // operand bytes - not on an instruction start - must reject. `LOAD_BOOL
    // dst, val` is a real 4-byte instruction (opcode + 2-byte dst register +
    // 1-byte bool literal) at instruction offset 0; offsets 1 and 2 are two
    // distinct interior bytes of its `dst` operand, proving this is a
    // boundary-membership check (`instr_starts.contains`), not a single
    // hardcoded interior offset.
    #[test]
    fn verifier_rejects_debug_pc_pointing_into_operand_middle_byte() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadBool { dst: 0, val: true },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            true,
        )
        .expect("emit");
        let mut mutated = bytes.clone();
        overwrite_debug_symbol_pc(&mut mutated, 0, 1);
        let report = verify_semcode(&mutated)
            .expect_err("debug pc pointing into the first interior operand byte must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidDebugSection
        );
        assert_eq!(report.diagnostics[0].offset, Some(1));
    }

    #[test]
    fn verifier_rejects_debug_pc_pointing_into_operand_final_byte() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadBool { dst: 0, val: true },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            true,
        )
        .expect("emit");
        let mut mutated = bytes.clone();
        overwrite_debug_symbol_pc(&mut mutated, 0, 2);
        let report = verify_semcode(&mutated)
            .expect_err("debug pc pointing into a second interior operand byte must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidDebugSection
        );
        assert_eq!(report.diagnostics[0].offset, Some(2));
    }

    // #1746 regression matrix (3/7 and 4/7): pc == 0 at the first
    // instruction, and pc pointing at a later real instruction start, must
    // both remain accepted - unmutated, genuinely compiler-emitted debug
    // symbols for a two-instruction function land exactly on the two real
    // instruction starts (0 and 4).
    #[test]
    fn verifier_accepts_debug_pc_at_first_and_later_instruction_starts() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadBool { dst: 0, val: true },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            true,
        )
        .expect("emit");
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let pcs: Vec<usize> = functions[0].debug_symbols.iter().map(|s| s.pc).collect();
        assert_eq!(
            pcs,
            vec![0, 4],
            "fixture must exercise pc=0 (first instruction) and a later instruction start"
        );
        let verified =
            verify_semcode(&bytes).expect("debug pcs on real instruction starts must verify");
        assert_eq!(verified.functions[0].debug_symbol_count, 2);
    }

    // #1746 regression matrix (5/7): out-of-range debug pc must still
    // reject with the existing diagnostic and message, unchanged by the new
    // boundary check (range is checked first).
    #[test]
    fn verifier_rejects_out_of_range_debug_pc() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadBool { dst: 0, val: true },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            true,
        )
        .expect("emit");
        let mut mutated = bytes.clone();
        overwrite_debug_symbol_pc(&mut mutated, 0, 100);
        let report = verify_semcode(&mutated).expect_err("out-of-range debug pc must still reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidDebugSection
        );
        assert_eq!(report.diagnostics[0].offset, Some(100));
        assert_eq!(
            report.diagnostics[0].message,
            "debug symbol pc points past the instruction stream"
        );
    }

    /// #1760 (FA-08-002) test helper: builds a single-function artifact with
    /// exactly `n` genuine debug symbols, one per instruction (matching
    /// `emit_ir_to_semcode`'s real emission - each instruction contributes
    /// exactly one `(pc, line, col)` entry, see `legacy_lowering.rs`'s `dbg.push`
    /// call site). The last instruction is `Ret`; every earlier instruction is
    /// a harmless repeated `LoadI32 { dst: 0, val: 0 }`, so the fixture is a
    /// real, fully-verifiable function - not a hand-crafted byte blob - and
    /// every debug pc genuinely lands on a real instruction start.
    fn build_function_with_n_debug_symbols(n: usize) -> Vec<u8> {
        assert!(n >= 1, "fixture needs at least the trailing Ret");
        let mut instrs: Vec<IrInstr> = (0..n - 1)
            .map(|_| IrInstr::LoadI32 { dst: 0, val: 0 })
            .collect();
        instrs.push(IrInstr::Ret { src: None });
        emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs,
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            true,
        )
        .expect("emit")
    }

    // #1760 (FA-08-002): the renamed `max_debug_symbols_per_function` must
    // preserve `pure_compute`'s pre-rename behavior exactly - 4096 genuine
    // debug symbols (one per instruction) must still admit under
    // `pure_compute`, whose configured limit is exactly 4096.
    #[test]
    fn verify_semcode_token_with_quotas_accepts_pure_compute_at_exactly_4096_debug_symbols() {
        let bytes = build_function_with_n_debug_symbols(4_096);
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        assert_eq!(functions[0].debug_symbols.len(), 4_096);
        verify_semcode_token_with_quotas(&bytes, RuntimeQuotas::pure_compute())
            .expect("exactly 4096 debug symbols must be admitted under pure_compute's own limit");
    }

    // #1760 (FA-08-002): the adversarial proof this checkpoint exists for.
    // 4097 debug symbols is within `sm-format`'s independent, unrelated fixed
    // decode-time cap (`MAX_DEBUG_SYMBOLS_PER_FUNCTION = 8192`), so this
    // artifact decodes cleanly - the rejection must come from `sm-verify`'s
    // own `pure_compute`-configured limit (4096), proving the rename did not
    // silently widen `pure_compute`'s admission to the decoder's 8192 cap.
    #[test]
    fn verify_semcode_token_with_quotas_rejects_pure_compute_at_4097_debug_symbols() {
        let bytes = build_function_with_n_debug_symbols(4_097);
        let report = verify_semcode_token_with_quotas(&bytes, RuntimeQuotas::pure_compute())
            .expect_err("4097 debug symbols must exceed pure_compute's 4096 limit");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::ResourceLimitExceeded
        );
        assert_eq!(
            report.diagnostics[0].message,
            "debug section uses 4097 entries, exceeding the per-function debug-symbol limit of 4096"
        );
    }

    // #1760 (FA-08-002): the same 4097-debug-symbol artifact that
    // `pure_compute` rejects must remain accepted under `verified_local`
    // (limit 8192) - this is the "dead code for verified_local/kernel_bound"
    // half of the decision's falsification: the check only has real
    // authority for `pure_compute`, whose 4096 is stricter than `sm-format`'s
    // fixed 8192 decode-time cap.
    #[test]
    fn verify_semcode_token_with_quotas_verified_local_accepts_4097_debug_symbols() {
        let bytes = build_function_with_n_debug_symbols(4_097);
        verify_semcode_token_with_quotas(&bytes, RuntimeQuotas::verified_local())
            .expect("4097 debug symbols must remain accepted under verified_local's 8192 limit");
    }

    // #1731 regression matrix (7/8): a truncated DBG0 tag or count must
    // still hit the existing deterministic malformed-section rejection,
    // unaffected by the new ambiguity check (which only runs once a full,
    // successfully-decoded DBG0 section is already present).
    #[test]
    fn verifier_rejects_truncated_debug_section_tag() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![IrInstr::Ret { src: None }],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            true,
        )
        .expect("emit");
        // Truncate the artifact to land inside the DBG0 tag/count region,
        // matching this file's existing truncation-test convention.
        let mut truncated = bytes.clone();
        truncated.truncate(truncated.len() - 1);
        let report =
            sm_format::semcode_decode::decode_semcode_envelope(&truncated).expect_err("decode");
        assert!(matches!(
            report,
            sm_format::semcode_decode::DecodeError::InvalidDebugSection { .. }
                | sm_format::semcode_decode::DecodeError::TruncatedFunction { .. }
        ));
    }

    // #1731 review follow-up: the ambiguity probe must be a purely
    // STRUCTURAL question (opcode recognized, every operand's byte shape
    // present, stream consumes exactly to the end) and must NOT be gated on
    // whether the alternative reading's operand *values* are semantically
    // canonical. Before this fix, `instruction_stream_parses_fully` reused
    // `decode_operands` with full canonical-domain enforcement, so an
    // alternative reading that was shape-complete but contained one
    // non-canonical literal (out-of-domain, not out-of-space) was wrongly
    // treated as "not a real competing interpretation" and the DBG0 reading
    // was silently kept.
    //
    // Fixture: reuses the exact #1731 TupleGet collision (dst=0x4742,
    // src=0x0030 spell `DBG0` + an empty debug_symbol_count, exactly like
    // `verifier_rejects_ambiguous_dbg0_tupleget_collision` above), followed
    // by a genuine `LoadBool{dst:0, val:true}` whose literal byte is then
    // hand-patched from canonical `1` to non-canonical `0xff`. This does not
    // change any operand's byte width (the literal is always exactly one
    // byte regardless of its value), so the alternative reading - decoding
    // the whole buffer from the start, ignoring the `DBG0` sniff entirely -
    // remains fully shape-complete: `TupleGet{..}; LoadBool{dst:0,
    // literal:0xff}; Ret{None}`, consuming every byte to the end, with one
    // non-canonical literal.
    //
    // What the `DBG0` reading's own (byte-shifted, effectively garbled)
    // instruction decode would produce is irrelevant here: the ambiguity
    // check runs before that decode is ever attempted, so this test only
    // needs the alternative reading to be genuinely shape-complete.
    #[test]
    fn verifier_rejects_ambiguous_framing_even_when_alternate_reading_has_non_canonical_operand() {
        let emitted = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::TupleGet {
                        dst: 0x4742,
                        src: 0x0030,
                        index: 0x6600,
                    },
                    IrInstr::LoadBool { dst: 0, val: true },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");

        // #1773 (FA-09-005): rebuild the bare `[string table][instruction
        // stream]` envelope shape (no mandatory OWN0/SIG0) so the DBG0 sniff
        // lands on the TupleGet collision again - see the identical comment
        // on `verifier_rejects_ambiguous_dbg0_tupleget_collision` above.
        let (_, emitted_functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&emitted).expect("decode");
        let emitted_env = &emitted_functions[0];
        let mut code = Vec::new();
        code.extend_from_slice(&emitted_env.code_slice[..2]);
        code.extend_from_slice(&emitted_env.code_slice[emitted_env.instr_start_offset..]);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC0);
        bytes.extend_from_slice(&4u16.to_le_bytes());
        bytes.extend_from_slice(b"main");
        bytes.extend_from_slice(&(code.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&code);

        let (_, code_start, _) = function_code_span(&bytes, "main");
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let env = &functions[0];
        assert!(env.has_debug_section);
        assert_eq!(&env.code_slice[2..6], b"DBG0");

        // TupleGet is 7 bytes (opcode + dst + src + index); LoadBool's
        // literal is the 4th byte of the instruction right after it
        // (opcode + dst(2) + literal).
        let literal_pos = code_start + env.string_table_end_offset + 7 + 1 + 2;
        assert_eq!(
            bytes[literal_pos], 1,
            "must be patching LoadBool's literal byte"
        );
        bytes[literal_pos] = 0xff;

        let report = verify_semcode(&bytes).expect_err("must reject ambiguous framing");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AmbiguousInstructionFraming
        );
    }

    // #1731 review follow-up (round 2): `MAKE_TUPLE`'s arity-`>=2` check and
    // `MAKE_RECORD`'s slot-count-`>=1` check are cardinality/value-domain
    // constraints, not byte-shape constraints - the count field itself is
    // always read unconditionally and always determines how many further
    // item-register bytes follow, regardless of whether that count value is
    // semantically canonical. They must be gated on
    // `enforce_canonical_domains` exactly like the literal/flag checks
    // above, or the same silent-ambiguity gap reopens for any alternative
    // reading built around a too-small tuple/record.
    //
    // Fixture: the exact #1731 TupleGet collision, followed by a genuine
    // `MakeTuple{dst:0, items:[5]}` - arity 1, non-canonical (`< 2`) but
    // fully shape-complete (the count field legitimately says "one item
    // follows", and one item follows).
    #[test]
    fn verifier_rejects_ambiguous_framing_with_non_canonical_maketuple_arity() {
        let emitted = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::TupleGet {
                        dst: 0x4742,
                        src: 0x0030,
                        index: 0x6600,
                    },
                    IrInstr::MakeTuple {
                        dst: 0,
                        items: vec![5],
                    },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");

        // #1773 (FA-09-005): rebuild the bare `[string table][instruction
        // stream]` envelope shape - see the identical comment on
        // `verifier_rejects_ambiguous_dbg0_tupleget_collision` above.
        let (_, emitted_functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&emitted).expect("decode");
        let emitted_env = &emitted_functions[0];
        let mut code = Vec::new();
        code.extend_from_slice(&emitted_env.code_slice[..2]);
        code.extend_from_slice(&emitted_env.code_slice[emitted_env.instr_start_offset..]);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC0);
        bytes.extend_from_slice(&4u16.to_le_bytes());
        bytes.extend_from_slice(b"main");
        bytes.extend_from_slice(&(code.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&code);

        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let env = &functions[0];
        assert!(env.has_debug_section);
        assert_eq!(&env.code_slice[2..6], b"DBG0");

        let report = verify_semcode(&bytes).expect_err("must reject ambiguous framing");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AmbiguousInstructionFraming
        );
    }

    #[test]
    fn verifier_rejects_unknown_call_target() {
        let mut bytes =
            compile_program_to_semcode("fn helper() { return; } fn main() { helper(); return; }")
                .expect("compile");
        let helper_pos = bytes
            .windows(b"helper".len())
            .rposition(|window| window == b"helper")
            .expect("helper string");
        bytes[helper_pos..helper_pos + b"helper".len()].copy_from_slice(b"gh0st!");
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UnknownCallTarget
        );
    }

    // #1773 (FA-09-005) permanent regressions: sm-verify's arity
    // enforcement. Built at the IR level (not the source front-end) because
    // the front-end always emits a call site whose argc matches the
    // callee's real declared arity - a genuine mismatch can only arise from
    // a malformed/adversarial artifact, exactly what the verifier's
    // admission gate must catch.

    #[test]
    fn verifier_rejects_call_argument_count_mismatch() {
        let bytes = emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "callee".to_string(),
                    instrs: vec![IrInstr::Ret { src: None }],
                    ownership_events: Vec::new(),
                    params: vec![CallableValueFamily::I32, CallableValueFamily::I32],
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: vec![
                        IrInstr::LoadI32 { dst: 0, val: 1 },
                        IrInstr::Call {
                            dst: None,
                            name: "callee".to_string(),
                            args: vec![0],
                        },
                        IrInstr::Ret { src: None },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit");

        let report = verify_semcode(&bytes).expect_err("must reject argc/arity mismatch");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::CallArgumentCountMismatch
        );
    }

    #[test]
    fn verifier_accepts_call_matching_argument_count() {
        let bytes = emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "callee".to_string(),
                    instrs: vec![IrInstr::Ret { src: None }],
                    ownership_events: Vec::new(),
                    params: vec![CallableValueFamily::I32, CallableValueFamily::I32],
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: vec![
                        IrInstr::LoadI32 { dst: 0, val: 1 },
                        IrInstr::LoadI32 { dst: 1, val: 2 },
                        IrInstr::Call {
                            dst: None,
                            name: "callee".to_string(),
                            args: vec![0, 1],
                        },
                        IrInstr::Ret { src: None },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit");

        let verified = verify_semcode(&bytes).expect("matching argc must verify");
        assert_eq!(verified.functions.len(), 2);
    }

    // --- #1756 (FA-07-016, umbrella #1617) regression matrix -------------
    //
    // Forward MUST dataflow proving every reachable register read is
    // definitely defined on every incoming execution path. `ENTRY_DEFS` is
    // exactly `{r0..signature.families.len()-1}` from the function's
    // canonical SIG0 signature (#1773); every other register starts
    // undefined until an instruction actually writes it. Cases 1-26 below
    // match the campaign's required regression matrix one-to-one.

    #[test]
    fn c1756_case1_rejects_zero_arg_entry_undefined_read() {
        let bytes = emit_test_function(vec![IrInstr::Ret { src: Some(0) }]);
        let report =
            verify_semcode(&bytes).expect_err("r0 undefined at a zero-arg entry must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    #[test]
    fn c1756_case2_accepts_one_parameter_entry_read() {
        let bytes = emit_test_function_with_params(
            vec![CallableValueFamily::I32],
            vec![IrInstr::Ret { src: Some(0) }],
        );
        verify_semcode(&bytes).expect("r0 is entry-defined by SIG0's one parameter");
    }

    #[test]
    fn c1756_case3_accepts_multiple_parameter_entry_reads() {
        let bytes = emit_test_function_with_params(
            vec![
                CallableValueFamily::I32,
                CallableValueFamily::I32,
                CallableValueFamily::I32,
            ],
            vec![
                IrInstr::AddI32 {
                    dst: 3,
                    lhs: 0,
                    rhs: 1,
                },
                IrInstr::AddI32 {
                    dst: 3,
                    lhs: 3,
                    rhs: 2,
                },
                IrInstr::Ret { src: Some(3) },
            ],
        );
        verify_semcode(&bytes).expect("r0..r2 are all entry-defined by SIG0's three parameters");
    }

    #[test]
    fn c1756_case4_rejects_register_just_past_parameter_prefix() {
        let bytes = emit_test_function_with_params(
            vec![CallableValueFamily::I32],
            vec![IrInstr::Ret { src: Some(1) }],
        );
        let report = verify_semcode(&bytes)
            .expect_err("r1 is not entry-defined by a one-parameter signature");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    #[test]
    fn c1756_case5_rejects_read_before_write_same_register() {
        let bytes = emit_test_function(vec![
            IrInstr::AddI32 {
                dst: 6,
                lhs: 5,
                rhs: 5,
            },
            IrInstr::LoadI32 { dst: 5, val: 1 },
            IrInstr::Ret { src: Some(6) },
        ]);
        let report = verify_semcode(&bytes).expect_err("r5 is read before any write reaches it");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    #[test]
    fn c1756_case6_accepts_write_before_read_same_register() {
        let bytes = emit_test_function(vec![
            IrInstr::LoadI32 { dst: 5, val: 1 },
            IrInstr::AddI32 {
                dst: 6,
                lhs: 5,
                rhs: 5,
            },
            IrInstr::Ret { src: Some(6) },
        ]);
        verify_semcode(&bytes).expect("r5 is written before it is read");
    }

    #[test]
    fn c1756_case7_accepts_read_when_both_branches_define() {
        let bytes = emit_test_function(vec![
            IrInstr::LoadBool { dst: 0, val: true },
            IrInstr::JmpIf {
                cond: 0,
                label: "then".to_string(),
            },
            IrInstr::LoadI32 { dst: 5, val: 1 }, // else path
            IrInstr::Jmp {
                label: "join".to_string(),
            },
            IrInstr::Label {
                name: "then".to_string(),
            },
            IrInstr::LoadI32 { dst: 5, val: 2 }, // then path
            IrInstr::Label {
                name: "join".to_string(),
            },
            IrInstr::Ret { src: Some(5) },
        ]);
        verify_semcode(&bytes).expect("both branches define r5 before the join reads it");
    }

    #[test]
    fn c1756_case8_rejects_read_when_only_one_branch_defines() {
        let bytes = emit_test_function(vec![
            IrInstr::LoadBool { dst: 0, val: true },
            IrInstr::JmpIf {
                cond: 0,
                label: "then".to_string(),
            },
            IrInstr::Jmp {
                label: "join".to_string(),
            }, // else path: no write
            IrInstr::Label {
                name: "then".to_string(),
            },
            IrInstr::LoadI32 { dst: 5, val: 2 }, // only the then path writes r5
            IrInstr::Label {
                name: "join".to_string(),
            },
            IrInstr::Ret { src: Some(5) },
        ]);
        let report = verify_semcode(&bytes).expect_err("only the 'then' branch defines r5");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    #[test]
    fn c1756_case9_accepts_definition_before_branch_survives_join() {
        let bytes = emit_test_function(vec![
            IrInstr::LoadI32 { dst: 5, val: 1 },
            IrInstr::LoadBool { dst: 0, val: true },
            IrInstr::JmpIf {
                cond: 0,
                label: "join".to_string(),
            },
            IrInstr::Label {
                name: "join".to_string(),
            },
            IrInstr::Ret { src: Some(5) },
        ]);
        verify_semcode(&bytes).expect("r5 defined before the branch remains defined on both paths");
    }

    #[test]
    fn c1756_case10_accepts_dominating_definition_before_loop() {
        let bytes = emit_test_function(vec![
            IrInstr::LoadI32 { dst: 5, val: 1 },
            IrInstr::Label {
                name: "loop_head".to_string(),
            },
            IrInstr::AddI32 {
                dst: 6,
                lhs: 5,
                rhs: 5,
            },
            IrInstr::Jmp {
                label: "loop_head".to_string(),
            },
        ]);
        verify_semcode(&bytes)
            .expect("r5 defined once before the loop remains valid on every iteration");
    }

    #[test]
    fn c1756_case11_rejects_first_iteration_reading_loop_carried_undefined_register() {
        let bytes = emit_test_function(vec![
            IrInstr::Label {
                name: "loop_head".to_string(),
            },
            IrInstr::AddI32 {
                dst: 6,
                lhs: 5,
                rhs: 5,
            }, // reads r5, which no path yet defines on entry to loop_head
            IrInstr::LoadI32 { dst: 5, val: 1 }, // defines r5 only for the NEXT iteration
            IrInstr::Jmp {
                label: "loop_head".to_string(),
            },
        ]);
        let report = verify_semcode(&bytes)
            .expect_err("the first iteration reads r5 before any path has defined it");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    /// Case 12 proves the non-negotiable TOP-initialization rule is load-
    /// bearing, not merely a stated preference. `loop_head` (the `AddI32`)
    /// has two predecessors: the entry `LoadI32` and the back-edge `JmpIf`.
    /// `r5` is written exactly once, before the loop, and never rewritten
    /// inside it - nothing inside the loop body ever re-adds `r5` to a set
    /// that excludes it. Under the correct rule (non-entry nodes start at
    /// `TOP = U`), `IN[loop_head]` in the very first fixed-point round is
    /// `OUT[entry] ∩ TOP = OUT[entry]`, which already (correctly) contains
    /// `r5`. Under the incorrect rule (non-entry nodes start at `∅`),
    /// `OUT[back-edge]` starts at `∅`, so `IN[loop_head] = OUT[entry] ∩ ∅ =
    /// ∅` in round one - and because MUST-intersection only ever shrinks a
    /// set and nothing in this loop ever re-adds `r5`, that wrongly-empty
    /// result can never self-correct in any later round: it is the loop's
    /// actual (wrong) converged fixed point, not a transient one. This is
    /// exactly the "correct fixed point" vs. "wrong fixed point" failure
    /// the rule exists to prevent - not just a slower convergence.
    #[test]
    fn c1756_case12_accepts_loop_read_requires_top_initialization() {
        let bytes = emit_test_function(vec![
            IrInstr::LoadI32 { dst: 5, val: 1 }, // entry: the only write to r5 anywhere
            IrInstr::Label {
                name: "loop_head".to_string(),
            },
            IrInstr::AddI32 {
                dst: 6,
                lhs: 5,
                rhs: 5,
            }, // loop_head: reads r5 every iteration
            IrInstr::JmpIf {
                cond: 6,
                label: "loop_head".to_string(),
            }, // conditional back-edge
            IrInstr::Ret { src: Some(6) }, // exit path
        ]);
        verify_semcode(&bytes).expect(
            "r5, defined once before the loop and never invalidated inside it, must remain \
             provably defined at loop_head - this is exactly the case that would incorrectly \
             reject under a non-entry-nodes-start-at-empty-set initialization strategy",
        );
    }

    #[test]
    fn c1756_case13_rejects_call_argument_undefined() {
        let bytes = emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "callee".to_string(),
                    instrs: vec![IrInstr::Ret { src: None }],
                    ownership_events: Vec::new(),
                    params: vec![CallableValueFamily::I32],
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: vec![
                        IrInstr::Call {
                            dst: None,
                            name: "callee".to_string(),
                            args: vec![7],
                        },
                        IrInstr::Ret { src: None },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit");
        let report = verify_semcode(&bytes).expect_err("call argument r7 is undefined");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    #[test]
    fn c1756_case14_accepts_call_argument_defined() {
        let bytes = emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "callee".to_string(),
                    instrs: vec![IrInstr::Ret { src: None }],
                    ownership_events: Vec::new(),
                    params: vec![CallableValueFamily::I32],
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: vec![
                        IrInstr::LoadI32 { dst: 7, val: 1 },
                        IrInstr::Call {
                            dst: None,
                            name: "callee".to_string(),
                            args: vec![7],
                        },
                        IrInstr::Ret { src: None },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit");
        verify_semcode(&bytes).expect("call argument r7 is defined");
    }

    #[test]
    fn c1756_case15_accepts_call_destination_defined_after_call() {
        let bytes = emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "callee".to_string(),
                    instrs: vec![
                        IrInstr::LoadI32 { dst: 0, val: 1 },
                        IrInstr::Ret { src: Some(0) },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: vec![
                        IrInstr::Call {
                            dst: Some(0),
                            name: "callee".to_string(),
                            args: vec![],
                        },
                        IrInstr::Ret { src: Some(0) },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit");
        verify_semcode(&bytes).expect("a CALL with a destination defines it on fallthrough");
    }

    #[test]
    fn c1756_case16_rejects_reading_call_without_destination_placeholder() {
        let bytes = emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "callee".to_string(),
                    instrs: vec![IrInstr::Ret { src: None }],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: vec![
                        IrInstr::Call {
                            dst: None,
                            name: "callee".to_string(),
                            args: vec![],
                        },
                        // r0 was never defined - the no-dst CALL's encoded
                        // dummy destination must not have defined it.
                        IrInstr::Ret { src: Some(0) },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit");
        let report = verify_semcode(&bytes)
            .expect_err("no-dst CALL must not define its encoded placeholder register");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    #[test]
    fn c1756_case17_rejects_undefined_return_source() {
        let bytes = emit_test_function_with_params(
            vec![CallableValueFamily::I32, CallableValueFamily::I32],
            vec![IrInstr::Ret { src: Some(5) }],
        );
        let report = verify_semcode(&bytes)
            .expect_err("return source r5 is unrelated to either declared parameter");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    /// Case 18 is distinct from case 15: it specifically proves that
    /// `Value::Unit` is an ordinary defined value for this pass's purposes,
    /// not an "undefined register" sentinel - the callee explicitly returns
    /// `Unit` (`Ret { src: None }`), and reading the caller's destination
    /// register afterward must still be accepted purely because the
    /// register was *written*, independent of what runtime value it holds.
    #[test]
    fn c1756_case18_accepts_reading_register_defined_by_unit_returning_call() {
        let bytes = emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "returns_unit".to_string(),
                    instrs: vec![IrInstr::Ret { src: None }],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
                IrFunction {
                    name: "main".to_string(),
                    instrs: vec![
                        IrInstr::Call {
                            dst: Some(0),
                            name: "returns_unit".to_string(),
                            args: vec![],
                        },
                        IrInstr::Ret { src: Some(0) },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit");
        verify_semcode(&bytes)
            .expect("r0 is defined by the call regardless of its runtime value being Unit");
    }

    #[test]
    fn c1756_case19_accepts_unit_parameter_entry_read() {
        let bytes = emit_test_function_with_params(
            vec![CallableValueFamily::Unit],
            vec![IrInstr::Ret { src: Some(0) }],
        );
        verify_semcode(&bytes)
            .expect("r0 is entry-defined even though its declared family is Unit");
    }

    #[test]
    fn c1756_case20_rejects_map_get_default_undefined() {
        let bytes = emit_test_function(vec![
            IrInstr::MapEmpty { dst: 0 },
            IrInstr::LoadI32 { dst: 1, val: 1 },
            IrInstr::MapGet {
                dst: 2,
                map: 0,
                key: 1,
                default_val: 9, // never written anywhere
            },
            IrInstr::Ret { src: Some(2) },
        ]);
        let report = verify_semcode(&bytes).expect_err(
            "MAP_GET's default register is a conservative, unconditional read (#1756) - key \
             presence is never statically proven, matching the runtime laziness rule (#1771) \
             being distinct from this static proof",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    #[test]
    fn c1756_case21_accepts_map_get_default_defined() {
        let bytes = emit_test_function(vec![
            IrInstr::MapEmpty { dst: 0 },
            IrInstr::LoadI32 { dst: 1, val: 1 },
            IrInstr::LoadI32 { dst: 9, val: 0 },
            IrInstr::MapGet {
                dst: 2,
                map: 0,
                key: 1,
                default_val: 9,
            },
            IrInstr::Ret { src: Some(2) },
        ]);
        verify_semcode(&bytes).expect("MAP_GET's default register is defined");
    }

    #[test]
    fn c1756_case22_rejects_map_get_map_source_undefined() {
        let bytes = emit_test_function(vec![
            IrInstr::LoadI32 { dst: 1, val: 1 },
            IrInstr::LoadI32 { dst: 9, val: 0 },
            IrInstr::MapGet {
                dst: 2,
                map: 0, // never written
                key: 1,
                default_val: 9,
            },
            IrInstr::Ret { src: Some(2) },
        ]);
        let report = verify_semcode(&bytes).expect_err("MAP_GET's map register is undefined");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    #[test]
    fn c1756_case23_rejects_conditional_branch_condition_undefined() {
        let bytes = emit_test_function(vec![
            IrInstr::JmpIf {
                cond: 0, // never written
                label: "target".to_string(),
            },
            IrInstr::Label {
                name: "target".to_string(),
            },
            IrInstr::Ret { src: None },
        ]);
        let report = verify_semcode(&bytes).expect_err("branch condition r0 is undefined");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    #[test]
    fn c1756_case24_ignores_undefined_read_in_unreachable_code() {
        let bytes = emit_test_function(vec![
            IrInstr::Ret { src: None },
            // Unreachable (RET has no successor). r99 is undefined, but
            // this pass only judges reachable reads, matching the
            // verifier's existing, separate policy on structurally valid
            // but unreachable code (see `verifier_accepts_structurally_
            // closed_infinite_loop`'s trailing-code precedent).
            IrInstr::AddI32 {
                dst: 1,
                lhs: 99,
                rhs: 99,
            },
        ]);
        verify_semcode(&bytes)
            .expect("an undefined read in unreachable code must not be judged by this pass");
    }

    /// Codex review round 1 on PR #1840: the original implementation
    /// allocated one `domain_size`-sized `RegSet` per DECODED instruction,
    /// reachable or not - so a structurally valid artifact consisting of an
    /// entry `RET` followed by many unreachable instructions referencing a
    /// high register number could force memory proportional to
    /// `total_instruction_count * domain_size`, with no existing quota
    /// bounding total instruction count. Fixed by allocating dataflow state
    /// only for reachable nodes. This regression exercises exactly that
    /// shape (reachable `RET` immediately, then many unreachable
    /// instructions touching a near-budget register) and proves it still
    /// verifies correctly - a real functional check on the fixed code path,
    /// not just a memory-bound argument.
    #[test]
    fn c1756_rejects_unreachable_bloat_stays_cheap_and_correct() {
        let mut instrs = vec![IrInstr::Ret { src: None }];
        for _ in 0..500 {
            instrs.push(IrInstr::AddI32 {
                dst: 4000,
                lhs: 4000,
                rhs: 4000,
            });
        }
        let bytes = emit_test_function(instrs);
        verify_semcode(&bytes).expect(
            "hundreds of unreachable instructions referencing a high register must not be \
             judged by this pass, and must not meaningfully slow or bloat verification",
        );
    }

    /// Codex review round 3 on PR #1840: even after round 1's reachable-only
    /// `RegSet` allocation fix, `verify_function_code`'s main decode loop
    /// still built a dense `Vec<Vec<u16>>` of reads/writes for every
    /// structurally decoded instruction, reachable or not - an empty
    /// `Vec<u16>` costs 24 bytes even with no heap allocation, so a
    /// signature-bearing function padded with a large number of cheap
    /// unreachable instructions (their own example: an entry `RET` followed
    /// by megabytes of two-byte `RET`s) still cost hundreds of MB for these
    /// two arrays alone, despite only one reachable node. Fixed by no longer
    /// building them at all in the main decode loop; `prove_definite_
    /// register_assignment` re-decodes only the instructions `reachable_
    /// offsets` actually names. 50,000 unreachable `RET`s (100 KB of dead
    /// code) is enough to prove this stays fast and correct without
    /// literally allocating gigabytes in a unit test.
    #[test]
    fn c1756_rejects_large_unreachable_ret_padding_without_dense_metadata() {
        let mut instrs = vec![IrInstr::Ret { src: None }];
        for _ in 0..50_000 {
            instrs.push(IrInstr::Ret { src: None });
        }
        let bytes = emit_test_function(instrs);
        verify_semcode(&bytes).expect(
            "tens of thousands of unreachable two-byte RETs must not require per-instruction \
             reads/writes metadata to be materialized for every one of them",
        );
    }

    /// #1756 Codex review round 17 (owner decision): Codex's own exact
    /// P2 shape - an entry `RET` (the only reachable instruction)
    /// followed by a large amount of structurally valid but UNREACHABLE
    /// `RET` padding (the same construction `c1756_rejects_large_
    /// unreachable_ret_padding_without_dense_metadata` already uses for
    /// a different reason - this pass never even attempts to judge
    /// unreachable code). Proves `reachable_count` stays 1 and the raw-
    /// state requirement stays fixed at 17 words (4 raw + 13 leader-
    /// state, both independently measured) no matter how much
    /// unreachable padding follows - preserving the verifier's pre-
    /// existing, deliberate "unreachable code is not judged by this
    /// pass" admission policy that round 16's `instr_len`-based proxy
    /// would otherwise have broken by charging the WHOLE structural
    /// instruction section, unreachable padding included. Also serves as
    /// the review's requested direct accounting comparison: padding = 0
    /// and padding = 1,000,000 must report the IDENTICAL exact word
    /// count, proving `reachable_instruction_bytes` genuinely does not
    /// grow with unreachable padding size.
    #[test]
    fn c1756_unreachable_padding_does_not_inflate_raw_state() {
        fn padded(padding: usize) -> Vec<u8> {
            let mut instrs = vec![IrInstr::Ret { src: None }];
            for _ in 0..padding {
                instrs.push(IrInstr::Ret { src: None });
            }
            emit_test_function(instrs)
        }

        for padding in [0usize, 1_000_000] {
            let bytes = padded(padding);

            // usage == limit (17 words) must be accepted regardless of
            // padding size.
            verify_semcode_token_with_quotas_and_limits(
                &bytes,
                RuntimeQuotas::verified_local(),
                VerificationLimits {
                    max_state_words: 17,
                    max_work_units: usize::MAX,
                },
            )
            .unwrap_or_else(|report| {
                panic!(
                    "padding={padding}: unreachable padding must not inflate the raw-state \
                     requirement past the single reachable RET's own cost: {report:?}"
                )
            });

            // usage == limit + 1 must still reject deterministically,
            // reporting the SAME exact word count regardless of padding.
            let report = verify_semcode_token_with_quotas_and_limits(
                &bytes,
                RuntimeQuotas::verified_local(),
                VerificationLimits {
                    max_state_words: 16,
                    max_work_units: usize::MAX,
                },
            )
            .expect_err("usage == limit + 1 must still reject deterministically");
            assert_eq!(
                report.diagnostics[0].code,
                VerificationCode::AnalysisStateLimitExceeded
            );
            assert!(
                report.diagnostics[0]
                    .message
                    .contains("17 logical verifier-analysis word"),
                "padding={padding}: exact required word count must stay 17 regardless of \
                 unreachable padding size, got: {}",
                report.diagnostics[0].message
            );
            assert!(
                report.diagnostics[0].message.contains("1 reachable node"),
                "padding={padding}: reachable_count must stay 1 regardless of unreachable \
                 padding size"
            );
        }
    }

    /// #1756 Codex review round 18 (owner decision): Codex's own exact P1
    /// shape - a reachable entry `RET` followed by millions of
    /// structurally valid but UNREACHABLE `LOAD_BOOL` instructions. Round
    /// 17 already proved unreachable padding cannot inflate the #1756
    /// resource-envelope's WORD accounting; this proves the earlier,
    /// SHARED structural decode pass (`verify_function_code`'s first walk,
    /// which every function pays regardless of signature, and which
    /// neither `max_state_words` nor `max_work_units` can see, since both
    /// are #1756-analysis-only budgets scoped to reachable state) does not
    /// pay one heap allocation per unreachable instruction either - the
    /// actual CPU/allocator-churn cost Codex's round-18 finding
    /// identified, one layer beneath what rounds 16-17 could bound at
    /// all. Uses genuine allocator instrumentation (`alloc_tracking`, a
    /// `System`-delegating counting `#[global_allocator]` active only in
    /// `cfg(test)` builds), not wall-clock timing, per the review's own
    /// explicit preference.
    #[test]
    fn c1756_structural_decode_never_allocates_operand_storage_for_unreachable_padding() {
        let mut instrs = vec![IrInstr::Ret { src: None }];
        for _ in 0..2_000_000 {
            instrs.push(IrInstr::LoadBool { dst: 0, val: true });
        }
        let bytes = emit_test_function(instrs);

        let before = alloc_tracking_calls();
        verify_semcode(&bytes)
            .expect("unreachable code is not judged by this pass and must still admit");
        let after = alloc_tracking_calls();

        // Every OTHER allocation this call legitimately makes (instr_
        // starts/instruction_successors/jump_targets/etc. growing via
        // amortized Vec doubling, HashSet growth for reachable_offsets,
        // the #1756 analysis's own O(1)-scaled structures for the single
        // reachable node) is bounded by O(log(instruction_count)) per
        // growing container, not O(instruction_count) - comfortably under
        // 1,000 total calls even at 2,000,001 instructions. Under the
        // pre-round-18 behavior, `writes.push(dst)` alone would have
        // triggered one allocation PER unreachable LOAD_BOOL (an empty
        // `Vec<u16>` allocates on its first push), i.e. ~2,000,000 - a
        // three-order-of-magnitude gap this threshold cannot miss.
        let allocations_during_verify = after.saturating_sub(before);
        assert!(
            allocations_during_verify < 1_000,
            "structural decode must not allocate operand storage per unreachable instruction - \
             saw {allocations_during_verify} allocator calls while verifying 2,000,000 \
             unreachable LOAD_BOOL instructions, expected a small O(log n) constant, not \
             one-or-more per instruction"
        );
    }

    /// #1756 Codex review round 21 (owner decision): Codex's own exact P1
    /// shape - a REACHABLE stream of zero-argument, no-destination `CALL`
    /// instructions to a real callee. Round 18 already proved the shared
    /// structural pass doesn't allocate per-instruction for `reads`/
    /// `writes`; this proves the SEPARATE reachable-only re-decode pass
    /// (which every reachable `CALL` also goes through, to obtain its
    /// register reads/writes for the definite-assignment CSR) does not
    /// pay a fresh `string_refs`/`call_argcs` allocation per call either,
    /// for metadata that re-decode pass never reads at all (it discards
    /// the whole `OperandRefs` via `.expect(...)`, keeping only what its
    /// `FlatSink` captured).
    #[test]
    fn c1756_reachable_redecode_never_allocates_metadata_for_ignored_fields() {
        let mut main_instrs = Vec::new();
        for _ in 0..2_000_000 {
            main_instrs.push(IrInstr::Call {
                dst: None,
                name: "helper".to_string(),
                args: Vec::new(),
            });
        }
        main_instrs.push(IrInstr::Ret { src: None });
        let bytes = emit_test_function_with_helper(main_instrs);

        // Explicit, generous limits - this test's own concern is decode-
        // time allocation behavior, not resource-limit correctness, so
        // it deliberately does not depend on wherever the candidate
        // default `max_state_words`/`max_work_units` happen to sit.
        let before = alloc_tracking_calls();
        verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: usize::MAX,
                max_work_units: usize::MAX,
            },
        )
        .expect("2,000,000 reachable no-op calls, zero-register domain, must still admit cheaply");
        let after = alloc_tracking_calls();

        // Unlike the round-18/round-20 allocation tests, this construction
        // has a genuine, UNRELATED, pre-existing O(reachable_count)
        // allocation baseline that is NOT part of round 21's fix and is
        // out of scope for it: `verify_function_code`'s own `call_
        // targets` construction (`env.strings[sid].clone()`, once per
        // "call target" `string_refs` entry) clones the callee name
        // string once per reachable CALL - unavoidable with real CALL
        // instructions, regardless of decode_operands's own behavior.
        // Measured directly: ~2,000,188 allocator calls with this
        // round's fix applied, matching that baseline almost exactly
        // (2,000,000 clones + a small O(log n) margin from every OTHER
        // growing container). The threshold below is generous relative
        // to that TRUE baseline, while remaining far below what
        // reverting round 21's fix would add on top: the reachable re-
        // decode pass would then ALSO allocate `string_refs` and `call_
        // argcs` storage per reachable CALL (each field's first push on
        // a freshly `Default::default()`d `OperandRefs` allocates),
        // roughly DOUBLING or more the total - a clear, discriminating
        // gap this threshold is chosen to catch (see the mutation
        // evidence in the PR reply for the exact reverted-order count).
        let allocations_during_verify = after.saturating_sub(before);
        assert!(
            allocations_during_verify < 2_200_000,
            "the reachable re-decode pass must not allocate string_refs/call_argcs storage per \
             reachable CALL on TOP of the unrelated call_targets string-cloning baseline - saw \
             {allocations_during_verify} allocator calls while verifying 2,000,000 reachable \
             no-op calls, expected close to the ~2,000,188-call baseline, not roughly double it"
        );
    }

    /// #1756 Codex review round 17's own requested "control test": the
    /// SAME long instruction stream, made genuinely REACHABLE instead of
    /// unreachable (no branch-free-but-unreachable padding - see
    /// `c1756_long_linear_low_domain_rejects_on_raw_state_not_leader_
    /// state`, unchanged by this round since it has no unreachable code
    /// at all, so `reachable_instruction_bytes == instr_len` for it
    /// exactly) still drives the raw-state requirement up and can still
    /// be rejected - proving round 17 narrowed accounting to genuinely
    /// REACHABLE bytes, not accidentally made instruction bytes free in
    /// general. That existing test's exact numbers were originally
    /// 312,508 raw-only / 312,528 combined; round 21's own P2 fix added
    /// a `leader_classification_state` term to the raw formula (charging
    /// `compute_leaders`'s consolidated scratch array), raising them to
    /// 325,008 raw-only / 325,028 combined - the STRUCTURAL claim this
    /// test makes (reachable-only byte scoping) is unaffected by that
    /// change and re-verified unchanged by this round's own full c1756
    /// suite run; only the unrelated numeric total moved.
    #[test]
    fn c1756_reachable_instruction_bytes_counts_only_reachable_spans() {
        // Five structural instructions at offsets 0, 4, 8, 20, 24; the
        // instruction section ends at offset 28 (the fifth instruction
        // is 4 bytes). Reachable: 0 (len 4), 4 (len 4), 24 (len 4, to
        // the section end) - 8 and 20 are unreachable and must not
        // count.
        let instr_starts = vec![0usize, 4, 8, 20, 24];
        let instr_len = 28usize;
        let mut reachable = HashSet::new();
        reachable.insert(0usize);
        reachable.insert(4usize);
        reachable.insert(24usize);

        assert_eq!(
            reachable_instruction_bytes(&instr_starts, &reachable, instr_len),
            Some(12),
            "must sum only the three reachable spans (4+4+4=12), ignoring the two \
             unreachable instructions at offsets 8 and 20 entirely"
        );
    }

    /// Codex review round 4 on PR #1840: even after round 3 stopped
    /// materializing metadata for unreachable instructions, a large fully
    /// REACHABLE straight-line stream (their example: millions of
    /// `LOAD_BOOL r0` in a row, no branching, no unreachable padding at
    /// all) still cost dozens of bytes of `Vec<Vec<u16>>`/`Vec<Vec<usize>>`
    /// heap-container overhead per instruction, from the reads/writes
    /// arrays AND the predecessor/successor lists. Fixed by CSR-flattening
    /// reads/writes (`dataflow_domain_accounting`) and eliminating the
    /// predecessor structure entirely in favor of edge relaxation over the
    /// verifier's own `instruction_successors` (`for_each_reachable_
    /// successor`). 300,000 reachable `LOAD_BOOL r0` instructions - every
    /// one of them a repeated write to the same already-defined register,
    /// so `Rc`-sharing (round 2) also keeps every `RegSet` after the first
    /// shared - is enough to prove this stays fast and correct without
    /// literally constructing a multi-million-instruction artifact in a
    /// unit test.
    #[test]
    fn c1756_accepts_large_fully_reachable_linear_stream() {
        let mut instrs: Vec<IrInstr> = Vec::new();
        for _ in 0..300_000 {
            instrs.push(IrInstr::LoadBool { dst: 0, val: true });
        }
        instrs.push(IrInstr::Ret { src: Some(0) });
        let bytes = emit_test_function(instrs);
        verify_semcode(&bytes).expect(
            "a long, fully reachable, non-branching stream that repeatedly writes the same \
             already-defined register must verify correctly and stay fast",
        );
    }

    /// Codex review round 5 on PR #1840: reproduces the exact adversarial
    /// shape described - a wide dispatch where every arm defines all but
    /// one (a DIFFERENT one per arm) of a shared register domain, joining
    /// into a single node, followed by a long fallthrough tail. Under a
    /// plain FIFO edge-relaxation worklist, each of the `ARMS` arms
    /// narrows the join's `IN` by exactly one more bit (regardless of
    /// processing order, since it's a strict, growing set difference), and
    /// every such narrowing that changes the join's `OUT` re-propagates
    /// through the *entire* `TAIL_LEN`-node tail again -
    /// `O(ARMS * TAIL_LEN)` relaxations. Fixed by processing positions in
    /// `reverse_postorder_ranks` order (see that function's doc comment):
    /// every arm is ordered ahead of the join it feeds, so all `ARMS`
    /// contributions have already narrowed `in_sets[join]` in place by the
    /// time the join is first popped and actually processed - the tail is
    /// walked once, not once per arm.
    ///
    /// Correctness is exercised too, not just performance: since arm `i`
    /// defines every register in `0..ARMS` except register `i`, the
    /// intersection over all arms is the EMPTY set within that domain - no
    /// register in `0..ARMS` survives the join, so the read at the far end
    /// of the tail (`r0`) is correctly rejected as undefined, regardless of
    /// how long the intervening tail is.
    #[test]
    fn c1756_rejects_wide_dispatch_join_feeding_long_tail_without_per_arm_blowup() {
        const ARMS: u16 = 128;
        const TAIL_LEN: usize = 20_000;
        const COND_REG: u16 = ARMS; // outside the 0..ARMS join domain

        let mut instrs: Vec<IrInstr> = vec![IrInstr::LoadBool {
            dst: COND_REG,
            val: true,
        }];
        for arm in 0..ARMS - 1 {
            instrs.push(IrInstr::JmpIf {
                cond: COND_REG,
                label: format!("arm{arm}"),
            });
        }
        // Fallthrough arm (ARMS - 1): reached only once every JmpIf above
        // has fallen through, so its body must be the next physical
        // instruction after the dispatch chain - defines every register
        // except r(ARMS - 1).
        for r in 0..ARMS {
            if r != ARMS - 1 {
                instrs.push(IrInstr::LoadI32 {
                    dst: r,
                    val: r as i32,
                });
            }
        }
        instrs.push(IrInstr::Jmp {
            label: "join".to_string(),
        });
        // The remaining ARMS - 1 arms, each reached by its own JmpIf target
        // above - defines every register except its own index.
        for arm in 0..ARMS - 1 {
            instrs.push(IrInstr::Label {
                name: format!("arm{arm}"),
            });
            for r in 0..ARMS {
                if r != arm {
                    instrs.push(IrInstr::LoadI32 {
                        dst: r,
                        val: r as i32,
                    });
                }
            }
            instrs.push(IrInstr::Jmp {
                label: "join".to_string(),
            });
        }
        instrs.push(IrInstr::Label {
            name: "join".to_string(),
        });
        for _ in 0..TAIL_LEN {
            // Writes a register outside the 0..ARMS join domain - carries
            // the join's (already-converged) state forward without adding
            // new information, the same "cheap tail" shape as the other
            // large stress regressions above.
            instrs.push(IrInstr::LoadI32 {
                dst: COND_REG,
                val: 0,
            });
        }
        instrs.push(IrInstr::Ret { src: Some(0) }); // r0: undefined by arm 0, so missing at the join

        let bytes = emit_test_function(instrs);
        let report = verify_semcode(&bytes).expect_err(
            "r0 is missing from arm 0's definitions, so no register in 0..ARMS survives the \
             128-arm join, and the tail read of r0 must be rejected",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    /// Codex review round 6 on PR #1840, second finding (a sibling to the
    /// frame-size one above): a loop header `H` fed by `ARMS` sibling
    /// nodes `S_0..S_{k-1}` via back edges, each defining every register
    /// in `0..ARMS` except a different one, `H` itself then feeding a
    /// long exit tail - `dispatch` reaches every `S_i` (each writes,
    /// jumps to `H`), and `H`'s own `JmpIf` back to `dispatch` closes the
    /// cycle (`dispatch` is `H`'s DFS ancestor - `H` is discovered only
    /// as some `S_i`'s descendant - so this is a genuine back edge; `H`
    /// and every `S_i` that participates in the cycle land in ONE SCC,
    /// mutually reachable through it). `H` has NO predecessor besides the
    /// arms' back edges, so it starts at TOP (the standard non-entry
    /// initialization) and needs a genuine, necessary contribution from
    /// ALL `ARMS` arms - not something a single predecessor already
    /// determines - before its fixed point is settled; `dispatch` itself
    /// starts empty (straight from entry, no SIG0 params), so each arm's
    /// own exclusion write is what determines its output, not something
    /// inherited from a prior pass.
    ///
    /// Correctness: since `S_i` defines every register in `0..ARMS`
    /// except `i`, the intersection over all `ARMS` arms at `H` is the
    /// EMPTY set within that domain (the same argument as the round-5
    /// test) - the read at the far end of the tail (`r0`) is correctly
    /// rejected as undefined, regardless of the intervening loop or tail
    /// length. This is the primary purpose of this regression: proving
    /// the SCC-based worklist converges a genuine, multi-source cycle to
    /// the mathematically correct fixed point. It is not, on its own,
    /// strong TIMING evidence against round 5's pure-RPO ordering: in
    /// this specific instruction layout, `dispatch` (and therefore every
    /// arm) is discovered by DFS before `H` even exists, so plain RPO
    /// rank already happens to order every arm ahead of `H` here too -
    /// the round-6 reply documents this honestly, together with the
    /// general, well-established argument for why SCC-restricted
    /// processing bounds total work for the broader class of CFGs RPO
    /// alone cannot.
    #[test]
    fn c1756_rejects_loop_header_fed_by_many_backedge_arms_without_per_arm_blowup() {
        const ARMS: u16 = 128;
        const TAIL_LEN: usize = 20_000;
        const COND_REG: u16 = ARMS;

        // `dispatch`'s own IN comes straight from entry (empty - no SIG0
        // params), so it - and every arm reached only through it - starts
        // genuinely empty, not full: each arm's "define everything except
        // my own index" write is what actually determines its output,
        // not something inherited from a prior loop pass. `header` has NO
        // direct edge from entry or dispatch - its only predecessors are
        // the arms' back edges - so it starts at TOP (the standard
        // non-entry initialization) and is narrowed ONE MEANINGFUL STEP
        // PER ARM as their outputs arrive, needing all `ARMS` of them
        // before its fixed point is fully determined - a genuine,
        // necessary multi-round convergence, not a value some single
        // predecessor already fixes on its own.
        let mut instrs: Vec<IrInstr> = vec![IrInstr::LoadBool {
            dst: COND_REG,
            val: true,
        }];
        instrs.push(IrInstr::Label {
            name: "dispatch".to_string(),
        });
        for arm in 0..ARMS - 1 {
            instrs.push(IrInstr::JmpIf {
                cond: COND_REG,
                label: format!("arm{arm}"),
            });
        }
        // Fallthrough arm (ARMS - 1): defines every register except
        // itself, then jumps to the header - closing the loop, since
        // header's own back edge (below) returns here.
        for r in 0..ARMS {
            if r != ARMS - 1 {
                instrs.push(IrInstr::LoadI32 {
                    dst: r,
                    val: r as i32,
                });
            }
        }
        instrs.push(IrInstr::Jmp {
            label: "header".to_string(),
        });
        // The remaining ARMS - 1 arms, each reached by its own JmpIf target
        // above - defines every register except its own index, then jumps
        // to the header too.
        for arm in 0..ARMS - 1 {
            instrs.push(IrInstr::Label {
                name: format!("arm{arm}"),
            });
            for r in 0..ARMS {
                if r != arm {
                    instrs.push(IrInstr::LoadI32 {
                        dst: r,
                        val: r as i32,
                    });
                }
            }
            instrs.push(IrInstr::Jmp {
                label: "header".to_string(),
            });
        }
        instrs.push(IrInstr::Label {
            name: "header".to_string(),
        });
        // The back edge: `dispatch` is `header`'s DFS ancestor here (every
        // arm was discovered as dispatch's descendant, and each leads to
        // header), so this is a genuine back edge, not a forward join -
        // `header` and every arm form one strongly-connected component.
        instrs.push(IrInstr::JmpIf {
            cond: COND_REG,
            label: "dispatch".to_string(),
        });
        // Exit tail: header's fallthrough - reached only once the loop
        // actually exits, with header's fully-converged fixed point.
        for _ in 0..TAIL_LEN {
            instrs.push(IrInstr::LoadI32 {
                dst: COND_REG,
                val: 0,
            });
        }
        instrs.push(IrInstr::Ret { src: Some(0) }); // r0: undefined by arm 0

        let bytes = emit_test_function(instrs);
        let report = verify_semcode(&bytes).expect_err(
            "r0 is missing from arm 0's definitions, so no register in 0..ARMS survives the \
             header's intersection over all back-edge arms, and the tail read of r0 must be \
             rejected",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    /// Codex review round 7 on PR #1840: reproduces the "bound convergence
    /// work inside each SCC" shape - one big cycle `c_0 -> c_1 -> ... ->
    /// c_{CHAIN_LEN-1} -> back to c_0`, laid out in REVERSE physical order
    /// (`c_{CHAIN_LEN-1}`'s bytes come first, `c_0`'s come last - the exact
    /// technique `c1756_backward_propagating_chain_still_rejects_and_
    /// converges_promptly` already uses for the whole-graph case), where
    /// the first `ARMS` cycle nodes (`c_0..c_{ARMS-1}`) each also have
    /// their own external arm `arm_i` (reached from a shared dispatch off
    /// entry) contributing "every register except r_i" within the `0..
    /// ARMS` domain - `arm_i`'s own write cost is bounded by `ARMS`, not
    /// `CHAIN_LEN`, so the ring can be made long independently of how many
    /// distinct exclusions feed it. `c_i`'s true fixed point (for `i <
    /// ARMS`) is the intersection of `arm_0..arm_i`'s contributions -
    /// genuinely cumulative along the ring, not something any single arm
    /// already determines. A plain FIFO seeded in ascending-position (=
    /// descending logical-index, per the reversed layout) order only
    /// advances newly-arriving information one hop per pass against that
    /// mismatch - round 5's exact ordering hazard, reproduced entirely
    /// INSIDE one SCC where round 6's cross-SCC isolation cannot help.
    /// Fixed by `local_reverse_postorder_ranks`: the same RPO-ordering
    /// technique round 5 applied to the whole graph, now applied to each
    /// SCC's own internal convergence.
    ///
    /// Correctness: the converged fixed point at the far end of the ring
    /// is the intersection of every arm's contribution - the empty set
    /// within `0..ARMS`, since `arm_i` excludes `r_i` for every `i` - so
    /// the tail read of `r0` is correctly rejected regardless of ring
    /// length or physical layout.
    #[test]
    fn c1756_rejects_reverse_physical_cycle_with_many_external_arms_without_per_arm_blowup() {
        const ARMS: u16 = 64;
        const CHAIN_LEN: u32 = 40_000;
        const COND_REG: u16 = ARMS;
        const TAIL_LEN: usize = 20_000;

        let mut instrs: Vec<IrInstr> = vec![IrInstr::LoadBool {
            dst: COND_REG,
            val: true,
        }];
        instrs.push(IrInstr::Label {
            name: "dispatch".to_string(),
        });
        for arm in 0..ARMS - 1 {
            instrs.push(IrInstr::JmpIf {
                cond: COND_REG,
                label: format!("arm{arm}"),
            });
        }
        // Fallthrough arm (ARMS - 1): defines every register in 0..ARMS
        // except itself, then jumps into the ring at c_{ARMS - 1}.
        for r in 0..ARMS {
            if r != ARMS - 1 {
                instrs.push(IrInstr::LoadI32 {
                    dst: r,
                    val: r as i32,
                });
            }
        }
        instrs.push(IrInstr::Jmp {
            label: format!("c{}", ARMS - 1),
        });
        // The remaining arms, each reached by its own JmpIf target above -
        // defines every register in 0..ARMS except its own index, then
        // jumps into the ring at the matching c_i.
        for arm in 0..ARMS - 1 {
            instrs.push(IrInstr::Label {
                name: format!("arm{arm}"),
            });
            for r in 0..ARMS {
                if r != arm {
                    instrs.push(IrInstr::LoadI32 {
                        dst: r,
                        val: r as i32,
                    });
                }
            }
            instrs.push(IrInstr::Jmp {
                label: format!("c{arm}"),
            });
        }
        // c_{CHAIN_LEN - 1}: the ring's last logical node - back edge to
        // c_0 (closing the loop) or fallthrough to the exit tail. Emitted
        // FIRST physically (lowest position), even though it's LAST in
        // logical/control-flow order.
        instrs.push(IrInstr::Label {
            name: format!("c{}", CHAIN_LEN - 1),
        });
        instrs.push(IrInstr::JmpIf {
            cond: COND_REG,
            label: "c0".to_string(),
        });
        for _ in 0..TAIL_LEN {
            instrs.push(IrInstr::LoadI32 {
                dst: COND_REG,
                val: 0,
            });
        }
        instrs.push(IrInstr::Ret { src: Some(0) }); // r0: undefined by arm 0
                                                    // c_{CHAIN_LEN - 2} down to c_0: each jumps to its logical
                                                    // successor - emitted in DESCENDING index order, so ascending
                                                    // physical position is the exact reverse of logical/control-flow
                                                    // order (c_{CHAIN_LEN - 1} lowest position .. c_0 highest).
                                                    // Positions ARMS..CHAIN_LEN-1 have no external arm at all - pure
                                                    // pass-through, carrying the already-converging state forward
                                                    // without adding new information - so the ring's own length can
                                                    // grow far beyond ARMS at negligible construction cost.
        for i in (0..CHAIN_LEN - 1).rev() {
            instrs.push(IrInstr::Label {
                name: format!("c{i}"),
            });
            instrs.push(IrInstr::Jmp {
                label: format!("c{}", i + 1),
            });
        }

        let bytes = emit_test_function(instrs);
        let report = verify_semcode(&bytes).expect_err(
            "each arm_i excludes r_i, so the intersection over all ARMS arms at the far end \
             of the ring is empty within 0..ARMS, and the tail read of r0 must be rejected",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    /// Codex review round 8 on PR #1840, first finding ("duplicate live
    /// RegSet states across branches"): many branch arms each performing
    /// the IDENTICAL state-changing write would, under every pre-round-8
    /// scheme (`Rc`-sharing included - `Rc`-sharing only ever avoided
    /// cloning along a *shared* predecessor path, it never interned
    /// independently-computed-but-equal values from *different* branches),
    /// each independently allocate an equal, duplicate `RegSet`. Under the
    /// round-8 MAY-missing formulation this class of duplication is
    /// structurally impossible: `compute_missing_sets` returns exactly one
    /// `RegSet` per reachable position - `missing.len() == reachable_count`
    /// always, by construction of the `Vec` it returns - mutated in place
    /// via `deliver`, with no clone-on-write path for any number of
    /// branches to trigger. This test uses `compute_missing_sets` directly
    /// (bypassing full SemCode construction, per Codex's own request for
    /// accounting evidence over timing) at Codex's stated scale
    /// (`|U|` = 4096) with 2,000 arms, all writing the identical register -
    /// direct accounting evidence that `event_count` scales with `arms +
    /// domain_size`, never with `arms * domain_size` or any product
    /// involving how many arms compute the same result.
    #[test]
    fn c1756_many_identical_branch_arms_allocate_exactly_one_regset_per_position() {
        const DOMAIN_SIZE: usize = 4096;
        const ARMS: usize = 2_000;
        // Positions: 0 = entry; 1..=ARMS = a dispatch chain (position i's
        // successors are [arm i's body, dispatch chain continues]); ARMS+1
        // ..=2*ARMS = arm bodies (every one writes bit 0, and ONLY bit 0);
        // 2*ARMS+1 = join (reached by every arm).
        let dispatch_start = 1usize;
        let arm_start = ARMS + 1;
        let join = 2 * ARMS + 1;
        let reachable_count = join + 1;

        let successors_of = |pos: usize| -> [Option<usize>; 2] {
            if pos == 0 {
                [Some(dispatch_start), None]
            } else if (dispatch_start..arm_start).contains(&pos) {
                let i = pos - dispatch_start;
                let next_dispatch = if i + 1 < ARMS {
                    Some(dispatch_start + i + 1)
                } else {
                    None
                };
                [Some(arm_start + i), next_dispatch]
            } else if (arm_start..join).contains(&pos) {
                [Some(join), None]
            } else {
                [None, None]
            }
        };
        // Every arm body writes bit 0 - the identical write, in every arm.
        let writes_contains = |pos: usize, bit: usize| (arm_start..join).contains(&pos) && bit == 0;

        let entry_missing = RegSet::full(DOMAIN_SIZE); // no SIG0 params: nothing defined at entry
        let (missing, event_count, peak_queue_len) = compute_missing_sets(
            reachable_count,
            DOMAIN_SIZE,
            entry_missing,
            writes_contains,
            successors_of,
            &mut AnalysisWorkMeter::new(usize::MAX),
        )
        .expect("test-only call: no verification-resource budget under test here");
        assert_eq!(
            missing.len(),
            reachable_count,
            "exactly one RegSet per reachable position, always - the structural property that \
             makes duplicate live states impossible regardless of arm count"
        );
        assert!(
            !missing[join].contains(0),
            "bit 0 is written by every one of the 2,000 arms, so it must be defined (not \
             missing) at the join"
        );
        assert!(
            missing[join].contains(1),
            "bit 1 is written by no arm, so it must still be missing at the join"
        );
        assert!(
            event_count <= reachable_count * DOMAIN_SIZE,
            "event_count ({event_count}) must be bounded by reachable_count * domain_size \
             ({}), never multiplied further by how many arms compute the identical result",
            reachable_count * DOMAIN_SIZE
        );
        assert!(
            peak_queue_len <= reachable_count * DOMAIN_SIZE,
            "peak_queue_len ({peak_queue_len}) must never exceed the total-work bound either"
        );
    }

    /// Codex review round 8 on PR #1840, second finding ("local RPO still
    /// reprocesses large SCC via backedges"): reproduces the same
    /// adversarial shape as `c1756_rejects_reverse_physical_cycle_with_
    /// many_external_arms_without_per_arm_blowup` - a ring laid out in
    /// reverse propagation order, fed at many distinct points by external
    /// arms each contributing a different missing register - directly via
    /// `compute_missing_sets`, asserting `event_count` explicitly (Codex's
    /// own request for a counter, not timing alone). Under the round-7
    /// local-RPO worklist this shape could still force repeated whole-
    /// component reprocessing (each backedge source re-queueing the header
    /// before the others had all contributed); under the round-8 bit-level
    /// worklist there is no "large SCC" concept left to reprocess, so the
    /// event count stays bounded by `reachable_count * domain_size`
    /// regardless of how many external arms feed the cycle or how it is
    /// physically laid out.
    #[test]
    fn c1756_many_external_arms_feeding_one_cycle_via_backedges_bounds_event_count() {
        const ARMS: usize = 200;
        const RING_LEN: usize = 4_000;
        const DOMAIN_SIZE: usize = ARMS;
        // Positions: 0 = entry; 1..=ARMS = a dispatch chain (no writes -
        // dispatch[i]'s successors are [arm i's body, dispatch[i+1]]);
        // ARMS+1..=2*ARMS = arm bodies (arm i writes every register except
        // i, then enters the ring at c_0); 2*ARMS+1..2*ARMS+RING_LEN = the
        // ring c_0..c_{RING_LEN-1}, laid out so position 2*ARMS+1+k
        // corresponds to LOGICAL ring index (RING_LEN-1-k) - reverse
        // propagation order, matching the physically-reversed byte-level
        // test - closing the loop from c_{RING_LEN-1} back to c_0.
        let dispatch_start = 1usize;
        let arm_start = ARMS + 1;
        let ring_start = 2 * ARMS + 1;
        let ring_pos = |logical_i: usize| -> usize { ring_start + (RING_LEN - 1 - logical_i) };
        let reachable_count = ring_start + RING_LEN;

        let successors_of = |pos: usize| -> [Option<usize>; 2] {
            if pos == 0 {
                [Some(dispatch_start), None]
            } else if (dispatch_start..arm_start).contains(&pos) {
                let i = pos - dispatch_start;
                let next_dispatch = if i + 1 < ARMS {
                    Some(dispatch_start + i + 1)
                } else {
                    None
                };
                [Some(arm_start + i), next_dispatch]
            } else if (arm_start..ring_start).contains(&pos) {
                [Some(ring_pos(0)), None]
            } else {
                // A ring position at physical `pos` corresponds to logical
                // index `logical_i` - find it by inverting `ring_pos`.
                let logical_i = RING_LEN - 1 - (pos - ring_start);
                if logical_i + 1 < RING_LEN {
                    [Some(ring_pos(logical_i + 1)), None]
                } else {
                    [Some(ring_pos(0)), None] // c_{RING_LEN-1}: back edge to c_0
                }
            }
        };
        // Arm i writes every register in 0..ARMS except i - reached by
        // every arm body position specifically (not the dispatch chain or
        // the ring, neither of which write anything).
        let writes_contains = |pos: usize, bit: usize| {
            (arm_start..ring_start).contains(&pos) && bit != (pos - arm_start)
        };

        let entry_missing = RegSet::full(DOMAIN_SIZE);
        let (missing, event_count, peak_queue_len) = compute_missing_sets(
            reachable_count,
            DOMAIN_SIZE,
            entry_missing,
            writes_contains,
            successors_of,
            &mut AnalysisWorkMeter::new(usize::MAX),
        )
        .expect("test-only call: no verification-resource budget under test here");
        // Every arm excludes a different register (arm i excludes bit i),
        // so the intersection over all ARMS arms - reflected at every ring
        // position once converged - is empty.
        for bit in 0..ARMS {
            assert!(
                missing[ring_pos(0)].contains(bit),
                "bit {bit} is excluded by arm {bit} specifically, so no register in 0..ARMS \
                 survives the ring's converged fixed point"
            );
        }
        assert!(
            event_count <= reachable_count * DOMAIN_SIZE,
            "event_count ({event_count}) must be bounded by reachable_count * domain_size \
             ({}), regardless of how many external arms feed the cycle via backedges",
            reachable_count * DOMAIN_SIZE
        );
        assert!(
            peak_queue_len <= reachable_count * DOMAIN_SIZE,
            "peak_queue_len ({peak_queue_len}) must never exceed the total-work bound either"
        );
    }

    /// Codex review round 9 on PR #1840: reproduces the exact adversarial
    /// shape described - a zero-parameter function (so `domain_size` = the
    /// full 4,096-register quota, all missing at entry) entering a wide,
    /// balanced binary dispatch tree with `LEAVES` = 32,768 leaves (~65,535
    /// total tree nodes), where nothing anywhere writes anything - every
    /// one of the 4,096 missing registers survives, unkilled, all the way
    /// down to every leaf. Under FIFO order, this accumulates an entire
    /// breadth level - up to `domain_size` bits at EACH of that level's
    /// nodes - before draining any of it, peaking near the full
    /// `reachable_count * domain_size` total-work bound (Codex's own
    /// figure: "about 134 million two-usize tuples, over 2 GiB"). Fixed by
    /// switching `compute_missing_sets`'s worklist from FIFO to LIFO (see
    /// its own doc comment for the full argument): popping `(pos, bit)`
    /// pushes at most 2 new items for that SAME bit, so a bit's delivery
    /// runs to completion - down to a leaf - before any sibling branch is
    /// touched, bounding peak size by the current delivery's own depth
    /// plus a bounded backlog, not by an entire wide level's width times
    /// `domain_size` at once.
    #[test]
    fn c1756_wide_binary_dispatch_tree_bounds_peak_queue_not_just_total_work() {
        const DOMAIN_SIZE: usize = 2_048;
        const LEAVES: usize = 4_096;
        // 1-indexed heap-array tree: position 1 = root, position i's
        // children are 2*i and 2*i+1. Internal nodes are 1..LEAVES
        // (32,767 of them - a complete binary tree with LEAVES leaves has
        // exactly LEAVES-1 internal nodes); leaves are LEAVES..2*LEAVES.
        // Position 0 = entry, whose only successor is the root (position
        // 1) - `reachable_count` = 1 (entry) + (2*LEAVES - 1) (tree).
        let reachable_count = 1 + (2 * LEAVES - 1);

        let successors_of = |pos: usize| -> [Option<usize>; 2] {
            if pos == 0 {
                [Some(1), None]
            } else if pos < LEAVES {
                [Some(2 * pos), Some(2 * pos + 1)]
            } else {
                [None, None] // a leaf
            }
        };
        // Nothing anywhere writes anything - the worst case for how many
        // registers survive, unkilled, all the way to every leaf.
        let writes_contains = |_pos: usize, _bit: usize| false;

        let entry_missing = RegSet::full(DOMAIN_SIZE); // zero params: everything missing
        let (missing, event_count, peak_queue_len) = compute_missing_sets(
            reachable_count,
            DOMAIN_SIZE,
            entry_missing,
            writes_contains,
            successors_of,
            &mut AnalysisWorkMeter::new(usize::MAX),
        )
        .expect("test-only call: no verification-resource budget under test here");

        // Correctness: every register is still missing at every leaf,
        // since nothing anywhere ever defines any of them.
        for leaf in [LEAVES, LEAVES + 1, 2 * LEAVES - 1] {
            assert!(
                missing[leaf].contains(0),
                "nothing writes anything anywhere in this tree, so every register must still \
                 be missing at every leaf"
            );
        }
        assert!(
            event_count <= reachable_count * DOMAIN_SIZE,
            "event_count ({event_count}) must stay within the total-work bound \
             ({}) - this is the property round 8 already established, unaffected by \
             switching from FIFO to LIFO",
            reachable_count * DOMAIN_SIZE
        );
        // The actual claim under test: LIFO keeps the LIVE queue dramatically
        // smaller than the total-work bound, not just bounded by it - a
        // generous but far-from-total-work ceiling (tree depth ~16, times a
        // small constant, plus domain_size for the initial seed) rules out
        // the "peaks near the full total-work bound" failure mode Codex
        // described without pinning an exact formula.
        let generous_ceiling = DOMAIN_SIZE * 64;
        assert!(
            peak_queue_len <= generous_ceiling,
            "peak_queue_len ({peak_queue_len}) must stay far below the total-work bound \
             ({}) - LIFO order should keep it within a small multiple of domain_size \
             ({DOMAIN_SIZE}), not anywhere near reachable_count * domain_size",
            reachable_count * DOMAIN_SIZE
        );
    }

    /// #1756 Codex review round 15: proves the work-stack's peak size
    /// genuinely scales with `leader_count`, independent of
    /// `domain_size` - not just the `domain_size`-only bound round 9's
    /// own test happened to observe. This construction's successor
    /// ORDER matters: each node's two successors are `[dead_end (pushed
    /// first/bottom), continue (pushed second/top)]`. LIFO always pops
    /// the most-recently-pushed item, so it dives into `continue`
    /// immediately, leaving `dead_end` buried and unresolved - and this
    /// repeats at every one of `d` nodes along the chain, so the buried
    /// backlog grows by one entry per node, independent of how many
    /// (or how few) registers are actually live. Measured directly:
    /// `peak_queue_len` tracks `d` almost exactly (within a small,
    /// domain_size-bounded margin from the seed phase), confirming the
    /// `leader_count + 2*domain_size` bound `check_analysis_state_
    /// budget` now enforces is both necessary (an attacker really can
    /// force this) and sound (it never exceeds the bound).
    #[test]
    fn c1756_reversed_order_diamond_chain_stack_peak_scales_with_leader_count() {
        const D: usize = 100_000;
        const DOMAIN_SIZE: usize = 64;
        let reachable_count = 2 * D;
        let successors_of = |pos: usize| -> [Option<usize>; 2] {
            if pos % 2 == 1 {
                return [None, None]; // dead_end
            }
            let i = pos / 2;
            let dead_end = pos + 1;
            if i + 1 < D {
                [Some(dead_end), Some(pos + 2)] // dead_end first, continue second
            } else {
                [Some(dead_end), None]
            }
        };
        let writes_contains = |_pos: usize, _bit: usize| false;
        let entry_missing = RegSet::full(DOMAIN_SIZE);
        let (_missing, _event_count, peak_queue_len) = compute_missing_sets(
            reachable_count,
            DOMAIN_SIZE,
            entry_missing,
            writes_contains,
            successors_of,
            &mut AnalysisWorkMeter::new(usize::MAX),
        )
        .expect("unbounded");
        assert!(
            peak_queue_len > D / 2,
            "peak_queue_len ({peak_queue_len}) must scale with D ({D}), not stay near \
             domain_size ({DOMAIN_SIZE}) - this is exactly the gap round 15's state-budget \
             correction closes"
        );
        assert!(
            peak_queue_len <= D + 2 * DOMAIN_SIZE,
            "peak_queue_len ({peak_queue_len}) must still respect the proven sound bound \
             leader_count + 2*domain_size ({})",
            D + 2 * DOMAIN_SIZE
        );
    }

    /// #1756 Codex review round 15 (owner decision): P1's own exact
    /// prescribed adversarial regression - `domain_size = 1`, a long
    /// chain of genuine diamonds, the condition register defined at
    /// entry (so it dies immediately and essentially nothing propagates
    /// on the lattice side). `work_units` stays tiny (there is nothing
    /// to deliver), while `leader_count = 2*count + 1` still drives real
    /// fixed structural overhead (`Vec<RegSet>` headers, `chain_
    /// targets`, `leader_positions`, and the worklist stack) that round
    /// 13/14's dense-payload-only formula charged almost nothing for.
    /// Proves the corrected state budget protects a dimension the work
    /// budget cannot: an artifact can be cheap to COMPUTE yet expensive
    /// to REPRESENT.
    #[test]
    fn c1756_low_domain_high_leader_count_rejects_on_state_not_work() {
        const COUNT: usize = 500; // leader_count = 1,001; required_state_words = 18,653 exactly (round 21: 3,633 raw + 15,020 leader-state)
        let bytes = emit_test_function(build_domain_one_diamond_chain(COUNT));

        // Work stays tiny - accepted even under a small work cap.
        verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: usize::MAX,
                max_work_units: 100,
            },
        )
        .expect("work_units must stay tiny - r0 dies at entry, nothing ever propagates");

        // But fixed leader-scaled structural overhead plus raw reachable-
        // node representation exceeds a state budget that round 13/14's
        // dense-payload-only formula would have comfortably admitted.
        let report = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 18_652, // one below the exact 18,653 words this needs
                max_work_units: 100,     // generous for work, tight for state
            },
        )
        .expect_err(
            "raw reachable-node representation plus fixed per-leader/stack structural overhead \
             must exceed the state budget even though work stays trivially small",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AnalysisStateLimitExceeded,
            "must reject specifically on STATE, not work - proving this budget protects a \
             dimension the work budget cannot"
        );
        assert!(report.diagnostics[0]
            .message
            .contains("18653 logical verifier-analysis word"));
    }

    /// #1756 Codex review round 10: round 8's `missing: Vec<RegSet>` array
    /// eagerly allocated `RegSet::word_count(domain_size)` words for EVERY
    /// reachable position, including ones whose missing set converges to
    /// genuinely empty - a signature-less-params function that defines
    /// every register `r0..DOMAIN_SIZE-1` once, then continues through a
    /// long fallthrough tail of cheap re-writes to an already-defined
    /// register (identical adversarial shape to
    /// `c1756_reachable_full_domain_then_long_redefine_chain_stays_correct`,
    /// which only checked correctness), forced `reachable_count *
    /// domain_size` bits of storage even though almost the entire tail's
    /// real content, after convergence, is empty. Fixed: `RegSet::insert`
    /// now grows each set's own backing storage lazily, only as far as the
    /// highest bit it has ever actually held - see the sibling doc
    /// comments on `RegSet` and `compute_missing_sets` for the full
    /// argument. This test measures real words allocated directly (Codex's
    /// established preference for accounting evidence over timing alone),
    /// not just correctness.
    #[test]
    fn c1756_long_redefine_chain_keeps_missing_sets_sparse_not_dense_per_node() {
        const DOMAIN_SIZE: usize = 4_096;
        const TAIL_LEN: usize = 100_000;
        // Position 0 = entry (zero params: everything missing). Positions
        // 1..=DOMAIN_SIZE each write exactly one, distinct, previously
        // undefined register (position i writes register i-1), so the
        // missing set shrinks by one bit per position until, by position
        // DOMAIN_SIZE, nothing is left missing. Positions
        // DOMAIN_SIZE+1..reachable_count re-write register 0 (already
        // defined) TAIL_LEN times - Codex's "long chain of four-byte
        // writes" tail that grows no new information at all.
        let reachable_count = 1 + DOMAIN_SIZE + TAIL_LEN;

        let successors_of = |pos: usize| -> [Option<usize>; 2] {
            if pos + 1 < reachable_count {
                [Some(pos + 1), None]
            } else {
                [None, None]
            }
        };
        let writes_contains = |pos: usize, bit: usize| -> bool {
            if pos == 0 {
                false
            } else if pos <= DOMAIN_SIZE {
                bit == pos - 1
            } else {
                bit == 0
            }
        };

        let entry_missing = RegSet::full(DOMAIN_SIZE); // zero params: everything missing
        let (missing, _event_count, _peak_queue_len) = compute_missing_sets(
            reachable_count,
            DOMAIN_SIZE,
            entry_missing,
            writes_contains,
            successors_of,
            &mut AnalysisWorkMeter::new(usize::MAX),
        )
        .expect("test-only call: no verification-resource budget under test here");

        // Correctness: early on, most registers are still missing; by the
        // end of the tail, nothing is.
        assert!(missing[1].contains(DOMAIN_SIZE - 1));
        assert!(!missing[reachable_count - 1].contains(0));
        assert!(!missing[reachable_count - 1].contains(DOMAIN_SIZE - 1));

        // The claim under test: real words allocated must stay far below
        // what eagerly sizing every position to the full domain would
        // cost, dominated here by the long, genuinely-empty tail.
        let dense_baseline_words = reachable_count * RegSet::word_count(DOMAIN_SIZE);
        let actual_words: usize = missing.iter().map(|s| s.words.len()).sum();
        assert!(
            actual_words < dense_baseline_words / 10,
            "actual_words ({actual_words}) must stay far below the old eager-dense baseline \
             ({dense_baseline_words}) - most of this chain's tail has a genuinely empty \
             missing set and must cost close to nothing, not domain_size bits regardless of \
             content"
        );
        // The tail itself (after every register is defined) must be
        // exactly zero words, not just "smaller than before".
        let tail_words: usize = missing[DOMAIN_SIZE + 1..]
            .iter()
            .map(|s| s.words.len())
            .sum();
        assert_eq!(
            tail_words, 0,
            "every tail position's missing set is genuinely empty and must allocate zero words"
        );
    }

    /// #1756 Codex review round 11: "Bound aggregate growth of missing
    /// sets" - round 10's lazy growth only helps a position whose OWN
    /// content stays small; it does nothing when MANY positions share a
    /// large, unchanging value, which is exactly what a long,
    /// non-branching run of redundant re-writes produces (nothing besides
    /// register 0 is ever written, so every position after the first
    /// holds the identical 4,095-of-4,096-registers-missing value).
    /// Leader compression (`compute_leaders`/`build_leader_chains`)
    /// collapses this run to its single leader (entry) regardless of
    /// length, since it never branches - closing both the per-position
    /// storage amplification AND the independent per-hop bit-delivery
    /// amplification (event_count) in one fix, measured directly here
    /// rather than by timing.
    #[test]
    fn c1756_long_non_branching_run_with_wide_missing_set_collapses_to_few_leaders() {
        const DOMAIN_SIZE: usize = 4_096;
        const RUN_LEN: usize = 2_000_000;
        let reachable_count = RUN_LEN + 1;
        let successors_of = |pos: usize| -> [Option<usize>; 2] {
            if pos + 1 < reachable_count {
                [Some(pos + 1), None]
            } else {
                [None, None]
            }
        };

        let is_leader = compute_leaders(reachable_count, successors_of);
        let leader_positions: Vec<usize> = (0..reachable_count).filter(|&p| is_leader[p]).collect();
        assert_eq!(
            leader_positions,
            vec![0],
            "a run with no branching at all must collapse to exactly one leader (entry), \
             regardless of how long the run is - the whole point of leader compression"
        );

        let leader_index_of =
            |pos: usize| -> usize { leader_positions.binary_search(&pos).unwrap() };
        let (chain_killed, chain_targets) = build_leader_chains(
            &leader_positions,
            &is_leader,
            successors_of,
            |pos, visit| {
                if pos != 0 {
                    visit(0); // every non-entry position redundantly re-writes register 0 only
                }
            },
            leader_index_of,
        );
        assert_eq!(
            chain_targets,
            vec![[None, None]],
            "the run dead-ends (no branch, no back edge), so entry's chain delivers nowhere"
        );
        assert!(chain_killed[0].contains(0));
        assert!(
            !chain_killed[0].contains(1),
            "register 1 is never written anywhere in this run"
        );

        let entry_missing = RegSet::full(DOMAIN_SIZE); // zero params: everything missing
        let (missing, event_count, _peak_queue_len) = compute_missing_sets(
            leader_positions.len(),
            DOMAIN_SIZE,
            entry_missing,
            |leader_idx, bit| chain_killed[leader_idx].contains(bit),
            |leader_idx| chain_targets[leader_idx],
            &mut AnalysisWorkMeter::new(usize::MAX),
        )
        .expect("test-only call: no verification-resource budget under test here");
        assert!(
            missing[0].contains(1),
            "register 1 is never written anywhere, so it must still be missing at entry"
        );
        // `missing[0]` is entry's own IN (pre-write) - it stays exactly
        // `entry_missing`, unchanged (see `deliver`'s `target != 0` guard),
        // so it still contains bit 0 too; `chain_killed[0]` (already
        // asserted above) is where register 0 being defined shows up.
        assert!(missing[0].contains(0));
        // The claim under test: with leader_count collapsed to 1, total
        // worklist events are bounded by domain_size alone, not
        // reachable_count * domain_size (which for RUN_LEN = 2,000,000
        // would be over 8 billion).
        assert!(
            event_count <= DOMAIN_SIZE,
            "event_count ({event_count}) must stay within domain_size ({DOMAIN_SIZE}) alone - \
             it must NOT scale with reachable_count * domain_size ({}), which is what round \
             8-10's per-position worklist would have driven this to",
            reachable_count * DOMAIN_SIZE
        );
    }

    /// #1756 Codex review round 11: stresses `find_first_violation`'s
    /// cross-leader minimum tracking. Leader A (position 1, processed
    /// BEFORE leader B in ascending leader-position order) has its own
    /// violation at position 3; leader B (position 4, processed AFTER A)
    /// walks to position 2 - numerically LOWER than A's violation, even
    /// though B's own leader position (4) is higher - which also
    /// violates. The deterministic-first-diagnostic contract
    /// (`c1756_case25`/`c1756_case26`) requires the lowest-position
    /// violation overall (position 2) - only a full cross-leader scan,
    /// not a per-chain short-circuit in leader-processing order, can
    /// guarantee that. (Every position here is genuinely targeted by
    /// something, matching the real caller's invariant that every
    /// non-entry position in `0..reachable_count` has at least one
    /// predecessor - `compute_leaders` treats an untargeted position as
    /// its own trivial leader, which is correct but does not arise for
    /// an actually-reachable graph, so this test avoids it.)
    #[test]
    fn c1756_cross_leader_scan_finds_the_true_lowest_position_violation() {
        const DOMAIN_SIZE: usize = 8;
        const REACHABLE_COUNT: usize = 5; // positions 0..4
        let successors_of = |pos: usize| -> [Option<usize>; 2] {
            match pos {
                0 => [Some(1), Some(4)],
                1 => [Some(3), None],
                4 => [Some(2), None],
                _ => [None, None], // 2 and 3 are dead ends
            }
        };
        let no_writes = |_pos: usize, _visit: &mut dyn FnMut(usize)| {};

        let is_leader = compute_leaders(REACHABLE_COUNT, successors_of);
        let leader_positions: Vec<usize> = (0..REACHABLE_COUNT).filter(|&p| is_leader[p]).collect();
        assert_eq!(leader_positions, vec![0, 1, 4]);

        let leader_index_of =
            |pos: usize| -> usize { leader_positions.binary_search(&pos).unwrap() };
        let (chain_killed, chain_targets) = build_leader_chains(
            &leader_positions,
            &is_leader,
            successors_of,
            no_writes,
            leader_index_of,
        );

        let entry_missing = RegSet::full(DOMAIN_SIZE); // zero params: everything missing
        let (missing, _event_count, _peak_queue_len) = compute_missing_sets(
            leader_positions.len(),
            DOMAIN_SIZE,
            entry_missing,
            |leader_idx, bit| chain_killed[leader_idx].contains(bit),
            |leader_idx| chain_targets[leader_idx],
            &mut AnalysisWorkMeter::new(usize::MAX),
        )
        .expect("test-only call: no verification-resource budget under test here");

        let reads_of = |pos: usize, visit: &mut dyn FnMut(usize)| {
            if pos == 3 || pos == 2 {
                visit(6); // both "instructions" read the same still-missing register
            }
        };
        let violation = find_first_violation(
            &leader_positions,
            &is_leader,
            &missing,
            successors_of,
            reads_of,
            no_writes,
        );
        assert_eq!(
            violation,
            Some((2, 6)),
            "position 2's violation must win even though leader B (position 4) is processed \
             after leader A (position 1), whose own violation at position 3 would incorrectly \
             win under a per-chain-order short-circuit"
        );
    }

    /// #1756 Codex review round 11: exercises the exact adversarial
    /// ORDER through the real `verify_semcode` pipeline (not just the
    /// standalone leader-compression functions) - many redundant
    /// re-writes to r0 BEFORE r1.. are ever defined, so the missing set
    /// stays wide for the whole prefix (the shape round 10's fix did not
    /// help), followed by the actual definitions and a read of the last
    /// one. Confirms `prove_definite_register_assignment`'s leader-
    /// compression wiring produces the correct ACCEPT result end to end,
    /// complementing the standalone accounting regressions above.
    #[test]
    fn c1756_redundant_prefix_then_late_definitions_verifies_correctly_through_full_pipeline() {
        let mut instrs = Vec::new();
        for _ in 0..5000 {
            instrs.push(IrInstr::LoadI32 { dst: 0, val: 0 }); // redundant after the first
        }
        for r in 1..64u16 {
            instrs.push(IrInstr::LoadI32 { dst: r, val: 0 });
        }
        instrs.push(IrInstr::Ret { src: Some(63) });
        let bytes = emit_test_function(instrs);
        verify_semcode(&bytes).expect(
            "every register is genuinely defined by the time it's read, regardless of how long \
             the redundant prefix that precedes its definition is",
        );
    }

    /// #1756 Codex review round 12: "Deduplicate identical branch
    /// successors before choosing leaders" - a `JMP_IF` whose taken
    /// branch targets its own fallthrough (a legal, if pointless, no-op
    /// conditional branch) makes `for_each_reachable_successor` report
    /// that one true successor twice. Before `dedup_successors`,
    /// `compute_leaders` would see that as out-degree 2 (a branch) and
    /// double-count the shared target's in-degree, forcing it to become
    /// a leader - repeated over a long chain of such instructions, this
    /// defeats round 11's compression entirely, exactly reproducing its
    /// adversarial shape. This test proves BOTH halves directly: the raw
    /// (undeduplicated) successors really do blow leader compression
    /// back open, and running them through `dedup_successors` first (as
    /// the real caller now does) collapses the chain back to a single
    /// leader.
    #[test]
    fn c1756_self_targeting_branch_chain_collapses_via_dedup_successors() {
        const CHAIN_LEN: usize = 500_000;
        let reachable_count = CHAIN_LEN + 2; // entry, then CHAIN_LEN self-targeting branches, then a dead end
                                             // Position 0 = entry -> 1. Positions 1..reachable_count-1 each
                                             // mimic a self-targeting JMP_IF: `for_each_reachable_successor`
                                             // reports its next position as BOTH the taken and fallthrough
                                             // successor, i.e. the SAME position twice, before dedup.
        let raw_successors_of = |pos: usize| -> [Option<usize>; 2] {
            if pos == 0 {
                [Some(1), None]
            } else if pos < reachable_count - 1 {
                [Some(pos + 1), Some(pos + 1)]
            } else {
                [None, None]
            }
        };

        let undeduped_leaders = compute_leaders(reachable_count, raw_successors_of);
        let undeduped_leader_positions: Vec<usize> = (0..reachable_count)
            .filter(|&p| undeduped_leaders[p])
            .collect();
        assert!(
            undeduped_leader_positions.len() > CHAIN_LEN / 2,
            "reproducing the finding: WITHOUT dedup, a chain of self-targeting branches must \
             blow leader compression back open (got {} leaders for a {CHAIN_LEN}-instruction \
             chain, expected close to all of them)",
            undeduped_leader_positions.len()
        );

        let successors_of =
            |pos: usize| -> [Option<usize>; 2] { dedup_successors(raw_successors_of(pos)) };
        let is_leader = compute_leaders(reachable_count, successors_of);
        let leader_positions: Vec<usize> = (0..reachable_count).filter(|&p| is_leader[p]).collect();
        assert_eq!(
            leader_positions,
            vec![0],
            "the fix under test: once raw successors are deduplicated, a chain of self-targeting \
             branches must collapse to exactly one leader (entry), same as any other \
             non-branching run"
        );

        let leader_index_of =
            |pos: usize| -> usize { leader_positions.binary_search(&pos).unwrap() };
        let no_writes = |_pos: usize, _visit: &mut dyn FnMut(usize)| {};
        let (chain_killed, chain_targets) = build_leader_chains(
            &leader_positions,
            &is_leader,
            successors_of,
            no_writes,
            leader_index_of,
        );
        const DOMAIN_SIZE: usize = 4096;
        let entry_missing = RegSet::full(DOMAIN_SIZE);
        let (_missing, event_count, _peak_queue_len) = compute_missing_sets(
            leader_positions.len(),
            DOMAIN_SIZE,
            entry_missing,
            |leader_idx, bit| chain_killed[leader_idx].contains(bit),
            |leader_idx| chain_targets[leader_idx],
            &mut AnalysisWorkMeter::new(usize::MAX),
        )
        .expect("test-only call: no verification-resource budget under test here");
        assert!(
            event_count <= DOMAIN_SIZE,
            "event_count ({event_count}) must stay within domain_size ({DOMAIN_SIZE}) alone, \
             not scale with CHAIN_LEN * DOMAIN_SIZE - Codex's own figure for this shape at \
             32,768 branches was ~134 million events"
        );
    }

    /// #1756 Codex review round 12: the same finding exercised through
    /// the real `verify_semcode` pipeline - a real `JMP_IF` whose label
    /// resolves to the very next instruction, repeated many times,
    /// confirming `for_each_reachable_successor` really does produce the
    /// duplicate-successor shape `dedup_successors` is fixing, and that
    /// the real caller's wiring wasn't the untested half of the fix.
    #[test]
    fn c1756_self_targeting_conditional_branch_chain_verifies_correctly_through_full_pipeline() {
        let mut instrs = vec![IrInstr::LoadBool { dst: 0, val: true }];
        for i in 0..2000 {
            instrs.push(IrInstr::JmpIf {
                cond: 0,
                label: format!("l{i}"),
            });
            instrs.push(IrInstr::Label {
                name: format!("l{i}"),
            });
        }
        instrs.push(IrInstr::Ret { src: Some(0) });
        let bytes = emit_test_function(instrs);
        verify_semcode(&bytes).expect(
            "r0 is defined before any read, regardless of how long the self-targeting \
             conditional-branch chain is",
        );
    }

    /// #1756 Codex review round 13 (owner decision): builds `count` real
    /// genuine diamonds (`JMP_IF` with one arm jumping straight to the
    /// merge, the other falling through one no-op instruction first) -
    /// `r0` is the always-defined condition register; every diamond's
    /// arms are otherwise vacuous. If `undefined_tail_read` is `true`,
    /// the function ends by reading `r1` without ever defining it
    /// (pulling `r1` into the register domain and keeping it missing
    /// across the whole chain, inflating work-unit propagation);
    /// otherwise it ends by defining `r1` then reading it (an accepting
    /// program, `r1` dying at entry just like `r0` - domain stays wide
    /// only in shape, not in propagated content).
    fn build_diamond_chain(count: usize, undefined_tail_read: bool) -> Vec<IrInstr> {
        let mut instrs = vec![IrInstr::LoadBool { dst: 0, val: true }];
        for i in 0..count {
            instrs.push(IrInstr::JmpIf {
                cond: 0,
                label: format!("merge{i}"),
            });
            instrs.push(IrInstr::LoadBool { dst: 2, val: false }); // false-arm body: one no-op write to an unrelated register
            instrs.push(IrInstr::Label {
                name: format!("merge{i}"),
            });
        }
        if !undefined_tail_read {
            instrs.push(IrInstr::LoadI32 { dst: 1, val: 0 });
        }
        instrs.push(IrInstr::Ret { src: Some(1) });
        instrs
    }

    /// #1756 Codex review round 15 (owner decision), P1's own exact
    /// prescribed case: `count` genuine diamonds referencing ONLY `r0`
    /// (the always-defined condition register - the false arm
    /// redundantly re-writes it, and the final read is of the same,
    /// already-defined register) - so `domain_size = 1` and `r0` dies at
    /// entry's own write, before any propagation ever begins. `leader_
    /// count = 2*count + 1` genuine leaders still exist (real branches,
    /// real merges), so fixed structural overhead scales with `count`
    /// even though almost nothing is happening on the lattice side.
    fn build_domain_one_diamond_chain(count: usize) -> Vec<IrInstr> {
        let mut instrs = vec![IrInstr::LoadBool { dst: 0, val: true }]; // r0 defined at entry
        for i in 0..count {
            instrs.push(IrInstr::JmpIf {
                cond: 0,
                label: format!("merge{i}"),
            });
            instrs.push(IrInstr::LoadBool { dst: 0, val: false }); // false arm: redundant re-write of r0
            instrs.push(IrInstr::Label {
                name: format!("merge{i}"),
            });
        }
        instrs.push(IrInstr::Ret { src: Some(0) }); // r0 is defined - accepts
        instrs
    }

    /// #1756 Codex review round 16 (owner decision): P1's own exact
    /// prescribed shape - a genuinely branch-FREE reachable chain, no
    /// `JmpIf` at all, so `compute_leaders` marks ONLY the entry
    /// (position 0) as a leader regardless of `count`. `leader_count`
    /// stays exactly 1 no matter how long this runs - the shape round
    /// 15's leader/domain-scaled formula alone could not bound at all,
    /// since it charges nothing beyond a fixed O(1) leader count however
    /// large `reachable_count` grows. Closed by `check_raw_node_state_
    /// budget`, which bounds `reachable_count` (and the reachable
    /// instruction stream's own byte length) directly, independent of
    /// leader or domain structure.
    fn build_long_linear_chain(count: usize) -> Vec<IrInstr> {
        let mut instrs = vec![IrInstr::LoadBool { dst: 0, val: true }]; // r0 defined at entry
        for _ in 0..count {
            instrs.push(IrInstr::LoadBool { dst: 0, val: false }); // redundant re-write, no branching at all
        }
        instrs.push(IrInstr::Ret { src: Some(0) }); // r0 is defined - accepts
        instrs
    }

    /// #1756 Codex review round 16 (owner decision): Codex's own exact P1
    /// regression - a signature-bearing, genuinely branch-free reachable
    /// chain (`leader_count = 1` regardless of length), `domain_size =
    /// 1`. `work_units` stays tiny (nothing to deliver - a single-leader
    /// graph never even touches the worklist). Leader-state alone
    /// (`dense_words = (2*1+1)*word_count(1) = 3`, `fixed_words = 17`,
    /// `leader_words = 20` ALWAYS for this shape, for any chain length,
    /// since `leader_count` and `domain_size` never change) comfortably
    /// fits any real budget. Yet raw reachable-node representation
    /// (`reachable_indices`, `reads_flat`/`writes_flat`, their offset
    /// tables, `is_leader`) scales directly with `reachable_count`,
    /// unbounded by leader or domain structure at all before this round -
    /// proving this is a dimension the pre-round-16 envelope could not
    /// bound, not merely under-charge.
    #[test]
    fn c1756_long_linear_low_domain_rejects_on_raw_state_not_leader_state() {
        const COUNT: usize = 100_000; // reachable_count = 100,002; raw_words = 325,008; leader_words = 20 (leader_count=1, domain_size=1); combined = 325,028 exactly
        let bytes = emit_test_function(build_long_linear_chain(COUNT));

        // Work stays tiny - accepted even under a small work cap.
        verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: usize::MAX,
                max_work_units: 100,
            },
        )
        .expect("work_units must stay tiny - a single-leader graph never touches the worklist");

        // Leader-state alone (20 words, fixed for ANY chain length here)
        // is nowhere near the problem - a budget of 1,000 is still far
        // below the true raw-node cost (325,008), proving the rejection
        // below is specifically about raw reachable-node representation,
        // not about leader/domain-scaled state.
        let report = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 1_000,
                max_work_units: usize::MAX,
            },
        )
        .expect_err(
            "raw reachable-node representation must exceed the state budget even though \
             leader-state alone (20 words) is nowhere near it, and rejection must occur BEFORE \
             dataflow_domain_accounting or compute_leaders ever allocates their reachable-node- \
             sized structures",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AnalysisStateLimitExceeded,
            "must reject specifically on STATE, not work"
        );
        assert!(report.diagnostics[0]
            .message
            .contains("raw reachable-node representation alone"));

        // Exact boundary: the raw check alone (before leader state is
        // even computed) reports 325,008 words for this construction.
        let raw_only = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 0,
                max_work_units: usize::MAX,
            },
        )
        .expect_err("must reject");
        assert!(raw_only.diagnostics[0]
            .message
            .contains("325008 logical verifier-analysis word"));

        // Exact combined boundary: raw_words (325,008) + leader_words
        // (20) = 325,028 exactly, once leader state is also computed.
        verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 325_028,
                max_work_units: usize::MAX,
            },
        )
        .expect("usage == limit must be accepted for state-budget accounting");
        let combined = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 325_027,
                max_work_units: usize::MAX,
            },
        )
        .expect_err("usage == limit + 1 must be a deterministic resource rejection");
        assert!(combined.diagnostics[0]
            .message
            .contains("325028 logical verifier-analysis word"));
    }

    /// #1756 Codex review round 21 (owner decision), P2's own required
    /// regression: "construct a low-domain/high-reachable-count shape
    /// where: old raw formula passes, corrected scratch-aware raw
    /// formula rejects, and prove rejection occurs before classifier
    /// scratch allocation." Reuses `c1756_long_linear_low_domain_
    /// rejects_on_raw_state_not_leader_state`'s exact construction
    /// (100,000 unreachable-free branch-free `LOAD_BOOL`s, `domain_size
    /// = 1`, `reachable_count = 100,002`) and that same test's own
    /// already-established pre-round-21 exact raw-only requirement of
    /// 312,508 words (what this construction needed before this round's
    /// P2 fix added the `leader_classification_state` term).
    ///
    /// Setting `max_state_words` to exactly that OLD value proves the
    /// point directly: a verifier still running the pre-round-21 formula
    /// would compute `raw_words == 312,508 == limit` and ACCEPT - this
    /// construction would sail through the raw precheck, then let
    /// `compute_leaders` allocate its consolidated `state: Vec<u8>`
    /// classifier scratch (`reachable_count` = 100,002 bytes nobody
    /// charged for). The corrected, round-21 formula instead computes
    /// `raw_words == 325,008 > 312,508` and rejects - and the verifier's
    /// own diagnostic text asserted below ("before any leader-compressed
    /// state is even computed") is produced ONLY by `check_raw_node_
    /// state_budget`'s own error path, never by the later combined
    /// check, so its presence here is direct evidence, from the
    /// verifier's own reporting, that this rejection fires strictly
    /// before `compute_leaders` (or `dataflow_domain_accounting`) ever
    /// runs - i.e. before the classifier scratch array they would
    /// allocate exists at all.
    #[test]
    fn c1756_pre_round21_raw_formula_would_have_admitted_uncharged_classifier_scratch() {
        const COUNT: usize = 100_000;
        const OLD_RAW_WORDS_BEFORE_ROUND21_FIX: usize = 312_508;
        let bytes = emit_test_function(build_long_linear_chain(COUNT));

        let report = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: OLD_RAW_WORDS_BEFORE_ROUND21_FIX,
                max_work_units: usize::MAX,
            },
        )
        .expect_err(
            "a limit that exactly equalled the pre-round-21 raw-only requirement must now be \
             rejected by the corrected, scratch-aware raw formula - proving the old formula \
             would have admitted a construction whose true raw cost (including classifier \
             scratch) exceeds it",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AnalysisStateLimitExceeded
        );
        assert!(
            report.diagnostics[0]
                .message
                .contains("before any leader-compressed state is even computed"),
            "this exact phrase is emitted only by check_raw_node_state_budget's own error path, \
             never by the later combined check - its presence proves rejection happened strictly \
             before compute_leaders (or dataflow_domain_accounting) ever ran, i.e. before the \
             classifier scratch array they would allocate exists at all"
        );
        assert!(report.diagnostics[0]
            .message
            .contains("325008 logical verifier-analysis word"));
    }

    /// #1756 Codex review round 20 (owner decision): the "linear control
    /// case" - proves this round's reordering did NOT regress to
    /// charging `reachable_count` in place of `leader_count` (which
    /// would silently undo round 11's entire leader-compression
    /// benefit). Reuses `c1756_long_linear_low_domain_rejects_on_raw_
    /// state_not_leader_state`'s exact construction and numbers (100,000
    /// unreachable-free `LOAD_BOOL`s, `reachable_count = 100,002`,
    /// `leader_count = 1` always for a branch-free stream). Round 21's P2
    /// fix added a `leader_classification_state` term to the raw
    /// formula, raising `raw_words` from 312,508 to 325,008 and true
    /// combined from 312,528 to 325,028 - `500,000` still sits
    /// comfortably above the TRUE (post-round-21) combined requirement
    /// but far below what the combined requirement would be if
    /// `leader_count` were wrongly substituted with `reachable_count`
    /// (`dense_words = (2*100_002+1)*1 = 200,005`, `fixed_bytes =
    /// 100_002*88 + (100_002+2)*16 = 10,400,240`, `fixed_words =
    /// 1,300,030`, wrong combined = `325,008 + 200,005 + 1,300,030 =
    /// 1,825,043`) - if the reordering had accidentally substituted the
    /// wrong count, this construction would reject at `500,000`; it must
    /// still accept.
    #[test]
    fn c1756_linear_control_case_still_uses_leader_count_not_reachable_count() {
        let bytes = emit_test_function(build_long_linear_chain(100_000));
        verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 500_000,
                max_work_units: usize::MAX,
            },
        )
        .expect(
            "a branch-free stream's TRUE leader_count (1) must still admit this construction - \
             a limit this low would reject if leader_count had regressed to reachable_count \
             (100,002)",
        );
    }

    /// #1756 Codex review round 20 (owner decision): Codex's own exact
    /// regression - a branch-dense signature-bearing function where the
    /// raw-only precheck passes but the COMBINED check (which needs
    /// `leader_count`, and therefore needs `leader_positions`
    /// authorized) rejects. Reuses `build_diamond_chain`'s already-
    /// established exact numbers (`N = 1,000` genuine diamonds:
    /// `reachable_count = 2,003`, `leader_count = 2,001`, `raw_words =
    /// 7,262`, true combined = `37,290`, re-verified after round 21's
    /// formula extension). `max_state_words = 20_000` sits strictly
    /// between `raw_words` (7,262 - the precheck alone passes) and the
    /// combined requirement (37,290 - the combined check must reject), proving
    /// the combined check's own rejection is what fires, using the
    /// CORRECTLY count-derived `leader_count = 2,001` (asserted in the
    /// message) - not a stale or wrong value, and specifically NOT by
    /// way of ever materializing `leader_positions` first (see the
    /// companion direct/low-level test below for the strongest form of
    /// this proof: an equivalent construction that never once calls
    /// `.filter().collect()` for leader positions anywhere in the test's
    /// own code, mirroring the production call order exactly).
    #[test]
    fn c1756_branch_dense_combined_check_rejects_using_count_only_leader_count() {
        const N: usize = 1_000;
        let bytes = emit_test_function(build_diamond_chain(N, false));
        let report = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 20_000,
                max_work_units: usize::MAX,
            },
        )
        .expect_err(
            "the combined check must reject even though the raw-only precheck alone (7,262) \
             would pass at this limit",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AnalysisStateLimitExceeded
        );
        assert!(
            report.diagnostics[0].message.contains("2001 leader(s)"),
            "the exact leader_count must still be correctly derived and reported via the \
             count-only path, proving it produces the right number: {}",
            report.diagnostics[0].message
        );
        assert!(report.diagnostics[0]
            .message
            .contains("37290 logical verifier-analysis word"));
    }

    /// #1756 Codex review round 20 (owner decision): the STRONGEST form
    /// of "leader_positions has not yet been allocated" evidence - a
    /// direct, low-level mirror of the production call order
    /// (`compute_leaders` -> count `is_leader`'s true bits -> `check_
    /// analysis_state_budget`) that NEVER calls `.filter().collect()` to
    /// build a `Vec<usize>` of leader positions anywhere in this test's
    /// own code, exactly as `prove_definite_register_assignment` no
    /// longer does before this same check. Deliberately does NOT use
    /// allocator-size instrumentation for this specific claim: unlike
    /// round 18's per-INSTRUCTION allocation count (a multi-order-of-
    /// magnitude gap, cleanly separable from every other legitimate
    /// allocation this call makes), a single `leader_positions` `Vec`
    /// among several other legitimately reachable_count/leader_count-
    /// scaled allocations (`reachable_indices`, `is_leader`, `chain_
    /// killed`, etc., several of comparable size) is not reliably
    /// distinguishable by allocation COUNT or even peak SIZE alone - a
    /// direct call-order proof is the honest, non-brittle choice instead
    /// (matching round 18's own precedent of preferring a structural
    /// proof when instrumentation would not cleanly discriminate).
    #[test]
    fn c1756_leader_count_derived_without_collecting_positions_before_combined_check() {
        const DIAMOND_COUNT: usize = 1_000;
        const DOMAIN_SIZE: usize = 3;
        let reachable_count = 1 + 3 * DIAMOND_COUNT;
        let successors_of = |pos: usize| -> [Option<usize>; 2] {
            if pos == 0 {
                [Some(1), None]
            } else {
                let rel = pos - 1;
                let diamond = rel / 3;
                let slot = rel % 3;
                match slot {
                    0 => {
                        let merge = 1 + 3 * diamond + 2;
                        let false_arm = 1 + 3 * diamond + 1;
                        [Some(merge), Some(false_arm)]
                    }
                    1 => {
                        let merge = 1 + 3 * diamond + 2;
                        [Some(merge), None]
                    }
                    2 => {
                        if diamond + 1 < DIAMOND_COUNT {
                            [Some(1 + 3 * (diamond + 1)), None]
                        } else {
                            [None, None]
                        }
                    }
                    _ => unreachable!(),
                }
            }
        };

        let is_leader = compute_leaders(reachable_count, successors_of);
        // Zero-allocation count - no `Vec<usize>` of positions exists at
        // this point, matching production exactly.
        let leader_count = is_leader.iter().filter(|&&leader| leader).count();
        assert_eq!(
            leader_count,
            2 * DIAMOND_COUNT + 1,
            "exact leader count must still be correct via the count-only path"
        );

        // This test bypasses real SemCode decoding entirely, so
        // `reachable_instruction_bytes` is a synthetic, small (relative
        // to the leader-scaled cost below) placeholder - only its
        // relationship to the chosen limit matters here, not its exact
        // real-world value.
        let reachable_instruction_bytes = reachable_count * 4;
        let raw_words = check_raw_node_state_budget(
            reachable_count,
            reachable_instruction_bytes,
            0,
            usize::MAX,
        )
        .expect("raw precheck must pass at usize::MAX");

        let limit = raw_words + 100; // comfortably above raw alone, comfortably below the combined requirement
        assert!(
            check_raw_node_state_budget(reachable_count, reachable_instruction_bytes, 0, limit)
                .is_ok(),
            "raw-only precheck must pass at this limit"
        );
        let result = check_analysis_state_budget(raw_words, leader_count, DOMAIN_SIZE, limit);
        assert!(
            result.is_err(),
            "the combined check must reject once leader-state is included, even though raw \
             alone fit - and it does so here using ONLY the count-derived leader_count, with no \
             leader_positions Vec ever constructed anywhere in this test"
        );
    }

    /// #1756 Codex review round 20 (owner decision): "count/collect
    /// consistency" - across representative CFG shapes, the zero-
    /// allocation count (`is_leader.iter().filter(..).count()`) and the
    /// materialized position list (`(0..n).filter(..).collect()`) must
    /// never disagree. Both views read the SAME already-computed
    /// `is_leader: Vec<bool>` in this implementation (not two separate
    /// traversals), so this is provable by construction rather than
    /// merely tested - but the test still guards against a FUTURE change
    /// accidentally introducing a second, independently-computed
    /// counting path that could drift from `compute_leaders`'s own
    /// classification.
    #[test]
    fn c1756_leader_count_and_collected_positions_agree_across_cfg_shapes() {
        // Linear: no branches at all - exactly one leader (entry).
        {
            let reachable_count = 6;
            let successors_of = |pos: usize| -> [Option<usize>; 2] {
                if pos + 1 < reachable_count {
                    [Some(pos + 1), None]
                } else {
                    [None, None]
                }
            };
            let is_leader = compute_leaders(reachable_count, successors_of);
            let count_only = is_leader.iter().filter(|&&l| l).count();
            let collected: Vec<usize> = (0..reachable_count).filter(|&p| is_leader[p]).collect();
            assert_eq!(count_only, collected.len(), "linear");
            assert_eq!(collected, vec![0], "linear: entry only");
        }

        // Genuine diamond: entry branches to two arms, both merge - every
        // position is a leader (entry, 2 branch targets, 1 merge).
        {
            let reachable_count = 4;
            let successors_of = |pos: usize| -> [Option<usize>; 2] {
                match pos {
                    0 => [Some(1), Some(2)],
                    1 => [Some(3), None],
                    2 => [Some(3), None],
                    3 => [None, None],
                    _ => unreachable!(),
                }
            };
            let is_leader = compute_leaders(reachable_count, successors_of);
            let count_only = is_leader.iter().filter(|&&l| l).count();
            let collected: Vec<usize> = (0..reachable_count).filter(|&p| is_leader[p]).collect();
            assert_eq!(count_only, collected.len(), "genuine diamond");
            assert_eq!(
                collected,
                vec![0, 1, 2, 3],
                "genuine diamond: all 4 positions"
            );
        }

        // Merge with non-leader interior: two arms each with an extra
        // non-branching hop before converging - the hops (2 and 4) must
        // NOT be leaders.
        {
            let reachable_count = 6;
            let successors_of = |pos: usize| -> [Option<usize>; 2] {
                match pos {
                    0 => [Some(1), Some(3)],
                    1 => [Some(2), None],
                    2 => [Some(5), None],
                    3 => [Some(4), None],
                    4 => [Some(5), None],
                    5 => [None, None],
                    _ => unreachable!(),
                }
            };
            let is_leader = compute_leaders(reachable_count, successors_of);
            let count_only = is_leader.iter().filter(|&&l| l).count();
            let collected: Vec<usize> = (0..reachable_count).filter(|&p| is_leader[p]).collect();
            assert_eq!(
                count_only,
                collected.len(),
                "merge with non-leader interior"
            );
            assert_eq!(
                collected,
                vec![0, 1, 3, 5],
                "merge: entry, both branch targets, and the merge point - the two single-\
                 predecessor hops (2, 4) stay non-leaders"
            );
        }

        // Self-targeting JMP_IF with duplicate (undeduplicated-then-
        // deduplicated) successors - round 12's own fix.
        {
            let reachable_count = 5;
            let raw_successors_of = |pos: usize| -> [Option<usize>; 2] {
                if pos == 0 {
                    [Some(1), None]
                } else if pos + 1 < reachable_count {
                    [Some(pos + 1), Some(pos + 1)]
                } else {
                    [None, None]
                }
            };
            let successors_of =
                |pos: usize| -> [Option<usize>; 2] { dedup_successors(raw_successors_of(pos)) };
            let is_leader = compute_leaders(reachable_count, successors_of);
            let count_only = is_leader.iter().filter(|&&l| l).count();
            let collected: Vec<usize> = (0..reachable_count).filter(|&p| is_leader[p]).collect();
            assert_eq!(count_only, collected.len(), "self-targeting JMP_IF");
            assert_eq!(
                collected,
                vec![0],
                "self-targeting JMP_IF chain collapses to exactly one leader (entry) once \
                 deduplicated"
            );
        }

        // Loop: entry -> header (2 predecessors: entry, body) -> body
        // (branches back to header or forward to exit).
        {
            let reachable_count = 4;
            let successors_of = |pos: usize| -> [Option<usize>; 2] {
                match pos {
                    0 => [Some(1), None],
                    1 => [Some(2), None],
                    2 => [Some(1), Some(3)],
                    3 => [None, None],
                    _ => unreachable!(),
                }
            };
            let is_leader = compute_leaders(reachable_count, successors_of);
            let count_only = is_leader.iter().filter(|&&l| l).count();
            let collected: Vec<usize> = (0..reachable_count).filter(|&p| is_leader[p]).collect();
            assert_eq!(count_only, collected.len(), "loop");
            assert_eq!(
                collected,
                vec![0, 1, 3],
                "loop: entry, the header (merge point via the back-edge), and the exit branch \
                 target - the loop body itself (2) stays non-leader"
            );
        }
    }

    /// #1756 Codex review round 16: exact boundary behavior for `check_
    /// raw_node_state_budget` in isolation, independently re-derived
    /// from the named constants directly (not calling the production
    /// formula), so this test can catch an implementation bug rather
    /// than just confirming the function agrees with itself.
    #[test]
    fn c1756_raw_node_state_budget_boundary_usage_equals_limit_vs_limit_plus_one() {
        const REACHABLE_COUNT: usize = 1_000;
        const INSTR_LEN: usize = 4_000;
        const ENTRY_PARAM_COUNT: usize = 2;

        let operand_count_bound = INSTR_LEN / 2;
        let raw_bytes = REACHABLE_COUNT * REACHABLE_INDEX_LOGICAL_BYTES
            + (REACHABLE_COUNT + 1) * 2 * CSR_OFFSET_LOGICAL_BYTES
            + REACHABLE_COUNT * LEADER_FLAG_LOGICAL_BYTES
            + REACHABLE_COUNT * LEADER_CLASSIFICATION_STATE_LOGICAL_BYTES
            + (2 * operand_count_bound + ENTRY_PARAM_COUNT) * REGISTER_OPERAND_LOGICAL_BYTES;
        let expected_raw_words = raw_bytes.div_ceil(WORD_BYTES);

        assert_eq!(
            check_raw_node_state_budget(
                REACHABLE_COUNT,
                INSTR_LEN,
                ENTRY_PARAM_COUNT,
                expected_raw_words
            ),
            Ok(expected_raw_words),
            "usage == limit must be accepted"
        );
        assert_eq!(
            check_raw_node_state_budget(
                REACHABLE_COUNT,
                INSTR_LEN,
                ENTRY_PARAM_COUNT,
                expected_raw_words - 1
            ),
            Err(expected_raw_words),
            "usage == limit + 1 must be a deterministic resource rejection reporting the true \
             required word count"
        );
    }

    /// #1756 Codex review round 16: overflow specifically in the NEW
    /// raw-node arithmetic (not the round-13/15 leader-side paths those
    /// tests already cover) must also be a deterministic rejection.
    #[test]
    fn c1756_raw_node_state_budget_overflow_is_deterministic_rejection() {
        let reachable_count = usize::MAX / 4;
        assert!(
            reachable_count
                .checked_mul(REACHABLE_INDEX_LOGICAL_BYTES)
                .is_none(),
            "test premise: reachable_index_bytes must overflow"
        );
        let result = check_raw_node_state_budget(reachable_count, 0, 0, usize::MAX - 1);
        assert_eq!(
            result,
            Err(usize::MAX),
            "raw-node byte arithmetic overflowing usize must report the overflow sentinel, not \
             panic or silently wrap"
        );
    }

    /// #1756 Codex review round 19 (owner decision): direct, isolated
    /// unit coverage for `AnalysisWorkMeter::charge_one` - proves the
    /// checked-arithmetic overflow guarantee without actually executing
    /// `usize::MAX` units of real work (which would never finish). Every
    /// case here is a plain function call in a normal (non-panicking)
    /// test - `charge_one` is built entirely on `checked_add`, which by
    /// construction can never panic, so the absence of a `#[should_
    /// panic]` anywhere in this test IS part of the proof, not an
    /// oversight: the correct behavior is an ordinary `Err` return, never
    /// a panic, on any target or build profile.
    #[test]
    fn c1756_analysis_work_meter_charge_one_boundary_and_overflow() {
        // (1) Normal exact boundary: used = limit - 1, charge -> used ==
        // limit, accepted.
        let mut meter = AnalysisWorkMeter { used: 9, limit: 10 };
        assert_eq!(
            meter.charge_one(),
            Ok(10),
            "charging up to the exact limit must be accepted"
        );
        assert_eq!(meter.used, 10);

        // (2) Limit exceed: used = limit, charge -> rejected, attempted
        // usage == limit + 1 (representable, so reported exactly).
        let mut meter = AnalysisWorkMeter {
            used: 10,
            limit: 10,
        };
        assert_eq!(
            meter.charge_one(),
            Err(11),
            "charging past the limit must report the exact attempted usage (limit + 1) when \
             representable"
        );

        // (3) Arithmetic overflow: used = usize::MAX, limit = usize::MAX,
        // charge -> deterministic rejection via the overflow sentinel,
        // never a panic, never a silent wrap to 0.
        let mut meter = AnalysisWorkMeter {
            used: usize::MAX,
            limit: usize::MAX,
        };
        assert_eq!(
            meter.charge_one(),
            Err(usize::MAX),
            "used + 1 overflowing usize must report the overflow sentinel, never panic or \
             silently wrap to 0"
        );
        assert_eq!(
            meter.used,
            usize::MAX,
            "an overflowing charge must leave `used` exactly as it was - never wrapped to 0, \
             which would let the verifier resume accepting work UNDER the configured limit"
        );

        // (4) Near-overflow: used = usize::MAX - 1, limit = usize::MAX,
        // charge -> accepted (used becomes exactly usize::MAX, the last
        // representable unit); the VERY NEXT charge must then overflow
        // deterministically.
        let mut meter = AnalysisWorkMeter {
            used: usize::MAX - 1,
            limit: usize::MAX,
        };
        assert_eq!(
            meter.charge_one(),
            Ok(usize::MAX),
            "the last representable unit of work must still be accepted"
        );
        assert_eq!(
            meter.charge_one(),
            Err(usize::MAX),
            "the NEXT charge, now genuinely overflowing usize::MAX + 1, must be a deterministic \
             rejection, not a panic or a wrap back to a small, still-under-limit number"
        );
    }

    /// #1756 Codex review round 14 (owner decision): like
    /// `emit_test_function`, but for a MULTI-FUNCTION artifact - needed
    /// to exercise the artifact-wide (not per-function) work budget.
    fn emit_test_functions(names_and_instrs: Vec<(&str, Vec<IrInstr>)>) -> Vec<u8> {
        emit_ir_to_semcode(
            &names_and_instrs
                .into_iter()
                .map(|(name, instrs)| IrFunction {
                    name: name.to_string(),
                    instrs,
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                })
                .collect::<Vec<_>>(),
            false,
        )
        .expect("emit test functions")
    }

    /// #1756 Codex review round 13 (owner decision): a small genuine-
    /// diamond program - real branches, real merges, nothing spurious -
    /// must still verify normally under the default resource envelope.
    /// The round-13 finding is about a resource BOUND, not a
    /// correctness defect; ordinary, modest programs must be entirely
    /// unaffected.
    #[test]
    fn c1756_small_genuine_diamond_program_verifies_normally() {
        let bytes = emit_test_function(build_diamond_chain(2, false));
        verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits::default_profile(),
        )
        .expect("a handful of genuine diamonds, entirely defined, must verify normally");
    }

    /// #1756 Codex review round 13 (owner decision), formula corrected in
    /// round 14 (see `check_analysis_state_budget`'s doc comment - peak
    /// state charges BOTH `chain_killed` and `missing`, simultaneously
    /// live, not just one; corrected again in round 15 to also charge the
    /// fixed per-leader/stack structural overhead, and again in round 16
    /// to also charge raw reachable-node representation via `check_raw_
    /// node_state_budget`, `check_analysis_state_budget`'s own doc
    /// comment inventories): a construction whose exact `required_state_
    /// words` (raw reachable-node words, from `check_raw_node_state_
    /// budget`, plus dense payload `(2*leader_count + 1) * ceil(domain_
    /// size / 64)` plus fixed structural overhead - here `leader_count =
    /// 2*n+1` for `n` genuine diamonds, `domain_size = 3`, giving dense =
    /// `4*n+3` and, following the same derivation, required = 37,290 for
    /// `n = 1,000`) exceeds `max_state_words` must be rejected
    /// deterministically, with `AnalysisStateLimitExceeded` - and, by
    /// construction (see `prove_definite_register_assignment`'s round-13
    /// pre-check, extended in round 16), BEFORE `dataflow_domain_
    /// accounting`, `build_leader_chains`, or `compute_missing_sets` ever
    /// allocates their reachable-node- or leader_count-sized dense
    /// state.
    #[test]
    fn c1756_construction_exceeding_state_budget_rejects_before_large_allocation() {
        const N: usize = 1_000; // leader_count = 2,001; required_state_words = 37,290 exactly (round 21: 7,262 raw + 30,028 leader-state)
        let bytes = emit_test_function(build_diamond_chain(N, false));
        let report = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 37_289, // one below the exact 37,290 words this needs
                max_work_units: usize::MAX,
            },
        )
        .expect_err("must reject on the state budget before any large allocation");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AnalysisStateLimitExceeded
        );
        assert!(report.diagnostics[0]
            .message
            .contains("37290 logical verifier-analysis word"));
        assert!(report.diagnostics[0]
            .message
            .contains("state budget of 37289"));
    }

    /// #1756 Codex review round 13 (owner decision): a construction whose
    /// `state_words` stays comfortably within budget but whose
    /// `work_units` (verified exactly, by binary search against the real
    /// pipeline, to be `4*n+3` for `n` genuine vacuous diamonds with an
    /// undefined tail read) exceeds `max_work_units` must be rejected
    /// deterministically on the work budget specifically -
    /// `AnalysisWorkLimitExceeded`, not `AnalysisStateLimitExceeded`.
    #[test]
    fn c1756_construction_within_state_budget_exceeding_work_budget_rejects_on_work() {
        const N: usize = 5; // state_words = 11, work_units = 4*5+3 = 23 exactly
        let bytes = emit_test_function(build_diamond_chain(N, true));
        let report = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 1_000, // far above the 11 words this actually needs
                max_work_units: 22,     // one below the exact 23 units this needs
            },
        )
        .expect_err("must reject on the work budget, state budget was never in danger");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AnalysisWorkLimitExceeded
        );
        assert!(report.diagnostics[0].message.contains("23"));
        assert!(report.diagnostics[0].message.contains("22"));
    }

    /// #1756 Codex review round 13 (owner decision): repeated
    /// verification of identical bytes against an identical resource
    /// envelope must yield an identical diagnostic every time - same
    /// error code, same resource-kind identity (`AnalysisStateLimitExceeded`
    /// vs `AnalysisWorkLimitExceeded` are distinct, stable variants,
    /// precisely so this is a plain code comparison), same used/limit
    /// accounting (the message text), and the diagnostic carries no
    /// per-function identity here because this is a whole-function
    /// admission decision (mirrors
    /// `c1756_case26_repeated_verification_yields_identical_diagnostic`
    /// for the definite-assignment code path itself).
    #[test]
    fn c1756_repeated_verification_of_resource_limit_yields_identical_diagnostic() {
        const N: usize = 1_000;
        let bytes = emit_test_function(build_diamond_chain(N, false));
        let limits = VerificationLimits {
            max_state_words: 2_000,
            max_work_units: usize::MAX,
        };
        let first = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            limits,
        )
        .expect_err("must reject");
        for _ in 0..9 {
            let again = verify_semcode_token_with_quotas_and_limits(
                &bytes,
                RuntimeQuotas::verified_local(),
                limits,
            )
            .expect_err("must reject");
            assert_eq!(
                again, first,
                "repeated verification against an identical resource envelope must produce an \
                 identical diagnostic"
            );
        }
    }

    /// #1756 Codex review round 13 (owner decision), formula corrected in
    /// rounds 14, 15, and 16: exact boundary behavior for the (now fully
    /// raw-reachable-node-plus-leader-scaled: `reachable_indices`,
    /// `reads_flat`/`writes_flat`, their offset tables, `is_leader`,
    /// `chain_killed` + `missing` dense payload, their `Vec<RegSet>`
    /// headers, `chain_targets`, `leader_positions`, and the worklist
    /// stack's `leader_count + 2*domain_size` peak) state-words budget -
    /// `usage == limit` is accepted for resource accounting purposes
    /// (the state check itself must not fire; the construction here is
    /// fully defined, so the overall result is a genuine accept),
    /// `usage == limit + 1` is a deterministic resource rejection. No
    /// off-by-one ambiguity.
    #[test]
    fn c1756_state_budget_boundary_usage_equals_limit_vs_limit_plus_one() {
        const N: usize = 5; // leader_count = 11; required_state_words = 226 exactly (round 21: 48 raw + 178 leader-state)
        let bytes = emit_test_function(build_diamond_chain(N, false));
        verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 226,
                max_work_units: usize::MAX,
            },
        )
        .expect("usage == limit must be accepted for state-budget accounting");

        let report = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 225,
                max_work_units: usize::MAX,
            },
        )
        .expect_err("usage == limit + 1 must be a deterministic resource rejection");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AnalysisStateLimitExceeded
        );
    }

    /// #1756 Codex review round 13 (owner decision): exact boundary
    /// behavior for the work-units budget, mirroring the state-budget
    /// boundary test above.
    #[test]
    fn c1756_work_budget_boundary_usage_equals_limit_vs_limit_plus_one() {
        const N: usize = 5; // work_units = 23 exactly (state_words = 11, well within budget)
        let bytes = emit_test_function(build_diamond_chain(N, false));
        verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 1_000,
                max_work_units: 23,
            },
        )
        .expect("usage == limit must be accepted for work-budget accounting");

        let report = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: 1_000,
                max_work_units: 22,
            },
        )
        .expect_err("usage == limit + 1 must be a deterministic resource rejection");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AnalysisWorkLimitExceeded
        );
    }

    /// #1756 Codex review round 13 (owner decision): `checked_mul`
    /// overflow itself must be treated as a deterministic rejection, not
    /// a panic or wrapping miscalculation - there is no finite word
    /// count to report in that case, only that the true count is
    /// unrepresentable and therefore certainly exceeds any finite
    /// `max_state_words`.
    #[test]
    fn c1756_analysis_state_budget_checked_mul_overflow_is_deterministic_rejection() {
        let result = check_analysis_state_budget(0, usize::MAX, 128, usize::MAX - 1);
        assert_eq!(
            result,
            Err(usize::MAX),
            "leader_count * word_count(domain_size) overflowing usize must report the \
             overflow sentinel, not panic or silently wrap"
        );
    }

    /// #1756 Codex review round 15: overflow specifically in the NEW
    /// fixed-structural-overhead arithmetic (not the round-13 dense-
    /// payload path the test above already covers) must also be a
    /// deterministic rejection. `leader_count` here is chosen small
    /// enough that `2 * leader_count + 1` (the dense-payload factor)
    /// does NOT overflow, but large enough that `leader_count *
    /// per_leader_fixed_bytes` does - proving every checked step in the
    /// chain is actually load-bearing, not just the first one.
    #[test]
    fn c1756_analysis_state_budget_fixed_overhead_overflow_is_deterministic_rejection() {
        let leader_count = usize::MAX / 50; // 2*leader_count+1 fits; leader_count*88 does not
        assert!(
            leader_count.checked_mul(2).is_some(),
            "test premise: dense factor must not overflow"
        );
        assert!(
            leader_count.checked_mul(88).is_none(),
            "test premise: per-leader fixed bytes must overflow"
        );
        let result = check_analysis_state_budget(0, leader_count, 1, usize::MAX - 1);
        assert_eq!(
            result,
            Err(usize::MAX),
            "fixed structural overhead overflowing usize must report the overflow sentinel too, \
             not panic, silently wrap, or fall through to only the dense-payload term"
        );
    }

    /// #1756 Codex review round 13 (owner decision): a permanent
    /// reproduction of Codex's genuine-diamond construction - direct
    /// accounting evidence (leader_count, event_count, real memory) via
    /// `compute_leaders`/`build_leader_chains`/`compute_missing_sets`,
    /// confirming the underlying solver's behavior on this shape is
    /// unchanged by round 13 (a resource envelope was added around it;
    /// the solver itself was not touched) and matches the finding's own
    /// figures at scale.
    #[test]
    fn c1756_genuine_diamond_chain_accounting_matches_finding() {
        const DOMAIN_SIZE: usize = 4096;
        const DIAMOND_COUNT: usize = 10_000;
        let reachable_count = 1 + 3 * DIAMOND_COUNT;
        let successors_of = |pos: usize| -> [Option<usize>; 2] {
            if pos == 0 {
                [Some(1), None]
            } else {
                let rel = pos - 1;
                let diamond = rel / 3;
                let slot = rel % 3;
                match slot {
                    0 => {
                        let merge = 1 + 3 * diamond + 2;
                        let false_arm = 1 + 3 * diamond + 1;
                        [Some(merge), Some(false_arm)]
                    }
                    1 => {
                        let merge = 1 + 3 * diamond + 2;
                        [Some(merge), None]
                    }
                    2 => {
                        if diamond + 1 < DIAMOND_COUNT {
                            [Some(1 + 3 * (diamond + 1)), None]
                        } else {
                            [None, None]
                        }
                    }
                    _ => unreachable!(),
                }
            }
        };
        let no_writes = |_pos: usize, _visit: &mut dyn FnMut(usize)| {};

        let is_leader = compute_leaders(reachable_count, successors_of);
        let leader_positions: Vec<usize> = (0..reachable_count).filter(|&p| is_leader[p]).collect();
        assert_eq!(
            leader_positions.len(),
            2 * DIAMOND_COUNT + 1,
            "each genuine diamond contributes exactly 2 leaders (branch target + merge), plus entry"
        );

        let leader_index_of =
            |pos: usize| -> usize { leader_positions.binary_search(&pos).unwrap() };
        let (chain_killed, chain_targets) = build_leader_chains(
            &leader_positions,
            &is_leader,
            successors_of,
            no_writes,
            leader_index_of,
        );
        let entry_missing = RegSet::full(DOMAIN_SIZE);
        let (missing, event_count, _peak_queue_len) = compute_missing_sets(
            leader_positions.len(),
            DOMAIN_SIZE,
            entry_missing,
            |leader_idx, bit| chain_killed[leader_idx].contains(bit),
            |leader_idx| chain_targets[leader_idx],
            &mut AnalysisWorkMeter::new(usize::MAX),
        )
        .expect("unbounded work budget must converge");
        let actual_words: usize = missing.iter().map(|s| s.words.len()).sum();

        assert_eq!(
            event_count,
            2 * DIAMOND_COUNT * DOMAIN_SIZE,
            "each diamond delivers every one of the domain's still-missing bits across exactly \
             2 dequeued events (branch-target and merge), matching the finding's own \
             ~2*leader_count*domain_size accounting"
        );
        assert_eq!(
            actual_words,
            leader_positions.len() * RegSet::word_count(DOMAIN_SIZE),
            "with nothing anywhere killing any bit, every leader's missing set genuinely spans \
             the full domain - this is the real, unavoidable cost this round's resource \
             envelope exists to bound, not a representation bug"
        );
    }

    /// #1756 Codex review round 14 (owner decision), P1: `max_work_units`
    /// is the budget for one WHOLE artifact verification call, not a
    /// fresh allowance per function. Two functions whose own work
    /// (`4*n+3` for `n` genuine diamonds, established in round 13) each
    /// individually fits under a cap - but whose SUM does not - must
    /// still reject the whole artifact with `AnalysisWorkLimitExceeded`,
    /// and the exact cumulative boundary (`usage == limit` accepted,
    /// `usage == limit + 1` rejected) must hold for the SUM, not either
    /// function's own local count.
    #[test]
    fn c1756_artifact_wide_work_budget_shared_across_functions() {
        const N_A: usize = 5; // work_units = 4*5+3 = 23
        const N_B: usize = 10; // work_units = 4*10+3 = 43
        const COMBINED: usize = 23 + 43; // = 66

        let fn_a_alone = emit_test_functions(vec![("fnA", build_diamond_chain(N_A, false))]);
        verify_semcode_token_with_quotas_and_limits(
            &fn_a_alone,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: usize::MAX,
                max_work_units: 50,
            },
        )
        .expect("fn A alone (23 units) fits comfortably under a 50-unit budget");
        let fn_b_alone = emit_test_functions(vec![("fnB", build_diamond_chain(N_B, false))]);
        verify_semcode_token_with_quotas_and_limits(
            &fn_b_alone,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: usize::MAX,
                max_work_units: 50,
            },
        )
        .expect("fn B alone (43 units) fits comfortably under a 50-unit budget");

        let bytes = emit_test_functions(vec![
            ("fnA", build_diamond_chain(N_A, false)),
            ("fnB", build_diamond_chain(N_B, false)),
        ]);
        let report = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: usize::MAX,
                max_work_units: 50,
            },
        )
        .expect_err(
            "combined 66 units must exceed a 50-unit artifact-wide budget even though each \
             function individually fits under it",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AnalysisWorkLimitExceeded
        );

        // Exact cumulative boundary: usage == limit (66) is accepted...
        verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: usize::MAX,
                max_work_units: COMBINED,
            },
        )
        .expect("cumulative usage == limit (66) must be accepted");
        // ...usage == limit + 1 (limit = 65) is a deterministic rejection.
        let boundary_report = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: usize::MAX,
                max_work_units: COMBINED - 1,
            },
        )
        .expect_err("cumulative usage == limit + 1 must be a deterministic resource rejection");
        assert_eq!(
            boundary_report.diagnostics[0].code,
            VerificationCode::AnalysisWorkLimitExceeded
        );
        assert!(boundary_report.diagnostics[0].message.contains("66"));
        assert!(boundary_report.diagnostics[0].message.contains("65"));
    }

    /// #1756 Codex review round 14 (owner decision), P1: once the
    /// artifact-wide work budget is exhausted, verification must stop
    /// immediately - not continue merely to discover another diagnostic.
    /// `fnC` here would produce a DIFFERENT diagnostic
    /// (`UndefinedRegisterRead`, reading an undefined `r99`) if it were
    /// ever reached; the only way the final result can still be
    /// `AnalysisWorkLimitExceeded` is if `fnC` was never analyzed at all.
    #[test]
    fn c1756_artifact_work_exhaustion_stops_before_later_functions_run() {
        let bytes = emit_test_functions(vec![
            ("fnA", build_diamond_chain(5, false)),
            ("fnB", build_diamond_chain(10, false)),
            ("fnC", vec![IrInstr::Ret { src: Some(99) }]),
        ]);
        let report = verify_semcode_token_with_quotas_and_limits(
            &bytes,
            RuntimeQuotas::verified_local(),
            VerificationLimits {
                max_state_words: usize::MAX,
                max_work_units: 50, // fnA (23) + fnB (43) alone already exceed this
            },
        )
        .expect_err("fnA + fnB alone already exhaust the 50-unit artifact budget");
        assert_eq!(
            report.diagnostics.len(),
            1,
            "must stop immediately on exhaustion - no fnC diagnostic, no extra partial work: {:?}",
            report.diagnostics
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::AnalysisWorkLimitExceeded,
            "the result must remain the work-limit error from the point of exhaustion, not \
             fnC's UndefinedRegisterRead - proving fnC was never reached"
        );
    }

    /// #1756 Codex review round 14 (owner decision), P2: reproduces
    /// Codex's exact case directly - a zero-param leader whose entry
    /// missing set is the full register domain, a genuine two-leader
    /// diamond where one leader's chain kills nearly the entire register
    /// universe (leaving exactly one register, the highest-numbered one,
    /// never killed anywhere) - proving `chain_killed` and `missing` are
    /// BOTH simultaneously dense (backing storage genuinely grown wide),
    /// not just one of them, and that the corrected `(2*leader_count+1) *
    /// word_count(domain_size)` formula (not round 13's undercounting
    /// `leader_count * word_count(domain_size)`) is what
    /// `check_analysis_state_budget` now enforces, with exact boundaries.
    #[test]
    fn c1756_peak_state_accounts_for_chain_killed_and_missing_simultaneously() {
        const DOMAIN_SIZE: usize = 4096;
        const REACHABLE_COUNT: usize = 3; // 0 = entry (folds a 2-way branch), 1 = false_arm, 2 = merge
        let successors_of = |pos: usize| -> [Option<usize>; 2] {
            match pos {
                0 => [Some(2), Some(1)], // entry: direct-to-merge, fallthrough-to-false_arm
                1 => [Some(2), None],    // false_arm -> merge
                _ => [None, None],       // merge: dead end
            }
        };
        let writes_of = |pos: usize, visit: &mut dyn FnMut(usize)| {
            if pos == 1 {
                // false_arm kills nearly the entire register universe -
                // every register EXCEPT the highest-numbered one, which
                // stays missing everywhere (including at `merge`, via the
                // direct entry->merge edge this chain never touches).
                for bit in 0..DOMAIN_SIZE - 1 {
                    visit(bit);
                }
            }
        };

        let is_leader = compute_leaders(REACHABLE_COUNT, successors_of);
        let leader_positions: Vec<usize> = (0..REACHABLE_COUNT).filter(|&p| is_leader[p]).collect();
        assert_eq!(
            leader_positions,
            vec![0, 1, 2],
            "entry, the branch target, and the genuine merge are all leaders"
        );

        let leader_index_of =
            |pos: usize| -> usize { leader_positions.binary_search(&pos).unwrap() };
        let (chain_killed, chain_targets) = build_leader_chains(
            &leader_positions,
            &is_leader,
            successors_of,
            writes_of,
            leader_index_of,
        );

        let entry_missing = RegSet::full(DOMAIN_SIZE); // zero params: everything missing
        let mut work_meter = AnalysisWorkMeter::new(usize::MAX);
        let (missing, _event_count, _peak_queue_len) = compute_missing_sets(
            leader_positions.len(),
            DOMAIN_SIZE,
            entry_missing,
            |leader_idx, bit| chain_killed[leader_idx].contains(bit),
            |leader_idx| chain_targets[leader_idx],
            &mut work_meter,
        )
        .expect("unbounded work budget must converge");

        let chain_killed_words: usize = chain_killed.iter().map(|s| s.words.len()).sum();
        let missing_words: usize = missing.iter().map(|s| s.words.len()).sum();
        assert!(
            chain_killed_words > 0,
            "chain_killed must retain dense backing words (false_arm's chain wrote nearly the \
             whole domain)"
        );
        assert!(
            missing_words > 0,
            "missing must SIMULTANEOUSLY retain dense backing words - the highest register is \
             never killed anywhere, so it stays missing (and wide) at every leader"
        );

        let leader_count = leader_positions.len();
        // Independent re-derivation of `check_analysis_state_budget`'s
        // round-15 formula (not a call into it), so this test can catch
        // an implementation bug rather than just confirming the function
        // agrees with itself.
        let words_per_leader = RegSet::word_count(DOMAIN_SIZE);
        let dense_words = (2 * leader_count + 1) * words_per_leader;
        let per_leader_fixed_bytes = 2 * REGSET_HEADER_LOGICAL_BYTES
            + CHAIN_TARGET_LOGICAL_BYTES
            + LEADER_INDEX_LOGICAL_BYTES;
        let stack_peak_entries = 2 * DOMAIN_SIZE + leader_count;
        let fixed_bytes =
            leader_count * per_leader_fixed_bytes + stack_peak_entries * STACK_ENTRY_LOGICAL_BYTES;
        let fixed_words = fixed_bytes.div_ceil(WORD_BYTES);
        let expected_required_words = dense_words + fixed_words;
        assert_eq!(
            check_analysis_state_budget(0, leader_count, DOMAIN_SIZE, expected_required_words),
            Ok(expected_required_words),
            "usage == limit must be accepted for state-budget accounting"
        );
        assert_eq!(
            check_analysis_state_budget(0, leader_count, DOMAIN_SIZE, expected_required_words - 1),
            Err(expected_required_words),
            "usage == limit + 1 must be a deterministic resource rejection reporting the true \
             required word count"
        );
    }

    /// Codex review round 1 on PR #1840, second fix: the register domain
    /// must be the actual registers in use (`U`), densely packed, not the
    /// raw `0..=max_register_id` numeric span. This function references
    /// only `r0` and `r4095` - two registers, nowhere near each other
    /// numerically - so `U` must have exactly 2 entries, not 4096, while
    /// still proving both cases correctly: `r0` (written, then read) is
    /// accepted; `r4095` (read without being written) is rejected. A bug in
    /// the dense-index remapping (an off-by-one, an unsorted `universe`, a
    /// stale raw-index lookup) would most likely surface as exactly this
    /// kind of correctness failure on sparse, wide-spanning register use.
    #[test]
    fn c1756_dense_domain_handles_sparse_wide_spanning_registers() {
        let accepted = emit_test_function(vec![
            IrInstr::LoadI32 { dst: 0, val: 1 },
            IrInstr::Ret { src: Some(0) },
        ]);
        verify_semcode(&accepted).expect("r0 is written before it is read");

        let rejected = emit_test_function(vec![IrInstr::Ret { src: Some(4095) }]);
        let report = verify_semcode(&rejected)
            .expect_err("r4095 is read but never written or entry-defined");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    /// Codex review round 2 on PR #1840, finding 1: making the padding
    /// reachable still amplified memory under round 1's fix, because
    /// `domain_size` scales with *actual* register usage, and a program can
    /// genuinely, reachably use every register in the budget. This
    /// reproduces exactly that shape - define every register `r0..r4095`
    /// once, then continue through a long fallthrough chain of writes to
    /// registers already defined (the realistic form of "cheap writes" that
    /// doesn't grow `U` further) - and proves it still verifies correctly.
    /// (The original fix here was `Rc`-sharing over a MUST/intersection
    /// formulation; round 8 replaced that formulation entirely with the
    /// MAY-missing/union dual, and round 10 separately bounds this exact
    /// shape's memory in
    /// `c1756_long_redefine_chain_keeps_missing_sets_sparse_not_dense_per_node`,
    /// so this test now only asserts the correctness half.)
    #[test]
    fn c1756_reachable_full_domain_then_long_redefine_chain_stays_correct() {
        let mut instrs = Vec::new();
        for r in 0..4096u16 {
            instrs.push(IrInstr::LoadI32 { dst: r, val: 0 });
        }
        for _ in 0..4000 {
            // Re-writes r0, already defined above - grows no new
            // information, the exact "cheap write" tail the finding
            // describes.
            instrs.push(IrInstr::LoadI32 { dst: 0, val: 1 });
        }
        instrs.push(IrInstr::Ret { src: Some(4095) });
        let bytes = emit_test_function(instrs);
        verify_semcode(&bytes).expect(
            "every register 0..4095 is genuinely defined before the final read, regardless of \
             how long the redefine tail is",
        );
    }

    /// Codex review round 2 on PR #1840, finding 2: the round-1 fixed-point
    /// loop rescanned reachable positions in ascending (offset) order every
    /// round, so information propagating *against* that order (as it does
    /// here: entry jumps to the LAST instruction, and each subsequent
    /// instruction jumps one step backward toward an early, undefined
    /// `RET`) only advanced one node per round - `O(reachable_count)`
    /// rounds of `O(reachable_count)` work each. Fixed by a genuine
    /// predecessor/successor worklist: each node's `IN` is computed
    /// directly from its actual predecessor(s)' `OUT`, propagated forward
    /// along real edges regardless of node-index order, so this chain
    /// converges in a single pass over its edges, not a quadratic scan.
    #[test]
    fn c1756_backward_propagating_chain_still_rejects_and_converges_promptly() {
        const CHAIN_LEN: usize = 3000;
        let mut instrs = vec![IrInstr::Jmp {
            label: format!("l{CHAIN_LEN}"),
        }];
        // "l1" resolves to the very next real instruction, this RET.
        instrs.push(IrInstr::Label {
            name: "l1".to_string(),
        });
        instrs.push(IrInstr::Ret { src: Some(0) }); // never defined
        for i in 2..=CHAIN_LEN {
            instrs.push(IrInstr::Label {
                name: format!("l{i}"),
            });
            instrs.push(IrInstr::Jmp {
                label: format!("l{}", i - 1),
            });
        }
        // Physical/offset order: entry, l1(RET), l2(Jmp l1), l3(Jmp l2), ...,
        // l{CHAIN_LEN}(Jmp l{CHAIN_LEN-1}). Execution/propagation order is
        // the reverse: entry -> l{CHAIN_LEN} -> l{CHAIN_LEN-1} -> ... -> l2
        // -> l1(RET) - directly opposite the ascending-offset scan order
        // the pre-fix round-robin loop used.
        let bytes = emit_test_function(instrs);
        let report = verify_semcode(&bytes).expect_err(
            "r0 is read by the RET at the far end of the backward chain and is never defined",
        );
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
    }

    /// Direct accounting evidence for the Codex-review round-1, round-3, and
    /// round-4 fixes, per the review's own request not to rely only on
    /// timing. `dataflow_domain_accounting` takes plain, hand-constructible
    /// inputs - this asserts its outputs' exact lengths (the quantities
    /// that bound `prove_definite_register_assignment`'s memory) directly,
    /// without needing to run a full decode walk. The `reads_writes_of`
    /// closure is backed by a small map covering ONLY the reachable indices
    /// actually queried - proving directly that unreachable indices are
    /// never even looked up, let alone materialized as a dense per-
    /// instruction entry (round 3's fix). The offsets arrays' lengths
    /// (`reachable_count + 1`, a `u32` each - not one `Vec<u16>` per node)
    /// are round 4's fix: reads/writes are now one flat allocation each,
    /// not `reachable_count` separate heap-container allocations.
    #[test]
    fn c1756_accounting_bounds_reachable_nodes_and_dense_universe_exactly() {
        // 501 structurally decoded nodes (an entry RET plus 500 unreachable
        // instructions), but only node 0 is reachable, and nothing is ever
        // read or written - the exact "RET followed by unreachable bloat"
        // shape Codex's finding described.
        let instr_starts: Vec<usize> = (0..501).collect();
        let mut reachable_offsets = HashSet::new();
        reachable_offsets.insert(0);
        let reads_writes: std::collections::HashMap<usize, (Vec<u16>, Vec<u16>)> =
            std::collections::HashMap::from([(0, (Vec::new(), Vec::new()))]);
        let (reachable_indices, reads_flat, reads_offsets, writes_flat, writes_offsets, universe) =
            dataflow_domain_accounting(&instr_starts, &reachable_offsets, 0, |idx, sink| {
                let (reads, writes) = reads_writes
                    .get(&idx)
                    .expect("must only be queried for the one reachable index");
                for &r in reads {
                    sink.read(r);
                }
                for &w in writes {
                    sink.write(w);
                }
            });
        assert_eq!(
            reachable_indices.len(),
            1,
            "500 unreachable nodes must not be allocated dataflow state"
        );
        assert_eq!(reads_flat.len(), 0, "nothing is ever read");
        assert_eq!(writes_flat.len(), 0, "nothing is ever written");
        assert_eq!(
            reads_offsets.len(),
            2,
            "one flat offset table sized reachable_count+1, not one Vec per node"
        );
        assert_eq!(writes_offsets.len(), 2);
        assert_eq!(
            universe.len(),
            0,
            "no register is ever read, written, or entry-defined"
        );

        // Two reachable nodes, referencing only r0 and r4095 - numerically
        // 4095 apart, but only 2 distinct registers actually in use.
        let instr_starts: Vec<usize> = vec![0, 1];
        let mut reachable_offsets = HashSet::new();
        reachable_offsets.insert(0);
        reachable_offsets.insert(1);
        let reads_writes: std::collections::HashMap<usize, (Vec<u16>, Vec<u16>)> =
            std::collections::HashMap::from([
                (0, (Vec::new(), vec![0])),
                (1, (vec![4095], Vec::new())),
            ]);
        let (reachable_indices, reads_flat, reads_offsets, writes_flat, writes_offsets, universe) =
            dataflow_domain_accounting(&instr_starts, &reachable_offsets, 0, |idx, sink| {
                let (reads, writes) = reads_writes.get(&idx).expect("both are reachable");
                for &r in reads {
                    sink.read(r);
                }
                for &w in writes {
                    sink.write(w);
                }
            });
        assert_eq!(reachable_indices.len(), 2);
        assert_eq!(reads_flat, vec![4095], "only node 1's one read");
        assert_eq!(writes_flat, vec![0], "only node 0's one write");
        assert_eq!(reads_offsets, vec![0, 0, 1]);
        assert_eq!(writes_offsets, vec![0, 1, 1]);
        assert_eq!(csr_slice(&reads_flat, &reads_offsets, 0), &[] as &[u16]);
        assert_eq!(csr_slice(&reads_flat, &reads_offsets, 1), &[4095]);
        assert_eq!(csr_slice(&writes_flat, &writes_offsets, 0), &[0]);
        assert_eq!(csr_slice(&writes_flat, &writes_offsets, 1), &[] as &[u16]);
        assert_eq!(
            universe.len(),
            2,
            "the dense register domain must track the 2 registers actually referenced \
             ({{r0, r4095}}), not the raw 4096-wide numeric span between them: {universe:?}"
        );
    }

    #[test]
    fn c1756_case25_reports_deterministic_first_diagnostic_among_multiple_undefined_reads() {
        let bytes = emit_test_function(vec![
            IrInstr::AddI32 {
                dst: 2,
                lhs: 10,
                rhs: 11,
            }, // r10, r11 both undefined
            IrInstr::AddI32 {
                dst: 3,
                lhs: 20,
                rhs: 21,
            }, // r20, r21 also undefined
            IrInstr::Ret { src: Some(3) },
        ]);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UndefinedRegisterRead
        );
        assert_eq!(report.diagnostics[0].offset, Some(0));
        assert!(
            report.diagnostics[0].message.contains("r10"),
            "the first reported undefined register must be the lowest-offset, first-decoded \
             operand (r10, not r11/r20/r21): {}",
            report.diagnostics[0].message
        );
    }

    #[test]
    fn c1756_case26_repeated_verification_yields_identical_diagnostic() {
        let bytes = emit_test_function(vec![
            IrInstr::AddI32 {
                dst: 2,
                lhs: 10,
                rhs: 11,
            },
            IrInstr::Ret { src: Some(2) },
        ]);
        let first = verify_semcode(&bytes).expect_err("must reject");
        for _ in 0..9 {
            let again = verify_semcode(&bytes).expect_err("must reject");
            assert_eq!(
                again, first,
                "repeated verification of identical bytes must produce an identical diagnostic"
            );
        }
    }

    #[test]
    fn verifier_rejects_unknown_closure_target() {
        let mut bytes = compile_program_to_semcode(
            "fn main() { let add: Closure(f64 -> f64) = (x => x + 1.0); add(2.0); return; }",
        )
        .expect("compile");
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let (closure_name_offset, closure_name_len) = functions
            .iter()
            .find(|function| function.name.starts_with("__closure_"))
            .map(|function| (function.name_offset + 2, function.name.len()))
            .expect("closure function");
        let replacement = "x".repeat(closure_name_len);
        bytes[closure_name_offset..closure_name_offset + closure_name_len]
            .copy_from_slice(replacement.as_bytes());

        let report = verify_semcode(&bytes).expect_err("must reject dangling closure target");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UnknownCallTarget
        );
    }

    #[test]
    fn verifier_rejects_builtin_closure_target() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadF64 { dst: 0, val: 1.0 },
                    IrInstr::MakeClosure {
                        dst: 1,
                        name: "sqrt".to_string(),
                        captures: Vec::new(),
                    },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");

        let report = verify_semcode(&bytes).expect_err("must reject builtin closure target");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::UnknownCallTarget
        );
    }

    #[test]
    fn verifier_rejects_f64_ops_under_v0_capabilities() {
        let src = r#"
            fn main() {
                let x: f64 = 1.0 + 2.0;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let bytes = downgrade_header_stripping_signature(&bytes, MAGIC0);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::CapabilityViolation
        );
    }

    #[test]
    fn verifier_rejects_missing_ownership_section_under_v11_capabilities() {
        // #1773 (FA-09-005): hand-built directly under MAGIC11 with no OWN0
        // section at all - `compile_program_to_semcode`'s output now always
        // includes a real (if empty) OWN0 section (mandatory once its own
        // chosen header reaches V11's floor), so stripping only the SIG0
        // span (as `downgrade_header_stripping_signature` does for the
        // capability-downgrade tests above) would leave OWN0 genuinely
        // present here - the opposite of what this test needs to prove.
        let new_code = [Opcode::Ret as u8, 0];
        let bytes = minimal_semcode_bytes_with_header(MAGIC11, "main", &new_code);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipSection
        );
    }

    #[test]
    fn verifier_rejects_invalid_ownership_event_kind() {
        let mut bytes = ownership_semcode_bytes();
        let (_, code_start, code_end) = function_code_span(&bytes, "main");
        let code = &mut bytes[code_start..code_end];
        let section_offset = ownership_section_offset(code);
        let kind_offset = section_offset + 4 + 2;
        code[kind_offset] = 0xff;
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipSection
        );
    }

    #[test]
    fn verifier_rejects_unsupported_ownership_component_kind() {
        let mut bytes = ownership_semcode_bytes();
        let (_, code_start, code_end) = function_code_span(&bytes, "main");
        let code = &mut bytes[code_start..code_end];
        let section_offset = ownership_section_offset(code);
        // #1726 Checkpoint D2a: `ownership_semcode_bytes()`'s Borrow event
        // now always carries a resolved ActivationSiteId (SEMCOD20/rev21),
        // so its layout is tag(4)+count(2)+kind(1)+activation_mode(1)+
        // anchor(4)+root(4)+component_count(2) before the component kind
        // byte -- 5 bytes past the legacy (pre-D2a) offset.
        let component_kind_offset = section_offset + 4 + 2 + 1 + 1 + 4 + 4 + 2;
        code[component_kind_offset] = 0xff;
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipSection
        );
    }

    #[test]
    fn verifier_rejects_truncated_ownership_path_payload() {
        let mut bytes = ownership_semcode_bytes();
        let (code_len_pos, code_start, code_end) = function_code_span(&bytes, "main");
        let code = &bytes[code_start..code_end];
        let section_offset = ownership_section_offset(code);
        // See `verifier_rejects_unsupported_ownership_component_kind` for why
        // this Borrow entry's component-kind byte sits at `+18` under
        // SEMCOD20/rev21, not `+13` under the legacy layout.
        let truncated_code_len = section_offset + 4 + 2 + 1 + 1 + 4 + 4 + 2 + 1 + 1;
        bytes[code_len_pos..code_len_pos + 4]
            .copy_from_slice(&(truncated_code_len as u32).to_le_bytes());
        bytes.truncate(code_start + truncated_code_len);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipSection
        );
    }

    #[test]
    fn verifier_rejects_invalid_record_field_component_kind() {
        let mut bytes = record_field_borrow_semcode_bytes();
        let (_, code_start, code_end) = function_code_span(&bytes, "main");
        let code = &mut bytes[code_start..code_end];
        let section_offset = ownership_section_offset(code);
        // See `verifier_rejects_unsupported_ownership_component_kind` for why
        // this Borrow entry's component-kind byte sits at `+18` under
        // SEMCOD20/rev21, not `+13` under the legacy layout.
        let component_kind_offset = section_offset + 4 + 2 + 1 + 1 + 4 + 4 + 2;
        code[component_kind_offset] = 0xff;
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipSection
        );
    }

    #[test]
    fn verifier_rejects_truncated_record_field_payload() {
        let mut bytes = record_field_borrow_semcode_bytes();
        let (code_len_pos, code_start, code_end) = function_code_span(&bytes, "main");
        let code = &bytes[code_start..code_end];
        let section_offset = ownership_section_offset(code);
        // See `verifier_rejects_unsupported_ownership_component_kind` for why
        // this Borrow entry's component-kind byte sits at `+18` under
        // SEMCOD20/rev21, not `+13` under the legacy layout.
        let truncated_code_len = section_offset + 4 + 2 + 1 + 1 + 4 + 4 + 2 + 1 + 3;
        bytes[code_len_pos..code_len_pos + 4]
            .copy_from_slice(&(truncated_code_len as u32).to_le_bytes());
        bytes.truncate(code_start + truncated_code_len);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipSection
        );
    }

    #[test]
    fn verifier_accepts_sequence_index_static_ownership_semcode() {
        let bytes = sequence_index_static_borrow_semcode_bytes();
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_rejects_truncated_sequence_index_static_payload() {
        let mut bytes = sequence_index_static_borrow_semcode_bytes();
        let (code_len_pos, code_start, code_end) = function_code_span(&bytes, "main");
        let code = &bytes[code_start..code_end];
        let section_offset = ownership_section_offset(code);
        let truncated_code_len = section_offset + 4 + 2 + 1 + 4 + 2 + 1 + 3;
        bytes[code_len_pos..code_len_pos + 4]
            .copy_from_slice(&(truncated_code_len as u32).to_le_bytes());
        bytes.truncate(code_start + truncated_code_len);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipSection
        );
    }

    #[test]
    fn verifier_accepts_adt_payload_ownership_semcode() {
        let bytes = adt_payload_ownership_semcode_bytes();
        let verified = verify_semcode(&bytes).expect("verify");
        assert_eq!(verified.functions.len(), 2);
    }

    #[test]
    fn verifier_rejects_invalid_adt_payload_component_kind() {
        let mut bytes = adt_payload_ownership_semcode_bytes();
        let (_, code_start, code_end) = function_code_span(&bytes, "read_payload");
        let code = &mut bytes[code_start..code_end];
        let section_offset = ownership_section_offset(code);
        let component_kind_offset = section_offset + 4 + 2 + 1 + 4 + 2;
        code[component_kind_offset] = 0xff;
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipSection
        );
    }

    #[test]
    fn verifier_rejects_truncated_adt_payload_missing_variant() {
        let mut bytes = adt_payload_ownership_semcode_bytes();
        let (code_len_pos, code_start, code_end) = function_code_span(&bytes, "read_payload");
        let code = &bytes[code_start..code_end];
        let section_offset = ownership_section_offset(code);
        let truncated_code_len = section_offset + 4 + 2 + 1 + 4 + 2 + 1;
        bytes[code_len_pos..code_len_pos + 4]
            .copy_from_slice(&(truncated_code_len as u32).to_le_bytes());
        bytes.truncate(code_start + truncated_code_len);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipSection
        );
    }

    #[test]
    fn verifier_rejects_truncated_adt_payload_missing_index() {
        let mut bytes = adt_payload_ownership_semcode_bytes();
        let (code_len_pos, code_start, code_end) = function_code_span(&bytes, "read_payload");
        let code = &bytes[code_start..code_end];
        let section_offset = ownership_section_offset(code);
        let truncated_code_len = section_offset + 4 + 2 + 1 + 4 + 2 + 1 + 4;
        bytes[code_len_pos..code_len_pos + 4]
            .copy_from_slice(&(truncated_code_len as u32).to_le_bytes());
        bytes.truncate(code_start + truncated_code_len);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::InvalidOwnershipSection
        );
    }

    #[test]
    fn verifier_rejects_record_field_payload_under_v11_capabilities() {
        let bytes = record_field_borrow_semcode_bytes();
        let bytes = downgrade_header_stripping_signature(&bytes, MAGIC11);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::CapabilityViolation
        );
    }

    #[test]
    fn verifier_rejects_ownership_section_under_v10_capabilities() {
        let bytes = ownership_semcode_bytes();
        let bytes = downgrade_header_stripping_signature(&bytes, MAGIC10);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::CapabilityViolation
        );
    }

    #[test]
    fn verifier_rejects_state_query_under_v3_capabilities() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::StateQuery {
                        dst: 0,
                        key: "decision.mode".to_string(),
                    },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let bytes = downgrade_header_stripping_signature(&bytes, MAGIC3);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::CapabilityViolation
        );
    }

    #[test]
    fn verifier_rejects_state_update_under_v4_capabilities() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadBool { dst: 0, val: true },
                    IrInstr::StateUpdate {
                        key: "decision.mode".to_string(),
                        src: 0,
                    },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let bytes = downgrade_header_stripping_signature(&bytes, MAGIC4);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::CapabilityViolation
        );
    }

    #[test]
    fn verifier_rejects_event_post_under_v5_capabilities() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::EventPost {
                        signal: "alert.raised".to_string(),
                    },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let bytes = downgrade_header_stripping_signature(&bytes, MAGIC5);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::CapabilityViolation
        );
    }

    #[test]
    fn verifier_rejects_clock_read_under_v6_capabilities() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![IrInstr::ClockRead { dst: 0 }, IrInstr::Ret { src: None }],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let bytes = downgrade_header_stripping_signature(&bytes, MAGIC6);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::CapabilityViolation
        );
    }

    #[test]
    fn verifier_rejects_text_under_v7_capabilities() {
        let src = r#"
            fn main() {
                let left: text = "alpha";
                let right: text = "alpha";
                assert(left == right);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let bytes = downgrade_header_stripping_signature(&bytes, MAGIC7);
        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::CapabilityViolation
        );
    }

    fn ownership_semcode_bytes() -> Vec<u8> {
        let src = r#"
            fn pair() -> (i32, i32) = (1, 2);

            fn main() {
                let pair: (i32, i32) = pair();
                let (ref left, _): (i32, i32) = pair;
                let total: f64 = 0.0;
                total += 1.0;
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    fn record_field_borrow_semcode_bytes() -> Vec<u8> {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext { camera: T, quality: 0.75 };
                let DecisionContext { camera: ref seen_camera, quality: _ } = ctx;
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    fn sequence_index_static_borrow_semcode_bytes() -> Vec<u8> {
        let mut bytes = record_field_borrow_semcode_bytes();
        let (_, code_start, code_end) = function_code_span(&bytes, "main");
        let code = &mut bytes[code_start..code_end];
        let section_offset = ownership_section_offset(code);
        // #1726 Checkpoint D2a: `record_field_borrow_semcode_bytes()` now
        // always promotes to SEMCOD20/rev21 (a frozen-producer Borrow event
        // always carries a resolved ActivationSiteId), so its Borrow entry's
        // layout is tag(4) + count(2) + kind(1) + activation_mode(1) +
        // anchor(4) + root(4) + component_count(2) before the first
        // component's own kind byte - 5 bytes further than the legacy
        // (pre-D2a) layout this offset used to assume. Verified by direct
        // decode inspection, not by re-deriving it a second time here: get
        // this wrong and the patch silently corrupts a different field
        // (root_symbol_id, in the pre-fix version of this helper) while the
        // test still passes for the wrong reason.
        let component_kind_offset = section_offset + 4 + 2 + 1 + 1 + 4 + 4 + 2;
        code[component_kind_offset] =
            sm_format::semcode_format::OWNERSHIP_PATH_COMPONENT_SEQUENCE_INDEX;
        // #1718: a real emitter faced with a `SequenceIndexStatic` component
        // would promote the header to `HEADER_V21`/`MAGIC21` (see
        // `has_v21_sequence_ownership_events` in `sm-ir`) - this helper
        // patches the component-kind byte in isolation (the base fixture's
        // real header only reflects a Field Borrow), so it must patch the
        // header magic to match what a real compiler would have chosen for
        // this content, or the resulting artifact would understate its own
        // capability requirement and get rejected for a reason unrelated to
        // whatever the calling test actually wants to exercise.
        bytes[0..8].copy_from_slice(&sm_format::semcode_format::MAGIC21);
        bytes
    }

    fn record_field_write_semcode_bytes() -> Vec<u8> {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext { camera: T, quality: 0.75 };
                let patched: DecisionContext = ctx with { quality: 1.0 };
                let _ = patched;
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    // #1891 Checkpoint W2E fixtures.

    fn multi_field_record_update_write_semcode_bytes() -> Vec<u8> {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext { camera: T, quality: 0.75 };
                let patched: DecisionContext = ctx with { camera: F, quality: 1.0 };
                let _ = patched;
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    fn repeated_assignment_write_semcode_bytes() -> Vec<u8> {
        let src = r#"
            fn main() {
                let mut x: i32 = 0;
                x = 1;
                x = 2;
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    fn mixed_store_var_and_make_record_write_semcode_bytes() -> Vec<u8> {
        let src = r#"
            record R {
                a: i32,
            }

            fn main() {
                let mut x: i32 = 0;
                x = 5;
                let base: R = R { a: 1 };
                let fresh: R = base with { a: 9 };
                let _ = fresh;
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    fn adt_payload_ownership_semcode_bytes() -> Vec<u8> {
        let src = r#"
            enum Maybe {
                None,
                Some(f64),
            }

            fn read_payload(value: Maybe) -> f64 {
                let ret: f64 = match value {
                    Maybe::None => { 0.0 }
                    Maybe::Some(ref inner) => {
                        let v: f64 = inner;
                        v
                    }
                };
                return ret;
            }

            fn main() {
                let out: f64 = read_payload(Maybe::Some(2.5));
                assert(out == 2.5);
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    fn function_code_span(bytes: &[u8], target: &str) -> (usize, usize, usize) {
        let mut cursor = 8usize;
        while cursor < bytes.len() {
            let name_len = read_u16_le(bytes, &mut cursor).expect("name length") as usize;
            let name = std::str::from_utf8(&bytes[cursor..cursor + name_len]).expect("utf8 name");
            cursor += name_len;
            let code_len_pos = cursor;
            let code_len = read_u32_le(bytes, &mut cursor).expect("code length") as usize;
            let code_start = cursor;
            let code_end = cursor + code_len;
            if name == target {
                return (code_len_pos, code_start, code_end);
            }
            cursor = code_end;
        }
        panic!("function '{target}' not found");
    }

    fn skip_string_table(code: &[u8]) -> usize {
        let mut cursor = 0usize;
        let count = read_u16_le(code, &mut cursor).expect("string count") as usize;
        for _ in 0..count {
            let len = read_u16_le(code, &mut cursor).expect("string length") as usize;
            cursor += len;
        }
        cursor
    }

    /// #1773 (FA-09-005): rebuilds `bytes` under `target_magic`, stripping
    /// each function's `SIG0` section along the way. Every artifact produced
    /// by `compile_program_to_semcode`/`emit_ir_to_semcode` now
    /// unconditionally carries a `SIG0` section (the new floor is
    /// `SEMCODE_SIGNATURE_MIN_REVISION`), so simply overwriting the header
    /// magic bytes to simulate an older revision - the technique several
    /// pre-#1773 tests in this module used - leaves a structurally
    /// inconsistent artifact: a header claiming a revision that structurally
    /// cannot carry `SIG0`, over a body that still does. This reconstructs
    /// what the body would genuinely have looked like had it been compiled
    /// under an older header, matching real pre-#1773 artifact shape.
    fn encode_legacy_ownership_component(
        code: &mut Vec<u8>,
        component: &sm_format::semcode_decode::DecodedAccessPathComponent,
    ) {
        use sm_format::semcode_decode::DecodedAccessPathComponent;
        match component {
            DecodedAccessPathComponent::TupleIndex(index) => {
                code.push(sm_format::semcode_format::OWNERSHIP_PATH_COMPONENT_TUPLE_INDEX);
                code.extend_from_slice(&index.to_le_bytes());
            }
            DecodedAccessPathComponent::FieldSymbol(symbol) => {
                code.push(sm_format::semcode_format::OWNERSHIP_PATH_COMPONENT_FIELD_SYMBOL);
                code.extend_from_slice(&symbol.to_le_bytes());
            }
            DecodedAccessPathComponent::AdtPayload { variant, index } => {
                code.push(sm_format::semcode_format::OWNERSHIP_PATH_COMPONENT_ADT_PAYLOAD);
                code.extend_from_slice(&variant.to_le_bytes());
                code.extend_from_slice(&index.to_le_bytes());
            }
            DecodedAccessPathComponent::SequenceIndexStatic(index) => {
                code.push(sm_format::semcode_format::OWNERSHIP_PATH_COMPONENT_SEQUENCE_INDEX);
                code.extend_from_slice(&index.to_le_bytes());
            }
        }
    }

    /// #1726 Checkpoint D2a: every one of this helper's 8 call sites
    /// downgrades to a revision below `SEMCODE_OWNERSHIP_ANCHOR_MIN_REVISION`
    /// (the highest, MAGIC11, is rev12; the threshold is rev21) - so any
    /// OWN0 section is always rebuilt in the *legacy* grammar (no activation
    /// prefix on Borrow events), from the already-decoded structured paths,
    /// rather than copied as raw bytes. Copying raw bytes silently broke this
    /// helper the moment a source artifact's OWN0 became rev21-shaped
    /// (SEMCOD20): the downgraded header's own revision correctly says
    /// "legacy grammar", but the copied bytes were still rev21-shaped,
    /// producing a decode-shape mismatch (`InvalidOwnershipSection`) before
    /// the capability check these tests exist to exercise ever ran. Content
    /// (root/components) is preserved exactly; only the wire shape and the
    /// cross-kind interleaving order are not (this helper does not need to
    /// preserve Borrow/Write order - that is #1726/#1891's own concern,
    /// orthogonal to what these capability-downgrade tests check).
    fn downgrade_header_stripping_signature(bytes: &[u8], target_magic: [u8; 8]) -> Vec<u8> {
        let (_, functions) = sm_format::semcode_decode::decode_semcode_envelope(bytes)
            .expect("decode current bytes");
        let mut out = Vec::new();
        out.extend_from_slice(&target_magic);
        for f in &functions {
            let mut code = Vec::new();
            code.extend_from_slice(&(f.strings.len() as u16).to_le_bytes());
            for s in &f.strings {
                code.extend_from_slice(&(s.len() as u16).to_le_bytes());
                code.extend_from_slice(s.as_bytes());
            }
            if f.has_debug_section {
                code.extend_from_slice(b"DBG0");
                code.extend_from_slice(&(f.debug_symbols.len() as u16).to_le_bytes());
                for d in &f.debug_symbols {
                    code.extend_from_slice(&(d.pc as u32).to_le_bytes());
                    code.extend_from_slice(&d.line.to_le_bytes());
                    code.extend_from_slice(&d.col.to_le_bytes());
                }
            }
            if f.has_ownership_section {
                code.extend_from_slice(&OWNERSHIP_SECTION_TAG);
                let total = f.borrowed_paths.len() + f.write_paths.len();
                code.extend_from_slice(&(total as u16).to_le_bytes());
                for p in &f.borrowed_paths {
                    code.push(sm_format::semcode_format::OWNERSHIP_EVENT_KIND_BORROW);
                    code.extend_from_slice(&p.root_symbol_id.to_le_bytes());
                    code.extend_from_slice(&(p.components.len() as u16).to_le_bytes());
                    for c in &p.components {
                        encode_legacy_ownership_component(&mut code, c);
                    }
                }
                for p in &f.write_paths {
                    code.push(sm_format::semcode_format::OWNERSHIP_EVENT_KIND_WRITE);
                    code.extend_from_slice(&p.root_symbol_id.to_le_bytes());
                    code.extend_from_slice(&(p.components.len() as u16).to_le_bytes());
                    for c in &p.components {
                        encode_legacy_ownership_component(&mut code, c);
                    }
                }
            }
            let name_bytes = f.name.as_bytes();
            out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(&(code.len() as u32).to_le_bytes());
            out.extend_from_slice(&code);
        }
        out
    }

    /// #1891 Checkpoint W2E, item 9.G support: identical to
    /// `downgrade_header_stripping_signature` (same legacy string/DBG0/OWN0
    /// re-encode, same unconditional omission of the rev21+ Write
    /// mode+anchor bytes for every Write event) except it also appends each
    /// function's real instruction bytes (`f.code_slice[f.instr_start_offset..]`)
    /// after its re-encoded header sections. `downgrade_header_stripping_signature`
    /// itself never copies instruction bytes at all - harmless for its own 8
    /// call sites, which only ever assert an early `CapabilityViolation`
    /// rejection that never reaches code execution, but fatal for a test
    /// that needs the downgraded artifact to fully verify AND admit as a
    /// genuine legacy Write. A prior attempt reused
    /// `downgrade_header_stripping_signature` directly for exactly that
    /// purpose and observed `ReachableFunctionFallthrough` on every
    /// function - the direct symptom of an empty instruction stream -
    /// confirming that helper was never meant for this.
    fn downgrade_to_legacy_write_preserving_code(bytes: &[u8], target_magic: [u8; 8]) -> Vec<u8> {
        let (_, functions) = sm_format::semcode_decode::decode_semcode_envelope(bytes)
            .expect("decode current bytes");
        let mut out = Vec::new();
        out.extend_from_slice(&target_magic);
        for f in &functions {
            let mut code = Vec::new();
            code.extend_from_slice(&(f.strings.len() as u16).to_le_bytes());
            for s in &f.strings {
                code.extend_from_slice(&(s.len() as u16).to_le_bytes());
                code.extend_from_slice(s.as_bytes());
            }
            if f.has_debug_section {
                code.extend_from_slice(b"DBG0");
                code.extend_from_slice(&(f.debug_symbols.len() as u16).to_le_bytes());
                for d in &f.debug_symbols {
                    code.extend_from_slice(&(d.pc as u32).to_le_bytes());
                    code.extend_from_slice(&d.line.to_le_bytes());
                    code.extend_from_slice(&d.col.to_le_bytes());
                }
            }
            if f.has_ownership_section {
                code.extend_from_slice(&OWNERSHIP_SECTION_TAG);
                let total = f.borrowed_paths.len() + f.write_paths.len();
                code.extend_from_slice(&(total as u16).to_le_bytes());
                for p in &f.borrowed_paths {
                    code.push(sm_format::semcode_format::OWNERSHIP_EVENT_KIND_BORROW);
                    code.extend_from_slice(&p.root_symbol_id.to_le_bytes());
                    code.extend_from_slice(&(p.components.len() as u16).to_le_bytes());
                    for c in &p.components {
                        encode_legacy_ownership_component(&mut code, c);
                    }
                }
                for p in &f.write_paths {
                    code.push(sm_format::semcode_format::OWNERSHIP_EVENT_KIND_WRITE);
                    code.extend_from_slice(&p.root_symbol_id.to_le_bytes());
                    code.extend_from_slice(&(p.components.len() as u16).to_le_bytes());
                    for c in &p.components {
                        encode_legacy_ownership_component(&mut code, c);
                    }
                }
            }
            code.extend_from_slice(&f.code_slice[f.instr_start_offset..]);
            let name_bytes = f.name.as_bytes();
            out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(&(code.len() as u32).to_le_bytes());
            out.extend_from_slice(&code);
        }
        out
    }

    /// #1773 (FA-09-005): a minimal, fully hand-built artifact - explicit
    /// header magic, empty string table, caller-supplied raw instruction
    /// bytes - with no OWN0/SIG0 section at all. Several pre-#1773 tests
    /// patched `compile_program_to_semcode`'s output at a hand-counted byte
    /// offset assuming the instruction stream began immediately after a
    /// bare empty string table; that assumption broke once every compiled
    /// artifact unconditionally gained OWN0+SIG0 sections. This sidesteps
    /// the compiler entirely for tests that need full, explicit control
    /// over byte layout under an arbitrary (possibly pre-#1732) header.
    fn minimal_semcode_bytes_with_header(magic: [u8; 8], name: &str, instrs: &[u8]) -> Vec<u8> {
        let mut code = Vec::new();
        code.extend_from_slice(&0u16.to_le_bytes()); // empty string table
        code.extend_from_slice(instrs);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&magic);
        bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(code.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&code);
        bytes
    }

    fn ownership_section_offset(code: &[u8]) -> usize {
        let cursor = skip_string_table(code);
        assert!(cursor + OWNERSHIP_SECTION_TAG.len() <= code.len());
        assert_eq!(
            &code[cursor..cursor + OWNERSHIP_SECTION_TAG.len()],
            OWNERSHIP_SECTION_TAG
        );
        cursor
    }

    /// #1726 Checkpoint D2b test support: walks a rev21+ OWN0 section
    /// exactly like `sm_format::semcode_decode` does (tag, count, then each
    /// event's kind/mode/[anchor]/root/component_count/components) and
    /// returns the byte offset (within `code`) of the 4-byte anchor field
    /// for every Borrow/StoreVarSite event, in encounter order. Deliberately
    /// re-derives the layout by walking it, the same discipline
    /// `sequence_index_static_borrow_semcode_bytes` now documents after its
    /// own hand-counted offset once silently drifted onto
    /// `root_symbol_id`'s byte instead of the field it claimed to target
    /// (the pre-D2a corruption `find_instruction`'s doc comment on this same
    /// class of mistake also warns about) - every offset this returns is
    /// re-verified against the real decoder's own view by every call site
    /// below before it is trusted.
    fn store_var_site_anchor_field_offsets(code: &[u8]) -> Vec<usize> {
        use sm_format::semcode_format::{
            ACTIVATION_MODE_STORE_VAR_SITE, OWNERSHIP_EVENT_KIND_BORROW,
            OWNERSHIP_EVENT_KIND_WRITE, OWNERSHIP_PATH_COMPONENT_ADT_PAYLOAD,
            OWNERSHIP_PATH_COMPONENT_FIELD_SYMBOL, OWNERSHIP_PATH_COMPONENT_SEQUENCE_INDEX,
            OWNERSHIP_PATH_COMPONENT_TUPLE_INDEX,
        };
        let section_offset = ownership_section_offset(code);
        let mut cursor = section_offset + OWNERSHIP_SECTION_TAG.len();
        let count = u16::from_le_bytes([code[cursor], code[cursor + 1]]) as usize;
        cursor += 2;
        let mut offsets = Vec::new();
        for _ in 0..count {
            let kind = code[cursor];
            cursor += 1;
            // Mirrors `sm_format::semcode_decode`'s own gate exactly: the
            // activation-mode byte exists for Borrow-kind events, and (as of
            // #1891 Checkpoint W2D) an execution-mode byte exists for
            // Write-kind events too - both at the same wire position, both
            // gated at the same revision. This function only cares about
            // Borrow/StoreVarSite anchor offsets, but it must still correctly
            // skip a Write event's own mode(+anchor) bytes to keep the
            // cursor aligned for every event after it; reading neither (as
            // an earlier version of this helper did for Write) silently
            // misparses the very next event's own `root_symbol_id` byte,
            // corrupting every offset after it.
            let is_store_var_site = if kind == OWNERSHIP_EVENT_KIND_BORROW {
                let mode = code[cursor];
                cursor += 1;
                mode == ACTIVATION_MODE_STORE_VAR_SITE
            } else {
                if kind == OWNERSHIP_EVENT_KIND_WRITE {
                    cursor += 1; // write execution mode
                    cursor += 4; // write executable anchor
                }
                false
            };
            if is_store_var_site {
                offsets.push(cursor);
                cursor += 4;
            }
            cursor += 4; // root_symbol_id
            let component_count = u16::from_le_bytes([code[cursor], code[cursor + 1]]) as usize;
            cursor += 2;
            for _ in 0..component_count {
                let tag = code[cursor];
                cursor += 1;
                cursor += match tag {
                    t if t == OWNERSHIP_PATH_COMPONENT_TUPLE_INDEX => 2,
                    t if t == OWNERSHIP_PATH_COMPONENT_FIELD_SYMBOL => 4,
                    t if t == OWNERSHIP_PATH_COMPONENT_ADT_PAYLOAD => 6,
                    t if t == OWNERSHIP_PATH_COMPONENT_SEQUENCE_INDEX => 4,
                    other => panic!("unknown ownership path component tag {other}"),
                };
            }
        }
        offsets
    }

    /// Overwrites the `occurrence`-th (0-indexed) Borrow/StoreVarSite
    /// event's anchor field in `function_name`'s OWN0 section with `anchor`,
    /// then re-decodes the whole artifact and asserts the new decoded
    /// anchor value is exactly `anchor` and every other event's activation
    /// is byte-for-byte unchanged - so a wrong offset fails loudly here
    /// (wrong field mutated, or an unrelated event disturbed) rather than
    /// silently producing a test that passes for the wrong reason.
    fn set_store_var_site_anchor(
        bytes: &mut [u8],
        function_name: &str,
        occurrence: usize,
        anchor: u32,
    ) {
        let before = {
            let (_, decoded) =
                sm_format::semcode_decode::decode_semcode_envelope(bytes).expect("decode before");
            decoded
                .iter()
                .find(|f| f.name == function_name)
                .expect("function")
                .borrowed_paths
                .clone()
        };
        let (_, code_start, code_end) = function_code_span(bytes, function_name);
        let code = &mut bytes[code_start..code_end];
        let offsets = store_var_site_anchor_field_offsets(code);
        let field_offset = *offsets.get(occurrence).unwrap_or_else(|| {
            panic!(
                "expected occurrence {occurrence}, found only {} StoreVarSite event(s)",
                offsets.len()
            )
        });
        code[field_offset..field_offset + 4].copy_from_slice(&anchor.to_le_bytes());
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(bytes).expect("decode after");
        let after = &decoded
            .iter()
            .find(|f| f.name == function_name)
            .expect("function")
            .borrowed_paths;
        for (index, (before_path, after_path)) in before.iter().zip(after.iter()).enumerate() {
            if index == occurrence {
                assert_eq!(
                    after_path.activation,
                    Some(sm_format::semcode_decode::DecodedBorrowActivation::StoreVarSite(anchor)),
                    "the intended event's anchor did not change to the intended value"
                );
                assert_eq!(before_path.root_symbol_id, after_path.root_symbol_id);
                assert_eq!(before_path.components, after_path.components);
            } else {
                assert_eq!(
                    before_path, after_path,
                    "an unrelated event's fields changed - wrong offset computed"
                );
            }
        }
    }

    /// Analogous to `store_var_site_anchor_field_offsets`, but records the
    /// 4-byte anchor field offset for every rev21+ Write event, whichever
    /// mode it declares - both `WRITE_EXECUTION_MODE_STORE_VAR_SITE` and
    /// `_MAKE_RECORD_SITE` carry an anchor immediately after their mode
    /// byte, unlike Borrow's `FrameEntry` mode, which carries none.
    fn write_execution_anchor_field_offsets(code: &[u8]) -> Vec<usize> {
        use sm_format::semcode_format::{
            ACTIVATION_MODE_STORE_VAR_SITE, OWNERSHIP_EVENT_KIND_BORROW,
            OWNERSHIP_EVENT_KIND_WRITE, OWNERSHIP_PATH_COMPONENT_ADT_PAYLOAD,
            OWNERSHIP_PATH_COMPONENT_FIELD_SYMBOL, OWNERSHIP_PATH_COMPONENT_SEQUENCE_INDEX,
            OWNERSHIP_PATH_COMPONENT_TUPLE_INDEX,
        };
        let section_offset = ownership_section_offset(code);
        let mut cursor = section_offset + OWNERSHIP_SECTION_TAG.len();
        let count = u16::from_le_bytes([code[cursor], code[cursor + 1]]) as usize;
        cursor += 2;
        let mut offsets = Vec::new();
        for _ in 0..count {
            let kind = code[cursor];
            cursor += 1;
            let is_write_with_anchor = if kind == OWNERSHIP_EVENT_KIND_BORROW {
                let mode = code[cursor];
                cursor += 1;
                if mode == ACTIVATION_MODE_STORE_VAR_SITE {
                    cursor += 4;
                }
                false
            } else if kind == OWNERSHIP_EVENT_KIND_WRITE {
                cursor += 1; // write execution mode - either value carries an anchor
                true
            } else {
                false
            };
            if is_write_with_anchor {
                offsets.push(cursor);
                cursor += 4;
            }
            cursor += 4; // root_symbol_id
            let component_count = u16::from_le_bytes([code[cursor], code[cursor + 1]]) as usize;
            cursor += 2;
            for _ in 0..component_count {
                let tag = code[cursor];
                cursor += 1;
                cursor += match tag {
                    t if t == OWNERSHIP_PATH_COMPONENT_TUPLE_INDEX => 2,
                    t if t == OWNERSHIP_PATH_COMPONENT_FIELD_SYMBOL => 4,
                    t if t == OWNERSHIP_PATH_COMPONENT_ADT_PAYLOAD => 6,
                    t if t == OWNERSHIP_PATH_COMPONENT_SEQUENCE_INDEX => 4,
                    other => panic!("unknown ownership path component tag {other}"),
                };
            }
        }
        offsets
    }

    /// Overwrites the `occurrence`-th (0-indexed) Write event's execution
    /// anchor field in `function_name`'s OWN0 section with `anchor`, keeping
    /// its declared mode (StoreVarSite/MakeRecordSite) unchanged, then
    /// re-decodes and asserts the new decoded anchor is exactly `anchor` and
    /// every other event is byte-for-byte unchanged - mirrors
    /// `set_store_var_site_anchor`'s exact discipline for the Write domain.
    fn set_write_execution_site_anchor(
        bytes: &mut [u8],
        function_name: &str,
        occurrence: usize,
        anchor: u32,
    ) {
        let before = {
            let (_, decoded) =
                sm_format::semcode_decode::decode_semcode_envelope(bytes).expect("decode before");
            decoded
                .iter()
                .find(|f| f.name == function_name)
                .expect("function")
                .write_paths
                .clone()
        };
        let (_, code_start, code_end) = function_code_span(bytes, function_name);
        let code = &mut bytes[code_start..code_end];
        let offsets = write_execution_anchor_field_offsets(code);
        let field_offset = *offsets.get(occurrence).unwrap_or_else(|| {
            panic!(
                "expected occurrence {occurrence}, found only {} Write execution-site event(s)",
                offsets.len()
            )
        });
        code[field_offset..field_offset + 4].copy_from_slice(&anchor.to_le_bytes());
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(bytes).expect("decode after");
        let after = &decoded
            .iter()
            .find(|f| f.name == function_name)
            .expect("function")
            .write_paths;
        for (index, (before_path, after_path)) in before.iter().zip(after.iter()).enumerate() {
            if index == occurrence {
                let expected = match before_path.write_execution {
                    Some(sm_format::semcode_decode::DecodedWriteExecution::StoreVarSite(_)) => {
                        sm_format::semcode_decode::DecodedWriteExecution::StoreVarSite(anchor)
                    }
                    Some(sm_format::semcode_decode::DecodedWriteExecution::MakeRecordSite(_)) => {
                        sm_format::semcode_decode::DecodedWriteExecution::MakeRecordSite(anchor)
                    }
                    None => panic!(
                        "occurrence {occurrence} has no write_execution to mutate (legacy revision?)"
                    ),
                };
                assert_eq!(
                    after_path.write_execution,
                    Some(expected),
                    "the intended event's anchor did not change to the intended value"
                );
                assert_eq!(before_path.root_symbol_id, after_path.root_symbol_id);
                assert_eq!(before_path.components, after_path.components);
            } else {
                assert_eq!(
                    before_path, after_path,
                    "an unrelated event's fields changed - wrong offset computed"
                );
            }
        }
    }

    /// Converts a real, decoded instruction offset (via `find_instruction`,
    /// which returns a position absolute within the whole artifact) into the
    /// function-relative `ExecutableAnchor` domain (`sm-vm`'s `Frame.pc`
    /// domain: a byte offset relative to the function's own `instr_start`).
    fn anchor_of(bytes: &[u8], function_name: &str, opcode: Opcode, occurrence: usize) -> u32 {
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(bytes).expect("decode");
        let env = functions
            .iter()
            .find(|f| f.name == function_name)
            .expect("function");
        let absolute = find_instruction(bytes, function_name, opcode, occurrence);
        u32::try_from(absolute - env.code_offset - env.instr_start_offset).expect("anchor fits u32")
    }

    /// Locates the absolute byte offset of the `occurrence`-th (0-indexed)
    /// instance of `opcode` within `function_name`'s actual decoded
    /// instruction stream.
    ///
    /// Walks instruction-by-instruction from the real `instr_start_offset`,
    /// reusing the same decode logic (`decode_semcode_envelope`,
    /// `decode_operands`) the verifier's own admission path uses, instead of
    /// scanning raw bytes for the first matching value anywhere in the
    /// buffer. This cannot match a header, function-name, string-table,
    /// debug-section, or ownership-section byte, and cannot match an earlier
    /// instruction's operand byte, because those are never mistaken for an
    /// opcode-position read during the walk.
    ///
    /// Panics with a precise message if the function isn't found, if
    /// decoding fails, or if fewer than `occurrence + 1` matches exist -
    /// never silently selects an arbitrary occurrence when more than one
    /// exists.
    fn find_instruction(
        bytes: &[u8],
        function_name: &str,
        opcode: Opcode,
        occurrence: usize,
    ) -> usize {
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(bytes).expect("decode semcode");
        let env = functions
            .iter()
            .find(|f| f.name == function_name)
            .unwrap_or_else(|| panic!("function '{function_name}' not found"));

        let code = env.code_slice;
        let mut cursor = env.instr_start_offset;
        let mut matches = Vec::new();
        while cursor < code.len() {
            let instr_offset = cursor;
            let raw = read_u8(code, &mut cursor).expect("read opcode byte");
            let decoded = Opcode::from_byte(raw).expect("valid opcode");
            if decoded == opcode {
                matches.push(instr_offset);
            }
            decode_operands(
                function_name,
                code,
                &mut cursor,
                instr_offset,
                decoded,
                true,
                &mut NoCollect,
                &mut NoCollect,
            )
            .expect("decode operands");
        }

        let relative = *matches.get(occurrence).unwrap_or_else(|| {
            panic!(
                "expected occurrence {occurrence} of {opcode:?} in '{function_name}', found \
                 {} match(es) at {matches:?}",
                matches.len()
            )
        });
        env.code_offset + relative
    }

    #[test]
    fn verify_semcode_token_accepts_no_main_helper_semcode() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "helper".to_string(),
                instrs: vec![IrInstr::Ret { src: None }],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let token = verify_semcode_token(&bytes).expect("token admission");
        assert!(token.has_entry("helper"));
        assert!(!token.has_entry("main"));
        assert_eq!(token.function_names().collect::<Vec<_>>(), vec!["helper"]);
    }

    #[test]
    fn verify_semcode_token_metadata_matches_verify_semcode() {
        let src = "fn main() { return; }";
        let bytes = compile_program_to_semcode(src).expect("compile");
        let metadata = verify_semcode(&bytes).expect("verify_semcode");
        let token = verify_semcode_token(&bytes).expect("token admission");
        assert_eq!(token.program(), &metadata);
    }

    #[test]
    fn verify_semcode_token_rejects_malformed_header() {
        let report = verify_semcode_token(b"SEMC").expect_err("must reject");
        assert_eq!(report.diagnostics[0].code, VerificationCode::BadHeader);
    }

    // #1736 (FA-05-006): before the sm-format checked-arithmetic repair, this
    // exact artifact shape (a claimed function code length that overflows
    // cursor arithmetic) could panic inside `decode_semcode_envelope` on a
    // 32-bit target instead of surfacing as a `RejectReport` - i.e. the
    // verifier boundary was not actually panic-safe against malformed input.
    #[test]
    fn verify_semcode_token_rejects_overflowing_code_length_without_panicking() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&sm_format::semcode_format::MAGIC0);
        bytes.extend_from_slice(&4u16.to_le_bytes());
        bytes.extend_from_slice(b"main");
        bytes.extend_from_slice(&0xFFFFFFF0u32.to_le_bytes());
        let report = verify_semcode_token(&bytes).expect_err("must reject, never panic");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::TruncatedFunction
        );
    }

    #[test]
    fn verified_semcode_token_keeps_original_bytes() {
        let src = "fn main() { return; }";
        let bytes = compile_program_to_semcode(src).expect("compile");
        let token = verify_semcode_token(&bytes).expect("token admission");
        assert_eq!(token.bytes(), bytes.as_slice());
    }

    #[test]
    fn verify_semcode_token_rejects_duplicate_function_names() {
        let mut bytes = compile_program_to_semcode("fn main() { return; }").expect("compile");
        let function_block = bytes[8..].to_vec();
        bytes.extend_from_slice(&function_block);
        let report = verify_semcode_token(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::DuplicateFunction
        );
    }

    #[test]
    fn require_entry_main_ok_for_executable_semcode() {
        let src = "fn main() { return; }";
        let bytes = compile_program_to_semcode(src).expect("compile");
        let token = verify_semcode_token(&bytes).expect("token admission");
        let entry_token = token.require_entry("main").expect("require_entry");
        assert_eq!(entry_token.entry(), "main");
    }

    #[test]
    fn require_entry_main_fails_for_helper_only_semcode() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "helper".to_string(),
                instrs: vec![IrInstr::Ret { src: None }],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let token = verify_semcode_token(&bytes).expect("token admission");
        let err = token.require_entry("main").expect_err("should fail");
        assert_eq!(
            err,
            EntryResolutionError::MissingEntry {
                entry: "main".to_string()
            }
        );
    }

    #[test]
    fn require_entry_helper_ok_for_helper_only_semcode() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "helper".to_string(),
                instrs: vec![IrInstr::Ret { src: None }],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let token = verify_semcode_token(&bytes).expect("token admission");
        let entry_token = token.require_entry("helper").expect("require_entry");
        assert_eq!(entry_token.entry(), "helper");
    }

    #[test]
    fn missing_entry_error_is_not_reject_report() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "helper".to_string(),
                instrs: vec![IrInstr::Ret { src: None }],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        let token = verify_semcode_token(&bytes).expect("token admission");
        let err = token.require_entry("main").expect_err("should fail");
        let error_msg = err.to_string();
        assert!(error_msg.contains("entry function 'main' not found"));
    }

    #[test]
    fn verified_entry_token_reuses_original_artifact() {
        let src = "fn main() { return; }";
        let bytes = compile_program_to_semcode(src).expect("compile");
        let token = verify_semcode_token(&bytes).expect("token admission");
        let entry_token = token.require_entry("main").expect("require_entry");
        assert_eq!(entry_token.bytes(), token.bytes());
        assert_eq!(entry_token.program(), token.program());
    }

    // #1732 (FA-05-002): QTruth opcodes (0x17..0x1A) were introduced after
    // the SEMCODE0 baseline vocabulary (see docs/roadmap/core_quad/) but
    // carry no capability bit and no minimum-header-revision gate, so a
    // baseline-header artifact containing them was previously admitted.
    // This is the RED->GREEN regression for that gap: same fixture the old
    // `verifier_accepts_unsupported_qtruth_opcodes` test used (a trivial
    // `fn main() { return; }` baseline-header artifact with its opcode
    // stream patched to raw QTruth bytes), but the artifact's header
    // revision (SEMCODE0, rev 1) is below QTruth's minimum (SEMCOD18,
    // rev 19), so admission must now fail closed.
    #[test]
    fn verifier_rejects_qtruth_opcodes_under_baseline_header() {
        // #1773 (FA-09-005): hand-built directly under MAGIC0 rather than
        // patching `compile_program_to_semcode`'s output - every compiled
        // artifact now unconditionally carries OWN0+SIG0 sections, so a
        // hand-counted byte offset assuming the instruction stream starts
        // right after a bare string table no longer lands on real opcode
        // bytes. See `minimal_semcode_bytes_with_header`'s doc comment.
        let mut new_code = Vec::new();
        // QTruthAnd (0x17)
        new_code.push(0x17);
        new_code.extend_from_slice(&0u16.to_le_bytes()); // dst
        new_code.extend_from_slice(&0u16.to_le_bytes()); // lhs
        new_code.extend_from_slice(&0u16.to_le_bytes()); // rhs
                                                         // QTruthOr (0x18)
        new_code.push(0x18);
        new_code.extend_from_slice(&0u16.to_le_bytes()); // dst
        new_code.extend_from_slice(&0u16.to_le_bytes()); // lhs
        new_code.extend_from_slice(&0u16.to_le_bytes()); // rhs
                                                         // QTruthNot (0x19)
        new_code.push(0x19);
        new_code.extend_from_slice(&0u16.to_le_bytes()); // dst
        new_code.extend_from_slice(&0u16.to_le_bytes()); // src
                                                         // QTruthImpl (0x1A)
        new_code.push(0x1A);
        new_code.extend_from_slice(&0u16.to_le_bytes()); // dst
        new_code.extend_from_slice(&0u16.to_le_bytes()); // lhs
        new_code.extend_from_slice(&0u16.to_le_bytes()); // rhs
                                                         // Ret
        new_code.push(Opcode::Ret as u8);
        new_code.push(0);

        let bytes = minimal_semcode_bytes_with_header(MAGIC0, "main", &new_code);

        assert_eq!(&bytes[0..8], b"SEMCODE0");
        let report = verify_semcode(&bytes).expect_err("must reject QTruth under baseline header");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::OpcodeRequiresNewerHeader
        );
    }

    // #1732 regression matrix (2): the exact same QTruth opcode stream must
    // be accepted once the artifact carries SEMCOD18 (QTruth's actual
    // minimum header revision) - proving the gate is a real revision
    // comparison, not a blanket QTruth rejection.
    #[test]
    fn verifier_accepts_qtruth_opcodes_under_semcod18_header() {
        // #1773 (FA-09-005): hand-built directly under MAGIC18 - see the
        // comment on `verifier_rejects_qtruth_opcodes_under_baseline_header`.

        // SEMCOD18 carries CAP_OWNERSHIP_PATHS (inherited from SEMCOD11+),
        // which requires a present (possibly empty) OWN0 section.
        let mut new_code = Vec::new();
        new_code.extend_from_slice(b"OWN0");
        new_code.extend_from_slice(&0u16.to_le_bytes()); // empty ownership event count
        new_code.push(0x17); // QTruthAnd
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.push(0x18); // QTruthOr
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.push(0x19); // QTruthNot
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.push(0x1A); // QTruthImpl
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.push(Opcode::Ret as u8);
        new_code.push(0);

        let bytes = minimal_semcode_bytes_with_header(MAGIC18, "main", &new_code);

        let verified = verify_semcode(&bytes).expect("QTruth under SEMCOD18 must verify");
        assert_eq!(verified.functions.len(), 1);
    }

    #[test]
    fn verifier_rejects_truncated_qtruth_opcodes() {
        // #1773 (FA-09-005): hand-built directly under MAGIC18 - see the
        // comment on `verifier_rejects_qtruth_opcodes_under_baseline_header`.
        // Use SEMCOD18 (QTruth's actual minimum header) so this test
        // isolates truncation handling from the #1732 header-revision gate
        // - both are legitimate rejections, but this test is specifically
        // about truncated operand bytes, not about the header revision.
        // SEMCOD18 carries CAP_OWNERSHIP_PATHS, which requires a present
        // (possibly empty) OWN0 section.
        let mut new_code = Vec::new();
        new_code.extend_from_slice(b"OWN0");
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.push(0x17); // QTruthAnd (requires 4 operand bytes, zero left)
        let bytes = minimal_semcode_bytes_with_header(MAGIC18, "main", &new_code);

        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::OperandOutOfBounds
        );
    }

    // Structural validity must be established before the header-revision
    // gate applies: a truncated QTruth instruction under a baseline
    // (SEMCODE0) header is not "structurally valid but from an older
    // header" - decode_operands cannot even establish it as a complete
    // instruction - so it must reject as OperandOutOfBounds, not
    // OpcodeRequiresNewerHeader, per docs/spec/verifier.md's diagnostic
    // contract for that code.
    #[test]
    fn verifier_rejects_truncated_qtruth_opcodes_under_baseline_header_as_operand_error() {
        // #1773 (FA-09-005): hand-built directly under MAGIC0 - see the
        // comment on `verifier_rejects_qtruth_opcodes_under_baseline_header`.
        let new_code = [0x17u8]; // QTruthAnd (requires 4 operand bytes, zero left)
        let bytes = minimal_semcode_bytes_with_header(MAGIC0, "main", &new_code);
        assert_eq!(&bytes[0..8], b"SEMCODE0");

        let report = verify_semcode(&bytes).expect_err("must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::OperandOutOfBounds
        );
    }

    // #1732 regression matrix (4/5): historically-baseline opcodes -
    // including the legacy lattice `QAnd`/`QOr`/`QNot`/`QImpl` opcodes
    // QTruth sits right next to - have minimum_semcode_revision() == 1, so
    // they must remain fully admissible under SEMCODE0, completely
    // unaffected by the new header-revision gate.
    #[test]
    fn verifier_accepts_baseline_logic_opcodes_under_semcode0_unchanged() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadI32 { dst: 0, val: 1 },
                    IrInstr::LoadI32 { dst: 1, val: 0 },
                    IrInstr::QAnd {
                        dst: 2,
                        lhs: 0,
                        rhs: 1,
                    },
                    IrInstr::QOr {
                        dst: 3,
                        lhs: 0,
                        rhs: 1,
                    },
                    IrInstr::QNot { dst: 4, src: 0 },
                    IrInstr::QImpl {
                        dst: 5,
                        lhs: 0,
                        rhs: 1,
                    },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        // #1773 (FA-09-005): SEMCOD19/rev20 is now the floor for every
        // emitted artifact regardless of which opcodes it uses (was
        // SEMCODE0/rev1, this program's own promotion floor) - see the
        // analogous sm-ir comment.
        assert_eq!(&bytes[0..8], b"SEMCOD19");
        let verified = verify_semcode(&bytes).expect("baseline logic opcodes must verify");
        assert_eq!(verified.header.rev, 20);
    }

    // #1732 regression matrix (6): a program mixing QTruth with an
    // unrelated capability-gated feature (here, f64 math, which alone
    // would only require SEMCODE1) must select the MAXIMUM required
    // header revision - originally SEMCOD18/rev19, now SEMCOD19/rev20
    // since #1773 (FA-09-005) raised the floor for every emitted artifact
    // regardless of opcodes used - not whichever feature's if/else-if
    // branch happens to match first.
    #[test]
    fn emitter_selects_maximum_required_revision_for_mixed_program() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadF64 { dst: 0, val: 1.5 },
                    IrInstr::QTruthAnd {
                        dst: 1,
                        lhs: 0,
                        rhs: 0,
                    },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        assert_eq!(&bytes[0..8], b"SEMCOD19");
        let verified = verify_semcode(&bytes).expect("mixed QTruth+f64 program must verify");
        assert_eq!(verified.header.rev, 20);
    }

    // #1732 regression matrix (3): the canonical emitter itself must choose
    // a header whose revision actually covers QTruth, not just structurally
    // valid bytes hand-patched by a test. Uses emit_ir_to_semcode (IR
    // level) so this doesn't depend on the source front-end's own QTruth
    // admission being wired for every feature-gate combination.
    #[test]
    fn emitter_promotes_qtruth_only_program_to_semcod18() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    // #1756 (FA-07-016): r0 must be definitely defined before
                    // `QTruthAnd` reads it as both `lhs` and `rhs` - this
                    // fixture only exercises header-revision promotion for
                    // the opcode, not runtime QTruth semantics, so a plain
                    // `LoadI32` satisfies definedness without needing this
                    // module to import `QuadVal` for a real `LoadQ`.
                    IrInstr::LoadI32 { dst: 0, val: 0 },
                    IrInstr::QTruthAnd {
                        dst: 0,
                        lhs: 0,
                        rhs: 0,
                    },
                    IrInstr::Ret { src: None },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit");
        // #1773 (FA-09-005): SEMCOD19/rev20 is now the floor (was SEMCOD18/
        // rev19, QTruth's own promotion floor) - see the comment on
        // `emitter_selects_maximum_required_revision_for_mixed_program`.
        assert_eq!(&bytes[0..8], b"SEMCOD19");
        let verified = verify_semcode(&bytes).expect("emitted QTruth program must verify");
        assert_eq!(verified.header.rev, 20);
    }

    // #1751 (FA-07-011): `pending_functions.len()` is a static count of
    // function definitions in the whole program; `quotas.max_frames` is a
    // dynamic runtime call-stack-depth budget enforced independently (and
    // correctly) at execution time in sm-vm's `push_frame`. Comparing the
    // two conflated "how many functions exist" with "how deep can the call
    // stack get". sm-format already owns the real static function-count
    // bound (`MAX_FUNCTIONS = 1024`, enforced at decode time), so the
    // verifier must not duplicate or misuse `max_frames` for this purpose.
    #[test]
    fn verify_semcode_token_accepts_many_function_definitions_regardless_of_frame_quota() {
        // Control: previously accepted (256 total functions, at the old
        // static bound) and must remain accepted.
        let bytes_256 = compile_many_functions(255);
        verify_semcode_token(&bytes_256).expect("256 functions must be accepted");

        // #1751 repro: previously rejected with ResourceLimitExceeded citing
        // "verified-local frame budget of 256" purely because the program
        // defines more functions than the runtime frame quota - a static
        // function count has nothing to do with live call-stack depth, so
        // this must now be accepted.
        let bytes_258 = compile_many_functions(257);
        verify_semcode_token(&bytes_258)
            .expect("function count must not be checked against the runtime frame quota");
    }

    fn compile_many_functions(extra_fn_count: usize) -> Vec<u8> {
        let mut src = String::new();
        for i in 0..extra_fn_count {
            src.push_str(&format!("fn fn_{i}() {{ return; }}\n"));
        }
        src.push_str("fn main() { return; }");
        compile_program_to_semcode(&src).expect("compile")
    }

    // #1754 (FA-07-014) test matrix: the verifier must sum DISTINCT
    // runtime symbols (function-local string-table entries, deduplicated
    // by exact value) across the WHOLE program against max_symbol_table
    // (16_384), matching the VM's single shared `RuntimeSymbolTable`
    // (crates/sm-runtime-core/src/lib.rs), not a per-function check.

    fn function_with_strings(name: &str, strings: &[String]) -> IrFunction {
        let mut instrs: Vec<IrInstr> = strings
            .iter()
            .map(|s| IrInstr::LoadText {
                dst: 0,
                val: s.clone(),
            })
            .collect();
        instrs.push(IrInstr::Ret { src: None });
        IrFunction {
            name: name.to_string(),
            instrs,
            ownership_events: Vec::new(),
            params: Vec::new(),
        }
    }

    /// Union of every function's local strings, deduplicated by value -
    /// the same quantity `RuntimeSymbolTable` (via
    /// `build_vm_program_view_from_decoded`) ends up holding.
    fn distinct_runtime_symbol_count(bytes: &[u8]) -> usize {
        let (_, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(bytes).expect("decode");
        decoded
            .iter()
            .flat_map(|env| env.strings.iter().map(|s| s.as_str()))
            .collect::<HashSet<_>>()
            .len()
    }

    #[test]
    fn verifier_accepts_heavy_duplication_under_program_wide_symbol_quota() {
        // 165 functions x 100 strings, but every function reuses the exact
        // same 100 string values: raw per-function-summed entries
        // (16_500) exceed max_symbol_table, while the DISTINCT program-wide
        // count (100) stays far under it. A naive "sum all local string
        // table lengths" implementation would wrongly reject this; the
        // correct dedup-by-value implementation must accept it.
        let shared_strings: Vec<String> = (0..100).map(|j| format!("dup_s{j}")).collect();
        let funcs: Vec<IrFunction> = (0..165)
            .map(|i| function_with_strings(&format!("f{i}"), &shared_strings))
            .collect();
        let bytes = emit_ir_to_semcode(&funcs, false).expect("emit");

        let raw_sum: usize = funcs.len() * shared_strings.len();
        assert!(
            raw_sum > 16_384,
            "raw sum must exceed the quota to be a meaningful control"
        );
        let distinct = distinct_runtime_symbol_count(&bytes);
        assert_eq!(distinct, 100);
        assert!(distinct <= 16_384);

        verify_semcode_token(&bytes)
            .expect("duplicate-heavy program under the distinct-symbol quota must be accepted");
    }

    #[test]
    fn verifier_accepts_program_wide_symbols_at_exact_quota_boundary() {
        // 64 functions x 256 strings, all globally distinct, each function
        // exactly at sm-format's MAX_STRINGS_PER_FUNCTION cap: 64 * 256 =
        // 16_384, exactly max_symbol_table. This hits the boundary exactly.
        let functions_count = 64usize;
        let strings_per_function = 256usize;
        assert_eq!(functions_count * strings_per_function, 16_384);
        assert_eq!(
            strings_per_function,
            sm_format::semcode_decode::MAX_STRINGS_PER_FUNCTION
        );

        let funcs: Vec<IrFunction> = (0..functions_count)
            .map(|i| {
                let strings: Vec<String> = (0..strings_per_function)
                    .map(|j| format!("b{i}_{j}"))
                    .collect();
                function_with_strings(&format!("f{i}"), &strings)
            })
            .collect();
        let bytes = emit_ir_to_semcode(&funcs, false).expect("emit");

        let distinct = distinct_runtime_symbol_count(&bytes);
        assert_eq!(distinct, 16_384);

        verify_semcode_token(&bytes)
            .expect("program exactly at the symbol quota boundary must be accepted");
    }

    #[test]
    fn verifier_rejects_program_wide_symbols_over_quota() {
        // Permanent regression test for the #1754 reproduction: 66
        // functions x 250 globally-distinct strings each (66 * 250 =
        // 16_500 > 16_384), with every individual function's local string
        // table staying at 250 <= MAX_STRINGS_PER_FUNCTION (256) - proving
        // this is caught only by GLOBAL, program-wide accounting, not by
        // any per-function limit.
        let functions_count = 66usize;
        let strings_per_function = 250usize;
        assert!(strings_per_function <= sm_format::semcode_decode::MAX_STRINGS_PER_FUNCTION);
        assert!(functions_count <= 256); // stay under the unrelated #1751 max_frames check

        let funcs: Vec<IrFunction> = (0..functions_count)
            .map(|i| {
                let strings: Vec<String> = (0..strings_per_function)
                    .map(|j| format!("f{i}_s{j}"))
                    .collect();
                function_with_strings(&format!("f{i}"), &strings)
            })
            .collect();
        let bytes = emit_ir_to_semcode(&funcs, false).expect("emit");

        let distinct = distinct_runtime_symbol_count(&bytes);
        assert_eq!(distinct, functions_count * strings_per_function);
        assert!(distinct > 16_384);

        let report = verify_semcode_token(&bytes)
            .expect_err("program-wide distinct runtime symbol count over quota must reject");
        assert_eq!(
            report.diagnostics[0].code,
            VerificationCode::ResourceLimitExceeded
        );
        assert!(
            report.diagnostics[0]
                .message
                .contains("program-wide runtime symbol table"),
            "diagnostic message must name the program-wide runtime symbol table: {}",
            report.diagnostics[0].message
        );
    }
}
