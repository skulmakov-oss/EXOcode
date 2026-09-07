use crate::semcode_format::{
    read_f64_le, read_i32_le, read_u16_le, read_u32_le, read_u8, CallableSignature,
    CallableValueFamily, Opcode, SemcodeFormatError, SemcodeHeaderSpec,
};
use crate::QuadVal;
use prom_abi::{
    AbiError, AbiFailureKind, AbiValue, ApplicationHostAbi, HostCallId, PrometheusHostAbi,
};
use prom_cap::{CapabilityChecker, CapabilityDenied};
use semantic_core_quad::{QuadState, QuadroReg32};
use sm_runtime_core::hello_observation_sink::{
    HelloObservationClass, HelloObservationEvent, HelloObservationSequenceIndex,
};
use sm_runtime_core::{
    AccessPath, AdtCarrier, ExecutionConfig, ExecutionContext, QuotaExceeded, QuotaKind,
    RecordCarrier, RuntimeQuotas, RuntimeSymbolTable, RuntimeTrap, SymbolId,
};
use sm_verify::RejectReport;
use sm_verify::{
    verify_semcode_token, verify_semcode_token_with_quotas, EntryResolutionError,
    VerifiedEntrySemCode,
};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Scalar key type for Map values.
///
/// Only scalar types that support deterministic equality are admitted as map keys.
/// This mirrors the type-checker's key guard (i32|u32|bool|text|quad) and avoids
/// making `Value` universally hashable or ordered.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MapKey {
    I32(i32),
    U32(u32),
    Bool(bool),
    Text(String),
    Quad(u8),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClosureValue {
    pub function_name: String,
    pub captures: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Quad(QuadVal),
    Bool(bool),
    Text(String),
    Sequence(Vec<Value>),
    Map(Vec<(MapKey, Value)>),
    Closure(ClosureValue),
    I32(i32),
    F64(f64),
    U32(u32),
    Fx(i32),
    Tuple(Vec<Value>),
    Record(RecordCarrier<Value>),
    Adt(AdtCarrier<Value>),
    Unit,
}

/// A single register slot.
///
/// Physical slot existence (storage capacity) and Semantic value
/// definition are distinct properties: a slot may exist without ever
/// having been written by an instruction. `Value::Unit` is an ordinary
/// Semantic value, never a stand-in for "nothing was written here" --
/// that state is represented explicitly by `Uninitialized`.
#[derive(Debug, Clone)]
enum RegisterSlot {
    Uninitialized,
    Value(Value),
}

/// #1726 Checkpoint D3: a Borrow event's runtime activation authority,
/// mirroring `sm_format::semcode_decode::DecodedBorrowActivation` in the VM's
/// own domain. `StoreVarSite`'s `usize` is frame-relative - the same
/// `Frame.pc` domain (byte offset relative to `instr_start`, the
/// opcode-byte position of the instruction being executed) D1 established
/// and D2b verified the anchor against; it is never a string-table index,
/// lowered-local symbol, or IR/event ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowActivation {
    FrameEntry,
    StoreVarSite(usize),
}

/// A decoded Borrow path paired with its runtime activation authority. Legacy
/// (pre-rev21) artifacts and explicit rev21 `FrameEntry` events both map to
/// `BorrowActivation::FrameEntry` here - the wire distinction between "no
/// activation field at all" and "explicit FrameEntry" is a D2a decode-layer
/// concern; by the time a path reaches the VM, both mean the same runtime
/// thing: active immediately at frame creation.
#[derive(Debug, Clone)]
pub struct BorrowedPath {
    pub path: AccessPath,
    pub activation: BorrowActivation,
}

/// A `StoreVarSite` Borrow not yet activated: it becomes active only after
/// the real StoreVar instruction at `anchor` (this frame's own PC domain)
/// successfully commits its write. Frame-local, non-static - never a
/// global/shared cursor (per Checkpoint D3 item 3).
#[derive(Debug, Clone)]
struct PendingSiteBorrow {
    anchor: usize,
    path: AccessPath,
}

/// #1891 Checkpoint W2F: every rev21+ Write ownership path attached to one
/// exact, verifier-authenticated (`StoreVarSite`/`MakeRecordSite`, Checkpoint
/// W2E) executable PC, grouped once when `FunctionBytecode` is built and
/// never mutated afterward. `pc` is the same frame-relative opcode-start
/// byte offset domain as `Frame.pc`/`BorrowActivation::StoreVarSite`/
/// `PendingSiteBorrow::anchor`. `paths` may hold more than one entry - W2A/W2E
/// already established that a single MakeRecord site legitimately carries
/// 1..N Write paths (multi-field `RecordUpdate`) - in the artifact's own
/// declared event order, never reordered. The runtime never needs to know
/// whether `pc` is a StoreVar or a MakeRecord instruction: W2E already
/// proved the anchor matches its declared opcode class, and re-deriving that
/// here would be a second verifier inside the VM.
#[derive(Debug, Clone)]
struct WriteExecutionSite {
    pc: usize,
    paths: Vec<AccessPath>,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub pc: usize,
    regs: Vec<RegisterSlot>,
    pub locals: HashMap<SymbolId, Value>,
    /// Borrow paths already active in this frame: `FrameEntry` paths from the
    /// moment the frame was pushed, plus any `StoreVarSite` paths whose
    /// anchor has since been visited and successfully committed.
    pub active_borrowed_paths: Vec<AccessPath>,
    /// `StoreVarSite` Borrow paths not yet activated in this frame. Checked
    /// (and, on a match, drained into `active_borrowed_paths`) only from
    /// inside the `StoreVar` opcode handler, after that instruction's own
    /// write has already committed.
    pending_site_borrows: Vec<PendingSiteBorrow>,
    pub func: String,
    pub return_dst: Option<u16>,
}

#[derive(Debug, Clone)]
pub struct FunctionBytecode {
    pub name: String,
    pub strings: Vec<String>,
    pub symbol_ids: Vec<SymbolId>,
    pub debug_symbols: Vec<DebugSymbol>,
    pub borrowed_paths: Vec<BorrowedPath>,
    /// #1891 Checkpoint W2F: sorted ascending by `pc` (built once via a
    /// `BTreeMap`, never mutated), so the dispatch loop's per-instruction
    /// lookup is `binary_search_by_key` - no allocation, no rebuilding, no
    /// HashMap-order nondeterminism.
    write_execution_sites: Vec<WriteExecutionSite>,
    pub code: Vec<u8>,
    pub instr_start: usize,
    /// The decoded canonical callable-signature record (#1773 / FA-09-005),
    /// retained unchanged from `DecodedFunctionEnvelope::signature`. `None`
    /// means the artifact's header predates canonical signatures
    /// (`SEMCODE_SIGNATURE_MIN_REVISION`) - `validate_call_arguments` treats
    /// that as "nothing to enforce", preserving pre-#1773 behavior exactly.
    pub signature: Option<CallableSignature>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugSymbol {
    pub pc: usize,
    pub line: u32,
    pub col: u16,
}

#[derive(Debug, Clone)]
pub struct VM {
    pub functions: HashMap<String, FunctionBytecode>,
    pub callstack: Vec<Frame>,
    pub config: ExecutionConfig,
    pub effect_calls: usize,
    /// #1759: execution-wide opcode-dispatch fuel counter, per
    /// `ssf08_1759_steps_calls_contract_decision.md` - charged once per
    /// successfully decoded opcode in `exec_loop_with_profile`, before that
    /// instruction's semantic body runs. Never reset by frame entry/exit.
    pub steps: usize,
    /// #1759: execution-wide admitted non-root call counter, per
    /// `ssf08_1759_steps_calls_contract_decision.md` - charged once per
    /// `push_frame` invocation that occurs while the callstack is already
    /// non-empty. The root/entry frame never charges this counter.
    pub calls: usize,
    pub symbols: RuntimeSymbolTable,
    /// PRNG state for random_seed / random_next_i32 (xorshift64; 0 = unseeded).
    pub prng_state: u64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum VmTestStatus {
    Completed,
    Failed,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct VmTestObservation {
    status: VmTestStatus,
    observable: Option<String>,
    trap: Option<String>,
}

#[cfg(test)]
thread_local! {
    static VM_TEST_TERMINAL_OBSERVATION: std::cell::RefCell<Option<VmTestObservation>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn vm_test_clear_terminal_observation() {
    VM_TEST_TERMINAL_OBSERVATION.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

#[cfg(test)]
fn vm_test_store_terminal_observation(observation: VmTestObservation) {
    VM_TEST_TERMINAL_OBSERVATION.with(|slot| {
        *slot.borrow_mut() = Some(observation);
    });
}

#[cfg(test)]
fn vm_test_take_terminal_observation() -> Option<VmTestObservation> {
    VM_TEST_TERMINAL_OBSERVATION.with(|slot| slot.borrow_mut().take())
}

#[cfg(test)]
fn vm_test_format_terminal_observable(
    ret_val: &Value,
    frame: &Frame,
    symbols: &sm_runtime_core::RuntimeSymbolTable,
) -> String {
    let mut locals = frame
        .locals
        .iter()
        .map(|(symbol, value)| {
            let name = symbols.resolve(*symbol).unwrap_or("<unknown>").to_string();
            let value = format!("{value:?}");
            (name, value)
        })
        .collect::<Vec<_>>();
    locals.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    let locals = if locals.is_empty() {
        "[]".to_string()
    } else {
        let joined = locals
            .into_iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("[{joined}]")
    };
    format!("return={ret_val:?}; locals={locals}")
}
trait OpcodeProfileSink {
    fn record_opcode(&mut self, opcode: Opcode);
}
struct NoopOpcodeProfile;

impl OpcodeProfileSink for NoopOpcodeProfile {
    #[inline(always)]
    fn record_opcode(&mut self, _opcode: Opcode) {}
}

#[cfg(feature = "vm-profile")]
const OPCODE_PROFILE_SLOT_COUNT: usize = 73;

#[cfg(feature = "vm-profile")]
const OPCODE_PROFILE_OPCODES: [Opcode; OPCODE_PROFILE_SLOT_COUNT] = [
    Opcode::LoadQ,
    Opcode::LoadBool,
    Opcode::LoadI32,
    Opcode::AddI32,
    Opcode::SubI32,
    Opcode::MulI32,
    Opcode::DivI32,
    Opcode::ModI32,
    Opcode::ConcatText,
    Opcode::LoadU32,
    Opcode::LoadVar,
    Opcode::StoreVar,
    Opcode::QAnd,
    Opcode::QOr,
    Opcode::QNot,
    Opcode::QImpl,
    Opcode::BoolAnd,
    Opcode::BoolOr,
    Opcode::BoolNot,
    Opcode::CmpEq,
    Opcode::CmpNe,
    Opcode::CmpI32Lt,
    Opcode::CmpI32Le,
    Opcode::Jmp,
    Opcode::JmpIf,
    Opcode::Call,
    Opcode::Ret,
    Opcode::Assert,
    Opcode::MakeTuple,
    Opcode::TupleGet,
    Opcode::MakeRecord,
    Opcode::RecordGet,
    Opcode::MakeAdt,
    Opcode::AdtTag,
    Opcode::AdtGet,
    Opcode::LoadF64,
    Opcode::AddF64,
    Opcode::SubF64,
    Opcode::MulF64,
    Opcode::DivF64,
    Opcode::LoadFx,
    Opcode::AddFx,
    Opcode::SubFx,
    Opcode::MulFx,
    Opcode::DivFx,
    Opcode::LoadText,
    Opcode::MakeSequence,
    Opcode::SequenceGet,
    Opcode::MakeClosure,
    Opcode::ClosureCall,
    Opcode::SequenceLen,
    Opcode::SequenceIsEmpty,
    Opcode::SequenceContains,
    Opcode::SequencePush,
    Opcode::SequencePrepend,
    Opcode::SequencePop,
    Opcode::MapEmpty,
    Opcode::MapContains,
    Opcode::MapGet,
    Opcode::MapSet,
    Opcode::RngSeed,
    Opcode::RngNextI32,
    Opcode::GateRead,
    Opcode::GateWrite,
    Opcode::PulseEmit,
    Opcode::StateQuery,
    Opcode::StateUpdate,
    Opcode::EventPost,
    Opcode::ClockRead,
    Opcode::QTruthAnd,
    Opcode::QTruthOr,
    Opcode::QTruthNot,
    Opcode::QTruthImpl,
];

#[cfg(feature = "vm-profile")]
fn opcode_profile_index(opcode: Opcode) -> usize {
    match opcode {
        Opcode::LoadQ => 0,
        Opcode::LoadBool => 1,
        Opcode::LoadI32 => 2,
        Opcode::AddI32 => 3,
        Opcode::SubI32 => 4,
        Opcode::MulI32 => 5,
        Opcode::DivI32 => 6,
        Opcode::ModI32 => 7,
        Opcode::ConcatText => 8,
        Opcode::LoadU32 => 9,
        Opcode::LoadVar => 10,
        Opcode::StoreVar => 11,
        Opcode::QAnd => 12,
        Opcode::QOr => 13,
        Opcode::QNot => 14,
        Opcode::QImpl => 15,
        Opcode::BoolAnd => 16,
        Opcode::BoolOr => 17,
        Opcode::BoolNot => 18,
        Opcode::CmpEq => 19,
        Opcode::CmpNe => 20,
        Opcode::CmpI32Lt => 21,
        Opcode::CmpI32Le => 22,
        Opcode::Jmp => 23,
        Opcode::JmpIf => 24,
        Opcode::Call => 25,
        Opcode::Ret => 26,
        Opcode::Assert => 27,
        Opcode::MakeTuple => 28,
        Opcode::TupleGet => 29,
        Opcode::MakeRecord => 30,
        Opcode::RecordGet => 31,
        Opcode::MakeAdt => 32,
        Opcode::AdtTag => 33,
        Opcode::AdtGet => 34,
        Opcode::LoadF64 => 35,
        Opcode::AddF64 => 36,
        Opcode::SubF64 => 37,
        Opcode::MulF64 => 38,
        Opcode::DivF64 => 39,
        Opcode::LoadFx => 40,
        Opcode::AddFx => 41,
        Opcode::SubFx => 42,
        Opcode::MulFx => 43,
        Opcode::DivFx => 44,
        Opcode::LoadText => 45,
        Opcode::MakeSequence => 46,
        Opcode::SequenceGet => 47,
        Opcode::MakeClosure => 48,
        Opcode::ClosureCall => 49,
        Opcode::SequenceLen => 50,
        Opcode::SequenceIsEmpty => 51,
        Opcode::SequenceContains => 52,
        Opcode::SequencePush => 53,
        Opcode::SequencePrepend => 54,
        Opcode::SequencePop => 55,
        Opcode::MapEmpty => 56,
        Opcode::MapContains => 57,
        Opcode::MapGet => 58,
        Opcode::MapSet => 59,
        Opcode::RngSeed => 60,
        Opcode::RngNextI32 => 61,
        Opcode::GateRead => 62,
        Opcode::GateWrite => 63,
        Opcode::PulseEmit => 64,
        Opcode::StateQuery => 65,
        Opcode::StateUpdate => 66,
        Opcode::EventPost => 67,
        Opcode::ClockRead => 68,
        Opcode::QTruthAnd => 69,
        Opcode::QTruthOr => 70,
        Opcode::QTruthNot => 71,
        Opcode::QTruthImpl => 72,
    }
}

#[cfg(feature = "vm-profile")]
#[allow(dead_code)]
fn opcode_from_profile_index(index: usize) -> Option<Opcode> {
    OPCODE_PROFILE_OPCODES.get(index).copied()
}

#[cfg(feature = "vm-profile")]
#[derive(Debug, Clone)]
pub struct VmOpcodeProfile {
    total_instructions: u64,
    opcode_counts: [u64; OPCODE_PROFILE_SLOT_COUNT],
}

#[cfg(feature = "vm-profile")]
impl Default for VmOpcodeProfile {
    fn default() -> Self {
        Self {
            total_instructions: 0,
            opcode_counts: [0; OPCODE_PROFILE_SLOT_COUNT],
        }
    }
}

#[cfg(feature = "vm-profile")]
impl VmOpcodeProfile {
    pub fn total_instructions(&self) -> u64 {
        self.total_instructions
    }

    pub fn count(&self, opcode: Opcode) -> u64 {
        self.opcode_counts[opcode_profile_index(opcode)]
    }

    pub fn is_empty(&self) -> bool {
        self.total_instructions == 0
    }

    pub fn top_n(&self, n: usize) -> Vec<(Opcode, u64)> {
        if n == 0 {
            return Vec::new();
        }
        let mut entries: Vec<(usize, Opcode, u64)> = OPCODE_PROFILE_OPCODES
            .iter()
            .enumerate()
            .filter_map(|(index, &opcode)| {
                let count = self.opcode_counts[index];
                (count > 0).then_some((index, opcode, count))
            })
            .collect();
        entries.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        entries
            .into_iter()
            .take(n)
            .map(|(_, opcode, count)| (opcode, count))
            .collect()
    }

    pub fn summary_top_n(&self, n: usize) -> String {
        use std::fmt::Write as _;

        let top = self.top_n(n);
        let mut out = String::new();
        let _ = write!(
            &mut out,
            "sm-vm opcode profile: total_instructions={} top={}",
            self.total_instructions,
            top.len()
        );
        for (opcode, count) in top {
            let _ = write!(&mut out, "\n  {:?}: {}", opcode, count);
        }
        out
    }

    fn record_opcode_slot(&mut self, opcode: Opcode) {
        self.total_instructions = self.total_instructions.saturating_add(1);
        let index = opcode_profile_index(opcode);
        self.opcode_counts[index] = self.opcode_counts[index].saturating_add(1);
    }
}

#[cfg(feature = "vm-profile")]
impl OpcodeProfileSink for VmOpcodeProfile {
    #[inline(always)]
    fn record_opcode(&mut self, opcode: Opcode) {
        self.record_opcode_slot(opcode);
    }
}

enum HelloObservationMode<'a> {
    Discard,
    Collect(&'a mut Vec<HelloObservationEvent>),
}

struct HelloObservationRuntime<'a> {
    mode: HelloObservationMode<'a>,
    sequence_index: u64,
}

impl<'a> HelloObservationRuntime<'a> {
    fn discard() -> Self {
        Self {
            mode: HelloObservationMode::Discard,
            sequence_index: 0,
        }
    }

    fn collect(events: &'a mut Vec<HelloObservationEvent>) -> Self {
        Self {
            mode: HelloObservationMode::Collect(events),
            sequence_index: 0,
        }
    }

    fn record_controlled_text_observation(&mut self, text: String) -> Result<(), RuntimeError> {
        if matches!(
            text.as_str(),
            "stdout" | "print" | "io.write" | "file" | "network" | "stdin"
        ) {
            return Err(RuntimeError::TypeMismatchRuntime(format!(
                "builtin 'print' does not admit host-output marker '{}'",
                text
            )));
        }

        let event = HelloObservationEvent {
            operation_kind: "controlled_observation_text",
            observation_class: HelloObservationClass::ControlledText,
            text,
            sequence_index: HelloObservationSequenceIndex(self.sequence_index),
        };
        self.sequence_index += 1;
        if let HelloObservationMode::Collect(events) = &mut self.mode {
            events.push(event);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    BadHeader,
    UnsupportedBytecodeVersion { found: String, supported: String },
    BadFormat(String),
    UnknownFunction(String),
    InvalidJumpAddress { func: String, addr: usize },
    TypeMismatchRuntime(String),
    StackUnderflow,
    StackOverflow,
    QuotaExceeded(QuotaExceeded),
    VerifierRejected(RejectReport),
    UnknownVariable(String),
    InvalidStringId(u16),
    HostAbi(AbiError),
    CapabilityDenied(CapabilityDenied),

    Trap(RuntimeTrap),
}

impl core::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RuntimeError::BadHeader => write!(f, "bad SemCode header"),
            RuntimeError::UnsupportedBytecodeVersion { found, supported } => write!(
                f,
                "unsupported SemCode version '{}'; supported versions: {}. hint: recompile source with current semantic",
                found, supported
            ),
            RuntimeError::BadFormat(m) => write!(f, "bad SemCode format: {}", m),
            RuntimeError::UnknownFunction(n) => write!(f, "unknown function '{}'", n),
            RuntimeError::InvalidJumpAddress { func, addr } => {
                write!(f, "invalid jump address {} in '{}'", addr, func)
            }
            RuntimeError::TypeMismatchRuntime(m) => write!(f, "runtime type mismatch: {}", m),
            RuntimeError::StackUnderflow => write!(f, "stack underflow"),
            RuntimeError::StackOverflow => write!(f, "stack overflow"),
            RuntimeError::QuotaExceeded(exceeded) => write!(
                f,
                "quota exceeded: {:?} limit={} used={}",
                exceeded.kind, exceeded.limit, exceeded.used
            ),
            RuntimeError::VerifierRejected(report) => write!(f, "{report}"),
            RuntimeError::UnknownVariable(n) => write!(f, "unknown variable '{}'", n),
            RuntimeError::InvalidStringId(id) => write!(f, "invalid string id {}", id),
            RuntimeError::HostAbi(err) => write!(f, "{err}"),
            RuntimeError::CapabilityDenied(err) => write!(f, "{err}"),

            RuntimeError::Trap(RuntimeTrap::AssertionFailed) => write!(f, "assertion failed"),
            RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict) => {
                write!(f, "write path overlaps active borrow")
            }
            RuntimeError::Trap(trap) => write!(f, "runtime trap: {:?}", trap),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Raw execution path.
///
/// Bypasses `sm-verify` admission and executes raw SemCode bytes.
/// Intentionally retained for tests, diagnostics, and malformed-byte behavior.
/// Must not be confused with verified execution.
pub fn run_semcode(bytes: &[u8]) -> Result<(), RuntimeError> {
    run_semcode_with_config(
        bytes,
        ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
    )
}

/// Compatibility verified-bytes shim.
///
/// Preserves older byte-based call sites. Internally performs admission through
/// `verify_semcode_token`, resolves the entry via `require_entry`, and delegates
/// to canonical token execution. It should not be preferred for new internal callers.
pub fn run_verified_semcode(bytes: &[u8]) -> Result<(), RuntimeError> {
    let token = verify_semcode_token(bytes).map_err(RuntimeError::VerifierRejected)?;
    let entry_token = token.require_entry("main").map_err(|err| match err {
        EntryResolutionError::MissingEntry { entry } => RuntimeError::UnknownFunction(entry),
    })?;
    run_verified_entry_semcode_with_config(
        &entry_token,
        ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
    )
}

/// Raw execution path.
///
/// Bypasses `sm-verify` admission and executes raw SemCode bytes.
/// Intentionally retained for tests, diagnostics, and malformed-byte behavior.
/// Must not be confused with verified execution.
pub fn run_semcode_collecting_hello_observations(
    bytes: &[u8],
) -> Result<Vec<HelloObservationEvent>, RuntimeError> {
    let mut events = Vec::new();
    let collected = run_semcode_with_entry_and_config_with_observation_runtime(
        bytes,
        "main",
        ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
        HelloObservationRuntime::collect(&mut events),
    )?;
    debug_assert!(events.is_empty());
    Ok(collected)
}

/// Raw execution path.
///
/// Bypasses `sm-verify` admission and executes raw SemCode bytes.
/// Intentionally retained for tests, diagnostics, and malformed-byte behavior.
/// Must not be confused with verified execution.
pub fn run_semcode_with_entry(bytes: &[u8], entry: &str) -> Result<(), RuntimeError> {
    run_semcode_with_entry_and_config(
        bytes,
        entry,
        ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
    )
}

/// Raw execution path.
///
/// Bypasses `sm-verify` admission and executes raw SemCode bytes.
/// Intentionally retained for tests, diagnostics, and malformed-byte behavior.
/// Must not be confused with verified execution.
pub fn run_semcode_with_config(bytes: &[u8], config: ExecutionConfig) -> Result<(), RuntimeError> {
    run_semcode_with_entry_and_config(bytes, "main", config)
}

/// Compatibility verified-bytes shim.
///
/// Preserves older byte-based call sites. Internally performs admission through
/// `verify_semcode_token`, resolves the entry via `require_entry`, and delegates
/// to canonical token execution. It should not be preferred for new internal callers.
pub fn run_verified_semcode_with_config(
    bytes: &[u8],
    config: ExecutionConfig,
) -> Result<(), RuntimeError> {
    let token = verify_semcode_token_with_quotas(bytes, config.quotas)
        .map_err(RuntimeError::VerifierRejected)?;
    let entry_token = token.require_entry("main").map_err(|err| match err {
        EntryResolutionError::MissingEntry { entry } => RuntimeError::UnknownFunction(entry),
    })?;
    run_verified_entry_semcode_with_config(&entry_token, config)
}

/// Compatibility verified-bytes shim.
///
/// Preserves older byte-based call sites. Internally performs admission through
/// `verify_semcode_token`, resolves the entry via `require_entry`, and delegates
/// to canonical token execution. It should not be preferred for new internal callers.
pub fn run_verified_semcode_with_entry(bytes: &[u8], entry: &str) -> Result<(), RuntimeError> {
    let token = verify_semcode_token(bytes).map_err(RuntimeError::VerifierRejected)?;
    let entry_token = token.require_entry(entry).map_err(|err| match err {
        EntryResolutionError::MissingEntry { entry } => RuntimeError::UnknownFunction(entry),
    })?;
    run_verified_entry_semcode_with_config(
        &entry_token,
        ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
    )
}

/// Compatibility verified-bytes shim.
///
/// Preserves older byte-based call sites. Internally performs admission through
/// `verify_semcode_token`, resolves the entry via `require_entry`, and delegates
/// to canonical token execution. It should not be preferred for new internal callers.
pub fn run_verified_semcode_with_entry_and_config(
    bytes: &[u8],
    entry: &str,
    config: ExecutionConfig,
) -> Result<(), RuntimeError> {
    let token = verify_semcode_token_with_quotas(bytes, config.quotas)
        .map_err(RuntimeError::VerifierRejected)?;
    let entry_token = token.require_entry(entry).map_err(|err| match err {
        EntryResolutionError::MissingEntry { entry } => RuntimeError::UnknownFunction(entry),
    })?;
    run_verified_entry_semcode_with_config(&entry_token, config)
}

/// Compatibility verified-bytes shim.
///
/// Preserves older byte-based call sites. Internally performs admission through
/// `verify_semcode_token`, resolves the entry via `require_entry`, and delegates
/// to canonical token execution. It should not be preferred for new internal callers.
pub fn run_verified_semcode_with_host_and_capabilities<
    H: PrometheusHostAbi,
    C: CapabilityChecker,
>(
    bytes: &[u8],
    host: &mut H,
    capabilities: &C,
) -> Result<(), RuntimeError> {
    let config = ExecutionConfig::for_context(ExecutionContext::KernelBound);
    let token = verify_semcode_token_with_quotas(bytes, config.quotas)
        .map_err(RuntimeError::VerifierRejected)?;
    let entry_token = token.require_entry("main").map_err(|err| match err {
        EntryResolutionError::MissingEntry { entry } => RuntimeError::UnknownFunction(entry),
    })?;
    run_verified_entry_semcode_with_host_and_capabilities_and_config(
        &entry_token,
        host,
        capabilities,
        config,
    )
}

/// Compatibility verified-bytes shim.
///
/// Preserves older byte-based call sites. Internally performs admission through
/// `verify_semcode_token`, resolves the entry via `require_entry`, and delegates
/// to canonical token execution. It should not be preferred for new internal callers.
pub fn run_verified_semcode_with_host_and_capabilities_and_config<
    H: PrometheusHostAbi,
    C: CapabilityChecker,
>(
    bytes: &[u8],
    entry: &str,
    host: &mut H,
    capabilities: &C,
    config: ExecutionConfig,
) -> Result<(), RuntimeError> {
    let token = verify_semcode_token_with_quotas(bytes, config.quotas)
        .map_err(RuntimeError::VerifierRejected)?;
    let entry_token = token.require_entry(entry).map_err(|err| match err {
        EntryResolutionError::MissingEntry { entry } => RuntimeError::UnknownFunction(entry),
    })?;
    run_verified_entry_semcode_with_host_and_capabilities_and_config(
        &entry_token,
        host,
        capabilities,
        config,
    )
}

pub fn run_verified_semcode_with_application_host_and_capabilities_and_config<
    H: ApplicationHostAbi,
    C: CapabilityChecker,
>(
    bytes: &[u8],
    entry: &str,
    host: &mut H,
    capabilities: &C,
    config: ExecutionConfig,
) -> Result<(), RuntimeError> {
    let token = verify_semcode_token_with_quotas(bytes, config.quotas)
        .map_err(RuntimeError::VerifierRejected)?;
    let entry_token = token.require_entry(entry).map_err(|err| match err {
        EntryResolutionError::MissingEntry { entry } => RuntimeError::UnknownFunction(entry),
    })?;
    run_verified_entry_semcode_with_application_host_and_capabilities_and_config(
        &entry_token,
        host,
        capabilities,
        config,
    )
}

/// Canonical verified token execution path.
///
/// This is the canonical and preferred path for internal verified execution.
/// It requires a `VerifiedEntrySemCode` token, ensuring that admission
/// and entry resolution have already occurred, and does not accept raw bytes.
pub fn run_verified_entry_semcode_with_host_and_capabilities_and_config<
    H: PrometheusHostAbi,
    C: CapabilityChecker,
>(
    token: &VerifiedEntrySemCode<'_, '_>,
    host: &mut H,
    capabilities: &C,
    config: ExecutionConfig,
) -> Result<(), RuntimeError> {
    let program = prepare_verified_execution(token)?;
    let mut vm = VM {
        functions: program.functions,
        callstack: Vec::new(),
        config,
        effect_calls: 0,
        steps: 0,
        calls: 0,
        symbols: program.runtime_symbols,
        prng_state: 0,
    };
    push_frame(&mut vm, token.entry(), Vec::new(), None)?;
    let mut bridge = PrometheusVmHost { host, capabilities };
    let mut observation = HelloObservationRuntime::discard();
    exec_loop(&mut vm, &mut bridge, &mut observation).map(|_| ())
}

pub fn run_verified_entry_semcode_with_application_host_and_capabilities_and_config<
    H: ApplicationHostAbi,
    C: CapabilityChecker,
>(
    token: &VerifiedEntrySemCode<'_, '_>,
    host: &mut H,
    capabilities: &C,
    config: ExecutionConfig,
) -> Result<(), RuntimeError> {
    let program = prepare_verified_execution(token)?;
    let mut vm = VM {
        functions: program.functions,
        callstack: Vec::new(),
        config,
        effect_calls: 0,
        steps: 0,
        calls: 0,
        symbols: program.runtime_symbols,
        prng_state: 0,
    };
    push_frame(&mut vm, token.entry(), Vec::new(), None)?;
    let mut bridge = ApplicationVmHost {
        host,
        capabilities,
        observed: false,
        quotas: vm.config.quotas,
        effect_calls: 0,
    };
    let mut observation = HelloObservationRuntime::discard();
    exec_loop(&mut vm, &mut bridge, &mut observation).map(|_| ())
}

/// Canonical verified token execution path.
///
/// This is the canonical and preferred path for internal verified execution.
/// It requires a `VerifiedEntrySemCode` token, ensuring that admission
/// and entry resolution have already occurred, and does not accept raw bytes.
pub fn run_verified_entry_semcode(
    token: &VerifiedEntrySemCode<'_, '_>,
) -> Result<(), RuntimeError> {
    run_verified_entry_semcode_with_config(
        token,
        ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
    )
}

/// Canonical verified token execution path.
///
/// This is the canonical and preferred path for internal verified execution.
/// It requires a `VerifiedEntrySemCode` token, ensuring that admission
/// and entry resolution have already occurred, and does not accept raw bytes.
pub fn run_verified_entry_semcode_with_config(
    token: &VerifiedEntrySemCode<'_, '_>,
    config: ExecutionConfig,
) -> Result<(), RuntimeError> {
    let program = prepare_verified_execution(token)?;
    run_vm_program_view_with_entry_and_config_with_observation_runtime(
        program,
        token.entry(),
        config,
        HelloObservationRuntime::discard(),
    )
    .map(|_| ())
}

/// Canonical verified function invocation path with arguments and structured return value.
///
/// Requires a `VerifiedEntrySemCode` token ensuring bytecode verification has
/// passed. The execution target is `token.entry()` -- the entry this token
/// was resolved against -- and is not independently selectable. Possession
/// of a token bound to one function must not authorize executing a
/// different function from the same verified artifact (#1772).
pub fn run_verified_function_semcode_with_args(
    token: &VerifiedEntrySemCode<'_, '_>,
    args: Vec<Value>,
) -> Result<Value, RuntimeError> {
    run_verified_function_semcode_with_args_and_config(
        token,
        args,
        ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
    )
}

/// See [`run_verified_function_semcode_with_args`]. The execution target is
/// `token.entry()`, not independently selectable (#1772).
pub fn run_verified_function_semcode_with_args_and_config(
    token: &VerifiedEntrySemCode<'_, '_>,
    args: Vec<Value>,
    config: ExecutionConfig,
) -> Result<Value, RuntimeError> {
    let program = prepare_verified_execution(token)?;
    let func_name = token.entry();
    if !program.functions.contains_key(func_name) {
        return Err(RuntimeError::UnknownFunction(func_name.to_string()));
    }
    let mut vm = VM {
        functions: program.functions,
        callstack: Vec::new(),
        config,
        effect_calls: 0,
        steps: 0,
        calls: 0,
        symbols: program.runtime_symbols,
        prng_state: 0,
    };
    // Argument validation happens inside `push_frame` itself (#1773 /
    // FA-09-005) - see `validate_call_arguments`'s doc comment.
    push_frame(&mut vm, func_name, args, None)?;
    let mut host = LegacyVmHost;
    let mut observation = HelloObservationRuntime::discard();
    exec_loop(&mut vm, &mut host, &mut observation)
}

/// Local opcode profiling path for verified token execution.
///
/// This is a feature-gated, local measurement harness that collects opcode
/// execution counts for verified VM execution. It does not change VM semantics
/// and it is not production telemetry.
#[cfg(feature = "vm-profile")]
pub fn run_verified_entry_semcode_with_profile(
    token: &VerifiedEntrySemCode<'_, '_>,
    config: ExecutionConfig,
) -> Result<VmOpcodeProfile, RuntimeError> {
    let program = prepare_verified_execution(token)?;
    let mut vm = VM {
        functions: program.functions,
        callstack: Vec::new(),
        config,
        effect_calls: 0,
        steps: 0,
        calls: 0,
        symbols: program.runtime_symbols,
        prng_state: 0,
    };
    push_frame(&mut vm, token.entry(), Vec::new(), None)?;
    let mut bridge = LegacyVmHost;
    let mut observation = HelloObservationRuntime::discard();
    let mut profile = VmOpcodeProfile::default();
    exec_loop_with_profile(&mut vm, &mut bridge, &mut observation, &mut profile)?;
    Ok(profile)
}

/// Raw execution path.
///
/// Bypasses `sm-verify` admission and executes raw SemCode bytes.
/// Intentionally retained for tests, diagnostics, and malformed-byte behavior.
/// Must not be confused with verified execution.
pub fn run_semcode_with_entry_and_config(
    bytes: &[u8],
    entry: &str,
    config: ExecutionConfig,
) -> Result<(), RuntimeError> {
    run_semcode_with_entry_and_config_with_observation_runtime(
        bytes,
        entry,
        config,
        HelloObservationRuntime::discard(),
    )
    .map(|_| ())
}

fn run_semcode_with_entry_and_config_with_observation_runtime<'a>(
    bytes: &[u8],
    entry: &str,
    config: ExecutionConfig,
    observation: HelloObservationRuntime<'a>,
) -> Result<Vec<HelloObservationEvent>, RuntimeError> {
    let program = parse_semcode(bytes)?;
    run_vm_program_view_with_entry_and_config_with_observation_runtime(
        program,
        entry,
        config,
        observation,
    )
}

fn run_vm_program_view_with_entry_and_config_with_observation_runtime<'a>(
    program: VmProgramView,
    entry: &str,
    config: ExecutionConfig,
    mut observation: HelloObservationRuntime<'a>,
) -> Result<Vec<HelloObservationEvent>, RuntimeError> {
    let mut vm = VM {
        functions: program.functions,
        callstack: Vec::new(),
        config,
        effect_calls: 0,
        steps: 0,
        calls: 0,
        symbols: program.runtime_symbols,
        prng_state: 0,
    };
    push_frame(&mut vm, entry, Vec::new(), None)?;
    let mut host = LegacyVmHost;
    exec_loop(&mut vm, &mut host, &mut observation)?;
    match observation.mode {
        HelloObservationMode::Discard => Ok(Vec::new()),
        HelloObservationMode::Collect(events) => Ok(std::mem::take(events)),
    }
}

/// Diagnostic raw-byte path.
///
/// Intentionally parses raw SemCode for diagnostic output without verifier enforcement.
/// Not a verified execution API.
#[cfg(feature = "disasm")]
pub fn disasm_semcode(bytes: &[u8]) -> Result<String, RuntimeError> {
    let program = parse_semcode(bytes)?;
    let spec = program.header;
    let functions = program.functions;
    let mut out = String::new();
    let header = if bytes.len() >= 8 { &bytes[0..8] } else { &[] };
    out.push_str(&format!(
        "{} epoch={}.{} caps=0x{:08x}\n",
        String::from_utf8_lossy(header),
        spec.epoch,
        spec.rev,
        spec.capabilities
    ));
    let mut ordered = functions.values().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.name.cmp(&right.name));
    for f in ordered {
        out.push_str(&format!(
            "fn {}: code={} bytes, strings={}, debug={}\n",
            f.name,
            f.code.len(),
            f.strings.len(),
            f.debug_symbols.len()
        ));
        let mut pc = 0usize;
        while pc < f.code.len().saturating_sub(f.instr_start) {
            let (line, next) = disasm_one(f, pc)?;
            out.push_str(&format!("  {:04x}: {}\n", pc, line));
            pc = next;
        }
    }
    Ok(out)
}

struct VmProgramView {
    // Only read by `disasm_semcode` (`#[cfg(feature = "disasm")]`) - a
    // default-feature build has no reader for it at all, which is a
    // genuine, pre-existing (not #1891-related) `dead_code` false positive
    // under that configuration rather than actually-unused data.
    #[cfg_attr(not(feature = "disasm"), allow(dead_code))]
    header: SemcodeHeaderSpec,
    runtime_symbols: RuntimeSymbolTable,
    functions: HashMap<String, FunctionBytecode>,
}

fn decode_and_map_errors(
    bytes: &[u8],
) -> Result<
    (
        SemcodeHeaderSpec,
        Vec<sm_format::semcode_decode::DecodedFunctionEnvelope<'_>>,
    ),
    RuntimeError,
> {
    sm_format::semcode_decode::decode_semcode_envelope(bytes).map_err(|e| match e {
        sm_format::semcode_decode::DecodeError::BadHeader => RuntimeError::BadHeader,
        sm_format::semcode_decode::DecodeError::UnsupportedVersion { found, supported } => {
            RuntimeError::UnsupportedBytecodeVersion { found, supported }
        }
        sm_format::semcode_decode::DecodeError::TruncatedFunction { msg, .. } => {
            RuntimeError::BadFormat(msg.to_string())
        }
        sm_format::semcode_decode::DecodeError::InvalidFunctionName { msg, .. } => {
            RuntimeError::BadFormat(msg.to_string())
        }
        sm_format::semcode_decode::DecodeError::InvalidStringTable { msg, .. } => {
            RuntimeError::BadFormat(msg.to_string())
        }
        sm_format::semcode_decode::DecodeError::InvalidDebugSection { msg, .. } => {
            RuntimeError::BadFormat(msg.to_string())
        }
        sm_format::semcode_decode::DecodeError::InvalidOwnershipSection { msg, .. } => {
            RuntimeError::BadFormat(msg.to_string())
        }
        sm_format::semcode_decode::DecodeError::InvalidSignatureSection { msg, .. } => {
            RuntimeError::BadFormat(msg.to_string())
        }
        sm_format::semcode_decode::DecodeError::ResourceLimit { msg, .. } => {
            RuntimeError::BadFormat(msg)
        }
    })
}

fn parse_semcode(bytes: &[u8]) -> Result<VmProgramView, RuntimeError> {
    let (header, decoded_functions) = decode_and_map_errors(bytes)?;
    build_vm_program_view_from_decoded(header, &decoded_functions)
}

fn prepare_verified_execution(
    token: &VerifiedEntrySemCode<'_, '_>,
) -> Result<VmProgramView, RuntimeError> {
    token
        .artifact()
        .with_decoded_envelopes(|header, decoded_functions| {
            build_vm_program_view_from_decoded(header.clone(), decoded_functions)
        })
}

fn build_vm_program_view_from_decoded(
    header: SemcodeHeaderSpec,
    decoded_functions: &[sm_format::semcode_decode::DecodedFunctionEnvelope<'_>],
) -> Result<VmProgramView, RuntimeError> {
    let mut out = HashMap::new();
    let mut runtime_symbols = RuntimeSymbolTable::new();

    for env in decoded_functions {
        let name = env.name.clone();
        let strings = env.strings.clone();

        let debug_symbols = env
            .debug_symbols
            .iter()
            .map(|s| DebugSymbol {
                pc: s.pc,
                line: s.line,
                col: s.col,
            })
            .collect();

        let symbol_ids = strings
            .iter()
            .map(|name| runtime_symbols.intern(name))
            .collect::<Vec<_>>();

        let remap_paths = |paths: &[sm_format::semcode_decode::DecodedAccessPath]| {
            paths
                .iter()
                .map(|path| {
                    // #1725 (FA-04-019): `root_symbol_id` is an index into
                    // this function's own string table (`symbol_ids`, built
                    // just above from `strings` in table order) - the
                    // producer (`emit_ownership_events`) now writes that
                    // index rather than a raw frontend SymbolId, so a
                    // lookup miss here means the artifact's OWN0 section
                    // references a local this function's string table
                    // never declared. Fails closed: no fallback to
                    // reinterpreting the raw wire number as if it were
                    // already a valid runtime SymbolId - that fallback is
                    // exactly the identity confusion #1725 exists to fix.
                    let local_root = path.root_symbol_id as usize;
                    let root = symbol_ids.get(local_root).copied().ok_or_else(|| {
                        RuntimeError::BadFormat(format!(
                            "ownership path root index {} out of bounds for function '{}' string table (len {})",
                            local_root,
                            name,
                            symbol_ids.len()
                        ))
                    })?;
                    let mut p = AccessPath::new(root);
                    for c in &path.components {
                        match c {
                            sm_format::semcode_decode::DecodedAccessPathComponent::TupleIndex(
                                i,
                            ) => {
                                p = p.tuple_index(*i);
                            }
                            sm_format::semcode_decode::DecodedAccessPathComponent::FieldSymbol(
                                s,
                            ) => {
                                p = p.field(SymbolId(*s));
                            }
                            sm_format::semcode_decode::DecodedAccessPathComponent::AdtPayload {
                                variant,
                                index,
                            } => {
                                p = p.adt_payload(SymbolId(*variant), *index);
                            }
                            sm_format::semcode_decode::DecodedAccessPathComponent::SequenceIndexStatic(
                                index,
                            ) => {
                                p = p.sequence_index_static(*index);
                            }
                        }
                    }
                    Ok(p)
                })
                .collect::<Result<Vec<_>, RuntimeError>>()
        };

        // #1726 Checkpoint D3: reuses `remap_paths` (unchanged) for the
        // `AccessPath` half, then zips back onto `env.borrowed_paths`'s own
        // `.activation` in the same order to attach each path's runtime
        // activation authority - never a second, independently-maintained
        // root/component resolver for the Borrow case.
        let borrowed_paths = remap_paths(&env.borrowed_paths)?
            .into_iter()
            .zip(env.borrowed_paths.iter())
            .map(|(path, decoded)| BorrowedPath {
                path,
                activation: match decoded.activation {
                    None | Some(sm_format::semcode_decode::DecodedBorrowActivation::FrameEntry) => {
                        BorrowActivation::FrameEntry
                    }
                    Some(sm_format::semcode_decode::DecodedBorrowActivation::StoreVarSite(
                        anchor,
                    )) => BorrowActivation::StoreVarSite(anchor as usize),
                },
            })
            .collect::<Vec<_>>();
        // #1891 Checkpoint W2F: a pre-rev21 Write ownership event carries a
        // path but no executable anchor at all - the wire grammar has no
        // field for one below `SEMCODE_OWNERSHIP_ANCHOR_MIN_REVISION` (see
        // `WRITE_EXECUTION_MODE_STORE_VAR_SITE`'s own doc comment). The
        // sequential cursor this checkpoint removes was the only mechanism
        // such an event was ever runtime-checked by, and it was never a
        // sound "exact execution site" authority even historically - a
        // static-order guess, exactly what SSF-08's own governing charter
        // (`docs/roadmap/stable_foundation/ssf08_ownership_position_decision.md`,
        // "Fail closed: cannot prove ownership state -> deterministic
        // rejection. Never: cannot find binding/path/state -> assume
        // Available.") forbids approximating. No producer in the current
        // compiler can emit such an artifact any more (#1891 Checkpoint W2A
        // made every Write event mint a resolved `WriteSiteId`
        // unconditionally, and Checkpoint W2D's floor extension promotes any
        // artifact carrying one to this revision), and no existing
        // compatibility/release contract (`tests/bytecode_compat.rs` and its
        // siblings) exercises a legacy Write-bearing artifact through real
        // execution - so this rejects deterministically rather than
        // inventing an authority that was never exact.
        if !env.write_paths.is_empty()
            && header.rev < sm_format::semcode_format::SEMCODE_OWNERSHIP_ANCHOR_MIN_REVISION
        {
            return Err(RuntimeError::BadFormat(format!(
                "function '{}' declares {} legacy (pre-rev21) Write ownership event(s) with no executable anchor; runtime execution of legacy Write-bearing artifacts is not supported - recompile with a current compiler",
                name,
                env.write_paths.len()
            )));
        }

        // #1891 Checkpoint W2F: group by the exact verifier-authenticated PC
        // (Checkpoint W2E) rather than keep one flat, order-matched list -
        // `BTreeMap` gives deterministic ascending-`pc` iteration for free,
        // matching the artifact's own per-PC event order within each group
        // since paths are appended in decode order.
        let write_execution_sites = {
            let mut by_pc: BTreeMap<usize, Vec<AccessPath>> = BTreeMap::new();
            for (path, decoded) in remap_paths(&env.write_paths)?
                .into_iter()
                .zip(env.write_paths.iter())
            {
                let anchor = match decoded.write_execution {
                    None => unreachable!("legacy Write events are rejected above"),
                    Some(sm_format::semcode_decode::DecodedWriteExecution::StoreVarSite(a)) => {
                        a as usize
                    }
                    Some(sm_format::semcode_decode::DecodedWriteExecution::MakeRecordSite(a)) => {
                        a as usize
                    }
                };
                by_pc.entry(anchor).or_default().push(path);
            }
            by_pc
                .into_iter()
                .map(|(pc, paths)| WriteExecutionSite { pc, paths })
                .collect::<Vec<_>>()
        };

        let f = FunctionBytecode {
            name: name.clone(),
            strings,
            symbol_ids,
            debug_symbols,
            borrowed_paths,
            write_execution_sites,
            code: env.code_slice.to_vec(),
            instr_start: env.instr_start_offset,
            signature: env.signature.clone(),
        };
        validate_function_bytecode(&f)?;
        if out.insert(name.clone(), f).is_some() {
            return Err(RuntimeError::BadFormat(format!(
                "duplicate function '{}'",
                name
            )));
        }
    }
    Ok(VmProgramView {
        header,
        runtime_symbols,
        functions: out,
    })
}

fn map_format_err(err: SemcodeFormatError) -> RuntimeError {
    match err {
        SemcodeFormatError::UnexpectedEof => RuntimeError::BadFormat("unexpected EOF".to_string()),
        SemcodeFormatError::InvalidUtf8 => RuntimeError::BadFormat("invalid utf8".to_string()),
        SemcodeFormatError::UnknownOpcode(op) => {
            RuntimeError::BadFormat(format!("unknown opcode 0x{:02x}", op))
        }
    }
}

fn validate_function_bytecode(f: &FunctionBytecode) -> Result<(), RuntimeError> {
    if f.instr_start > f.code.len() {
        return Err(RuntimeError::BadFormat(format!(
            "invalid instr_start in '{}'",
            f.name
        )));
    }
    let mut cur = f.instr_start;
    let mut starts: HashSet<usize> = HashSet::new();
    let mut jumps: Vec<usize> = Vec::new();
    while cur < f.code.len() {
        starts.insert(cur - f.instr_start);
        let opcode = Opcode::from_byte(read_u8(&f.code, &mut cur).map_err(map_format_err)?)
            .map_err(map_format_err)?;
        match opcode {
            Opcode::LoadQ => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::LoadBool => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::LoadI32 => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_i32_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::AddI32 => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::SubI32 | Opcode::MulI32 | Opcode::DivI32 | Opcode::ModI32 => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::LoadU32 => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u32_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::LoadF64 => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_f64_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::LoadFx => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_i32_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::AddFx | Opcode::SubFx | Opcode::MulFx | Opcode::DivFx => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::MakeTuple => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                if count < 2 {
                    return Err(RuntimeError::BadFormat(format!(
                        "tuple literal arity must be at least 2 in '{}'",
                        f.name
                    )));
                }
                for _ in 0..count {
                    let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                }
            }
            Opcode::MakeSequence => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                for _ in 0..count {
                    let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                }
            }
            Opcode::MakeClosure => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                for _ in 0..count {
                    let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                }
            }
            Opcode::MakeRecord => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                if count == 0 {
                    return Err(RuntimeError::BadFormat(format!(
                        "record literal slot count must be at least 1 in '{}'",
                        f.name
                    )));
                }
                for _ in 0..count {
                    let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                }
            }
            Opcode::MakeAdt => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                for _ in 0..count {
                    let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                }
            }
            Opcode::AdtTag => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::AdtGet => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::RecordGet => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::TupleGet => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::SequenceGet => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::SequenceLen => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::SequenceIsEmpty => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::SequenceContains | Opcode::SequencePush | Opcode::SequencePrepend => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::SequencePop => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::MapEmpty => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::MapContains => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::MapGet => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::MapSet => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::RngSeed => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::RngNextI32 => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::ClosureCall => {
                let _ = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::LoadVar => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::StoreVar => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::QNot | Opcode::BoolNot | Opcode::QTruthNot => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::LoadText => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::ConcatText => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::QAnd
            | Opcode::QOr
            | Opcode::QImpl
            | Opcode::QTruthAnd
            | Opcode::QTruthOr
            | Opcode::QTruthImpl
            | Opcode::BoolAnd
            | Opcode::BoolOr
            | Opcode::CmpEq
            | Opcode::CmpNe
            | Opcode::CmpI32Lt
            | Opcode::CmpI32Le
            | Opcode::AddF64
            | Opcode::SubF64
            | Opcode::MulF64
            | Opcode::DivF64 => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::Jmp => {
                let addr = read_u32_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                jumps.push(addr);
            }
            Opcode::JmpIf => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let addr = read_u32_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                jumps.push(addr);
            }
            Opcode::Call => {
                let _ = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let argc = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                for _ in 0..argc {
                    let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                }
            }
            Opcode::Assert => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::GateRead => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::GateWrite => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::PulseEmit => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::StateQuery => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::StateUpdate => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::EventPost => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::ClockRead => {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            Opcode::Ret => {
                let has_src = read_u8(&f.code, &mut cur).map_err(map_format_err)? != 0;
                if has_src {
                    let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                }
            }
        }
    }
    if cur != f.code.len() {
        return Err(RuntimeError::BadFormat(format!(
            "trailing bytes in '{}'",
            f.name
        )));
    }
    let instr_len = f.code.len().saturating_sub(f.instr_start);
    for ds in &f.debug_symbols {
        if ds.pc >= instr_len {
            return Err(RuntimeError::BadFormat(format!(
                "debug pc out of range in '{}': {}",
                f.name, ds.pc
            )));
        }
        // #1746 (FA-07-006): mirrors sm-verify's canonical admission check.
        // `starts` (built above, same instruction walk) already proves
        // which offsets are real instruction boundaries; range alone does
        // not rule out a pc landing inside a decoded operand's bytes.
        if !starts.contains(&ds.pc) {
            return Err(RuntimeError::BadFormat(format!(
                "debug pc not on an instruction boundary in '{}': {}",
                f.name, ds.pc
            )));
        }
    }
    for addr in jumps {
        if addr >= instr_len {
            return Err(RuntimeError::BadFormat(format!(
                "jump out of range in '{}': {}",
                f.name, addr
            )));
        }
        if !starts.contains(&addr) {
            return Err(RuntimeError::BadFormat(format!(
                "jump to non-instruction boundary in '{}': {}",
                f.name, addr
            )));
        }
    }
    Ok(())
}

trait VmHostBridge {
    fn gate_read(&mut self, device_id: u16, port: u16) -> Result<Value, RuntimeError>;
    fn gate_write(&mut self, device_id: u16, port: u16, value: Value) -> Result<(), RuntimeError>;
    fn pulse_emit(&mut self, signal: &str) -> Result<(), RuntimeError>;
    fn state_query(&mut self, key: &str) -> Result<Value, RuntimeError>;
    fn state_update(&mut self, key: &str, value: Value) -> Result<(), RuntimeError>;
    fn event_post(&mut self, signal: &str) -> Result<(), RuntimeError>;
    fn clock_read(&mut self) -> Result<Value, RuntimeError>;
    fn args_read(&mut self, _index: u32) -> Result<Value, RuntimeError> {
        Err(unavailable_host_call(HostCallId::ArgsRead))
    }
    fn stdin_read_text(&mut self) -> Result<Value, RuntimeError> {
        Err(unavailable_host_call(HostCallId::StdinReadText))
    }
    fn stdout_write(&mut self, _text: &str) -> Result<(), RuntimeError> {
        Err(unavailable_host_call(HostCallId::StdoutWrite))
    }
    fn stderr_write(&mut self, _text: &str) -> Result<(), RuntimeError> {
        Err(unavailable_host_call(HostCallId::StderrWrite))
    }
    fn path_inspect(&mut self, _path: &str) -> Result<Value, RuntimeError> {
        Err(unavailable_host_call(HostCallId::PathInspect))
    }
    fn fs_read_text(&mut self, _path: &str) -> Result<Value, RuntimeError> {
        Err(unavailable_host_call(HostCallId::FsRead))
    }
    fn fs_write_text(&mut self, _path: &str, _text: &str) -> Result<(), RuntimeError> {
        Err(unavailable_host_call(HostCallId::FsWrite))
    }
    fn time_duration_millis(&mut self) -> Result<Value, RuntimeError> {
        Err(unavailable_host_call(HostCallId::TimeDuration))
    }
}

fn unavailable_host_call(call: HostCallId) -> RuntimeError {
    RuntimeError::HostAbi(AbiError::new(
        call,
        AbiFailureKind::Unavailable,
        "host boundary does not provide this application operation",
    ))
}

struct LegacyVmHost;

impl VmHostBridge for LegacyVmHost {
    fn gate_read(&mut self, device_id: u16, port: u16) -> Result<Value, RuntimeError> {
        Ok(Value::I32(((device_id as i32) << 16) | (port as i32)))
    }

    fn gate_write(
        &mut self,
        _device_id: u16,
        _port: u16,
        _value: Value,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn pulse_emit(&mut self, _signal: &str) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn state_query(&mut self, key: &str) -> Result<Value, RuntimeError> {
        Ok(Value::I32(stable_state_query_fallback(key)))
    }

    fn state_update(&mut self, _key: &str, _value: Value) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn event_post(&mut self, _signal: &str) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn clock_read(&mut self) -> Result<Value, RuntimeError> {
        Ok(Value::U32(0))
    }
}

struct PrometheusVmHost<'a, H: PrometheusHostAbi, C: CapabilityChecker> {
    host: &'a mut H,
    capabilities: &'a C,
}

impl<'a, H: PrometheusHostAbi, C: CapabilityChecker> VmHostBridge for PrometheusVmHost<'a, H, C> {
    fn gate_read(&mut self, device_id: u16, port: u16) -> Result<Value, RuntimeError> {
        self.capabilities
            .require_call(HostCallId::GateRead)
            .map_err(RuntimeError::CapabilityDenied)?;
        let raw = self
            .host
            .gate_read(device_id, port)
            .map_err(RuntimeError::HostAbi)?;
        value_from_abi(raw, HostCallId::GateRead)
    }

    fn gate_write(&mut self, device_id: u16, port: u16, value: Value) -> Result<(), RuntimeError> {
        self.capabilities
            .require_call(HostCallId::GateWrite)
            .map_err(RuntimeError::CapabilityDenied)?;
        self.host
            .gate_write(device_id, port, value_to_abi(value)?)
            .map_err(RuntimeError::HostAbi)
    }

    fn pulse_emit(&mut self, signal: &str) -> Result<(), RuntimeError> {
        self.capabilities
            .require_call(HostCallId::PulseEmit)
            .map_err(RuntimeError::CapabilityDenied)?;
        self.host.pulse_emit(signal).map_err(RuntimeError::HostAbi)
    }

    fn state_query(&mut self, key: &str) -> Result<Value, RuntimeError> {
        self.capabilities
            .require_call(HostCallId::StateQuery)
            .map_err(RuntimeError::CapabilityDenied)?;
        let raw = self.host.state_query(key).map_err(RuntimeError::HostAbi)?;
        value_from_abi(raw, HostCallId::StateQuery)
    }

    fn state_update(&mut self, key: &str, value: Value) -> Result<(), RuntimeError> {
        self.capabilities
            .require_call(HostCallId::StateUpdate)
            .map_err(RuntimeError::CapabilityDenied)?;
        self.host
            .state_update(key, value_to_abi(value)?)
            .map_err(RuntimeError::HostAbi)
    }

    fn event_post(&mut self, signal: &str) -> Result<(), RuntimeError> {
        self.capabilities
            .require_call(HostCallId::EventPost)
            .map_err(RuntimeError::CapabilityDenied)?;
        self.host.event_post(signal).map_err(RuntimeError::HostAbi)
    }

    fn clock_read(&mut self) -> Result<Value, RuntimeError> {
        self.capabilities
            .require_call(HostCallId::ClockRead)
            .map_err(RuntimeError::CapabilityDenied)?;
        self.host
            .clock_read()
            .map(Value::U32)
            .map_err(RuntimeError::HostAbi)
    }
}

struct ApplicationVmHost<'a, H: ApplicationHostAbi, C: CapabilityChecker> {
    host: &'a mut H,
    capabilities: &'a C,
    observed: bool,
    quotas: RuntimeQuotas,
    effect_calls: usize,
}

impl<'a, H: ApplicationHostAbi, C: CapabilityChecker> ApplicationVmHost<'a, H, C> {
    fn require(&self, call: HostCallId) -> Result<(), RuntimeError> {
        self.capabilities
            .require_call(call)
            .map_err(RuntimeError::CapabilityDenied)
    }

    fn require_observation(&self, call: HostCallId) -> Result<(), RuntimeError> {
        if self.observed {
            Ok(())
        } else {
            Err(RuntimeError::HostAbi(AbiError::new(
                call,
                AbiFailureKind::InvalidInput,
                "application writes require a preceding captured observation",
            )))
        }
    }

    /// Charges one effect-call quota unit for an application host operation that
    /// already passed its capability check. Must run after `require`/`require_observation`
    /// and before the actual host dispatch, so quota exhaustion blocks the effect itself.
    fn bump_effect_calls(&mut self) -> Result<(), RuntimeError> {
        let next = self.effect_calls + 1;
        enforce_quota(&self.quotas, QuotaKind::EffectCalls, next)?;
        self.effect_calls = next;
        Ok(())
    }
}

impl<'a, H: ApplicationHostAbi, C: CapabilityChecker> VmHostBridge for ApplicationVmHost<'a, H, C> {
    fn gate_read(&mut self, _device_id: u16, _port: u16) -> Result<Value, RuntimeError> {
        Err(unavailable_host_call(HostCallId::GateRead))
    }

    fn gate_write(
        &mut self,
        _device_id: u16,
        _port: u16,
        _value: Value,
    ) -> Result<(), RuntimeError> {
        Err(unavailable_host_call(HostCallId::GateWrite))
    }

    fn pulse_emit(&mut self, _signal: &str) -> Result<(), RuntimeError> {
        Err(unavailable_host_call(HostCallId::PulseEmit))
    }

    fn state_query(&mut self, _key: &str) -> Result<Value, RuntimeError> {
        Err(unavailable_host_call(HostCallId::StateQuery))
    }

    fn state_update(&mut self, _key: &str, _value: Value) -> Result<(), RuntimeError> {
        Err(unavailable_host_call(HostCallId::StateUpdate))
    }

    fn event_post(&mut self, _signal: &str) -> Result<(), RuntimeError> {
        Err(unavailable_host_call(HostCallId::EventPost))
    }

    fn clock_read(&mut self) -> Result<Value, RuntimeError> {
        Err(unavailable_host_call(HostCallId::ClockRead))
    }

    fn args_read(&mut self, index: u32) -> Result<Value, RuntimeError> {
        self.require(HostCallId::ArgsRead)?;
        self.bump_effect_calls()?;
        let value = self.host.args_read(index).map_err(RuntimeError::HostAbi)?;
        self.observed = true;
        Ok(Value::Text(value))
    }

    fn stdin_read_text(&mut self) -> Result<Value, RuntimeError> {
        self.require(HostCallId::StdinReadText)?;
        self.bump_effect_calls()?;
        let value = self.host.stdin_read_text().map_err(RuntimeError::HostAbi)?;
        self.observed = true;
        Ok(Value::Text(value))
    }

    fn stdout_write(&mut self, text: &str) -> Result<(), RuntimeError> {
        self.require(HostCallId::StdoutWrite)?;
        self.require_observation(HostCallId::StdoutWrite)?;
        self.bump_effect_calls()?;
        self.host.stdout_write(text).map_err(RuntimeError::HostAbi)
    }

    fn stderr_write(&mut self, text: &str) -> Result<(), RuntimeError> {
        self.require(HostCallId::StderrWrite)?;
        self.require_observation(HostCallId::StderrWrite)?;
        self.bump_effect_calls()?;
        self.host.stderr_write(text).map_err(RuntimeError::HostAbi)
    }

    fn path_inspect(&mut self, path: &str) -> Result<Value, RuntimeError> {
        self.require(HostCallId::PathInspect)?;
        self.bump_effect_calls()?;
        let value = self
            .host
            .path_inspect(path)
            .map_err(RuntimeError::HostAbi)?;
        self.observed = true;
        Ok(Value::Bool(value))
    }

    fn fs_read_text(&mut self, path: &str) -> Result<Value, RuntimeError> {
        self.require(HostCallId::FsRead)?;
        self.bump_effect_calls()?;
        let value = self
            .host
            .fs_read_text(path)
            .map_err(RuntimeError::HostAbi)?;
        self.observed = true;
        Ok(Value::Text(value))
    }

    fn fs_write_text(&mut self, path: &str, text: &str) -> Result<(), RuntimeError> {
        self.require(HostCallId::FsWrite)?;
        self.require_observation(HostCallId::FsWrite)?;
        self.bump_effect_calls()?;
        self.host
            .fs_write_text(path, text)
            .map_err(RuntimeError::HostAbi)
    }

    fn time_duration_millis(&mut self) -> Result<Value, RuntimeError> {
        self.require(HostCallId::TimeDuration)?;
        self.bump_effect_calls()?;
        let value = self
            .host
            .time_duration_millis()
            .map_err(RuntimeError::HostAbi)?;
        self.observed = true;
        Ok(Value::U32(value))
    }
}

fn exec_loop<'a, H: VmHostBridge>(
    vm: &mut VM,
    host: &mut H,
    observation: &mut HelloObservationRuntime<'a>,
) -> Result<Value, RuntimeError> {
    let mut profile = NoopOpcodeProfile;
    exec_loop_with_profile(vm, host, observation, &mut profile)
}

fn exec_loop_with_profile<'a, H, P>(
    vm: &mut VM,
    host: &mut H,
    observation: &mut HelloObservationRuntime<'a>,
    profile: &mut P,
) -> Result<Value, RuntimeError>
where
    H: VmHostBridge,
    P: OpcodeProfileSink,
{
    loop {
        let Some(frame_idx) = vm.callstack.len().checked_sub(1) else {
            return Ok(Value::Unit);
        };
        let func_name = vm.callstack[frame_idx].func.clone();
        let f = vm
            .functions
            .get(&func_name)
            .cloned()
            .ok_or_else(|| RuntimeError::UnknownFunction(func_name.clone()))?;
        let pc = vm.callstack[frame_idx].pc;
        let instr_rel_len = f.code.len().saturating_sub(f.instr_start);
        if pc >= instr_rel_len {
            return Err(RuntimeError::BadFormat(format!(
                "pc out of range in '{}': {}",
                func_name, pc
            )));
        }
        let mut cur = f.instr_start + pc;
        let opcode = Opcode::from_byte(read_u8(&f.code, &mut cur).map_err(map_format_err)?)
            .map_err(map_format_err)?;
        // #1759: the sole Steps choke point - charged immediately after a
        // successful opcode decode, before profiling (non-authoritative),
        // the write-execution-site check, and this opcode's own semantic
        // body. A failed decode above never reaches this line, so a
        // malformed opcode byte charges no Step. No refund on a later
        // failure within this same dispatch - see
        // ssf08_1759_steps_calls_contract_decision.md §2-§3.
        vm.steps = charge_counter(vm.steps, vm.config.quotas.max_steps, QuotaKind::Steps)?;
        profile.record_opcode(opcode);
        // #1891 Checkpoint W2F: check every Write path attached to this
        // exact PC against the frame's currently active borrows BEFORE the
        // instruction below commits any effect - covers both StoreVar and
        // MakeRecord uniformly (see `WriteExecutionSite`'s own doc comment
        // for why the VM never needs to know which opcode class `pc`
        // belongs to). Reads `active_borrowed_paths` as it stood before this
        // instruction, so a Borrow this same StoreVar is about to activate
        // (`pending_site_borrows`, checked further below only after the
        // write commits) cannot retroactively conflict with the write that
        // introduced it.
        check_write_execution_site(&f, pc, &vm.callstack[frame_idx].active_borrowed_paths)?;
        let next_pc: usize;

        match opcode {
            Opcode::LoadQ => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let q = match read_u8(&f.code, &mut cur).map_err(map_format_err)? {
                    0 => QuadVal::N,
                    1 => QuadVal::F,
                    2 => QuadVal::T,
                    3 => QuadVal::S,
                    v => {
                        return Err(RuntimeError::BadFormat(format!(
                            "invalid quad literal {}",
                            v
                        )))
                    }
                };
                set_reg(vm, frame_idx, dst, Value::Quad(q))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::LoadBool => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let b = read_u8(&f.code, &mut cur).map_err(map_format_err)? != 0;
                set_reg(vm, frame_idx, dst, Value::Bool(b))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::LoadI32 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let v = read_i32_le(&f.code, &mut cur).map_err(map_format_err)?;
                set_reg(vm, frame_idx, dst, Value::I32(v))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::AddI32 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let l = as_i32(get_reg(vm, frame_idx, lhs)?)?;
                let r = as_i32(get_reg(vm, frame_idx, rhs)?)?;
                let out = l.wrapping_add(r);
                set_reg(vm, frame_idx, dst, Value::I32(out))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::SubI32 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let l = as_i32(get_reg(vm, frame_idx, lhs)?)?;
                let r = as_i32(get_reg(vm, frame_idx, rhs)?)?;
                let out = l.wrapping_sub(r);
                set_reg(vm, frame_idx, dst, Value::I32(out))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::MulI32 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let l = as_i32(get_reg(vm, frame_idx, lhs)?)?;
                let r = as_i32(get_reg(vm, frame_idx, rhs)?)?;
                let out = l.wrapping_mul(r);
                set_reg(vm, frame_idx, dst, Value::I32(out))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::DivI32 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let l = as_i32(get_reg(vm, frame_idx, lhs)?)?;
                let r = as_i32(get_reg(vm, frame_idx, rhs)?)?;
                let out = i32_div_raw(l, r)?;
                set_reg(vm, frame_idx, dst, Value::I32(out))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::ModI32 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let l = as_i32(get_reg(vm, frame_idx, lhs)?)?;
                let r = as_i32(get_reg(vm, frame_idx, rhs)?)?;
                let out = i32_mod_raw(l, r)?;
                set_reg(vm, frame_idx, dst, Value::I32(out))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::LoadU32 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let v = read_u32_le(&f.code, &mut cur).map_err(map_format_err)?;
                set_reg(vm, frame_idx, dst, Value::U32(v))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::LoadF64 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let v = read_f64_le(&f.code, &mut cur).map_err(map_format_err)?;
                set_reg(vm, frame_idx, dst, Value::F64(v))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::LoadFx => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let v = read_i32_le(&f.code, &mut cur).map_err(map_format_err)?;
                set_reg(vm, frame_idx, dst, Value::Fx(v))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::LoadText => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let value = decode_text_literal(lookup_str(&f, sid)?);
                set_reg(vm, frame_idx, dst, Value::Text(value))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::ConcatText => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let mut out = as_text(get_reg(vm, frame_idx, lhs)?)?;
                out.push_str(&as_text(get_reg(vm, frame_idx, rhs)?)?);
                set_reg(vm, frame_idx, dst, Value::Text(out))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::MakeSequence => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                    items.push(get_reg(vm, frame_idx, src)?);
                }
                set_reg(vm, frame_idx, dst, Value::Sequence(items))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::MakeClosure => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let function_name = lookup_str(&f, sid)?.to_string();
                let mut captures = Vec::with_capacity(count);
                for _ in 0..count {
                    let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                    captures.push(get_reg(vm, frame_idx, src)?);
                }
                set_reg(
                    vm,
                    frame_idx,
                    dst,
                    Value::Closure(ClosureValue {
                        function_name,
                        captures,
                    }),
                )?;
                next_pc = cur - f.instr_start;
            }
            Opcode::MakeTuple => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                    items.push(get_reg(vm, frame_idx, src)?);
                }
                set_reg(vm, frame_idx, dst, Value::Tuple(items))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::MakeRecord => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let type_name = lookup_str(&f, sid)?.to_string();
                let mut slots = Vec::with_capacity(count);
                for _ in 0..count {
                    let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                    slots.push(get_reg(vm, frame_idx, src)?);
                }
                set_reg(
                    vm,
                    frame_idx,
                    dst,
                    Value::Record(RecordCarrier { type_name, slots }),
                )?;
                next_pc = cur - f.instr_start;
            }
            Opcode::MakeAdt => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let variant_sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let tag = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let type_name = lookup_str(&f, sid)?.to_string();
                let variant_name = lookup_str(&f, variant_sid)?.to_string();
                let mut payload = Vec::with_capacity(count);
                for _ in 0..count {
                    let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                    payload.push(get_reg(vm, frame_idx, src)?);
                }
                set_reg(
                    vm,
                    frame_idx,
                    dst,
                    Value::Adt(AdtCarrier {
                        type_name,
                        variant_name,
                        tag,
                        payload,
                    }),
                )?;
                next_pc = cur - f.instr_start;
            }
            Opcode::AdtTag => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let expected_name = lookup_str(&f, sid)?.to_string();
                let adt = get_reg(vm, frame_idx, src)?;
                let Value::Adt(adt) = adt else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "ADT_TAG source must be enum".to_string(),
                    ));
                };
                if adt.type_name != expected_name {
                    return Err(RuntimeError::TypeMismatchRuntime(format!(
                        "ADT_TAG expected enum '{}', got '{}'",
                        expected_name, adt.type_name
                    )));
                }
                set_reg(vm, frame_idx, dst, Value::I32(i32::from(adt.tag)))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::AdtGet => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let index = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let expected_name = lookup_str(&f, sid)?.to_string();
                let adt = get_reg(vm, frame_idx, src)?;
                let Value::Adt(adt) = adt else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "ADT_GET source must be enum".to_string(),
                    ));
                };
                if adt.type_name != expected_name {
                    return Err(RuntimeError::TypeMismatchRuntime(format!(
                        "ADT_GET expected enum '{}', got '{}'",
                        expected_name, adt.type_name
                    )));
                }
                let item = adt.payload.get(index).cloned().ok_or_else(|| {
                    RuntimeError::BadFormat(format!("adt-get index out of bounds: {}", index))
                })?;
                set_reg(vm, frame_idx, dst, item)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::RecordGet => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let index = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let expected_name = lookup_str(&f, sid)?.to_string();
                let record = get_reg(vm, frame_idx, src)?;
                let Value::Record(record) = record else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "RECORD_GET source must be record".to_string(),
                    ));
                };
                if record.type_name != expected_name {
                    return Err(RuntimeError::TypeMismatchRuntime(format!(
                        "RECORD_GET expected record '{}', got '{}'",
                        expected_name, record.type_name
                    )));
                }
                let item = record.slots.get(index).cloned().ok_or_else(|| {
                    RuntimeError::BadFormat(format!("record-get index out of bounds: {}", index))
                })?;
                set_reg(vm, frame_idx, dst, item)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::TupleGet => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let index = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let tuple = get_reg(vm, frame_idx, src)?;
                let Value::Tuple(items) = tuple else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "TUPLE_GET source must be tuple".to_string(),
                    ));
                };
                let item = items.get(index).cloned().ok_or_else(|| {
                    RuntimeError::BadFormat(format!("tuple-get index out of bounds: {}", index))
                })?;
                set_reg(vm, frame_idx, dst, item)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::SequenceGet => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let index_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sequence = get_reg(vm, frame_idx, src)?;
                let Value::Sequence(items) = sequence else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "SEQUENCE_GET source must be sequence".to_string(),
                    ));
                };
                let index = as_i32(get_reg(vm, frame_idx, index_reg)?)?;
                if index < 0 {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "SEQUENCE_GET index must be non-negative".to_string(),
                    ));
                }
                let item = items.get(index as usize).cloned().ok_or_else(|| {
                    RuntimeError::TypeMismatchRuntime(format!(
                        "SEQUENCE_GET index out of bounds: {}",
                        index
                    ))
                })?;
                set_reg(vm, frame_idx, dst, item)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::SequenceLen => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sequence = get_reg(vm, frame_idx, src)?;
                let Value::Sequence(items) = sequence else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "SEQUENCE_LEN source must be sequence".to_string(),
                    ));
                };
                let len = i32::try_from(items.len()).map_err(|_| {
                    RuntimeError::BadFormat("SEQUENCE_LEN exceeds i32 range".to_string())
                })?;
                set_reg(vm, frame_idx, dst, Value::I32(len))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::SequenceIsEmpty => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sequence = get_reg(vm, frame_idx, src)?;
                let Value::Sequence(items) = sequence else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "SEQUENCE_IS_EMPTY source must be sequence".to_string(),
                    ));
                };
                set_reg(vm, frame_idx, dst, Value::Bool(items.is_empty()))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::SequenceContains => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let seq_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let val_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sequence = get_reg(vm, frame_idx, seq_reg)?;
                let Value::Sequence(items) = sequence else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "SEQUENCE_CONTAINS first argument must be sequence".to_string(),
                    ));
                };
                let search = get_reg(vm, frame_idx, val_reg)?;
                set_reg(vm, frame_idx, dst, Value::Bool(items.contains(&search)))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::SequencePush => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let seq_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let val_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sequence = get_reg(vm, frame_idx, seq_reg)?;
                let Value::Sequence(items) = sequence else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "SEQUENCE_PUSH first argument must be sequence".to_string(),
                    ));
                };
                let new_val = get_reg(vm, frame_idx, val_reg)?;
                let mut new_items = items;
                new_items.push(new_val);
                set_reg(vm, frame_idx, dst, Value::Sequence(new_items))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::SequencePrepend => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let seq_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let val_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sequence = get_reg(vm, frame_idx, seq_reg)?;
                let Value::Sequence(items) = sequence else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "SEQUENCE_PREPEND first argument must be sequence".to_string(),
                    ));
                };
                let new_val = get_reg(vm, frame_idx, val_reg)?;
                let mut new_items = Vec::with_capacity(items.len() + 1);
                new_items.push(new_val);
                new_items.extend(items);
                set_reg(vm, frame_idx, dst, Value::Sequence(new_items))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::SequencePop => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sequence = get_reg(vm, frame_idx, src)?;
                let Value::Sequence(items) = sequence else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "SEQUENCE_POP source must be sequence".to_string(),
                    ));
                };
                if items.is_empty() {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "SEQUENCE_POP source must be non-empty".to_string(),
                    ));
                }
                let mut new_items = items.clone();
                new_items.pop();
                set_reg(vm, frame_idx, dst, Value::Sequence(new_items))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::MapEmpty => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                set_reg(vm, frame_idx, dst, Value::Map(Vec::new()))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::MapContains => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let map_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let key_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let map_val = get_reg(vm, frame_idx, map_reg)?;
                let Value::Map(pairs) = map_val else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "MAP_CONTAINS first argument must be a map".to_string(),
                    ));
                };
                let key = map_key_from_value(get_reg(vm, frame_idx, key_reg)?)?;
                let found = pairs.iter().any(|(k, _)| k == &key);
                set_reg(vm, frame_idx, dst, Value::Bool(found))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::MapGet => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let map_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let key_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let default_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let map_val = get_reg(vm, frame_idx, map_reg)?;
                let Value::Map(pairs) = map_val else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "MAP_GET first argument must be a map".to_string(),
                    ));
                };
                let key = map_key_from_value(get_reg(vm, frame_idx, key_reg)?)?;
                let result = match pairs.iter().find(|(k, _)| k == &key) {
                    Some((_, v)) => v.clone(),
                    None => get_reg(vm, frame_idx, default_reg)?,
                };
                set_reg(vm, frame_idx, dst, result)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::MapSet => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let map_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let key_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let val_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let map_val = get_reg(vm, frame_idx, map_reg)?;
                let Value::Map(pairs) = map_val else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "MAP_SET first argument must be a map".to_string(),
                    ));
                };
                let key = map_key_from_value(get_reg(vm, frame_idx, key_reg)?)?;
                let val = get_reg(vm, frame_idx, val_reg)?;
                let mut new_pairs: Vec<(MapKey, Value)> =
                    pairs.iter().filter(|(k, _)| k != &key).cloned().collect();
                new_pairs.push((key, val));
                set_reg(vm, frame_idx, dst, Value::Map(new_pairs))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::RngSeed => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let seed_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let seed_val = get_reg(vm, frame_idx, seed_reg)?;
                let Value::I32(seed) = seed_val else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "RNG_SEED seed argument must be i32".to_string(),
                    ));
                };
                // Map i32 seed to a non-zero u64.  seed==0 becomes 1 to avoid
                // the xorshift64 zero fixed-point.
                let raw = seed as u64;
                vm.prng_state = if raw == 0 { 1 } else { raw };
                set_reg(vm, frame_idx, dst, Value::Unit)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::RngNextI32 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lo_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let hi_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lo_val = get_reg(vm, frame_idx, lo_reg)?;
                let hi_val = get_reg(vm, frame_idx, hi_reg)?;
                let (Value::I32(lo), Value::I32(hi)) = (lo_val, hi_val) else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "RNG_NEXT_I32 lo and hi must be i32".to_string(),
                    ));
                };
                if lo >= hi {
                    return Err(RuntimeError::TypeMismatchRuntime(format!(
                        "random_next_i32: lo ({lo}) must be strictly less than hi ({hi})"
                    )));
                }
                // Compute range through i64 to handle the full i32 span without
                // overflow (e.g. lo=i32::MIN, hi=i32::MAX gives range=4294967295).
                let range = i64::from(hi) - i64::from(lo);
                let raw = xorshift64_step(&mut vm.prng_state);
                let offset = (raw % (range as u64)) as i64;
                let result = i64::from(lo) + offset;
                let result = i32::try_from(result)
                    .map_err(|_| RuntimeError::Trap(RuntimeTrap::ArithmeticOverflow))?;
                set_reg(vm, frame_idx, dst, Value::I32(result))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::LoadVar => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let symbol = lookup_symbol(&f, sid)?;
                let val = vm.callstack[frame_idx]
                    .locals
                    .get(&symbol)
                    .cloned()
                    .ok_or_else(|| {
                        let name = lookup_str(&f, sid).unwrap_or("<unknown>");
                        RuntimeError::UnknownVariable(name.to_string())
                    })?;
                set_reg(vm, frame_idx, dst, val)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::StoreVar => {
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let symbol = lookup_symbol(&f, sid)?;
                let val = get_reg(vm, frame_idx, src)?;
                // #1891 Checkpoint W2F: the exact-PC write check above
                // (`check_write_execution_site`) already ran before this
                // match started, using `pc` and `active_borrowed_paths` as
                // they stood before this instruction's own write - commit
                // unconditionally here.
                vm.callstack[frame_idx].locals.insert(symbol, val);
                // #1726 Checkpoint D3: activation happens strictly after the
                // write above has committed, never before - this instruction
                // (`pc`, captured at the top of the dispatch loop before any
                // operand was read, the same frame-relative "opcode-byte
                // position" domain `ExecutableAnchor` and D2b's admission
                // check already use) is only now eligible to satisfy a
                // pending site. Uses `position` + `remove` (not `retain`) so
                // a loop's second visit to the same anchor - already
                // consumed on the first - finds no match and is a harmless
                // no-op, never re-activating or erroring; nothing is ever
                // removed just because control passed elsewhere without this
                // exact PC executing.
                if let Some(index) = vm.callstack[frame_idx]
                    .pending_site_borrows
                    .iter()
                    .position(|pending| pending.anchor == pc)
                {
                    let activated = vm.callstack[frame_idx].pending_site_borrows.remove(index);
                    vm.callstack[frame_idx]
                        .active_borrowed_paths
                        .push(activated.path);
                }
                next_pc = cur - f.instr_start;
            }
            Opcode::QAnd
            | Opcode::QOr
            | Opcode::QImpl
            | Opcode::QTruthAnd
            | Opcode::QTruthOr
            | Opcode::QTruthImpl => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lq = as_quad(get_reg(vm, frame_idx, lhs)?)?;
                let rq = as_quad(get_reg(vm, frame_idx, rhs)?)?;
                let out_q = match opcode {
                    Opcode::QAnd => quad_and(lq, rq),
                    Opcode::QOr => quad_or(lq, rq),
                    Opcode::QImpl => quad_implies(lq, rq),
                    Opcode::QTruthAnd => quad_truth_and(lq, rq),
                    Opcode::QTruthOr => quad_truth_or(lq, rq),
                    Opcode::QTruthImpl => quad_truth_implies(lq, rq),
                    _ => unreachable!(),
                };
                set_reg(vm, frame_idx, dst, Value::Quad(out_q))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::QNot | Opcode::QTruthNot => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let q = as_quad(get_reg(vm, frame_idx, src)?)?;
                let out_q = if opcode == Opcode::QNot {
                    quad_not(q)
                } else {
                    quad_truth_not(q)
                };
                set_reg(vm, frame_idx, dst, Value::Quad(out_q))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::BoolAnd | Opcode::BoolOr => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lb = as_bool(get_reg(vm, frame_idx, lhs)?)?;
                let rb = as_bool(get_reg(vm, frame_idx, rhs)?)?;
                let out_b = if opcode == Opcode::BoolAnd {
                    lb && rb
                } else {
                    lb || rb
                };
                set_reg(vm, frame_idx, dst, Value::Bool(out_b))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::BoolNot => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let b = as_bool(get_reg(vm, frame_idx, src)?)?;
                set_reg(vm, frame_idx, dst, Value::Bool(!b))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::CmpEq | Opcode::CmpNe => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lv = get_reg(vm, frame_idx, lhs)?;
                let rv = get_reg(vm, frame_idx, rhs)?;
                let eq = value_eq(&lv, &rv)?;
                let out = if opcode == Opcode::CmpEq { eq } else { !eq };
                set_reg(vm, frame_idx, dst, Value::Bool(out))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::CmpI32Lt | Opcode::CmpI32Le => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let l = as_i32(get_reg(vm, frame_idx, lhs)?)?;
                let r = as_i32(get_reg(vm, frame_idx, rhs)?)?;
                let out = if opcode == Opcode::CmpI32Lt {
                    l < r
                } else {
                    l <= r
                };
                set_reg(vm, frame_idx, dst, Value::Bool(out))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::AddF64 | Opcode::SubF64 | Opcode::MulF64 | Opcode::DivF64 => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let l = as_f64(get_reg(vm, frame_idx, lhs)?)?;
                let r = as_f64(get_reg(vm, frame_idx, rhs)?)?;
                let out = match opcode {
                    Opcode::AddF64 => l + r,
                    Opcode::SubF64 => l - r,
                    Opcode::MulF64 => l * r,
                    Opcode::DivF64 => l / r,
                    _ => unreachable!(),
                };
                set_reg(vm, frame_idx, dst, Value::F64(out))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::AddFx | Opcode::SubFx | Opcode::MulFx | Opcode::DivFx => {
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let lhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let rhs = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let l = as_fx(get_reg(vm, frame_idx, lhs)?)?;
                let r = as_fx(get_reg(vm, frame_idx, rhs)?)?;
                let out = match opcode {
                    Opcode::AddFx => fx_add_raw(l, r)?,
                    Opcode::SubFx => fx_sub_raw(l, r)?,
                    Opcode::MulFx => fx_mul_raw(l, r)?,
                    Opcode::DivFx => fx_div_raw(l, r)?,
                    _ => unreachable!(),
                };
                set_reg(vm, frame_idx, dst, Value::Fx(out))?;
                next_pc = cur - f.instr_start;
            }
            Opcode::Jmp => {
                let addr = read_u32_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                if addr >= instr_rel_len {
                    return Err(RuntimeError::InvalidJumpAddress {
                        func: func_name,
                        addr,
                    });
                }
                next_pc = addr;
            }
            Opcode::JmpIf => {
                let cond = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let addr = read_u32_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let b = as_bool(get_reg(vm, frame_idx, cond)?)?;
                if b {
                    if addr >= instr_rel_len {
                        return Err(RuntimeError::InvalidJumpAddress {
                            func: func_name,
                            addr,
                        });
                    }
                    next_pc = addr;
                } else {
                    next_pc = cur - f.instr_start;
                }
            }
            // Call target resolution precedence (#1653 / #1750, umbrella #1617):
            // an internal/program function named `callee` always wins over a
            // same-named builtin. This mirrors the verifier's already-correct
            // rule (crates/sm-verify/src/lib.rs: `known_functions.contains`
            // short-circuits before any builtin capability check) and the
            // source-level rule (crates/sm-front: user table checked before
            // `builtin_sig`). Before this fix, `try_eval_builtin_call` ran
            // unconditionally first, so a user-defined `sin`/`to_text`/etc.
            // was silently shadowed by the builtin at the one layer (the VM)
            // that actually executes code, even though both the frontend and
            // the verifier had already admitted/proven the user function.
            // Builtin resolution now only happens when no internal function
            // by that name exists.
            Opcode::Call => {
                let has_dst = read_u8(&f.code, &mut cur).map_err(map_format_err)? != 0;
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let callee_sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let argc = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
                let callee = lookup_str(&f, callee_sid)?.to_string();
                let mut args = Vec::with_capacity(argc);
                for _ in 0..argc {
                    let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                    args.push(get_reg(vm, frame_idx, r)?);
                }
                if vm.functions.contains_key(&callee) {
                    // Argument validation happens inside `push_frame` itself
                    // (#1773 / FA-09-005) - see `validate_call_arguments`'s
                    // doc comment.
                    vm.callstack[frame_idx].pc = cur - f.instr_start;
                    push_frame(vm, &callee, args, if has_dst { Some(dst) } else { None })?;
                    continue;
                } else if let Some(result) =
                    try_eval_builtin_call(host, observation, &callee, &args)?
                {
                    if has_dst {
                        set_reg(vm, frame_idx, dst, result)?;
                    }
                    next_pc = cur - f.instr_start;
                } else {
                    // Truly unknown callee: neither an internal function nor
                    // a builtin. Route through push_frame so the error
                    // (RuntimeError::UnknownFunction) is unchanged from
                    // pre-fix behavior.
                    vm.callstack[frame_idx].pc = cur - f.instr_start;
                    push_frame(vm, &callee, args, if has_dst { Some(dst) } else { None })?;
                    continue;
                }
            }
            Opcode::ClosureCall => {
                let has_dst = read_u8(&f.code, &mut cur).map_err(map_format_err)? != 0;
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let closure_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let arg_reg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let closure = get_reg(vm, frame_idx, closure_reg)?;
                let Value::Closure(closure) = closure else {
                    return Err(RuntimeError::TypeMismatchRuntime(
                        "CLOSURE_CALL source must be closure".to_string(),
                    ));
                };
                let mut args = closure.captures;
                args.push(get_reg(vm, frame_idx, arg_reg)?);
                vm.callstack[frame_idx].pc = cur - f.instr_start;
                push_frame(
                    vm,
                    &closure.function_name,
                    args,
                    if has_dst { Some(dst) } else { None },
                )?;
                continue;
            }
            Opcode::Assert => {
                let cond = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                match get_reg(vm, frame_idx, cond)? {
                    Value::Bool(true) => {
                        next_pc = cur - f.instr_start;
                    }
                    Value::Bool(false) => {
                        return Err(RuntimeError::Trap(RuntimeTrap::AssertionFailed));
                    }
                    other => {
                        return Err(RuntimeError::TypeMismatchRuntime(format!(
                            "ASSERT requires bool register, got {:?}",
                            other
                        )));
                    }
                }
            }
            Opcode::GateRead => {
                bump_effect_calls(vm)?;
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let device_id = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let port = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let value = host.gate_read(device_id, port)?;
                set_reg(vm, frame_idx, dst, value)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::GateWrite => {
                bump_effect_calls(vm)?;
                let device_id = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let port = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let value = get_reg(vm, frame_idx, src)?;
                host.gate_write(device_id, port, value)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::PulseEmit => {
                bump_effect_calls(vm)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let signal = lookup_str(&f, sid)?;
                host.pulse_emit(signal)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::StateQuery => {
                bump_effect_calls(vm)?;
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let key = lookup_str(&f, sid)?;
                let value = host.state_query(key)?;
                set_reg(vm, frame_idx, dst, value)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::StateUpdate => {
                bump_effect_calls(vm)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let key = lookup_str(&f, sid)?;
                let value = get_reg(vm, frame_idx, src)?;
                host.state_update(key, value)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::EventPost => {
                bump_effect_calls(vm)?;
                let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let signal = lookup_str(&f, sid)?;
                host.event_post(signal)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::ClockRead => {
                bump_effect_calls(vm)?;
                let dst = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                let value = host.clock_read()?;
                set_reg(vm, frame_idx, dst, value)?;
                next_pc = cur - f.instr_start;
            }
            Opcode::Ret => {
                let has_src = read_u8(&f.code, &mut cur).map_err(map_format_err)? != 0;
                let ret_val = if has_src {
                    let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                    get_reg(vm, frame_idx, src)?
                } else {
                    Value::Unit
                };
                let finished = vm.callstack.pop().ok_or(RuntimeError::StackUnderflow)?;
                if let Some(caller) = vm.callstack.last_mut() {
                    if let Some(dst) = finished.return_dst {
                        write_reg(caller, dst as usize, ret_val, &vm.config.quotas)?;
                    }
                } else {
                    #[cfg(test)]
                    {
                        let observable =
                            vm_test_format_terminal_observable(&ret_val, &finished, &vm.symbols);
                        vm_test_store_terminal_observation(VmTestObservation {
                            status: VmTestStatus::Completed,
                            observable: Some(observable),
                            trap: None,
                        });
                    }
                    return Ok(ret_val);
                }
                continue;
            }
        }

        vm.callstack[frame_idx].pc = next_pc;
    }
}

fn value_to_abi(value: Value) -> Result<AbiValue, RuntimeError> {
    match value {
        Value::Quad(q) => Ok(AbiValue::Quad(quad_to_u8(q))),
        Value::Bool(v) => Ok(AbiValue::Bool(v)),
        Value::Text(_) => Err(RuntimeError::TypeMismatchRuntime(
            "text values are not part of the PROMETHEUS host ABI surface".to_string(),
        )),
        Value::Sequence(_) => Err(RuntimeError::TypeMismatchRuntime(
            "sequence values are not part of the PROMETHEUS host ABI surface".to_string(),
        )),
        Value::Map(_) => Err(RuntimeError::TypeMismatchRuntime(
            "map values are not part of the PROMETHEUS host ABI surface".to_string(),
        )),
        Value::Closure(_) => Err(RuntimeError::TypeMismatchRuntime(
            "closure values are not part of the PROMETHEUS host ABI surface".to_string(),
        )),
        Value::I32(v) => Ok(AbiValue::I32(v)),
        Value::F64(v) => Ok(AbiValue::F64(v)),
        Value::U32(v) => Ok(AbiValue::U32(v)),
        Value::Fx(v) => Ok(AbiValue::Fx(v)),
        Value::Tuple(_) => Err(RuntimeError::TypeMismatchRuntime(
            "tuple values are not part of the PROMETHEUS host ABI surface".to_string(),
        )),
        Value::Record(_) => Err(RuntimeError::TypeMismatchRuntime(
            "record values are not part of the PROMETHEUS host ABI surface".to_string(),
        )),
        Value::Adt(_) => Err(RuntimeError::TypeMismatchRuntime(
            "enum values are not part of the PROMETHEUS host ABI surface".to_string(),
        )),
        Value::Unit => Ok(AbiValue::Unit),
    }
}

/// Converts a host-supplied `AbiValue` into a VM `Value`.
///
/// This is the untrusted host-ABI -> VM boundary (FA-09-007 / #1775): a
/// `PrometheusHostAbi` implementation is arbitrary caller-supplied code, and
/// its return values must be validated before they acquire VM-internal
/// meaning. `call` identifies which host operation produced `value`, so a
/// rejection carries accurate `AbiError` context.
fn value_from_abi(value: AbiValue, call: HostCallId) -> Result<Value, RuntimeError> {
    match value {
        AbiValue::Quad(raw) => quad_from_abi(raw).map(Value::Quad).map_err(|raw| {
            RuntimeError::HostAbi(AbiError::new(
                call,
                AbiFailureKind::InvalidInput,
                format!("host returned out-of-domain quad byte {raw} (valid domain is 0..=3)"),
            ))
        }),
        AbiValue::Bool(v) => Ok(Value::Bool(v)),
        AbiValue::I32(v) => Ok(Value::I32(v)),
        AbiValue::F64(v) => Ok(Value::F64(v)),
        AbiValue::U32(v) => Ok(Value::U32(v)),
        AbiValue::Fx(v) => Ok(Value::Fx(v)),
        AbiValue::Unit => Ok(Value::Unit),
    }
}

fn stable_state_query_fallback(key: &str) -> i32 {
    key.bytes().fold(0i32, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(i32::from(byte))
    })
}

/// The executable runtime family a `Value` actually belongs to (#1773 /
/// FA-09-005). Mirrors `sm-format`'s `CallableValueFamily` 1:1 - this is
/// the runtime-side half of the wire family tag; `callable_family_for_type`
/// in `sm-ir` is the compile-time half deriving the same tag from a source
/// `Type`. Exhaustive match, no wildcard, so a future `Value` variant added
/// without a family arm here is a compile-time error.
fn value_family(value: &Value) -> CallableValueFamily {
    match value {
        Value::Quad(_) => CallableValueFamily::Quad,
        Value::Bool(_) => CallableValueFamily::Bool,
        Value::Text(_) => CallableValueFamily::Text,
        Value::Sequence(_) => CallableValueFamily::Sequence,
        Value::Map(_) => CallableValueFamily::Map,
        Value::Closure(_) => CallableValueFamily::Closure,
        Value::I32(_) => CallableValueFamily::I32,
        Value::F64(_) => CallableValueFamily::F64,
        Value::U32(_) => CallableValueFamily::U32,
        Value::Fx(_) => CallableValueFamily::Fx,
        Value::Tuple(_) => CallableValueFamily::Tuple,
        Value::Record(_) => CallableValueFamily::Record,
        Value::Adt(_) => CallableValueFamily::Adt,
        Value::Unit => CallableValueFamily::Unit,
    }
}

/// The one authoritative invocation validator (#1773 / FA-09-005),
/// fused directly into `push_frame` (see below) rather than called
/// separately at each of its callers. An earlier revision of this PR called
/// it explicitly from just the public invocation helper and internal
/// `Opcode::Call`, reasoning that `push_frame` also backs the deliberately
/// unverified raw execution path and other verified entry routes shouldn't
/// need to remember to call it too. Codex review on this PR correctly
/// identified that reasoning as incomplete on both counts: several other
/// genuinely *verified* entry routes (`run_verified_entry_semcode_with_config`
/// and the PROMETHEUS/application host routes) call `push_frame` directly
/// and were left unvalidated, and `Opcode::ClosureCall` - not just
/// `Opcode::Call` - also calls `push_frame` and was left unvalidated too,
/// letting a `MAKE_CLOSURE` with the wrong capture count/family enter its
/// target frame unchecked. Fusing into `push_frame` closes every call site
/// at once, including any future one, rather than relying on each caller to
/// remember to call this separately. The raw/unverified path is safe to
/// include: it always passes empty args, so this only ever produces a
/// clean, deterministic rejection in place of a later, more confusing
/// uninitialized-register error - never a false rejection of a scenario the
/// raw path is trying to exercise.
///
/// `target_contract: None` means the callee's artifact header predates
/// canonical signatures (`SEMCODE_SIGNATURE_MIN_REVISION`) - nothing to
/// enforce, preserving pre-#1773 behavior for old artifacts exactly.
/// Checks too-few/too-many args and wrong family in both used AND unused
/// parameters (family is checked for every supplied argument regardless of
/// whether the callee's body ever reads that register), so no body
/// instruction is needed to expose a type mismatch.
fn validate_call_arguments(
    func_name: &str,
    target_contract: Option<&CallableSignature>,
    args: &[Value],
) -> Result<(), RuntimeError> {
    let Some(contract) = target_contract else {
        return Ok(());
    };
    if args.len() != contract.families.len() {
        return Err(RuntimeError::TypeMismatchRuntime(format!(
            "call to '{}' passed {} argument(s), but its canonical signature declares {} parameter(s)",
            func_name,
            args.len(),
            contract.families.len()
        )));
    }
    for (index, (arg, expected_family)) in args.iter().zip(&contract.families).enumerate() {
        let actual_family = value_family(arg);
        if actual_family != *expected_family {
            return Err(RuntimeError::TypeMismatchRuntime(format!(
                "call to '{}' argument {} has runtime family {:?}, but its canonical signature declares {:?}",
                func_name, index, actual_family, expected_family
            )));
        }
    }
    Ok(())
}

fn push_frame(
    vm: &mut VM,
    func_name: &str,
    args: Vec<Value>,
    return_dst: Option<u16>,
) -> Result<(), RuntimeError> {
    let f = vm
        .functions
        .get(func_name)
        .ok_or_else(|| RuntimeError::UnknownFunction(func_name.to_string()))?;
    validate_call_arguments(func_name, f.signature.as_ref(), &args)?;
    let next_depth = vm.callstack.len() + 1;
    enforce_quota(&vm.config.quotas, QuotaKind::Frames, next_depth)?;
    match enforce_quota(&vm.config.quotas, QuotaKind::StackDepth, next_depth) {
        Err(RuntimeError::QuotaExceeded(_)) => return Err(RuntimeError::StackOverflow),
        other => other?,
    }
    let initial_reg_count = 16usize.max(args.len());
    enforce_quota(&vm.config.quotas, QuotaKind::Registers, initial_reg_count)?;
    // #1759: the sole Calls choke point. `push_frame` is the single,
    // universal frame-construction path (root/entry and every nested
    // Opcode::Call/ClosureCall invocation alike), so the root exemption
    // uses the same structural signal already used above for the Frames
    // quota: the callstack was empty before this push only for the very
    // first (root/entry) frame of an execution. Charged immediately after
    // Frames/StackDepth/Registers admit this invocation and strictly before
    // any frame state (registers, ownership-borrow setup, the `Frame`
    // itself) is materialized - not merely before the eventual
    // `vm.callstack.push`, per the exact admission order frozen in
    // ssf08_1759_steps_calls_contract_decision.md §4 and §6: "construct
    // frame, push" is one combined final step that comes entirely after
    // the Calls charge, not interleaved with it.
    if !vm.callstack.is_empty() {
        vm.calls = charge_counter(vm.calls, vm.config.quotas.max_calls, QuotaKind::Calls)?;
    }
    let mut regs = vec![RegisterSlot::Uninitialized; initial_reg_count];
    for (i, v) in args.into_iter().enumerate() {
        regs[i] = RegisterSlot::Value(v);
    }
    // #1726 Checkpoint D3: `FrameEntry` Borrow paths are active from the
    // moment this frame exists (unchanged legacy/eager behavior, including
    // every rev20 artifact and every explicit rev21 `FrameEntry` event).
    // `StoreVarSite` paths start pending; they only move into
    // `active_borrowed_paths` when the `StoreVar` opcode handler observes
    // its own instruction's frame-relative PC match the anchor, after that
    // instruction's write has already committed - see the `StoreVar` arm
    // below.
    let mut active_borrowed_paths = Vec::new();
    let mut pending_site_borrows = Vec::new();
    for borrowed in &f.borrowed_paths {
        match borrowed.activation {
            BorrowActivation::FrameEntry => active_borrowed_paths.push(borrowed.path.clone()),
            BorrowActivation::StoreVarSite(anchor) => {
                pending_site_borrows.push(PendingSiteBorrow {
                    anchor,
                    path: borrowed.path.clone(),
                })
            }
        }
    }
    let frame = Frame {
        pc: 0,
        regs,
        locals: HashMap::new(),
        active_borrowed_paths,
        pending_site_borrows,
        func: f.name.clone(),
        return_dst,
    };
    vm.callstack.push(frame);
    Ok(())
}

fn access_paths_overlap(lhs: &AccessPath, rhs: &AccessPath) -> bool {
    if lhs.root != rhs.root {
        return false;
    }
    let shared_len = lhs.components.len().min(rhs.components.len());
    lhs.components[..shared_len] == rhs.components[..shared_len]
}

fn ensure_write_path_allowed(
    write_path: &AccessPath,
    borrowed_paths: &[AccessPath],
) -> Result<(), RuntimeError> {
    if borrowed_paths
        .iter()
        .any(|borrowed_path| access_paths_overlap(write_path, borrowed_path))
    {
        return Err(RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict));
    }
    Ok(())
}

/// #1891 Checkpoint W2F: the single dynamic hook - called once per
/// instruction dispatch, before that instruction's own effects commit (see
/// the call site in the dispatch loop above). Looks up every Write path
/// verifier-authenticated (Checkpoint W2E) against this exact PC and checks
/// each in the artifact's own declared order (`write_execution_sites` is a
/// plain sorted `Vec`, never a HashMap, so this is always deterministic).
/// Uniform across StoreVar and MakeRecord - see `WriteExecutionSite`'s own
/// doc comment for why the VM never re-derives which opcode class `pc`
/// belongs to. A PC with no matching site (the overwhelming majority of
/// instructions) costs one `binary_search_by_key` and nothing else - no
/// allocation, no per-visit rebuilding.
fn check_write_execution_site(
    f: &FunctionBytecode,
    pc: usize,
    active_borrowed_paths: &[AccessPath],
) -> Result<(), RuntimeError> {
    if let Ok(index) = f
        .write_execution_sites
        .binary_search_by_key(&pc, |site| site.pc)
    {
        for write_path in &f.write_execution_sites[index].paths {
            ensure_write_path_allowed(write_path, active_borrowed_paths)?;
        }
    }
    Ok(())
}

fn lookup_str<'a>(f: &'a FunctionBytecode, sid: u16) -> Result<&'a str, RuntimeError> {
    f.strings
        .get(sid as usize)
        .map(|s| s.as_str())
        .ok_or(RuntimeError::InvalidStringId(sid))
}

fn lookup_symbol(f: &FunctionBytecode, sid: u16) -> Result<SymbolId, RuntimeError> {
    f.symbol_ids
        .get(sid as usize)
        .copied()
        .ok_or(RuntimeError::InvalidStringId(sid))
}

fn get_reg(vm: &VM, frame_idx: usize, r: u16) -> Result<Value, RuntimeError> {
    match vm
        .callstack
        .get(frame_idx)
        .and_then(|fr| fr.regs.get(r as usize))
    {
        Some(RegisterSlot::Value(v)) => Ok(v.clone()),
        Some(RegisterSlot::Uninitialized) => Err(RuntimeError::BadFormat(format!(
            "read uninitialized reg r{}",
            r
        ))),
        None => Err(RuntimeError::BadFormat(format!("read invalid reg r{}", r))),
    }
}

fn set_reg(vm: &mut VM, frame_idx: usize, r: u16, v: Value) -> Result<(), RuntimeError> {
    if let Some(frame) = vm.callstack.get_mut(frame_idx) {
        write_reg(frame, r as usize, v, &vm.config.quotas)?;
    }
    Ok(())
}

fn write_reg(
    frame: &mut Frame,
    r: usize,
    v: Value,
    quotas: &RuntimeQuotas,
) -> Result<(), RuntimeError> {
    if frame.regs.len() <= r {
        let required = r + 1;
        enforce_quota(quotas, QuotaKind::Registers, required)?;
        frame.regs.resize(required, RegisterSlot::Uninitialized);
    }
    frame.regs[r] = RegisterSlot::Value(v);
    Ok(())
}

fn enforce_quota(quotas: &RuntimeQuotas, kind: QuotaKind, used: usize) -> Result<(), RuntimeError> {
    if let Some(exceeded) = quotas.exceed(kind, used) {
        return Err(RuntimeError::QuotaExceeded(exceeded));
    }
    Ok(())
}

/// #1759: shared charge primitive for the `Steps`/`Calls` execution-wide
/// counters, per `ssf08_1759_steps_calls_contract_decision.md` §10 - the
/// only place either counter is incremented. Overflow-safe by construction:
/// `used.checked_add(1)` never wraps. When the increment itself would
/// overflow (`used` was already `usize::MAX`), this fails closed with the
/// same `RuntimeError::QuotaExceeded` channel, reporting `used: usize::MAX`
/// as SATURATED REPORTING for that one unrepresentable attempted-usage case
/// only - never a silent wraparound, never a new error channel.
fn charge_counter(used: usize, limit: usize, kind: QuotaKind) -> Result<usize, RuntimeError> {
    match used.checked_add(1) {
        Some(next) if next > limit => Err(RuntimeError::QuotaExceeded(QuotaExceeded {
            kind,
            limit,
            used: next,
        })),
        Some(next) => Ok(next),
        None => Err(RuntimeError::QuotaExceeded(QuotaExceeded {
            kind,
            limit,
            used: usize::MAX,
        })),
    }
}

fn bump_effect_calls(vm: &mut VM) -> Result<(), RuntimeError> {
    let next = vm.effect_calls + 1;
    enforce_quota(&vm.config.quotas, QuotaKind::EffectCalls, next)?;
    vm.effect_calls = next;
    Ok(())
}

fn as_quad(v: Value) -> Result<QuadVal, RuntimeError> {
    if let Value::Quad(q) = v {
        Ok(q)
    } else {
        Err(RuntimeError::TypeMismatchRuntime(
            "expected quad".to_string(),
        ))
    }
}

fn as_bool(v: Value) -> Result<bool, RuntimeError> {
    if let Value::Bool(b) = v {
        Ok(b)
    } else {
        Err(RuntimeError::TypeMismatchRuntime(
            "expected bool".to_string(),
        ))
    }
}

fn as_i32(v: Value) -> Result<i32, RuntimeError> {
    if let Value::I32(x) = v {
        Ok(x)
    } else {
        Err(RuntimeError::TypeMismatchRuntime(
            "expected i32".to_string(),
        ))
    }
}

fn as_f64(v: Value) -> Result<f64, RuntimeError> {
    if let Value::F64(x) = v {
        Ok(x)
    } else {
        Err(RuntimeError::TypeMismatchRuntime(
            "expected f64".to_string(),
        ))
    }
}

fn as_fx(v: Value) -> Result<i32, RuntimeError> {
    if let Value::Fx(x) = v {
        Ok(x)
    } else {
        Err(RuntimeError::TypeMismatchRuntime("expected fx".to_string()))
    }
}

fn map_key_from_value(v: Value) -> Result<MapKey, RuntimeError> {
    match v {
        Value::I32(x) => Ok(MapKey::I32(x)),
        Value::U32(x) => Ok(MapKey::U32(x)),
        Value::Bool(x) => Ok(MapKey::Bool(x)),
        Value::Text(x) => Ok(MapKey::Text(x)),
        Value::Quad(q) => Ok(MapKey::Quad(quad_to_u8(q))),
        other => Err(RuntimeError::TypeMismatchRuntime(format!(
            "map key must be a scalar (i32, u32, bool, text, or quad), got {:?}",
            other
        ))),
    }
}

/// Advance the xorshift64 PRNG state by one step and return the new raw value.
/// If state is 0 (unseeded), treat it as seed 1 to avoid the zero fixed point.
fn xorshift64_step(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 1;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn fx_add_raw(lhs: i32, rhs: i32) -> Result<i32, RuntimeError> {
    i32::try_from(i64::from(lhs) + i64::from(rhs))
        .map_err(|_| RuntimeError::Trap(RuntimeTrap::ArithmeticOverflow))
}

fn fx_sub_raw(lhs: i32, rhs: i32) -> Result<i32, RuntimeError> {
    i32::try_from(i64::from(lhs) - i64::from(rhs))
        .map_err(|_| RuntimeError::Trap(RuntimeTrap::ArithmeticOverflow))
}

fn fx_mul_raw(lhs: i32, rhs: i32) -> Result<i32, RuntimeError> {
    let scaled = (i64::from(lhs) * i64::from(rhs)) / 1_000;
    i32::try_from(scaled).map_err(|_| RuntimeError::Trap(RuntimeTrap::ArithmeticOverflow))
}

fn fx_div_raw(lhs: i32, rhs: i32) -> Result<i32, RuntimeError> {
    if rhs == 0 {
        return Err(RuntimeError::Trap(RuntimeTrap::DivisionByZero));
    }
    let scaled = (i64::from(lhs) * 1_000) / i64::from(rhs);
    i32::try_from(scaled).map_err(|_| RuntimeError::Trap(RuntimeTrap::ArithmeticOverflow))
}

fn i32_div_raw(lhs: i32, rhs: i32) -> Result<i32, RuntimeError> {
    if rhs == 0 {
        return Err(RuntimeError::Trap(RuntimeTrap::DivisionByZero));
    }
    lhs.checked_div(rhs)
        .ok_or(RuntimeError::Trap(RuntimeTrap::ArithmeticOverflow))
}

fn i32_mod_raw(lhs: i32, rhs: i32) -> Result<i32, RuntimeError> {
    if rhs == 0 {
        return Err(RuntimeError::Trap(RuntimeTrap::DivisionByZero));
    }
    lhs.checked_rem(rhs)
        .ok_or(RuntimeError::Trap(RuntimeTrap::ArithmeticOverflow))
}

fn quad_to_u8(q: QuadVal) -> u8 {
    match q {
        QuadVal::N => 0,
        QuadVal::F => 1,
        QuadVal::T => 2,
        QuadVal::S => 3,
    }
}

// Only exercised by `mod tests`' legacy reference implementations (see
// `legacy_qnot`/`legacy_qand`/`legacy_qor`), which round-trip already-valid
// `QuadVal`s through `u8` and are mathematically closed under `0..=3`. The
// untrusted host-ABI boundary uses `quad_from_abi` instead (see #1775).
#[cfg(test)]
fn u8_to_quad(v: u8) -> QuadVal {
    match v & 0b11 {
        0 => QuadVal::N,
        1 => QuadVal::F,
        2 => QuadVal::T,
        _ => QuadVal::S,
    }
}

/// Validates a raw host-ABI quad byte against the canonical four-value
/// domain (0=N, 1=F, 2=T, 3=S).
///
/// Unlike `u8_to_quad` (which masks bytes that are trusted, internally
/// produced, and already known to be closed under `0..=3`, e.g. bitwise
/// combinations of two already-valid `QuadVal`s), this function is the
/// gatekeeper for bytes arriving from an untrusted host ABI implementation.
/// It rejects anything outside `0..=3` instead of silently truncating it
/// into a plausible value. See `value_from_abi`.
fn quad_from_abi(raw: u8) -> Result<QuadVal, u8> {
    match raw {
        0 => Ok(QuadVal::N),
        1 => Ok(QuadVal::F),
        2 => Ok(QuadVal::T),
        3 => Ok(QuadVal::S),
        other => Err(other),
    }
}

fn quadval_to_quadstate(q: QuadVal) -> QuadState {
    match q {
        QuadVal::N => QuadState::N,
        QuadVal::F => QuadState::F,
        QuadVal::T => QuadState::T,
        QuadVal::S => QuadState::S,
    }
}

fn quadstate_to_quadval(s: QuadState) -> QuadVal {
    match s {
        QuadState::N => QuadVal::N,
        QuadState::F => QuadVal::F,
        QuadState::T => QuadVal::T,
        QuadState::S => QuadVal::S,
    }
}

fn quad_lane0(q: QuadVal) -> QuadroReg32 {
    let mut reg = QuadroReg32::from_raw(0);
    reg.set_unchecked(0, quadval_to_quadstate(q));
    reg
}

fn quad_lane0_value(reg: QuadroReg32) -> QuadVal {
    quadstate_to_quadval(reg.try_get(0).unwrap())
}

fn quad_not(a: QuadVal) -> QuadVal {
    quad_lane0_value(quad_lane0(a).lattice_inverse())
}

fn quad_and(a: QuadVal, b: QuadVal) -> QuadVal {
    quad_lane0_value(quad_lane0(a).lattice_meet(quad_lane0(b)))
}

fn quad_or(a: QuadVal, b: QuadVal) -> QuadVal {
    quad_lane0_value(quad_lane0(a).lattice_join(quad_lane0(b)))
}

fn quad_implies(a: QuadVal, b: QuadVal) -> QuadVal {
    quad_lane0_value(quad_lane0(a).lattice_inverse().lattice_join(quad_lane0(b)))
}

fn quad_truth_not(a: QuadVal) -> QuadVal {
    quad_lane0_value(quad_lane0(a).map_not())
}

fn quad_truth_and(a: QuadVal, b: QuadVal) -> QuadVal {
    quad_lane0_value(quad_lane0(a).map_and(quad_lane0(b)))
}

fn quad_truth_or(a: QuadVal, b: QuadVal) -> QuadVal {
    quad_lane0_value(quad_lane0(a).map_or(quad_lane0(b)))
}

fn quad_truth_implies(a: QuadVal, b: QuadVal) -> QuadVal {
    quad_lane0_value(quad_lane0(a).map_implies(quad_lane0(b)))
}

fn value_eq(a: &Value, b: &Value) -> Result<bool, RuntimeError> {
    match (a, b) {
        (Value::Quad(x), Value::Quad(y)) => Ok(x == y),
        (Value::Bool(x), Value::Bool(y)) => Ok(x == y),
        (Value::Text(x), Value::Text(y)) => Ok(x == y),
        (Value::Sequence(xs), Value::Sequence(ys)) => {
            if xs.len() != ys.len() {
                return Ok(false);
            }
            for (x, y) in xs.iter().zip(ys.iter()) {
                if !value_eq(x, y)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (Value::Map(_), Value::Map(_)) => Err(RuntimeError::TypeMismatchRuntime(
            "Map values are not comparable with == / !=".to_string(),
        )),
        (Value::Closure(_), Value::Closure(_)) => Err(RuntimeError::TypeMismatchRuntime(
            "closure values are not comparable with CmpEq/CmpNe".to_string(),
        )),
        (Value::I32(x), Value::I32(y)) => Ok(x == y),
        (Value::F64(x), Value::F64(y)) => Ok(x == y),
        (Value::U32(x), Value::U32(y)) => Ok(x == y),
        (Value::Fx(x), Value::Fx(y)) => Ok(x == y),
        (Value::Tuple(xs), Value::Tuple(ys)) => {
            if xs.len() != ys.len() {
                return Ok(false);
            }
            for (x, y) in xs.iter().zip(ys.iter()) {
                if !value_eq(x, y)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (Value::Record(xs), Value::Record(ys)) => {
            if xs.type_name != ys.type_name {
                return Ok(false);
            }
            if xs.slots.len() != ys.slots.len() {
                return Ok(false);
            }
            for (x, y) in xs.slots.iter().zip(ys.slots.iter()) {
                if !value_eq(x, y)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (Value::Adt(xs), Value::Adt(ys)) => {
            if xs.type_name != ys.type_name
                || xs.variant_name != ys.variant_name
                || xs.tag != ys.tag
                || xs.payload.len() != ys.payload.len()
            {
                return Ok(false);
            }
            for (x, y) in xs.payload.iter().zip(ys.payload.iter()) {
                if !value_eq(x, y)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (Value::Unit, Value::Unit) => Ok(true),
        _ => Err(RuntimeError::TypeMismatchRuntime(
            "CmpEq/CmpNe operands must have same runtime type".to_string(),
        )),
    }
}

fn try_eval_builtin_call<'a, H: VmHostBridge>(
    host: &mut H,
    observation: &mut HelloObservationRuntime<'a>,
    name: &str,
    args: &[Value],
) -> Result<Option<Value>, RuntimeError> {
    let value = match name {
        "sin" => Value::F64(expect_builtin_unary_f64(name, args)?.sin()),
        "cos" => Value::F64(expect_builtin_unary_f64(name, args)?.cos()),
        "tan" => Value::F64(expect_builtin_unary_f64(name, args)?.tan()),
        "sqrt" => Value::F64(expect_builtin_unary_f64(name, args)?.sqrt()),
        "abs" => Value::F64(expect_builtin_unary_f64(name, args)?.abs()),
        "pow" => {
            let (lhs, rhs) = expect_builtin_binary_f64(name, args)?;
            Value::F64(lhs.powf(rhs))
        }
        "to_text" => Value::Text(value_to_text(expect_builtin_to_text_arg(name, args)?)?),
        "print" => {
            if args.len() != 1 {
                return Err(RuntimeError::TypeMismatchRuntime(format!(
                    "builtin 'print' expects 1 argument, got {}",
                    args.len()
                )));
            }
            let text = match &args[0] {
                Value::Text(s) => s.clone(),
                other => {
                    return Err(RuntimeError::TypeMismatchRuntime(format!(
                        "builtin 'print' expects text, got {:?}",
                        other
                    )));
                }
            };
            observation.record_controlled_text_observation(text)?;
            Value::Unit
        }
        "args_read" => {
            let index = expect_builtin_u32(name, args)?;
            host.args_read(index)?
        }
        "stdin_read_text" => {
            expect_builtin_arity(name, args, 0)?;
            host.stdin_read_text()?
        }
        "stdout_write" => {
            let text = expect_builtin_text(name, args, 1)?;
            host.stdout_write(text)?;
            Value::Unit
        }
        "stderr_write" => {
            let text = expect_builtin_text(name, args, 1)?;
            host.stderr_write(text)?;
            Value::Unit
        }
        "path_inspect" => {
            let path = expect_builtin_text(name, args, 1)?;
            host.path_inspect(path)?
        }
        "fs_read_text" => {
            let path = expect_builtin_text(name, args, 1)?;
            host.fs_read_text(path)?
        }
        "fs_write_text" => {
            expect_builtin_arity(name, args, 2)?;
            let path = match &args[0] {
                Value::Text(value) => value.as_str(),
                _ => {
                    return Err(RuntimeError::TypeMismatchRuntime(format!(
                        "builtin '{name}' expects text path"
                    )))
                }
            };
            let text = match &args[1] {
                Value::Text(value) => value.as_str(),
                _ => {
                    return Err(RuntimeError::TypeMismatchRuntime(format!(
                        "builtin '{name}' expects text payload"
                    )))
                }
            };
            host.fs_write_text(path, text)?;
            Value::Unit
        }
        "time_duration_ms" => {
            expect_builtin_arity(name, args, 0)?;
            host.time_duration_millis()?
        }
        _ => return Ok(None),
    };
    Ok(Some(value))
}

fn expect_builtin_arity(name: &str, args: &[Value], expected: usize) -> Result<(), RuntimeError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(RuntimeError::TypeMismatchRuntime(format!(
            "builtin '{name}' expects {expected} arguments, got {}",
            args.len()
        )))
    }
}

fn expect_builtin_u32(name: &str, args: &[Value]) -> Result<u32, RuntimeError> {
    expect_builtin_arity(name, args, 1)?;
    match args[0] {
        Value::U32(value) => Ok(value),
        ref other => Err(RuntimeError::TypeMismatchRuntime(format!(
            "builtin '{name}' expects u32, got {other:?}"
        ))),
    }
}

fn expect_builtin_text<'a>(
    name: &str,
    args: &'a [Value],
    expected: usize,
) -> Result<&'a str, RuntimeError> {
    expect_builtin_arity(name, args, expected)?;
    match &args[0] {
        Value::Text(value) => Ok(value.as_str()),
        _ => Err(RuntimeError::TypeMismatchRuntime(format!(
            "builtin '{name}' expects text"
        ))),
    }
}

fn decode_text_literal(raw: &str) -> String {
    raw.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(raw)
        .to_string()
}

fn as_text(v: Value) -> Result<String, RuntimeError> {
    if let Value::Text(x) = v {
        Ok(x)
    } else {
        Err(RuntimeError::TypeMismatchRuntime(
            "expected text".to_string(),
        ))
    }
}

fn value_to_text(v: Value) -> Result<String, RuntimeError> {
    match v {
        Value::Text(x) => Ok(x),
        Value::Bool(b) => Ok(if b { "true" } else { "false" }.to_string()),
        Value::I32(x) => Ok(x.to_string()),
        Value::U32(x) => Ok(x.to_string()),
        Value::Quad(q) => Ok(match q {
            QuadVal::N => "N",
            QuadVal::T => "T",
            QuadVal::F => "F",
            QuadVal::S => "S",
        }
        .to_string()),
        other => Err(RuntimeError::TypeMismatchRuntime(format!(
            "builtin 'to_text' currently supports text, bool, i32, u32, and quad; got {:?}",
            other
        ))),
    }
}

fn expect_builtin_to_text_arg(name: &str, args: &[Value]) -> Result<Value, RuntimeError> {
    if args.len() != 1 {
        return Err(RuntimeError::TypeMismatchRuntime(format!(
            "builtin '{name}' expects 1 argument, got {}",
            args.len()
        )));
    }
    Ok(args[0].clone())
}

fn expect_builtin_unary_f64(name: &str, args: &[Value]) -> Result<f64, RuntimeError> {
    if args.len() != 1 {
        return Err(RuntimeError::TypeMismatchRuntime(format!(
            "builtin '{name}' expects 1 f64 argument, got {}",
            args.len()
        )));
    }
    as_f64(args[0].clone())
}

fn expect_builtin_binary_f64(name: &str, args: &[Value]) -> Result<(f64, f64), RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::TypeMismatchRuntime(format!(
            "builtin '{name}' expects 2 f64 arguments, got {}",
            args.len()
        )));
    }
    Ok((as_f64(args[0].clone())?, as_f64(args[1].clone())?))
}

#[cfg(feature = "disasm")]
fn disasm_one(f: &FunctionBytecode, pc: usize) -> Result<(String, usize), RuntimeError> {
    let mut cur = f.instr_start + pc;
    let opcode = Opcode::from_byte(read_u8(&f.code, &mut cur).map_err(map_format_err)?)
        .map_err(map_format_err)?;
    let text = match opcode {
        Opcode::LoadQ => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let q = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_Q r{}, {}", d, q)
        }
        Opcode::LoadBool => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let b = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_BOOL r{}, {}", d, b)
        }
        Opcode::LoadText => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let text = lookup_str(f, sid)?;
            format!("LOAD_TEXT r{}, {:?}", d, text)
        }
        Opcode::ConcatText => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let l = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("CONCAT_TEXT r{}, r{}, r{}", d, l, r)
        }
        Opcode::LoadI32 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let n = read_i32_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_I32 r{}, {}", d, n)
        }
        Opcode::AddI32 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let l = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("ADD_I32 r{}, r{}, r{}", d, l, r)
        }
        Opcode::SubI32 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let l = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("SUB_I32 r{}, r{}, r{}", d, l, r)
        }
        Opcode::MulI32 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let l = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("MUL_I32 r{}, r{}, r{}", d, l, r)
        }
        Opcode::DivI32 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let l = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("DIV_I32 r{}, r{}, r{}", d, l, r)
        }
        Opcode::ModI32 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let l = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("MOD_I32 r{}, r{}, r{}", d, l, r)
        }
        Opcode::LoadU32 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let n = read_u32_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_U32 r{}, {}", d, n)
        }
        Opcode::LoadF64 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let n = read_f64_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_F64 r{}, {}", d, n)
        }
        Opcode::LoadFx => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let n = read_i32_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_FX r{}, raw:{}", d, n)
        }
        Opcode::MakeSequence => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
            let mut regs = Vec::with_capacity(count);
            for _ in 0..count {
                regs.push(read_u16_le(&f.code, &mut cur).map_err(map_format_err)?);
            }
            let regs = regs
                .iter()
                .map(|reg| format!("r{}", reg))
                .collect::<Vec<_>>()
                .join(", ");
            format!("MAKE_SEQUENCE r{}, [{}]", d, regs)
        }
        Opcode::MakeClosure => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
            let mut regs = Vec::with_capacity(count);
            for _ in 0..count {
                regs.push(read_u16_le(&f.code, &mut cur).map_err(map_format_err)?);
            }
            let regs = regs
                .iter()
                .map(|reg| format!("r{}", reg))
                .collect::<Vec<_>>()
                .join(", ");
            let target = lookup_str(f, sid)?;
            format!("MAKE_CLOSURE r{}, {}, [{}]", d, target, regs)
        }
        Opcode::AddFx | Opcode::SubFx | Opcode::MulFx | Opcode::DivFx => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let l = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let op = match opcode {
                Opcode::AddFx => "ADD_FX",
                Opcode::SubFx => "SUB_FX",
                Opcode::MulFx => "MUL_FX",
                Opcode::DivFx => "DIV_FX",
                _ => unreachable!(),
            };
            format!("{} r{}, r{}, r{}", op, d, l, r)
        }
        Opcode::MakeTuple => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
            let mut regs = Vec::with_capacity(count);
            for _ in 0..count {
                regs.push(read_u16_le(&f.code, &mut cur).map_err(map_format_err)?);
            }
            let regs = regs
                .iter()
                .map(|reg| format!("r{}", reg))
                .collect::<Vec<_>>()
                .join(", ");
            format!("MAKE_TUPLE r{}, [{}]", d, regs)
        }
        Opcode::MakeRecord => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
            let mut regs = Vec::with_capacity(count);
            for _ in 0..count {
                regs.push(read_u16_le(&f.code, &mut cur).map_err(map_format_err)?);
            }
            let regs = regs
                .iter()
                .map(|reg| format!("r{}", reg))
                .collect::<Vec<_>>()
                .join(", ");
            let name = lookup_str(f, sid)?;
            format!("MAKE_RECORD r{}, {}, [{}]", d, name, regs)
        }
        Opcode::MakeAdt => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let variant_sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let tag = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let count = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
            let mut regs = Vec::with_capacity(count);
            for _ in 0..count {
                regs.push(read_u16_le(&f.code, &mut cur).map_err(map_format_err)?);
            }
            let regs = regs
                .iter()
                .map(|reg| format!("r{}", reg))
                .collect::<Vec<_>>()
                .join(", ");
            let name = lookup_str(f, sid)?;
            let variant = lookup_str(f, variant_sid)?;
            format!(
                "MAKE_ADT r{}, {}::{}, tag={}, [{}]",
                d, name, variant, tag, regs
            )
        }
        Opcode::AdtTag => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let name = lookup_str(f, sid)?;
            format!("ADT_TAG r{}, r{}, {}", d, s, name)
        }
        Opcode::AdtGet => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let i = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let name = lookup_str(f, sid)?;
            format!("ADT_GET r{}, r{}, {}, {}", d, s, name, i)
        }
        Opcode::RecordGet => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let i = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let name = lookup_str(f, sid)?;
            format!("RECORD_GET r{}, r{}, {}, {}", d, s, name, i)
        }
        Opcode::TupleGet => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let i = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("TUPLE_GET r{}, r{}, {}", d, s, i)
        }
        Opcode::SequenceGet => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let i = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("SEQUENCE_GET r{}, r{}, r{}", d, s, i)
        }
        Opcode::SequenceLen => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("SEQUENCE_LEN r{}, r{}", d, s)
        }
        Opcode::SequenceIsEmpty => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("SEQUENCE_IS_EMPTY r{}, r{}", d, s)
        }
        Opcode::SequenceContains => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let v = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("SEQUENCE_CONTAINS r{}, r{}, r{}", d, s, v)
        }
        Opcode::SequencePush => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let v = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("SEQUENCE_PUSH r{}, r{}, r{}", d, s, v)
        }
        Opcode::SequencePrepend => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let v = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("SEQUENCE_PREPEND r{}, r{}, r{}", d, s, v)
        }
        Opcode::SequencePop => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("SEQUENCE_POP r{}, r{}", d, s)
        }
        Opcode::MapEmpty => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("MAP_EMPTY r{}", d)
        }
        Opcode::MapContains => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let m = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let k = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("MAP_CONTAINS r{}, r{}, r{}", d, m, k)
        }
        Opcode::MapGet => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let m = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let k = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let dv = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("MAP_GET r{}, r{}, r{}, r{}", d, m, k, dv)
        }
        Opcode::MapSet => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let m = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let k = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let v = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("MAP_SET r{}, r{}, r{}, r{}", d, m, k, v)
        }
        Opcode::RngSeed => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("RNG_SEED r{}, r{}", d, s)
        }
        Opcode::RngNextI32 => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let lo = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let hi = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("RNG_NEXT_I32 r{}, r{}, r{}", d, lo, hi)
        }
        Opcode::ClosureCall => {
            let has_dst = read_u8(&f.code, &mut cur).map_err(map_format_err)? != 0;
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let closure = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let arg = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            if has_dst {
                format!("CLOSURE_CALL r{}, r{}, r{}", d, closure, arg)
            } else {
                format!("CLOSURE_CALL -, r{}, r{}", closure, arg)
            }
        }
        Opcode::LoadVar => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("LOAD_VAR r{}, s{}", d, s)
        }
        Opcode::StoreVar => {
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("STORE_VAR s{}, r{}", s, r)
        }
        Opcode::QAnd
        | Opcode::QOr
        | Opcode::QImpl
        | Opcode::QTruthAnd
        | Opcode::QTruthOr
        | Opcode::QTruthImpl
        | Opcode::BoolAnd
        | Opcode::BoolOr
        | Opcode::CmpI32Lt
        | Opcode::CmpI32Le
        | Opcode::AddF64
        | Opcode::SubF64
        | Opcode::MulF64
        | Opcode::DivF64
        | Opcode::CmpEq
        | Opcode::CmpNe => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let l = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let op = match opcode {
                Opcode::QAnd => "Q_AND",
                Opcode::QOr => "Q_OR",
                Opcode::QImpl => "Q_IMPL",
                Opcode::QTruthAnd => "Q_TRUTH_AND",
                Opcode::QTruthOr => "Q_TRUTH_OR",
                Opcode::QTruthImpl => "Q_TRUTH_IMPL",
                Opcode::BoolAnd => "BOOL_AND",
                Opcode::BoolOr => "BOOL_OR",
                Opcode::CmpI32Lt => "CMP_I32_LT",
                Opcode::CmpI32Le => "CMP_I32_LE",
                Opcode::AddI32 => "ADD_I32",
                Opcode::SubI32 => "SUB_I32",
                Opcode::MulI32 => "MUL_I32",
                Opcode::DivI32 => "DIV_I32",
                Opcode::ModI32 => "MOD_I32",
                Opcode::AddF64 => "ADD_F64",
                Opcode::SubF64 => "SUB_F64",
                Opcode::MulF64 => "MUL_F64",
                Opcode::DivF64 => "DIV_F64",
                Opcode::AddFx => "ADD_FX",
                Opcode::SubFx => "SUB_FX",
                Opcode::MulFx => "MUL_FX",
                Opcode::DivFx => "DIV_FX",
                Opcode::CmpEq => "CMP_EQ",
                _ => "CMP_NE",
            };
            format!("{} r{}, r{}, r{}", op, d, l, r)
        }
        Opcode::QNot | Opcode::BoolNot | Opcode::QTruthNot => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let op = match opcode {
                Opcode::QNot => "Q_NOT",
                Opcode::QTruthNot => "Q_TRUTH_NOT",
                Opcode::BoolNot => "BOOL_NOT",
                _ => unreachable!(),
            };
            format!("{} r{}, r{}", op, d, s)
        }
        Opcode::Jmp => {
            let a = read_u32_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("JMP {}", a)
        }
        Opcode::JmpIf => {
            let c = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let a = read_u32_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("JMP_IF r{}, {}", c, a)
        }
        Opcode::Call => {
            let has = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let n = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let argc = read_u16_le(&f.code, &mut cur).map_err(map_format_err)? as usize;
            for _ in 0..argc {
                let _ = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            }
            format!("CALL dst?{} r{} fn#{} argc={}", has, d, n, argc)
        }
        Opcode::Assert => {
            let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("ASSERT r{}", r)
        }
        Opcode::GateRead => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let dev = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let port = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("GATE_READ r{}, dev={}, port={}", d, dev, port)
        }
        Opcode::GateWrite => {
            let dev = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let port = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let s = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("GATE_WRITE dev={}, port={}, r{}", dev, port, s)
        }
        Opcode::PulseEmit => {
            let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("PULSE_EMIT s{}", sid)
        }
        Opcode::StateQuery => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("STATE_QUERY r{}, s{}", d, sid)
        }
        Opcode::StateUpdate => {
            let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            let src = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("STATE_UPDATE s{}, r{}", sid, src)
        }
        Opcode::EventPost => {
            let sid = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("EVENT_POST s{}", sid)
        }
        Opcode::ClockRead => {
            let d = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
            format!("CLOCK_READ r{}", d)
        }
        Opcode::Ret => {
            let has = read_u8(&f.code, &mut cur).map_err(map_format_err)?;
            if has != 0 {
                let r = read_u16_le(&f.code, &mut cur).map_err(map_format_err)?;
                format!("RET r{}", r)
            } else {
                "RET".to_string()
            }
        }
    };
    Ok((text, cur - f.instr_start))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semcode_format::read_utf8;
    use sm_emit::{
        compile_program_to_semcode, compile_program_to_semcode_with_options, CompileProfile,
        OptLevel, ACTIVATION_MODE_FRAME_ENTRY, OWNERSHIP_EVENT_KIND_BORROW,
        OWNERSHIP_EVENT_KIND_WRITE, OWNERSHIP_PATH_COMPONENT_FIELD_SYMBOL,
        OWNERSHIP_PATH_COMPONENT_TUPLE_INDEX, OWNERSHIP_SECTION_TAG,
        SEMCODE_OWNERSHIP_ANCHOR_MIN_REVISION, SIGNATURE_SECTION_TAG,
        WRITE_EXECUTION_MODE_STORE_VAR_SITE,
    };
    use sm_ir::{emit_ir_to_semcode, IrFunction, IrInstr};
    use sm_runtime_core::{
        ExecutionConfig, ExecutionContext, PathComponent, QuotaExceeded, QuotaKind, RuntimeTrap,
    };

    #[cfg(feature = "vm-profile")]
    mod profile_tests {
        use super::*;
        use sm_emit::Opcode;

        #[test]
        fn opcode_profile_starts_empty_and_records_counts() {
            let mut profile = VmOpcodeProfile::default();
            assert!(profile.is_empty());
            assert_eq!(profile.total_instructions(), 0);
            assert_eq!(profile.count(Opcode::LoadQ), 0);
            assert!(profile.top_n(0).is_empty());

            profile.record_opcode_slot(Opcode::LoadQ);
            profile.record_opcode_slot(Opcode::LoadQ);
            profile.record_opcode_slot(Opcode::QAnd);

            assert!(!profile.is_empty());
            assert_eq!(profile.total_instructions(), 3);
            assert_eq!(profile.count(Opcode::LoadQ), 2);
            assert_eq!(profile.count(Opcode::QAnd), 1);
            assert_eq!(profile.count(Opcode::Ret), 0);
        }

        #[test]
        fn opcode_profile_ranks_and_summarizes_top_entries() {
            let mut profile = VmOpcodeProfile::default();
            profile.record_opcode_slot(Opcode::LoadQ);
            profile.record_opcode_slot(Opcode::LoadQ);
            profile.record_opcode_slot(Opcode::QAnd);
            profile.record_opcode_slot(Opcode::QAnd);
            profile.record_opcode_slot(Opcode::QAnd);
            profile.record_opcode_slot(Opcode::CmpEq);

            let top = profile.top_n(3);
            assert_eq!(top.len(), 3);
            assert_eq!(top[0], (Opcode::QAnd, 3));
            assert_eq!(top[1], (Opcode::LoadQ, 2));
            assert_eq!(top[2], (Opcode::CmpEq, 1));

            let summary = profile.summary_top_n(3);
            assert!(summary.contains("sm-vm opcode profile: total_instructions=6"));
            assert!(summary.contains("QAnd"));
            assert!(summary.contains("LoadQ"));
            assert!(summary.contains("CmpEq"));
        }

        #[test]
        fn opcode_profile_table_round_trips_all_slots() {
            for (index, opcode) in OPCODE_PROFILE_OPCODES.iter().enumerate() {
                assert_eq!(opcode_profile_index(*opcode), index);
                assert_eq!(opcode_from_profile_index(index), Some(*opcode));
            }
            assert_eq!(opcode_from_profile_index(OPCODE_PROFILE_SLOT_COUNT), None);
        }
    }

    mod helper_boundary_result_observation_tests {
        use super::*;

        fn observe_verified_entry_semcode(name: &str, source: &str) -> VmTestObservation {
            let bytes = compile_program_to_semcode(source)
                .unwrap_or_else(|err| panic!("{name}: compile failed: {err:?}"));
            let token = sm_verify::verify_semcode_token(&bytes)
                .unwrap_or_else(|err| panic!("{name}: verify failed: {err}"));
            let entry = token
                .require_entry("main")
                .unwrap_or_else(|err| panic!("{name}: entry resolution failed: {err:?}"));

            vm_test_clear_terminal_observation();
            let result = run_verified_entry_semcode_with_config(
                &entry,
                ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
            );
            match result {
                Ok(()) => vm_test_take_terminal_observation().unwrap_or(VmTestObservation {
                    status: VmTestStatus::Failed,
                    observable: None,
                    trap: Some(format!("{name}: missing terminal observation")),
                }),
                Err(err) => VmTestObservation {
                    status: VmTestStatus::Failed,
                    observable: None,
                    trap: Some(format!("{err}")),
                },
            }
        }

        fn canonicalize_terminal_observable(observable: &str, semantic_locals: &[&str]) -> String {
            let (ret_part, locals_part) = observable
                .split_once("; locals=")
                .expect("terminal observable return/locals split");
            let locals_part = locals_part
                .strip_prefix('[')
                .and_then(|s| s.strip_suffix(']'))
                .expect("terminal observable locals list");

            let locals = locals_part
                .split(", ")
                .filter_map(|entry| entry.split_once('='))
                .map(|(name, value)| (name.trim(), value.trim()))
                .collect::<Vec<_>>();

            let mut selected = Vec::new();
            for wanted in semantic_locals {
                if let Some((_, value)) = locals.iter().find(|(name, _)| name == wanted) {
                    selected.push(format!("{wanted}={value}"));
                }
            }

            format!("{ret_part}; locals=[{}]", selected.join(", "))
        }

        #[test]
        fn private_vm_observation_helper_captures_terminal_state() {
            let observation = observe_verified_entry_semcode("empty_main", "fn main() { return; }");
            assert_eq!(observation.status, VmTestStatus::Completed);
            assert!(observation.trap.is_none());

            let observable = observation
                .observable
                .as_deref()
                .expect("terminal observable");
            assert!(observable.contains("return=Unit"));
            assert!(observable.contains("locals=[]"));
        }

        #[test]
        fn source_qtruth_intrinsics_compile_verify_and_execute() {
            let cases = [
                (
                    "qtruth_and",
                    "fn main() { let result: quad = qtruth_and(T, F); return; }",
                    "return=Unit; locals=[__sm_local_0_result=Quad(F)]",
                ),
                (
                    "qtruth_or",
                    "fn main() { let result: quad = qtruth_or(T, F); return; }",
                    "return=Unit; locals=[__sm_local_0_result=Quad(T)]",
                ),
                (
                    "qtruth_not",
                    "fn main() { let result: quad = qtruth_not(T); return; }",
                    "return=Unit; locals=[__sm_local_0_result=Quad(F)]",
                ),
                (
                    "qtruth_impl",
                    "fn main() { let result: quad = qtruth_impl(T, F); return; }",
                    "return=Unit; locals=[__sm_local_0_result=Quad(F)]",
                ),
            ];

            for (name, source, expected) in cases {
                let observation = observe_verified_entry_semcode(name, source);
                assert_eq!(
                    observation.status,
                    VmTestStatus::Completed,
                    "{name} did not complete"
                );
                assert!(
                    observation.trap.is_none(),
                    "{name} trapped: {:?}",
                    observation.trap
                );
                assert_eq!(
                    observation.observable.as_deref(),
                    Some(expected),
                    "{name} result"
                );
            }
        }

        struct HelperBoundaryPair {
            name: &'static str,
            helper_fixture: &'static str,
            inline_fixture: &'static str,
            helper_source: &'static str,
            inline_source: &'static str,
            semantic_locals: &'static [&'static str],
        }

        const HELPER_BOUNDARY_PAIRS: &[HelperBoundaryPair] = &[
            HelperBoundaryPair {
                name: "vm-m9 helper boundary",
                helper_fixture: "scalar_helper_boundary_helper.sm",
                inline_fixture: "scalar_helper_boundary_inline.sm",
                helper_source: include_str!(
                    "../tests/fixtures/profiling/scalar_movement/scalar_helper_boundary_helper.sm"
                ),
                inline_source: include_str!(
                    "../tests/fixtures/profiling/scalar_movement/scalar_helper_boundary_inline.sm"
                ),
                semantic_locals: &["checksum", "merged_count", "score"],
            },
            HelperBoundaryPair {
                name: "g2 helper single-call",
                helper_fixture: "scalar_helper_boundary_single_call_helper.sm",
                inline_fixture: "scalar_helper_boundary_single_call_inline.sm",
                helper_source: include_str!(
                    "../tests/fixtures/profiling/scalar_movement/g2/scalar_helper_boundary_single_call_helper.sm"
                ),
                inline_source: include_str!(
                    "../tests/fixtures/profiling/scalar_movement/g2/scalar_helper_boundary_single_call_inline.sm"
                ),
                semantic_locals: &["checksum", "hit_count", "score"],
            },
            HelperBoundaryPair {
                name: "g2 helper call-chain",
                helper_fixture: "scalar_helper_boundary_call_chain_helper.sm",
                inline_fixture: "scalar_helper_boundary_call_chain_inline.sm",
                helper_source: include_str!(
                    "../tests/fixtures/profiling/scalar_movement/g2/scalar_helper_boundary_call_chain_helper.sm"
                ),
                inline_source: include_str!(
                    "../tests/fixtures/profiling/scalar_movement/g2/scalar_helper_boundary_call_chain_inline.sm"
                ),
                semantic_locals: &["checksum", "chain_hits", "score"],
            },
        ];

        fn assert_helper_boundary_pair_equivalence(pair: &HelperBoundaryPair) {
            let helper = observe_verified_entry_semcode(pair.helper_fixture, pair.helper_source);
            let inline = observe_verified_entry_semcode(pair.inline_fixture, pair.inline_source);
            let helper_observable = helper.observable.as_deref().map(|observable| {
                canonicalize_terminal_observable(observable, pair.semantic_locals)
            });
            let inline_observable = inline.observable.as_deref().map(|observable| {
                canonicalize_terminal_observable(observable, pair.semantic_locals)
            });

            println!("pair: {}", pair.name);
            println!(
                "  helper: status={:?} observable={:?}",
                helper.status, helper.observable
            );
            println!(
                "  inline:  status={:?} observable={:?}",
                inline.status, inline.observable
            );

            assert_eq!(
                helper.status,
                VmTestStatus::Completed,
                "{} helper run did not complete",
                pair.name
            );
            assert_eq!(
                inline.status,
                VmTestStatus::Completed,
                "{} inline run did not complete",
                pair.name
            );
            assert!(helper.trap.is_none(), "{} helper run trapped", pair.name);
            assert!(inline.trap.is_none(), "{} inline run trapped", pair.name);
            assert_eq!(
                helper_observable, inline_observable,
                "{} helper and inline observations diverged",
                pair.name
            );
        }

        #[test]
        fn helper_boundary_pair_equivalence_harness_matches_terminal_observations() {
            for pair in HELPER_BOUNDARY_PAIRS {
                assert_helper_boundary_pair_equivalence(pair);
            }
        }
    }

    #[test]
    fn vm_runs_empty_main() {
        let src = "fn main() { return; }";
        let bytes = compile_program_to_semcode(src).expect("compile");
        run_semcode(&bytes).expect("run");
    }

    #[test]
    fn run_verified_entry_semcode_executes_main_token() {
        let src = "fn main() { return; }";
        let bytes = compile_program_to_semcode(src).expect("compile");
        let token = sm_verify::verify_semcode_token(&bytes).expect("verify");
        let entry_token = token.require_entry("main").expect("require entry");
        run_verified_entry_semcode(&entry_token).expect("token execution");
    }

    #[test]
    fn run_verified_entry_semcode_executes_helper_token() {
        let src = r#"
            fn helper() { return; }
            fn main() { return; }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let token = sm_verify::verify_semcode_token(&bytes).expect("verify");
        let entry_token = token.require_entry("helper").expect("require entry");
        run_verified_entry_semcode(&entry_token).expect("token execution");
    }

    #[test]
    fn run_verified_entry_semcode_with_config_matches_old_verified_path() {
        let src = "fn main() { return; }";
        let bytes = compile_program_to_semcode(src).expect("compile");
        let token = sm_verify::verify_semcode_token(&bytes).expect("verify");
        let entry_token = token.require_entry("main").expect("require entry");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_stack_depth = 1;
        run_verified_entry_semcode_with_config(&entry_token, config).expect("run");
    }

    // #1757 (FA-07-017): builds "fn main() { let a: bool = true; return; }"
    // and patches LOAD_BOOL's dst register operand to r5000 - the same
    // fixture shape as sm-verify's own r5000 regression tests. LOAD_BOOL is
    // the function's first instruction, so its absolute byte offset is just
    // the function's code-slice start (recovered via pointer arithmetic,
    // since `code_slice` is a genuine subslice of `bytes`) plus
    // `instr_start_offset`.
    //
    // #1770 follow-up: `let a: bool = true;` also emits a STORE_VAR
    // immediately after LOAD_BOOL, reading the same register LOAD_BOOL
    // wrote (`src`), to bind it to local `a`. Relocating only LOAD_BOOL's
    // `dst` to r5000 without also relocating STORE_VAR's `src` left the
    // original register (r0) referenced-but-never-written -- which used to
    // read back as a fail-open `Value::Unit` (the exact #1770 bug) and
    // silently worked by accident. Now that undefined reads are correctly
    // rejected, both operands of this load/store pair must be relocated
    // together so the artifact stays internally consistent; the r5000
    // register-budget property under test is otherwise unaffected.
    fn build_r5000_register_artifact() -> Vec<u8> {
        let mut bytes = compile_program_to_semcode("fn main() { let a: bool = true; return; }")
            .expect("compile");
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode semcode");
        let env = functions
            .iter()
            .find(|f| f.name == "main")
            .expect("main function present");
        let code_start = env.code_slice.as_ptr() as usize - bytes.as_ptr() as usize;
        let opcode_pos = code_start + env.instr_start_offset;
        assert_eq!(
            bytes[opcode_pos],
            Opcode::LoadBool as u8,
            "expected LOAD_BOOL as main's first instruction"
        );
        let dst_pos = opcode_pos + 1;
        bytes[dst_pos..dst_pos + 2].copy_from_slice(&5000u16.to_le_bytes());
        // LOAD_BOOL is [opcode:1][dst:u16][val:u8] = 4 bytes; STORE_VAR
        // follows immediately as [opcode:1][sid:u16][src:u16].
        let store_var_pos = opcode_pos + 4;
        assert_eq!(
            bytes[store_var_pos],
            Opcode::StoreVar as u8,
            "expected STORE_VAR immediately after LOAD_BOOL"
        );
        let store_var_src_pos = store_var_pos + 1 + 2;
        bytes[store_var_src_pos..store_var_src_pos + 2].copy_from_slice(&5000u16.to_le_bytes());
        bytes
    }

    // Config-aware byte shim, KernelBound config: admission must now reach
    // the shim's caller-supplied quotas, so r5000 (within KernelBound's 8192
    // register budget) must no longer be rejected by the old hardcoded
    // VerifiedLocal (4096) admission profile.
    #[test]
    fn run_verified_semcode_with_config_admits_r5000_under_kernel_bound_config() {
        let bytes = build_r5000_register_artifact();
        let config = ExecutionConfig::for_context(ExecutionContext::KernelBound);
        run_verified_semcode_with_config(&bytes, config)
            .expect("KernelBound config must admit and run an r5000 artifact");
    }

    // Same r5000 bytes, VerifiedLocal config: still rejected, proving the
    // canonical stricter path is unaffected by the fix.
    #[test]
    fn run_verified_semcode_with_config_still_rejects_r5000_under_verified_local_config() {
        let bytes = build_r5000_register_artifact();
        let config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        let err = run_verified_semcode_with_config(&bytes, config)
            .expect_err("VerifiedLocal config must still reject an r5000 artifact");
        assert!(matches!(err, RuntimeError::VerifierRejected(_)));
    }

    // #1757 follow-up (found by adversarial review): the no-`_and_config`
    // sibling `run_verified_semcode_with_host_and_capabilities` hardcodes
    // KernelBound execution but was still admitting via the old
    // hardcoded-VerifiedLocal `verify_semcode_token`. Fixed to admit under
    // the same KernelBound quotas it executes with. r5000 (within
    // KernelBound's 8192 register budget, over VerifiedLocal's 4096) must
    // now be admitted and run instead of rejected at admission.
    #[test]
    fn run_verified_semcode_with_host_and_capabilities_admits_r5000_under_kernel_bound() {
        let bytes = build_r5000_register_artifact();
        let manifest = prom_cap::CapabilityManifest::new();
        let mut host = prom_abi::RecordingHostAbi::default();
        run_verified_semcode_with_host_and_capabilities(&bytes, &mut host, &manifest)
            .expect("KernelBound-hardcoded shim must admit and run an r5000 artifact");
    }

    #[test]
    fn run_verified_semcode_missing_main_keeps_old_error() {
        let mut bytes = compile_program_to_semcode("fn main() { return; }").expect("compile");
        let target = b"main";
        for i in 0..bytes.len() - target.len() {
            if &bytes[i..i + target.len()] == target {
                bytes[i..i + target.len()].copy_from_slice(b"help");
                break;
            }
        }
        let err = run_verified_semcode(&bytes).expect_err("must fail missing main");
        assert!(matches!(err, RuntimeError::UnknownFunction(func) if func == "main"));
    }

    #[test]
    fn vm_runs_assert_statement_when_condition_holds() {
        let src = r#"
            fn main() {
                assert(true);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        run_semcode(&bytes).expect("assert(true) should pass");
    }

    #[test]
    fn vm_traps_on_failed_assert() {
        let src = r#"
            fn main() {
                assert(false);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let err = run_semcode(&bytes).expect_err("assert(false) should trap");
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::AssertionFailed)
        ));
    }

    #[test]
    fn vm_runs_function_requires_clause_when_condition_holds() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn decide(ctx: DecisionContext, expected: quad) -> quad
                requires(ctx.camera == expected)
                requires(ctx.quality == 0.75) {
                return ctx.camera;
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext { camera: T, quality: 0.75 };
                let seen: quad = decide(ctx, T);
                assert(seen == T);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        run_semcode(&bytes).expect("requires clause should pass");
    }

    #[test]
    fn vm_traps_on_failed_function_requires_clause() {
        let src = r#"
            fn must_be_true(flag: bool) -> bool requires(flag == true) {
                return flag;
            }

            fn main() {
                let seen: bool = must_be_true(false);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let err = run_semcode(&bytes).expect_err("requires clause should trap");
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::AssertionFailed)
        ));
    }

    #[test]
    fn vm_runs_function_ensures_clause_when_condition_holds() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn decide(ctx: DecisionContext) -> quad
                ensures(result == ctx.camera)
                ensures(ctx.quality == 0.75) {
                return ctx.camera;
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext { camera: T, quality: 0.75 };
                let seen: quad = decide(ctx);
                assert(seen == T);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        run_semcode(&bytes).expect("ensures clause should pass");
    }

    #[test]
    fn vm_traps_on_failed_function_ensures_clause() {
        let src = r#"
            fn must_return_true(flag: bool) -> bool ensures(result == true) {
                return flag;
            }

            fn main() {
                let seen: bool = must_return_true(false);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let err = run_semcode(&bytes).expect_err("ensures clause should trap");
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::AssertionFailed)
        ));
    }

    #[test]
    fn vm_runs_function_invariant_clauses_when_conditions_hold() {
        let src = r#"
            fn keep(flag: bool) -> bool
                invariant(flag == true)
                invariant(result == flag) {
                return flag;
            }

            fn main() {
                let seen: bool = keep(true);
                assert(seen == true);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        run_semcode(&bytes).expect("invariant clauses should pass");
    }

    #[test]
    fn vm_traps_on_failed_function_invariant_clause() {
        let src = r#"
            fn must_stay_true(flag: bool) -> bool invariant(result == true) {
                return flag;
            }

            fn main() {
                let seen: bool = must_stay_true(false);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let err = run_semcode(&bytes).expect_err("invariant clause should trap");
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::AssertionFailed)
        ));
    }

    #[test]
    fn vm_runs_bool_ops() {
        let src = r#"
			fn main() {
				let a: bool = true;
				let b: bool = false;
				let c = a && b;
				if c == false { return; } else { return; }
			}
		"#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        run_semcode(&bytes).expect("run");
    }

    #[test]
    fn vm_runs_quad_ops() {
        let src = r#"
			fn main() {
				let a: quad = T;
				let b: quad = S;
				let c = a && b;
				if c == T { return; } else { return; }
			}
		"#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        run_semcode(&bytes).expect("run");
    }

    #[test]
    fn vm_runs_call_ret() {
        let src = r#"
			fn one() -> i32 { return 1; }
			fn main() { let x: i32 = one(); return; }
		"#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_fx_literal_call_and_compare_path() {
        let src = r#"
            fn id(x: fx) -> fx { return x; }

            fn make() -> fx {
                return -1.25;
            }

            fn main() {
                let x: fx = 1.25;
                let y: fx = id(2);
                let z: fx = make();
                let a = x == x;
                let b = y != z;
                if a == b { return; } else { return; }
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("LOAD_FX"));
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_plain_fx_arithmetic_path() {
        let src = r#"
            fn main() {
                let a: fx = 2.5;
                let b: fx = 1.5;
                let sum: fx = a + b;
                let diff: fx = a - b;
                let prod: fx = a * b;
                let quo: fx = a / b;
                let neg: fx = -a;
                let expected_sum: fx = 4.0;
                let expected_diff: fx = 1.0;
                let expected_prod: fx = 3.75;
                let expected_quo: fx = 1.666;
                let expected_neg: fx = -2.5;
                assert(sum == expected_sum);
                assert(diff == expected_diff);
                assert(prod == expected_prod);
                assert(quo == expected_quo);
                assert(neg == expected_neg);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("ADD_FX"));
        assert!(disasm.contains("SUB_FX"));
        assert!(disasm.contains("MUL_FX"));
        assert!(disasm.contains("DIV_FX"));
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_text_literal_and_equality_path() {
        let src = r#"
            fn echo(x: text) -> text { return x; }

            fn main() {
                let a: text = "alpha";
                let b: text = echo("alpha");
                assert(a == b);
                assert(a != "beta");
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("LOAD_TEXT"));
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_sequence_literal_index_and_equality_path() {
        let src = r#"
            fn main() {
                let values: Sequence(i32) = [1, 2, 3];
                let head: i32 = values[0];
                assert(head == 1);
                assert(values == [1, 2, 3]);
                assert(values != [1, 2, 4]);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("MAKE_SEQUENCE"));
        assert!(disasm.contains("SEQUENCE_GET"));
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_first_class_closure_direct_invocation_path() {
        let src = r#"
            fn main() {
                let offset: f64 = 1.0;
                let add: Closure(f64 -> f64) = (x => x + offset);
                let total: f64 = add(2.0);
                assert(total == 3.0);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("MAKE_CLOSURE"));
        assert!(disasm.contains("CLOSURE_CALL"));
        run_semcode(&bytes).expect("run");
    }

    #[test]
    fn vm_traps_on_fx_division_by_zero() {
        let src = r#"
            fn main() {
                let a: fx = 1.0;
                let b: fx = 0.0;
                let bad: fx = a / b;
                assert(bad == a);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let err = run_semcode(&bytes).expect_err("fx division by zero should trap");
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::DivisionByZero)
        ));
    }

    // SSF-07 (issue #1578) freezes i32's existing overflow contract: AddI32/
    // SubI32/MulI32 wrap silently (wrapping_add/sub/mul), while DivI32/ModI32
    // trap on the i32::MIN/-1 overflow edge case and on division/modulo by
    // zero (checked_div/checked_rem, i32_div_raw/i32_mod_raw). This behavior
    // already exists; these tests turn implicit behavior into a guarded
    // contract that would fail loudly if a future change accidentally
    // switched wrapping to panicking (or vice versa). i32::MIN cannot be
    // written as a literal directly (parse_i32_literal parses unsigned digit
    // text before unary negation applies), hence the `0 - 2147483647 - 1`
    // construction throughout.
    // The i32 overflow contract has two independent implementations of the
    // same wrap/trap policy: the VM's runtime opcodes (semcode_vm.rs) used at
    // O0, and crystalfold.rs's O1 constant-folding rewrites, which duplicate
    // the same wrapping_*/checked_* calls at compile time. Nothing forces
    // those two to stay in sync, so every case below runs under both O0 and
    // O1 rather than only the O0 default `compile_program_to_semcode` uses.
    fn assert_wraps_under_all_opt_levels(src: &str, msg: &str) {
        for opt in [OptLevel::O0, OptLevel::O1] {
            let bytes = compile_program_to_semcode_with_options(src, CompileProfile::RustLike, opt)
                .unwrap_or_else(|e| panic!("compile failed under {opt:?}: {e:?}"));
            run_semcode(&bytes).unwrap_or_else(|e| panic!("{msg} (opt={opt:?}): {e:?}"));
        }
    }

    fn assert_traps_under_all_opt_levels(src: &str, msg: &str, expected: RuntimeTrap) {
        for opt in [OptLevel::O0, OptLevel::O1] {
            let bytes = compile_program_to_semcode_with_options(src, CompileProfile::RustLike, opt)
                .unwrap_or_else(|e| panic!("compile failed under {opt:?}: {e:?}"));
            let err = run_semcode(&bytes)
                .err()
                .unwrap_or_else(|| panic!("{msg} (opt={opt:?}) should trap"));
            assert!(
                matches!(&err, RuntimeError::Trap(t) if *t == expected),
                "{msg} (opt={opt:?}): expected trap {expected:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn vm_wraps_i32_addition_past_max() {
        let src = r#"
            fn main() {
                let max_val: i32 = 2147483647;
                let one: i32 = 1;
                let wrapped: i32 = max_val + one;
                let min_val: i32 = 0 - 2147483647 - 1;
                assert(wrapped == min_val);
                return;
            }
        "#;
        assert_wraps_under_all_opt_levels(src, "i32 addition must wrap past i32::MAX, not trap");
    }

    #[test]
    fn vm_wraps_i32_subtraction_past_min() {
        let src = r#"
            fn main() {
                let min_val: i32 = 0 - 2147483647 - 1;
                let one: i32 = 1;
                let wrapped: i32 = min_val - one;
                let max_val: i32 = 2147483647;
                assert(wrapped == max_val);
                return;
            }
        "#;
        assert_wraps_under_all_opt_levels(src, "i32 subtraction must wrap past i32::MIN, not trap");
    }

    #[test]
    fn vm_wraps_i32_multiplication_past_max() {
        let src = r#"
            fn main() {
                let max_val: i32 = 2147483647;
                let two: i32 = 2;
                let wrapped: i32 = max_val * two;
                let expected: i32 = 0 - 2;
                assert(wrapped == expected);
                return;
            }
        "#;
        assert_wraps_under_all_opt_levels(
            src,
            "i32 multiplication must wrap per two's-complement, not trap",
        );
    }

    #[test]
    fn vm_wraps_i32_unary_negation_of_min() {
        let src = r#"
            fn main() {
                let min_val: i32 = 0 - 2147483647 - 1;
                let negated: i32 = -min_val;
                assert(negated == min_val);
                return;
            }
        "#;
        assert_wraps_under_all_opt_levels(
            src,
            "unary negation of i32::MIN lowers through SubI32 and must wrap, not trap",
        );
    }

    #[test]
    fn vm_traps_on_i32_division_min_by_negative_one() {
        let src = r#"
            fn main() {
                let min_val: i32 = 0 - 2147483647 - 1;
                let neg_one: i32 = 0 - 1;
                let bad: i32 = min_val / neg_one;
                assert(bad == min_val);
                return;
            }
        "#;
        assert_traps_under_all_opt_levels(
            src,
            "i32::MIN / -1 should trap with overflow, not wrap",
            RuntimeTrap::ArithmeticOverflow,
        );
    }

    #[test]
    fn vm_traps_on_i32_modulo_min_by_negative_one() {
        let src = r#"
            fn main() {
                let min_val: i32 = 0 - 2147483647 - 1;
                let neg_one: i32 = 0 - 1;
                let bad: i32 = min_val % neg_one;
                assert(bad == min_val);
                return;
            }
        "#;
        assert_traps_under_all_opt_levels(
            src,
            "i32::MIN % -1 should trap with overflow, not wrap",
            RuntimeTrap::ArithmeticOverflow,
        );
    }

    #[test]
    fn vm_traps_on_i32_division_by_zero() {
        let src = r#"
            fn main() {
                let a: i32 = 10;
                let b: i32 = 0;
                let bad: i32 = a / b;
                assert(bad == a);
                return;
            }
        "#;
        assert_traps_under_all_opt_levels(
            src,
            "i32 division by zero should trap",
            RuntimeTrap::DivisionByZero,
        );
    }

    #[test]
    fn vm_traps_on_i32_modulo_by_zero() {
        let src = r#"
            fn main() {
                let a: i32 = 10;
                let b: i32 = 0;
                let bad: i32 = a % b;
                assert(bad == a);
                return;
            }
        "#;
        assert_traps_under_all_opt_levels(
            src,
            "i32 modulo by zero should trap",
            RuntimeTrap::DivisionByZero,
        );
    }

    #[test]
    fn vm_evaluates_named_call_expression_arguments_in_source_order() {
        // FA-04-016 / #1722, real pipeline proof: `add(b = ten / zero,
        // a = min_val % neg_one)` writes `b`'s expression first in source
        // even though `a` is the declared first parameter. `b`'s expression
        // traps with DivisionByZero; `a`'s expression traps with
        // ArithmeticOverflow instead. Only one of the two instructions ever
        // executes (the VM halts on the first trap), so which trap kind
        // surfaces is a direct, deterministic witness of evaluation order:
        // source order traps DivisionByZero, parameter order would trap
        // ArithmeticOverflow.
        let src = r#"
            fn add(a: i32, b: i32) -> i32 {
                return a + b;
            }

            fn main() {
                let ten: i32 = 10;
                let zero: i32 = 0;
                let min_val: i32 = 0 - 2147483647 - 1;
                let neg_one: i32 = 0 - 1;
                let x: i32 = add(b = ten / zero, a = min_val % neg_one);
                assert(x == 0);
                return;
            }
        "#;
        assert_traps_under_all_opt_levels(
            src,
            "named call-expression argument 'b' (division by zero) is written before 'a' (overflow) in source and must trap first",
            RuntimeTrap::DivisionByZero,
        );
    }

    #[test]
    fn vm_evaluates_named_call_statement_arguments_in_source_order() {
        // FA-04-016 / #1722, real pipeline proof for the independent
        // call-statement lowering path: same distinguishing-trap construction
        // as vm_evaluates_named_call_expression_arguments_in_source_order,
        // but with a Unit-returning function invoked as a bare statement.
        let src = r#"
            fn take(a: i32, b: i32) {
                return;
            }

            fn main() {
                let ten: i32 = 10;
                let zero: i32 = 0;
                let min_val: i32 = 0 - 2147483647 - 1;
                let neg_one: i32 = 0 - 1;
                take(b = ten / zero, a = min_val % neg_one);
                return;
            }
        "#;
        assert_traps_under_all_opt_levels(
            src,
            "named call-statement argument 'b' (division by zero) is written before 'a' (overflow) in source and must trap first",
            RuntimeTrap::DivisionByZero,
        );
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_u32_literal_compare_path() {
        let src = r#"
            fn main() {
                let left: u32 = 1_000u32;
                let right: u32 = 0x3e8u32;
                assert(left == right);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("LOAD_U32"));
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_tuple_return_and_equality_path() {
        let src = r#"
            fn pair(flag: bool) -> (i32, bool) {
                return (1, flag);
            }

            fn main() {
                let left: (i32, bool) = pair(true);
                let right: (i32, bool) = (1, true);
                assert(left == right);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("MAKE_TUPLE"));
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_tuple_destructuring_bind_path() {
        let src = r#"
            fn pair(flag: bool) -> (i32, bool) = (1, flag);

            fn main() {
                let (count, ready): (i32, bool) = pair(true);
                assert(count == 1);
                assert(ready == true);
                return;
            }
        "#;

        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("TUPLE_GET"));
        run_semcode(&bytes).expect("run");
    }

    #[test]
    fn vm_tracks_borrowed_paths_on_frame_push() {
        let bytes = ownership_tracking_bytes();
        let program = parse_semcode(&bytes).expect("parse");
        let mut vm = VM {
            functions: program.functions,
            callstack: Vec::new(),
            config: ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
            effect_calls: 0,
            steps: 0,
            calls: 0,
            symbols: program.runtime_symbols,
            prng_state: 0,
        };

        push_frame(&mut vm, "main", Vec::new(), None).expect("push frame");

        // #1726 Checkpoint D3: `ownership_tracking_bytes()` compiles a real
        // tuple `ref` destructure, which is a `StoreVarSite` (not
        // `FrameEntry`) Borrow since D2a - so at push time, before its
        // annotated `StoreVar` has executed, this path is pending, not
        // active. This test proves the decoded path/anchor data flows
        // correctly into pending state; `tests/borrow_activation_v20.rs`
        // proves the later activation step itself end to end (a real
        // conflicting write only rejects once the introduction has run).
        assert_eq!(vm.callstack.len(), 1);
        let frame = &vm.callstack[0];
        assert!(frame.active_borrowed_paths.is_empty());
        assert_eq!(frame.pending_site_borrows.len(), 1);
        assert_eq!(
            frame.pending_site_borrows[0].path.components,
            vec![PathComponent::TupleIndex(0)]
        );
        // #1725 (FA-04-019): before this fix, `root_symbol_id` was a raw
        // frontend SymbolId, out of bounds for this function's own string
        // table, so the (now-removed) fail-open fallback treated it as an
        // already-valid *global* runtime SymbolId - which happened to
        // resolve to "pair" only by coincidence (the call `pair(true)`
        // interns that exact raw string as a Call target elsewhere in the
        // artifact). The correct identity is the local binding's own
        // lowered key, not that unrelated string.
        assert_eq!(
            vm.symbols.resolve(frame.pending_site_borrows[0].path.root),
            Some("__sm_local_0_pair")
        );
    }

    #[test]
    fn vm_clears_borrowed_paths_on_frame_exit() {
        let bytes = helper_borrow_bytes();
        let program = parse_semcode(&bytes).expect("parse");
        let mut vm = VM {
            functions: program.functions,
            callstack: Vec::new(),
            config: ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
            effect_calls: 0,
            steps: 0,
            calls: 0,
            symbols: program.runtime_symbols,
            prng_state: 0,
        };

        push_frame(&mut vm, "main", Vec::new(), None).expect("push main");
        push_frame(
            &mut vm,
            "helper",
            vec![Value::Tuple(vec![Value::I32(1), Value::Bool(true)])],
            None,
        )
        .expect("push helper");

        // #1726 Checkpoint D3: `helper`'s tuple `ref` destructure is a
        // `StoreVarSite` Borrow, pending (not active) before its `StoreVar`
        // executes - this test never runs an instruction, so it stays
        // pending throughout, and frame exit must clear it exactly like it
        // used to clear the (pre-D3) eager active set.
        assert_eq!(vm.callstack.len(), 2);
        assert_eq!(vm.callstack[1].pending_site_borrows.len(), 1);
        assert!(vm.callstack[1].active_borrowed_paths.is_empty());

        let finished = vm.callstack.pop().expect("helper frame");
        assert_eq!(finished.pending_site_borrows.len(), 1);
        assert_eq!(vm.callstack.len(), 1);
        assert!(vm.callstack[0].pending_site_borrows.is_empty());
        assert!(vm.callstack[0].active_borrowed_paths.is_empty());
    }

    #[test]
    fn vm_tracks_record_field_borrowed_paths_on_frame_push() {
        let bytes = record_field_borrow_tracking_bytes();
        let program = parse_semcode(&bytes).expect("parse");
        let mut vm = VM {
            functions: program.functions,
            callstack: Vec::new(),
            config: ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
            effect_calls: 0,
            steps: 0,
            calls: 0,
            symbols: program.runtime_symbols,
            prng_state: 0,
        };

        push_frame(&mut vm, "main", Vec::new(), None).expect("push frame");

        // #1726 Checkpoint D3: same reasoning as
        // `vm_tracks_borrowed_paths_on_frame_push` above, for the record
        // field producer.
        assert_eq!(vm.callstack.len(), 1);
        let frame = &vm.callstack[0];
        assert!(frame.active_borrowed_paths.is_empty());
        assert_eq!(frame.pending_site_borrows.len(), 1);
        assert!(matches!(
            frame.pending_site_borrows[0].path.components.as_slice(),
            [PathComponent::Field(_)]
        ));
    }

    #[test]
    fn vm_clears_record_field_borrowed_paths_on_frame_exit() {
        let bytes = helper_record_field_borrow_bytes();
        let program = parse_semcode(&bytes).expect("parse");
        let mut vm = VM {
            functions: program.functions,
            callstack: Vec::new(),
            config: ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
            effect_calls: 0,
            steps: 0,
            calls: 0,
            symbols: program.runtime_symbols,
            prng_state: 0,
        };

        push_frame(&mut vm, "main", Vec::new(), None).expect("push main");
        push_frame(
            &mut vm,
            "helper",
            vec![Value::Record(RecordCarrier {
                type_name: "DecisionContext".to_string(),
                slots: vec![Value::Quad(QuadVal::T), Value::F64(0.75)],
            })],
            None,
        )
        .expect("push helper");

        assert_eq!(vm.callstack.len(), 2);
        assert_eq!(vm.callstack[1].pending_site_borrows.len(), 1);
        assert!(matches!(
            vm.callstack[1].pending_site_borrows[0]
                .path
                .components
                .as_slice(),
            [PathComponent::Field(_)]
        ));

        let finished = vm.callstack.pop().expect("helper frame");
        assert_eq!(finished.pending_site_borrows.len(), 1);
        assert_eq!(vm.callstack.len(), 1);
        assert!(vm.callstack[0].pending_site_borrows.is_empty());
        assert!(vm.callstack[0].active_borrowed_paths.is_empty());
    }

    #[test]
    fn vm_rejects_write_after_borrow_same_path() {
        let bytes = ownership_write_overlap_bytes(&[0], &[0]);
        let err = run_semcode(&bytes).expect_err("overlapping write must fail");
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict)
        ));
        assert_eq!(format!("{err}"), "write path overlaps active borrow");
    }

    #[test]
    fn vm_rejects_write_when_borrowed_parent_overlaps_child_path() {
        let bytes = ownership_write_overlap_bytes(&[], &[0]);
        let err = run_semcode(&bytes).expect_err("parent-child overlap must fail");
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict)
        ));
        assert_eq!(format!("{err}"), "write path overlaps active borrow");
    }

    #[test]
    fn vm_rejects_write_when_borrowed_child_overlaps_parent_path() {
        let bytes = ownership_write_overlap_bytes(&[0], &[]);
        let err = run_semcode(&bytes).expect_err("child-parent overlap must fail");
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict)
        ));
        assert_eq!(format!("{err}"), "write path overlaps active borrow");
    }

    #[test]
    fn vm_allows_write_to_sibling_path_with_active_borrow() {
        let bytes = ownership_write_overlap_bytes(&[0], &[1]);
        run_semcode(&bytes).expect("sibling write must stay allowed");
    }

    #[test]
    fn vm_rejects_record_field_write_after_borrow_same_field() {
        let bytes = record_field_write_overlap_bytes(Some("camera"), Some("camera"));
        let err = run_semcode(&bytes).expect_err("same-field record write must fail");
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict)
        ));
        assert_eq!(format!("{err}"), "write path overlaps active borrow");
    }

    #[test]
    fn vm_rejects_record_field_write_when_borrowed_parent_overlaps_child_field() {
        let bytes = record_field_write_overlap_bytes(None, Some("camera"));
        let err = run_semcode(&bytes).expect_err("record parent-child overlap must fail");
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict)
        ));
        assert_eq!(format!("{err}"), "write path overlaps active borrow");
    }

    #[test]
    fn vm_rejects_record_parent_write_when_borrowed_child_field() {
        let bytes = record_field_write_overlap_bytes(Some("camera"), None);
        let err = run_semcode(&bytes).expect_err("record child-parent overlap must fail");
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict)
        ));
        assert_eq!(format!("{err}"), "write path overlaps active borrow");
    }

    #[test]
    fn vm_allows_record_field_write_to_sibling_field_with_active_borrow() {
        let bytes = record_field_write_overlap_bytes(Some("camera"), Some("quality"));
        run_semcode(&bytes).expect("sibling record field write must stay allowed");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_stage1_record_literal_path() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext { quality: 0.75, camera: T };
                let shadow: DecisionContext = ctx;
                let _ = shadow;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("MAKE_RECORD"));
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_stage1_enum_constructor_path() {
        let src = r#"
            enum Maybe {
                None,
                Some(bool),
            }

            fn choose(flag: bool) -> Maybe {
                return Maybe::Some(flag);
            }

            fn main() {
                let left: Maybe = choose(true);
                let right: Maybe = Maybe::None;
                let _ = left;
                let _ = right;
                return;
            }
        "#;

        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("MAKE_ADT"));
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_option_and_result_standard_form_paths() {
        let src = r#"
            fn keep(flag: bool) -> Option(bool) {
                let fallback: Option(bool) = Option::None;
                let _ = fallback;
                return Option::Some(flag);
            }

            fn settle(flag: bool) -> Result(bool, quad) {
                if flag {
                    let value: Result(bool, quad) = Result::Ok(true);
                    return value;
                }
                let value: Result(bool, quad) = Result::Err(N);
                return value;
            }

            fn main() {
                let left: Option(bool) = keep(true);
                let right: Result(bool, quad) = settle(false);
                let _ = left;
                let _ = right;
                return;
            }
        "#;

        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("MAKE_ADT"));
        run_semcode(&bytes).expect("Option/Result standard-form paths should run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_option_and_result_match_ergonomics_paths() {
        let src = r#"
            fn unwrap(opt: Option(bool)) -> bool {
                let out: bool = match opt {
                    Option::Some(value) => { value }
                    Option::None => { false }
                };
                return out;
            }

            fn settle(res: Result(quad, quad)) -> quad {
                let out: quad = match res {
                    Result::Ok(value) => { value }
                    Result::Err(code) => { code }
                };
                return out;
            }

            fn main() {
                let left: bool = unwrap(Option::Some(true));
                let right: bool = unwrap(Option::None);
                let code: quad = settle(Result::Err(S));
                assert(left == true);
                assert(right == false);
                assert(code == S);
                return;
            }
        "#;

        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("ADT_TAG"));
        assert!(disasm.contains("ADT_GET"));
        run_semcode(&bytes).expect("Option/Result match ergonomics paths should run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_stage1_adt_match_path() {
        let src = r#"
            enum Maybe {
                None,
                Some(f64),
            }

            fn unwrap(value: Maybe) -> f64 {
                let total: f64 = match value {
                    Maybe::Some(inner) => { inner }
                    _ => { 0.0 }
                };
                return total;
            }

            fn main() {
                let total: f64 = unwrap(Maybe::Some(2.5));
                assert(total == 2.5);
                return;
            }
        "#;

        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("ADT_TAG"));
        assert!(disasm.contains("ADT_GET"));
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_exhaustive_adt_match_without_default_path() {
        let src = r#"
            enum Maybe {
                None,
                Some(f64),
            }

            fn unwrap(value: Maybe) -> f64 {
                let total: f64 = match value {
                    Maybe::None => { 0.0 }
                    Maybe::Some(inner) => { inner }
                };
                return total;
            }

            fn main() {
                let total: f64 = unwrap(Maybe::Some(2.5));
                assert(total == 2.5);
                return;
            }
        "#;

        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("ADT_TAG"));
        assert!(disasm.contains("ASSERT"));
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_adt_payload_ownership_positive_e2e_path() {
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
                return;
            }
        "#;

        let bytes = compile_program_to_semcode(src).expect("compile");
        let (_, envs) = sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let mut found_adt_payload = false;
        for env in &envs {
            for path in env.borrowed_paths.iter().chain(env.write_paths.iter()) {
                for component in &path.components {
                    if let sm_format::semcode_decode::DecodedAccessPathComponent::AdtPayload {
                        ..
                    } = component
                    {
                        found_adt_payload = true;
                    }
                }
            }
        }
        assert!(
            found_adt_payload,
            "Expected AdtPayload ownership component to be emitted in SemCode"
        );

        run_semcode(&bytes).expect("run");
    }

    // NOTE(ADT-4): A negative E2E test cannot be expressed cleanly right now because the Semantic frontend
    // does not yet support mutable bindings or mutable re-assignments of ADT payloads (e.g. `inner += 1.0` or mutating the parent `value` while `inner` is borrowed).
    // Therefore, we rely on the runtime-patched tests added in ADT-3 (like `vm_rejects_adt_payload_write_when_borrowed_same_payload`)
    // to prove the VM borrow-checker correctness, and we will add the negative E2E test once the language surface is ready.

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_stage1_record_field_access_path() {
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
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("RECORD_GET"));
        run_semcode(&bytes).expect("run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_record_pass_return_and_safe_equality_path() {
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
        run_semcode(&bytes).expect("record pass/return/equality should run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_record_access_policy_scenario() {
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
        run_semcode(&bytes).expect("record access-policy scenario should run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_record_destructuring_bind_path() {
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
        let disasm = disasm_semcode(&bytes).expect("disassemble");
        assert!(disasm.contains("RECORD_GET"));
        run_semcode(&bytes).expect("record destructuring bind path should run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_record_let_else_path() {
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
        let disasm = disasm_semcode(&bytes).expect("disassemble");
        assert!(disasm.contains("RECORD_GET"));
        run_semcode(&bytes).expect("record let-else path should run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_record_copy_with_path() {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext { camera: T, quality: 0.75 };
                let patched: DecisionContext = ctx with { quality: 1.0 };
                assert(patched.camera == T);
                assert(patched.quality == 1.0);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disassemble");
        assert!(disasm.contains("RECORD_GET"));
        assert!(disasm.contains("MAKE_RECORD"));
        run_semcode(&bytes).expect("record copy-with path should run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_record_stage2_ergonomics_scenario() {
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
                let DecisionContext { camera, quality: _ } = ctx;
                let DecisionContext { camera: T, override_state, quality } =
                    patched else return;
                assert(camera == T);
                assert(override_state == N);
                assert(quality == 0.75);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disassemble");
        assert!(disasm.contains("RECORD_GET"));
        assert!(disasm.contains("MAKE_RECORD"));
        run_semcode(&bytes).expect("record stage-2 ergonomics scenario should run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_for_range_inclusive_path() {
        let src = r#"
            fn main() {
                let saw_start: bool = false;
                let saw_end: bool = false;
                for i in 0..=2 {
                    if i == 0 {
                        saw_start ||= true;
                    } else {
                        saw_start ||= false;
                    }
                    if i == 2 {
                        saw_end ||= true;
                    } else {
                        saw_end ||= false;
                    }
                }
                assert(saw_start == true);
                assert(saw_end == true);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("CMP_I32_LE"));
        assert!(disasm.contains("ADD_I32"));
        run_semcode(&bytes).expect("inclusive for-range should run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_for_range_empty_half_open_path() {
        let src = r#"
            fn main() {
                let visited: bool = false;
                for i in 3..3 {
                    visited ||= true;
                }
                assert(visited == false);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("CMP_I32_LT"));
        run_semcode(&bytes).expect("empty half-open for-range should skip body");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_iterable_for_over_sequence_path() {
        let src = r#"
            fn main() {
                let items: Sequence(i32) = [1, 2, 3];
                let saw_two: bool = false;
                for item in items {
                    if item == 2 {
                        saw_two ||= true;
                    }
                }
                assert(saw_two == true);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("SEQUENCE_LEN"));
        assert!(disasm.contains("SEQUENCE_GET"));
        run_semcode(&bytes).expect("Sequence(T) iterable loop should run");
    }

    #[cfg(feature = "disasm")]
    #[test]
    fn vm_runs_iterable_for_over_explicit_record_impl_path() {
        let src = r#"
            trait Iterable {
                fn next(self: Self, index: i32) -> Option(i32);
            }

            record Numbers {
                limit: i32,
            }

            impl Iterable for Numbers {
                fn next(self: Self, index: i32) -> Option(i32) {
                    let _ = self.limit;
                    if index == 0 {
                        return Option::Some(0);
                    }
                    if index == 1 {
                        return Option::Some(1);
                    }
                    if index == 2 {
                        return Option::Some(2);
                    }
                    return Option::None;
                }
            }

            fn main() {
                let numbers: Numbers = Numbers { limit: 4 };
                let saw_two: bool = false;
                for value in numbers {
                    if value == 2 {
                        saw_two ||= true;
                    }
                }
                assert(saw_two == true);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let disasm = disasm_semcode(&bytes).expect("disasm");
        assert!(disasm.contains("__impl::Iterable::Numbers::next"));
        assert!(disasm.contains("ADT_TAG"));
        assert!(disasm.contains("ADT_GET"));
        run_semcode(&bytes).expect("direct record Iterable loop should run");
    }

    #[test]
    fn vm_rejects_unknown_opcode_on_load() {
        let src = "fn main() { return; }";
        let mut bytes = compile_program_to_semcode(src).expect("compile");
        // #1773 (FA-09-005): the first opcode byte's absolute offset now
        // depends on the mandatory OWN0/SIG0 sections preceding the
        // instruction stream - located via the real decode rather than a
        // hand-counted literal, which silently drifted out of the
        // instruction stream (and even out of the function's own code
        // region) once those sections became mandatory.
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let opcode_pos = 8 + 2 + 4 + 4 + functions[0].instr_start_offset;
        bytes[opcode_pos] = 0xff;
        let err = run_semcode(&bytes).expect_err("must fail");
        assert!(matches!(err, RuntimeError::BadFormat(_)));
    }

    fn build_qtruth_test_program(opcode: u8) -> Vec<u8> {
        use sm_emit::Opcode;
        let src = "fn main() { return; }";
        let mut bytes = compile_program_to_semcode(src).expect("compile");

        let mut new_code = Vec::new();

        // 0: LoadQ r0, T
        new_code.push(Opcode::LoadQ as u8);
        new_code.extend_from_slice(&0u16.to_le_bytes());
        new_code.push(2); // T

        // 4: LoadQ r1, F
        new_code.push(Opcode::LoadQ as u8);
        new_code.extend_from_slice(&1u16.to_le_bytes());
        new_code.push(1); // F

        // 8: Opcode r2, r0, r1 (or r0 for Not)
        new_code.push(opcode);
        new_code.extend_from_slice(&2u16.to_le_bytes()); // dst = 2
        new_code.extend_from_slice(&0u16.to_le_bytes()); // lhs/src = 0 (T)
        if opcode != 0x19 {
            // QTruthNot = 0x19
            new_code.extend_from_slice(&1u16.to_le_bytes()); // rhs = 1 (F)
        }

        // 21 or 19: Ret r2
        new_code.push(Opcode::Ret as u8);
        new_code.push(1); // has_src
        new_code.extend_from_slice(&2u16.to_le_bytes());

        // #1773 (FA-09-005): the instruction stream's real start/prefix
        // length now depends on the mandatory OWN0/SIG0 sections - located
        // via the real decode rather than hand-counted literals, which
        // silently corrupted this splice (and the resulting `code_len`)
        // once those sections became mandatory. `prefix_len` covers
        // everything before the instruction stream (string table + OWN0 +
        // SIG0), which the splice below leaves untouched.
        let (_, functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let prefix_len = functions[0].instr_start_offset;
        let code_len_pos = 8 + 2 + 4;
        let opcode_pos = code_len_pos + 4 + prefix_len;
        let old_instr_len = functions[0].code_slice.len() - prefix_len;
        bytes.splice(
            opcode_pos..opcode_pos + old_instr_len,
            new_code.iter().copied(),
        );

        let new_code_len = prefix_len + new_code.len();
        bytes[code_len_pos..code_len_pos + 4].copy_from_slice(&(new_code_len as u32).to_le_bytes());

        bytes
    }

    #[test]
    fn vm_executes_qtruth_opcodes_correctly() {
        vm_test_clear_terminal_observation();

        // T map_and F == F
        let bytes_and = build_qtruth_test_program(0x17);
        run_semcode(&bytes_and).expect("must run");
        assert_eq!(
            vm_test_take_terminal_observation()
                .unwrap()
                .observable
                .unwrap(),
            "return=Quad(F); locals=[]"
        );

        // T map_or F == T
        let bytes_or = build_qtruth_test_program(0x18);
        run_semcode(&bytes_or).expect("must run");
        assert_eq!(
            vm_test_take_terminal_observation()
                .unwrap()
                .observable
                .unwrap(),
            "return=Quad(T); locals=[]"
        );

        // T map_implies F == F
        let bytes_impl = build_qtruth_test_program(0x1A);
        run_semcode(&bytes_impl).expect("must run");
        assert_eq!(
            vm_test_take_terminal_observation()
                .unwrap()
                .observable
                .unwrap(),
            "return=Quad(F); locals=[]"
        );

        // T map_not == F
        let bytes_not = build_qtruth_test_program(0x19);
        run_semcode(&bytes_not).expect("must run");
        assert_eq!(
            vm_test_take_terminal_observation()
                .unwrap()
                .observable
                .unwrap(),
            "return=Quad(F); locals=[]"
        );
    }

    #[test]
    fn vm_rejects_unsupported_bytecode_version_with_hint() {
        let src = "fn main() { return; }";
        let mut bytes = compile_program_to_semcode(src).expect("compile");
        bytes[7] = b'X';
        let err = run_semcode(&bytes).expect_err("must fail");
        match err {
            RuntimeError::UnsupportedBytecodeVersion { found, supported } => {
                // #1773 (FA-09-005): the natural compile of this program is
                // now SEMCOD19 (was SEMCODE0), so the corrupted last byte
                // produces "SEMCOD1X", not "SEMCODEX" - "SEMCOD" is the
                // stable 6-byte prefix shared by every header revision's
                // magic, single- and double-digit alike.
                assert!(found.starts_with("SEMCOD"));
                assert!(supported.contains("SEMCODE0"));
                assert!(supported.contains("SEMCODE1"));
                assert!(supported.contains("SEMCODE2"));
                assert!(supported.contains("SEMCODE3"));
                assert!(supported.contains("SEMCODE4"));
                assert!(supported.contains("SEMCODE5"));
                assert!(supported.contains("SEMCODE6"));
                assert!(supported.contains("SEMCODE7"));
                assert!(supported.contains("SEMCODE8"));
                assert!(supported.contains("SEMCODE9"));
                assert!(supported.contains("SEMCOD10"));
                assert!(supported.contains("SEMCOD11"));
                assert!(supported.contains("SEMCOD12"));
                assert!(supported.contains("SEMCOD13"));
            }
            other => panic!("expected UnsupportedBytecodeVersion, got {other:?}"),
        }
    }

    #[test]
    fn vm_enforces_configured_stack_depth() {
        let src = r#"
            fn helper() { return; }
            fn main() { helper(); return; }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_stack_depth = 1;
        let err = run_semcode_with_config(&bytes, config).expect_err("must fail");
        assert_eq!(err, RuntimeError::StackOverflow);
    }

    // #1751 (FA-07-011): proves `max_frames` is still enforced against live
    // call-stack depth at execution time (in `push_frame`) after sm-verify
    // stopped misusing the same field name as a static function-definition
    // count. A real 2-level call chain (main -> helper) must still trap once
    // it genuinely exceeds a shrunk frame budget.
    #[test]
    fn vm_enforces_configured_frame_budget() {
        let src = r#"
            fn helper() { return; }
            fn main() { helper(); return; }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_frames = 1;
        let err = run_semcode_with_config(&bytes, config).expect_err("must fail");
        assert_eq!(
            err,
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Frames,
                limit: 1,
                used: 2,
            })
        );
    }

    #[test]
    fn vm_enforces_configured_register_budget() {
        let src = r#"
            fn main() {
                let a: bool = true;
                let b: bool = false;
                let c = a && b;
                if c == false { return; } else { return; }
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_registers = 2;
        let err = run_semcode_with_config(&bytes, config).expect_err("must fail");
        assert_eq!(
            err,
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Registers,
                limit: 2,
                used: 16,
            })
        );
    }

    // --- #1759 (FA-08-001) Steps/Calls runtime enforcement ------------------
    //
    // Implements the contract frozen by PR #1899
    // (docs/roadmap/stable_foundation/ssf08_1759_steps_calls_contract_decision.md).
    // Two test tiers, per that decision's own §10/§12 reproduction design:
    //   1. direct helper-level tests against `charge_counter` itself, so the
    //      `usize::MAX` ceiling case is provable without ever executing
    //      `usize::MAX` real instructions/calls;
    //   2. real end-to-end VM tests proving the frozen Step/Call semantics,
    //      precedence, and root-exemption against actually compiled programs.

    #[test]
    fn charge_counter_admits_when_next_is_under_limit() {
        assert_eq!(
            charge_counter(4, 5, QuotaKind::Steps).expect("must admit"),
            5
        );
    }

    #[test]
    fn charge_counter_admits_when_next_exactly_equals_limit() {
        assert_eq!(
            charge_counter(4, 5, QuotaKind::Calls).expect("must admit"),
            5
        );
    }

    #[test]
    fn charge_counter_rejects_at_zero_limit() {
        let err = charge_counter(0, 0, QuotaKind::Steps).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Steps,
                limit: 0,
                used: 1,
            })
        );
    }

    #[test]
    fn charge_counter_rejects_one_past_limit() {
        let err = charge_counter(5, 5, QuotaKind::Calls).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Calls,
                limit: 5,
                used: 6,
            })
        );
    }

    /// #1759 §10: the single unrepresentable-attempted-usage case. `used` was
    /// already `usize::MAX`, so `checked_add(1)` overflows - this must fail
    /// closed via the ordinary `QuotaExceeded` channel with `used` SATURATED
    /// at `usize::MAX` (reporting-only), never a silent wraparound and never
    /// a new error variant. Exercised directly against the helper so this is
    /// provable without executing `usize::MAX` real instructions/calls.
    #[test]
    fn charge_counter_saturates_used_at_usize_max_on_overflow() {
        let err = charge_counter(usize::MAX, usize::MAX, QuotaKind::Steps).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Steps,
                limit: usize::MAX,
                used: usize::MAX,
            })
        );
    }

    /// Same overflow case as above but with a small, distinct `limit`, to
    /// prove the primitive reports the caller's real configured `limit`
    /// unchanged - saturation applies only to the unrepresentable `used`
    /// value, not to `limit`.
    #[test]
    fn charge_counter_overflow_reports_the_real_configured_limit() {
        let err = charge_counter(usize::MAX, 5, QuotaKind::Calls).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Calls,
                limit: 5,
                used: usize::MAX,
            })
        );
    }

    /// Runs `main` from freshly parsed SemCode bytes to completion or error,
    /// returning the final `VM` alongside the `Result` so tests can inspect
    /// `vm.steps`/`vm.calls` in BOTH the success and the failure path (e.g.
    /// proving a rejected call left the Calls counter unmoved).
    fn run_semcode_for_test(
        bytes: &[u8],
        config: ExecutionConfig,
    ) -> (VM, Result<(), RuntimeError>) {
        let program = parse_semcode(bytes).expect("parse must succeed for a compiled test fixture");
        let mut vm = VM {
            functions: program.functions,
            callstack: Vec::new(),
            config,
            effect_calls: 0,
            steps: 0,
            calls: 0,
            symbols: program.runtime_symbols,
            prng_state: 0,
        };
        let result = push_frame(&mut vm, "main", Vec::new(), None).and_then(|()| {
            let mut host = LegacyVmHost;
            let mut observation = HelloObservationRuntime::discard();
            exec_loop(&mut vm, &mut host, &mut observation).map(|_| ())
        });
        (vm, result)
    }

    #[test]
    fn vm_steps_zero_limit_rejects_the_first_opcode() {
        let src = r#"
            fn main() { return; }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_steps = 0;
        let (vm, result) = run_semcode_for_test(&bytes, config);
        assert_eq!(
            result.unwrap_err(),
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Steps,
                limit: 0,
                used: 1,
            })
        );
        assert_eq!(
            vm.steps, 0,
            "the rejected opcode itself must not be charged"
        );
    }

    /// Derives the real Step count of a tiny fixed program empirically (a
    /// generous quota first), then proves the frozen `used > limit`
    /// exhaustion boundary exactly: the derived count itself must complete,
    /// and one fewer must fail with `used` equal to that same derived count.
    #[test]
    fn vm_steps_exact_limit_completes_and_one_under_rejects() {
        let src = r#"
            fn main() {
                let x: i32 = 1;
                let y: i32 = x + 1;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");

        let mut generous = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        generous.quotas.max_steps = 1_000_000;
        let (vm, result) = run_semcode_for_test(&bytes, generous);
        result.expect("must complete under a generous step budget");
        let exact = vm.steps;
        assert!(
            exact > 0,
            "a non-trivial program must dispatch at least one opcode"
        );

        let mut exact_config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        exact_config.quotas.max_steps = exact;
        let (_, result) = run_semcode_for_test(&bytes, exact_config);
        result.expect("exactly enough steps must still complete");

        let mut one_under = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        one_under.quotas.max_steps = exact - 1;
        let (vm, result) = run_semcode_for_test(&bytes, one_under);
        assert_eq!(
            result.unwrap_err(),
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Steps,
                limit: exact - 1,
                used: exact,
            })
        );
        assert_eq!(
            vm.steps,
            exact - 1,
            "no partial charge past the rejected opcode"
        );
    }

    /// The reproduction case #1759's own contract decision specifies: a
    /// verified backward loop whose natural iteration count vastly exceeds
    /// the configured Step budget must be stopped deterministically by
    /// `Steps`, not run forever and not stopped by any other quota (no
    /// calls, no growing registers, no effect opcodes are in this program).
    ///
    /// The natural bound (10_000, ~50x the configured budget) is
    /// deliberately small rather than near-infinite: this keeps the test
    /// itself fast, and - more importantly - keeps its own mutation proof
    /// (removing the Steps charge) a fast, ordinary test *failure* rather
    /// than a multi-minute test *hang*. A near-infinite bound would still
    /// prove the same property when the implementation is correct, but
    /// would turn "the charge was removed" into "the test suite hangs",
    /// which is a poor permanent regression guard.
    #[test]
    fn vm_steps_deterministically_stops_a_verified_backward_loop() {
        let src = r#"
            fn main() {
                let mut i: i32 = 0;
                while i < 10000 {
                    i = i + 1;
                }
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_steps = 200;
        let (_, result) = run_semcode_for_test(&bytes, config);
        match result.unwrap_err() {
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Steps,
                ..
            }) => {}
            other => panic!("expected Steps exhaustion to stop the loop, got {other:?}"),
        }
    }

    #[test]
    fn vm_calls_zero_limit_root_only_program_succeeds() {
        let src = r#"
            fn main() { return; }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_calls = 0;
        let (vm, result) = run_semcode_for_test(&bytes, config);
        result.expect("a root-only program must run under max_calls = 0");
        assert_eq!(vm.calls, 0);
    }

    #[test]
    fn vm_calls_zero_limit_rejects_the_first_nested_call() {
        let src = r#"
            fn helper() { return; }
            fn main() { helper(); return; }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_calls = 0;
        let (vm, result) = run_semcode_for_test(&bytes, config);
        assert_eq!(
            result.unwrap_err(),
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Calls,
                limit: 0,
                used: 1,
            })
        );
        assert_eq!(vm.calls, 0, "the rejected call itself must not be charged");
    }

    /// Primary Calls proof, deliberately iterative/non-recursive so
    /// `StackDepth` cannot mask this quota (the call stack never exceeds
    /// depth 2 - `main` plus one `helper` frame at a time - regardless of
    /// how many times the loop below calls `helper`).
    #[test]
    fn vm_calls_exactly_n_nested_calls_succeeds() {
        let src = r#"
            fn helper() { return; }
            fn main() {
                let mut i: i32 = 0;
                while i < 3 {
                    helper();
                    i = i + 1;
                }
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_calls = 3;
        let (vm, result) = run_semcode_for_test(&bytes, config);
        result.expect("exactly N nested calls must be admitted under max_calls = N");
        assert_eq!(vm.calls, 3);
    }

    #[test]
    fn vm_calls_n_plus_one_nested_calls_rejects() {
        let src = r#"
            fn helper() { return; }
            fn main() {
                let mut i: i32 = 0;
                while i < 3 {
                    helper();
                    i = i + 1;
                }
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_calls = 2;
        let (vm, result) = run_semcode_for_test(&bytes, config);
        assert_eq!(
            result.unwrap_err(),
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Calls,
                limit: 2,
                used: 3,
            })
        );
        assert_eq!(vm.calls, 2, "no partial charge past the rejected call");
    }

    #[test]
    fn vm_closure_call_consumes_one_call() {
        let src = r#"
            fn main() {
                let offset: f64 = 1.0;
                let add: Closure(f64 -> f64) = (x => x + offset);
                let total: f64 = add(2.0);
                assert(total == 3.0);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        let (vm, result) = run_semcode_for_test(&bytes, config);
        result.expect("closure invocation must complete");
        assert_eq!(vm.calls, 1, "ClosureCall must consume exactly one Call");
    }

    /// `sqrt` resolves via `try_eval_builtin_call` (inline, no `push_frame`)
    /// rather than admitting a real Semantic function - proves builtins are
    /// structurally excluded from the Calls budget, not merely uncounted by
    /// omission. `assert` is a dedicated opcode (never `push_frame`-routed
    /// either), so its presence here does not confound the result.
    #[test]
    fn vm_builtin_call_does_not_consume_calls() {
        let src = r#"
            fn main() {
                let y: f64 = sqrt(4.0);
                assert(y == 2.0);
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        let (vm, result) = run_semcode_for_test(&bytes, config);
        result.expect("builtin call must complete");
        assert_eq!(
            vm.calls, 0,
            "a builtin resolved inline must not consume a Call"
        );
    }

    /// A `CALL` instruction dispatched with zero Step fuel remaining must be
    /// stopped by `Steps` before the dispatch loop's `match` body - and
    /// therefore before `push_frame` - ever runs, so it never reaches the
    /// point where a Calls charge would apply. Proves the frozen precedence
    /// (#1759 §3) specifically for the one opcode that spends both budgets.
    #[test]
    fn vm_call_opcode_with_no_step_fuel_is_stopped_by_steps_before_calls() {
        let src = r#"
            fn helper() { return; }
            fn main() { helper(); return; }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_steps = 0;
        let (vm, result) = run_semcode_for_test(&bytes, config);
        assert_eq!(
            result.unwrap_err(),
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Steps,
                limit: 0,
                used: 1,
            })
        );
        assert_eq!(
            vm.calls, 0,
            "Steps exhaustion must preempt any Calls charge"
        );
    }

    /// A nested call rejected by an unrelated structural quota (`Frames`
    /// here) must not consume a Call - `push_frame` returns via `?` before
    /// ever reaching the Calls charge at its own tail end (#1759 §6).
    #[test]
    fn vm_nested_call_rejected_by_frames_does_not_consume_calls() {
        let src = r#"
            fn helper() { return; }
            fn main() { helper(); return; }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_frames = 1;
        let (vm, result) = run_semcode_for_test(&bytes, config);
        assert_eq!(
            result.unwrap_err(),
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Frames,
                limit: 1,
                used: 2,
            })
        );
        assert_eq!(
            vm.calls, 0,
            "a call rejected by Frames must not itself consume a Call"
        );
    }

    /// Same proof as above, for `StackDepth` - which fails closed as
    /// `RuntimeError::StackOverflow` rather than `QuotaExceeded`, per the
    /// existing compatibility mapping, but must equally leave `Calls`
    /// unmoved since it too is checked before the Calls charge in
    /// `push_frame`'s admission order (#1759 §6).
    #[test]
    fn vm_nested_call_rejected_by_stack_depth_does_not_consume_calls() {
        let src = r#"
            fn helper() { return; }
            fn main() { helper(); return; }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_stack_depth = 1;
        let (vm, result) = run_semcode_for_test(&bytes, config);
        assert_eq!(result.unwrap_err(), RuntimeError::StackOverflow);
        assert_eq!(
            vm.calls, 0,
            "a call rejected by StackDepth must not itself consume a Call"
        );
    }

    /// Same proof for `Registers`, the last structural admission check
    /// before Calls in the frozen order (#1759 §6) - the C1 corrective
    /// checkpoint moved the Calls charge to fire immediately after this
    /// check succeeds and strictly before any frame state (registers,
    /// ownership setup, the `Frame` itself) is materialized, so this test
    /// also guards against Calls being charged even though no frame content
    /// was ever built for the rejected invocation.
    ///
    /// Built via raw `IrInstr` rather than compiled source: `push_frame`'s
    /// own `initial_reg_count` check is `16usize.max(args.len())`, so
    /// exceeding it for a *nested* call while the root stays within budget
    /// requires `args.len() > 16` - but through the compiled surface
    /// language every one of N call arguments must simultaneously occupy
    /// its own register at the call site, so the *caller* would need N
    /// live registers to marshal them, tripping the caller's own
    /// register-growth check before `push_frame` for the callee is ever
    /// reached (a real construction bug caught by re-running this exact
    /// mutation proof after C1). The raw `IrInstr::Call.args: Vec<u16>`
    /// list has no such constraint - listing the same register 17 times
    /// gives `helper` an `args.len()` of 17 while `main` only ever
    /// populates that one register, so `main`'s own registers never grow
    /// past 1.
    #[test]
    fn vm_nested_call_rejected_by_registers_does_not_consume_calls() {
        let main_fn = IrFunction {
            name: "main".to_string(),
            instrs: vec![
                IrInstr::LoadI32 { dst: 0, val: 1 },
                IrInstr::Call {
                    dst: None,
                    name: "helper".to_string(),
                    args: vec![0; 17],
                },
                IrInstr::Ret { src: None },
            ],
            ownership_events: Vec::new(),
            params: Vec::new(),
        };
        let helper_fn = IrFunction {
            name: "helper".to_string(),
            instrs: vec![IrInstr::Ret { src: None }],
            ownership_events: Vec::new(),
            params: vec![CallableValueFamily::I32; 17],
        };
        let bytes = emit_ir_to_semcode(&[main_fn, helper_fn], false).expect("emit test program");
        let mut config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        config.quotas.max_registers = 16;
        let (vm, result) = run_semcode_for_test(&bytes, config);
        assert_eq!(
            result.unwrap_err(),
            RuntimeError::QuotaExceeded(QuotaExceeded {
                kind: QuotaKind::Registers,
                limit: 16,
                used: 17,
            })
        );
        assert_eq!(
            vm.calls, 0,
            "a call rejected by Registers must not itself consume a Call"
        );
    }

    /// Same proof for a signature/arity (`validate_call_arguments`)
    /// rejection - the *earliest* admission check in `push_frame`, ahead of
    /// even Frames. Built via raw `IrInstr` (bypassing the source-level
    /// frontend's own static type checking, which would otherwise reject
    /// this mismatch before it ever reached the VM) and run through the
    /// same raw/unverified `parse_semcode` path `run_semcode_for_test` uses,
    /// so the VM's own independent runtime-family check - not the verifier -
    /// is what actually fires here.
    #[test]
    fn vm_nested_call_rejected_by_signature_mismatch_does_not_consume_calls() {
        let main_fn = IrFunction {
            name: "main".to_string(),
            instrs: vec![
                IrInstr::LoadI32 { dst: 0, val: 42 },
                IrInstr::Call {
                    dst: None,
                    name: "helper".to_string(),
                    args: vec![0],
                },
                IrInstr::Ret { src: None },
            ],
            ownership_events: Vec::new(),
            params: Vec::new(),
        };
        let helper_fn = IrFunction {
            name: "helper".to_string(),
            instrs: vec![IrInstr::Ret { src: None }],
            ownership_events: Vec::new(),
            params: vec![CallableValueFamily::Bool],
        };
        let bytes = emit_ir_to_semcode(&[main_fn, helper_fn], false).expect("emit test program");
        let config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        let (vm, result) = run_semcode_for_test(&bytes, config);
        match result.unwrap_err() {
            RuntimeError::TypeMismatchRuntime(_) => {}
            other => panic!("expected a runtime family mismatch, got {other:?}"),
        }
        assert_eq!(
            vm.calls, 0,
            "a call rejected by signature/arity validation must not itself consume a Call"
        );
    }

    /// Runs a raw `IrInstr` program (bypassing the source-level frontend, so
    /// a real host+capability effect opcode can be exercised directly) and
    /// returns the final `VM` alongside the result, mirroring
    /// `run_prometheus_program_capturing_value` but preserving `vm` so
    /// `vm.calls`/`vm.steps` remain inspectable afterward.
    fn run_prometheus_program_returning_vm<H: PrometheusHostAbi, C: CapabilityChecker>(
        instrs: Vec<IrInstr>,
        host: &mut H,
        capabilities: &C,
    ) -> (VM, Result<Value, RuntimeError>) {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs,
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit test program");
        let config = ExecutionConfig::for_context(ExecutionContext::KernelBound);
        let token = verify_semcode_token_with_quotas(&bytes, config.quotas)
            .expect("test program must admit");
        let entry_token = token.require_entry("main").expect("main entry");
        let program = prepare_verified_execution(&entry_token).expect("prepare");
        let mut vm = VM {
            functions: program.functions,
            callstack: Vec::new(),
            config,
            effect_calls: 0,
            steps: 0,
            calls: 0,
            symbols: program.runtime_symbols,
            prng_state: 0,
        };
        let result = push_frame(&mut vm, entry_token.entry(), Vec::new(), None).and_then(|()| {
            let mut bridge = PrometheusVmHost { host, capabilities };
            let mut observation = HelloObservationRuntime::discard();
            exec_loop(&mut vm, &mut bridge, &mut observation)
        });
        (vm, result)
    }

    /// A `GateRead` effect opcode never calls `push_frame` (it dispatches
    /// its host call inline in its own `match` arm - see #1759's own
    /// architecture inspection), so it must not consume a Call either.
    #[test]
    fn vm_gate_read_effect_opcode_does_not_consume_calls() {
        let mut host = prom_abi::RecordingHostAbi::with_read_value(AbiValue::Quad(1));
        let capabilities = gate_read_capabilities();
        let (vm, result) = run_prometheus_program_returning_vm(
            gate_read_into_return_program(),
            &mut host,
            &capabilities,
        );
        result.expect("gate read program must succeed");
        assert_eq!(
            vm.calls, 0,
            "an effect opcode must not consume the Calls quota"
        );
    }

    // --- #1771 (FA-09-003, umbrella #1617) regression matrix ---------------
    //
    // `Opcode::MapGet`'s missing-key path reads `default_reg` lazily (only
    // when the key is absent) via `get_reg(vm, frame_idx, default_reg)`.
    // Built with `emit_ir_to_semcode` directly (bypassing the source-level
    // `map_get(...)` builtin) specifically so `default_val` can reference a
    // register the program never writes — the source-level builtin always
    // lowers its third argument expression into a register write before
    // `IrInstr::MapGet` is emitted, which would make "physically absent
    // default register" unconstructible from source.
    //
    // `push_frame` materializes at least 16 register slots regardless of
    // arguments, so `r100` is guaranteed physically absent from a fresh
    // zero-arg frame in every case below that uses it, and no instruction
    // in these programs writes a register anywhere near that high.

    /// Builds a two-function program (`helper` returns unit; `main` runs
    /// `main_instrs`) via direct IR emission.
    fn build_map_get_test_program(main_instrs: Vec<IrInstr>) -> Vec<u8> {
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
        .expect("emit map_get test program")
    }

    fn run_map_get_test_program(bytes: &[u8]) -> Result<Value, RuntimeError> {
        let token = verify_semcode_token(bytes).expect("verifier must accept: default_reg is a legal in-budget register reference regardless of physical materialization");
        let entry = token.require_entry("main").expect("entry");
        run_verified_function_semcode_with_args(&entry, vec![])
    }

    /// #1756 (FA-07-016): raw/unverified execution with caller-supplied
    /// arguments - bypasses `sm-verify` admission (via `parse_semcode`
    /// rather than a verified token) but exercises the exact same
    /// `push_frame`/`validate_call_arguments`/`get_reg` VM machinery a
    /// verified call would. Several #1770/#1771 regressions constructed a
    /// program whose #1756 proof now correctly fails canonical
    /// verification entirely (that is the intended, documented
    /// consequence: the new static proof subsumes what used to be a
    /// runtime-only VM defense) - this lets those regressions keep proving
    /// their original claim about the VM's own independent execution
    /// semantics, on the identical bytes, without a verified token.
    fn run_raw_semcode_with_args(
        bytes: &[u8],
        entry: &str,
        args: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        let program = parse_semcode(bytes).expect("parse");
        let mut vm = VM {
            functions: program.functions,
            callstack: Vec::new(),
            config: ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
            effect_calls: 0,
            steps: 0,
            calls: 0,
            symbols: program.runtime_symbols,
            prng_state: 0,
        };
        push_frame(&mut vm, entry, args, None)?;
        let mut host = LegacyVmHost;
        let mut observation = HelloObservationRuntime::discard();
        exec_loop(&mut vm, &mut host, &mut observation)
    }

    /// A5.1: missing key + valid initialized default register -> exact
    /// default value.
    #[test]
    fn map_get_missing_key_returns_initialized_default_register_value() {
        let bytes = build_map_get_test_program(vec![
            IrInstr::MapEmpty { dst: 0 },
            IrInstr::LoadI32 { dst: 1, val: 1 },
            IrInstr::LoadI32 { dst: 2, val: 10 },
            IrInstr::MapSet {
                dst: 0,
                map: 0,
                key: 1,
                val: 2,
            },
            IrInstr::LoadI32 { dst: 3, val: 2 }, // requested key (absent: map only has key 1)
            IrInstr::LoadI32 { dst: 4, val: 999 }, // initialized default
            IrInstr::MapGet {
                dst: 5,
                map: 0,
                key: 3,
                default_val: 4,
            },
            IrInstr::Ret { src: Some(5) },
        ]);
        let res = run_map_get_test_program(&bytes).expect("run");
        assert_eq!(res, Value::I32(999));
    }

    /// A5.2 (the core #1771 regression): missing key + a `default_reg` that
    /// is numerically in-budget but physically absent from the frame must
    /// propagate `get_reg`'s own read error, not fabricate `Value::Unit`.
    ///
    /// Confirmed RED against the pre-fix `unwrap_or(Value::Unit)` swallow
    /// (returned `Ok(Value::Unit)` instead of this error) before the
    /// production fix in this same PR was applied.
    #[test]
    fn map_get_missing_key_with_physically_absent_default_register_propagates_error() {
        let bytes = build_map_get_test_program(vec![
            IrInstr::MapEmpty { dst: 0 },
            IrInstr::LoadI32 { dst: 1, val: 1 },
            IrInstr::LoadI32 { dst: 2, val: 10 },
            IrInstr::MapSet {
                dst: 0,
                map: 0,
                key: 1,
                val: 2,
            },
            IrInstr::LoadI32 { dst: 3, val: 2 }, // requested key (absent)
            IrInstr::MapGet {
                dst: 5,
                map: 0,
                key: 3,
                default_val: 100, // never written anywhere in this program
            },
            IrInstr::Ret { src: Some(5) },
        ]);
        // #1756 (FA-07-016): an unwritten `default_reg` is now rejected by
        // canonical verification itself (conservative - key presence is
        // never statically proven), so this program's #1771 runtime-
        // laziness claim is exercised via the raw/unverified path instead.
        let err = run_raw_semcode_with_args(&bytes, "main", Vec::new())
            .expect_err("must propagate the read error");
        assert_eq!(
            err,
            RuntimeError::BadFormat("read invalid reg r100".to_string())
        );
    }

    /// A5.3: present key + physically absent default register -> the map
    /// value is returned and `default_reg` is never observed (laziness
    /// boundary case).
    #[test]
    fn map_get_present_key_ignores_physically_absent_default_register() {
        let bytes = build_map_get_test_program(vec![
            IrInstr::MapEmpty { dst: 0 },
            IrInstr::LoadI32 { dst: 1, val: 1 },
            IrInstr::LoadI32 { dst: 2, val: 10 },
            IrInstr::MapSet {
                dst: 0,
                map: 0,
                key: 1,
                val: 2,
            },
            IrInstr::LoadI32 { dst: 3, val: 1 }, // requested key (present)
            IrInstr::MapGet {
                dst: 5,
                map: 0,
                key: 3,
                default_val: 100, // never written; must not be read
            },
            IrInstr::Ret { src: Some(5) },
        ]);
        // #1756 (FA-07-016): see the doc comment on
        // `run_raw_semcode_with_args` - the verifier's conservative MAP_GET
        // rule now rejects this program regardless of the present-key
        // runtime path, so #1771's laziness claim is exercised raw.
        let res = run_raw_semcode_with_args(&bytes, "main", Vec::new()).expect("run");
        assert_eq!(res, Value::I32(10));
    }

    /// A5.4: present key + valid default register -> the map value still
    /// wins over the default.
    #[test]
    fn map_get_present_key_prefers_map_value_over_valid_default() {
        let bytes = build_map_get_test_program(vec![
            IrInstr::MapEmpty { dst: 0 },
            IrInstr::LoadI32 { dst: 1, val: 1 },
            IrInstr::LoadI32 { dst: 2, val: 10 },
            IrInstr::MapSet {
                dst: 0,
                map: 0,
                key: 1,
                val: 2,
            },
            IrInstr::LoadI32 { dst: 3, val: 1 }, // requested key (present)
            IrInstr::LoadI32 { dst: 4, val: 999 }, // valid default, must be ignored
            IrInstr::MapGet {
                dst: 5,
                map: 0,
                key: 3,
                default_val: 4,
            },
            IrInstr::Ret { src: Some(5) },
        ]);
        let res = run_map_get_test_program(&bytes).expect("run");
        assert_eq!(res, Value::I32(10));
    }

    /// A5.5: the default register genuinely, explicitly holds `Value::Unit`
    /// (written by a real call to a function that returns unit) -> the
    /// missing-key result is a legitimate `Unit`, proving `Unit` itself is
    /// never rejected -- only a *fabricated* `Unit` (from a swallowed read
    /// error) is forbidden.
    #[test]
    fn map_get_missing_key_with_explicit_unit_default_returns_legitimate_unit() {
        let bytes = build_map_get_test_program(vec![
            IrInstr::MapEmpty { dst: 0 },
            IrInstr::LoadI32 { dst: 1, val: 1 },
            IrInstr::LoadI32 { dst: 2, val: 10 },
            IrInstr::MapSet {
                dst: 0,
                map: 0,
                key: 1,
                val: 2,
            },
            IrInstr::LoadI32 { dst: 3, val: 2 }, // requested key (absent)
            IrInstr::Call {
                dst: Some(4),
                name: "helper".to_string(),
                args: vec![],
            }, // r4 = Value::Unit, via a real RET-with-no-src write
            IrInstr::MapGet {
                dst: 5,
                map: 0,
                key: 3,
                default_val: 4,
            },
            IrInstr::Ret { src: Some(5) },
        ]);
        let res = run_map_get_test_program(&bytes).expect("run");
        assert_eq!(res, Value::Unit);
    }

    /// B5 layout evidence: `RegisterSlot` has the same layout as the stdlib
    /// `Option<Value>` alternative would have. This locks in the *relative*
    /// comparison the representation decision was based on; it does not by
    /// itself bound `Value`'s own size (a Rust layout detail of the
    /// compiler/target, not a semantic contract this test should gate) --
    /// the concrete `size_of::<Value>() == size_of::<Option<Value>>() ==
    /// size_of::<RegisterSlot>() == 80` measurement is recorded as PR
    /// evidence instead.
    #[test]
    fn register_slot_layout_matches_option_value() {
        assert_eq!(
            std::mem::size_of::<RegisterSlot>(),
            std::mem::size_of::<Option<Value>>(),
            "RegisterSlot must not cost more than the stdlib Option<Value> alternative"
        );
    }

    // --- #1770 (FA-09-002, umbrella #1617) regression matrix ---------------
    //
    // `push_frame` used to materialize at least 16 register slots with
    // `Value::Unit` regardless of arguments, and `write_reg` filled any gap
    // created by a high-register write the same way. That made "physical
    // slot exists" and "a Semantic value was actually defined here"
    // indistinguishable: an entirely undefined register read back as a
    // plausible `Unit` instead of failing. Built with `emit_ir_to_semcode`
    // directly for the same reason as #1771's tests: precise control over
    // exactly which registers are written, independent of what the
    // source-level compiler happens to allocate.

    fn build_register_init_test_program(functions: Vec<IrFunction>) -> Vec<u8> {
        emit_ir_to_semcode(&functions, false).expect("emit register-init test program")
    }

    // #1773 (FA-09-005): `params` must match what each call site below
    // actually invokes this function with - the new canonical-signature
    // contract is enforced before `push_frame`, so a declared arity that
    // disagreed with the real call would reject before ever reaching the
    // register-initialization behavior these #1770 tests exist to probe.
    fn ret_only_function(
        name: &str,
        src: Option<u16>,
        params: Vec<CallableValueFamily>,
    ) -> IrFunction {
        IrFunction {
            name: name.to_string(),
            instrs: vec![IrInstr::Ret { src }],
            ownership_events: Vec::new(),
            params,
        }
    }

    /// B12.1 + B13 cross-layer defense, updated for #1756 (FA-07-016): a
    /// zero-arg entry reading `r0` with no preceding definition is now
    /// rejected by the verifier itself (`prove_definite_register_assignment`
    /// proves `ENTRY_DEFS` is empty for a zero-parameter signature, so `r0`
    /// is never definitely defined). The canonical verified route therefore
    /// never reaches the VM for this artifact at all - but the VM's own
    /// independent defense (#1770's `RegisterSlot::Uninitialized`) must
    /// still catch the identical malformed bytes if some other, unverified
    /// path reaches them, which the second half of this test proves via the
    /// raw/unverified execution route that intentionally bypasses
    /// `sm-verify` admission.
    #[test]
    fn frame_reads_zero_arg_entry_register_as_uninitialized_error() {
        let bytes =
            build_register_init_test_program(vec![ret_only_function("main", Some(0), vec![])]);
        let report = verify_semcode_token(&bytes)
            .expect_err("verifier must now reject the undefined r0 read (#1756)");
        assert_eq!(
            report.diagnostics[0].code,
            sm_verify::VerificationCode::UndefinedRegisterRead
        );

        let err = run_semcode_with_entry_and_config(
            &bytes,
            "main",
            ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
        )
        .expect_err("VM must independently reject the undefined read on the raw/unverified path");
        assert_eq!(
            err,
            RuntimeError::BadFormat("read uninitialized reg r0".to_string())
        );
    }

    /// B12.2, updated for #1756 (FA-07-016): a write to a high register must
    /// not materialize the gap below it as legitimate values. The verifier
    /// now proves this statically too (r50 is read but never written on any
    /// path reaching the `RET`, so it is never definitely defined), so the
    /// canonical verified route rejects before the VM ever runs; the VM's
    /// own independent defense against the identical bytes is proven
    /// separately via the raw/unverified execution route.
    #[test]
    fn write_high_register_then_read_gap_is_uninitialized_error() {
        let bytes = build_register_init_test_program(vec![IrFunction {
            name: "main".to_string(),
            instrs: vec![
                IrInstr::LoadI32 { dst: 100, val: 1 },
                IrInstr::Ret { src: Some(50) },
            ],
            ownership_events: Vec::new(),
            params: Vec::new(),
        }]);
        let report = verify_semcode_token(&bytes)
            .expect_err("verifier must now reject the undefined r50 read (#1756)");
        assert_eq!(
            report.diagnostics[0].code,
            sm_verify::VerificationCode::UndefinedRegisterRead
        );

        let err = run_semcode_with_entry_and_config(
            &bytes,
            "main",
            ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
        )
        .expect_err("gap register must remain uninitialized on the raw/unverified path");
        assert_eq!(
            err,
            RuntimeError::BadFormat("read uninitialized reg r50".to_string())
        );
    }

    /// B12.3 + B12.9: `Value::Unit` remains an ordinary, readable value once
    /// genuinely written -- including via a real `CALL` whose callee
    /// returns unit (`Ret { src: None }`), proving the call's destination
    /// register becomes explicitly initialized rather than only
    /// incidentally holding `Unit` by virtue of unwritten storage.
    #[test]
    fn explicit_unit_write_then_read_succeeds_and_call_result_is_initialized() {
        let bytes = build_register_init_test_program(vec![
            ret_only_function("helper", None, vec![]),
            IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::Call {
                        dst: Some(5),
                        name: "helper".to_string(),
                        args: vec![],
                    },
                    IrInstr::Ret { src: Some(5) },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            },
        ]);
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry = token.require_entry("main").expect("entry");
        let res = run_verified_function_semcode_with_args(&entry, vec![]).expect("run");
        assert_eq!(res, Value::Unit);
    }

    /// B12.4 + B12.5 (single-argument half), updated for #1756 (FA-07-016):
    /// exactly one supplied argument defines `r0`; a sibling register that
    /// was never supplied and never written must remain uninitialized, not
    /// "magically" defined. `read_r1`'s canonical signature declares one
    /// parameter (`ENTRY_DEFS = {r0}`), yet its body reads `r1` - the
    /// verifier now proves that read undefined and rejects the whole
    /// artifact containing it (verification is per-artifact, not
    /// per-function), so the "sibling register" half of this claim can no
    /// longer share one verified token with the well-formed half; it is
    /// proven on its own via the raw/unverified execution path instead,
    /// preserving the VM's own independent defense.
    #[test]
    fn single_argument_initializes_only_r0() {
        let good_bytes = build_register_init_test_program(vec![
            ret_only_function("main", None, vec![]),
            ret_only_function("read_r0", Some(0), vec![CallableValueFamily::I32]),
        ]);
        let token = verify_semcode_token(&good_bytes).expect("verify");
        let entry_r0 = token.require_entry("read_r0").expect("entry read_r0");
        let ok = run_verified_function_semcode_with_args(&entry_r0, vec![Value::I32(42)])
            .expect("supplied arg register must be defined");
        assert_eq!(ok, Value::I32(42));

        let bad_bytes = build_register_init_test_program(vec![ret_only_function(
            "read_r1",
            Some(1),
            vec![CallableValueFamily::I32],
        )]);
        let report = verify_semcode_token(&bad_bytes)
            .expect_err("verifier must now reject the undefined r1 read (#1756)");
        assert_eq!(
            report.diagnostics[0].code,
            sm_verify::VerificationCode::UndefinedRegisterRead
        );
        // #1756: `run_semcode_with_entry_and_config` always supplies zero
        // args (see its own doc comment on why that's safe for #1773's
        // fused `push_frame` arity check), which would reject on argc
        // mismatch before ever reaching the register read this test needs
        // to observe. `run_raw_semcode_with_args` bypasses only
        // verification, not argument supply, so the real one-argument call
        // shape is preserved.
        let err = run_raw_semcode_with_args(&bad_bytes, "read_r1", vec![Value::I32(42)])
            .expect_err("unsupplied sibling register must not be magically defined");
        assert_eq!(
            err,
            RuntimeError::BadFormat("read uninitialized reg r1".to_string())
        );
    }

    /// B12.5 (multiple-arguments half), updated for #1756 (FA-07-016):
    /// exactly the supplied registers are initialized -- not one more.
    /// `read_r2`'s canonical signature declares two parameters
    /// (`ENTRY_DEFS = {r0, r1}`), yet its body reads `r2` - the verifier now
    /// proves that read undefined and rejects the whole artifact containing
    /// it, so (as in `single_argument_initializes_only_r0` above) the two
    /// halves of this claim can no longer share one verified artifact; the
    /// "past the supplied count" half is proven via the raw/unverified path.
    #[test]
    fn multiple_arguments_initialize_exactly_supplied_registers() {
        let args = vec![Value::I32(1), Value::I32(2)];

        let good_bytes = build_register_init_test_program(vec![
            ret_only_function("main", None, vec![]),
            ret_only_function(
                "read_r1",
                Some(1),
                vec![CallableValueFamily::I32, CallableValueFamily::I32],
            ),
        ]);
        let token = verify_semcode_token(&good_bytes).expect("verify");
        let entry_r1 = token.require_entry("read_r1").expect("entry read_r1");
        let ok = run_verified_function_semcode_with_args(&entry_r1, args.clone())
            .expect("second supplied arg register must be defined");
        assert_eq!(ok, Value::I32(2));

        let bad_bytes = build_register_init_test_program(vec![ret_only_function(
            "read_r2",
            Some(2),
            vec![CallableValueFamily::I32, CallableValueFamily::I32],
        )]);
        let report = verify_semcode_token(&bad_bytes)
            .expect_err("verifier must now reject the undefined r2 read (#1756)");
        assert_eq!(
            report.diagnostics[0].code,
            sm_verify::VerificationCode::UndefinedRegisterRead
        );
        let err = run_raw_semcode_with_args(&bad_bytes, "read_r2", args)
            .expect_err("register past the supplied argument count must be uninitialized");
        assert_eq!(
            err,
            RuntimeError::BadFormat("read uninitialized reg r2".to_string())
        );
    }

    /// B12.10, updated for #1756 (FA-07-016): a `CALL` with no destination
    /// operand must define no register at all -- not even incidentally. The
    /// verifier now proves this too (`r7` is read but nothing on any path
    /// writes it, since the no-dst `CALL` correctly defines nothing), so the
    /// canonical route rejects before the VM runs; the VM's own defense
    /// against the identical bytes is proven via the raw/unverified path.
    #[test]
    fn call_without_destination_creates_no_register_definition() {
        let bytes = build_register_init_test_program(vec![
            ret_only_function("helper", None, vec![]),
            IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::Call {
                        dst: None,
                        name: "helper".to_string(),
                        args: vec![],
                    },
                    IrInstr::Ret { src: Some(7) },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            },
        ]);
        let report = verify_semcode_token(&bytes)
            .expect_err("verifier must now reject the undefined r7 read (#1756)");
        assert_eq!(
            report.diagnostics[0].code,
            sm_verify::VerificationCode::UndefinedRegisterRead
        );

        let err = run_semcode_with_entry_and_config(
            &bytes,
            "main",
            ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
        )
        .expect_err("no-dst call must not define r7, proven on the raw/unverified path");
        assert_eq!(
            err,
            RuntimeError::BadFormat("read uninitialized reg r7".to_string())
        );
    }

    /// B12.11: a `ClosureCall`'s result register is initialized the same
    /// way a plain `CALL`'s is.
    #[test]
    fn closure_call_result_initialization_preserved() {
        let bytes = build_register_init_test_program(vec![
            // #1773 (FA-09-005): ClosureCall now validates too (see the
            // updated doc comment on `validate_call_arguments`) - this
            // closure has zero captures, so its real invocation passes
            // exactly one argument (the closure's own parameter).
            ret_only_function("closure_target", Some(0), vec![CallableValueFamily::I32]),
            IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadI32 { dst: 0, val: 77 },
                    IrInstr::MakeClosure {
                        dst: 1,
                        name: "closure_target".to_string(),
                        captures: vec![],
                    },
                    IrInstr::ClosureCall {
                        dst: Some(2),
                        closure: 1,
                        arg: 0,
                    },
                    IrInstr::Ret { src: Some(2) },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            },
        ]);
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry = token.require_entry("main").expect("entry");
        let res = run_verified_function_semcode_with_args(&entry, vec![]).expect("run");
        assert_eq!(res, Value::I32(77));
    }

    // --- #1773 (FA-09-005, umbrella #1617) regression matrix ---------------
    //
    // `sm-verify` can only prove argument COUNT statically (registers are
    // untyped storage - see the #1773 architecture checkpoint on issue
    // #1773). Runtime FAMILY enforcement is uniquely `validate_call_
    // arguments`'s responsibility, checked immediately before `push_frame`
    // for both the public invocation helper and internal `Opcode::Call` -
    // these promote the original architecture-checkpoint RED reproductions
    // to permanent coverage for that half of the contract.

    /// Public invocation: a family mismatch (Bool supplied where the
    /// canonical signature declares I32) must be rejected before the frame
    /// is even pushed - not merely "the body traps on the wrong value".
    #[test]
    fn public_invocation_rejects_argument_family_mismatch() {
        let bytes = build_register_init_test_program(vec![ret_only_function(
            "main",
            Some(0),
            vec![CallableValueFamily::I32],
        )]);
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry = token.require_entry("main").expect("entry");
        let err = run_verified_function_semcode_with_args(&entry, vec![Value::Bool(true)])
            .expect_err("family mismatch must be rejected");
        assert!(matches!(err, RuntimeError::TypeMismatchRuntime(_)));
    }

    /// Public invocation: the family check applies even to a parameter the
    /// callee's body never reads - no body instruction should be needed to
    /// expose the mismatch (mirrors #1770's "unused parameter register
    /// storage must not fabricate a value" invariant, one layer up).
    #[test]
    fn public_invocation_rejects_family_mismatch_in_unused_parameter() {
        let bytes = build_register_init_test_program(vec![ret_only_function(
            "main",
            None, // body never reads r0
            vec![CallableValueFamily::I32],
        )]);
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry = token.require_entry("main").expect("entry");
        let err = run_verified_function_semcode_with_args(&entry, vec![Value::Text("x".into())])
            .expect_err("family mismatch in an unused parameter must still be rejected");
        assert!(matches!(err, RuntimeError::TypeMismatchRuntime(_)));
    }

    /// Internal `Opcode::Call`: the exact same family enforcement applies
    /// to a call made from inside another function's body, not just the
    /// public entry point - proving the single shared validator, not two
    /// independently-maintained checks.
    #[test]
    fn internal_call_rejects_argument_family_mismatch() {
        let bytes = build_register_init_test_program(vec![
            ret_only_function("callee", Some(0), vec![CallableValueFamily::I32]),
            IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadBool { dst: 0, val: true },
                    IrInstr::Call {
                        dst: Some(1),
                        name: "callee".to_string(),
                        args: vec![0],
                    },
                    IrInstr::Ret { src: Some(1) },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            },
        ]);
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry = token.require_entry("main").expect("entry");
        let err = run_verified_function_semcode_with_args(&entry, vec![])
            .expect_err("internal call family mismatch must be rejected");
        assert!(matches!(err, RuntimeError::TypeMismatchRuntime(_)));
    }

    /// Internal `Opcode::Call`: multiple call sites to the same two-
    /// parameter callee, each with correct argc/family, must all succeed -
    /// the validator is not a one-shot/first-call-only check.
    #[test]
    fn internal_call_accepts_multiple_correct_call_sites_to_same_callee() {
        let bytes = build_register_init_test_program(vec![
            IrFunction {
                name: "add".to_string(),
                instrs: vec![
                    IrInstr::AddI32 {
                        dst: 2,
                        lhs: 0,
                        rhs: 1,
                    },
                    IrInstr::Ret { src: Some(2) },
                ],
                ownership_events: Vec::new(),
                params: vec![CallableValueFamily::I32, CallableValueFamily::I32],
            },
            IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadI32 { dst: 0, val: 1 },
                    IrInstr::LoadI32 { dst: 1, val: 2 },
                    IrInstr::Call {
                        dst: Some(2),
                        name: "add".to_string(),
                        args: vec![0, 1],
                    },
                    IrInstr::LoadI32 { dst: 3, val: 10 },
                    IrInstr::LoadI32 { dst: 4, val: 20 },
                    IrInstr::Call {
                        dst: Some(5),
                        name: "add".to_string(),
                        args: vec![3, 4],
                    },
                    IrInstr::AddI32 {
                        dst: 6,
                        lhs: 2,
                        rhs: 5,
                    },
                    IrInstr::Ret { src: Some(6) },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            },
        ]);
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry = token.require_entry("main").expect("entry");
        let res = run_verified_function_semcode_with_args(&entry, vec![]).expect("run");
        assert_eq!(res, Value::I32(33)); // (1+2) + (10+20)
    }

    /// Internal `Opcode::Call`: a self-recursive callee's own declared
    /// signature is enforced identically on every recursive call, not only
    /// the initial (non-recursive) entry.
    #[test]
    fn internal_call_enforces_signature_across_recursive_calls() {
        let bytes = build_register_init_test_program(vec![IrFunction {
            name: "count_down".to_string(),
            instrs: vec![
                IrInstr::LoadI32 { dst: 1, val: 0 },
                IrInstr::CmpEq {
                    dst: 2,
                    lhs: 0,
                    rhs: 1,
                },
                IrInstr::JmpIf {
                    cond: 2,
                    label: "base_case".to_string(),
                },
                IrInstr::LoadI32 { dst: 3, val: 1 },
                IrInstr::SubI32 {
                    dst: 4,
                    lhs: 0,
                    rhs: 3,
                },
                IrInstr::Call {
                    dst: Some(5),
                    name: "count_down".to_string(),
                    args: vec![4],
                },
                IrInstr::Ret { src: Some(5) },
                IrInstr::Label {
                    name: "base_case".to_string(),
                },
                IrInstr::Ret { src: Some(0) },
            ],
            ownership_events: Vec::new(),
            params: vec![CallableValueFamily::I32],
        }]);
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry = token.require_entry("count_down").expect("entry");
        let res =
            run_verified_function_semcode_with_args(&entry, vec![Value::I32(3)]).expect("run");
        assert_eq!(res, Value::I32(0));
    }

    /// Builtin-precedence preservation (#1817, re-asserted for #1773): a
    /// call to a name that resolves to a known internal function must still
    /// win over a same-named builtin, unaffected by the new arity/family
    /// validator sitting in front of that same `push_frame` call.
    #[test]
    fn internal_call_precedence_over_builtin_unaffected_by_signature_validation() {
        let bytes = build_register_init_test_program(vec![
            ret_only_function("sqrt", Some(0), vec![CallableValueFamily::I32]),
            IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadI32 { dst: 0, val: 777 },
                    IrInstr::Call {
                        dst: Some(1),
                        name: "sqrt".to_string(),
                        args: vec![0],
                    },
                    IrInstr::Ret { src: Some(1) },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            },
        ]);
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry = token.require_entry("main").expect("entry");
        let res = run_verified_function_semcode_with_args(&entry, vec![]).expect("run");
        assert_eq!(
            res,
            Value::I32(777),
            "internal 'sqrt' must win over the builtin, and its I32 signature must validate \
             cleanly against the I32 arg it is actually called with"
        );
    }

    /// Codex review finding on this PR: `Opcode::ClosureCall` calls
    /// `push_frame` the same way `Opcode::Call` does, and was left
    /// unvalidated by an earlier revision of this fix. A structurally
    /// valid, verifier-admitted artifact could `MAKE_CLOSURE` a target
    /// whose real signature disagrees with what the closure supplies
    /// (captures + the closure's own parameter) and still enter its frame.
    /// Fixed by fusing `validate_call_arguments` directly into
    /// `push_frame`, the one function every real call path - `Call`,
    /// `ClosureCall`, and every verified entry route - already funnels
    /// through.
    #[test]
    fn closure_call_rejects_argument_family_mismatch() {
        let bytes = build_register_init_test_program(vec![
            ret_only_function("closure_target", Some(0), vec![CallableValueFamily::I32]),
            IrFunction {
                name: "main".to_string(),
                instrs: vec![
                    IrInstr::LoadBool { dst: 0, val: true },
                    IrInstr::MakeClosure {
                        dst: 1,
                        name: "closure_target".to_string(),
                        captures: vec![],
                    },
                    IrInstr::ClosureCall {
                        dst: Some(2),
                        closure: 1,
                        arg: 0,
                    },
                    IrInstr::Ret { src: Some(2) },
                ],
                ownership_events: Vec::new(),
                params: Vec::new(),
            },
        ]);
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry = token.require_entry("main").expect("entry");
        let err = run_verified_function_semcode_with_args(&entry, vec![])
            .expect_err("closure call family mismatch must be rejected");
        assert!(matches!(err, RuntimeError::TypeMismatchRuntime(_)));
    }

    /// Codex review finding on this PR: an earlier revision validated only
    /// the argument-taking public helper
    /// (`run_verified_function_semcode_with_args_and_config`) and internal
    /// `Opcode::Call`, leaving every other verified entry route (which
    /// always passes zero args to `push_frame`) unvalidated - a
    /// parameterized function invoked through one of those routes with a
    /// required-but-unused parameter would succeed with no argument at all.
    /// `run_verified_entry_semcode_with_config` is representative of the
    /// whole family (the PROMETHEUS/application host routes share the same
    /// `push_frame` call shape).
    #[test]
    fn verified_entry_route_rejects_zero_args_for_parameterized_function() {
        let bytes = build_register_init_test_program(vec![ret_only_function(
            "main",
            None, // body never reads the unsupplied parameter
            vec![CallableValueFamily::I32],
        )]);
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry = token.require_entry("main").expect("entry");
        let err = run_verified_entry_semcode_with_config(
            &entry,
            ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
        )
        .expect_err("a required parameter must be rejected, not silently supplied zero args");
        assert!(matches!(err, RuntimeError::TypeMismatchRuntime(_)));
    }

    // --- #1772 (FA-09-004, umbrella #1617) regression matrix ---------------
    //
    // `run_verified_function_semcode_with_args[_and_config]` used to accept
    // an independent `func_name: &str` alongside the entry-bound token,
    // letting a token resolved for "a_fn" execute an unrelated "b_fn" from
    // the same verified artifact. Confirmed RED against pre-fix code before
    // this fix (a temporary probe -- since removed, as its call shape no
    // longer compiles -- called `run_verified_function_semcode_with_args
    // (&entry_a, "b_fn", vec![])` and observed `Ok(Value::I32(222))`,
    // proving b_fn's body genuinely executed). `func_name` is now removed
    // entirely: the execution target is always `token.entry()`, making the
    // divergence structurally impossible to express, not merely rejected
    // at runtime.

    fn build_ab_sentinel_program() -> Vec<u8> {
        emit_ir_to_semcode(
            &[
                IrFunction {
                    name: "a_fn".to_string(),
                    instrs: vec![
                        IrInstr::LoadI32 { dst: 0, val: 111 },
                        IrInstr::Ret { src: Some(0) },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
                IrFunction {
                    name: "b_fn".to_string(),
                    instrs: vec![
                        IrInstr::LoadI32 { dst: 0, val: 222 },
                        IrInstr::Ret { src: Some(0) },
                    ],
                    ownership_events: Vec::new(),
                    params: Vec::new(),
                },
            ],
            false,
        )
        .expect("emit")
    }

    /// Item 1: token(A) + request A -> A executes.
    #[test]
    fn entry_token_a_executes_a() {
        let bytes = build_ab_sentinel_program();
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry_a = token.require_entry("a_fn").expect("entry a_fn");
        let res = run_verified_function_semcode_with_args(&entry_a, vec![]).expect("run");
        assert_eq!(res, Value::I32(111));
    }

    /// Item 2: token(A) + "request B" is no longer expressible at all --
    /// `func_name` does not exist as a parameter on this API. The bug is
    /// eliminated by construction, not by a runtime check.
    #[test]
    fn entry_token_execution_target_is_not_independently_selectable() {
        // This test's existence and the crate's own successful compilation
        // ARE the proof: there is no `func_name` parameter through which a
        // caller holding `entry_a` (bound to "a_fn") could name "b_fn".
        let bytes = build_ab_sentinel_program();
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry_a = token.require_entry("a_fn").expect("entry a_fn");
        let res = run_verified_function_semcode_with_args(&entry_a, vec![]).expect("run");
        assert_eq!(
            res,
            Value::I32(111),
            "the only reachable execution target for entry_a is a_fn"
        );
    }

    /// Item 3: independently resolving token(B) -> B executes normally.
    #[test]
    fn entry_token_b_executes_b() {
        let bytes = build_ab_sentinel_program();
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry_b = token.require_entry("b_fn").expect("entry b_fn");
        let res = run_verified_function_semcode_with_args(&entry_b, vec![]).expect("run");
        assert_eq!(res, Value::I32(222));
    }

    /// Item 4: missing entry still produces the existing `require_entry`
    /// failure, unchanged by this fix.
    #[test]
    fn entry_token_missing_entry_still_fails_at_resolution() {
        let bytes = build_ab_sentinel_program();
        let token = verify_semcode_token(&bytes).expect("verify");
        let err = token.require_entry("c_fn").expect_err("must fail");
        assert!(matches!(
            err,
            sm_verify::EntryResolutionError::MissingEntry { .. }
        ));
    }

    /// Item 5: the config-aware sibling obeys the identical identity rule
    /// (execution target is `token.entry()`, args still delivered).
    #[test]
    fn entry_token_config_aware_sibling_uses_token_entry_and_delivers_args() {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "id_fn".to_string(),
                instrs: vec![IrInstr::Ret { src: Some(0) }],
                ownership_events: Vec::new(),
                params: vec![CallableValueFamily::I32],
            }],
            false,
        )
        .expect("emit");
        let token = verify_semcode_token(&bytes).expect("verify");
        let entry = token.require_entry("id_fn").expect("entry id_fn");
        let config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);
        let res = run_verified_function_semcode_with_args_and_config(
            &entry,
            vec![Value::I32(99)],
            config,
        )
        .expect("run");
        assert_eq!(
            res,
            Value::I32(99),
            "args must still be delivered to the bound target"
        );
    }

    // Item 6 (args delivered correctly) is covered directly by
    // `entry_token_config_aware_sibling_uses_token_entry_and_delivers_args`
    // above and by the many pre-existing single-argument tests in this
    // module and in `crates/sm-vm/src/lib.rs` (e.g.
    // `test_2_invoke_function_accepting_i32`), all of which pass unchanged
    // after `func_name` was dropped from the call.
    //
    // Item 7 (ordinary canonical verified-entry execution unchanged) is
    // covered by the full existing `vm_runs_*` / `test_N_*` suite (~90
    // tests), which passes unchanged.

    #[test]
    fn verified_run_rejects_invalid_bytecode_before_execution() {
        let src = "fn main() { return; }";
        let mut bytes = compile_program_to_semcode(src).expect("compile");
        let opcode_pos = 8 + 2 + 4 + 4 + 2;
        bytes[opcode_pos] = 0xff;
        let err = run_verified_semcode(&bytes).expect_err("must fail");
        assert!(matches!(err, RuntimeError::VerifierRejected(_)));
    }

    #[test]
    fn vm_builtin_print_collects_controlled_text_observation_in_memory() {
        let src = r#"
            fn main() {
                print("Hello, World!");
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let events = run_semcode_collecting_hello_observations(&bytes).expect("run");

        assert_eq!(events.len(), 1);
        let event = &events[0];
        assert_eq!(event.operation_kind, "controlled_observation_text");
        assert_eq!(
            event.observation_class,
            HelloObservationClass::ControlledText
        );
        assert_eq!(event.text, "Hello, World!");
        assert_eq!(event.sequence_index, HelloObservationSequenceIndex(0));
    }

    #[test]
    fn vm_builtin_print_assigns_deterministic_sequence_indexes() {
        let src = r#"
            fn main() {
                print("Hello, World!");
                print("Hello, World!");
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let events = run_semcode_collecting_hello_observations(&bytes).expect("run");

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence_index, HelloObservationSequenceIndex(0));
        assert_eq!(events[1].sequence_index, HelloObservationSequenceIndex(1));
    }

    #[test]
    fn vm_builtin_print_rejects_non_text_values_without_implicit_conversion() {
        let mut observation = HelloObservationRuntime::discard();

        let mut host = LegacyVmHost;
        let err = try_eval_builtin_call(&mut host, &mut observation, "print", &[Value::I32(10)])
            .expect_err("i32 must fail");
        assert!(matches!(err, RuntimeError::TypeMismatchRuntime(_)));

        let err = try_eval_builtin_call(
            &mut host,
            &mut observation,
            "print",
            &[Value::Quad(QuadVal::T)],
        )
        .expect_err("quad must fail");
        assert!(matches!(err, RuntimeError::TypeMismatchRuntime(_)));
    }

    #[test]
    fn vm_builtin_print_rejects_host_output_markers_without_observation_leakage() {
        for forbidden in ["stdout", "print", "io.write", "file", "network", "stdin"] {
            let src = format!(
                r#"
                    fn main() {{
                        print("{forbidden}");
                        return;
                    }}
                "#
            );
            let bytes = compile_program_to_semcode(&src).expect("compile");
            let err = run_semcode_collecting_hello_observations(&bytes)
                .expect_err("host-output marker must fail");
            assert!(
                matches!(err, RuntimeError::TypeMismatchRuntime(_)),
                "unexpected error for {forbidden}: {err}"
            );
        }
    }

    fn ownership_tracking_bytes() -> Vec<u8> {
        let src = r#"
            fn pair(flag: bool) -> (i32, bool) = (1, flag);

            fn main() {
                let pair: (i32, bool) = pair(true);
                let (ref left, _): (i32, bool) = pair;
                let _ = left;
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    fn helper_borrow_bytes() -> Vec<u8> {
        let src = r#"
            fn helper(pair: (i32, bool)) {
                let (ref left, _): (i32, bool) = pair;
                let _ = left;
                return;
            }

            fn main() {
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    fn record_field_borrow_tracking_bytes() -> Vec<u8> {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn main() {
                let ctx: DecisionContext = DecisionContext { camera: T, quality: 0.75 };
                let DecisionContext { camera: ref seen_camera, quality: _ } = ctx;
                let _ = seen_camera;
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    fn helper_record_field_borrow_bytes() -> Vec<u8> {
        let src = r#"
            record DecisionContext {
                camera: quad,
                quality: f64,
            }

            fn helper(ctx: DecisionContext) {
                let DecisionContext { camera: ref seen_camera, quality: _ } = ctx;
                let _ = seen_camera;
                return;
            }

            fn main() {
                return;
            }
        "#;
        compile_program_to_semcode(src).expect("compile")
    }

    // #1724 (FA-04-018): lowering now assigns each lexical binding its own
    // `__sm_local_<id>_<source_name>` runtime-local key instead of reusing
    // the raw source spelling, so these ownership-overlap fixtures can no
    // longer find their synthetic root by matching the raw name in the
    // string table. This resolver is fail-closed (0 or >1 matches panics)
    // for the same reason `tests/runtime_ownership_e2e.rs`'s resolver is:
    // silently falling back to "first match" would hide a real ambiguity.
    fn resolve_unique_lowered_local_key(strings: &[String], source_name: &str) -> u32 {
        let matches: Vec<usize> = strings
            .iter()
            .enumerate()
            .filter(|(_, candidate)| is_lowered_local_key_for(candidate, source_name))
            .map(|(index, _)| index)
            .collect();
        match matches.as_slice() {
            [] => panic!(
                "no lowered runtime-local key found for source root '{source_name}' - \
                 looked for '__sm_local_<id>_{source_name}' in the function's string \
                 table {strings:?}"
            ),
            [single] => *single as u32,
            multiple => panic!(
                "ambiguous lowered runtime-local key for source root '{source_name}': \
                 indexes {multiple:?} all match - the test must disambiguate which \
                 binding it means"
            ),
        }
    }

    fn is_lowered_local_key_for(candidate: &str, source_name: &str) -> bool {
        let Some(rest) = candidate.strip_prefix("__sm_local_") else {
            return false;
        };
        let digits_end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        if digits_end == 0 {
            return false;
        }
        let Some(after_digits) = rest.get(digits_end..) else {
            return false;
        };
        let Some(spelling) = after_digits.strip_prefix('_') else {
            return false;
        };
        spelling == source_name
    }

    fn adt_payload_write_overlap_bytes(
        borrowed_adt: Option<(u32, u16)>,
        write_adt: Option<(u32, u16)>,
    ) -> Vec<u8> {
        let src = r#"
            fn main() {
                let e: f64 = 0.0;
                let other: f64 = 1.0;
                e += 2.0;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        rewrite_adt_main_ownership_section(bytes, borrowed_adt, write_adt)
    }

    /// #1891 Checkpoint W2F: harvests a genuine `StoreVarSite` anchor from
    /// the REAL (pre-rewrite) compiled artifact's own Write event, for reuse
    /// as the anchor of a synthetic Write event the fixtures below
    /// hand-author. `check_write_execution_site` (Checkpoint W2F's own
    /// runtime hook) never requires an anchor's instruction to correspond to
    /// the synthetic path's root/component - that provenance is explicitly
    /// out of scope for both the verifier (Checkpoint W2E) and the runtime
    /// (see `WriteExecutionSite`'s own doc comment) - so any genuine
    /// StoreVar PC in this function satisfies it, not only one tied to a
    /// particular local. Every fixture below reassigns a real local via
    /// compound assignment (`x += ...`), which the current compiler
    /// (#1891 Checkpoints W2A/W2C) always resolves to a genuine
    /// `StoreVarSite` Write event - reusing that real anchor here avoids
    /// both a hand-rolled instruction decoder and the `disasm` feature gate
    /// `disasm_semcode` sits behind (these fixtures back tests that run
    /// without that feature enabled).
    fn real_store_var_anchor(env: &sm_format::semcode_decode::DecodedFunctionEnvelope) -> u32 {
        env.write_paths
            .iter()
            .find_map(|p| match p.write_execution {
                Some(sm_format::semcode_decode::DecodedWriteExecution::StoreVarSite(a)) => Some(a),
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!(
                    "function '{}' has no real StoreVarSite Write event to borrow an anchor from",
                    env.name
                )
            })
    }

    /// #1891 Checkpoint W2F, item 5/11.J: white-box proof of the required
    /// Write-check -> StoreVar-commit -> Borrow-activation ordering. No real
    /// source can produce a Borrow and a Write sharing the identical
    /// StoreVar anchor - there is no construct that both reassigns a local
    /// and introduces a `ref` borrow of that same local at the same
    /// instruction - so this hand-declares a synthetic
    /// `Borrow(StoreVarSite(A))` on `x`, rooted at zero components, at the
    /// exact same real anchor `x`'s own genuine `Write(StoreVarSite(A))`
    /// events already carry (harvested via `real_store_var_anchor`, never
    /// invented). Every real Write event is re-emitted verbatim (its own
    /// real anchor/root, not reconstructed from scratch) so this never risks
    /// silently changing what the real compiler actually emitted; only a
    /// deliberate hand-authored Borrow event is added, and only real
    /// instruction bytes execute.
    fn rewrite_main_with_self_referencing_store_var_borrow(bytes: Vec<u8>) -> Vec<u8> {
        let mut cursor = 8usize;
        let name_len = read_u16_le(&bytes, &mut cursor).expect("name len") as usize;
        let name = read_utf8(&bytes, &mut cursor, name_len).expect("name");
        assert_eq!(name, "main");
        let code_len_pos = cursor;
        let code_len = read_u32_le(&bytes, &mut cursor).expect("code len") as usize;
        let code_start = cursor;
        let code_end = code_start + code_len;
        let code = &bytes[code_start..code_end];
        let (_, mut decoded_functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let env = decoded_functions.remove(0);
        assert!(
            !env.write_paths.is_empty(),
            "expected at least one real Write event to borrow an anchor from"
        );
        assert!(
            env.write_paths.iter().all(|p| p.components.is_empty()),
            "this fixture only supports bare scalar reassignments (zero-component paths)"
        );
        let real_anchor = real_store_var_anchor(&env);
        let x_root = env.write_paths[0].root_symbol_id;
        let instr_start = env.instr_start_offset;
        let sig0_len = env
            .signature
            .as_ref()
            .map(|sig| SIGNATURE_SECTION_TAG.len() + 2 + sig.families.len())
            .unwrap_or(0);
        let own0_end = instr_start - sig0_len;
        let ownership_start = code[..own0_end]
            .windows(OWNERSHIP_SECTION_TAG.len())
            .position(|window| window == OWNERSHIP_SECTION_TAG)
            .expect("OWN0 section");

        let mut new_own0 = Vec::new();
        new_own0.extend_from_slice(&OWNERSHIP_SECTION_TAG);
        let total = 1 + env.write_paths.len();
        new_own0.extend_from_slice(&(total as u16).to_le_bytes());
        new_own0.push(OWNERSHIP_EVENT_KIND_BORROW);
        new_own0.push(sm_format::semcode_format::ACTIVATION_MODE_STORE_VAR_SITE);
        new_own0.extend_from_slice(&real_anchor.to_le_bytes());
        new_own0.extend_from_slice(&x_root.to_le_bytes());
        new_own0.extend_from_slice(&0u16.to_le_bytes());
        for p in &env.write_paths {
            let anchor = match p.write_execution {
                Some(sm_format::semcode_decode::DecodedWriteExecution::StoreVarSite(a)) => a,
                other => {
                    panic!("expected every real Write event here to be StoreVarSite, got {other:?}")
                }
            };
            new_own0.push(OWNERSHIP_EVENT_KIND_WRITE);
            new_own0.push(WRITE_EXECUTION_MODE_STORE_VAR_SITE);
            new_own0.extend_from_slice(&anchor.to_le_bytes());
            new_own0.extend_from_slice(&p.root_symbol_id.to_le_bytes());
            new_own0.extend_from_slice(&0u16.to_le_bytes());
        }

        let mut new_code = Vec::with_capacity(code.len());
        new_code.extend_from_slice(&code[..ownership_start]);
        new_code.extend_from_slice(&new_own0);
        new_code.extend_from_slice(&code[own0_end..]);

        let mut out = Vec::with_capacity(bytes.len() + new_code.len().saturating_sub(code.len()));
        out.extend_from_slice(&bytes[..code_len_pos]);
        out.extend_from_slice(&(new_code.len() as u32).to_le_bytes());
        out.extend_from_slice(&new_code);
        out.extend_from_slice(&bytes[code_end..]);
        out
    }

    #[test]
    fn j1_write_does_not_conflict_with_the_borrow_it_is_about_to_activate() {
        let src = r#"
            fn main() {
                let mut x: i32 = 1;
                x = 2;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let bytes = rewrite_main_with_self_referencing_store_var_borrow(bytes);
        run_semcode(&bytes).expect(
            "a StoreVar's own write must not conflict with the Borrow it is about to activate \
             at the identical PC - the write check must run before activation, using the \
             active-borrow set as it stood BEFORE this instruction",
        );
    }

    #[test]
    fn j2_later_write_conflicts_once_the_self_activated_borrow_is_active() {
        let src = r#"
            fn main() {
                let mut x: i32 = 1;
                x = 2;
                x = 3;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        let bytes = rewrite_main_with_self_referencing_store_var_borrow(bytes);
        let err = run_semcode(&bytes).expect_err(
            "once `x = 2;` has committed and activated its own same-PC Borrow, the LATER, \
             independent `x = 3;` write must now conflict - proving the Borrow really did \
             activate after the first write, not merely that the first write was never checked",
        );
        assert!(matches!(
            err,
            RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict)
        ));
    }

    fn rewrite_adt_main_ownership_section(
        bytes: Vec<u8>,
        borrowed_adt: Option<(u32, u16)>,
        write_adt: Option<(u32, u16)>,
    ) -> Vec<u8> {
        let mut cursor = 8usize;
        let name_len = read_u16_le(&bytes, &mut cursor).expect("name len") as usize;
        let name = read_utf8(&bytes, &mut cursor, name_len).expect("name");
        assert_eq!(name, "main");
        let code_len_pos = cursor;
        let code_len = read_u32_le(&bytes, &mut cursor).expect("code len") as usize;
        let code_start = cursor;
        let code_end = code_start + code_len;
        let code = &bytes[code_start..code_end];
        let (header, mut decoded_functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let header_rev = header.rev;
        let env = decoded_functions.remove(0);
        // #1891 Checkpoint W2F: harvested from the REAL (pre-rewrite)
        // compiled artifact's own genuine `StoreVarSite` Write event (minted
        // unconditionally by the current compiler for `e += 2.0;` below,
        // per Checkpoints W2A/W2C) - see `real_store_var_anchor`'s own doc
        // comment for why any genuine StoreVar boundary satisfies
        // `check_write_execution_site` regardless of which local it belongs
        // to.
        let real_anchor = real_store_var_anchor(&env);
        let strings = env.strings;
        let instr_start = env.instr_start_offset;
        // #1773 (FA-09-005): see the identical fix (and rationale) in
        // `rewrite_main_ownership_section` above - a real SIG0 section now
        // sits between OWN0 and the instruction stream, so the OWN0
        // replacement must stop at OWN0's own end, not `instr_start`.
        let sig0_len = env
            .signature
            .as_ref()
            .map(|sig| SIGNATURE_SECTION_TAG.len() + 2 + sig.families.len())
            .unwrap_or(0);
        let own0_end = instr_start - sig0_len;
        let e_root = resolve_unique_lowered_local_key(&strings, "e");
        let ownership_start = code[..own0_end]
            .windows(OWNERSHIP_SECTION_TAG.len())
            .position(|window| window == OWNERSHIP_SECTION_TAG)
            .expect("OWN0 section");
        let mut new_code = Vec::with_capacity(code.len());
        new_code.extend_from_slice(&code[..ownership_start]);

        let mut out_ownership = Vec::new();
        out_ownership.extend_from_slice(&OWNERSHIP_SECTION_TAG);
        out_ownership.extend_from_slice(&2u16.to_le_bytes());
        append_adt_payload_ownership_event(
            &mut out_ownership,
            OWNERSHIP_EVENT_KIND_BORROW,
            e_root,
            borrowed_adt,
            header_rev,
            real_anchor,
        );
        append_adt_payload_ownership_event(
            &mut out_ownership,
            OWNERSHIP_EVENT_KIND_WRITE,
            e_root,
            write_adt,
            header_rev,
            real_anchor,
        );

        new_code.extend_from_slice(&out_ownership);
        new_code.extend_from_slice(&code[own0_end..]);

        let mut out = Vec::with_capacity(bytes.len() + new_code.len().saturating_sub(code.len()));
        out.extend_from_slice(&bytes[..code_len_pos]);
        out.extend_from_slice(&(new_code.len() as u32).to_le_bytes());
        out.extend_from_slice(&new_code);
        out.extend_from_slice(&bytes[code_end..]);
        out
    }

    /// #1891 Checkpoint W2F: `real_anchor` must be a genuine StoreVar PC
    /// harvested from this same function's own real compile
    /// (`real_store_var_anchor`) - Checkpoint W2F's runtime now looks up
    /// Write paths by exact PC (`check_write_execution_site`), so a
    /// placeholder like `0` would either miss entirely or, worse,
    /// coincidentally collide with an unrelated real instruction's PC. Any
    /// genuine StoreVar boundary works: this checkpoint's own trust
    /// boundary never requires the anchor's instruction to correspond to
    /// the synthetic path's root/component (see `WriteExecutionSite`'s doc
    /// comment).
    fn append_adt_payload_ownership_event(
        out: &mut Vec<u8>,
        kind: u8,
        root: u32,
        adt: Option<(u32, u16)>,
        header_rev: u16,
        real_anchor: u32,
    ) {
        out.push(kind);
        if header_rev >= SEMCODE_OWNERSHIP_ANCHOR_MIN_REVISION {
            if kind == OWNERSHIP_EVENT_KIND_BORROW {
                out.push(ACTIVATION_MODE_FRAME_ENTRY);
            } else if kind == OWNERSHIP_EVENT_KIND_WRITE {
                out.push(WRITE_EXECUTION_MODE_STORE_VAR_SITE);
                out.extend_from_slice(&real_anchor.to_le_bytes());
            }
        }
        out.extend_from_slice(&root.to_le_bytes());
        match adt {
            Some((variant, index)) => {
                out.extend_from_slice(&1u16.to_le_bytes());
                out.push(sm_format::semcode_format::OWNERSHIP_PATH_COMPONENT_ADT_PAYLOAD);
                out.extend_from_slice(&variant.to_le_bytes());
                out.extend_from_slice(&index.to_le_bytes());
            }
            None => out.extend_from_slice(&0u16.to_le_bytes()),
        }
    }

    // #1718 reconciliation: before this checkpoint, the five tests below
    // proved the VM's borrow-conflict machinery correctly evaluated
    // `Write(AdtPayload)` overlap (same-payload conflicts, different
    // index/variant don't, parent/child overlaps either direction conflict)
    // via a synthetic, hand-patched artifact (`adt_payload_write_overlap_bytes`,
    // `rewrite_adt_main_ownership_section`). That artifact-level surface is
    // now, correctly, not admitted at all: `Write(AdtPayload)` fails closed
    // at decode/verify regardless of header, before the VM's overlap logic
    // is ever reached (see `docs/roadmap/stable_foundation/ssf08_1718_path_family_contract_decision.md`).
    // The five scenarios below are consolidated into one test proving this
    // rejection is unconditional - it does not matter what the synthetic
    // events' *would-be* overlap relationship was, every one of them is
    // refused identically and for the identical reason. This is exactly the
    // "artifact-level Write(AdtPayload) must now fail before execution"
    // requirement, using an even more adversarial construction than a
    // hand-built decode fixture: a *rewritten* real artifact, proving the
    // rejection survives this specific downgrade/re-encode path too.
    //
    // The overlap-detection logic itself (`access_paths_overlap`) is still
    // real production code other path families depend on - its correctness
    // for `AdtPayload`'s specific shape is now proven directly, at the Rust
    // level, by `access_paths_overlap_adt_payload_mechanism_tests` below,
    // entirely independent of whether `AdtPayload` is an admitted artifact
    // surface for any particular event kind.
    #[test]
    fn vm_rejects_synthetic_adt_payload_write_artifact_regardless_of_overlap_shape() {
        type AdtPayloadSpec = Option<(u32, u16)>;
        let scenarios: [(AdtPayloadSpec, AdtPayloadSpec); 5] = [
            (Some((42, 0)), Some((42, 0))), // same payload
            (Some((42, 0)), Some((42, 1))), // different index
            (Some((42, 0)), Some((43, 0))), // different variant
            (None, Some((42, 0))),          // parent borrow, child write
            (Some((42, 0)), None),          // child borrow, parent write
        ];
        for (borrowed_adt, write_adt) in scenarios {
            let bytes = adt_payload_write_overlap_bytes(borrowed_adt, write_adt);
            let err = run_semcode(&bytes).expect_err(
                "a Write(AdtPayload) event must be rejected before execution regardless of \
                 its overlap relationship with any Borrow event",
            );
            assert!(
                matches!(err, RuntimeError::BadFormat(_)),
                "expected artifact-level rejection (BadFormat), got {err:?} for \
                 borrowed={borrowed_adt:?} write={write_adt:?}"
            );
        }
    }

    // #1718: internal overlap-mechanism tests only - these call
    // `access_paths_overlap`/`ensure_write_path_allowed` directly, at the
    // Rust level, with no SemCode encoding/decoding/verification involved
    // at all. They prove the overlap math itself still correctly handles
    // `PathComponent::AdtPayload`'s equality-based shape (same reasoning
    // `#1725` already established for `Field`'s opaque `SymbolId` treatment)
    // - they are NOT evidence that `Write(AdtPayload)` is an admitted
    // artifact surface, which the test above proves it is not.
    mod access_paths_overlap_adt_payload_mechanism_tests {
        use super::*;

        fn adt_payload_path(root: u32, variant: u32, index: u16) -> AccessPath {
            AccessPath::new(SymbolId(root)).adt_payload(SymbolId(variant), index)
        }

        #[test]
        fn same_payload_overlaps() {
            let borrow = adt_payload_path(1, 42, 0);
            let write = adt_payload_path(1, 42, 0);
            assert!(access_paths_overlap(&write, &borrow));
            assert!(matches!(
                ensure_write_path_allowed(&write, &[borrow]),
                Err(RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict))
            ));
        }

        #[test]
        fn different_index_does_not_overlap() {
            let borrow = adt_payload_path(1, 42, 0);
            let write = adt_payload_path(1, 42, 1);
            assert!(!access_paths_overlap(&write, &borrow));
            assert_eq!(ensure_write_path_allowed(&write, &[borrow]), Ok(()));
        }

        #[test]
        fn different_variant_does_not_overlap() {
            let borrow = adt_payload_path(1, 42, 0);
            let write = adt_payload_path(1, 43, 0);
            assert!(!access_paths_overlap(&write, &borrow));
            assert_eq!(ensure_write_path_allowed(&write, &[borrow]), Ok(()));
        }

        #[test]
        fn parent_borrow_overlaps_child_write() {
            let borrow = AccessPath::new(SymbolId(1)); // parent: root only
            let write = adt_payload_path(1, 42, 0); // child: root.AdtPayload
            assert!(access_paths_overlap(&write, &borrow));
            assert!(matches!(
                ensure_write_path_allowed(&write, &[borrow]),
                Err(RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict))
            ));
        }

        #[test]
        fn child_borrow_overlaps_parent_write() {
            let borrow = adt_payload_path(1, 42, 0); // child: root.AdtPayload
            let write = AccessPath::new(SymbolId(1)); // parent: root only
            assert!(access_paths_overlap(&write, &borrow));
            assert!(matches!(
                ensure_write_path_allowed(&write, &[borrow]),
                Err(RuntimeError::Trap(RuntimeTrap::BorrowWriteConflict))
            ));
        }
    }

    fn record_field_write_overlap_bytes(
        borrowed_field: Option<&str>,
        write_field: Option<&str>,
    ) -> Vec<u8> {
        let src = r#"
            fn main() {
                let camera: f64 = 0.0;
                let quality: f64 = 1.0;
                let ctx: f64 = 1.0;
                ctx += 2.0;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        rewrite_record_main_ownership_section(bytes, borrowed_field, write_field)
    }

    fn rewrite_record_main_ownership_section(
        bytes: Vec<u8>,
        borrowed_field: Option<&str>,
        write_field: Option<&str>,
    ) -> Vec<u8> {
        let mut cursor = 8usize;
        let name_len = read_u16_le(&bytes, &mut cursor).expect("name len") as usize;
        let name = read_utf8(&bytes, &mut cursor, name_len).expect("name");
        assert_eq!(name, "main");
        let code_len_pos = cursor;
        let code_len = read_u32_le(&bytes, &mut cursor).expect("code len") as usize;
        let code_start = cursor;
        let code_end = code_start + code_len;
        let code = &bytes[code_start..code_end];
        let (header, mut decoded_functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let header_rev = header.rev;
        let env = decoded_functions.remove(0);
        // #1891 Checkpoint W2F: see the identical rationale on
        // `append_adt_payload_ownership_event` - harvested from `ctx += 2.0;`
        // below's own real StoreVarSite Write event.
        let real_anchor = real_store_var_anchor(&env);
        let strings = env.strings;
        let instr_start = env.instr_start_offset;
        // #1773 (FA-09-005): see the identical fix (and rationale) in
        // `rewrite_main_ownership_section` above.
        let sig0_len = env
            .signature
            .as_ref()
            .map(|sig| SIGNATURE_SECTION_TAG.len() + 2 + sig.families.len())
            .unwrap_or(0);
        let own0_end = instr_start - sig0_len;
        let ctx_root = resolve_unique_lowered_local_key(&strings, "ctx");
        let ownership_start = code[..own0_end]
            .windows(OWNERSHIP_SECTION_TAG.len())
            .position(|window| window == OWNERSHIP_SECTION_TAG)
            .expect("OWN0 section");
        let mut new_code = Vec::with_capacity(code.len());
        new_code.extend_from_slice(&code[..ownership_start]);
        new_code.extend_from_slice(&record_field_ownership_section_bytes(
            ctx_root,
            &strings,
            borrowed_field,
            write_field,
            header_rev,
            real_anchor,
        ));
        new_code.extend_from_slice(&code[own0_end..]);

        let mut out = Vec::with_capacity(bytes.len() + new_code.len().saturating_sub(code.len()));
        out.extend_from_slice(&bytes[..code_len_pos]);
        out.extend_from_slice(&(new_code.len() as u32).to_le_bytes());
        out.extend_from_slice(&new_code);
        out.extend_from_slice(&bytes[code_end..]);
        out
    }

    fn record_field_ownership_section_bytes(
        root: u32,
        strings: &[String],
        borrowed_field: Option<&str>,
        write_field: Option<&str>,
        header_rev: u16,
        real_anchor: u32,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&OWNERSHIP_SECTION_TAG);
        out.extend_from_slice(&2u16.to_le_bytes());
        append_record_field_ownership_event(
            &mut out,
            OWNERSHIP_EVENT_KIND_BORROW,
            root,
            strings,
            borrowed_field,
            header_rev,
            real_anchor,
        );
        append_record_field_ownership_event(
            &mut out,
            OWNERSHIP_EVENT_KIND_WRITE,
            root,
            strings,
            write_field,
            header_rev,
            real_anchor,
        );
        out
    }

    /// #1891 Checkpoint W2F: see `append_adt_payload_ownership_event`'s
    /// identical rationale for `real_anchor`.
    fn append_record_field_ownership_event(
        out: &mut Vec<u8>,
        kind: u8,
        root: u32,
        strings: &[String],
        field: Option<&str>,
        header_rev: u16,
        real_anchor: u32,
    ) {
        out.push(kind);
        if header_rev >= SEMCODE_OWNERSHIP_ANCHOR_MIN_REVISION {
            if kind == OWNERSHIP_EVENT_KIND_BORROW {
                out.push(ACTIVATION_MODE_FRAME_ENTRY);
            } else if kind == OWNERSHIP_EVENT_KIND_WRITE {
                out.push(WRITE_EXECUTION_MODE_STORE_VAR_SITE);
                out.extend_from_slice(&real_anchor.to_le_bytes());
            }
        }
        out.extend_from_slice(&root.to_le_bytes());
        match field {
            Some(field_name) => {
                out.extend_from_slice(&1u16.to_le_bytes());
                out.push(OWNERSHIP_PATH_COMPONENT_FIELD_SYMBOL);
                // This fixture's "field" names (`camera`/`quality`) are
                // actually plain local-variable declarations in the
                // synthetic source above, not real record-type fields, so
                // they go through the same lowered-local mangling as any
                // other local under #1724 and need the same resolver.
                let field_symbol = resolve_unique_lowered_local_key(strings, field_name);
                out.extend_from_slice(&field_symbol.to_le_bytes());
            }
            None => out.extend_from_slice(&0u16.to_le_bytes()),
        }
    }

    fn ownership_write_overlap_bytes(
        borrowed_components: &[u16],
        write_components: &[u16],
    ) -> Vec<u8> {
        let src = r#"
            fn main() {
                let total: f64 = 1.0;
                total += 2.0;
                return;
            }
        "#;
        let bytes = compile_program_to_semcode(src).expect("compile");
        rewrite_main_ownership_section(bytes, borrowed_components, write_components)
    }

    fn rewrite_main_ownership_section(
        bytes: Vec<u8>,
        borrowed_components: &[u16],
        write_components: &[u16],
    ) -> Vec<u8> {
        let mut cursor = 8usize;
        let name_len = read_u16_le(&bytes, &mut cursor).expect("name len") as usize;
        let name = read_utf8(&bytes, &mut cursor, name_len).expect("name");
        assert_eq!(name, "main");
        let code_len_pos = cursor;
        let code_len = read_u32_le(&bytes, &mut cursor).expect("code len") as usize;
        let code_start = cursor;
        let code_end = code_start + code_len;
        let code = &bytes[code_start..code_end];
        let (header, mut decoded_functions) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let header_rev = header.rev;
        let env = decoded_functions.remove(0);
        // #1891 Checkpoint W2F: see the identical rationale on
        // `append_adt_payload_ownership_event` - harvested from
        // `total += 2.0;` below's own real StoreVarSite Write event.
        let real_anchor = real_store_var_anchor(&env);
        let strings = env.strings;
        let instr_start = env.instr_start_offset;
        // #1773 (FA-09-005): a real SIG0 section now sits between OWN0 and
        // the instruction stream (mandatory once the compiled header
        // reaches SEMCODE_SIGNATURE_MIN_REVISION), so the OWN0 replacement
        // below must stop at OWN0's own end - not `instr_start` - or it
        // silently drops SIG0 along with the OWN0 bytes it means to
        // replace, corrupting the artifact its own header still claims to
        // carry a signature for.
        let sig0_len = env
            .signature
            .as_ref()
            .map(|sig| SIGNATURE_SECTION_TAG.len() + 2 + sig.families.len())
            .unwrap_or(0);
        let own0_end = instr_start - sig0_len;
        let total_root = resolve_unique_lowered_local_key(&strings, "total");
        let ownership_start = code[..own0_end]
            .windows(OWNERSHIP_SECTION_TAG.len())
            .position(|window| window == OWNERSHIP_SECTION_TAG)
            .expect("OWN0 section");
        let mut new_code = Vec::with_capacity(code.len());
        new_code.extend_from_slice(&code[..ownership_start]);
        new_code.extend_from_slice(&ownership_section_bytes(
            total_root,
            borrowed_components,
            write_components,
            header_rev,
            real_anchor,
        ));
        new_code.extend_from_slice(&code[own0_end..]);

        let mut out = Vec::with_capacity(bytes.len() + new_code.len().saturating_sub(code.len()));
        out.extend_from_slice(&bytes[..code_len_pos]);
        out.extend_from_slice(&(new_code.len() as u32).to_le_bytes());
        out.extend_from_slice(&new_code);
        out.extend_from_slice(&bytes[code_end..]);
        out
    }

    fn ownership_section_bytes(
        root: u32,
        borrowed_components: &[u16],
        write_components: &[u16],
        header_rev: u16,
        real_anchor: u32,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&OWNERSHIP_SECTION_TAG);
        out.extend_from_slice(&2u16.to_le_bytes());
        append_ownership_event(
            &mut out,
            OWNERSHIP_EVENT_KIND_BORROW,
            root,
            borrowed_components,
            header_rev,
            real_anchor,
        );
        append_ownership_event(
            &mut out,
            OWNERSHIP_EVENT_KIND_WRITE,
            root,
            write_components,
            header_rev,
            real_anchor,
        );
        out
    }

    /// #1891 Checkpoint W2F: see `append_adt_payload_ownership_event`'s
    /// identical rationale for `real_anchor`.
    fn append_ownership_event(
        out: &mut Vec<u8>,
        kind: u8,
        root: u32,
        components: &[u16],
        header_rev: u16,
        real_anchor: u32,
    ) {
        out.push(kind);
        if header_rev >= SEMCODE_OWNERSHIP_ANCHOR_MIN_REVISION {
            if kind == OWNERSHIP_EVENT_KIND_BORROW {
                out.push(ACTIVATION_MODE_FRAME_ENTRY);
            } else if kind == OWNERSHIP_EVENT_KIND_WRITE {
                out.push(WRITE_EXECUTION_MODE_STORE_VAR_SITE);
                out.extend_from_slice(&real_anchor.to_le_bytes());
            }
        }
        out.extend_from_slice(&root.to_le_bytes());
        out.extend_from_slice(&(components.len() as u16).to_le_bytes());
        for index in components {
            out.push(OWNERSHIP_PATH_COMPONENT_TUPLE_INDEX);
            out.extend_from_slice(&index.to_le_bytes());
        }
    }

    fn legacy_qnot(a: QuadVal) -> QuadVal {
        let v = quad_to_u8(a);
        let r = ((v & 0b10) >> 1) | ((v & 0b01) << 1);
        u8_to_quad(r)
    }

    fn legacy_qand(a: QuadVal, b: QuadVal) -> QuadVal {
        u8_to_quad(quad_to_u8(a) & quad_to_u8(b))
    }

    fn legacy_qor(a: QuadVal, b: QuadVal) -> QuadVal {
        u8_to_quad(quad_to_u8(a) | quad_to_u8(b))
    }

    fn legacy_qimpl(a: QuadVal, b: QuadVal) -> QuadVal {
        legacy_qor(legacy_qnot(a), b)
    }

    const ALL_QUADS: [QuadVal; 4] = [QuadVal::N, QuadVal::F, QuadVal::T, QuadVal::S];

    #[test]
    fn vm_quad_lattice_bridge_qnot_matches_legacy_truth() {
        for a in ALL_QUADS {
            assert_eq!(quad_not(a), legacy_qnot(a), "QNot mismatch for {:?}", a);
        }
    }

    #[test]
    fn vm_quad_lattice_bridge_qand_matches_legacy_truth() {
        for a in ALL_QUADS {
            for b in ALL_QUADS {
                assert_eq!(
                    quad_and(a, b),
                    legacy_qand(a, b),
                    "QAnd mismatch for {:?}, {:?}",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn vm_quad_lattice_bridge_qor_matches_legacy_truth() {
        for a in ALL_QUADS {
            for b in ALL_QUADS {
                assert_eq!(
                    quad_or(a, b),
                    legacy_qor(a, b),
                    "QOr mismatch for {:?}, {:?}",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn vm_quad_lattice_bridge_qimpl_matches_legacy_truth() {
        for a in ALL_QUADS {
            for b in ALL_QUADS {
                assert_eq!(
                    quad_implies(a, b),
                    legacy_qimpl(a, b),
                    "QImpl mismatch for {:?}, {:?}",
                    a,
                    b
                );
            }
        }
    }

    // #1746 (FA-07-006): mirrors sm-verify's own regression coverage for the
    // same structural invariant, exercised here through `run_semcode` - the
    // raw path that calls `parse_semcode`/`validate_function_bytecode`
    // directly without going through `sm-verify` admission at all, so this
    // check is this path's only structural gate against a malformed debug
    // pc, not merely a redundant defense-in-depth mirror.
    #[test]
    fn vm_rejects_debug_pc_pointing_into_operand_byte() {
        let src = r#"
            fn main() {
                let x: bool = true;
                return;
            }
        "#;
        let bytes = sm_emit::compile_program_to_semcode_with_options_debug(
            src,
            sm_emit::CompileProfile::RustLike,
            sm_emit::OptLevel::O0,
            true,
        )
        .expect("compile with debug symbols");

        let entry_offset = {
            let (_, functions) =
                sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
            let f = &functions[0];
            assert_eq!(
                f.debug_symbols.iter().map(|s| s.pc).collect::<Vec<_>>(),
                vec![0, 4, 9],
                "fixture must have real instruction-start debug pcs before mutation"
            );
            f.code_offset + f.string_table_end_offset + 4 + 2
        };
        let mut mutated = bytes.clone();
        // First debug entry's pc: overwrite 0 -> 1, landing inside
        // LOAD_BOOL's 2-byte `dst` operand (instruction 0 spans bytes 0..4),
        // not on any instruction start (0, 4, 9).
        mutated[entry_offset..entry_offset + 4].copy_from_slice(&1u32.to_le_bytes());

        let err =
            run_semcode(&mutated).expect_err("debug pc pointing into an operand byte must reject");
        assert!(
            matches!(err, RuntimeError::BadFormat(ref msg) if msg.contains("debug pc not on an instruction boundary")),
            "unexpected error: {err:?}"
        );
    }

    // #1754 (FA-07-014): cross-check that sm-verify's program-wide distinct
    // runtime symbol count (union of every function's local strings,
    // deduped by exact value) matches what `build_vm_program_view_from_decoded`
    // actually builds into `RuntimeSymbolTable`, for a representative
    // multi-function artifact that reuses some strings across functions.
    // No new production API: `build_vm_program_view_from_decoded` is an
    // existing private fn in this same module, called directly from this
    // in-crate test.
    #[test]
    fn program_wide_symbol_table_matches_verifier_distinct_count() {
        use sm_ir::{IrFunction, IrInstr};

        // f0: a,b,c ; f1: b,c,d (b,c shared with f0) ; f2: d,e (d shared with f1)
        let make = |name: &str, strings: &[&str]| IrFunction {
            name: name.to_string(),
            instrs: {
                let mut instrs: Vec<IrInstr> = strings
                    .iter()
                    .map(|s| IrInstr::LoadText {
                        dst: 0,
                        val: s.to_string(),
                    })
                    .collect();
                instrs.push(IrInstr::Ret { src: None });
                instrs
            },
            ownership_events: Vec::new(),
            params: Vec::new(),
        };
        let funcs = vec![
            make("f0", &["a", "b", "c"]),
            make("f1", &["b", "c", "d"]),
            make("f2", &["d", "e"]),
        ];
        let bytes = sm_emit::emit_ir_to_semcode(&funcs, false).expect("emit");

        // sm-verify must accept it (well under quota).
        sm_verify::verify_semcode_token(&bytes).expect("small duplicate-string program admits");

        // The verifier's counting algorithm, mirrored here: union of every
        // function's local strings, deduped by value.
        let (header, decoded) =
            sm_format::semcode_decode::decode_semcode_envelope(&bytes).expect("decode");
        let verifier_style_distinct = decoded
            .iter()
            .flat_map(|env| env.strings.iter().map(|s| s.as_str()))
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(verifier_style_distinct, 5); // a, b, c, d, e

        // The REAL VM resource, built the same way execution builds it.
        let program =
            build_vm_program_view_from_decoded(header, &decoded).expect("build vm program view");
        assert_eq!(program.runtime_symbols.len(), verifier_style_distinct);
    }

    // --- #1775 (FA-09-007, umbrella #1617) regression matrix ---------------
    //
    // The host ABI boundary (`PrometheusHostAbi` -> `Value` via
    // `value_from_abi`/`quad_from_abi`) must reject a raw host `AbiValue::
    // Quad(u8)` byte outside the canonical `0..=3` domain with a
    // deterministic `RuntimeError::HostAbi` error, never silently mask it
    // (e.g. `& 0b11`) into a plausible-but-wrong valid quad value.
    //
    // Pre-fix (confirmed RED against the original `u8_to_quad(q)` masking
    // used directly in `value_from_abi`, before `quad_from_abi` existed):
    // `AbiValue::Quad(4)` -> `4 & 3 == 0` -> `QuadVal::N`, and
    // `AbiValue::Quad(0xff)` -> `0xff & 3 == 3` -> `QuadVal::S`; both
    // executions completed successfully with no error at all.

    /// Runs a single-function `main` program against a real
    /// `PrometheusHostAbi` + `CapabilityChecker` pair and returns `main`'s
    /// terminal `Value`, unlike the public `run_verified_semcode_with_host_
    /// and_capabilities*` family (which discards the return value and only
    /// reports success/failure).
    fn run_prometheus_program_capturing_value<H: PrometheusHostAbi, C: CapabilityChecker>(
        instrs: Vec<IrInstr>,
        host: &mut H,
        capabilities: &C,
    ) -> Result<Value, RuntimeError> {
        let bytes = emit_ir_to_semcode(
            &[IrFunction {
                name: "main".to_string(),
                instrs,
                ownership_events: Vec::new(),
                params: Vec::new(),
            }],
            false,
        )
        .expect("emit #1775 test program");
        let config = ExecutionConfig::for_context(ExecutionContext::KernelBound);
        let token = verify_semcode_token_with_quotas(&bytes, config.quotas)
            .expect("#1775 test program must admit");
        let entry_token = token.require_entry("main").expect("main entry");
        let program = prepare_verified_execution(&entry_token).expect("prepare");
        let mut vm = VM {
            functions: program.functions,
            callstack: Vec::new(),
            config,
            effect_calls: 0,
            steps: 0,
            calls: 0,
            symbols: program.runtime_symbols,
            prng_state: 0,
        };
        push_frame(&mut vm, entry_token.entry(), Vec::new(), None).expect("push main frame");
        let mut bridge = PrometheusVmHost { host, capabilities };
        let mut observation = HelloObservationRuntime::discard();
        exec_loop(&mut vm, &mut bridge, &mut observation)
    }

    /// A `GateRead` into r0 followed by `Ret r0`: the real opcode/host-call
    /// path that turns a host-returned `AbiValue` into a VM `Value`.
    fn gate_read_into_return_program() -> Vec<IrInstr> {
        vec![
            IrInstr::GateRead {
                dst: 0,
                device_id: 0,
                port: 0,
            },
            IrInstr::Ret { src: Some(0) },
        ]
    }

    fn gate_read_capabilities() -> prom_cap::CapabilityManifest {
        let mut manifest = prom_cap::CapabilityManifest::new();
        manifest.allow(prom_cap::CapabilityKind::GateRead);
        manifest
    }

    /// A5.1-A5.4: canonical raw bytes 0/1/2/3 still succeed end-to-end
    /// through the real `GateRead` opcode and decode to exactly N/F/T/S,
    /// unchanged from before this fix.
    #[test]
    fn gate_read_admits_every_canonical_quad_byte() {
        for (raw, expected) in [
            (0u8, QuadVal::N),
            (1u8, QuadVal::F),
            (2u8, QuadVal::T),
            (3u8, QuadVal::S),
        ] {
            let mut host = prom_abi::RecordingHostAbi::with_read_value(AbiValue::Quad(raw));
            let capabilities = gate_read_capabilities();
            let result = run_prometheus_program_capturing_value(
                gate_read_into_return_program(),
                &mut host,
                &capabilities,
            )
            .unwrap_or_else(|err| panic!("raw quad byte {raw} must still be admitted: {err:?}"));
            assert_eq!(result, Value::Quad(expected), "mismatch for raw byte {raw}");
        }
    }

    /// A5.5-A5.7 (the core #1775 regression): raw bytes 4, 5, and 0xff are
    /// outside the canonical quad domain and must now be rejected with a
    /// deterministic `RuntimeError::HostAbi` error instead of being masked
    /// (pre-fix: `4 & 3 == 0` -> silently `N`; `5 & 3 == 1` -> silently `F`;
    /// `0xff & 3 == 3` -> silently `S`, all with no error).
    #[test]
    fn gate_read_rejects_every_out_of_domain_quad_byte() {
        for raw in [4u8, 5u8, 0xffu8] {
            let mut host = prom_abi::RecordingHostAbi::with_read_value(AbiValue::Quad(raw));
            let capabilities = gate_read_capabilities();
            let err = run_prometheus_program_capturing_value(
                gate_read_into_return_program(),
                &mut host,
                &capabilities,
            )
            .expect_err(&format!(
                "out-of-domain raw quad byte {raw} must now be rejected, not masked"
            ));
            match err {
                RuntimeError::HostAbi(abi_err) => {
                    assert_eq!(abi_err.call, HostCallId::GateRead);
                    assert_eq!(abi_err.kind, AbiFailureKind::InvalidInput);
                    assert!(
                        abi_err.message.contains(&raw.to_string()),
                        "error message should name the offending byte {raw}: {}",
                        abi_err.message
                    );
                }
                other => panic!("expected RuntimeError::HostAbi for raw byte {raw}, got {other:?}"),
            }
        }
    }

    /// Direct unit-level check of the boundary function's exact matching
    /// table, independent of VM plumbing: 0/1/2/3 map to N/F/T/S and
    /// everything else is rejected (proven exhaustively over all 256 byte
    /// values, not just the issue's representative samples).
    #[test]
    fn quad_from_abi_matches_canonical_domain_exhaustively() {
        for raw in 0u8..=255 {
            match raw {
                0 => assert_eq!(quad_from_abi(raw), Ok(QuadVal::N)),
                1 => assert_eq!(quad_from_abi(raw), Ok(QuadVal::F)),
                2 => assert_eq!(quad_from_abi(raw), Ok(QuadVal::T)),
                3 => assert_eq!(quad_from_abi(raw), Ok(QuadVal::S)),
                other => assert_eq!(quad_from_abi(other), Err(other)),
            }
        }
    }

    /// Ordinary non-`Quad` `AbiValue` variants must be completely
    /// unaffected by the quad-domain validation added at this boundary.
    #[test]
    fn gate_read_of_non_quad_abi_values_is_unaffected() {
        let cases: Vec<(AbiValue, Value)> = vec![
            (AbiValue::Bool(true), Value::Bool(true)),
            (AbiValue::I32(-7), Value::I32(-7)),
            (AbiValue::U32(9), Value::U32(9)),
            (AbiValue::F64(1.5), Value::F64(1.5)),
            (AbiValue::Unit, Value::Unit),
        ];
        for (abi_val, expected) in cases {
            let mut host = prom_abi::RecordingHostAbi::with_read_value(abi_val.clone());
            let capabilities = gate_read_capabilities();
            let result = run_prometheus_program_capturing_value(
                gate_read_into_return_program(),
                &mut host,
                &capabilities,
            )
            .unwrap_or_else(|err| panic!("{abi_val:?} must be unaffected by #1775: {err:?}"));
            assert_eq!(result, expected);
        }
    }

    /// A legitimate `AbiError` returned directly by the host (independent of
    /// any quad-domain issue) must propagate completely unchanged: same
    /// `RuntimeError::HostAbi(..)` wrapping the host's own `AbiError`, not
    /// reinterpreted or swallowed by the new validation.
    #[test]
    fn gate_read_existing_host_error_propagation_is_unchanged() {
        struct FailingHost;
        impl PrometheusHostAbi for FailingHost {
            fn gate_read(&mut self, _device_id: u16, _port: u16) -> Result<AbiValue, AbiError> {
                Err(AbiError::new(
                    HostCallId::GateRead,
                    AbiFailureKind::HostFault,
                    "simulated host fault, unrelated to #1775",
                ))
            }
            fn gate_write(&mut self, _: u16, _: u16, _: AbiValue) -> Result<(), AbiError> {
                unreachable!()
            }
            fn pulse_emit(&mut self, _: &str) -> Result<(), AbiError> {
                unreachable!()
            }
            fn state_query(&mut self, _: &str) -> Result<AbiValue, AbiError> {
                unreachable!()
            }
            fn state_update(&mut self, _: &str, _: AbiValue) -> Result<(), AbiError> {
                unreachable!()
            }
            fn event_post(&mut self, _: &str) -> Result<(), AbiError> {
                unreachable!()
            }
            fn clock_read(&mut self) -> Result<u32, AbiError> {
                unreachable!()
            }
        }

        let mut host = FailingHost;
        let capabilities = gate_read_capabilities();
        let err = run_prometheus_program_capturing_value(
            gate_read_into_return_program(),
            &mut host,
            &capabilities,
        )
        .expect_err("host-returned AbiError must propagate");
        match err {
            RuntimeError::HostAbi(abi_err) => {
                assert_eq!(abi_err.call, HostCallId::GateRead);
                assert_eq!(abi_err.kind, AbiFailureKind::HostFault);
                assert_eq!(abi_err.message, "simulated host fault, unrelated to #1775");
            }
            other => panic!("expected unchanged RuntimeError::HostAbi passthrough, got {other:?}"),
        }
    }

    /// Confirms the second untrusted-boundary call site (`state_query`)
    /// gets the identical repair: canonical bytes still succeed, and an
    /// out-of-domain byte is now rejected with `HostCallId::StateQuery`
    /// context (not `GateRead`'s), proving `call` is threaded through
    /// correctly rather than hardcoded.
    #[test]
    fn state_query_quad_boundary_matches_gate_read() {
        let mut ok_host = prom_abi::RecordingHostAbi::with_state_query_value(AbiValue::Quad(2));
        let mut manifest = prom_cap::CapabilityManifest::new();
        manifest.allow(prom_cap::CapabilityKind::StateQuery);
        let program = vec![
            IrInstr::StateQuery {
                dst: 0,
                key: "k".to_string(),
            },
            IrInstr::Ret { src: Some(0) },
        ];
        let result =
            run_prometheus_program_capturing_value(program.clone(), &mut ok_host, &manifest)
                .expect("canonical raw byte 2 via state_query must succeed");
        assert_eq!(result, Value::Quad(QuadVal::T));

        let mut bad_host = prom_abi::RecordingHostAbi::with_state_query_value(AbiValue::Quad(0xff));
        let err = run_prometheus_program_capturing_value(program, &mut bad_host, &manifest)
            .expect_err("out-of-domain raw byte via state_query must now be rejected");
        match err {
            RuntimeError::HostAbi(abi_err) => {
                assert_eq!(abi_err.call, HostCallId::StateQuery);
                assert_eq!(abi_err.kind, AbiFailureKind::InvalidInput);
            }
            other => panic!("expected RuntimeError::HostAbi, got {other:?}"),
        }
    }
}
