#![allow(clippy::len_without_is_empty)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(any(feature = "alloc", feature = "std"))]
extern crate alloc;

#[cfg(any(feature = "alloc", feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(any(feature = "alloc", feature = "std"))]
use alloc::string::{String, ToString};
#[cfg(any(feature = "alloc", feature = "std"))]
use alloc::vec::Vec;

#[cfg(any(feature = "alloc", feature = "std"))]
pub mod hello_observation_route;
#[cfg(any(feature = "alloc", feature = "std"))]
pub mod hello_observation_sink;

#[cfg(any(feature = "alloc", feature = "std"))]
#[derive(Debug, Clone, PartialEq)]
pub struct RecordCarrier<T> {
    pub type_name: String,
    pub slots: Vec<T>,
}

#[cfg(any(feature = "alloc", feature = "std"))]
#[derive(Debug, Clone, PartialEq)]
pub struct AdtCarrier<T> {
    pub type_name: String,
    pub variant_name: String,
    pub tag: u16,
    pub payload: Vec<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(pub u32);

impl SymbolId {
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AccessPath {
    pub root: SymbolId,
    pub components: Vec<PathComponent>,
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl AccessPath {
    pub fn new(root: SymbolId) -> Self {
        Self {
            root,
            components: Vec::new(),
        }
    }

    pub fn tuple_index(&self, index: u16) -> Self {
        let mut components = self.components.clone();
        components.push(PathComponent::TupleIndex(index));
        Self {
            root: self.root,
            components,
        }
    }

    pub fn sequence_index_static(&self, index: u32) -> Self {
        let mut components = self.components.clone();
        components.push(PathComponent::SequenceIndexStatic(index));
        Self {
            root: self.root,
            components,
        }
    }

    pub fn field(&self, name: SymbolId) -> Self {
        let mut components = self.components.clone();
        components.push(PathComponent::Field(name));
        Self {
            root: self.root,
            components,
        }
    }

    pub fn adt_payload(&self, variant: SymbolId, index: u16) -> Self {
        let mut components = self.components.clone();
        components.push(PathComponent::AdtPayload { variant, index });
        Self {
            root: self.root,
            components,
        }
    }
}

/// #1725 (FA-04-019) follow-up finding, verified by direct inspection of
/// every consumer in `sm-vm`: `Field`'s and `AdtPayload::variant`'s
/// `SymbolId` are used *exclusively* as opaque, root-gated equality keys
/// inside `access_paths_overlap` (`lhs.components == rhs.components` after
/// `lhs.root == rhs.root` already holds) - unlike `AccessPath.root`
/// pre-#1725, they are never resolved through `RuntimeSymbolTable`/
/// `vm.symbols`, never used as an index into any table, and never
/// displayed. Because the overlap check is root-gated first, a field or
/// variant spelling colliding with an unrelated one on a *different* root
/// (e.g. two distinct record types both declaring a `value` field) can
/// never produce a false conflict - only same-root component paths are
/// ever compared, and within one root's own record/ADT type every field or
/// variant spelling is unique by construction. This is a materially
/// different, already-sound property from root identity, which #1725
/// repairs because roots *are* resolved through a table. See
/// `field_symbol_collision_across_unrelated_record_types_does_not_false_conflict`
/// in `tests/own0_root_identity_e2e.rs` for the proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PathComponent {
    TupleIndex(u16),
    Field(SymbolId),
    AdtPayload { variant: SymbolId, index: u16 },
    SequenceIndexStatic(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionContext {
    PureCompute,
    VerifiedLocal,
    RuleExecution,
    KernelBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaKind {
    Steps,
    Calls,
    StackDepth,
    Frames,
    Registers,
    SymbolTable,
    EffectCalls,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotaExceeded {
    pub kind: QuotaKind,
    pub limit: usize,
    pub used: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTrap {
    AssertionFailed,
    BorrowWriteConflict,
    StackOverflow,
    StackUnderflow,
    TypeMismatch,
    InvalidOpcode,
    InvalidJump,
    DivisionByZero,
    ArithmeticOverflow,
    CapabilityDenied,
    AbiViolation,
    VerifierRejected,
    QuotaExceeded(QuotaExceeded),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeQuotas {
    pub max_steps: usize,
    pub max_calls: usize,
    pub max_stack_depth: usize,
    pub max_frames: usize,
    pub max_registers: usize,
    pub max_symbol_table: usize,
    pub max_effect_calls: usize,
    /// Per-function debug-symbol-table admission limit, enforced by
    /// `sm-verify::verify_function_code` against `env.debug_symbols.len()`.
    /// Not a `QuotaKind` member: never charged by `sm-vm`, never surfaces
    /// as `RuntimeError::QuotaExceeded`. Kept on `RuntimeQuotas` only for
    /// compatibility with existing per-profile admission plumbing (#1760).
    pub max_debug_symbols_per_function: usize,
}

impl RuntimeQuotas {
    pub const fn verified_local() -> Self {
        Self {
            max_steps: 100_000,
            max_calls: 16_384,
            max_stack_depth: 256,
            max_frames: 256,
            max_registers: 4_096,
            max_symbol_table: 16_384,
            max_effect_calls: 1_024,
            max_debug_symbols_per_function: 8_192,
        }
    }

    pub const fn pure_compute() -> Self {
        Self {
            max_steps: 100_000,
            max_calls: 16_384,
            max_stack_depth: 256,
            max_frames: 256,
            max_registers: 4_096,
            max_symbol_table: 16_384,
            max_effect_calls: 0,
            max_debug_symbols_per_function: 4_096,
        }
    }

    pub const fn kernel_bound() -> Self {
        Self {
            max_steps: 250_000,
            max_calls: 32_768,
            max_stack_depth: 256,
            max_frames: 256,
            max_registers: 8_192,
            max_symbol_table: 16_384,
            max_effect_calls: 4_096,
            max_debug_symbols_per_function: 16_384,
        }
    }

    pub fn exceed(self, kind: QuotaKind, used: usize) -> Option<QuotaExceeded> {
        let limit = match kind {
            QuotaKind::Steps => self.max_steps,
            QuotaKind::Calls => self.max_calls,
            QuotaKind::StackDepth => self.max_stack_depth,
            QuotaKind::Frames => self.max_frames,
            QuotaKind::Registers => self.max_registers,
            QuotaKind::SymbolTable => self.max_symbol_table,
            QuotaKind::EffectCalls => self.max_effect_calls,
        };
        (used > limit).then_some(QuotaExceeded { kind, limit, used })
    }
}

impl Default for RuntimeQuotas {
    fn default() -> Self {
        Self::verified_local()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[cfg(any(feature = "alloc", feature = "std"))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeSymbolTable {
    ordered: Vec<String>,
    index: BTreeMap<String, SymbolId>,
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl RuntimeSymbolTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, name: &str) -> SymbolId {
        if let Some(id) = self.index.get(name) {
            return *id;
        }
        let id = SymbolId(self.ordered.len() as u32);
        self.ordered.push(name.to_string());
        self.index.insert(name.to_string(), id);
        id
    }

    pub fn resolve(&self, id: SymbolId) -> Option<&str> {
        self.ordered.get(id.0 as usize).map(|name| name.as_str())
    }

    pub fn len(&self) -> usize {
        self.ordered.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ordered.is_empty()
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DebugNameMap {
    names: Vec<String>,
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl DebugNameMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, name: &str) -> SymbolId {
        let id = SymbolId(self.names.len() as u32);
        self.names.push(name.to_string());
        id
    }

    pub fn resolve(&self, id: SymbolId) -> Option<&str> {
        self.names.get(id.0 as usize).map(|name| name.as_str())
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn access_path_root_starts_with_empty_component_list() {
        let path = AccessPath::new(SymbolId(7));
        assert_eq!(path.root, SymbolId(7));
        assert!(path.components.is_empty());
    }

    #[test]
    fn access_path_tuple_indices_preserve_append_order() {
        let path = AccessPath::new(SymbolId(3)).tuple_index(1).tuple_index(4);
        assert_eq!(path.root, SymbolId(3));
        assert_eq!(
            path.components,
            vec![PathComponent::TupleIndex(1), PathComponent::TupleIndex(4)]
        );
    }

    #[test]
    fn access_path_record_field_can_be_represented() {
        let camera = SymbolId(11);
        let path = AccessPath::new(SymbolId(3)).field(camera);
        assert_eq!(path.root, SymbolId(3));
        assert_eq!(path.components, vec![PathComponent::Field(camera)]);
    }

    #[test]
    fn access_path_component_order_is_deterministic() {
        let field = SymbolId(12);
        let left = AccessPath::new(SymbolId(9)).tuple_index(0).field(field);
        let right = AccessPath::new(SymbolId(9)).tuple_index(0).field(field);
        let different = AccessPath::new(SymbolId(9)).field(field).tuple_index(0);
        assert_eq!(left, right);
        assert_ne!(left, different);

        let adt = AccessPath::new(SymbolId(9)).adt_payload(SymbolId(42), 0);
        let adt_diff_idx = AccessPath::new(SymbolId(9)).adt_payload(SymbolId(42), 1);
        let adt_diff_var = AccessPath::new(SymbolId(9)).adt_payload(SymbolId(43), 0);
        assert_eq!(
            adt,
            AccessPath::new(SymbolId(9)).adt_payload(SymbolId(42), 0)
        );
        assert_ne!(adt, adt_diff_idx);
        assert_ne!(adt, adt_diff_var);
        assert_ne!(adt, left);

        let sequence = AccessPath::new(SymbolId(9)).sequence_index_static(2);
        let same_sequence = AccessPath::new(SymbolId(9)).sequence_index_static(2);
        let different_sequence = AccessPath::new(SymbolId(9)).sequence_index_static(4);
        assert_eq!(sequence, same_sequence);
        assert_ne!(sequence, different_sequence);
        assert_ne!(sequence, left);
    }

    #[test]
    fn runtime_symbol_table_assigns_deterministic_ids() {
        let mut table = RuntimeSymbolTable::new();
        let alpha = table.intern("alpha");
        let beta = table.intern("beta");
        let alpha_again = table.intern("alpha");

        assert_eq!(alpha, SymbolId(0));
        assert_eq!(beta, SymbolId(1));
        assert_eq!(alpha_again, alpha);
        assert_eq!(table.resolve(beta), Some("beta"));
    }

    #[test]
    fn debug_name_map_is_append_only_and_stable() {
        let mut names = DebugNameMap::new();
        let first = names.push("main");
        let second = names.push("helper");

        assert_eq!(first, SymbolId(0));
        assert_eq!(second, SymbolId(1));
        assert_eq!(names.resolve(first), Some("main"));
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn verified_local_quotas_keep_stack_depth_at_contract_value() {
        let quotas = RuntimeQuotas::verified_local();
        assert_eq!(quotas.max_stack_depth, 256);
        assert_eq!(quotas.max_effect_calls, 1_024);
    }

    /// #1761 (FA-08-003): removing `max_const_pool`/`QuotaKind::ConstPool`
    /// must not change any surviving quota value in any baseline profile.
    /// Pins every remaining field of all three profiles to its exact
    /// pre-removal value, so an accidental baseline change hiding inside
    /// this removal would fail loudly here rather than pass silently.
    #[test]
    fn removing_const_pool_leaves_every_surviving_baseline_value_unchanged() {
        let verified_local = RuntimeQuotas::verified_local();
        assert_eq!(verified_local.max_steps, 100_000);
        assert_eq!(verified_local.max_calls, 16_384);
        assert_eq!(verified_local.max_stack_depth, 256);
        assert_eq!(verified_local.max_frames, 256);
        assert_eq!(verified_local.max_registers, 4_096);
        assert_eq!(verified_local.max_symbol_table, 16_384);
        assert_eq!(verified_local.max_effect_calls, 1_024);
        assert_eq!(verified_local.max_debug_symbols_per_function, 8_192);

        let pure_compute = RuntimeQuotas::pure_compute();
        assert_eq!(pure_compute.max_steps, 100_000);
        assert_eq!(pure_compute.max_calls, 16_384);
        assert_eq!(pure_compute.max_stack_depth, 256);
        assert_eq!(pure_compute.max_frames, 256);
        assert_eq!(pure_compute.max_registers, 4_096);
        assert_eq!(pure_compute.max_symbol_table, 16_384);
        assert_eq!(pure_compute.max_effect_calls, 0);
        assert_eq!(pure_compute.max_debug_symbols_per_function, 4_096);

        let kernel_bound = RuntimeQuotas::kernel_bound();
        assert_eq!(kernel_bound.max_steps, 250_000);
        assert_eq!(kernel_bound.max_calls, 32_768);
        assert_eq!(kernel_bound.max_stack_depth, 256);
        assert_eq!(kernel_bound.max_frames, 256);
        assert_eq!(kernel_bound.max_registers, 8_192);
        assert_eq!(kernel_bound.max_symbol_table, 16_384);
        assert_eq!(kernel_bound.max_effect_calls, 4_096);
        assert_eq!(kernel_bound.max_debug_symbols_per_function, 16_384);
    }

    #[test]
    fn quota_exceed_reports_precise_limit_and_usage() {
        let quotas = RuntimeQuotas::pure_compute();
        let exceed = quotas
            .exceed(QuotaKind::EffectCalls, 1)
            .expect("must exceed");

        assert_eq!(
            exceed,
            QuotaExceeded {
                kind: QuotaKind::EffectCalls,
                limit: 0,
                used: 1,
            }
        );
    }

    #[test]
    fn execution_config_uses_context_defaults() {
        let config = ExecutionConfig::for_context(ExecutionContext::KernelBound);
        assert_eq!(config.context, ExecutionContext::KernelBound);
        assert_eq!(config.quotas.max_effect_calls, 4_096);
    }
}
