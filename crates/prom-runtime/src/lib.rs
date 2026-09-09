#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
use prom_abi::PrometheusHostAbi;
use prom_audit::{AuditEventId, AuditEventKind, AuditSessionMetadata, AuditTrail};
use prom_cap::{CapabilityChecker, CapabilityManifestMetadata};
use prom_gates::{GateBinding, GateHostAdapter, GateRegistry};
use prom_rules::{Agenda, AgendaEntry, RuleDefinition, RuleEffect, RuleEngine, RuleId};
use prom_state::{
    ContextWindow, FactResolution, SemanticStateStore, StateEpoch, StateTransitionMetadata,
    StateUpdate, StateValidationError,
};
use sm_runtime_core::{ExecutionConfig, ExecutionContext, RuntimeQuotas};
use sm_verify::{verify_semcode_token_with_quotas, EntryResolutionError};
use sm_vm::{run_verified_entry_semcode_with_host_and_capabilities_and_config, RuntimeError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSessionDescriptor {
    pub context: ExecutionContext,
    pub quotas: RuntimeQuotas,
    pub capability_manifest: CapabilityManifestMetadata,
    pub gate_registry_bound: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeIntegrationSnapshot {
    pub session: RuntimeSessionDescriptor,
    pub state_epoch: StateEpoch,
    pub active_rules: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationSelection {
    pub entry: AgendaEntry,
    pub remaining_rules: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeStateAdvance {
    pub transition: StateTransitionMetadata,
    pub agenda: Agenda,
    pub snapshot: RuntimeIntegrationSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleStateWriteAdvance {
    pub rule_id: RuleId,
    pub effect_ordinal: usize,
    pub advance: RuntimeStateAdvance,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleAuditNoteAdvance {
    pub rule_id: RuleId,
    pub effect_ordinal: usize,
    pub event_id: AuditEventId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleEffectExecutionCode {
    UnsupportedEffectFamily,
    StateValidationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleEffectExecutionError {
    pub code: RuleEffectExecutionCode,
    pub rule_id: RuleId,
    pub effect_ordinal: usize,
    pub message: String,
}

impl RuleEffectExecutionError {
    pub fn new(
        code: RuleEffectExecutionCode,
        rule_id: RuleId,
        effect_ordinal: usize,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            rule_id,
            effect_ordinal,
            message: message.into(),
        }
    }
}

impl core::fmt::Display for RuleEffectExecutionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RuleEffectExecutionError {}

fn build_audit_session(descriptor: &RuntimeSessionDescriptor) -> AuditSessionMetadata {
    AuditSessionMetadata {
        context: descriptor.context,
        quotas: descriptor.quotas,
        capability_manifest: descriptor.capability_manifest.clone(),
        gate_registry_bound: descriptor.gate_registry_bound,
    }
}

fn build_integration_snapshot(
    descriptor: &RuntimeSessionDescriptor,
    state: &SemanticStateStore,
    agenda: &Agenda,
) -> RuntimeIntegrationSnapshot {
    RuntimeIntegrationSnapshot {
        session: descriptor.clone(),
        state_epoch: state.epoch(),
        active_rules: agenda.entries().len(),
    }
}

fn apply_update_refresh_agenda(
    descriptor: &RuntimeSessionDescriptor,
    state: &mut SemanticStateStore,
    update: StateUpdate,
    rules: &RuleEngine,
    trail: &mut AuditTrail,
) -> Result<RuntimeStateAdvance, StateValidationError> {
    let transition = state.apply(update)?;
    trail.record(AuditEventKind::StateTransition {
        key: transition.key.clone(),
        from_epoch: transition.from_epoch.0,
        to_epoch: transition.to_epoch.0,
    });
    let agenda = rules.evaluate(state);
    let snapshot = build_integration_snapshot(descriptor, state, &agenda);
    Ok(RuntimeStateAdvance {
        transition,
        agenda,
        snapshot,
    })
}

fn apply_rule_state_write_effects(
    descriptor: &RuntimeSessionDescriptor,
    state: &mut SemanticStateStore,
    rule: &RuleDefinition,
    rules: &RuleEngine,
    trail: &mut AuditTrail,
) -> Result<Vec<RuleStateWriteAdvance>, RuleEffectExecutionError> {
    let mut advances = Vec::new();

    for (effect_ordinal, effect) in rule.effect_plan().effects().iter().enumerate() {
        let RuleEffect::StateWrite(effect) = effect else {
            return Err(RuleEffectExecutionError::new(
                RuleEffectExecutionCode::UnsupportedEffectFamily,
                rule.id.clone(),
                effect_ordinal,
                format!(
                    "rule '{}' effect {} is not admitted by the current state-write execution slice",
                    rule.id.0, effect_ordinal
                ),
            ));
        };

        let advance = apply_update_refresh_agenda(
            descriptor,
            state,
            StateUpdate::new(
                effect.key.clone(),
                FactResolution::Certain(effect.value.clone()),
                ContextWindow::new(effect.context.clone()),
                effect.reason.clone(),
            ),
            rules,
            trail,
        )
        .map_err(|err| {
            RuleEffectExecutionError::new(
                RuleEffectExecutionCode::StateValidationFailed,
                rule.id.clone(),
                effect_ordinal,
                err.to_string(),
            )
        })?;

        advances.push(RuleStateWriteAdvance {
            rule_id: rule.id.clone(),
            effect_ordinal,
            advance,
        });
    }

    Ok(advances)
}

fn apply_rule_audit_note_effects(
    trail: &mut AuditTrail,
    rule: &RuleDefinition,
) -> Result<Vec<RuleAuditNoteAdvance>, RuleEffectExecutionError> {
    let mut advances = Vec::new();

    for (effect_ordinal, effect) in rule.effect_plan().effects().iter().enumerate() {
        let RuleEffect::AuditNote(effect) = effect else {
            return Err(RuleEffectExecutionError::new(
                RuleEffectExecutionCode::UnsupportedEffectFamily,
                rule.id.clone(),
                effect_ordinal,
                format!(
                    "rule '{}' effect {} is not admitted by the current audit-note execution slice",
                    rule.id.0, effect_ordinal
                ),
            ));
        };

        let event_id = trail.record(AuditEventKind::Note {
            message: effect.message.clone(),
        });
        advances.push(RuleAuditNoteAdvance {
            rule_id: rule.id.clone(),
            effect_ordinal,
            event_id,
        });
    }

    Ok(advances)
}

pub struct ExecutionSession<'a, H: PrometheusHostAbi, C: CapabilityChecker> {
    host: &'a mut H,
    capabilities: &'a C,
    config: ExecutionConfig,
    descriptor: RuntimeSessionDescriptor,
}

impl<'a, H: PrometheusHostAbi, C: CapabilityChecker> ExecutionSession<'a, H, C> {
    pub fn new(
        host: &'a mut H,
        capabilities: &'a C,
        config: ExecutionConfig,
        capability_manifest: CapabilityManifestMetadata,
    ) -> Self {
        Self {
            host,
            capabilities,
            descriptor: RuntimeSessionDescriptor {
                context: config.context,
                quotas: config.quotas,
                capability_manifest,
                gate_registry_bound: false,
            },
            config,
        }
    }

    pub fn kernel_bound(
        host: &'a mut H,
        capabilities: &'a C,
        capability_manifest: CapabilityManifestMetadata,
    ) -> Self {
        Self::new(
            host,
            capabilities,
            ExecutionConfig::for_context(ExecutionContext::KernelBound),
            capability_manifest,
        )
    }

    pub fn descriptor(&self) -> &RuntimeSessionDescriptor {
        &self.descriptor
    }

    pub fn derive_agenda(&self, state: &SemanticStateStore, rules: &RuleEngine) -> Agenda {
        rules.evaluate(state)
    }

    pub fn select_next_activation(&self, agenda: &Agenda) -> Option<ActivationSelection> {
        agenda
            .entries()
            .first()
            .cloned()
            .map(|entry| ActivationSelection {
                entry,
                remaining_rules: agenda.entries().len().saturating_sub(1),
            })
    }

    pub fn begin_audit_trail(&self) -> AuditTrail {
        AuditTrail::new(build_audit_session(&self.descriptor))
    }

    pub fn record_session_started(&self, trail: &mut AuditTrail, entry: &str) -> AuditEventId {
        trail.record(AuditEventKind::SessionStarted {
            entry: entry.into(),
        })
    }

    pub fn record_session_finished(&self, trail: &mut AuditTrail) -> AuditEventId {
        trail.record(AuditEventKind::SessionFinished)
    }

    pub fn record_rule_activation(
        &self,
        trail: &mut AuditTrail,
        selection: &ActivationSelection,
    ) -> AuditEventId {
        trail.record(AuditEventKind::RuleActivated {
            rule_id: selection.entry.rule_id.0.clone(),
            salience: selection.entry.salience.0,
        })
    }

    pub fn record_state_transition(
        &self,
        trail: &mut AuditTrail,
        transition: &StateTransitionMetadata,
    ) -> AuditEventId {
        trail.record(AuditEventKind::StateTransition {
            key: transition.key.clone(),
            from_epoch: transition.from_epoch.0,
            to_epoch: transition.to_epoch.0,
        })
    }

    pub fn integration_snapshot(
        &self,
        state: &SemanticStateStore,
        agenda: &Agenda,
    ) -> RuntimeIntegrationSnapshot {
        build_integration_snapshot(&self.descriptor, state, agenda)
    }

    pub fn apply_state_update_and_refresh_agenda(
        &self,
        state: &mut SemanticStateStore,
        update: StateUpdate,
        rules: &RuleEngine,
        trail: &mut AuditTrail,
    ) -> Result<RuntimeStateAdvance, StateValidationError> {
        apply_update_refresh_agenda(&self.descriptor, state, update, rules, trail)
    }

    pub fn apply_rule_state_write_effects(
        &self,
        state: &mut SemanticStateStore,
        rule: &RuleDefinition,
        rules: &RuleEngine,
        trail: &mut AuditTrail,
    ) -> Result<Vec<RuleStateWriteAdvance>, RuleEffectExecutionError> {
        apply_rule_state_write_effects(&self.descriptor, state, rule, rules, trail)
    }

    pub fn apply_rule_audit_note_effects(
        &self,
        trail: &mut AuditTrail,
        rule: &RuleDefinition,
    ) -> Result<Vec<RuleAuditNoteAdvance>, RuleEffectExecutionError> {
        apply_rule_audit_note_effects(trail, rule)
    }

    /// Public compatibility API accepting bytes.
    ///
    /// Internally routes through verified token admission and VM token execution.
    /// Retained for public compatibility and is not evidence that `sm-vm` canonical execution is byte-first.
    pub fn run_verified_semcode(&mut self, bytes: &[u8]) -> Result<(), RuntimeError> {
        self.run_verified_semcode_entry(bytes, "main")
    }

    /// Public compatibility API accepting bytes.
    ///
    /// Internally routes through verified token admission and VM token execution.
    /// Retained for public compatibility and is not evidence that `sm-vm` canonical execution is byte-first.
    pub fn run_verified_semcode_entry(
        &mut self,
        bytes: &[u8],
        entry: &str,
    ) -> Result<(), RuntimeError> {
        let token = verify_semcode_token_with_quotas(bytes, self.config.quotas)
            .map_err(RuntimeError::VerifierRejected)?;
        let entry_token = token.require_entry(entry).map_err(|err| match err {
            EntryResolutionError::MissingEntry { entry } => RuntimeError::UnknownFunction(entry),
        })?;
        run_verified_entry_semcode_with_host_and_capabilities_and_config(
            &entry_token,
            self.host,
            self.capabilities,
            self.config,
        )
    }
}

pub struct GateExecutionSession<'a, B: GateBinding, C: CapabilityChecker> {
    registry: &'a GateRegistry,
    binding: &'a mut B,
    capabilities: &'a C,
    config: ExecutionConfig,
    descriptor: RuntimeSessionDescriptor,
}

impl<'a, B: GateBinding, C: CapabilityChecker> GateExecutionSession<'a, B, C> {
    pub fn new(
        registry: &'a GateRegistry,
        binding: &'a mut B,
        capabilities: &'a C,
        config: ExecutionConfig,
        capability_manifest: CapabilityManifestMetadata,
    ) -> Self {
        Self {
            registry,
            binding,
            capabilities,
            descriptor: RuntimeSessionDescriptor {
                context: config.context,
                quotas: config.quotas,
                capability_manifest,
                gate_registry_bound: true,
            },
            config,
        }
    }

    pub fn kernel_bound(
        registry: &'a GateRegistry,
        binding: &'a mut B,
        capabilities: &'a C,
        capability_manifest: CapabilityManifestMetadata,
    ) -> Self {
        Self::new(
            registry,
            binding,
            capabilities,
            ExecutionConfig::for_context(ExecutionContext::KernelBound),
            capability_manifest,
        )
    }

    pub fn descriptor(&self) -> &RuntimeSessionDescriptor {
        &self.descriptor
    }

    pub fn derive_agenda(&self, state: &SemanticStateStore, rules: &RuleEngine) -> Agenda {
        rules.evaluate(state)
    }

    pub fn select_next_activation(&self, agenda: &Agenda) -> Option<ActivationSelection> {
        agenda
            .entries()
            .first()
            .cloned()
            .map(|entry| ActivationSelection {
                entry,
                remaining_rules: agenda.entries().len().saturating_sub(1),
            })
    }

    pub fn begin_audit_trail(&self) -> AuditTrail {
        AuditTrail::new(build_audit_session(&self.descriptor))
    }

    pub fn record_session_started(&self, trail: &mut AuditTrail, entry: &str) -> AuditEventId {
        trail.record(AuditEventKind::SessionStarted {
            entry: entry.into(),
        })
    }

    pub fn record_session_finished(&self, trail: &mut AuditTrail) -> AuditEventId {
        trail.record(AuditEventKind::SessionFinished)
    }

    pub fn record_rule_activation(
        &self,
        trail: &mut AuditTrail,
        selection: &ActivationSelection,
    ) -> AuditEventId {
        trail.record(AuditEventKind::RuleActivated {
            rule_id: selection.entry.rule_id.0.clone(),
            salience: selection.entry.salience.0,
        })
    }

    pub fn record_state_transition(
        &self,
        trail: &mut AuditTrail,
        transition: &StateTransitionMetadata,
    ) -> AuditEventId {
        trail.record(AuditEventKind::StateTransition {
            key: transition.key.clone(),
            from_epoch: transition.from_epoch.0,
            to_epoch: transition.to_epoch.0,
        })
    }

    pub fn integration_snapshot(
        &self,
        state: &SemanticStateStore,
        agenda: &Agenda,
    ) -> RuntimeIntegrationSnapshot {
        build_integration_snapshot(&self.descriptor, state, agenda)
    }

    pub fn apply_state_update_and_refresh_agenda(
        &self,
        state: &mut SemanticStateStore,
        update: StateUpdate,
        rules: &RuleEngine,
        trail: &mut AuditTrail,
    ) -> Result<RuntimeStateAdvance, StateValidationError> {
        apply_update_refresh_agenda(&self.descriptor, state, update, rules, trail)
    }

    pub fn apply_rule_state_write_effects(
        &self,
        state: &mut SemanticStateStore,
        rule: &RuleDefinition,
        rules: &RuleEngine,
        trail: &mut AuditTrail,
    ) -> Result<Vec<RuleStateWriteAdvance>, RuleEffectExecutionError> {
        apply_rule_state_write_effects(&self.descriptor, state, rule, rules, trail)
    }

    pub fn apply_rule_audit_note_effects(
        &self,
        trail: &mut AuditTrail,
        rule: &RuleDefinition,
    ) -> Result<Vec<RuleAuditNoteAdvance>, RuleEffectExecutionError> {
        apply_rule_audit_note_effects(trail, rule)
    }

    /// Public compatibility API accepting bytes.
    ///
    /// Internally routes through verified token admission and VM token execution.
    /// Retained for public compatibility and is not evidence that `sm-vm` canonical execution is byte-first.
    pub fn run_verified_semcode(&mut self, bytes: &[u8]) -> Result<(), RuntimeError> {
        self.run_verified_semcode_entry(bytes, "main")
    }

    /// Public compatibility API accepting bytes.
    ///
    /// Internally routes through verified token admission and VM token execution.
    /// Retained for public compatibility and is not evidence that `sm-vm` canonical execution is byte-first.
    pub fn run_verified_semcode_entry(
        &mut self,
        bytes: &[u8],
        entry: &str,
    ) -> Result<(), RuntimeError> {
        let mut host = GateHostAdapter::new(self.registry, self.binding);
        let token = verify_semcode_token_with_quotas(bytes, self.config.quotas)
            .map_err(RuntimeError::VerifierRejected)?;
        let entry_token = token.require_entry(entry).map_err(|err| match err {
            EntryResolutionError::MissingEntry { entry } => RuntimeError::UnknownFunction(entry),
        })?;
        run_verified_entry_semcode_with_host_and_capabilities_and_config(
            &entry_token,
            &mut host,
            self.capabilities,
            self.config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prom_abi::{AbiValue, RecordingHostAbi};
    use prom_cap::CapabilityManifest;
    use prom_gates::{DeterministicGateMock, GateDescriptor, GateId};
    use prom_rules::{RuleCondition, RuleDefinition, RuleEffect, RuleEngine};
    use prom_state::{ContextWindow, FactResolution, FactValue, StateUpdate};

    #[test]
    fn execution_session_descriptor_reports_context_and_manifest() {
        let manifest = CapabilityManifest::gate_surface();
        let metadata = manifest.metadata();
        let mut host = RecordingHostAbi::with_read_value(AbiValue::I32(1));
        let session = ExecutionSession::kernel_bound(&mut host, &manifest, metadata.clone());
        assert_eq!(session.descriptor().context, ExecutionContext::KernelBound);
        assert_eq!(session.descriptor().capability_manifest, metadata);
        assert!(!session.descriptor().gate_registry_bound);
    }

    #[test]
    fn gate_execution_session_descriptor_marks_gate_binding() {
        let manifest = CapabilityManifest::gate_surface();
        let metadata = manifest.metadata();
        let mut registry = GateRegistry::new();
        registry
            .register(GateDescriptor::read_write(7, 4, "gate.alpha"))
            .expect("register");
        let mut binding = DeterministicGateMock::new();
        binding.seed_read(GateId::new(7, 4), AbiValue::I32(1));
        let session =
            GateExecutionSession::kernel_bound(&registry, &mut binding, &manifest, metadata);
        assert!(session.descriptor().gate_registry_bound);
    }

    #[test]
    fn gate_execution_session_derives_agenda_and_audit_without_owning_subdomains() {
        let manifest = CapabilityManifest::gate_surface();
        let metadata = manifest.metadata();
        let mut registry = GateRegistry::new();
        registry
            .register(GateDescriptor::read_write(7, 4, "gate.alpha"))
            .expect("register");
        let mut binding = DeterministicGateMock::new();
        binding.seed_read(GateId::new(7, 4), AbiValue::I32(1));
        let session =
            GateExecutionSession::kernel_bound(&registry, &mut binding, &manifest, metadata);

        let mut state = SemanticStateStore::new();
        state
            .apply(StateUpdate::new(
                "fact.alpha",
                FactResolution::Certain(FactValue::Bool(true)),
                ContextWindow::new("root"),
                "seed alpha",
            ))
            .expect("seed");
        let mut rules = RuleEngine::new();
        rules
            .register(RuleDefinition::new(
                "rule.alpha",
                5,
                vec![RuleCondition::equals("fact.alpha", FactValue::Bool(true))],
            ))
            .expect("register rule");

        let agenda = session.derive_agenda(&state, &rules);
        assert_eq!(agenda.entries().len(), 1);
        let activation = session
            .select_next_activation(&agenda)
            .expect("activation selection");
        assert_eq!(activation.entry.rule_id.0, "rule.alpha");
        assert_eq!(activation.remaining_rules, 0);

        let mut audit = session.begin_audit_trail();
        session.record_session_started(&mut audit, "main");
        session.record_rule_activation(&mut audit, &activation);
        let transition = state.transitions().last().expect("transition");
        session.record_state_transition(&mut audit, transition);
        session.record_session_finished(&mut audit);
        assert!(audit.session().gate_registry_bound);
        assert_eq!(audit.events().len(), 4);

        let snapshot = session.integration_snapshot(&state, &agenda);
        assert_eq!(snapshot.state_epoch, StateEpoch(1));
        assert_eq!(snapshot.active_rules, 1);
    }

    #[test]
    fn gate_execution_session_applies_state_update_refreshes_agenda_and_emits_audit() {
        let manifest = CapabilityManifest::gate_surface();
        let metadata = manifest.metadata();
        let mut registry = GateRegistry::new();
        registry
            .register(GateDescriptor::read_write(7, 4, "gate.alpha"))
            .expect("register");
        let mut binding = DeterministicGateMock::new();
        binding.seed_read(GateId::new(7, 4), AbiValue::I32(1));
        let session =
            GateExecutionSession::kernel_bound(&registry, &mut binding, &manifest, metadata);

        let mut state = SemanticStateStore::new();
        let mut rules = RuleEngine::new();
        rules
            .register(RuleDefinition::new(
                "rule.alpha",
                5,
                vec![RuleCondition::equals("fact.alpha", FactValue::Bool(true))],
            ))
            .expect("register rule");

        let mut audit = session.begin_audit_trail();
        let advance = session
            .apply_state_update_and_refresh_agenda(
                &mut state,
                StateUpdate::new(
                    "fact.alpha",
                    FactResolution::Certain(FactValue::Bool(true)),
                    ContextWindow::new("root"),
                    "seed alpha",
                ),
                &rules,
                &mut audit,
            )
            .expect("advance state");

        assert_eq!(advance.transition.from_epoch, StateEpoch(0));
        assert_eq!(advance.transition.to_epoch, StateEpoch(1));
        assert_eq!(advance.agenda.entries().len(), 1);
        assert_eq!(advance.snapshot.state_epoch, StateEpoch(1));
        assert_eq!(advance.snapshot.active_rules, 1);
        assert!(matches!(
            &audit.events()[0].kind,
            AuditEventKind::StateTransition {
                key,
                from_epoch,
                to_epoch
            } if key == "fact.alpha" && *from_epoch == 0 && *to_epoch == 1
        ));
    }

    #[test]
    fn execution_session_applies_rule_state_write_effects_in_declared_order() {
        let manifest = CapabilityManifest::gate_surface();
        let metadata = manifest.metadata();
        let mut host = RecordingHostAbi::default();
        let session = ExecutionSession::kernel_bound(&mut host, &manifest, metadata);

        let mut state = SemanticStateStore::new();
        state
            .apply(StateUpdate::new(
                "fact.alpha",
                FactResolution::Certain(FactValue::Bool(true)),
                ContextWindow::new("root"),
                "seed alpha",
            ))
            .expect("seed");

        let mut rules = RuleEngine::new();
        let rule = RuleDefinition::new(
            "rule.alpha",
            5,
            vec![RuleCondition::equals("fact.alpha", FactValue::Bool(true))],
        )
        .with_effects(vec![
            RuleEffect::state_write("fact.beta", FactValue::I32(2), "window.beta", "derive beta"),
            RuleEffect::state_write(
                "fact.gamma",
                FactValue::Text("ready".to_string()),
                "window.gamma",
                "derive gamma",
            ),
        ]);
        rules.register(rule.clone()).expect("register rule");

        let mut audit = session.begin_audit_trail();
        let advances = session
            .apply_rule_state_write_effects(&mut state, &rule, &rules, &mut audit)
            .expect("apply rule state-write effects");

        assert_eq!(advances.len(), 2);
        assert_eq!(advances[0].effect_ordinal, 0);
        assert_eq!(advances[0].advance.transition.key, "fact.beta");
        assert_eq!(advances[1].effect_ordinal, 1);
        assert_eq!(advances[1].advance.transition.key, "fact.gamma");
        assert!(matches!(
            state.get("fact.beta").expect("fact.beta").resolution,
            FactResolution::Certain(FactValue::I32(2))
        ));
        assert!(matches!(
            state.get("fact.gamma").expect("fact.gamma").resolution,
            FactResolution::Certain(FactValue::Text(ref text)) if text == "ready"
        ));
        assert!(matches!(
            &audit.events()[0].kind,
            AuditEventKind::StateTransition {
                key,
                from_epoch,
                to_epoch
            } if key == "fact.beta" && *from_epoch == 1 && *to_epoch == 2
        ));
        assert!(matches!(
            &audit.events()[1].kind,
            AuditEventKind::StateTransition {
                key,
                from_epoch,
                to_epoch
            } if key == "fact.gamma" && *from_epoch == 2 && *to_epoch == 3
        ));
    }

    #[test]
    fn execution_session_rejects_non_state_write_effect_families_in_first_wave() {
        let manifest = CapabilityManifest::gate_surface();
        let metadata = manifest.metadata();
        let mut host = RecordingHostAbi::default();
        let session = ExecutionSession::kernel_bound(&mut host, &manifest, metadata);

        let mut state = SemanticStateStore::new();
        state
            .apply(StateUpdate::new(
                "fact.alpha",
                FactResolution::Certain(FactValue::Bool(true)),
                ContextWindow::new("root"),
                "seed alpha",
            ))
            .expect("seed");

        let mut rules = RuleEngine::new();
        let rule = RuleDefinition::new(
            "rule.alpha",
            5,
            vec![RuleCondition::equals("fact.alpha", FactValue::Bool(true))],
        )
        .with_effects(vec![RuleEffect::audit_note("not yet admitted")]);
        rules.register(rule.clone()).expect("register rule");

        let mut audit = session.begin_audit_trail();
        let err = session
            .apply_rule_state_write_effects(&mut state, &rule, &rules, &mut audit)
            .expect_err("audit-note execution is not admitted in first wave");

        assert_eq!(err.code, RuleEffectExecutionCode::UnsupportedEffectFamily);
        assert_eq!(err.rule_id.0, "rule.alpha");
        assert_eq!(err.effect_ordinal, 0);
        assert!(audit.events().is_empty());
        assert!(state.get("fact.alpha").is_some());
        assert!(state.get("fact.beta").is_none());
    }

    #[test]
    fn execution_session_applies_rule_audit_note_effects_in_declared_order() {
        let manifest = CapabilityManifest::gate_surface();
        let metadata = manifest.metadata();
        let mut host = RecordingHostAbi::default();
        let session = ExecutionSession::kernel_bound(&mut host, &manifest, metadata);

        let rule = RuleDefinition::new(
            "rule.alpha",
            5,
            vec![RuleCondition::equals("fact.alpha", FactValue::Bool(true))],
        )
        .with_effects(vec![
            RuleEffect::audit_note("rule.alpha note one"),
            RuleEffect::audit_note("rule.alpha note two"),
        ]);

        let mut audit = session.begin_audit_trail();
        let advances = session
            .apply_rule_audit_note_effects(&mut audit, &rule)
            .expect("apply rule audit-note effects");

        assert_eq!(advances.len(), 2);
        assert_eq!(advances[0].effect_ordinal, 0);
        assert_eq!(advances[0].event_id, AuditEventId(0));
        assert_eq!(advances[1].effect_ordinal, 1);
        assert_eq!(advances[1].event_id, AuditEventId(1));
        assert!(matches!(
            &audit.events()[0].kind,
            AuditEventKind::Note { message } if message == "rule.alpha note one"
        ));
        assert!(matches!(
            &audit.events()[1].kind,
            AuditEventKind::Note { message } if message == "rule.alpha note two"
        ));
    }

    #[test]
    fn execution_session_rejects_non_audit_note_effect_families_in_audit_slice() {
        let manifest = CapabilityManifest::gate_surface();
        let metadata = manifest.metadata();
        let mut host = RecordingHostAbi::default();
        let session = ExecutionSession::kernel_bound(&mut host, &manifest, metadata);

        let rule = RuleDefinition::new(
            "rule.alpha",
            5,
            vec![RuleCondition::equals("fact.alpha", FactValue::Bool(true))],
        )
        .with_effects(vec![RuleEffect::state_write(
            "fact.beta",
            FactValue::I32(2),
            "window.beta",
            "derive beta",
        )]);

        let mut audit = session.begin_audit_trail();
        let err = session
            .apply_rule_audit_note_effects(&mut audit, &rule)
            .expect_err("state-write execution is not admitted in audit-note slice");

        assert_eq!(err.code, RuleEffectExecutionCode::UnsupportedEffectFamily);
        assert_eq!(err.rule_id.0, "rule.alpha");
        assert_eq!(err.effect_ordinal, 0);
        assert!(audit.events().is_empty());
    }

    // #1762 (FA-08-004) primary trust invariant: the same context label
    // paired with a deliberately custom, distinct RuntimeQuotas envelope
    // must propagate that exact envelope - not a context-derived baseline -
    // all the way from `ExecutionConfig` through `RuntimeSessionDescriptor`
    // into the audit trail's own session metadata.
    fn custom_quota_envelope() -> RuntimeQuotas {
        RuntimeQuotas {
            max_steps: 101,
            max_calls: 102,
            max_stack_depth: 103,
            max_frames: 104,
            max_registers: 105,
            max_symbol_table: 106,
            max_effect_calls: 107,
            max_debug_symbols_per_function: 108,
        }
    }

    #[test]
    fn execution_session_propagates_custom_envelope_into_descriptor_and_audit_session() {
        let manifest = CapabilityManifest::gate_surface();
        let metadata = manifest.metadata();
        let mut host = RecordingHostAbi::with_read_value(AbiValue::I32(1));
        let config = ExecutionConfig::new(ExecutionContext::VerifiedLocal, custom_quota_envelope());
        let session = ExecutionSession::new(&mut host, &manifest, config, metadata);

        assert_eq!(
            session.descriptor().context,
            ExecutionContext::VerifiedLocal
        );
        assert_eq!(session.descriptor().quotas, custom_quota_envelope());

        let audit = session.begin_audit_trail();
        assert_eq!(audit.session().context, ExecutionContext::VerifiedLocal);
        assert_eq!(audit.session().quotas, custom_quota_envelope());
    }

    #[test]
    fn gate_execution_session_propagates_custom_envelope_into_descriptor_and_audit_session() {
        let manifest = CapabilityManifest::gate_surface();
        let metadata = manifest.metadata();
        let registry = GateRegistry::new();
        let mut binding = DeterministicGateMock::new();
        let config = ExecutionConfig::new(ExecutionContext::VerifiedLocal, custom_quota_envelope());
        let session =
            GateExecutionSession::new(&registry, &mut binding, &manifest, config, metadata);

        assert_eq!(
            session.descriptor().context,
            ExecutionContext::VerifiedLocal
        );
        assert_eq!(session.descriptor().quotas, custom_quota_envelope());

        let audit = session.begin_audit_trail();
        assert_eq!(audit.session().context, ExecutionContext::VerifiedLocal);
        assert_eq!(audit.session().quotas, custom_quota_envelope());
    }
}
