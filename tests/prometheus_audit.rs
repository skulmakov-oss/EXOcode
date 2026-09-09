use semantic_language::frontend::{emit_ir_to_semcode, IrFunction, IrInstr};
use semantic_language::prom_abi::AbiValue;
use semantic_language::prom_audit::{AuditEventKind, AuditSessionMetadata, AuditTrail};
use semantic_language::prom_cap::{CapabilityKind, CapabilityManifest};
use semantic_language::prom_gates::{DeterministicGateMock, GateDescriptor, GateId, GateRegistry};
use semantic_language::prom_runtime::GateExecutionSession;
use semantic_language::runtime_core::{ExecutionContext, RuntimeQuotas};

fn runtime_program() -> Vec<IrFunction> {
    vec![IrFunction {
        name: "main".to_string(),
        instrs: vec![
            IrInstr::GateRead {
                dst: 0,
                device_id: 7,
                port: 3,
            },
            IrInstr::GateWrite {
                device_id: 7,
                port: 4,
                src: 0,
            },
            IrInstr::Ret { src: None },
        ],
        ownership_events: Vec::new(),
        params: Vec::new(),
    }]
}

#[test]
fn audit_trail_reuses_runtime_session_descriptor_without_owning_runtime_logic() {
    let bytes = emit_ir_to_semcode(&runtime_program(), false).expect("emit");

    let mut registry = GateRegistry::new();
    registry
        .register(GateDescriptor::read_only(7, 3, "sensor.alpha"))
        .expect("register read gate");
    registry
        .register(GateDescriptor::read_write(7, 4, "actuator.beta"))
        .expect("register write gate");

    let manifest = CapabilityManifest::gate_surface();
    let mut binding = DeterministicGateMock::new();
    binding.seed_read(GateId::new(7, 3), AbiValue::I32(88));

    let mut session =
        GateExecutionSession::kernel_bound(&registry, &mut binding, &manifest, manifest.metadata());
    let audit_session = AuditSessionMetadata {
        context: session.descriptor().context,
        quotas: session.descriptor().quotas,
        capability_manifest: session.descriptor().capability_manifest.clone(),
        gate_registry_bound: session.descriptor().gate_registry_bound,
    };
    let mut audit = AuditTrail::new(audit_session);
    audit.record(AuditEventKind::SessionStarted {
        entry: "main".to_string(),
    });

    // Intentional byte-shim compatibility coverage: this test protects the public run_verified_semcode* API surface.
    session.run_verified_semcode(&bytes).expect("run");
    drop(session);

    audit.record(AuditEventKind::GateRead {
        device_id: 7,
        port: 3,
    });
    audit.record(AuditEventKind::GateWrite {
        device_id: 7,
        port: 4,
    });
    audit.record(AuditEventKind::SessionFinished);

    let replay = audit.replay_metadata();
    assert_eq!(replay.session.context, ExecutionContext::KernelBound);
    assert!(replay.session.gate_registry_bound);
    assert_eq!(replay.event_count, 4);
    assert_eq!(binding.writes(), &[(GateId::new(7, 4), AbiValue::I32(88))]);
}

#[test]
fn audit_trail_records_capability_denial_with_manifest_context() {
    let mut audit = AuditTrail::new(AuditSessionMetadata {
        context: ExecutionContext::KernelBound,
        quotas: RuntimeQuotas::kernel_bound(),
        capability_manifest: CapabilityManifest::new().metadata(),
        gate_registry_bound: true,
    });
    audit.record(AuditEventKind::CapabilityDenied {
        capability: CapabilityKind::PulseEmit,
        call: Some("PulseEmit".to_string()),
    });

    let replay = audit.replay_metadata();
    assert_eq!(replay.event_count, 1);
    match &audit.events()[0].kind {
        AuditEventKind::CapabilityDenied { capability, call } => {
            assert_eq!(*capability, CapabilityKind::PulseEmit);
            assert_eq!(call.as_deref(), Some("PulseEmit"));
            assert_eq!(
                replay.session.capability_manifest.schema,
                "prom.cap.manifest"
            );
        }
        other => panic!("unexpected event {other:?}"),
    }
}
