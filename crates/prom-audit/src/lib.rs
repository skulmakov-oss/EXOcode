#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod hello_observation_audit;

use alloc::string::String;
use alloc::vec::Vec;
use hello_observation_audit::ControlledObservationAuditDecision;
use prom_cap::{CapabilityKind, CapabilityManifestMetadata, CapabilityManifestVersion};
use sm_runtime_core::{ExecutionContext, RuntimeQuotas};

const AUDIT_REPLAY_ARCHIVE_MAGIC: &str = "semantic_audit_replay_archive";
const MULTI_SESSION_REPLAY_ARCHIVE_MAGIC: &str = "semantic_multi_session_replay_archive";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AuditEventId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditSessionMetadata {
    pub context: ExecutionContext,
    pub quotas: RuntimeQuotas,
    pub capability_manifest: CapabilityManifestMetadata,
    pub gate_registry_bound: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditEventKind {
    SessionStarted {
        entry: String,
    },
    SessionFinished,
    RuleActivated {
        rule_id: String,
        salience: i32,
    },
    StateTransition {
        key: String,
        from_epoch: u64,
        to_epoch: u64,
    },
    CapabilityDenied {
        capability: CapabilityKind,
        call: Option<String>,
    },
    ControlledObservation {
        operation_kind: String,
        observation_class: String,
        payload_hash: Option<u64>,
        redacted: bool,
        sequence_index: u64,
        policy: ControlledObservationAuditDecision,
        linkage: crate::hello_observation_audit::HelloObservationAuditLinkage,
    },
    GateRead {
        device_id: u16,
        port: u16,
    },
    GateWrite {
        device_id: u16,
        port: u16,
    },
    PulseEmit {
        signal: String,
    },
    Note {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEvent {
    pub id: AuditEventId,
    pub kind: AuditEventKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayMetadata {
    pub session: AuditSessionMetadata,
    pub event_count: usize,
    pub last_event_id: Option<AuditEventId>,
}

pub const AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION: u32 = 2;
pub const MULTI_SESSION_REPLAY_ARCHIVE_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditReplayArchive {
    pub format_version: u32,
    pub session: AuditSessionMetadata,
    pub events: Vec<AuditEvent>,
    pub replay: ReplayMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiSessionReplayArchiveSession {
    pub session_ordinal: u32,
    pub archive: AuditReplayArchive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiSessionReplayArchive {
    pub format_version: u32,
    pub sessions: Vec<MultiSessionReplayArchiveSession>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditReplayArchiveFormatError {
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiSessionReplayArchiveFormatError {
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditTrail {
    session: AuditSessionMetadata,
    next_id: u64,
    events: Vec<AuditEvent>,
}

impl AuditTrail {
    pub fn new(session: AuditSessionMetadata) -> Self {
        Self {
            session,
            next_id: 0,
            events: Vec::new(),
        }
    }

    pub fn session(&self) -> &AuditSessionMetadata {
        &self.session
    }

    pub fn events(&self) -> &[AuditEvent] {
        &self.events
    }

    pub fn record(&mut self, kind: AuditEventKind) -> AuditEventId {
        let id = AuditEventId(self.next_id);
        self.next_id += 1;
        self.events.push(AuditEvent { id, kind });
        id
    }

    pub fn replay_metadata(&self) -> ReplayMetadata {
        ReplayMetadata {
            session: self.session.clone(),
            event_count: self.events.len(),
            last_event_id: self.events.last().map(|event| event.id),
        }
    }

    pub fn replay_archive(&self) -> AuditReplayArchive {
        AuditReplayArchive::new(
            self.session.clone(),
            self.events.clone(),
            self.replay_metadata(),
        )
    }
}

impl AuditReplayArchive {
    pub fn new(
        session: AuditSessionMetadata,
        events: Vec<AuditEvent>,
        replay: ReplayMetadata,
    ) -> Self {
        Self {
            format_version: AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION,
            session,
            events,
            replay,
        }
    }

    pub fn to_canonical_text(&self) -> String {
        let mut out = String::new();
        out.push_str(AUDIT_REPLAY_ARCHIVE_MAGIC);
        out.push('\t');
        out.push_str(&self.format_version.to_string());
        out.push('\n');
        out.push_str("session\t");
        out.push_str(display_execution_context(self.session.context));
        out.push('\t');
        out.push_str(&escape_archive_field(
            &self.session.capability_manifest.schema,
        ));
        out.push('\t');
        out.push_str(display_manifest_version(
            self.session.capability_manifest.version,
        ));
        out.push('\t');
        out.push_str(if self.session.gate_registry_bound {
            "true"
        } else {
            "false"
        });
        out.push('\t');
        out.push_str(&self.session.quotas.max_steps.to_string());
        out.push('\t');
        out.push_str(&self.session.quotas.max_calls.to_string());
        out.push('\t');
        out.push_str(&self.session.quotas.max_stack_depth.to_string());
        out.push('\t');
        out.push_str(&self.session.quotas.max_frames.to_string());
        out.push('\t');
        out.push_str(&self.session.quotas.max_registers.to_string());
        out.push('\t');
        out.push_str(&self.session.quotas.max_symbol_table.to_string());
        out.push('\t');
        out.push_str(&self.session.quotas.max_effect_calls.to_string());
        out.push('\t');
        out.push_str(
            &self
                .session
                .quotas
                .max_debug_symbols_per_function
                .to_string(),
        );
        out.push('\n');
        out.push_str("events\t");
        out.push_str(&self.events.len().to_string());
        out.push('\n');

        for event in &self.events {
            out.push_str("event\t");
            out.push_str(&event.id.0.to_string());
            encode_event_kind(&mut out, &event.kind);
            out.push('\n');
        }

        out.push_str("replay\t");
        out.push_str(&self.replay.event_count.to_string());
        out.push('\t');
        match self.replay.last_event_id {
            Some(id) => out.push_str(&id.0.to_string()),
            None => out.push_str("none"),
        }
        out.push('\n');

        out
    }

    pub fn from_canonical_text(src: &str) -> Result<Self, AuditReplayArchiveFormatError> {
        let mut lines = src.lines();
        let header = lines
            .next()
            .ok_or_else(|| AuditReplayArchiveFormatError::new("missing archive header"))?;
        let header_parts = split_archive_line(header);
        if header_parts.len() != 2 || header_parts[0] != AUDIT_REPLAY_ARCHIVE_MAGIC {
            return Err(AuditReplayArchiveFormatError::new("invalid archive header"));
        }
        let format_version = parse_u32_field(header_parts[1], "archive format version")?;
        if format_version != AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION {
            return Err(AuditReplayArchiveFormatError::new(format!(
                "unsupported archive format version {}; expected {}",
                format_version, AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION
            )));
        }

        let session_line = lines
            .next()
            .ok_or_else(|| AuditReplayArchiveFormatError::new("missing session line"))?;
        let session_parts = split_archive_line(session_line);
        if session_parts.len() != 13 || session_parts[0] != "session" {
            return Err(AuditReplayArchiveFormatError::new("invalid session line"));
        }
        let session = AuditSessionMetadata {
            context: parse_execution_context(session_parts[1])?,
            capability_manifest: CapabilityManifestMetadata {
                schema: unescape_archive_field(session_parts[2])?,
                version: parse_manifest_version(session_parts[3])?,
            },
            gate_registry_bound: parse_bool_field(session_parts[4], "gate registry bound")?,
            quotas: RuntimeQuotas {
                max_steps: parse_usize_field(session_parts[5], "max_steps")?,
                max_calls: parse_usize_field(session_parts[6], "max_calls")?,
                max_stack_depth: parse_usize_field(session_parts[7], "max_stack_depth")?,
                max_frames: parse_usize_field(session_parts[8], "max_frames")?,
                max_registers: parse_usize_field(session_parts[9], "max_registers")?,
                max_symbol_table: parse_usize_field(session_parts[10], "max_symbol_table")?,
                max_effect_calls: parse_usize_field(session_parts[11], "max_effect_calls")?,
                max_debug_symbols_per_function: parse_usize_field(
                    session_parts[12],
                    "max_debug_symbols_per_function",
                )?,
            },
        };

        let events_line = lines
            .next()
            .ok_or_else(|| AuditReplayArchiveFormatError::new("missing event count line"))?;
        let event_count_parts = split_archive_line(events_line);
        if event_count_parts.len() != 2 || event_count_parts[0] != "events" {
            return Err(AuditReplayArchiveFormatError::new(
                "invalid event count line",
            ));
        }
        let expected_event_count = parse_usize_field(event_count_parts[1], "event count")?;

        let mut events = Vec::with_capacity(expected_event_count);
        for _ in 0..expected_event_count {
            let event_line = lines
                .next()
                .ok_or_else(|| AuditReplayArchiveFormatError::new("missing event line"))?;
            let parts = split_archive_line(event_line);
            if !parts.is_empty() && parts[0] == "replay" {
                return Err(AuditReplayArchiveFormatError::new(
                    "replay event count does not match archive events",
                ));
            }
            if parts.len() < 3 || parts[0] != "event" {
                return Err(AuditReplayArchiveFormatError::new("invalid event line"));
            }
            let id = AuditEventId(parse_u64_field(parts[1], "event id")?);
            let kind = decode_event_kind(&parts[2..])?;
            events.push(AuditEvent { id, kind });
        }

        let replay_line = lines
            .next()
            .ok_or_else(|| AuditReplayArchiveFormatError::new("missing replay line"))?;
        let replay_parts = split_archive_line(replay_line);
        if replay_parts.len() != 3 || replay_parts[0] != "replay" {
            return Err(AuditReplayArchiveFormatError::new("invalid replay line"));
        }
        let replay = ReplayMetadata {
            session: session.clone(),
            event_count: parse_usize_field(replay_parts[1], "replay event count")?,
            last_event_id: parse_optional_event_id(replay_parts[2])?,
        };

        if replay.event_count != events.len() {
            return Err(AuditReplayArchiveFormatError::new(
                "replay event count does not match archive events",
            ));
        }
        let actual_last_event_id = events.last().map(|event| event.id);
        if replay.last_event_id != actual_last_event_id {
            return Err(AuditReplayArchiveFormatError::new(
                "replay last event id does not match archive events",
            ));
        }
        for (index, event) in events.iter().enumerate() {
            if event.id != AuditEventId(index as u64) {
                return Err(AuditReplayArchiveFormatError::new(
                    "archive event ids must be monotonic from zero",
                ));
            }
        }

        if lines.any(|line| !line.trim().is_empty()) {
            return Err(AuditReplayArchiveFormatError::new(
                "unexpected trailing archive lines",
            ));
        }

        Ok(Self {
            format_version,
            session,
            events,
            replay,
        })
    }
}

impl MultiSessionReplayArchiveSession {
    pub fn new(session_ordinal: u32, archive: AuditReplayArchive) -> Self {
        Self {
            session_ordinal,
            archive,
        }
    }
}

impl MultiSessionReplayArchive {
    pub fn new(sessions: Vec<MultiSessionReplayArchiveSession>) -> Self {
        Self {
            format_version: MULTI_SESSION_REPLAY_ARCHIVE_FORMAT_VERSION,
            sessions,
        }
    }

    pub fn to_canonical_text(&self) -> String {
        let mut out = String::new();
        out.push_str(MULTI_SESSION_REPLAY_ARCHIVE_MAGIC);
        out.push('\t');
        out.push_str(&self.format_version.to_string());
        out.push('\n');
        out.push_str("sessions\t");
        out.push_str(&self.sessions.len().to_string());
        out.push('\n');

        for session in &self.sessions {
            let archive_text = session.archive.to_canonical_text();
            let archive_lines = archive_text.lines().collect::<Vec<_>>();
            out.push_str("session\t");
            out.push_str(&session.session_ordinal.to_string());
            out.push('\t');
            out.push_str(&archive_lines.len().to_string());
            out.push('\n');
            for line in archive_lines {
                out.push_str("archive\t");
                out.push_str(line);
                out.push('\n');
            }
        }

        out
    }

    pub fn from_canonical_text(src: &str) -> Result<Self, MultiSessionReplayArchiveFormatError> {
        let mut lines = src.lines();
        let header = lines.next().ok_or_else(|| {
            MultiSessionReplayArchiveFormatError::new("missing multi-session archive header")
        })?;
        let header_parts = split_archive_line(header);
        if header_parts.len() != 2 || header_parts[0] != MULTI_SESSION_REPLAY_ARCHIVE_MAGIC {
            return Err(MultiSessionReplayArchiveFormatError::new(
                "invalid multi-session archive header",
            ));
        }
        let format_version =
            parse_u32_field(header_parts[1], "multi-session archive format version")
                .map_err(|err| MultiSessionReplayArchiveFormatError::new(err.message))?;
        if format_version != MULTI_SESSION_REPLAY_ARCHIVE_FORMAT_VERSION {
            return Err(MultiSessionReplayArchiveFormatError::new(format!(
                "unsupported multi-session archive format version {}; expected {}",
                format_version, MULTI_SESSION_REPLAY_ARCHIVE_FORMAT_VERSION
            )));
        }

        let session_count_line = lines.next().ok_or_else(|| {
            MultiSessionReplayArchiveFormatError::new(
                "missing multi-session archive session-count line",
            )
        })?;
        let session_count_parts = split_archive_line(session_count_line);
        if session_count_parts.len() != 2 || session_count_parts[0] != "sessions" {
            return Err(MultiSessionReplayArchiveFormatError::new(
                "invalid multi-session archive session-count line",
            ));
        }
        let expected_session_count = parse_usize_field(
            session_count_parts[1],
            "multi-session archive session count",
        )
        .map_err(|err| MultiSessionReplayArchiveFormatError::new(err.message))?;

        let mut sessions = Vec::with_capacity(expected_session_count);
        while let Some(line) = lines.next() {
            if line.trim().is_empty() {
                continue;
            }
            let session_parts = split_archive_line(line);
            if session_parts.len() != 3 || session_parts[0] != "session" {
                return Err(MultiSessionReplayArchiveFormatError::new(
                    "invalid multi-session archive session line",
                ));
            }
            let session_ordinal =
                parse_u32_field(session_parts[1], "multi-session session ordinal")
                    .map_err(|err| MultiSessionReplayArchiveFormatError::new(err.message))?;
            let expected_archive_line_count = parse_usize_field(
                session_parts[2],
                "multi-session embedded archive line count",
            )
            .map_err(|err| MultiSessionReplayArchiveFormatError::new(err.message))?;
            if expected_archive_line_count == 0 {
                return Err(MultiSessionReplayArchiveFormatError::new(
                    "embedded audit archive line count must not be zero",
                ));
            }

            let mut archive_text = String::new();
            for index in 0..expected_archive_line_count {
                let archive_line = lines.next().ok_or_else(|| {
                    MultiSessionReplayArchiveFormatError::new("missing embedded audit archive line")
                })?;
                let archive_parts = split_archive_line(archive_line);
                if archive_parts.is_empty() || archive_parts[0] != "archive" {
                    return Err(MultiSessionReplayArchiveFormatError::new(
                        "invalid embedded audit archive line",
                    ));
                }
                let embedded_line = archive_line.strip_prefix("archive\t").ok_or_else(|| {
                    MultiSessionReplayArchiveFormatError::new(
                        "embedded audit archive line missing prefix payload",
                    )
                })?;
                archive_text.push_str(embedded_line);
                if index + 1 != expected_archive_line_count {
                    archive_text.push('\n');
                }
            }

            let archive = AuditReplayArchive::from_canonical_text(&archive_text)
                .map_err(|err| MultiSessionReplayArchiveFormatError::new(err.message))?;
            sessions.push(MultiSessionReplayArchiveSession {
                session_ordinal,
                archive,
            });
        }

        if sessions.len() != expected_session_count {
            return Err(MultiSessionReplayArchiveFormatError::new(format!(
                "session count mismatch: header says {}, parsed {}",
                expected_session_count,
                sessions.len()
            )));
        }
        for (index, session) in sessions.iter().enumerate() {
            if session.session_ordinal != index as u32 {
                return Err(MultiSessionReplayArchiveFormatError::new(
                    "multi-session replay session ordinals must be monotonic from zero",
                ));
            }
        }

        Ok(Self {
            format_version,
            sessions,
        })
    }
}

impl AuditReplayArchiveFormatError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl core::fmt::Display for AuditReplayArchiveFormatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "audit replay archive format error: {}", self.message)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AuditReplayArchiveFormatError {}

impl MultiSessionReplayArchiveFormatError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl core::fmt::Display for MultiSessionReplayArchiveFormatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "multi-session replay archive format error: {}",
            self.message
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MultiSessionReplayArchiveFormatError {}

fn encode_event_kind(out: &mut String, kind: &AuditEventKind) {
    match kind {
        AuditEventKind::SessionStarted { entry } => {
            out.push_str("\tsession-started\t");
            out.push_str(&escape_archive_field(entry));
        }
        AuditEventKind::SessionFinished => {
            out.push_str("\tsession-finished");
        }
        AuditEventKind::RuleActivated { rule_id, salience } => {
            out.push_str("\trule-activated\t");
            out.push_str(&escape_archive_field(rule_id));
            out.push('\t');
            out.push_str(&salience.to_string());
        }
        AuditEventKind::StateTransition {
            key,
            from_epoch,
            to_epoch,
        } => {
            out.push_str("\tstate-transition\t");
            out.push_str(&escape_archive_field(key));
            out.push('\t');
            out.push_str(&from_epoch.to_string());
            out.push('\t');
            out.push_str(&to_epoch.to_string());
        }
        AuditEventKind::CapabilityDenied { capability, call } => {
            out.push_str("\tcapability-denied\t");
            out.push_str(display_capability_kind(*capability));
            out.push('\t');
            match call {
                Some(call) => out.push_str(&escape_archive_field(call)),
                None => out.push_str("none"),
            }
        }
        AuditEventKind::ControlledObservation {
            operation_kind,
            observation_class,
            payload_hash,
            redacted,
            sequence_index,
            policy,
            linkage,
        } => {
            out.push_str("\tcontrolled-observation\t");
            out.push_str(&escape_archive_field(operation_kind));
            out.push('\t');
            out.push_str(&escape_archive_field(observation_class));
            out.push('\t');
            match payload_hash {
                Some(hash) => out.push_str(&hash.to_string()),
                None => out.push_str("none"),
            }
            out.push('\t');
            out.push_str(if *redacted { "true" } else { "false" });
            out.push('\t');
            out.push_str(&sequence_index.to_string());
            out.push('\t');
            out.push_str(display_controlled_observation_audit_decision(*policy));
            out.push('\t');
            match linkage.verifier_admission_ref {
                Some(value) => out.push_str(&value.to_string()),
                None => out.push_str("none"),
            }
            out.push('\t');
            match linkage.capability_policy_ref {
                Some(value) => out.push_str(&value.to_string()),
                None => out.push_str("none"),
            }
            out.push('\t');
            match linkage.sink_policy_ref {
                Some(value) => out.push_str(&value.to_string()),
                None => out.push_str("none"),
            }
        }
        AuditEventKind::GateRead { device_id, port } => {
            out.push_str("\tgate-read\t");
            out.push_str(&device_id.to_string());
            out.push('\t');
            out.push_str(&port.to_string());
        }
        AuditEventKind::GateWrite { device_id, port } => {
            out.push_str("\tgate-write\t");
            out.push_str(&device_id.to_string());
            out.push('\t');
            out.push_str(&port.to_string());
        }
        AuditEventKind::PulseEmit { signal } => {
            out.push_str("\tpulse-emit\t");
            out.push_str(&escape_archive_field(signal));
        }
        AuditEventKind::Note { message } => {
            out.push_str("\tnote\t");
            out.push_str(&escape_archive_field(message));
        }
    }
}

fn decode_event_kind(parts: &[&str]) -> Result<AuditEventKind, AuditReplayArchiveFormatError> {
    if parts.is_empty() {
        return Err(AuditReplayArchiveFormatError::new(
            "missing event kind payload",
        ));
    }
    match parts[0] {
        "session-started" => {
            if parts.len() != 2 {
                return Err(AuditReplayArchiveFormatError::new(
                    "invalid session-started payload",
                ));
            }
            Ok(AuditEventKind::SessionStarted {
                entry: unescape_archive_field(parts[1])?,
            })
        }
        "session-finished" => {
            if parts.len() != 1 {
                return Err(AuditReplayArchiveFormatError::new(
                    "invalid session-finished payload",
                ));
            }
            Ok(AuditEventKind::SessionFinished)
        }
        "rule-activated" => {
            if parts.len() != 3 {
                return Err(AuditReplayArchiveFormatError::new(
                    "invalid rule-activated payload",
                ));
            }
            Ok(AuditEventKind::RuleActivated {
                rule_id: unescape_archive_field(parts[1])?,
                salience: parse_i32_field(parts[2], "rule salience")?,
            })
        }
        "state-transition" => {
            if parts.len() != 4 {
                return Err(AuditReplayArchiveFormatError::new(
                    "invalid state-transition payload",
                ));
            }
            Ok(AuditEventKind::StateTransition {
                key: unescape_archive_field(parts[1])?,
                from_epoch: parse_u64_field(parts[2], "transition from epoch")?,
                to_epoch: parse_u64_field(parts[3], "transition to epoch")?,
            })
        }
        "capability-denied" => {
            if parts.len() != 3 {
                return Err(AuditReplayArchiveFormatError::new(
                    "invalid capability-denied payload",
                ));
            }
            Ok(AuditEventKind::CapabilityDenied {
                capability: parse_capability_kind(parts[1])?,
                call: parse_optional_string(parts[2])?,
            })
        }
        "controlled-observation" => {
            if parts.len() != 10 {
                return Err(AuditReplayArchiveFormatError::new(
                    "invalid controlled-observation payload",
                ));
            }
            Ok(AuditEventKind::ControlledObservation {
                operation_kind: unescape_archive_field(parts[1])?,
                observation_class: unescape_archive_field(parts[2])?,
                payload_hash: parse_optional_u64(parts[3])?,
                redacted: parse_bool_field(parts[4], "controlled observation redacted flag")?,
                sequence_index: parse_u64_field(parts[5], "controlled observation sequence index")?,
                policy: parse_controlled_observation_audit_decision(parts[6])?,
                linkage: crate::hello_observation_audit::HelloObservationAuditLinkage {
                    verifier_admission_ref: parse_optional_u64(parts[7])?,
                    capability_policy_ref: parse_optional_u64(parts[8])?,
                    sink_policy_ref: parse_optional_u64(parts[9])?,
                },
            })
        }
        "gate-read" => {
            if parts.len() != 3 {
                return Err(AuditReplayArchiveFormatError::new(
                    "invalid gate-read payload",
                ));
            }
            Ok(AuditEventKind::GateRead {
                device_id: parse_u16_field(parts[1], "gate-read device id")?,
                port: parse_u16_field(parts[2], "gate-read port")?,
            })
        }
        "gate-write" => {
            if parts.len() != 3 {
                return Err(AuditReplayArchiveFormatError::new(
                    "invalid gate-write payload",
                ));
            }
            Ok(AuditEventKind::GateWrite {
                device_id: parse_u16_field(parts[1], "gate-write device id")?,
                port: parse_u16_field(parts[2], "gate-write port")?,
            })
        }
        "pulse-emit" => {
            if parts.len() != 2 {
                return Err(AuditReplayArchiveFormatError::new(
                    "invalid pulse-emit payload",
                ));
            }
            Ok(AuditEventKind::PulseEmit {
                signal: unescape_archive_field(parts[1])?,
            })
        }
        "note" => {
            if parts.len() != 2 {
                return Err(AuditReplayArchiveFormatError::new("invalid note payload"));
            }
            Ok(AuditEventKind::Note {
                message: unescape_archive_field(parts[1])?,
            })
        }
        _ => Err(AuditReplayArchiveFormatError::new(
            "unknown audit event kind in archive",
        )),
    }
}

fn display_execution_context(context: ExecutionContext) -> &'static str {
    match context {
        ExecutionContext::PureCompute => "pure-compute",
        ExecutionContext::VerifiedLocal => "verified-local",
        ExecutionContext::RuleExecution => "rule-execution",
        ExecutionContext::KernelBound => "kernel-bound",
    }
}

fn parse_execution_context(raw: &str) -> Result<ExecutionContext, AuditReplayArchiveFormatError> {
    match raw {
        "pure-compute" => Ok(ExecutionContext::PureCompute),
        "verified-local" => Ok(ExecutionContext::VerifiedLocal),
        "rule-execution" => Ok(ExecutionContext::RuleExecution),
        "kernel-bound" => Ok(ExecutionContext::KernelBound),
        _ => Err(AuditReplayArchiveFormatError::new(
            "unknown execution context in archive",
        )),
    }
}

fn display_manifest_version(version: CapabilityManifestVersion) -> &'static str {
    match version {
        CapabilityManifestVersion::V1 => "v1",
    }
}

fn parse_manifest_version(
    raw: &str,
) -> Result<CapabilityManifestVersion, AuditReplayArchiveFormatError> {
    match raw {
        "v1" => Ok(CapabilityManifestVersion::V1),
        _ => Err(AuditReplayArchiveFormatError::new(
            "unknown capability manifest version in archive",
        )),
    }
}

fn display_capability_kind(kind: CapabilityKind) -> &'static str {
    match kind {
        CapabilityKind::GateRead => "GateRead",
        CapabilityKind::GateWrite => "GateWrite",
        CapabilityKind::PulseEmit => "PulseEmit",
        CapabilityKind::ControlledObservationSink => "ControlledObservationSink",
        CapabilityKind::StateQuery => "StateQuery",
        CapabilityKind::StateUpdate => "StateUpdate",
        CapabilityKind::EventPost => "EventPost",
        CapabilityKind::ClockRead => "ClockRead",
        CapabilityKind::ArgsRead => "ArgsRead",
        CapabilityKind::StdinReadText => "StdinReadText",
        CapabilityKind::StdoutWrite => "StdoutWrite",
        CapabilityKind::StderrWrite => "StderrWrite",
        CapabilityKind::PathInspect => "PathInspect",
        CapabilityKind::FsRead => "FsRead",
        CapabilityKind::FsWrite => "FsWrite",
        CapabilityKind::TimeDuration => "TimeDuration",
    }
}

fn display_controlled_observation_audit_decision(
    decision: ControlledObservationAuditDecision,
) -> &'static str {
    match decision {
        ControlledObservationAuditDecision::Record => "record",
        ControlledObservationAuditDecision::Redact => "redact",
        ControlledObservationAuditDecision::NoStore => "no_store",
        ControlledObservationAuditDecision::Deny => "deny",
    }
}

fn parse_capability_kind(raw: &str) -> Result<CapabilityKind, AuditReplayArchiveFormatError> {
    match raw {
        "GateRead" => Ok(CapabilityKind::GateRead),
        "GateWrite" => Ok(CapabilityKind::GateWrite),
        "PulseEmit" => Ok(CapabilityKind::PulseEmit),
        "ControlledObservationSink" => Ok(CapabilityKind::ControlledObservationSink),
        "StateQuery" => Ok(CapabilityKind::StateQuery),
        "StateUpdate" => Ok(CapabilityKind::StateUpdate),
        "EventPost" => Ok(CapabilityKind::EventPost),
        "ClockRead" => Ok(CapabilityKind::ClockRead),
        "ArgsRead" => Ok(CapabilityKind::ArgsRead),
        "StdinReadText" => Ok(CapabilityKind::StdinReadText),
        "StdoutWrite" => Ok(CapabilityKind::StdoutWrite),
        "StderrWrite" => Ok(CapabilityKind::StderrWrite),
        "PathInspect" => Ok(CapabilityKind::PathInspect),
        "FsRead" => Ok(CapabilityKind::FsRead),
        "FsWrite" => Ok(CapabilityKind::FsWrite),
        "TimeDuration" => Ok(CapabilityKind::TimeDuration),
        _ => Err(AuditReplayArchiveFormatError::new(
            "unknown capability kind in archive",
        )),
    }
}

fn parse_controlled_observation_audit_decision(
    raw: &str,
) -> Result<ControlledObservationAuditDecision, AuditReplayArchiveFormatError> {
    match raw {
        "record" => Ok(ControlledObservationAuditDecision::Record),
        "redact" => Ok(ControlledObservationAuditDecision::Redact),
        "no_store" => Ok(ControlledObservationAuditDecision::NoStore),
        "deny" => Ok(ControlledObservationAuditDecision::Deny),
        _ => Err(AuditReplayArchiveFormatError::new(
            "unknown controlled observation audit decision in archive",
        )),
    }
}

fn parse_bool_field(raw: &str, label: &str) -> Result<bool, AuditReplayArchiveFormatError> {
    match raw {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(AuditReplayArchiveFormatError::new(format!(
            "invalid {}",
            label
        ))),
    }
}

fn parse_u16_field(raw: &str, label: &str) -> Result<u16, AuditReplayArchiveFormatError> {
    raw.parse::<u16>()
        .map_err(|_| AuditReplayArchiveFormatError::new(format!("invalid {}", label)))
}

fn parse_u32_field(raw: &str, label: &str) -> Result<u32, AuditReplayArchiveFormatError> {
    raw.parse::<u32>()
        .map_err(|_| AuditReplayArchiveFormatError::new(format!("invalid {}", label)))
}

fn parse_u64_field(raw: &str, label: &str) -> Result<u64, AuditReplayArchiveFormatError> {
    raw.parse::<u64>()
        .map_err(|_| AuditReplayArchiveFormatError::new(format!("invalid {}", label)))
}

fn parse_i32_field(raw: &str, label: &str) -> Result<i32, AuditReplayArchiveFormatError> {
    raw.parse::<i32>()
        .map_err(|_| AuditReplayArchiveFormatError::new(format!("invalid {}", label)))
}

fn parse_usize_field(raw: &str, label: &str) -> Result<usize, AuditReplayArchiveFormatError> {
    raw.parse::<usize>()
        .map_err(|_| AuditReplayArchiveFormatError::new(format!("invalid {}", label)))
}

fn parse_optional_event_id(
    raw: &str,
) -> Result<Option<AuditEventId>, AuditReplayArchiveFormatError> {
    if raw == "none" {
        return Ok(None);
    }
    Ok(Some(AuditEventId(parse_u64_field(
        raw,
        "replay last event id",
    )?)))
}

fn parse_optional_u64(raw: &str) -> Result<Option<u64>, AuditReplayArchiveFormatError> {
    if raw == "none" {
        return Ok(None);
    }
    Ok(Some(parse_u64_field(raw, "optional u64")?))
}

fn parse_optional_string(raw: &str) -> Result<Option<String>, AuditReplayArchiveFormatError> {
    if raw == "none" {
        return Ok(None);
    }
    Ok(Some(unescape_archive_field(raw)?))
}

fn split_archive_line(line: &str) -> Vec<&str> {
    line.split('\t').collect()
}

fn escape_archive_field(value: &str) -> String {
    let mut escaped = String::new();
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '\t' => escaped.push_str("\\t"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            ':' => escaped.push_str("\\:"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn unescape_archive_field(value: &str) -> Result<String, AuditReplayArchiveFormatError> {
    let mut out = String::new();
    let mut chars = value.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        let escaped = chars
            .next()
            .ok_or_else(|| AuditReplayArchiveFormatError::new("unterminated archive escape"))?;
        match escaped {
            '\\' => out.push('\\'),
            't' => out.push('\t'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            ':' => out.push(':'),
            _ => {
                return Err(AuditReplayArchiveFormatError::new(
                    "unsupported archive escape sequence",
                ))
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hello_observation_audit::{
        apply_controlled_observation_audit_policy, ControlledObservationAuditDecision,
        ControlledObservationAuditResult, HelloObservationAuditLinkage,
    };

    fn sample_session() -> AuditSessionMetadata {
        AuditSessionMetadata {
            context: ExecutionContext::KernelBound,
            quotas: RuntimeQuotas::kernel_bound(),
            capability_manifest: CapabilityManifestMetadata {
                schema: "prom.cap.manifest".to_string(),
                version: prom_cap::CapabilityManifestVersion::V1,
            },
            gate_registry_bound: true,
        }
    }

    #[test]
    fn audit_trail_assigns_monotonic_event_ids() {
        let mut trail = AuditTrail::new(sample_session());
        let first = trail.record(AuditEventKind::SessionStarted {
            entry: "main".to_string(),
        });
        let second = trail.record(AuditEventKind::SessionFinished);
        assert_eq!(first, AuditEventId(0));
        assert_eq!(second, AuditEventId(1));
    }

    #[test]
    fn replay_metadata_reflects_session_and_event_count() {
        let session = sample_session();
        let mut trail = AuditTrail::new(session.clone());
        trail.record(AuditEventKind::SessionStarted {
            entry: "main".to_string(),
        });
        trail.record(AuditEventKind::GateRead {
            device_id: 7,
            port: 3,
        });

        let replay = trail.replay_metadata();
        assert_eq!(replay.session, session);
        assert_eq!(replay.event_count, 2);
        assert_eq!(replay.last_event_id, Some(AuditEventId(1)));
    }

    #[test]
    fn capability_denial_event_retains_call_context() {
        let mut trail = AuditTrail::new(sample_session());
        trail.record(AuditEventKind::CapabilityDenied {
            capability: CapabilityKind::PulseEmit,
            call: Some("PulseEmit".to_string()),
        });
        match &trail.events()[0].kind {
            AuditEventKind::CapabilityDenied { capability, call } => {
                assert_eq!(*capability, CapabilityKind::PulseEmit);
                assert_eq!(call.as_deref(), Some("PulseEmit"));
            }
            other => panic!("unexpected event {other:?}"),
        }
    }

    fn controlled_observation_linkage() -> HelloObservationAuditLinkage {
        HelloObservationAuditLinkage {
            verifier_admission_ref: Some(17),
            capability_policy_ref: Some(29),
            sink_policy_ref: Some(31),
        }
    }

    #[test]
    fn controlled_observation_record_policy_appends_deterministic_event() {
        let mut trail = AuditTrail::new(sample_session());
        let result = apply_controlled_observation_audit_policy(
            &mut trail,
            0xfeed_beef,
            7,
            ControlledObservationAuditDecision::Record,
            controlled_observation_linkage(),
        );

        assert_eq!(
            result,
            ControlledObservationAuditResult::Recorded(AuditEventId(0))
        );
        assert_eq!(trail.events().len(), 1);
        match &trail.events()[0].kind {
            AuditEventKind::ControlledObservation {
                operation_kind,
                observation_class,
                payload_hash,
                redacted,
                sequence_index,
                policy,
                linkage,
            } => {
                assert_eq!(operation_kind, "controlled_observation_text");
                assert_eq!(observation_class, "ControlledText");
                assert_eq!(*payload_hash, Some(0xfeed_beef));
                assert!(!redacted);
                assert_eq!(*sequence_index, 7);
                assert_eq!(*policy, ControlledObservationAuditDecision::Record);
                assert_eq!(*linkage, controlled_observation_linkage());
            }
            other => panic!("unexpected event {other:?}"),
        }

        let archive = trail.replay_archive();
        let text = archive.to_canonical_text();
        let parsed = AuditReplayArchive::from_canonical_text(&text).expect("parse");
        assert_eq!(parsed, archive);
        assert!(text.contains("controlled-observation"));
        assert!(!text.contains("ControlledObservationSink"));
    }

    #[test]
    fn controlled_observation_redact_policy_appends_redacted_event() {
        let mut trail = AuditTrail::new(sample_session());
        let result = apply_controlled_observation_audit_policy(
            &mut trail,
            0xfeed_beef,
            9,
            ControlledObservationAuditDecision::Redact,
            controlled_observation_linkage(),
        );

        assert_eq!(
            result,
            ControlledObservationAuditResult::Recorded(AuditEventId(0))
        );
        assert_eq!(trail.events().len(), 1);
        match &trail.events()[0].kind {
            AuditEventKind::ControlledObservation {
                payload_hash,
                redacted,
                policy,
                sequence_index,
                ..
            } => {
                assert_eq!(*payload_hash, None);
                assert!(*redacted);
                assert_eq!(*policy, ControlledObservationAuditDecision::Redact);
                assert_eq!(*sequence_index, 9);
            }
            other => panic!("unexpected event {other:?}"),
        }

        let archive = trail.replay_archive();
        let text = archive.to_canonical_text();
        let parsed = AuditReplayArchive::from_canonical_text(&text).expect("parse");
        assert_eq!(parsed, archive);
        assert!(text.contains("controlled-observation"));
    }

    #[test]
    fn controlled_observation_no_store_and_deny_are_explicit() {
        let mut no_store_trail = AuditTrail::new(sample_session());
        let no_store = apply_controlled_observation_audit_policy(
            &mut no_store_trail,
            123,
            1,
            ControlledObservationAuditDecision::NoStore,
            controlled_observation_linkage(),
        );
        assert_eq!(no_store, ControlledObservationAuditResult::NoStore);
        assert!(no_store_trail.events().is_empty());

        let mut deny_trail = AuditTrail::new(sample_session());
        let denied = apply_controlled_observation_audit_policy(
            &mut deny_trail,
            456,
            2,
            ControlledObservationAuditDecision::Deny,
            controlled_observation_linkage(),
        );
        assert_eq!(denied, ControlledObservationAuditResult::Denied);
        assert!(deny_trail.events().is_empty());
    }

    #[test]
    fn controlled_observation_archive_representation_is_deterministic() {
        let linkage = controlled_observation_linkage();

        let mut first = AuditTrail::new(sample_session());
        apply_controlled_observation_audit_policy(
            &mut first,
            0x1234,
            3,
            ControlledObservationAuditDecision::Record,
            linkage,
        );

        let mut second = AuditTrail::new(sample_session());
        apply_controlled_observation_audit_policy(
            &mut second,
            0x1234,
            3,
            ControlledObservationAuditDecision::Record,
            linkage,
        );

        assert_eq!(
            first.replay_archive().to_canonical_text(),
            second.replay_archive().to_canonical_text()
        );
    }

    #[test]
    fn replay_archive_roundtrips_controlled_observation_sink_capability_kind() {
        let mut trail = AuditTrail::new(sample_session());
        trail.record(AuditEventKind::CapabilityDenied {
            capability: CapabilityKind::ControlledObservationSink,
            call: Some("ControlledObservationSink".to_string()),
        });

        let archive = trail.replay_archive();
        let text = archive.to_canonical_text();
        let parsed = AuditReplayArchive::from_canonical_text(&text).expect("parse");

        assert_eq!(parsed, archive);
        assert!(text.contains("ControlledObservationSink"));
    }

    #[test]
    fn audit_trail_preserves_rule_activation_and_state_transition_events() {
        let mut trail = AuditTrail::new(sample_session());
        trail.record(AuditEventKind::RuleActivated {
            rule_id: "rule.alpha".to_string(),
            salience: 9,
        });
        trail.record(AuditEventKind::StateTransition {
            key: "fact.alpha".to_string(),
            from_epoch: 1,
            to_epoch: 2,
        });

        assert!(matches!(
            &trail.events()[0].kind,
            AuditEventKind::RuleActivated { rule_id, salience }
                if rule_id == "rule.alpha" && *salience == 9
        ));
        assert!(matches!(
            &trail.events()[1].kind,
            AuditEventKind::StateTransition {
                key,
                from_epoch,
                to_epoch
            } if key == "fact.alpha" && *from_epoch == 1 && *to_epoch == 2
        ));
    }

    #[test]
    fn replay_archive_uses_canonical_format_and_copies_trail_state() {
        let mut trail = AuditTrail::new(sample_session());
        trail.record(AuditEventKind::SessionStarted {
            entry: "main".to_string(),
        });
        trail.record(AuditEventKind::SessionFinished);

        let archive = trail.replay_archive();

        assert_eq!(archive.format_version, AUDIT_REPLAY_ARCHIVE_FORMAT_VERSION);
        assert_eq!(archive.session, trail.session().clone());
        assert_eq!(archive.events, trail.events());
        assert_eq!(archive.replay.event_count, 2);
        assert_eq!(archive.replay.last_event_id, Some(AuditEventId(1)));
    }

    #[test]
    fn replay_archive_roundtrips_through_canonical_text() {
        let mut trail = AuditTrail::new(sample_session());
        trail.record(AuditEventKind::SessionStarted {
            entry: "main\tentry".to_string(),
        });
        trail.record(AuditEventKind::CapabilityDenied {
            capability: CapabilityKind::StateUpdate,
            call: Some("StateUpdate".to_string()),
        });
        trail.record(AuditEventKind::StateTransition {
            key: "fact.alpha".to_string(),
            from_epoch: 2,
            to_epoch: 3,
        });
        trail.record(AuditEventKind::Note {
            message: "note:done".to_string(),
        });

        let archive = trail.replay_archive();
        let text = archive.to_canonical_text();
        let parsed = AuditReplayArchive::from_canonical_text(&text).expect("parse");

        assert_eq!(parsed, archive);
        let lines = text.lines().collect::<Vec<_>>();
        assert!(lines[3].contains("event\t0\tsession-started"));
        assert!(lines[6].contains("event\t3\tnote"));
    }

    #[test]
    fn replay_archive_rejects_event_count_mismatch() {
        let text = "\
semantic_audit_replay_archive\t2\n\
session\tkernel-bound\tprom.cap.manifest\tv1\ttrue\t250000\t32768\t256\t256\t8192\t16384\t4096\t16384\n\
events\t1\n\
replay\t0\tnone\n";

        let err = AuditReplayArchive::from_canonical_text(text).expect_err("must reject");

        assert!(err.message.contains("event count"));
    }

    #[test]
    fn replay_archive_rejects_non_monotonic_event_ids() {
        let text = "\
semantic_audit_replay_archive\t2\n\
session\tkernel-bound\tprom.cap.manifest\tv1\ttrue\t250000\t32768\t256\t256\t8192\t16384\t4096\t16384\n\
events\t1\n\
event\t9\tsession-finished\n\
replay\t1\t9\n";

        let err = AuditReplayArchive::from_canonical_text(text).expect_err("must reject");

        assert!(err.message.contains("monotonic"));
    }

    #[test]
    fn multi_session_replay_archive_uses_explicit_format_version() {
        let mut first = AuditTrail::new(sample_session());
        first.record(AuditEventKind::SessionStarted {
            entry: "alpha".to_string(),
        });

        let mut second = AuditTrail::new(sample_session());
        second.record(AuditEventKind::SessionStarted {
            entry: "beta".to_string(),
        });
        second.record(AuditEventKind::SessionFinished);

        let archive = MultiSessionReplayArchive::new(vec![
            MultiSessionReplayArchiveSession::new(0, first.replay_archive()),
            MultiSessionReplayArchiveSession::new(1, second.replay_archive()),
        ]);

        assert_eq!(
            archive.format_version,
            MULTI_SESSION_REPLAY_ARCHIVE_FORMAT_VERSION
        );
        assert_eq!(archive.sessions.len(), 2);
        assert_eq!(archive.sessions[0].session_ordinal, 0);
        assert_eq!(archive.sessions[1].session_ordinal, 1);
    }

    #[test]
    fn multi_session_replay_archive_roundtrips_through_canonical_text() {
        let mut first = AuditTrail::new(sample_session());
        first.record(AuditEventKind::SessionStarted {
            entry: "alpha".to_string(),
        });
        first.record(AuditEventKind::Note {
            message: "first:done".to_string(),
        });

        let mut second = AuditTrail::new(sample_session());
        second.record(AuditEventKind::SessionStarted {
            entry: "beta".to_string(),
        });
        second.record(AuditEventKind::StateTransition {
            key: "fact.beta".to_string(),
            from_epoch: 2,
            to_epoch: 3,
        });

        let archive = MultiSessionReplayArchive::new(vec![
            MultiSessionReplayArchiveSession::new(0, first.replay_archive()),
            MultiSessionReplayArchiveSession::new(1, second.replay_archive()),
        ]);

        let text = archive.to_canonical_text();
        let parsed = MultiSessionReplayArchive::from_canonical_text(&text).expect("parse");

        assert_eq!(parsed, archive);
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines[0], "semantic_multi_session_replay_archive\t1");
        assert_eq!(lines[1], "sessions\t2");
        assert!(lines[2].starts_with("session\t0\t"));
        assert!(lines[3].starts_with("archive\tsemantic_audit_replay_archive\t2"));
    }

    #[test]
    fn multi_session_replay_archive_rejects_session_count_mismatch() {
        let text = "\
semantic_multi_session_replay_archive\t1\n\
sessions\t2\n\
session\t0\t4\n\
archive\tsemantic_audit_replay_archive\t2\n\
archive\tsession\tkernel-bound\tprom.cap.manifest\tv1\ttrue\t250000\t32768\t256\t256\t8192\t16384\t4096\t16384\n\
archive\tevents\t0\n\
archive\treplay\t0\tnone\n";

        let err = MultiSessionReplayArchive::from_canonical_text(text).expect_err("must reject");

        assert!(err.message.contains("session count mismatch"));
    }

    #[test]
    fn multi_session_replay_archive_rejects_non_monotonic_session_ordinals() {
        let text = "\
semantic_multi_session_replay_archive\t1\n\
sessions\t1\n\
session\t9\t4\n\
archive\tsemantic_audit_replay_archive\t2\n\
archive\tsession\tkernel-bound\tprom.cap.manifest\tv1\ttrue\t250000\t32768\t256\t256\t8192\t16384\t4096\t16384\n\
archive\tevents\t0\n\
archive\treplay\t0\tnone\n";

        let err = MultiSessionReplayArchive::from_canonical_text(text).expect_err("must reject");

        assert!(err.message.contains("monotonic"));
    }

    // #1762 (FA-08-004): a custom envelope whose eight values cannot
    // plausibly be mistaken for any baseline profile - makes any
    // re-derivation-from-context or field-order mistake obvious.
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

    fn custom_session(context: ExecutionContext) -> AuditSessionMetadata {
        AuditSessionMetadata {
            context,
            quotas: custom_quota_envelope(),
            capability_manifest: CapabilityManifestMetadata {
                schema: "prom.cap.manifest".to_string(),
                version: prom_cap::CapabilityManifestVersion::V1,
            },
            gate_registry_bound: false,
        }
    }

    // #1762 primary regression: the distinct custom envelope must survive
    // AuditSessionMetadata -> AuditReplayArchive -> canonical text ->
    // parsed archive -> ReplayMetadata bit-for-bit, with no re-derivation
    // from `context` anywhere in the pipeline.
    #[test]
    fn effective_quota_envelope_survives_full_canonical_pipeline() {
        let mut trail = AuditTrail::new(custom_session(ExecutionContext::VerifiedLocal));
        trail.record(AuditEventKind::SessionStarted {
            entry: "main".to_string(),
        });

        let archive = trail.replay_archive();
        assert_eq!(archive.session.quotas, custom_quota_envelope());

        let text = archive.to_canonical_text();
        let parsed = AuditReplayArchive::from_canonical_text(&text).expect("parse");

        assert_eq!(parsed, archive);
        assert_eq!(parsed.session.quotas, custom_quota_envelope());
        assert_eq!(parsed.replay.session.quotas, custom_quota_envelope());
    }

    // #1762: exact wire shape - 13 tokens, quota fields in RuntimeQuotas
    // struct declaration order.
    #[test]
    fn canonical_session_line_emits_thirteen_tokens_in_frozen_quota_order() {
        let trail = AuditTrail::new(custom_session(ExecutionContext::VerifiedLocal));
        let text = trail.replay_archive().to_canonical_text();
        let session_line = text.lines().nth(1).expect("session line");

        assert_eq!(
            session_line,
            "session\tverified-local\tprom.cap.manifest\tv1\tfalse\t101\t102\t103\t104\t105\t106\t107\t108"
        );
        assert_eq!(session_line.split('\t').count(), 13);
    }

    // #1762 (§12): ReplayMetadata does not own a second, independently
    // derived quota field - `archive.session.quotas` and
    // `archive.replay.session.quotas` must be the identical value, both
    // immediately after construction and after a canonical round-trip.
    #[test]
    fn replay_metadata_session_matches_archive_session_including_quotas() {
        let mut trail = AuditTrail::new(custom_session(ExecutionContext::KernelBound));
        trail.record(AuditEventKind::SessionFinished);

        let archive = trail.replay_archive();
        assert_eq!(archive.session.quotas, archive.replay.session.quotas);

        let parsed =
            AuditReplayArchive::from_canonical_text(&archive.to_canonical_text()).expect("parse");
        assert_eq!(parsed.session.quotas, parsed.replay.session.quotas);
        assert_eq!(parsed.session.quotas, custom_quota_envelope());
    }

    // #1762 (§10): a legacy v1 archive (5-token session line, no quota
    // data ever recorded) must be rejected by the v2 reader through the
    // existing version-mismatch channel - never silently backfilled with
    // a context-derived baseline.
    #[test]
    fn replay_archive_rejects_legacy_v1_format_version() {
        let text = "\
semantic_audit_replay_archive\t1\n\
session\tkernel-bound\tprom.cap.manifest\tv1\ttrue\n\
events\t0\n\
replay\t0\tnone\n";

        let err = AuditReplayArchive::from_canonical_text(text).expect_err("must reject v1");

        assert!(err.message.contains("unsupported archive format version 1"));
        assert!(err.message.contains("expected 2"));
    }

    // #1762 (§9): a malformed quota numeric field must fail deterministically
    // through the existing `AuditReplayArchiveFormatError` channel, exactly
    // like any other malformed numeric field in this format.
    #[test]
    fn replay_archive_rejects_malformed_quota_numeric_field() {
        let text = "\
semantic_audit_replay_archive\t2\n\
session\tverified-local\tprom.cap.manifest\tv1\tfalse\t101\tnot-a-number\t103\t104\t105\t106\t107\t108\n\
events\t0\n\
replay\t0\tnone\n";

        let err = AuditReplayArchive::from_canonical_text(text).expect_err("must reject");

        assert!(err.message.contains("max_calls"));
    }

    // #1762 (§9): no permissive parser - a session line with one field too
    // few or too many must reject, not silently truncate/pad.
    #[test]
    fn replay_archive_rejects_session_line_with_wrong_token_count() {
        let too_few = "\
semantic_audit_replay_archive\t2\n\
session\tverified-local\tprom.cap.manifest\tv1\tfalse\t101\t102\t103\t104\t105\t106\t107\n\
events\t0\n\
replay\t0\tnone\n";
        let err =
            AuditReplayArchive::from_canonical_text(too_few).expect_err("12 tokens must reject");
        assert!(err.message.contains("invalid session line"));

        let too_many = "\
semantic_audit_replay_archive\t2\n\
session\tverified-local\tprom.cap.manifest\tv1\tfalse\t101\t102\t103\t104\t105\t106\t107\t108\t109\n\
events\t0\n\
replay\t0\tnone\n";
        let err =
            AuditReplayArchive::from_canonical_text(too_many).expect_err("14 tokens must reject");
        assert!(err.message.contains("invalid session line"));
    }

    // #1762 (§11, §16): the outer multi-session format stays at version 1
    // while its embedded archives are independently versioned v2 - proving
    // the frozen "outer version unchanged" decision round-trips for real,
    // not just in principle.
    #[test]
    fn multi_session_replay_archive_keeps_outer_v1_with_embedded_v2_archives() {
        let mut first = AuditTrail::new(custom_session(ExecutionContext::VerifiedLocal));
        first.record(AuditEventKind::SessionStarted {
            entry: "alpha".to_string(),
        });

        let mut second = AuditTrail::new(sample_session());
        second.record(AuditEventKind::SessionFinished);

        let archive = MultiSessionReplayArchive::new(vec![
            MultiSessionReplayArchiveSession::new(0, first.replay_archive()),
            MultiSessionReplayArchiveSession::new(1, second.replay_archive()),
        ]);
        assert_eq!(archive.format_version, 1);
        assert_eq!(archive.sessions[0].archive.format_version, 2);
        assert_eq!(archive.sessions[1].archive.format_version, 2);

        let text = archive.to_canonical_text();
        assert!(text.starts_with("semantic_multi_session_replay_archive\t1\n"));
        assert!(text.contains("archive\tsemantic_audit_replay_archive\t2\n"));

        let parsed = MultiSessionReplayArchive::from_canonical_text(&text).expect("parse");
        assert_eq!(parsed, archive);
        assert_eq!(
            parsed.sessions[0].archive.session.quotas,
            custom_quota_envelope()
        );
    }

    // #1762 (§16.D): the especially valuable regression - two sessions
    // sharing the identical ExecutionContext but carrying different
    // RuntimeQuotas must remain distinguishable after a full multi-session
    // canonical round-trip. Same label, different effective envelope,
    // provenance tells them apart.
    #[test]
    fn multi_session_replay_archive_distinguishes_same_context_different_quotas() {
        let mut baseline = AuditTrail::new(AuditSessionMetadata {
            context: ExecutionContext::VerifiedLocal,
            quotas: RuntimeQuotas::verified_local(),
            capability_manifest: CapabilityManifestMetadata {
                schema: "prom.cap.manifest".to_string(),
                version: prom_cap::CapabilityManifestVersion::V1,
            },
            gate_registry_bound: false,
        });
        baseline.record(AuditEventKind::SessionStarted {
            entry: "main".to_string(),
        });

        let mut tightened = AuditTrail::new(AuditSessionMetadata {
            context: ExecutionContext::VerifiedLocal,
            quotas: RuntimeQuotas {
                max_effect_calls: 1,
                ..RuntimeQuotas::verified_local()
            },
            capability_manifest: CapabilityManifestMetadata {
                schema: "prom.cap.manifest".to_string(),
                version: prom_cap::CapabilityManifestVersion::V1,
            },
            gate_registry_bound: false,
        });
        tightened.record(AuditEventKind::SessionStarted {
            entry: "main".to_string(),
        });

        let archive = MultiSessionReplayArchive::new(vec![
            MultiSessionReplayArchiveSession::new(0, baseline.replay_archive()),
            MultiSessionReplayArchiveSession::new(1, tightened.replay_archive()),
        ]);
        let parsed = MultiSessionReplayArchive::from_canonical_text(&archive.to_canonical_text())
            .expect("parse");

        assert_eq!(
            parsed.sessions[0].archive.session.context,
            parsed.sessions[1].archive.session.context
        );
        assert_eq!(
            parsed.sessions[0].archive.session.quotas,
            RuntimeQuotas::verified_local()
        );
        assert_eq!(
            parsed.sessions[1].archive.session.quotas.max_effect_calls,
            1
        );
        assert_ne!(
            parsed.sessions[0].archive.session.quotas,
            parsed.sessions[1].archive.session.quotas
        );
    }

    // #1762 (§11, §16.E): an old inner v1 archive embedded under an
    // (unchanged) outer v1 multi-session archive must still be rejected -
    // proving `MultiSessionReplayArchive` genuinely delegates to
    // `AuditReplayArchive::from_canonical_text` rather than duplicating a
    // now-stale copy of the session-line parsing logic.
    #[test]
    fn multi_session_replay_archive_rejects_legacy_v1_embedded_archive() {
        let text = "\
semantic_multi_session_replay_archive\t1\n\
sessions\t1\n\
session\t0\t4\n\
archive\tsemantic_audit_replay_archive\t1\n\
archive\tsession\tkernel-bound\tprom.cap.manifest\tv1\ttrue\n\
archive\tevents\t0\n\
archive\treplay\t0\tnone\n";

        let err = MultiSessionReplayArchive::from_canonical_text(text)
            .expect_err("embedded v1 archive must reject");

        assert!(err.message.contains("unsupported archive format version 1"));
        assert!(err.message.contains("expected 2"));
    }
}
