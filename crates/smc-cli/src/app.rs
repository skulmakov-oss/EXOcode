use crate::application_host::CliApplicationHost;
use crate::executable_bundle::read_source_with_package_admission;
use crate::incremental::{
    emit_trace, module_graph_fingerprint, module_graph_module_count, read_graph_hash,
    update_cache_index, CacheEvent, CacheReason, ModuleGraphSnapshot,
};
use crate::package_manifest::{
    admit_package_entry_module, inspect_local_package_graph, reset_declared_dependency_graph_cache,
    reset_pinned_dependency_fingerprint_cache, resolve_package_import_path,
    resolve_project_root_check_entry,
};
use crate::{format_path, FormatterMode};
use prom_audit::hello_observation_audit::{
    apply_controlled_observation_audit_policy, ControlledObservationAuditDecision,
    ControlledObservationAuditResult, HelloObservationAuditLinkage,
};
use prom_audit::{AuditEventId, AuditSessionMetadata, AuditTrail};
use prom_cap::{
    hello_observation_capability::{
        require_hello_observation_sink_capability, HelloObservationCapabilityContext,
        HelloObservationCapabilityDecision,
    },
    ApplicationCapabilityProfile, CapabilityKind, CapabilityManifest,
};
use sm_emit::{
    compile_program_to_semcode, compile_program_to_semcode_with_options_debug, CompileProfile,
    OptLevel,
};
use sm_front::{lex, parse_logos_program_with_profile, parse_program_with_profile, ParserProfile};
use sm_ir::{compile_program_to_ir_with_options_and_profile, lower_logos_laws_to_ir};
use sm_runtime_core::hello_observation_sink::HelloObservationClass;
use sm_runtime_core::{ExecutionConfig, ExecutionContext};
use sm_sema::{check_file_with_provider_and_profile, check_source_with_profile, ModuleProvider};
use sm_verify::{verify_semcode, verify_semcode_token, verify_semcode_token_with_quotas};
use sm_vm::{
    disasm_semcode, run_semcode_collecting_hello_observations_with_config,
    run_verified_entry_semcode_with_application_host_and_capabilities_and_config, RuntimeError,
};
use std::collections::HashSet;
use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant};
use ton618_core::diagnostics::diagnostic_catalog;

struct CliFsModuleProvider;

impl ModuleProvider for CliFsModuleProvider {
    fn read_module(&self, module_id: &str) -> Result<Vec<u8>, String> {
        ensure_package_module_admission(Path::new(module_id))?;
        std::fs::read(module_id).map_err(|e| e.to_string())
    }

    fn resolve_import(&self, importer_module_id: &str, spec: &str) -> Result<String, String> {
        resolve_package_import_path(Path::new(importer_module_id), spec)
            .map(|path| {
                let text = path.to_string_lossy();
                // Only fold '\' into '/' on Windows -- on Unix it is an ordinary filename
                // character, and resolve_package_import_path already preserves it, so
                // folding it here would make this module_id identify a different file
                // than the one that was actually resolved (same class as DL-012/DL-016).
                if cfg!(windows) {
                    text.replace('\\', "/")
                } else {
                    text.into_owned()
                }
            })
            .map_err(|e| e.to_string())
    }
}

fn ensure_package_module_admission(path: &Path) -> Result<(), String> {
    admit_package_entry_module(path)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn cli_profile() -> ParserProfile {
    ParserProfile::foundation_default()
}

fn reject_leading_unknown_flag(input: &str) -> Result<(), String> {
    if input.starts_with('-') {
        Err(format!("unknown flag '{}'", input))
    } else {
        Ok(())
    }
}

pub fn main_entry() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

pub fn run(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() {
        return Err(usage());
    }
    match args[0].as_str() {
        "compile" => cmd_compile(&args[1..]),
        "check" => cmd_check(&args[1..]),
        "lint" => cmd_lint(&args[1..]),
        "watch" => cmd_watch(&args[1..]),
        "fmt" => cmd_fmt(&args[1..]),
        "dump-ast" => cmd_dump_ast(&args[1..]),
        "dump-ir" => cmd_dump_ir(&args[1..]),
        "dump-bytecode" => cmd_dump_bytecode(&args[1..]),
        "hash-ast" => cmd_hash_ast(&args[1..]),
        "hash-ir" => cmd_hash_ir(&args[1..]),
        "hash-smc" => cmd_hash_smc(&args[1..]),
        "snapshots" => cmd_snapshots(&args[1..]),
        "features" => cmd_features(&args[1..]),
        "explain" => cmd_explain(&args[1..]),
        "repl" => cmd_repl(&args[1..]),
        "verify" => cmd_verify(&args[1..]),
        "test" => cmd_test(&args[1..]),
        "package" => cmd_package(&args[1..]),
        "run" => cmd_run(&args[1..]),
        "run-smc" => cmd_run_smc(&args[1..]),
        "disasm" => cmd_disasm(&args[1..]),
        "work" => cmd_work(&args[1..]),
        "look" => crate::ui_frame_inspect::cmd_look(&args[1..]),
        "hub" => crate::hub::cmd_hub(&args[1..]),
        "help" | "--help" | "-h" => {
            println!("{}", usage());
            Ok(())
        }
        other => Err(format!("unknown command '{}'\n\n{}", other, usage())),
    }
}

fn cmd_package(args: &[String]) -> Result<(), String> {
    if args.len() != 2 || args[0] != "inspect" {
        return Err("usage: smc package inspect <project-root>".to_string());
    }
    reject_leading_unknown_flag(&args[1])?;
    let root = Path::new(&args[1]);
    if !root.is_dir() {
        return Err(format!(
            "package inspect project root '{}' is not a directory",
            root.display()
        ));
    }
    println!("{}", inspect_local_package_graph(root)?);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct WorkControlFrame {
    pub subject: String,
    pub intent: String,
    pub target: Option<String>,
    pub profile: Option<String>,
}

fn cmd_work(args: &[String]) -> Result<(), String> {
    let frame = parse_work_control_frame(args)?;
    let subject = frame.subject.clone();

    match frame.intent.as_str() {
        "check" => cmd_check(&[subject]),
        "prove" => cmd_work_prove(&subject, frame.profile.as_deref()),
        "wake" => cmd_work_wake(&subject),
        "seal" => cmd_work_seal(&frame),
        _ => Err(format!(
            "Unknown intent '{}'. Did you mean 'work <subject> prove' or 'work <subject> seal'?",
            frame.intent
        )),
    }
}

fn parse_work_control_frame(args: &[String]) -> Result<WorkControlFrame, String> {
    if args.len() < 2 {
        return Err(
            "usage: smc work <subject> <intent> [to <target>] [with <profile>]".to_string(),
        );
    }
    let subject = args[0].clone();
    let intent = args[1].clone();
    let mut target = None;
    let mut profile = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "to" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "missing target after 'to'".to_string())?;
                target = Some(value.clone());
                i += 2;
            }
            "with" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "missing profile after 'with'".to_string())?;
                profile = Some(value.clone());
                i += 2;
            }
            other => {
                return Err(format!(
                    "Unexpected token '{}'. The grammar supports: work <subject> <intent> [to <target>] [with <profile>]",
                    other
                ));
            }
        }
    }

    Ok(WorkControlFrame {
        subject,
        intent,
        target,
        profile,
    })
}

fn work_subject_path(subject: &str) -> Result<PathBuf, String> {
    let input_path = Path::new(subject);
    if input_path.is_dir() {
        resolve_project_root_check_entry(input_path)
    } else {
        Ok(input_path.to_path_buf())
    }
}

fn work_is_semcode_artifact(subject: &str) -> bool {
    Path::new(subject)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("smc"))
}

fn cmd_work_prove(subject: &str, _profile: Option<&str>) -> Result<(), String> {
    if work_is_semcode_artifact(subject) {
        return cmd_verify(&[subject.to_string()]);
    }

    let root = work_subject_path(subject)?;
    let src = read_source_with_package_admission(&root)?;
    let bytes = compile_program_to_semcode_with_options_debug(
        &src,
        CompileProfile::Auto,
        OptLevel::O0,
        false,
    )
    .map_err(|e| e.to_string())?;
    let verified = verify_semcode(&bytes).map_err(|report| report.to_string())?;
    println!(
        "verified '{}' ({} function(s), header={}, epoch={}.{})",
        subject,
        verified.functions.len(),
        String::from_utf8_lossy(&verified.header.magic),
        verified.header.epoch,
        verified.header.rev
    );
    Ok(())
}

fn cmd_work_wake(subject: &str) -> Result<(), String> {
    if work_is_semcode_artifact(subject) {
        cmd_run_smc(&[subject.to_string()])
    } else {
        cmd_run(&[subject.to_string()])
    }
}

fn cmd_work_seal(frame: &WorkControlFrame) -> Result<(), String> {
    if work_is_semcode_artifact(&frame.subject) {
        return Err("work seal expects source input, not SemCode".to_string());
    }
    let target = frame
        .target
        .as_ref()
        .ok_or_else(|| "missing target after 'to'".to_string())?;

    let mut args = vec![frame.subject.clone(), "-o".to_string(), target.clone()];
    if let Some(profile) = &frame.profile {
        args.push("--profile".to_string());
        args.push(profile.clone());
    }
    cmd_compile(&args)
}

fn cmd_compile(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "usage: smc compile <input.sm|project-root> -o <out.smc> [--profile auto|rust] [--opt-level O0|O1] [--debug-symbols] [--metrics]"
                .to_string(),
        );
    }
    let input = args[0].as_str();
    reject_leading_unknown_flag(input)?;
    if args.len() < 3 {
        return Err(
            "usage: smc compile <input.sm|project-root> -o <out.smc> [--profile auto|rust] [--opt-level O0|O1] [--debug-symbols] [--metrics]"
                .to_string(),
        );
    }
    let mut out: Option<&str> = None;
    let mut metrics = false;
    let mut profile = CompileProfile::Auto;
    let mut opt = OptLevel::O0;
    let mut debug_symbols = false;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                i += 1;
                out = args.get(i).map(|s| s.as_str());
            }
            "--profile" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --profile".to_string())?;
                profile = parse_compile_profile(v)?;
            }
            "--opt-level" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --opt-level".to_string())?;
                opt = parse_opt_level(v)?;
            }
            "--opt" => {
                opt = OptLevel::O1;
            }
            "--debug-symbols" => {
                debug_symbols = true;
            }
            "--metrics" => {
                metrics = true;
            }
            other => return Err(format!("unknown flag '{}'", other)),
        }
        i += 1;
    }
    let out = out.ok_or_else(|| "missing -o <out.smc>".to_string())?;
    let t0 = Instant::now();
    let input_path = Path::new(input);
    let root = if input_path.is_dir() {
        resolve_project_root_check_entry(input_path)?
    } else {
        input_path.to_path_buf()
    };
    let src = read_source_with_package_admission(&root)?;
    let t_read = Instant::now();
    let parser_profile = cli_profile();
    let bytes = compile_program_to_semcode_with_options_debug(&src, profile, opt, debug_symbols)
        .map_err(|e| e.to_string())?;
    let t_compile = Instant::now();
    std::fs::write(out, &bytes).map_err(|e| format!("failed to write '{}': {}", out, e))?;
    let t_write = Instant::now();
    println!("compiled '{}' -> '{}' ({} bytes)", input, out, bytes.len());
    if debug_symbols {
        println!(
            "note: --debug-symbols requested (debug section emission reserved for next revision)"
        );
    }
    if metrics {
        let token_count = lex(&src).map(|t| t.len()).unwrap_or(0);
        let mut fn_count = 0usize;
        let mut expr_count = 0usize;
        let mut stmt_count = 0usize;
        let mut symbol_count = 0usize;
        if let Ok(p) = parse_program_with_profile(&src, &parser_profile) {
            fn_count = p.functions.len();
            expr_count = p.arena.expr_count();
            stmt_count = p.arena.stmt_count();
            symbol_count = p.arena.symbol_count();
        }
        let mut ir_func_count = 0usize;
        let mut ir_instr_count = 0usize;
        if let Ok(ir) =
            compile_program_to_ir_with_options_and_profile(&src, profile, opt, &parser_profile)
        {
            ir_func_count = ir.len();
            ir_instr_count = ir.iter().map(|f| f.instrs.len()).sum();
        }
        println!(
            "metrics: read={}ms compile={}ms write={}ms total={}ms tokens={} fns={} exprs={} stmts={} symbols={} ir_funcs={} ir_instrs={} exb_bytes={} hash={:016x}",
            (t_read - t0).as_millis(),
            (t_compile - t_read).as_millis(),
            (t_write - t_compile).as_millis(),
            (t_write - t0).as_millis(),
            token_count,
            fn_count,
            expr_count,
            stmt_count,
            symbol_count,
            ir_func_count,
            ir_instr_count,
            bytes.len(),
            fnv1a64(&bytes)
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColorMode {
    Auto,
    Always,
    Never,
}

fn parse_color_mode(v: &str) -> Result<ColorMode, String> {
    match v.to_ascii_lowercase().as_str() {
        "auto" => Ok(ColorMode::Auto),
        "always" => Ok(ColorMode::Always),
        "never" => Ok(ColorMode::Never),
        _ => Err(format!(
            "invalid --color '{}', expected auto|always|never",
            v
        )),
    }
}

fn resolve_color_mode(mode: ColorMode) -> bool {
    match mode {
        ColorMode::Always => true,
        ColorMode::Never => false,
        ColorMode::Auto => env::var("NO_COLOR").is_err(),
    }
}

fn color_wrap(enabled: bool, s: &str, code: &str) -> String {
    if enabled {
        format!("\x1b[{}m{}\x1b[0m", code, s)
    } else {
        s.to_string()
    }
}

fn print_diag_colored(enabled: bool, text: &str) {
    let mut out = text.to_string();
    out = out.replace(
        "Error [",
        &format!("{}[", color_wrap(enabled, "Error", "31;1")),
    );
    out = out.replace(
        "Warning [",
        &format!("{}[", color_wrap(enabled, "Warning", "33;1")),
    );
    out = out.replace(
        "help:",
        &format!("{}:", color_wrap(enabled, "help", "36;1")),
    );
    eprintln!("{}", out.trim_end());
}

fn cmd_check(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "usage: smc check <input.sm|project-root> [--no-cache] [--trace-cache] [--metrics] [--deny warnings|<CODE>]"
                .to_string(),
        );
    }
    let input = args[0].as_str();
    reject_leading_unknown_flag(input)?;
    let input_path = Path::new(input);
    let root = if input_path.is_dir() {
        resolve_project_root_check_entry(input_path)?
    } else {
        input_path.to_path_buf()
    };
    let mut no_cache = false;
    let mut metrics = false;
    let mut trace_cache_enabled = false;
    let mut color = ColorMode::Auto;
    let mut deny = DenyPolicy::default();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--no-cache" => no_cache = true,
            "--metrics" => metrics = true,
            "--trace-cache" => trace_cache_enabled = true,
            "--color" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --color".to_string())?;
                color = parse_color_mode(v)?;
            }
            "--deny" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --deny".to_string())?;
                parse_deny_value(v, &mut deny);
            }
            other => return Err(format!("unknown flag '{}'", other)),
        }
        i += 1;
    }
    if deny.has_rules() {
        no_cache = true;
        trace_cache(
            trace_cache_enabled,
            CacheEvent::Invalidate,
            CacheReason::DenyPolicy,
            &root,
            "SEMP",
            "",
        );
    }
    let t0 = Instant::now();
    let src = read_source_with_package_admission(&root)?;
    let t_read = Instant::now();
    let prev_graph_hash = read_graph_hash(Path::new(CACHE_GRAPH_FILE));
    let mut graph_hash_now = None;
    if let Ok(snapshot) = ModuleGraphSnapshot::read_from_root(&root) {
        graph_hash_now = Some(snapshot.hash(CACHE_SCHEMA_VERSION));
        let _ = snapshot.write_to(Path::new(CACHE_GRAPH_FILE), CACHE_SCHEMA_VERSION);
    }
    if !no_cache {
        if let Ok(fp) = module_graph_fingerprint(&root, CACHE_SCHEMA_VERSION) {
            let cache_path = cache_file_for_root(&root)?;
            match load_cache_entry_ex(&cache_path, fp) {
                Ok(CacheLookup::Hit(cached)) => {
                    let key = format!("{:016x}", fp);
                    trace_cache(
                        trace_cache_enabled,
                        CacheEvent::Hit,
                        CacheReason::Reused,
                        &root,
                        "SEMP",
                        &key,
                    );
                    println!(
                        "smc check passed (cached): {} warning(s), {} scheduled law(s)",
                        cached.warning_count, cached.law_count
                    );
                    for w in cached.warnings {
                        eprintln!("{}", w.trim_end());
                    }
                    if metrics {
                        let t1 = Instant::now();
                        let token_count = lex(&src).map(|t| t.len()).unwrap_or(0);
                        println!(
                            "metrics: read={}ms check={}ms total={}ms cached=1 tokens={}",
                            (t_read - t0).as_millis(),
                            (t1 - t_read).as_millis(),
                            (t1 - t0).as_millis(),
                            token_count
                        );
                    }
                    return Ok(());
                }
                Ok(CacheLookup::Miss(reason)) => {
                    let key = format!("{:016x}", fp);
                    trace_cache(
                        trace_cache_enabled,
                        CacheEvent::Miss,
                        reason,
                        &root,
                        "SEMP",
                        &key,
                    );
                }
                Err(_) => {}
            }
        }
    } else {
        trace_cache(
            trace_cache_enabled,
            CacheEvent::Miss,
            CacheReason::CacheDisabled,
            &root,
            "SEMP",
            "",
        );
    }

    let provider = CliFsModuleProvider;
    let parser_profile = cli_profile();
    let root_canon = root
        .canonicalize()
        .map_err(|e| format!("failed to resolve '{}': {}", root.display(), e))?;
    let report = check_file_with_provider_and_profile(&root_canon, &provider, &parser_profile)
        .or_else(|_| check_source_with_profile(&src, &parser_profile))
        .map_err(|e| e.to_string())?;
    let t_check = Instant::now();
    let color_enabled = resolve_color_mode(color);
    for w in &report.warnings {
        print_diag_colored(color_enabled, &w.rendered);
    }
    println!(
        "smc check passed: {} warning(s), {} scheduled law(s)",
        report.warnings.len(),
        report.scheduled_laws.len()
    );
    if !no_cache {
        if let Ok(fp) = module_graph_fingerprint(&root, CACHE_SCHEMA_VERSION) {
            let cache_path = cache_file_for_root(&root)?;
            let entry = CacheEntry {
                fingerprint: fp,
                warning_count: report.warnings.len(),
                law_count: report.scheduled_laws.len(),
                warnings: report.warnings.iter().map(|w| w.rendered.clone()).collect(),
            };
            let _ = save_cache_entry(&cache_path, &entry);
            let mc = module_graph_module_count(&root).unwrap_or(1);
            let _ = update_cache_index(Path::new(CACHE_INDEX_FILE), &root, fp, graph_hash_now, mc);
            if trace_cache_enabled {
                if prev_graph_hash != graph_hash_now {
                    trace_cache(
                        true,
                        CacheEvent::Invalidate,
                        CacheReason::GraphChanged,
                        &root,
                        "GRAPH",
                        &format!("{:016x}", fp),
                    );
                }
            }
        }
    }
    if metrics {
        let t_end = Instant::now();
        let token_count = lex(&src).map(|t| t.len()).unwrap_or(0);
        let module_count = module_graph_module_count(&root).unwrap_or(1);
        println!(
            "metrics: read={}ms check={}ms cache_write={}ms total={}ms cached=0 tokens={} modules={} warnings={} scheduled_laws={} arena_nodes={}",
            (t_read - t0).as_millis(),
            (t_check - t_read).as_millis(),
            (t_end - t_check).as_millis(),
            (t_end - t0).as_millis(),
            token_count,
            module_count,
            report.warnings.len(),
            report.scheduled_laws.len(),
            report.arena_nodes
        );
    }
    let denied = collect_denied_warning_lines(&report, &deny);
    if !denied.is_empty() {
        return Err(format!(
            "check failed by deny policy ({}):\n{}",
            denied.len(),
            denied.join("\n")
        ));
    }
    Ok(())
}

fn cmd_watch(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "usage: smc watch <input.sm> [--metrics] [--color auto|always|never]".to_string(),
        );
    }
    let root = PathBuf::from(&args[0]);
    let mut metrics = false;
    let mut color = ColorMode::Auto;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--metrics" => metrics = true,
            "--color" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --color".to_string())?;
                color = parse_color_mode(v)?;
            }
            other => return Err(format!("unknown flag '{}'", other)),
        }
        i += 1;
    }
    let color_enabled = resolve_color_mode(color);
    println!("watching '{}'", root.display());
    let mut last_fp: Option<u64> = None;
    let mut last_snapshot: Option<String> = None;
    loop {
        match module_graph_fingerprint(&root, CACHE_SCHEMA_VERSION) {
            Ok(fp) => {
                if last_fp != Some(fp) {
                    let t0 = Instant::now();
                    // Reset only when the fingerprint actually changed, not on every idle
                    // tick: `module_graph_fingerprint` now folds each module's governing
                    // manifest content into its hash (see collect_module_graph in
                    // incremental.rs), so a change to a declared dependency in that
                    // manifest already changes `fp` and reaches this branch -- the
                    // unconditional-every-tick reset DL-022 introduced was itself a real
                    // regression (rehashing every declared package's full content on every
                    // idle tick, not just real rebuilds). See DL-023.
                    reset_pinned_dependency_fingerprint_cache();
                    reset_declared_dependency_graph_cache();
                    last_fp = Some(fp);
                    let src = match read_source_with_package_admission(&root) {
                        Ok(s) => s,
                        Err(e) => {
                            let snap = format!("error: {}", e);
                            let changed = last_snapshot
                                .as_ref()
                                .map(|prev| prev != &snap)
                                .unwrap_or(true);
                            if changed {
                                println!("{snap}");
                                last_snapshot = Some(snap);
                            } else {
                                println!("change detected, smc output unchanged");
                            }
                            thread::sleep(Duration::from_millis(600));
                            continue;
                        }
                    };
                    let provider = CliFsModuleProvider;
                    let parser_profile = cli_profile();
                    let snapshot = match root
                        .canonicalize()
                        .map_err(|e| e.to_string())
                        .and_then(|p| {
                            check_file_with_provider_and_profile(&p, &provider, &parser_profile)
                                .map_err(|e| e.to_string())
                        })
                        .or_else(|_| {
                            check_source_with_profile(&src, &parser_profile)
                                .map_err(|e| e.to_string())
                        }) {
                        Ok(report) => {
                            let mut out = String::new();
                            for w in &report.warnings {
                                out.push_str(w.rendered.trim_end());
                                out.push('\n');
                            }
                            out.push_str(&format!(
                                "ok: {} warning(s), {} scheduled law(s)",
                                report.warnings.len(),
                                report.scheduled_laws.len()
                            ));
                            out
                        }
                        Err(e) => format!("{e}"),
                    };
                    let changed = last_snapshot
                        .as_ref()
                        .map(|prev| prev != &snapshot)
                        .unwrap_or(true);
                    if changed {
                        if snapshot.starts_with("Error [") || snapshot.starts_with("Warning [") {
                            print_diag_colored(color_enabled, &snapshot);
                        } else {
                            println!("{snapshot}");
                        }
                        last_snapshot = Some(snapshot);
                        if metrics {
                            let t1 = Instant::now();
                            let token_count = lex(&src).map(|t| t.len()).unwrap_or(0);
                            let modules = module_graph_module_count(&root).unwrap_or(1);
                            println!(
                                "metrics: total={}ms tokens={} modules={} fingerprint={:016x}",
                                (t1 - t0).as_millis(),
                                token_count,
                                modules,
                                fp
                            );
                        }
                    } else {
                        println!("change detected, smc output unchanged");
                    }
                }
            }
            Err(e) => eprintln!("{e}"),
        }
        thread::sleep(Duration::from_millis(600));
    }
}

fn cmd_fmt(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("usage: smc fmt [--check] <path>".to_string());
    }

    let mut check = false;
    let mut target: Option<&str> = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--check" => check = true,
            value if value.starts_with('-') => {
                return Err(format!("unknown flag '{}'", value));
            }
            value => {
                if target.is_some() {
                    return Err("usage: smc fmt [--check] <path>".to_string());
                }
                target = Some(value);
            }
        }
        i += 1;
    }

    let target = target.ok_or_else(|| "usage: smc fmt [--check] <path>".to_string())?;
    let target_path = Path::new(target);
    let mode = if check {
        FormatterMode::Check
    } else {
        FormatterMode::Write
    };
    let summary = format_path(target_path, mode)?;
    let display = target_path.display();

    if check {
        if summary.files_changed == 0 {
            println!(
                "format check passed: '{}' ({} file(s) scanned)",
                display, summary.files_scanned
            );
            return Ok(());
        }

        let changed = summary
            .changed_paths
            .iter()
            .map(|path| format!("  {}", path.display()))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!(
            "format check failed: {} file(s) need formatting under '{}'\n{}",
            summary.files_changed, display, changed
        ));
    }

    if summary.files_changed == 0 {
        println!(
            "already formatted: '{}' ({} file(s) scanned)",
            display, summary.files_scanned
        );
    } else {
        println!(
            "formatted '{}' ({} file(s) changed out of {})",
            display, summary.files_changed, summary.files_scanned
        );
        for path in summary.changed_paths {
            println!("formatted: {}", path.display());
        }
    }

    Ok(())
}

fn cmd_lint(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("usage: smc lint <input.sm> [--no-cache] [--trace-cache] [--deny warnings|<CODE>] [--color auto|always|never]".to_string());
    }
    let input = args[0].as_str();
    let mut no_cache = false;
    let mut trace_cache_enabled = false;
    let mut color = ColorMode::Auto;
    let mut deny = DenyPolicy::default();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--no-cache" => no_cache = true,
            "--trace-cache" => trace_cache_enabled = true,
            "--color" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --color".to_string())?;
                color = parse_color_mode(v)?;
            }
            "--deny" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --deny".to_string())?;
                parse_deny_value(v, &mut deny);
            }
            other => return Err(format!("unknown flag '{}'", other)),
        }
        i += 1;
    }
    if !deny.has_rules() {
        deny.deny_all_warnings = true;
    }
    if !deny.deny_codes.is_empty() {
        no_cache = true;
        trace_cache(
            trace_cache_enabled,
            CacheEvent::Invalidate,
            CacheReason::DenyPolicy,
            &PathBuf::from(input),
            "SEMP",
            "",
        );
    }
    let root = PathBuf::from(input);
    let src = read_source_with_package_admission(Path::new(input))?;
    if let Ok(snapshot) = ModuleGraphSnapshot::read_from_root(&root) {
        let _ = snapshot.write_to(Path::new(CACHE_GRAPH_FILE), CACHE_SCHEMA_VERSION);
    }
    if !no_cache && deny.deny_all_warnings {
        if let Ok(fp) = module_graph_fingerprint(&root, CACHE_SCHEMA_VERSION) {
            let cache_path = cache_file_for_root(&root)?;
            match load_cache_entry_ex(&cache_path, fp) {
                Ok(CacheLookup::Hit(cached)) => {
                    trace_cache(
                        trace_cache_enabled,
                        CacheEvent::Hit,
                        CacheReason::Reused,
                        &root,
                        "SEMP",
                        &format!("{:016x}", fp),
                    );
                    let color_enabled = resolve_color_mode(color);
                    for w in &cached.warnings {
                        print_diag_colored(color_enabled, w);
                    }
                    if cached.warning_count == 0 {
                        println!("lint passed: no warnings");
                        return Ok(());
                    }
                    return Err(format!(
                        "lint failed by deny policy ({}):\n{}\nfile: {}",
                        cached.warning_count,
                        cached
                            .warnings
                            .iter()
                            .take(16)
                            .map(|w| w.lines().next().unwrap_or(""))
                            .collect::<Vec<_>>()
                            .join("\n"),
                        root.display()
                    ));
                }
                Ok(CacheLookup::Miss(reason)) => {
                    trace_cache(
                        trace_cache_enabled,
                        CacheEvent::Miss,
                        reason,
                        &root,
                        "SEMP",
                        &format!("{:016x}", fp),
                    );
                }
                Err(_) => {}
            }
        }
    } else if no_cache {
        trace_cache(
            trace_cache_enabled,
            CacheEvent::Miss,
            CacheReason::CacheDisabled,
            &root,
            "SEMP",
            "",
        );
    }

    let provider = CliFsModuleProvider;
    let parser_profile = cli_profile();
    let report = if !no_cache {
        Path::new(input)
            .canonicalize()
            .map_err(|e| format!("failed to resolve '{}': {}", input, e))
            .and_then(|p| {
                check_file_with_provider_and_profile(&p, &provider, &parser_profile)
                    .map_err(|e| e.to_string())
            })
            .or_else(|_| check_source_with_profile(&src, &parser_profile))
            .map_err(|e| e.to_string())?
    } else {
        check_source_with_profile(&src, &parser_profile).map_err(|e| e.to_string())?
    };

    let color_enabled = resolve_color_mode(color);
    for w in &report.warnings {
        print_diag_colored(color_enabled, &w.rendered);
    }
    let denied = collect_denied_warning_lines(&report, &deny);
    if denied.is_empty() {
        println!("lint passed: no warnings");
        Ok(())
    } else {
        Err(format!(
            "lint failed by deny policy ({}):\n{}\nfile: {}",
            denied.len(),
            denied.join("\n"),
            root.display()
        ))
    }
}

fn cmd_explain(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: smc explain <error-code|--list>".to_string());
    }
    if args[0] == "--list" {
        for (code, text) in diagnostic_catalog() {
            println!("{}: {}", code, text);
        }
        return Ok(());
    }
    let code = args[0].trim().to_ascii_uppercase();
    let Some((_, text)) = diagnostic_catalog().iter().find(|(c, _)| *c == code) else {
        if let Some(s) = suggest_code(&code, diagnostic_catalog()) {
            return Err(format!(
                "unknown diagnostic code '{}'\nhelp: did you mean '{}'?",
                code, s
            ));
        }
        return Err(format!("unknown diagnostic code '{}'", code));
    };
    println!("{}: {}", code, text);
    Ok(())
}

fn suggest_code(input: &str, candidates: &[(&str, &str)]) -> Option<String> {
    let mut best: Option<(&str, usize)> = None;
    for (c, _) in candidates {
        let d = edit_distance(input, c);
        if d <= 2 {
            match best {
                Some((_, bd)) if d >= bd => {}
                _ => best = Some((c, d)),
            }
        }
    }
    best.map(|(c, _)| c.to_string())
}

fn edit_distance(a: &str, b: &str) -> usize {
    let aa: Vec<char> = a.chars().collect();
    let bb: Vec<char> = b.chars().collect();
    let mut dp: Vec<usize> = (0..=bb.len()).collect();
    for (i, ca) in aa.iter().enumerate() {
        let mut prev = dp[0];
        dp[0] = i + 1;
        for (j, cb) in bb.iter().enumerate() {
            let tmp = dp[j + 1];
            let cost = if ca == cb { 0 } else { 1 };
            dp[j + 1] = (dp[j + 1] + 1).min(dp[j] + 1).min(prev + cost);
            prev = tmp;
        }
    }
    dp[bb.len()]
}

fn cmd_dump_ast(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: smc dump-ast <input.sm|project-root>".to_string());
    }
    let input = args[0].as_str();
    let input_path = Path::new(input);
    let root = if input_path.is_dir() {
        resolve_project_root_check_entry(input_path)?
    } else {
        input_path.to_path_buf()
    };
    let src = read_source_with_package_admission(&root)?;
    let parser_profile = cli_profile();
    let ast_key = ast_pack_key(&root, &src)?;
    let ast_pack = cache_ast_file_for_key(ast_key)?;
    if let Some(cached) = load_text_pack(&ast_pack, PACK_KIND_AST)? {
        println!("{}", cached);
        return Ok(());
    }
    let rendered = if let Ok(logos) = parse_logos_program_with_profile(&src, &parser_profile) {
        format!("{:#?}", logos)
    } else {
        format!(
            "{:#?}",
            parse_program_with_profile(&src, &parser_profile).map_err(|e| e.to_string())?
        )
    };
    let _ = save_text_pack(&ast_pack, PACK_KIND_AST, &rendered);
    println!("{}", rendered);
    Ok(())
}

fn cmd_dump_ir(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "usage: smc dump-ir <input.sm|project-root> [--profile auto|rust|logos] [--opt-level O0|O1|--opt]"
                .to_string(),
        );
    }
    let input = args[0].as_str();
    let mut profile = CompileProfile::Auto;
    let mut opt = OptLevel::O0;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --profile".to_string())?;
                profile = parse_compile_profile(v)?;
            }
            "--opt-level" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --opt-level".to_string())?;
                opt = parse_opt_level(v)?;
            }
            "--opt" => opt = OptLevel::O1,
            other => return Err(format!("unknown flag '{}'", other)),
        }
        i += 1;
    }
    let input_path = Path::new(input);
    let root = if input_path.is_dir() {
        resolve_project_root_check_entry(input_path)?
    } else {
        input_path.to_path_buf()
    };
    let src = read_source_with_package_admission(&root)?;
    let parser_profile = cli_profile();
    let ir_key = ir_pack_key(&root, &src, profile, opt)?;
    let ir_pack = cache_ir_file_for_key(ir_key)?;
    if let Some(cached) = load_text_pack(&ir_pack, PACK_KIND_IR)? {
        println!("{}", cached);
        return Ok(());
    }
    let rendered = match profile {
        CompileProfile::Logos => {
            let logos = parse_logos_program_with_profile(&src, &parser_profile)
                .map_err(|e| e.to_string())?;
            format!("{:#?}", lower_logos_laws_to_ir(&logos))
        }
        CompileProfile::RustLike => format!(
            "{:#?}",
            compile_program_to_ir_with_options_and_profile(
                &src,
                CompileProfile::RustLike,
                opt,
                &parser_profile,
            )
            .map_err(|e| e.to_string())?
        ),
        CompileProfile::Auto => {
            if let Ok(logos) = parse_logos_program_with_profile(&src, &parser_profile) {
                format!("{:#?}", lower_logos_laws_to_ir(&logos))
            } else {
                format!(
                    "{:#?}",
                    compile_program_to_ir_with_options_and_profile(
                        &src,
                        CompileProfile::RustLike,
                        opt,
                        &parser_profile,
                    )
                    .map_err(|e| e.to_string())?
                )
            }
        }
    };
    let _ = save_text_pack(&ir_pack, PACK_KIND_IR, &rendered);
    println!("{}", rendered);
    Ok(())
}

fn cmd_dump_bytecode(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "usage: smc dump-bytecode <input.sm|project-root> [--profile auto|rust] [--opt-level O0|O1|--opt] [--debug-symbols]"
                .to_string(),
        );
    }
    let input = args[0].as_str();
    let mut profile = CompileProfile::Auto;
    let mut opt = OptLevel::O0;
    let mut debug_symbols = false;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --profile".to_string())?;
                profile = parse_compile_profile(v)?;
            }
            "--opt-level" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --opt-level".to_string())?;
                opt = parse_opt_level(v)?;
            }
            "--opt" => opt = OptLevel::O1,
            "--debug-symbols" => debug_symbols = true,
            other => return Err(format!("unknown flag '{}'", other)),
        }
        i += 1;
    }
    let input_path = Path::new(input);
    let root = if input_path.is_dir() {
        resolve_project_root_check_entry(input_path)?
    } else {
        input_path.to_path_buf()
    };
    let src = read_source_with_package_admission(&root)?;
    let _parser_profile = cli_profile();
    let exb_key = smc_pack_key(&root, &src, profile, opt, debug_symbols)?;
    let exb_pack = cache_smc_file_for_key(exb_key)?;
    let bytes = if let Some(cached) = load_blob_pack(&exb_pack, PACK_KIND_SMC)? {
        cached
    } else {
        let built =
            compile_program_to_semcode_with_options_debug(&src, profile, opt, debug_symbols)
                .map_err(|e| e.to_string())?;
        let _ = save_blob_pack(&exb_pack, PACK_KIND_SMC, &built);
        built
    };
    for (i, chunk) in bytes.chunks(16).enumerate() {
        print!("{:04x}: ", i * 16);
        for b in chunk {
            print!("{:02x} ", b);
        }
        println!();
    }
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
fn parse_import_specs(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in source.lines() {
        let t = line.trim_start();
        if t.is_empty() || t.starts_with("//") || t.starts_with('#') {
            continue;
        }
        if !t.starts_with("Import") {
            continue;
        }
        let mut rest = t["Import".len()..].trim();
        if rest.is_empty() {
            continue;
        }
        if let Some(after_pub) = rest.strip_prefix("pub ") {
            rest = after_pub.trim_start();
        }
        let spec = if let Some(stripped) = rest.strip_prefix('"') {
            if let Some(end) = stripped.find('"') {
                stripped[..end].to_string()
            } else {
                stripped.to_string()
            }
        } else {
            rest.split_whitespace().next().unwrap_or("").to_string()
        };
        if !spec.is_empty() {
            out.push(spec);
        }
    }
    out
}

#[derive(Debug, Clone)]
struct CacheEntry {
    fingerprint: u64,
    warning_count: usize,
    law_count: usize,
    warnings: Vec<String>,
}

const CACHE_SCHEMA_VERSION: u32 = 2;
const PACK_MAGIC: [u8; 4] = *b"EXOP";
const PACK_KIND_SEM: [u8; 4] = *b"SEMP";
const PACK_KIND_AST: [u8; 4] = *b"ASTP";
const PACK_KIND_IR: [u8; 4] = *b"IRPK";
const PACK_KIND_SMC: [u8; 4] = *b"SMCP";
const CACHE_ROOT_DIR: &str = ".semantic-cache";
const CACHE_PACKS_SEM_DIR: &str = ".semantic-cache/packs/sem";
const CACHE_PACKS_AST_DIR: &str = ".semantic-cache/packs/ast";
const CACHE_PACKS_IR_DIR: &str = ".semantic-cache/packs/ir";
const CACHE_PACKS_SMC_DIR: &str = ".semantic-cache/packs/smc";
const CACHE_SCHEMA_FILE: &str = ".semantic-cache/schema.json";
const CACHE_INDEX_FILE: &str = ".semantic-cache/index.bin";
const CACHE_GRAPH_FILE: &str = ".semantic-cache/graph.bin";

#[derive(Debug, Clone, Default)]
struct DenyPolicy {
    deny_all_warnings: bool,
    deny_codes: HashSet<String>,
}

impl DenyPolicy {
    fn has_rules(&self) -> bool {
        self.deny_all_warnings || !self.deny_codes.is_empty()
    }
}

fn trace_cache(
    enabled: bool,
    event: CacheEvent,
    reason: CacheReason,
    module: &Path,
    pack_kind: &str,
    key: &str,
) {
    emit_trace(enabled, event, reason, module, pack_kind, key);
}

fn parse_deny_value(v: &str, policy: &mut DenyPolicy) {
    if v.eq_ignore_ascii_case("warnings") || v.eq_ignore_ascii_case("all") {
        policy.deny_all_warnings = true;
    } else {
        policy.deny_codes.insert(v.to_ascii_uppercase());
    }
}

fn collect_denied_warning_lines(
    report: &sm_sema::SemanticReport,
    policy: &DenyPolicy,
) -> Vec<String> {
    let mut out = Vec::new();
    for w in &report.warnings {
        if policy.deny_all_warnings || policy.deny_codes.contains(w.code) {
            out.push(format!("{}: {}", w.code, w.message));
        }
    }
    out
}

fn cache_file_for_root(root: &Path) -> Result<PathBuf, String> {
    let canonical = root
        .canonicalize()
        .map_err(|e| format!("resolve '{}': {}", root.display(), e))?;
    let key = fnv1a64(canonical.to_string_lossy().as_bytes());
    ensure_cache_layout()?;
    Ok(PathBuf::from(CACHE_PACKS_SEM_DIR).join(format!("{:016x}.smpack", key)))
}

fn cache_ast_file_for_key(key: u64) -> Result<PathBuf, String> {
    ensure_cache_layout()?;
    Ok(PathBuf::from(CACHE_PACKS_AST_DIR).join(format!("{:016x}.astpack", key)))
}

fn cache_ir_file_for_key(key: u64) -> Result<PathBuf, String> {
    ensure_cache_layout()?;
    Ok(PathBuf::from(CACHE_PACKS_IR_DIR).join(format!("{:016x}.irpack", key)))
}

fn cache_smc_file_for_key(key: u64) -> Result<PathBuf, String> {
    ensure_cache_layout()?;
    Ok(PathBuf::from(CACHE_PACKS_SMC_DIR).join(format!("{:016x}.smcpack", key)))
}

fn ast_pack_key(path: &Path, source: &str) -> Result<u64, String> {
    let canonical = path
        .canonicalize()
        .map_err(|e| format!("resolve '{}': {}", path.display(), e))?;
    let mut blob = Vec::new();
    blob.extend_from_slice(canonical.to_string_lossy().as_bytes());
    blob.push(0);
    blob.extend_from_slice(format!("{:016x}", root_source_fingerprint(source)).as_bytes());
    blob.push(0);
    blob.extend_from_slice(b"frontend-v1-auto");
    Ok(fnv1a64(&blob))
}

fn root_source_fingerprint(source: &str) -> u64 {
    fnv1a64(source.as_bytes())
}

fn downstream_pack_fingerprint(path: &Path, source: &str) -> Result<u64, String> {
    module_graph_fingerprint(path, CACHE_SCHEMA_VERSION)
        .or_else(|_| Ok(root_source_fingerprint(source)))
}

fn ir_pack_key(
    path: &Path,
    source: &str,
    profile: CompileProfile,
    opt: OptLevel,
) -> Result<u64, String> {
    let canonical = path
        .canonicalize()
        .map_err(|e| format!("resolve '{}': {}", path.display(), e))?;
    let mut blob = Vec::new();
    blob.extend_from_slice(canonical.to_string_lossy().as_bytes());
    blob.push(0);
    blob.extend_from_slice(
        format!("{:016x}", downstream_pack_fingerprint(path, source)?).as_bytes(),
    );
    blob.push(0);
    blob.extend_from_slice(format!("profile={:?};opt={:?};lowering=v1", profile, opt).as_bytes());
    Ok(fnv1a64(&blob))
}

fn smc_pack_key(
    path: &Path,
    source: &str,
    profile: CompileProfile,
    opt: OptLevel,
    debug_symbols: bool,
) -> Result<u64, String> {
    let canonical = path
        .canonicalize()
        .map_err(|e| format!("resolve '{}': {}", path.display(), e))?;
    let mut blob = Vec::new();
    blob.extend_from_slice(canonical.to_string_lossy().as_bytes());
    blob.push(0);
    blob.extend_from_slice(
        format!("{:016x}", downstream_pack_fingerprint(path, source)?).as_bytes(),
    );
    blob.push(0);
    blob.extend_from_slice(
        format!(
            "profile={:?};opt={:?};debug={};emit=v1",
            profile, opt, debug_symbols
        )
        .as_bytes(),
    );
    Ok(fnv1a64(&blob))
}

fn ensure_cache_layout() -> Result<(), String> {
    std::fs::create_dir_all(CACHE_PACKS_SEM_DIR).map_err(|e| format!("create cache dir: {}", e))?;
    std::fs::create_dir_all(CACHE_PACKS_AST_DIR).map_err(|e| format!("create cache dir: {}", e))?;
    std::fs::create_dir_all(CACHE_PACKS_IR_DIR).map_err(|e| format!("create cache dir: {}", e))?;
    std::fs::create_dir_all(CACHE_PACKS_SMC_DIR).map_err(|e| format!("create cache dir: {}", e))?;
    if !Path::new(CACHE_SCHEMA_FILE).exists() {
        let schema = format!(
            "{{\"schema_version\":{},\"pack_magic\":\"EXOP\",\"layout\":\"v0.1\"}}\n",
            CACHE_SCHEMA_VERSION
        );
        std::fs::write(CACHE_SCHEMA_FILE, schema)
            .map_err(|e| format!("write cache schema '{}': {}", CACHE_SCHEMA_FILE, e))?;
    }
    if !Path::new(CACHE_INDEX_FILE).exists() {
        std::fs::write(CACHE_INDEX_FILE, b"EXOIDX\n")
            .map_err(|e| format!("write cache index '{}': {}", CACHE_INDEX_FILE, e))?;
    }
    if !Path::new(CACHE_GRAPH_FILE).exists() {
        std::fs::write(CACHE_GRAPH_FILE, b"EXOGRAPH 2 0\n")
            .map_err(|e| format!("write cache graph '{}': {}", CACHE_GRAPH_FILE, e))?;
    }
    let _ = CACHE_ROOT_DIR;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct PackHeader {
    kind: [u8; 4],
    schema_version: u32,
    toolchain_hash: u64,
    feature_hash: u64,
    payload_len: u64,
    payload_checksum: u64,
}

/// Builds the tag string that identifies "this compiler build" for cache
/// purposes: the crate's package version plus a discriminator for the
/// actual compiler/build semantics. Pulled out as a pure function so the
/// dependency on `compiler_generation` is independently testable (DL-024:
/// a semantic cache entry produced by one compiler generation must not be
/// accepted as valid by a differently-behaving generation, even when
/// `pkg_version` is unchanged).
fn toolchain_identity_tag(pkg_version: &str, compiler_generation: &str) -> String {
    format!("smc-cli:{}:{}", pkg_version, compiler_generation)
}

// A malformed `SM_*` override below (set but not parseable) is deliberately
// left to silently fall through to the computed default, not treated as a
// build/cache-usage error (DL-025). This is safe, not merely convenient: the
// default in every one of these functions is itself a real, complete,
// fail-closed identity computed from this exact binary's own build-time
// constants (`SM_COMPILER_SOURCE_HASH`/`SM_ENABLED_FEATURES`, both emitted by
// `build.rs`, which fails the *build* rather than emitting an incomplete
// value - see its module doc). Two genuinely different compiler generations
// therefore still compute two different default hashes even if both somehow
// inherited the same malformed override string, so a malformed override can
// never cause a false cache `HIT` the way an incomplete *construction*-time
// fallback could. This is a different situation from the fail-closed
// requirements on `build.rs`'s own identity construction: there, a fallback
// value is silently wrong (e.g. missing real source content); here, the
// fallback value is the same correct value these functions would compute
// with no override at all.
fn current_toolchain_hash() -> u64 {
    if let Ok(v) = std::env::var("SM_TOOLCHAIN_HASH") {
        if let Ok(parsed) = u64::from_str_radix(v.trim(), 16).or_else(|_| v.trim().parse::<u64>()) {
            return parsed;
        }
    }
    // SM_COMPILER_SOURCE_HASH is set by build.rs from the content of every
    // crate that determines check/compile/verify/run semantics (parse,
    // typecheck, lowering, verification, execution). A rebuild from edited
    // source in any of them - committed or not - changes this value, so an
    // old cache entry from a semantically different build cannot alias with
    // a new one just because CARGO_PKG_VERSION didn't change.
    let tag = toolchain_identity_tag(env!("CARGO_PKG_VERSION"), env!("SM_COMPILER_SOURCE_HASH"));
    fnv1a64(tag.as_bytes())
}

fn current_feature_hash() -> u64 {
    if let Ok(v) = std::env::var("SM_FEATURE_HASH") {
        if let Ok(parsed) = u64::from_str_radix(v.trim(), 16).or_else(|_| v.trim().parse::<u64>()) {
            return parsed;
        }
    }
    // SM_ENABLED_FEATURES (set by build.rs from this build's own
    // CARGO_FEATURE_* env vars) folds Cargo feature flags such as
    // profile-rust/profile-logos/debug-symbols into this identity: they
    // change lowering/verification behavior independently of source
    // content, so a --no-default-features build must not accept an
    // artifact/cache entry produced by a default-features build.
    let flags = format!(
        "debug_assertions={};target_pointer_width={};features={}",
        cfg!(debug_assertions),
        std::mem::size_of::<usize>() * 8,
        env!("SM_ENABLED_FEATURES")
    );
    fnv1a64(flags.as_bytes())
}

fn current_cache_schema_version() -> u32 {
    if let Ok(v) = std::env::var("SM_CACHE_SCHEMA") {
        if let Ok(parsed) = v.trim().parse::<u32>() {
            return parsed;
        }
    }
    CACHE_SCHEMA_VERSION
}

fn current_caps_hash() -> u64 {
    if let Ok(v) = std::env::var("SM_CAPS_HASH") {
        if let Ok(parsed) = u64::from_str_radix(v.trim(), 16).or_else(|_| v.trim().parse::<u64>()) {
            return parsed;
        }
    }
    0
}

fn expected_feature_hash_for_kind(kind: [u8; 4]) -> u64 {
    let base = current_feature_hash();
    if kind == PACK_KIND_SMC {
        let mut blob = Vec::new();
        blob.extend_from_slice(&base.to_le_bytes());
        blob.extend_from_slice(&current_caps_hash().to_le_bytes());
        fnv1a64(&blob)
    } else {
        base
    }
}

fn encode_pack_header(header: &PackHeader) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 4 + 4 + 8 + 8 + 8 + 8);
    out.extend_from_slice(&PACK_MAGIC);
    out.extend_from_slice(&header.kind);
    out.extend_from_slice(&header.schema_version.to_le_bytes());
    out.extend_from_slice(&header.toolchain_hash.to_le_bytes());
    out.extend_from_slice(&header.feature_hash.to_le_bytes());
    out.extend_from_slice(&header.payload_len.to_le_bytes());
    out.extend_from_slice(&header.payload_checksum.to_le_bytes());
    out
}

fn decode_pack_header(bytes: &[u8]) -> Option<(PackHeader, usize)> {
    let header_len = 44usize;
    if bytes.len() < header_len {
        return None;
    }
    if bytes[0..4] != PACK_MAGIC {
        return None;
    }
    let kind = [bytes[4], bytes[5], bytes[6], bytes[7]];
    let schema_version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let toolchain_hash = u64::from_le_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
    ]);
    let feature_hash = u64::from_le_bytes([
        bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26], bytes[27],
    ]);
    let payload_len = u64::from_le_bytes([
        bytes[28], bytes[29], bytes[30], bytes[31], bytes[32], bytes[33], bytes[34], bytes[35],
    ]);
    let payload_checksum = u64::from_le_bytes([
        bytes[36], bytes[37], bytes[38], bytes[39], bytes[40], bytes[41], bytes[42], bytes[43],
    ]);
    Some((
        PackHeader {
            kind,
            schema_version,
            toolchain_hash,
            feature_hash,
            payload_len,
            payload_checksum,
        },
        header_len,
    ))
}

fn save_text_pack(path: &Path, kind: [u8; 4], payload: &str) -> Result<(), String> {
    let payload_bytes = payload.as_bytes();
    let header = PackHeader {
        kind,
        schema_version: current_cache_schema_version(),
        toolchain_hash: current_toolchain_hash(),
        feature_hash: expected_feature_hash_for_kind(kind),
        payload_len: payload_bytes.len() as u64,
        payload_checksum: fnv1a64(payload_bytes),
    };
    let mut out = encode_pack_header(&header);
    out.extend_from_slice(payload_bytes);
    std::fs::write(path, out).map_err(|e| format!("write pack '{}': {}", path.display(), e))
}

fn load_text_pack(path: &Path, expected_kind: [u8; 4]) -> Result<Option<String>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes =
        std::fs::read(path).map_err(|e| format!("read pack '{}': {}", path.display(), e))?;
    let (header, header_len) = match decode_pack_header(&bytes) {
        Some(v) => v,
        None => return Ok(None),
    };
    if header.kind != expected_kind {
        return Ok(None);
    }
    if header.schema_version != current_cache_schema_version() {
        return Ok(None);
    }
    if header.toolchain_hash != current_toolchain_hash() {
        return Ok(None);
    }
    if header.feature_hash != expected_feature_hash_for_kind(expected_kind) {
        return Ok(None);
    }
    if header.payload_len != (bytes.len().saturating_sub(header_len)) as u64 {
        return Ok(None);
    }
    let payload = &bytes[header_len..];
    if header.payload_checksum != fnv1a64(payload) {
        return Ok(None);
    }
    String::from_utf8(payload.to_vec())
        .map(Some)
        .map_err(|_| format!("read pack '{}': payload is not valid utf-8", path.display()))
}

fn save_blob_pack(path: &Path, kind: [u8; 4], payload: &[u8]) -> Result<(), String> {
    let header = PackHeader {
        kind,
        schema_version: current_cache_schema_version(),
        toolchain_hash: current_toolchain_hash(),
        feature_hash: expected_feature_hash_for_kind(kind),
        payload_len: payload.len() as u64,
        payload_checksum: fnv1a64(payload),
    };
    let mut out = encode_pack_header(&header);
    out.extend_from_slice(payload);
    std::fs::write(path, out).map_err(|e| format!("write pack '{}': {}", path.display(), e))
}

fn load_blob_pack(path: &Path, expected_kind: [u8; 4]) -> Result<Option<Vec<u8>>, String> {
    match load_blob_pack_ex(path, expected_kind)? {
        BlobPackLookup::Hit(bytes) => Ok(Some(bytes)),
        BlobPackLookup::Miss(_) => Ok(None),
    }
}

enum BlobPackLookup {
    Hit(Vec<u8>),
    Miss(CacheReason),
}

fn load_blob_pack_ex(path: &Path, expected_kind: [u8; 4]) -> Result<BlobPackLookup, String> {
    if !path.exists() {
        return Ok(BlobPackLookup::Miss(CacheReason::NotFound));
    }
    let bytes =
        std::fs::read(path).map_err(|e| format!("read pack '{}': {}", path.display(), e))?;
    let (header, header_len) = match decode_pack_header(&bytes) {
        Some(v) => v,
        None => return Ok(BlobPackLookup::Miss(CacheReason::HeaderInvalid)),
    };
    if header.kind != expected_kind {
        return Ok(BlobPackLookup::Miss(CacheReason::KindMismatch));
    }
    if header.schema_version != current_cache_schema_version() {
        return Ok(BlobPackLookup::Miss(CacheReason::VersionMismatch));
    }
    if header.toolchain_hash != current_toolchain_hash() {
        return Ok(BlobPackLookup::Miss(CacheReason::ToolchainMismatch));
    }
    if header.feature_hash != expected_feature_hash_for_kind(expected_kind) {
        if expected_kind == PACK_KIND_SMC && current_caps_hash() != 0 {
            return Ok(BlobPackLookup::Miss(CacheReason::CapsMismatch));
        }
        return Ok(BlobPackLookup::Miss(CacheReason::FeatureMismatch));
    }
    if header.payload_len != (bytes.len().saturating_sub(header_len)) as u64 {
        return Ok(BlobPackLookup::Miss(CacheReason::PayloadSizeMismatch));
    }
    let payload = &bytes[header_len..];
    if header.payload_checksum != fnv1a64(payload) {
        return Ok(BlobPackLookup::Miss(CacheReason::ChecksumMismatch));
    }
    Ok(BlobPackLookup::Hit(payload.to_vec()))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err("odd hex length".to_string());
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let hi = (bytes[i] as char)
            .to_digit(16)
            .ok_or_else(|| "invalid hex".to_string())?;
        let lo = (bytes[i + 1] as char)
            .to_digit(16)
            .ok_or_else(|| "invalid hex".to_string())?;
        out.push(((hi << 4) | lo) as u8);
        i += 2;
    }
    Ok(out)
}

fn save_cache_entry(path: &Path, entry: &CacheEntry) -> Result<(), String> {
    let mut payload = String::new();
    payload.push_str(&format!("FP {:016x}\n", entry.fingerprint));
    payload.push_str(&format!("WARN {}\n", entry.warning_count));
    payload.push_str(&format!("LAW {}\n", entry.law_count));
    let mut checksum_blob = Vec::new();
    for w in &entry.warnings {
        payload.push_str("W ");
        payload.push_str(&hex_encode(w.as_bytes()));
        payload.push('\n');
        checksum_blob.extend_from_slice(w.as_bytes());
        checksum_blob.push(0);
    }
    payload.push_str(&format!("WSUM {:016x}\n", fnv1a64(&checksum_blob)));
    save_text_pack(path, PACK_KIND_SEM, &payload)
}

enum CacheLookup {
    Hit(CacheEntry),
    Miss(CacheReason),
}

fn load_cache_entry_ex(path: &Path, expected_fp: u64) -> Result<CacheLookup, String> {
    if !path.exists() {
        return Ok(CacheLookup::Miss(CacheReason::NotFound));
    }
    let bytes =
        std::fs::read(path).map_err(|e| format!("read pack '{}': {}", path.display(), e))?;
    let (header, header_len) = match decode_pack_header(&bytes) {
        Some(v) => v,
        None => return Ok(CacheLookup::Miss(CacheReason::HeaderInvalid)),
    };
    if header.kind != PACK_KIND_SEM {
        return Ok(CacheLookup::Miss(CacheReason::KindMismatch));
    }
    if header.schema_version != current_cache_schema_version() {
        return Ok(CacheLookup::Miss(CacheReason::VersionMismatch));
    }
    if header.toolchain_hash != current_toolchain_hash() {
        return Ok(CacheLookup::Miss(CacheReason::ToolchainMismatch));
    }
    if header.feature_hash != current_feature_hash() {
        return Ok(CacheLookup::Miss(CacheReason::FeatureMismatch));
    }
    if header.payload_len != (bytes.len().saturating_sub(header_len)) as u64 {
        return Ok(CacheLookup::Miss(CacheReason::PayloadSizeMismatch));
    }
    let payload = &bytes[header_len..];
    if header.payload_checksum != fnv1a64(payload) {
        return Ok(CacheLookup::Miss(CacheReason::ChecksumMismatch));
    }
    let text = String::from_utf8(payload.to_vec())
        .map_err(|_| format!("read pack '{}': payload is not valid utf-8", path.display()))?;

    let mut fp = None;
    let mut warn = 0usize;
    let mut law = 0usize;
    let mut wsum = None;
    let mut warnings = Vec::new();
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("FP ") {
            fp = u64::from_str_radix(v.trim(), 16).ok();
            continue;
        }
        if let Some(v) = line.strip_prefix("WARN ") {
            warn = v.trim().parse::<usize>().unwrap_or(0);
            continue;
        }
        if let Some(v) = line.strip_prefix("LAW ") {
            law = v.trim().parse::<usize>().unwrap_or(0);
            continue;
        }
        if let Some(v) = line.strip_prefix("WSUM ") {
            wsum = u64::from_str_radix(v.trim(), 16).ok();
            continue;
        }
        if let Some(v) = line.strip_prefix("W ") {
            let raw = hex_decode(v.trim())?;
            warnings.push(String::from_utf8_lossy(&raw).to_string());
        }
    }
    if fp != Some(expected_fp) {
        return Ok(CacheLookup::Miss(CacheReason::FingerprintMismatch));
    }
    if warn != warnings.len() {
        return Ok(CacheLookup::Miss(CacheReason::ChecksumMismatch));
    }
    let mut checksum_blob = Vec::new();
    for w in &warnings {
        checksum_blob.extend_from_slice(w.as_bytes());
        checksum_blob.push(0);
    }
    if wsum != Some(fnv1a64(&checksum_blob)) {
        return Ok(CacheLookup::Miss(CacheReason::ChecksumMismatch));
    }
    Ok(CacheLookup::Hit(CacheEntry {
        fingerprint: expected_fp,
        warning_count: warn,
        law_count: law,
        warnings,
    }))
}

#[cfg(test)]
fn load_cache_entry(path: &Path, expected_fp: u64) -> Result<Option<CacheEntry>, String> {
    match load_cache_entry_ex(path, expected_fp)? {
        CacheLookup::Hit(v) => Ok(Some(v)),
        CacheLookup::Miss(_) => Ok(None),
    }
}

fn cmd_hash_ast(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: smc hash-ast <input.sm|project-root>".to_string());
    }
    let input = args[0].as_str();
    let input_path = Path::new(input);
    let root = if input_path.is_dir() {
        resolve_project_root_check_entry(input_path)?
    } else {
        input_path.to_path_buf()
    };
    let src = read_source_with_package_admission(&root)?;
    let parser_profile = cli_profile();
    let ast_key = ast_pack_key(&root, &src)?;
    let ast_pack = cache_ast_file_for_key(ast_key)?;
    let text = if let Some(cached) = load_text_pack(&ast_pack, PACK_KIND_AST)? {
        cached
    } else {
        let rendered = if let Ok(logos) = parse_logos_program_with_profile(&src, &parser_profile) {
            format!("{:#?}", logos)
        } else {
            format!(
                "{:#?}",
                parse_program_with_profile(&src, &parser_profile).map_err(|e| e.to_string())?
            )
        };
        let _ = save_text_pack(&ast_pack, PACK_KIND_AST, &rendered);
        rendered
    };
    println!("{:016x}", fnv1a64(text.as_bytes()));
    Ok(())
}

fn cmd_hash_ir(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "usage: smc hash-ir <input.sm|project-root> [--profile auto|rust|logos] [--opt-level O0|O1|--opt]"
                .to_string(),
        );
    }
    let input = args[0].as_str();
    let mut profile = CompileProfile::Auto;
    let mut opt = OptLevel::O0;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --profile".to_string())?;
                profile = parse_compile_profile(v)?;
            }
            "--opt-level" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --opt-level".to_string())?;
                opt = parse_opt_level(v)?;
            }
            "--opt" => opt = OptLevel::O1,
            other => return Err(format!("unknown flag '{}'", other)),
        }
        i += 1;
    }
    let input_path = Path::new(input);
    let root = if input_path.is_dir() {
        resolve_project_root_check_entry(input_path)?
    } else {
        input_path.to_path_buf()
    };
    let src = read_source_with_package_admission(&root)?;
    let parser_profile = cli_profile();
    let ir_key = ir_pack_key(&root, &src, profile, opt)?;
    let ir_pack = cache_ir_file_for_key(ir_key)?;
    let text = if let Some(cached) = load_text_pack(&ir_pack, PACK_KIND_IR)? {
        cached
    } else {
        let rendered = match profile {
            CompileProfile::Logos => {
                let logos = parse_logos_program_with_profile(&src, &parser_profile)
                    .map_err(|e| e.to_string())?;
                format!("{:#?}", lower_logos_laws_to_ir(&logos))
            }
            CompileProfile::RustLike => format!(
                "{:#?}",
                compile_program_to_ir_with_options_and_profile(
                    &src,
                    CompileProfile::RustLike,
                    opt,
                    &parser_profile,
                )
                .map_err(|e| e.to_string())?
            ),
            CompileProfile::Auto => {
                if let Ok(logos) = parse_logos_program_with_profile(&src, &parser_profile) {
                    format!("{:#?}", lower_logos_laws_to_ir(&logos))
                } else {
                    format!(
                        "{:#?}",
                        compile_program_to_ir_with_options_and_profile(
                            &src,
                            CompileProfile::RustLike,
                            opt,
                            &parser_profile,
                        )
                        .map_err(|e| e.to_string())?
                    )
                }
            }
        };
        let _ = save_text_pack(&ir_pack, PACK_KIND_IR, &rendered);
        rendered
    };
    println!("{:016x}", fnv1a64(text.as_bytes()));
    Ok(())
}

fn cmd_hash_smc(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "usage: smc hash-smc <input.sm|project-root> [--profile auto|rust] [--opt-level O0|O1|--opt] [--trace-cache]"
                .to_string(),
        );
    }
    let input = args[0].as_str();
    let mut profile = CompileProfile::Auto;
    let mut opt = OptLevel::O0;
    let mut debug_symbols = false;
    let mut trace_cache_enabled = false;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --profile".to_string())?;
                profile = parse_compile_profile(v)?;
            }
            "--opt-level" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "missing value for --opt-level".to_string())?;
                opt = parse_opt_level(v)?;
            }
            "--opt" => opt = OptLevel::O1,
            "--debug-symbols" => debug_symbols = true,
            "--trace-cache" => trace_cache_enabled = true,
            other => return Err(format!("unknown flag '{}'", other)),
        }
        i += 1;
    }
    let input_path = Path::new(input);
    let root = if input_path.is_dir() {
        resolve_project_root_check_entry(input_path)?
    } else {
        input_path.to_path_buf()
    };
    let src = read_source_with_package_admission(&root)?;
    let prev_graph_hash = read_graph_hash(Path::new(CACHE_GRAPH_FILE));
    let graph_hash_now = if let Ok(snapshot) = ModuleGraphSnapshot::read_from_root(&root) {
        let hash = snapshot.hash(CACHE_SCHEMA_VERSION);
        let _ = snapshot.write_to(Path::new(CACHE_GRAPH_FILE), CACHE_SCHEMA_VERSION);
        Some(hash)
    } else {
        None
    };
    let exb_key = smc_pack_key(&root, &src, profile, opt, debug_symbols)?;
    if trace_cache_enabled && prev_graph_hash.is_some() && prev_graph_hash != graph_hash_now {
        trace_cache(
            true,
            CacheEvent::Invalidate,
            CacheReason::GraphChanged,
            &root,
            "GRAPH",
            &format!("{:016x}", exb_key),
        );
    }
    let exb_pack = cache_smc_file_for_key(exb_key)?;
    let bytes = match load_blob_pack_ex(&exb_pack, PACK_KIND_SMC)? {
        BlobPackLookup::Hit(cached) => {
            trace_cache(
                trace_cache_enabled,
                CacheEvent::Hit,
                CacheReason::Reused,
                &root,
                "SMCP",
                &format!("{:016x}", exb_key),
            );
            cached
        }
        BlobPackLookup::Miss(reason) => {
            trace_cache(
                trace_cache_enabled,
                CacheEvent::Miss,
                reason,
                &root,
                "SMCP",
                &format!("{:016x}", exb_key),
            );
            let built =
                compile_program_to_semcode_with_options_debug(&src, profile, opt, debug_symbols)
                    .map_err(|e| e.to_string())?;
            let _ = save_blob_pack(&exb_pack, PACK_KIND_SMC, &built);
            built
        }
    };
    println!("{:016x}", fnv1a64(&bytes));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn mk_temp_dir(prefix: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "{}_{}_{}",
            prefix,
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).expect("mkdir");
        base
    }

    #[test]
    fn import_specs_parse_pub_and_alias() {
        let src = r#"
Import "a.sm"
Import pub "b.sm"
Import "c.sm" as Core
"#;
        let specs = parse_import_specs(src);
        assert_eq!(specs, vec!["a.sm", "b.sm", "c.sm"]);
    }

    // CliFsModuleProvider::resolve_import folded '\' into '/' unconditionally, even though
    // resolve_package_import_path (which it wraps) already preserves a literal backslash
    // on Unix. On Unix a resolved import whose path contains a literal '\' must come back
    // unchanged from this provider method too, or check/compile/run would try to read a
    // different or nonexistent module than the one actually resolved (DL-018).
    #[cfg(unix)]
    #[test]
    fn cli_fs_module_provider_resolve_import_preserves_literal_backslash_on_unix() {
        let dir = mk_temp_dir("cli_fs_module_provider_backslash_import");
        let importer = dir.join("main.sm");
        std::fs::write(&importer, "fn main() { return; }\n").expect("write importer");
        std::fs::write(dir.join("dep\\lib.sm"), "fn dep() { return; }\n")
            .expect("write literal-backslash dependency");

        let provider = CliFsModuleProvider;
        let resolved = provider
            .resolve_import(importer.to_str().expect("importer utf8"), "dep\\lib.sm")
            .expect("resolve");
        assert!(
            resolved.ends_with("dep\\lib.sm"),
            "resolved module_id must preserve the literal backslash, got: {resolved}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn executable_bundle_includes_direct_local_helper_modules() {
        let dir = mk_temp_dir("smc_exec_bundle_local_helper");
        let root = dir.join("main.sm");
        let helper = dir.join("helper.sm");
        std::fs::write(
            &root,
            r#"
Import "helper.sm"

fn main() {
    let value: i32 = score(1);
    assert(value == 1);
    return;
}
"#,
        )
        .expect("write root");
        std::fs::write(
            &helper,
            r#"
fn score(value: i32) -> i32 {
    return value;
}
"#,
        )
        .expect("write helper");

        let bundled = read_source_with_package_admission(&root).expect("bundle");
        assert!(bundled.contains("Import \"helper.sm\""));
        assert!(bundled.contains("fn score(value: i32) -> i32"));
        assert!(bundled.contains("fn main()"));
        assert!(bundled.find("fn score").unwrap() < bundled.find("fn main").unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn executable_bundle_admits_selected_imports_in_wave2() {
        let dir = mk_temp_dir("smc_exec_bundle_selected_allowed");
        let root = dir.join("main.sm");
        let helper = dir.join("helper.sm");
        std::fs::write(
            &root,
            r#"
Import "helper.sm" { score }

fn main() {
    return;
}
"#,
        )
        .expect("write root");
        std::fs::write(
            &helper,
            r#"
fn score(value: i32) -> i32 {
    return value;
}
"#,
        )
        .expect("write helper");

        let bundled = read_source_with_package_admission(&root).expect("bundle selected import");
        assert!(bundled.contains("Import \"helper.sm\" { score }"));
        assert!(bundled.contains("fn execsel_"));
        assert!(bundled.contains("fn score(value: i32) -> i32"));
        assert!(bundled.contains("return execsel_"));
        assert!(bundled.find("fn score").unwrap() < bundled.find("fn main").unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_roundtrip_and_integrity_check() {
        let dir = mk_temp_dir("smc_cache_roundtrip");
        let path = dir.join("entry.cache");
        let entry = CacheEntry {
            fingerprint: 0x1234_5678_90ab_cdef,
            warning_count: 2,
            law_count: 5,
            warnings: vec!["w1".into(), "w2".into()],
        };
        save_cache_entry(&path, &entry).expect("save");
        let loaded = load_cache_entry(&path, entry.fingerprint)
            .expect("load")
            .expect("some");
        assert_eq!(loaded.warning_count, 2);
        assert_eq!(loaded.law_count, 5);
        assert_eq!(loaded.warnings, vec!["w1", "w2"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_version_mismatch_is_ignored() {
        let dir = mk_temp_dir("smc_cache_version");
        let path = dir.join("entry.cache");
        let payload = b"FP 0000000000000001\nWARN 0\nLAW 0\nWSUM 14650fb0739d0383\n".to_vec();
        let header = PackHeader {
            kind: PACK_KIND_SEM,
            schema_version: CACHE_SCHEMA_VERSION - 1,
            toolchain_hash: current_toolchain_hash(),
            feature_hash: current_feature_hash(),
            payload_len: payload.len() as u64,
            payload_checksum: fnv1a64(&payload),
        };
        let mut bytes = encode_pack_header(&header);
        bytes.extend_from_slice(&payload);
        std::fs::write(&path, bytes).expect("write");
        let got = load_cache_entry(&path, 1).expect("load");
        assert!(got.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_checksum_mismatch_is_ignored() {
        let dir = mk_temp_dir("smc_cache_checksum");
        let path = dir.join("entry.cache");
        let payload = b"FP 0000000000000001\nWARN 0\nLAW 0\nWSUM 14650fb0739d0383\n".to_vec();
        let header = PackHeader {
            kind: PACK_KIND_SEM,
            schema_version: CACHE_SCHEMA_VERSION,
            toolchain_hash: current_toolchain_hash(),
            feature_hash: current_feature_hash(),
            payload_len: payload.len() as u64,
            payload_checksum: 0,
        };
        let mut bytes = encode_pack_header(&header);
        bytes.extend_from_slice(&payload);
        std::fs::write(&path, bytes).expect("write");
        let got = load_cache_entry(&path, 1).expect("load");
        assert!(got.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- Compiler-generation cache invalidation (DL-024, DL-025) ---
    //
    // Root cause: current_toolchain_hash()'s default value used to depend
    // only on CARGO_PKG_VERSION, which does not change when compiler
    // semantics change without a version bump (the normal case between
    // commits). Two builds that behave differently but share a package
    // version therefore produced the identical toolchain_hash, letting an
    // old build's cached "PASS" survive a rebuild that would now reject the
    // same source. See build.rs and toolchain_identity_tag's doc comment.

    #[test]
    fn build_script_identity_env_vars_are_wired_into_the_runtime_formulas() {
        // build.rs's own logic (dependency-closure discovery, path-relative
        // content hashing, enabled-feature collection) is not directly unit
        // -testable here - it runs in a separate process before this crate
        // compiles, not as one of its modules - and was instead verified
        // empirically by hand (closure discovery found every crate a review
        // pass flagged as missing; two checkouts of identical content at
        // different absolute paths produced the identical hash; a
        // --no-default-features build produced a different feature string
        // than the default build). This test locks in the one thing that
        // *is* checkable from here: that current_toolchain_hash() and
        // current_feature_hash() actually consume the env vars build.rs
        // sets, not just that the env vars exist.
        assert_eq!(
            env!("SM_COMPILER_SOURCE_HASH").len(),
            16,
            "build.rs must emit a 16-hex-digit content hash"
        );
        assert!(
            env!("SM_ENABLED_FEATURES").contains("STD"),
            "a normal `cargo test` build always has the std feature enabled"
        );

        let tag_without_generation = toolchain_identity_tag(env!("CARGO_PKG_VERSION"), "");
        let tag_with_real_generation =
            toolchain_identity_tag(env!("CARGO_PKG_VERSION"), env!("SM_COMPILER_SOURCE_HASH"));
        assert_ne!(
            fnv1a64(tag_without_generation.as_bytes()),
            fnv1a64(tag_with_real_generation.as_bytes()),
            "current_toolchain_hash()'s formula must actually depend on \
             SM_COMPILER_SOURCE_HASH, not silently ignore it"
        );
    }

    #[test]
    fn semantic_cache_toolchain_hash_differs_across_compiler_generations() {
        // Direct test of the fix's core claim: the tag function that feeds
        // current_toolchain_hash() must produce different output for two
        // "generations" sharing the same package version. Before this fix,
        // no parameter representing compiler generation existed at all -
        // the formula was `format!("smc-cli:{}", pkg_version)` with nothing
        // else, so it could not have varied no matter what the generation
        // was.
        let generation_a =
            fnv1a64(toolchain_identity_tag("0.1.0", "sourceHashAAAAAAAA").as_bytes());
        let generation_b =
            fnv1a64(toolchain_identity_tag("0.1.0", "sourceHashBBBBBBBB").as_bytes());
        assert_ne!(
            generation_a, generation_b,
            "same package version, different compiler generation, must not collide"
        );

        let generation_a_again =
            fnv1a64(toolchain_identity_tag("0.1.0", "sourceHashAAAAAAAA").as_bytes());
        assert_eq!(
            generation_a, generation_a_again,
            "the same generation must hash identically every time (no nondeterminism)"
        );
    }

    #[test]
    fn semantic_cache_hits_for_same_compiler_generation() {
        // 8.1: source fingerprint X, compiler generation A (this test
        // binary's own, real current_toolchain_hash()) saved then loaded
        // under the identical generation -> HIT.
        let dir = mk_temp_dir("smc_cache_same_generation");
        let path = dir.join("entry.cache");
        let entry = CacheEntry {
            fingerprint: 0xABCD_1234,
            warning_count: 0,
            law_count: 0,
            warnings: Vec::new(),
        };
        save_cache_entry(&path, &entry).expect("save");
        let got = load_cache_entry(&path, entry.fingerprint).expect("load");
        assert!(
            got.is_some(),
            "same compiler generation + same source fingerprint must HIT"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn semantic_cache_misses_when_compiler_generation_changes() {
        // 8.2 (the key regression): source fingerprint X saved under a
        // compiler generation that is NOT this test binary's own
        // current_toolchain_hash() (constructed directly, the same
        // technique cache_version_mismatch_is_ignored and
        // cache_checksum_mismatch_is_ignored already use to inject a
        // specific header field) must MISS(ToolchainMismatch) when loaded
        // under the real current generation, even though the source
        // fingerprint and schema/feature hashes all still match.
        let dir = mk_temp_dir("smc_cache_cross_generation");
        let path = dir.join("entry.cache");
        let payload = b"FP 000000000000abcd\nWARN 0\nLAW 0\nWSUM 14650fb0739d0383\n".to_vec();
        let other_generation_hash = current_toolchain_hash() ^ 0xDEAD_BEEF_DEAD_BEEF;
        assert_ne!(
            other_generation_hash,
            current_toolchain_hash(),
            "sanity: the injected generation must actually differ from the real one"
        );
        let header = PackHeader {
            kind: PACK_KIND_SEM,
            schema_version: CACHE_SCHEMA_VERSION,
            toolchain_hash: other_generation_hash,
            feature_hash: current_feature_hash(),
            payload_len: payload.len() as u64,
            payload_checksum: fnv1a64(&payload),
        };
        let mut bytes = encode_pack_header(&header);
        bytes.extend_from_slice(&payload);
        std::fs::write(&path, bytes).expect("write");

        match load_cache_entry_ex(&path, 0xabcd) {
            Ok(CacheLookup::Miss(CacheReason::ToolchainMismatch)) => {}
            Ok(CacheLookup::Hit(_)) => panic!(
                "a cache entry from a different compiler generation must not be accepted as a HIT"
            ),
            other => panic!("expected Miss(ToolchainMismatch), got {:?}", other.is_ok()),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn semantic_cache_still_misses_when_source_fingerprint_changes() {
        // 8.3: existing fingerprint invalidation must keep working
        // regardless of this fix - same (real, current) compiler
        // generation, different source fingerprint -> MISS.
        let dir = mk_temp_dir("smc_cache_fingerprint_change");
        let path = dir.join("entry.cache");
        let entry = CacheEntry {
            fingerprint: 0x1111_1111,
            warning_count: 0,
            law_count: 0,
            warnings: Vec::new(),
        };
        save_cache_entry(&path, &entry).expect("save");
        match load_cache_entry_ex(&path, 0x2222_2222) {
            Ok(CacheLookup::Miss(CacheReason::FingerprintMismatch)) => {}
            other => panic!(
                "expected Miss(FingerprintMismatch), got {:?}",
                other.is_ok()
            ),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn semantic_check_command_discards_cache_entry_from_another_compiler_generation() {
        // 8.5: exercises the real `smc check` path (cmd_check), not just
        // the cache helper functions. Pre-seeds a SEM-kind pack at the
        // exact path cmd_check itself will look up (cache_file_for_root),
        // stamped with a different compiler generation, then runs a real
        // check on the same source. If the stale entry were wrongly
        // accepted, cmd_check returns before ever touching the cache file
        // again, so it would still carry the injected other_generation_hash
        // afterward. A correct MISS makes cmd_check perform a fresh check
        // and unconditionally overwrite the file with a fresh entry stamped
        // with the real current_toolchain_hash() - that overwrite is what
        // this test observes, since cmd_check does not expose HIT/MISS as a
        // return value.
        let dir = mk_temp_dir("smc_check_cross_generation");
        let source_path = dir.join("main.sm");
        std::fs::write(&source_path, "fn main() {\n    return;\n}\n").expect("write source");

        let canonical_root = source_path.canonicalize().expect("canonicalize");
        let fp = module_graph_fingerprint(&canonical_root, CACHE_SCHEMA_VERSION)
            .expect("fingerprint the fixture");
        let cache_path = cache_file_for_root(&canonical_root).expect("cache path for root");

        let other_generation_hash = current_toolchain_hash() ^ 0xDEAD_BEEF_DEAD_BEEF;
        let payload = format!("FP {:016x}\nWARN 0\nLAW 0\nWSUM 14650fb0739d0383\n", fp);
        let header = PackHeader {
            kind: PACK_KIND_SEM,
            schema_version: CACHE_SCHEMA_VERSION,
            toolchain_hash: other_generation_hash,
            feature_hash: current_feature_hash(),
            payload_len: payload.len() as u64,
            payload_checksum: fnv1a64(payload.as_bytes()),
        };
        let mut bytes = encode_pack_header(&header);
        bytes.extend_from_slice(payload.as_bytes());
        std::fs::write(&cache_path, bytes).expect("seed stale cross-generation cache entry");

        cmd_check(&[source_path.to_string_lossy().into_owned()]).expect("check must pass");

        let rewritten = std::fs::read(&cache_path).expect("cache file must exist after check");
        let (rewritten_header, _) =
            decode_pack_header(&rewritten).expect("rewritten cache header must decode");
        assert_eq!(
            rewritten_header.toolchain_hash,
            current_toolchain_hash(),
            "a check that correctly rejected the other-generation entry must rewrite the cache \
             with the real current generation, not silently keep serving the stale one"
        );
        assert_ne!(
            rewritten_header.toolchain_hash, other_generation_hash,
            "the stale cross-generation entry must not have been left in place"
        );

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&cache_path);
    }

    #[test]
    fn module_fingerprint_changes_on_dependency_edit() {
        let dir = mk_temp_dir("smc_mod_fp");
        let root = dir.join("root.sm");
        let child = dir.join("child.sm");
        std::fs::write(
            &root,
            r#"
Import "child.sm"
Law "R" [priority 1]:
    When true -> System.recovery()
"#,
        )
        .expect("write root");
        std::fs::write(
            &child,
            r#"
Law "C" [priority 1]:
    When true -> System.recovery()
"#,
        )
        .expect("write child");
        let fp1 = module_graph_fingerprint(&root, CACHE_SCHEMA_VERSION).expect("fp1");
        std::fs::write(
            &child,
            r#"
Law "C2" [priority 2]:
    When true -> System.recovery()
"#,
        )
        .expect("rewrite child");
        let fp2 = module_graph_fingerprint(&root, CACHE_SCHEMA_VERSION).expect("fp2");
        assert_ne!(fp1, fp2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn downstream_pack_keys_follow_dependency_changes_but_ast_stays_root_scoped() {
        let dir = mk_temp_dir("smc_pack_dep_keys");
        let root = dir.join("root.sm");
        let child = dir.join("child.sm");
        std::fs::write(
            &root,
            r#"
Import "child.sm"
Law "Root" [priority 1]:
    When true -> System.recovery()
"#,
        )
        .expect("write root");
        std::fs::write(
            &child,
            r#"
Law "Child" [priority 1]:
    When true -> System.recovery()
"#,
        )
        .expect("write child");

        let source_before = std::fs::read_to_string(&root).expect("read root before");
        let ast_before = ast_pack_key(&root, &source_before).expect("ast before");
        let ir_before = ir_pack_key(&root, &source_before, CompileProfile::Auto, OptLevel::O0)
            .expect("ir before");
        let smc_before = smc_pack_key(
            &root,
            &source_before,
            CompileProfile::Auto,
            OptLevel::O0,
            false,
        )
        .expect("smc before");

        std::fs::write(
            &child,
            r#"
Law "Child2" [priority 2]:
    When true -> System.recovery()
"#,
        )
        .expect("rewrite child");

        let source_after = std::fs::read_to_string(&root).expect("read root after");
        let ast_after = ast_pack_key(&root, &source_after).expect("ast after");
        let ir_after = ir_pack_key(&root, &source_after, CompileProfile::Auto, OptLevel::O0)
            .expect("ir after");
        let smc_after = smc_pack_key(
            &root,
            &source_after,
            CompileProfile::Auto,
            OptLevel::O0,
            false,
        )
        .expect("smc after");

        assert_eq!(
            ast_before, ast_after,
            "AST pack key should stay root-scoped"
        );
        assert_ne!(
            ir_before, ir_after,
            "IR pack key should track dependency changes"
        );
        assert_ne!(
            smc_before, smc_after,
            "SMC pack key should track dependency changes"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn parse_compile_profile(v: &str) -> Result<CompileProfile, String> {
    match v.to_ascii_lowercase().as_str() {
        "auto" => Ok(CompileProfile::Auto),
        "rust" | "rustlike" | "rust-like" => Ok(CompileProfile::RustLike),
        "logos" => Ok(CompileProfile::Logos),
        _ => Err(format!(
            "invalid --profile '{}', expected auto|rust|logos",
            v
        )),
    }
}

fn parse_opt_level(v: &str) -> Result<OptLevel, String> {
    match v.to_ascii_uppercase().as_str() {
        "O0" => Ok(OptLevel::O0),
        "O1" => Ok(OptLevel::O1),
        _ => Err(format!("invalid --opt-level '{}', expected O0|O1", v)),
    }
}

fn cmd_snapshots(args: &[String]) -> Result<(), String> {
    if args.len() > 1 {
        return Err("usage: smc snapshots [--update]".to_string());
    }
    let update = args.first().map(|s| s.as_str()) == Some("--update");
    let mut cmd = Command::new("cargo");
    cmd.arg("test")
        .arg("--test")
        .arg("golden_snapshots")
        .arg("-q");
    if update {
        cmd.env("SM_UPDATE_SNAPSHOTS", "1");
    }
    let status = cmd
        .status()
        .map_err(|e| format!("failed to run cargo test: {}", e))?;
    if status.success() {
        if update {
            println!("snapshot tests passed and snapshots updated");
        } else {
            println!("snapshot tests passed");
        }
        Ok(())
    } else {
        Err(format!("snapshot tests failed with status {}", status))
    }
}

fn cmd_features(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("usage: smc features".to_string());
    }
    let mut enabled = Vec::new();
    let mut disabled = Vec::new();
    for (name, on) in [
        ("std", cfg!(feature = "std")),
        ("profile-rust", cfg!(feature = "profile-rust")),
        ("profile-logos", cfg!(feature = "profile-logos")),
        ("debug-symbols", cfg!(feature = "debug-symbols")),
        ("simd", cfg!(feature = "simd")),
        ("bench", cfg!(feature = "bench")),
    ] {
        if on {
            enabled.push(name);
        } else {
            disabled.push(name);
        }
    }
    println!("enabled: {}", enabled.join(", "));
    println!("disabled: {}", disabled.join(", "));
    Ok(())
}

fn cmd_repl(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("usage: smc repl".to_string());
    }

    println!("Semantic Language REPL (Semantic Language check mode)");
    println!("commands: :help, :check, :clear, :quit");

    let mut buffer = String::new();
    let mut line = String::new();

    loop {
        if buffer.trim().is_empty() {
            print!("smc> ");
        } else {
            print!("...> ");
        }
        io::stdout()
            .flush()
            .map_err(|e| format!("stdout flush failed: {}", e))?;

        line.clear();
        let n = io::stdin()
            .read_line(&mut line)
            .map_err(|e| format!("stdin read failed: {}", e))?;
        if n == 0 {
            println!();
            break;
        }

        let trimmed = line.trim_end_matches(['\r', '\n']);
        match trimmed {
            ":quit" | ":q" | ":exit" => break,
            ":help" => {
                println!(":check   run smc analysis for current buffer");
                println!(":clear   clear current buffer");
                println!(":quit    exit REPL");
                continue;
            }
            ":clear" => {
                buffer.clear();
                println!("buffer cleared");
                continue;
            }
            ":check" => {
                if buffer.trim().is_empty() {
                    println!("buffer is empty");
                    continue;
                }
                run_repl_check(&buffer);
                continue;
            }
            _ => {}
        }

        buffer.push_str(trimmed);
        buffer.push('\n');

        if trimmed.is_empty() {
            run_repl_check(&buffer);
        }
    }
    Ok(())
}

fn run_repl_check(buffer: &str) {
    let color_enabled = resolve_color_mode(ColorMode::Auto);
    let parser_profile = cli_profile();
    match check_source_with_profile(buffer, &parser_profile) {
        Ok(report) => {
            for w in &report.warnings {
                print_diag_colored(color_enabled, &w.rendered);
            }
            println!(
                "ok: {} warning(s), {} scheduled law(s)",
                report.warnings.len(),
                report.scheduled_laws.len()
            );
        }
        Err(e) => {
            print_diag_colored(color_enabled, &e.to_string());
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ControlledObservationCliEnvelope {
    capability_decision: HelloObservationCapabilityDecision,
    audit_results: Vec<ControlledObservationAuditResult>,
    rendered_lines: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlledObservationSummary {
    pub sequence_index: u64,
    pub observation_class: HelloObservationClass,
    pub text_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlledObservationQualificationEnvelope {
    pub capability_decision: HelloObservationCapabilityDecision,
    pub audit_results: Vec<ControlledObservationAuditResult>,
    pub observations: Vec<ControlledObservationSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ControlledObservationCollectedEvent {
    sequence_index: u64,
    observation_class: HelloObservationClass,
    text_hash: u64,
    text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ControlledObservationInternalEnvelope {
    capability_decision: HelloObservationCapabilityDecision,
    audit_results: Vec<ControlledObservationAuditResult>,
    events: Vec<ControlledObservationCollectedEvent>,
}

fn stable_text_hash(text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn collect_controlled_observation_envelope(
    bytes: &[u8],
) -> Result<ControlledObservationInternalEnvelope, String> {
    // #1762 (FA-08-004): construct exactly one ExecutionConfig and thread
    // it, unchanged, through admission, execution, and provenance -
    // never reconstruct a second, independently-hardcoded config that
    // happens to agree today. Both `verify_semcode_token_with_quotas` and
    // `run_semcode_collecting_hello_observations_with_config` consume
    // this same instance, so the audit metadata built from it below
    // records the actual authority, not a value derived after the fact.
    let execution_config = ExecutionConfig::for_context(ExecutionContext::VerifiedLocal);

    verify_semcode_token_with_quotas(bytes, execution_config.quotas)
        .map_err(|report| report.to_string())?;

    let events = run_semcode_collecting_hello_observations_with_config(bytes, execution_config)
        .map_err(|e| e.to_string())?;
    let mut capability_manifest = CapabilityManifest::new();
    capability_manifest.allow(CapabilityKind::ControlledObservationSink);

    let capability_context = HelloObservationCapabilityContext {
        observation_sink_present: true,
        sink_available: true,
        requested_host_channel: None,
    };
    let capability_decision =
        require_hello_observation_sink_capability(&capability_manifest, &capability_context);
    let HelloObservationCapabilityDecision::Allow = capability_decision else {
        return Err(format!(
            "controlled observation capability denied: {:?}",
            capability_decision
        ));
    };

    // Reuses the SAME `execution_config` already passed to
    // `verify_semcode_token_with_quotas` and
    // `run_semcode_collecting_hello_observations_with_config` above - the
    // recorded provenance is the actual authority, not a value
    // reconstructed after the fact (#1762).
    let mut audit_trail = AuditTrail::new(AuditSessionMetadata {
        context: execution_config.context,
        quotas: execution_config.quotas,
        capability_manifest: capability_manifest.metadata(),
        gate_registry_bound: true,
    });
    let linkage = HelloObservationAuditLinkage {
        verifier_admission_ref: Some(1),
        capability_policy_ref: Some(2),
        sink_policy_ref: Some(3),
    };

    let mut audit_results = Vec::with_capacity(events.len());
    let mut collected_events = Vec::with_capacity(events.len());
    for (expected_index, event) in events.into_iter().enumerate() {
        if event.observation_class != HelloObservationClass::ControlledText {
            return Err(format!(
                "unexpected observation class in CLI envelope: {:?}",
                event.observation_class
            ));
        }
        if event.sequence_index.0 != expected_index as u64 {
            return Err(format!(
                "nondeterministic observation order: expected sequence_index {} but got {}",
                expected_index, event.sequence_index.0
            ));
        }
        let audit_result = apply_controlled_observation_audit_policy(
            &mut audit_trail,
            stable_text_hash(&event.text),
            event.sequence_index.0,
            ControlledObservationAuditDecision::Record,
            linkage,
        );
        match audit_result {
            ControlledObservationAuditResult::Recorded(AuditEventId(_)) => {}
            ControlledObservationAuditResult::NoStore => {
                return Err("audit policy unexpectedly returned no_store".to_string());
            }
            ControlledObservationAuditResult::Denied => {
                return Err("audit policy unexpectedly denied controlled observation".to_string());
            }
        }
        audit_results.push(audit_result);
        collected_events.push(ControlledObservationCollectedEvent {
            sequence_index: event.sequence_index.0,
            observation_class: event.observation_class,
            text_hash: stable_text_hash(&event.text),
            text: event.text,
        });
    }

    Ok(ControlledObservationInternalEnvelope {
        capability_decision,
        audit_results,
        events: collected_events,
    })
}

pub(crate) fn qualify_controlled_observation_envelope(
    bytes: &[u8],
) -> Result<ControlledObservationQualificationEnvelope, String> {
    let internal = collect_controlled_observation_envelope(bytes)?;
    Ok(ControlledObservationQualificationEnvelope {
        capability_decision: internal.capability_decision,
        audit_results: internal.audit_results,
        observations: internal
            .events
            .into_iter()
            .map(|event| ControlledObservationSummary {
                sequence_index: event.sequence_index,
                observation_class: event.observation_class,
                text_hash: event.text_hash,
            })
            .collect(),
    })
}

fn render_controlled_observation_envelope(
    bytes: &[u8],
) -> Result<ControlledObservationCliEnvelope, String> {
    let internal = collect_controlled_observation_envelope(bytes)?;
    Ok(ControlledObservationCliEnvelope {
        capability_decision: internal.capability_decision,
        audit_results: internal.audit_results,
        rendered_lines: internal
            .events
            .into_iter()
            .map(|event| event.text)
            .collect(),
    })
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    if args.len() == 1 {
        return cmd_run_controlled_observation(&args[0]);
    }
    let options = parse_application_run_options(args)?;
    let input = options.input.as_str();
    reject_leading_unknown_flag(input)?;
    let input_path = Path::new(input);
    let root = if input_path.is_dir() {
        resolve_project_root_check_entry(input_path)?
    } else {
        input_path.to_path_buf()
    };
    let src = read_source_with_package_admission(&root)?;
    let bytes = compile_program_to_semcode(&src).map_err(|e| e.to_string())?;
    let token = verify_semcode_token(&bytes).map_err(|error| error.to_string())?;
    let entry = token
        .require_entry("main")
        .map_err(|error| error.to_string())?;
    let capabilities = CapabilityManifest::for_application_profile(options.profile);
    let mut host = CliApplicationHost::new(
        &options.root,
        options.application_args,
        options.duration_millis,
    )?;
    let result = run_verified_entry_semcode_with_application_host_and_capabilities_and_config(
        &entry,
        &mut host,
        &capabilities,
        ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
    );
    match &result {
        Err(RuntimeError::CapabilityDenied(denied)) => {
            if let Some(call) = denied.call {
                host.record_denied(call);
            }
        }
        Err(RuntimeError::HostAbi(denied)) => host.record_denied(denied.call),
        _ => {}
    }
    for record in host.audit() {
        eprintln!("{}", record.render());
    }
    result.map_err(|error| error.to_string())
}

fn cmd_run_controlled_observation(input: &str) -> Result<(), String> {
    reject_leading_unknown_flag(input)?;
    let input_path = Path::new(input);
    let root = if input_path.is_dir() {
        resolve_project_root_check_entry(input_path)?
    } else {
        input_path.to_path_buf()
    };
    let src = read_source_with_package_admission(&root)?;
    let bytes = compile_program_to_semcode(&src).map_err(|e| e.to_string())?;
    let envelope = render_controlled_observation_envelope(&bytes)?;
    for line in envelope.rendered_lines {
        println!("{line}");
    }
    Ok(())
}

struct ApplicationRunOptions {
    input: String,
    profile: ApplicationCapabilityProfile,
    root: PathBuf,
    duration_millis: Option<u32>,
    application_args: Vec<String>,
}

fn parse_application_run_options(args: &[String]) -> Result<ApplicationRunOptions, String> {
    if args.is_empty() {
        return Err(application_run_usage());
    }
    let input = args[0].clone();
    let mut profile = None;
    let mut root = None;
    let mut duration_millis = None;
    let mut application_args = Vec::new();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--profile" => {
                index += 1;
                let value = args.get(index).ok_or_else(application_run_usage)?;
                profile =
                    Some(match value.as_str() {
                        "pure" => ApplicationCapabilityProfile::Pure,
                        "cli-read-only" => ApplicationCapabilityProfile::CliReadOnly,
                        "cli-file-transform" => ApplicationCapabilityProfile::CliFileTransform,
                        "ui-bounded" => return Err(
                            "ui-bounded is a catalogued profile and is not a CLI execution mode"
                                .to_string(),
                        ),
                        _ => return Err(format!("unknown application profile '{value}'")),
                    });
            }
            "--root" => {
                index += 1;
                root = Some(PathBuf::from(
                    args.get(index).ok_or_else(application_run_usage)?,
                ));
            }
            "--duration-ms" => {
                index += 1;
                duration_millis = Some(
                    args.get(index)
                        .ok_or_else(application_run_usage)?
                        .parse::<u32>()
                        .map_err(|_| "--duration-ms must be a u32 value".to_string())?,
                );
            }
            "--" => {
                application_args.extend_from_slice(&args[index + 1..]);
                break;
            }
            other => {
                return Err(format!(
                    "unknown run option '{other}'\n{}",
                    application_run_usage()
                ))
            }
        }
        index += 1;
    }
    Ok(ApplicationRunOptions {
        input,
        profile: profile.ok_or_else(application_run_usage)?,
        root: root.ok_or_else(application_run_usage)?,
        duration_millis,
        application_args,
    })
}

fn application_run_usage() -> String {
    "usage: smc run <input.sm|project-root> --profile <pure|cli-read-only|cli-file-transform> --root <directory> [--duration-ms <u32>] [-- <application-args...>]".to_string()
}

fn cmd_verify(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: smc verify <input.smc|project-root>".to_string());
    }
    let input = args[0].as_str();
    reject_leading_unknown_flag(input)?;
    let input_path = Path::new(input);
    let bytes = if input_path.is_dir() {
        let entry = resolve_project_root_check_entry(input_path)?;
        let source = read_source_with_package_admission(&entry)?;
        compile_program_to_semcode(&source).map_err(|error| error.to_string())?
    } else {
        std::fs::read(input).map_err(|e| format!("failed to read '{}': {}", input, e))?
    };
    let verified = verify_semcode(&bytes).map_err(|report| report.to_string())?;
    println!(
        "verified '{}' ({} function(s), header={}, epoch={}.{})",
        input,
        verified.functions.len(),
        String::from_utf8_lossy(&verified.header.magic),
        verified.header.epoch,
        verified.header.rev
    );
    Ok(())
}

fn cmd_test(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: smc test <project-root>".to_string());
    }
    let input = args[0].as_str();
    reject_leading_unknown_flag(input)?;
    let project_root = Path::new(input);
    if !project_root.is_dir() {
        return Err("smc test requires a project root directory".to_string());
    }

    resolve_project_root_check_entry(project_root)?;
    let project_root = project_root
        .canonicalize()
        .map_err(|error| format!("failed to resolve project root '{}': {error}", input))?;
    let tests = discover_project_test_sources(&project_root)?;
    if tests.is_empty() {
        return Err(format!(
            "project '{}' contains no tests/*.sm programs",
            project_root.display()
        ));
    }

    for test in &tests {
        let source = std::fs::read_to_string(test)
            .map_err(|error| format!("failed to read test '{}': {error}", test.display()))?;
        let bytes = compile_program_to_semcode(&source)
            .map_err(|error| format!("test '{}' failed to compile: {error}", test.display()))?;
        let token = verify_semcode_token(&bytes)
            .map_err(|error| format!("test '{}' failed verification: {error}", test.display()))?;
        let entry = token
            .require_entry("main")
            .map_err(|error| format!("test '{}' has no main entry: {error}", test.display()))?;
        let mut host = CliApplicationHost::new(&project_root, Vec::new(), None)?;
        run_verified_entry_semcode_with_application_host_and_capabilities_and_config(
            &entry,
            &mut host,
            &CapabilityManifest::for_application_profile(ApplicationCapabilityProfile::Pure),
            ExecutionConfig::for_context(ExecutionContext::VerifiedLocal),
        )
        .map_err(|error| format!("test '{}' failed: {error}", test.display()))?;
        let relative = test
            .strip_prefix(&project_root)
            .expect("discovered test remains inside project root")
            .to_string_lossy()
            .replace('\\', "/");
        println!("ok {relative}");
    }
    println!("test result: ok. {} passed", tests.len());
    Ok(())
}

fn discover_project_test_sources(project_root: &Path) -> Result<Vec<PathBuf>, String> {
    fn is_link_or_reparse(metadata: &std::fs::Metadata) -> bool {
        if metadata.file_type().is_symlink() {
            return true;
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
            metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
        }
        #[cfg(not(windows))]
        false
    }

    fn visit(root: &Path, directory: &Path, tests: &mut Vec<PathBuf>) -> Result<(), String> {
        let entries = std::fs::read_dir(directory).map_err(|error| {
            format!(
                "failed to read test directory '{}': {error}",
                directory.display()
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!(
                    "failed to inspect test directory '{}': {error}",
                    directory.display()
                )
            })?;
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
                format!("failed to inspect test path '{}': {error}", path.display())
            })?;
            if is_link_or_reparse(&metadata) {
                return Err(format!(
                    "test discovery rejects symbolic link or reparse path '{}'",
                    path.display()
                ));
            }
            if metadata.is_dir() {
                visit(root, &path, tests)?;
            } else if metadata.is_file() && path.extension().is_some_and(|ext| ext == "sm") {
                let canonical = path.canonicalize().map_err(|error| {
                    format!(
                        "failed to resolve test source '{}': {error}",
                        path.display()
                    )
                })?;
                let relative = canonical.strip_prefix(root).map_err(|_| {
                    format!("test source '{}' escapes the project root", path.display())
                })?;
                if relative.to_str().is_none() {
                    return Err(format!(
                        "test source '{}' is not valid UTF-8; project test paths must be UTF-8",
                        path.display()
                    ));
                }
                tests.push(canonical);
            }
        }
        Ok(())
    }

    let test_root = project_root.join("tests");
    if !test_root.exists() {
        return Ok(Vec::new());
    }
    let metadata = std::fs::symlink_metadata(&test_root).map_err(|error| {
        format!(
            "failed to inspect test directory '{}': {error}",
            test_root.display()
        )
    })?;
    if is_link_or_reparse(&metadata) || !metadata.is_dir() {
        return Err(format!(
            "project test root '{}' must be a real directory",
            test_root.display()
        ));
    }
    let mut tests = Vec::new();
    visit(project_root, &test_root, &mut tests)?;
    tests.sort_by_key(|path| {
        path.strip_prefix(project_root)
            .expect("discovered test remains inside project root")
            .to_str()
            .expect("test source path validated as UTF-8 during discovery")
            .replace('\\', "/")
    });
    Ok(tests)
}

fn cmd_run_smc(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: smc run-smc <input.smc>".to_string());
    }
    let input = args[0].as_str();
    reject_leading_unknown_flag(input)?;
    let bytes = std::fs::read(input).map_err(|e| format!("failed to read '{}': {}", input, e))?;
    let envelope = render_controlled_observation_envelope(&bytes)?;
    for line in envelope.rendered_lines {
        println!("{line}");
    }
    Ok(())
}

fn cmd_disasm(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: smc disasm <input.smc>".to_string());
    }
    let input = args[0].as_str();
    let bytes = std::fs::read(input).map_err(|e| format!("failed to read '{}': {}", input, e))?;
    let text = disasm_semcode(&bytes).map_err(|e| e.to_string())?;
    print!("{text}");
    Ok(())
}

fn usage() -> String {
    [
        "Semantic Language toolchain v0",
        "  smc compile <input.sm|project-root> -o <out.smc> [--profile auto|rust] [--opt-level O0|O1] [--debug-symbols] [--metrics]",
        "  smc check <input.sm|project-root> [--no-cache] [--trace-cache] [--metrics] [--deny warnings|<CODE>] [--color auto|always|never]",
        "  smc lint <input.sm> [--no-cache] [--trace-cache] [--deny warnings|<CODE>] [--color auto|always|never]",
        "  smc watch <input.sm> [--metrics] [--color auto|always|never]",
        "  smc fmt [--check] <path>",
        "  smc dump-ast <input.sm>",
        "  smc dump-ir <input.sm> [--profile auto|rust|logos] [--opt-level O0|O1|--opt]",
        "  smc dump-bytecode <input.sm> [--profile auto|rust] [--opt-level O0|O1|--opt] [--debug-symbols]",
        "  smc hash-ast <input.sm|project-root>",
        "  smc hash-ir <input.sm|project-root> [--profile auto|rust|logos] [--opt-level O0|O1|--opt]",
        "  smc hash-smc <input.sm|project-root> [--profile auto|rust] [--opt-level O0|O1|--opt] [--debug-symbols]",
        "  smc snapshots [--update]",
        "  smc features",
        "  smc explain <error-code|--list>",
        "  smc repl",
        "  smc work <subject> <intent> [to <target>] [with <profile>]",
        "  smc verify <input.smc|project-root>",
        "  smc test <project-root>",
        "  smc package inspect <project-root>",
        "  smc run <input.sm|project-root>",
        "  smc run <input.sm|project-root> --profile <pure|cli-read-only|cli-file-transform> --root <directory> [--duration-ms <u32>] [-- <application-args...>]",
        "  smc run-smc <input.smc>",
        "  smc disasm <input.smc>",
        "  smc look ui frame --from <snapshot> [--frame <n>] [--format text|draw-json] [--out <path>]",
        "  smc look ui frame <source-file> [--events <script>] [--frame <n>] [--format text|draw-json] [--out <path>]",
        "  smc hub tools",
        "  smc hub describe <tool-id>",
        "  smc hub invoke <tool-id> <operation-id> --input <file> [--out <file>]",
        "  smc hub audit --request <request-id>",
    ]
    .join("\n")
}

#[cfg(test)]
mod cli_observation_envelope_tests {
    use super::*;

    #[test]
    fn render_controlled_observation_envelope_applies_production_capability_and_audit() {
        let src = r#"
fn main() {
    print("Hello, World!");
}
"#;
        let bytes = compile_program_to_semcode(src).expect("compile canonical hello source");

        let envelope = render_controlled_observation_envelope(&bytes)
            .expect("controlled observation envelope should render");

        assert_eq!(
            envelope.capability_decision,
            HelloObservationCapabilityDecision::Allow
        );
        assert_eq!(envelope.rendered_lines, vec!["Hello, World!".to_string()]);
        assert_eq!(envelope.audit_results.len(), 1);
        assert!(matches!(
            envelope.audit_results[0],
            ControlledObservationAuditResult::Recorded(AuditEventId(0))
        ));
    }

    #[test]
    fn render_controlled_observation_envelope_preserves_sequence_order() {
        let src = r#"
fn main() {
    print("hello from Semantic");
    print("score=42");
}
"#;
        let bytes = compile_program_to_semcode(src).expect("compile two-event print source");

        let envelope = render_controlled_observation_envelope(&bytes)
            .expect("controlled observation envelope should render");

        assert_eq!(
            envelope.rendered_lines,
            vec!["hello from Semantic".to_string(), "score=42".to_string(),]
        );
        assert_eq!(envelope.audit_results.len(), 2);
        assert!(matches!(
            envelope.audit_results[0],
            ControlledObservationAuditResult::Recorded(AuditEventId(0))
        ));
        assert!(matches!(
            envelope.audit_results[1],
            ControlledObservationAuditResult::Recorded(AuditEventId(1))
        ));
    }

    #[test]
    fn qualify_controlled_observation_envelope_is_non_rendering() {
        let src = r#"
fn main() {
    print("Hello, World!");
}
"#;
        let bytes = compile_program_to_semcode(src).expect("compile canonical hello source");

        let envelope = qualify_controlled_observation_envelope(&bytes)
            .expect("controlled observation qualification should succeed");

        assert_eq!(
            envelope.capability_decision,
            HelloObservationCapabilityDecision::Allow
        );
        assert_eq!(envelope.observations.len(), 1);
        assert_eq!(envelope.observations[0].sequence_index, 0);
        assert_eq!(
            envelope.observations[0].observation_class,
            HelloObservationClass::ControlledText
        );
        assert_eq!(
            envelope.observations[0].text_hash,
            stable_text_hash("Hello, World!")
        );
        assert_eq!(envelope.audit_results.len(), 1);
        assert!(matches!(
            envelope.audit_results[0],
            ControlledObservationAuditResult::Recorded(AuditEventId(0))
        ));
    }

    #[test]
    fn qualification_and_rendering_share_observation_order() {
        let src = r#"
fn main() {
    print("hello from Semantic");
    print("score=42");
}
"#;
        let bytes = compile_program_to_semcode(src).expect("compile two-event print source");

        let qualification = qualify_controlled_observation_envelope(&bytes)
            .expect("controlled observation qualification should succeed");
        let rendering = render_controlled_observation_envelope(&bytes)
            .expect("controlled observation envelope should render");

        assert_eq!(
            qualification
                .observations
                .iter()
                .map(|observation| observation.sequence_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            qualification
                .observations
                .iter()
                .map(|observation| observation.text_hash)
                .collect::<Vec<_>>(),
            vec![
                stable_text_hash("hello from Semantic"),
                stable_text_hash("score=42")
            ]
        );
        assert_eq!(
            rendering.rendered_lines,
            vec!["hello from Semantic".to_string(), "score=42".to_string()]
        );
        assert_eq!(qualification.audit_results.len(), 2);
        assert!(matches!(
            qualification.audit_results[0],
            ControlledObservationAuditResult::Recorded(AuditEventId(0))
        ));
        assert!(matches!(
            qualification.audit_results[1],
            ControlledObservationAuditResult::Recorded(AuditEventId(1))
        ));
    }
}
