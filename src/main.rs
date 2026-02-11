use exocode_core::language::{
    compile_human_program, execute_machine_program, parse_machine_program, render_machine_program,
};
use exocode_core::parser::{train_profile_in_place, ParserProfile, TrainingSample};
use exocode_core::{QuadroReg, F, N, S, T};
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::process::ExitCode;

#[derive(Debug, Deserialize)]
struct JsonSample {
    input: String,
    target: String,
}

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(1)
        }
    }
}

fn run(args: Vec<String>) -> Result<ExitCode, String> {
    if args.is_empty() {
        println!("{}", usage());
        return Ok(ExitCode::SUCCESS);
    }

    match args[0].as_str() {
        "help" | "--help" | "-h" => {
            println!("{}", usage());
            Ok(ExitCode::SUCCESS)
        }
        "profile" => {
            run_profile(&args[1..])?;
            Ok(ExitCode::SUCCESS)
        }
        "lang" => {
            run_lang(&args[1..])?;
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!("unknown command '{}'\n\n{}", other, usage())),
    }
}

fn run_profile(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(profile_usage());
    }

    match args[0].as_str() {
        "train" => cmd_profile_train(&args[1..]),
        "save" => cmd_profile_save(&args[1..]),
        "load" => cmd_profile_load(&args[1..]),
        other => Err(format!(
            "unknown profile subcommand '{}'\n\n{}",
            other,
            profile_usage()
        )),
    }
}

fn cmd_profile_train(args: &[String]) -> Result<(), String> {
    let mut samples_path: Option<&str> = None;
    let mut out_path: Option<&str> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--samples" => {
                i += 1;
                samples_path = Some(req_arg(args, i, "--samples")?);
            }
            "--out" => {
                i += 1;
                out_path = Some(req_arg(args, i, "--out")?);
            }
            other => return Err(format!("unknown flag '{}'\n\n{}", other, train_usage())),
        }
        i += 1;
    }

    let samples_path = samples_path.ok_or_else(train_usage)?;
    let out_path = out_path.ok_or_else(train_usage)?;

    let raw = std::fs::read_to_string(samples_path)
        .map_err(|e| format!("failed to read samples '{}': {}", samples_path, e))?;
    let samples_json: Vec<JsonSample> = serde_json::from_str(&raw)
        .map_err(|e| format!("invalid samples JSON '{}': {}", samples_path, e))?;

    let mut profile = ParserProfile::default();
    let borrowed: Vec<TrainingSample<'_>> = samples_json
        .iter()
        .map(|s| TrainingSample {
            input: &s.input,
            target: &s.target,
        })
        .collect();
    train_profile_in_place(&mut profile, &borrowed);
    profile
        .save_to_file(out_path)
        .map_err(|e| format!("failed to save profile '{}': {}", out_path, e))?;

    println!(
        "trained profile saved to '{}' ({} aliases)",
        out_path,
        profile.aliases.len()
    );
    Ok(())
}

fn cmd_profile_save(args: &[String]) -> Result<(), String> {
    let mut out_path: Option<&str> = None;
    let mut aliases: Vec<(String, String)> = Vec::new();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out_path = Some(req_arg(args, i, "--out")?);
            }
            "--alias" => {
                i += 1;
                let raw = req_arg(args, i, "--alias")?;
                aliases.push(parse_alias_pair(raw)?);
            }
            other => return Err(format!("unknown flag '{}'\n\n{}", other, save_usage())),
        }
        i += 1;
    }

    let out_path = out_path.ok_or_else(save_usage)?;
    let mut profile = ParserProfile::default();
    for (raw, canonical) in aliases {
        profile.add_alias(raw, canonical);
    }

    profile
        .save_to_file(out_path)
        .map_err(|e| format!("failed to save profile '{}': {}", out_path, e))?;
    println!(
        "profile saved to '{}' ({} aliases)",
        out_path,
        profile.aliases.len()
    );
    Ok(())
}

fn cmd_profile_load(args: &[String]) -> Result<(), String> {
    let mut in_path: Option<&str> = None;
    let mut as_json = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--in" => {
                i += 1;
                in_path = Some(req_arg(args, i, "--in")?);
            }
            "--json" => as_json = true,
            other => return Err(format!("unknown flag '{}'\n\n{}", other, load_usage())),
        }
        i += 1;
    }

    let in_path = in_path.ok_or_else(load_usage)?;
    let profile = ParserProfile::load_from_file(in_path)
        .map_err(|e| format!("failed to load profile '{}': {}", in_path, e))?;

    if as_json {
        let json = profile
            .to_json()
            .map_err(|e| format!("failed to encode profile as JSON: {}", e))?;
        println!("{json}");
    } else {
        println!("profile '{}' aliases: {}", in_path, profile.aliases.len());
        let mut keys: Vec<_> = profile.aliases.keys().collect();
        keys.sort_unstable();
        for key in keys {
            if let Some(val) = profile.aliases.get(key) {
                println!("{key} => {val}");
            }
        }
    }

    Ok(())
}

fn run_lang(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(lang_usage());
    }

    match args[0].as_str() {
        "compile" => cmd_lang_compile(&args[1..]),
        "run-human" => cmd_lang_run_human(&args[1..]),
        "run-machine" => cmd_lang_run_machine(&args[1..]),
        other => Err(format!(
            "unknown lang subcommand '{}'\n\n{}",
            other,
            lang_usage()
        )),
    }
}

fn cmd_lang_compile(args: &[String]) -> Result<(), String> {
    let mut in_path: Option<&str> = None;
    let mut out_path: Option<&str> = None;
    let mut profile_path: Option<&str> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--in" => {
                i += 1;
                in_path = Some(req_arg(args, i, "--in")?);
            }
            "--out" => {
                i += 1;
                out_path = Some(req_arg(args, i, "--out")?);
            }
            "--profile" => {
                i += 1;
                profile_path = Some(req_arg(args, i, "--profile")?);
            }
            other => {
                return Err(format!(
                    "unknown flag '{}'\n\n{}",
                    other,
                    lang_compile_usage()
                ))
            }
        }
        i += 1;
    }

    let in_path = in_path.ok_or_else(lang_compile_usage)?;
    let out_path = out_path.ok_or_else(lang_compile_usage)?;
    let human = std::fs::read_to_string(in_path)
        .map_err(|e| format!("failed to read input '{}': {}", in_path, e))?;
    let profile = load_profile_opt(profile_path)?;
    let program = compile_human_program(&human, profile.as_ref())
        .map_err(|e| format!("failed to compile human program: {}", e))?;
    let machine = render_machine_program(&program);
    std::fs::write(out_path, machine)
        .map_err(|e| format!("failed to write machine output '{}': {}", out_path, e))?;
    println!(
        "compiled '{}' -> '{}' ({} instructions)",
        in_path,
        out_path,
        program.instructions.len()
    );
    Ok(())
}

fn cmd_lang_run_human(args: &[String]) -> Result<(), String> {
    let mut in_path: Option<&str> = None;
    let mut profile_path: Option<&str> = None;
    let mut seeds: Vec<(String, u8)> = Vec::new();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--in" => {
                i += 1;
                in_path = Some(req_arg(args, i, "--in")?);
            }
            "--profile" => {
                i += 1;
                profile_path = Some(req_arg(args, i, "--profile")?);
            }
            "--set" => {
                i += 1;
                let raw = req_arg(args, i, "--set")?;
                seeds.push(parse_seed_pair(raw)?);
            }
            other => {
                return Err(format!(
                    "unknown flag '{}'\n\n{}",
                    other,
                    lang_run_human_usage()
                ))
            }
        }
        i += 1;
    }

    let in_path = in_path.ok_or_else(lang_run_human_usage)?;
    let human = std::fs::read_to_string(in_path)
        .map_err(|e| format!("failed to read input '{}': {}", in_path, e))?;
    let profile = load_profile_opt(profile_path)?;
    let program = compile_human_program(&human, profile.as_ref())
        .map_err(|e| format!("failed to compile human program: {}", e))?;

    let mut env = HashMap::<String, QuadroReg>::new();
    seed_env(&mut env, &seeds);
    execute_machine_program(&program, &mut env)
        .map_err(|e| format!("failed to execute machine program: {}", e))?;
    print_env(&env);
    Ok(())
}

fn cmd_lang_run_machine(args: &[String]) -> Result<(), String> {
    let mut in_path: Option<&str> = None;
    let mut seeds: Vec<(String, u8)> = Vec::new();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--in" => {
                i += 1;
                in_path = Some(req_arg(args, i, "--in")?);
            }
            "--set" => {
                i += 1;
                let raw = req_arg(args, i, "--set")?;
                seeds.push(parse_seed_pair(raw)?);
            }
            other => {
                return Err(format!(
                    "unknown flag '{}'\n\n{}",
                    other,
                    lang_run_machine_usage()
                ))
            }
        }
        i += 1;
    }

    let in_path = in_path.ok_or_else(lang_run_machine_usage)?;
    let machine_src = std::fs::read_to_string(in_path)
        .map_err(|e| format!("failed to read machine input '{}': {}", in_path, e))?;
    let program = parse_machine_program(&machine_src)
        .map_err(|e| format!("failed to parse machine program: {}", e))?;
    let mut env = HashMap::<String, QuadroReg>::new();
    seed_env(&mut env, &seeds);
    execute_machine_program(&program, &mut env)
        .map_err(|e| format!("failed to execute machine program: {}", e))?;
    print_env(&env);
    Ok(())
}

fn load_profile_opt(path: Option<&str>) -> Result<Option<ParserProfile>, String> {
    match path {
        Some(p) => ParserProfile::load_from_file(p)
            .map(Some)
            .map_err(|e| format!("failed to load profile '{}': {}", p, e)),
        None => Ok(None),
    }
}

fn seed_env(env: &mut HashMap<String, QuadroReg>, seeds: &[(String, u8)]) {
    for (name, state) in seeds {
        env.insert(name.clone(), state_to_reg(*state));
    }
}

fn state_to_reg(state: u8) -> QuadroReg {
    match state {
        N => QuadroReg::from_raw(0),
        F => QuadroReg::from_raw(exocode_core::LSB_MASK),
        T => QuadroReg::from_raw(exocode_core::MSB_MASK),
        S => QuadroReg::from_raw(u64::MAX),
        _ => QuadroReg::from_raw(0),
    }
}

fn parse_seed_pair(raw: &str) -> Result<(String, u8), String> {
    let (name, state) = raw
        .split_once('=')
        .ok_or_else(|| format!("invalid --set '{}', expected name=N|F|T|S", raw))?;
    if name.is_empty() {
        return Err(format!("invalid --set '{}', empty variable name", raw));
    }
    let state = match state {
        "N" => N,
        "F" => F,
        "T" => T,
        "S" => S,
        _ => {
            return Err(format!(
                "invalid state in --set '{}', expected N|F|T|S",
                raw
            ))
        }
    };
    Ok((name.to_string(), state))
}

fn print_env(env: &HashMap<String, QuadroReg>) {
    let mut keys: Vec<_> = env.keys().collect();
    keys.sort_unstable();
    for key in keys {
        if let Some(v) = env.get(key) {
            println!("{} = {:#018x}", key, v.raw());
        }
    }
}

fn req_arg<'a>(args: &'a [String], idx: usize, flag: &str) -> Result<&'a str, String> {
    args.get(idx)
        .map(|s| s.as_str())
        .ok_or_else(|| format!("missing value for {}\n\n{}", flag, usage()))
}

fn parse_alias_pair(raw: &str) -> Result<(String, String), String> {
    let (lhs, rhs) = raw
        .split_once('=')
        .ok_or_else(|| format!("invalid alias '{}', expected RAW=CANONICAL", raw))?;
    if lhs.is_empty() || rhs.is_empty() {
        return Err(format!("invalid alias '{}', expected RAW=CANONICAL", raw));
    }
    Ok((lhs.to_string(), rhs.to_string()))
}

fn usage() -> String {
    let mut out = String::new();
    out.push_str("Usage:\n");
    out.push_str("  exocode_core help\n");
    out.push_str("  exocode_core profile <train|save|load> [options]\n");
    out.push_str("  exocode_core lang <compile|run-human|run-machine> [options]\n");
    out.push_str("\nQuick Start:\n");
    out.push_str(
		"  exocode_core profile save --out profile.json --alias \"AND=&\" --alias \"OR=|\" --alias \"NOT=!\" --alias \"TRUE=T\" --alias \"FALSE=F\"\n",
	);
    out.push_str(
        "  exocode_core lang compile --in human.exo --out machine.exoasm --profile profile.json\n",
    );
    out.push_str("  exocode_core lang run-machine --in machine.exoasm\n\n");
    out.push_str(&profile_usage());
    out.push('\n');
    out.push_str(&lang_usage());
    out
}

fn profile_usage() -> String {
    format!("{}\n{}\n{}", train_usage(), save_usage(), load_usage())
}

fn train_usage() -> String {
    "  exocode_core profile train --samples <samples.json> --out <profile.json>\n  samples.json format: [{\"input\":\"out = a AND b\", \"target\":\"out = a & b\"}]".to_string()
}

fn save_usage() -> String {
    "  exocode_core profile save --out <profile.json> --alias RAW=CANONICAL [--alias RAW=CANONICAL ...]".to_string()
}

fn load_usage() -> String {
    "  exocode_core profile load --in <profile.json> [--json]".to_string()
}

fn lang_usage() -> String {
    format!(
        "{}\n{}\n{}",
        lang_compile_usage(),
        lang_run_human_usage(),
        lang_run_machine_usage()
    )
}

fn lang_compile_usage() -> String {
    "  exocode_core lang compile --in <human.exo> --out <machine.exoasm> [--profile <profile.json>]"
        .to_string()
}

fn lang_run_human_usage() -> String {
    "  exocode_core lang run-human --in <human.exo> [--profile <profile.json>] [--set var=N|F|T|S ...]".to_string()
}

fn lang_run_machine_usage() -> String {
    "  exocode_core lang run-machine --in <machine.exoasm> [--set var=N|F|T|S ...]".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_alias_pair_ok() {
        let (a, b) = parse_alias_pair("AND=&").expect("must parse");
        assert_eq!(a, "AND");
        assert_eq!(b, "&");
    }

    #[test]
    fn parse_alias_pair_rejects_invalid() {
        assert!(parse_alias_pair("ANDEQ").is_err());
        assert!(parse_alias_pair("=x").is_err());
        assert!(parse_alias_pair("x=").is_err());
    }

    #[test]
    fn parse_seed_pair_ok() {
        let (k, v) = parse_seed_pair("a=T").expect("must parse");
        assert_eq!(k, "a");
        assert_eq!(v, T);
    }

    #[test]
    fn parse_seed_pair_rejects_invalid() {
        assert!(parse_seed_pair("a=").is_err());
        assert!(parse_seed_pair("=T").is_err());
        assert!(parse_seed_pair("a=X").is_err());
    }
}
