use exocode_core::exobyte_vm::{disasm_exobyte, run_exobyte};
use exocode_core::frontend::compile_program_to_exobyte;
use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() {
        return Err(usage());
    }
    match args[0].as_str() {
        "compile" => cmd_compile(&args[1..]),
        "run" => cmd_run(&args[1..]),
        "runb" => cmd_runb(&args[1..]),
        "disasm" => cmd_disasm(&args[1..]),
        "help" | "--help" | "-h" => Err(usage()),
        other => Err(format!("unknown command '{}'\n\n{}", other, usage())),
    }
}

fn cmd_compile(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: exoc compile <input.exo> -o <out.exb>".to_string());
    }
    let input = args[0].as_str();
    let mut out: Option<&str> = None;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                i += 1;
                out = args.get(i).map(|s| s.as_str());
            }
            other => return Err(format!("unknown flag '{}'", other)),
        }
        i += 1;
    }
    let out = out.ok_or_else(|| "missing -o <out.exb>".to_string())?;
    let src =
        std::fs::read_to_string(input).map_err(|e| format!("failed to read '{}': {}", input, e))?;
    let bytes = compile_program_to_exobyte(&src).map_err(|e| e.to_string())?;
    std::fs::write(out, &bytes).map_err(|e| format!("failed to write '{}': {}", out, e))?;
    println!("compiled '{}' -> '{}' ({} bytes)", input, out, bytes.len());
    Ok(())
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: exoc run <input.exo>".to_string());
    }
    let input = args[0].as_str();
    let src =
        std::fs::read_to_string(input).map_err(|e| format!("failed to read '{}': {}", input, e))?;
    let bytes = compile_program_to_exobyte(&src).map_err(|e| e.to_string())?;
    run_exobyte(&bytes).map_err(|e| e.to_string())
}

fn cmd_runb(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: exoc runb <input.exb>".to_string());
    }
    let input = args[0].as_str();
    let bytes = std::fs::read(input).map_err(|e| format!("failed to read '{}': {}", input, e))?;
    run_exobyte(&bytes).map_err(|e| e.to_string())
}

fn cmd_disasm(args: &[String]) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: exoc disasm <input.exb>".to_string());
    }
    let input = args[0].as_str();
    let bytes = std::fs::read(input).map_err(|e| format!("failed to read '{}': {}", input, e))?;
    let text = disasm_exobyte(&bytes).map_err(|e| e.to_string())?;
    print!("{text}");
    Ok(())
}

fn usage() -> String {
    [
        "EXOcode toolchain v0",
        "  exoc compile <input.exo> -o <out.exb>",
        "  exoc run <input.exo>",
        "  exoc runb <input.exb>",
        "  exoc disasm <input.exb>",
    ]
    .join("\n")
}
