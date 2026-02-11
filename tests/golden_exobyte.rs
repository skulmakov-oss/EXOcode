use exocode_core::frontend::compile_program_to_exobyte;

fn check_golden(base: &str) {
    let src_path = format!("tests/golden/{}.exo", base);
    let bin_path = format!("tests/golden/{}.exb", base);
    let src = std::fs::read_to_string(&src_path).expect("read .exo");
    let expected = std::fs::read(&bin_path).expect("read .exb");
    let got = compile_program_to_exobyte(&src).expect("compile");
    if got != expected {
        panic!("{}", format_diff(&expected, &got));
    }
}

fn format_diff(expected: &[u8], got: &[u8]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "golden mismatch: expected {} bytes, got {} bytes\n",
        expected.len(),
        got.len()
    ));
    let n = expected.len().min(got.len());
    let mut mismatches = 0usize;
    for i in 0..n {
        if expected[i] != got[i] {
            out.push_str(&format!(
                "  @{:04x}: expected {:02x}, got {:02x}\n",
                i, expected[i], got[i]
            ));
            mismatches += 1;
            if mismatches >= 32 {
                out.push_str("  ... more mismatches omitted\n");
                break;
            }
        }
    }
    if expected.len() != got.len() {
        let shorter = n;
        let tail = if expected.len() > got.len() {
            &expected[shorter..]
        } else {
            &got[shorter..]
        };
        let shown = tail
            .iter()
            .take(16)
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>();
        out.push_str(&format!("  tail at {:04x}: {}\n", shorter, shown.join(" ")));
    }
    out
}

#[test]
fn golden_empty_main() {
    check_golden("empty_main");
}

#[test]
fn golden_call_ret() {
    check_golden("call_ret");
}

#[test]
fn golden_match_quad() {
    check_golden("match_quad");
}
