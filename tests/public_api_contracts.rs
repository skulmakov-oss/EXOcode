use std::fs;

const PROM_REF_WRAPPERS: &[&str] = &[
    "CapabilityRef",
    "ActorRef",
    "SessionRef",
    "ClientRef",
    "RevisionRef",
    "EpochRef",
];

const TARGETS: &[(&str, &str)] = &[
    (
        "crates/sm-emit/src/lib.rs",
        "tests/golden_snapshots/public_api/sm_emit_lib.txt",
    ),
    (
        "crates/sm-emit/src/hello_real_semcode.rs",
        "tests/golden_snapshots/public_api/sm_emit_hello_real_semcode.txt",
    ),
    (
        "crates/sm-emit/src/hello_observation_bytes.rs",
        "tests/golden_snapshots/public_api/sm_emit_hello_observation_bytes.txt",
    ),
    (
        "crates/sm-format/src/lib.rs",
        "tests/golden_snapshots/public_api/sm_format_lib.txt",
    ),
    (
        "crates/sm-format/src/local_format.rs",
        "tests/golden_snapshots/public_api/sm_format_local_format.txt",
    ),
    (
        "crates/sm-format/src/semcode_decode.rs",
        "tests/golden_snapshots/public_api/sm_format_semcode_decode.txt",
    ),
    (
        "crates/sm-ir/src/lib.rs",
        "tests/golden_snapshots/public_api/sm_ir_lib.txt",
    ),
    (
        "crates/sm-profile/src/lib.rs",
        "tests/golden_snapshots/public_api/sm_profile_lib.txt",
    ),
    (
        "crates/sm-runtime-core/src/lib.rs",
        "tests/golden_snapshots/public_api/sm_runtime_core_lib.txt",
    ),
    (
        "crates/sm-verify/src/lib.rs",
        "tests/golden_snapshots/public_api/sm_verify_lib.txt",
    ),
    (
        "crates/sm-verify/src/hello_pending_admission.rs",
        "tests/golden_snapshots/public_api/sm_verify_hello_pending_admission.txt",
    ),
    (
        "crates/sm-verify/src/hello_real_semcode_admission.rs",
        "tests/golden_snapshots/public_api/sm_verify_hello_real_semcode_admission.txt",
    ),
    (
        "crates/sm-vm/src/lib.rs",
        "tests/golden_snapshots/public_api/sm_vm_lib.txt",
    ),
    (
        "crates/sm-vm/src/semcode_vm.rs",
        "tests/golden_snapshots/public_api/sm_vm_semcode_vm.txt",
    ),
    (
        "crates/prom-abi/src/lib.rs",
        "tests/golden_snapshots/public_api/prom_abi_lib.txt",
    ),
    (
        "crates/prom-cap/src/lib.rs",
        "tests/golden_snapshots/public_api/prom_cap_lib.txt",
    ),
    (
        "crates/prom-audit/src/lib.rs",
        "tests/golden_snapshots/public_api/prom_audit_lib.txt",
    ),
    (
        "crates/prom-runtime/src/lib.rs",
        "tests/golden_snapshots/public_api/prom_runtime_lib.txt",
    ),
    (
        "crates/smc-cli/src/lib.rs",
        "tests/golden_snapshots/public_api/smc_cli_lib.txt",
    ),
    (
        "crates/prom-refs/src/lib.rs",
        "tests/golden_snapshots/public_api/prom_refs_lib.txt",
    ),
    (
        "crates/prom-ui/src/shell_bridge.rs",
        "tests/golden_snapshots/public_api/prom_ui_shell_bridge.txt",
    ),
    (
        "crates/prom-ui-runtime/src/shell_player.rs",
        "tests/golden_snapshots/public_api/prom_ui_runtime_shell_player.txt",
    ),
    (
        "crates/prom-ui-runtime/src/reference_admission.rs",
        "tests/golden_snapshots/public_api/prom_ui_runtime_reference_admission.txt",
    ),
    (
        "crates/prom-ui-runtime/src/reference_contour.rs",
        "tests/golden_snapshots/public_api/prom_ui_runtime_reference_contour.txt",
    ),
    (
        "crates/semantic-hub/src/lib.rs",
        "tests/golden_snapshots/public_api/semantic_hub_lib.txt",
    ),
    (
        "crates/semantic-hub-turbovec/src/lib.rs",
        "tests/golden_snapshots/public_api/semantic_hub_turbovec_lib.txt",
    ),
];

fn normalize_ws(line: &str) -> String {
    line.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LexState {
    Normal,
    BlockComment(usize),
    NormalString,
    NormalStringContinuation,
    RawString(usize),
    ByteNormalString,
    ByteNormalStringContinuation,
    RawByteString(usize),
}

impl LexState {
    fn is_in_string_literal(self) -> bool {
        matches!(
            self,
            LexState::NormalString
                | LexState::NormalStringContinuation
                | LexState::RawString(_)
                | LexState::ByteNormalString
                | LexState::ByteNormalStringContinuation
                | LexState::RawByteString(_)
        )
    }

    fn is_escaped_continuation(self) -> bool {
        matches!(
            self,
            LexState::NormalStringContinuation | LexState::ByteNormalStringContinuation
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VisibleSegment {
    Code(String),
    Literal(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuralEventKind {
    TopLevelOpenBrace,
    TopLevelCloseBrace,
    TopLevelSemicolon,
    TopLevelComma,
    BaselineOpenBrace,
    MethodBodyClose,
    BaselineSemicolon,
    BaselineComma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StructuralEvent {
    seg_idx: usize,
    char_idx: usize,
    kind: StructuralEventKind,
}

#[derive(Debug, Clone)]
struct ScannedLine {
    segments: Vec<VisibleSegment>,
    code_tokens: String,
    events: Vec<StructuralEvent>,
    has_top_level_semicolon: bool,
    has_top_level_open_paren: bool,
    top_level_open_paren_segs: Vec<(usize, usize)>,
    top_level_close_paren_segs: Vec<(usize, usize)>,
    has_top_level_open_brace: bool,
    first_top_level_open_brace_seg: Option<(usize, usize)>,
    first_outer_attribute_close_seg: Option<(usize, usize)>,
    ends_in_string_literal: bool,
    /// True when this line's own scan ended with the lexer having just
    /// crossed an escaped-newline continuation (a trailing `\` inside a
    /// normal/byte string, e.g. `"abc\` at end of line) - narrower than
    /// `ends_in_string_literal` (which is also true for a genuine,
    /// unescaped multi-line string). `render_visible_range` never
    /// renders that trailing `\` (it is not part of the literal's actual
    /// value), so any caller extracting a SUB-RANGE of this line's text
    /// to be replayed through a later, independent `scan_line` call
    /// (rather than continuing to use this same `ScannedLine`'s own
    /// events/segments directly) needs this flag to know it must
    /// manually re-append `\` to that extracted text, or the
    /// continuation is silently lost on replay.
    ends_in_escaped_continuation: bool,
}

impl ScannedLine {
    fn render_visible(&self) -> String {
        self.render_visible_range(None, None)
    }

    fn text_up_to_function_body_open_brace(&self) -> String {
        self.render_visible_range(None, self.first_top_level_open_brace_seg)
    }

    fn render_visible_range(
        &self,
        start: Option<(usize, usize)>,
        end: Option<(usize, usize)>,
    ) -> String {
        self.render_visible_range_tagged(start, end).0
    }

    /// Same rendering as `render_visible_range`, but also returns the byte
    /// ranges within the returned string that came from
    /// `VisibleSegment::Literal` content, verbatim from the authoritative
    /// lexer - not re-derived by scanning the returned string for quote
    /// characters. Downstream consumers that need to know which bytes are
    /// literal (e.g. `normalize_variant`) must use these ranges instead of
    /// re-lexing the rendered text, so a literal's own content (including
    /// any embedded `"`, arbitrary `#` counts in raw strings, or internal
    /// whitespace) can never be misclassified as code.
    fn render_visible_range_tagged(
        &self,
        start: Option<(usize, usize)>,
        end: Option<(usize, usize)>,
    ) -> (String, Vec<(usize, usize)>) {
        if let (Some(s), Some(e)) = (start, end) {
            if s > e {
                return (String::new(), Vec::new());
            }
        }
        let mut out = String::new();
        let mut literal_ranges: Vec<(usize, usize)> = Vec::new();
        for (seg_idx, seg) in self.segments.iter().enumerate() {
            if let Some((s_seg, _)) = start {
                if seg_idx < s_seg {
                    continue;
                }
            }
            if let Some((e_seg, _)) = end {
                if seg_idx > e_seg {
                    break;
                }
            }

            match seg {
                VisibleSegment::Code(s) => {
                    let s_char = if start.is_some_and(|(s_seg, _)| s_seg == seg_idx) {
                        start.unwrap().1
                    } else {
                        0
                    };
                    let e_char = if end.is_some_and(|(e_seg, _)| e_seg == seg_idx) {
                        end.unwrap().1
                    } else {
                        s.len().saturating_sub(1)
                    };
                    if s_char <= e_char && s_char < s.len() {
                        let chunk = &s[s_char..=e_char.min(s.len() - 1)];
                        let norm = normalize_ws(chunk);
                        if !norm.is_empty() {
                            if !out.is_empty() && !out.ends_with(' ') {
                                let first = norm.chars().next().unwrap();
                                let last = out.chars().last().unwrap();
                                let skip_space = is_attached_punctuation_prefix(first)
                                    || is_attached_opening_delimiter(last)
                                    || (last == '#' && first == '[')
                                    || (out.ends_with("pub") && first == '(');
                                if !skip_space {
                                    out.push(' ');
                                }
                            }
                            out.push_str(&norm);
                        }
                    }
                }
                VisibleSegment::Literal(s) => {
                    let s_char = if start.is_some_and(|(s_seg, _)| s_seg == seg_idx) {
                        start.unwrap().1
                    } else {
                        0
                    };
                    let e_char = if end.is_some_and(|(e_seg, _)| e_seg == seg_idx) {
                        end.unwrap().1
                    } else {
                        s.len().saturating_sub(1)
                    };
                    if s_char <= e_char && s_char < s.len() {
                        let chunk = &s[s_char..=e_char.min(s.len() - 1)];
                        if !out.is_empty() && !out.ends_with(' ') {
                            let last = out.chars().last().unwrap();
                            if !is_attached_opening_delimiter(last) {
                                out.push(' ');
                            }
                        }
                        let lit_start = out.len();
                        out.push_str(chunk);
                        let lit_end = out.len();
                        literal_ranges.push((lit_start, lit_end));
                    }
                }
            }
        }
        (out, literal_ranges)
    }

    fn code_tokens_range(
        &self,
        start: Option<(usize, usize)>,
        end: Option<(usize, usize)>,
    ) -> String {
        if let (Some(s), Some(e)) = (start, end) {
            if s > e {
                return String::new();
            }
        }
        let mut out = String::new();
        for (seg_idx, seg) in self.segments.iter().enumerate() {
            if let Some((s_seg, _)) = start {
                if seg_idx < s_seg {
                    continue;
                }
            }
            if let Some((e_seg, _)) = end {
                if seg_idx > e_seg {
                    break;
                }
            }

            match seg {
                VisibleSegment::Code(s) => {
                    let s_char = if start.is_some_and(|(s_seg, _)| s_seg == seg_idx) {
                        start.unwrap().1
                    } else {
                        0
                    };
                    let e_char = if end.is_some_and(|(e_seg, _)| e_seg == seg_idx) {
                        end.unwrap().1
                    } else {
                        s.len().saturating_sub(1)
                    };
                    if s_char <= e_char && s_char < s.len() {
                        let chunk = &s[s_char..=e_char.min(s.len() - 1)];
                        if !out.is_empty() && !out.ends_with(' ') {
                            out.push(' ');
                        }
                        out.push_str(chunk);
                    }
                }
                VisibleSegment::Literal(_) => {
                    if !out.is_empty() && !out.ends_with(' ') {
                        out.push(' ');
                    }
                    out.push_str("\"\"");
                }
            }
        }
        out
    }
}

fn next_pos(segments: &[VisibleSegment], pos: (usize, usize)) -> Option<(usize, usize)> {
    let (seg_idx, char_idx) = pos;
    if seg_idx >= segments.len() {
        return None;
    }
    let seg_len = match &segments[seg_idx] {
        VisibleSegment::Code(s) => s.len(),
        VisibleSegment::Literal(s) => s.len(),
    };
    if char_idx + 1 < seg_len {
        Some((seg_idx, char_idx + 1))
    } else if seg_idx + 1 < segments.len() {
        Some((seg_idx + 1, 0))
    } else {
        None
    }
}

fn prev_pos(segments: &[VisibleSegment], pos: (usize, usize)) -> Option<(usize, usize)> {
    let (seg_idx, char_idx) = pos;
    if char_idx > 0 {
        Some((seg_idx, char_idx - 1))
    } else if seg_idx > 0 {
        let prev_seg = seg_idx - 1;
        let prev_len = match &segments[prev_seg] {
            VisibleSegment::Code(s) => s.len(),
            VisibleSegment::Literal(s) => s.len(),
        };
        if prev_len > 0 {
            Some((prev_seg, prev_len - 1))
        } else {
            None
        }
    } else {
        None
    }
}

fn is_attached_punctuation_prefix(c: char) -> bool {
    matches!(c, ';' | ',' | ':' | ')' | ']' | '}' | '>')
}

fn is_attached_opening_delimiter(c: char) -> bool {
    matches!(c, '(' | '[' | '{' | '<' | '*' | '&' | '!')
}

fn is_likely_comparison_less_than(code_tokens: &str) -> bool {
    let t = code_tokens.trim_end();
    if t.is_empty() {
        return false;
    }
    if t.ends_with(['\'', '"', ')', ']', '}']) {
        return true;
    }
    if t.ends_with("true") || t.ends_with("false") {
        return true;
    }
    if is_public_const_or_static(code_tokens) {
        if let Some((_, after_eq)) = t.rsplit_once('=') {
            let trimmed_after = after_eq.trim();
            if !trimmed_after.ends_with("::") && !trimmed_after.contains('<') {
                return true;
            }
        }
    }
    let mut last_token_start = None;
    for (idx, c) in t.char_indices().rev() {
        if c.is_alphanumeric() || c == '_' {
            last_token_start = Some(idx);
        } else {
            break;
        }
    }
    if let Some(start) = last_token_start {
        let token = &t[start..];
        if token.starts_with(|c: char| c.is_ascii_digit()) {
            return true;
        }
    }
    false
}

/// True when `s`'s trailing (trimmed) tokens end with the Rust keyword
/// `kw` at a genuine word boundary - never matched as a suffix of a longer
/// identifier (`"sometype"` does not end with keyword `"type"`). Used to
/// track, across physical lines, whether the declaration currently being
/// scanned started with a keyword whose `=` right-hand side is a TYPE
/// rather than a value expression (`type X = ...`), so that case can be
/// excluded from `CodeLexer::expr_initializer_active`.
fn ends_with_keyword(s: &str, kw: &str) -> bool {
    let t = s.trim_end();
    match t.strip_suffix(kw) {
        Some(before) => {
            before.is_empty() || before.ends_with(|c: char| !c.is_alphanumeric() && c != '_')
        }
        None => false,
    }
}

fn is_macro_bang(code_tokens: &str) -> bool {
    let s = code_tokens.trim_end();
    if s.is_empty() {
        return false;
    }
    let mut last_valid_byte = None;
    for (byte_idx, c) in s.char_indices().rev() {
        if c.is_alphanumeric() || c == '_' {
            last_valid_byte = Some(byte_idx);
        } else {
            break;
        }
    }
    let Some(start_byte) = last_valid_byte else {
        return false;
    };
    let ident = &s[start_byte..];
    if ident.starts_with(|c: char| c.is_numeric()) {
        return false;
    }
    true
}

#[derive(Debug, Clone)]
struct CodeLexer {
    state: LexState,
    angle_depth: usize,
    paren_depth: usize,
    bracket_depth: usize,
    brace_depth: usize,
    pending_macro_bang: bool,
    macro_brace_stack: Vec<usize>,
    const_brace_stack: Vec<(usize, usize)>, // (brace depth, angle depth at entry)
    /// Set the moment a plain `=` is seen at a declaration boundary
    /// (top-level: `paren_depth == bracket_depth == angle_depth == 0 &&
    /// brace_depth == 0`, or baseline-member: same but `brace_depth == 1`,
    /// which is where a top-level `const`/`static` initializer, an enum
    /// discriminant, or a trait/impl associated-const initializer/default
    /// all start). While set, `<`/`>`/`<<`/`>>` default to OPERATORS
    /// rather than generic delimiters, exactly like the existing
    /// `const_brace_stack` mechanism already does for const-generic
    /// `{ EXPR }` positions, closing the gap for the two declaration
    /// shapes that are const/expression contexts without ever being
    /// inside a const-generic angle frame at all. A trailing `::` still
    /// opens a turbofish/qualified-path generic exactly as before
    /// (`Foo::<Bar>`, `<T as Trait>::Assoc`), since that check is
    /// unchanged; this flag only widens which initializer shapes get the
    /// same treatment `const_brace_stack` already provides, not how a
    /// `<` is judged once inside one. Cleared at the same structural-
    /// event sites that already end a top-level or baseline declaration
    /// (`TopLevelSemicolon`, `BaselineSemicolon`, `TopLevelComma`,
    /// `BaselineComma`, `TopLevelCloseBrace`), never tied to
    /// `MethodBodyClose`, since a discriminant's own `{ ... }` block-
    /// expression closing brace can coincidentally match that same
    /// brace-depth pattern without ending the declaration itself (the
    /// terminating comma/semicolon that follows still does).
    expr_initializer_active: bool,
    /// Set once the keyword `type` is recognized (via `ends_with_keyword`,
    /// at a whitespace boundary) at the same declaration-boundary depth
    /// `expr_initializer_active` cares about, and cleared at the same
    /// sites. A plain `=` at that depth is only treated as the start of a
    /// value-expression initializer when this is FALSE - `type X = ...`
    /// and `type Item = DefaultType;` (an associated type default) have a
    /// TYPE on their right-hand side, not an expression, so their `<`/`>`
    /// must keep the ordinary generic-delimiter treatment (`Vec<Foo>` must
    /// still open a generic frame). This is a Rust reserved-keyword check,
    /// not an identifier/name heuristic - it is the same kind of
    /// structural keyword-prefix test `is_public_type_alias` already uses
    /// for top-level dispatch, just tracked persistently so it still
    /// applies when the keyword and the `=` land on different physical
    /// lines.
    pending_type_alias: bool,
    /// Set the moment a genuine, whole-word `where` keyword is recognized
    /// at true baseline depth (`paren_depth == bracket_depth ==
    /// angle_depth == brace_depth == 0` - an `impl`/`fn`/... header is
    /// always scanned before its own body brace opens, so this is the
    /// same depth `pending_type_alias` tracks at its top-level case).
    /// Once set, it marks every keyword read afterward as being inside
    /// this header's where-clause, not its own leading bound list -
    /// see `header_saw_trait_for` below, the sole consumer.
    header_saw_where: bool,
    /// Set the moment a genuine, whole-word `for` keyword is recognized
    /// at true baseline depth (same depth as `header_saw_where`) BEFORE
    /// `header_saw_where` has been set. This is the structural signature
    /// of an impl header's own trait-for-type separator (`impl<T> Trait
    /// for S<T>`) - a higher-ranked trait bound's `for<'a>` can never
    /// match it: written inside the header's own leading generic-bound
    /// list (`impl<T: for<'a> Fn(&'a T)> S<T>`), that `for` sits inside
    /// the OUTER `impl<...>` angle frame, so `angle_depth > 0` at the
    /// point it is read; written inside a trailing where-clause (`impl<T>
    /// S<T> where T: for<'a> Fn(&'a T)`), `header_saw_where` is already
    /// true by the time that `for` is read, since `where` always precedes
    /// every bound in its own clause. Checked once, after a fully-read
    /// impl header, by `is_inherent_impl_header` in place of the unsafe
    /// `contains_keyword`/`find_keyword` textual "for" search it used to
    /// do - this field IS the impl's own trait-for-type separator, read
    /// live off real-time depth-tracked lexer state rather than searched
    /// for after the fact in already-rendered text, so a legally
    /// reformatted or line-wrapped header (`for` and its type on
    /// different physical lines, `for<'a>` split across lines) cannot
    /// change the answer the way a position-based text search could.
    /// Reset at the same top-level boundaries as `pending_type_alias`,
    /// since an impl header (like a type alias) is scanned entirely
    /// before any of those boundaries and never spans across one.
    header_saw_trait_for: bool,
}

impl CodeLexer {
    fn new() -> Self {
        Self {
            state: LexState::Normal,
            angle_depth: 0,
            paren_depth: 0,
            bracket_depth: 0,
            brace_depth: 0,
            pending_macro_bang: false,
            macro_brace_stack: Vec::new(),
            const_brace_stack: Vec::new(),
            expr_initializer_active: false,
            pending_type_alias: false,
            header_saw_where: false,
            header_saw_trait_for: false,
        }
    }

    fn reset_top_level_depths(&mut self) {
        self.angle_depth = 0;
        self.paren_depth = 0;
        self.bracket_depth = 0;
        self.brace_depth = 0;
        self.pending_macro_bang = false;
        self.macro_brace_stack.clear();
        self.const_brace_stack.clear();
        self.expr_initializer_active = false;
        self.header_saw_where = false;
        self.header_saw_trait_for = false;
        self.pending_type_alias = false;
    }

    /// Finalizes `header_saw_where`/`header_saw_trait_for` against
    /// whatever `code_tokens` holds so far, at true baseline depth. This
    /// is the SOLE place either flag is set - called at every genuine
    /// Rust token boundary reachable while scanning a header (whitespace,
    /// `<`, a line or block comment starting, and the end of a physical
    /// line), not just whitespace, so a keyword immediately followed by
    /// trivia other than a plain space (`where/**/T`, `for//comment\n<`)
    /// is recognized exactly the same as one followed by a space. Callers
    /// pass `code_tokens` as it stands immediately BEFORE the boundary
    /// character/trivia is itself appended, matching how `ends_with_
    /// keyword` needs to see the keyword as the literal suffix.
    fn finalize_header_keywords(&mut self, code_tokens: &str) {
        if self.paren_depth == 0
            && self.bracket_depth == 0
            && self.angle_depth == 0
            && self.brace_depth == 0
        {
            if ends_with_keyword(code_tokens, "where") {
                self.header_saw_where = true;
            } else if !self.header_saw_where && ends_with_keyword(code_tokens, "for") {
                self.header_saw_trait_for = true;
            }
        }
    }

    fn scan_line(&mut self, line: &str) -> ScannedLine {
        let mut segments = Vec::new();
        let mut cur_code = String::new();
        let mut code_tokens = String::with_capacity(line.len());
        let mut events = Vec::new();
        let mut has_top_level_semicolon = false;
        let mut has_top_level_open_paren = false;
        let mut top_level_open_paren_segs = Vec::new();
        let mut top_level_close_paren_segs = Vec::new();
        let mut has_top_level_open_brace = false;
        let mut first_top_level_open_brace_seg = None;
        let mut first_outer_attribute_close_seg = None;

        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            match self.state {
                LexState::Normal => {
                    // 1. Line comment //
                    if chars[i] == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
                        // A comment is lexical trivia, a genuine Rust token
                        // boundary - `where//comment` (or `for//comment`)
                        // must finalize the keyword exactly like a plain
                        // space would, before the comment (and the rest of
                        // the physical line) is discarded below.
                        self.finalize_header_keywords(&code_tokens);
                        code_tokens.push(' ');
                        break;
                    }

                    // 2. Block comment start /*
                    if chars[i] == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
                        // Same token-boundary reasoning as the line-comment
                        // case above - `where/**/T` must finalize `where`
                        // here, since the comment (and whatever text starts
                        // the NEXT token right after it closes) never
                        // passes through the whitespace arm.
                        self.finalize_header_keywords(&code_tokens);
                        if !cur_code.is_empty() {
                            segments.push(VisibleSegment::Code(std::mem::take(&mut cur_code)));
                        }
                        self.state = LexState::BlockComment(1);
                        code_tokens.push(' ');
                        i += 2;
                        continue;
                    }

                    // 3. Raw byte string br#"..."# or br"..."
                    if chars[i] == 'b' && i + 1 < chars.len() && chars[i + 1] == 'r' {
                        let mut j = i + 2;
                        let mut hashes = 0;
                        while j < chars.len() && chars[j] == '#' {
                            hashes += 1;
                            j += 1;
                        }
                        if j < chars.len() && chars[j] == '"' {
                            if !cur_code.is_empty() {
                                segments.push(VisibleSegment::Code(std::mem::take(&mut cur_code)));
                            }
                            self.state = LexState::RawByteString(hashes);
                            segments.push(VisibleSegment::Literal(chars[i..=j].iter().collect()));
                            code_tokens.push_str("\"\"");
                            i = j + 1;
                            continue;
                        }
                    }

                    // 4. Raw string r#"..."# or r"..."
                    if chars[i] == 'r' {
                        let mut j = i + 1;
                        let mut hashes = 0;
                        while j < chars.len() && chars[j] == '#' {
                            hashes += 1;
                            j += 1;
                        }
                        if j < chars.len() && chars[j] == '"' {
                            if !cur_code.is_empty() {
                                segments.push(VisibleSegment::Code(std::mem::take(&mut cur_code)));
                            }
                            self.state = LexState::RawString(hashes);
                            segments.push(VisibleSegment::Literal(chars[i..=j].iter().collect()));
                            code_tokens.push_str("\"\"");
                            i = j + 1;
                            continue;
                        }
                    }

                    // 5. Byte string b"..."
                    if chars[i] == 'b' && i + 1 < chars.len() && chars[i + 1] == '"' {
                        if !cur_code.is_empty() {
                            segments.push(VisibleSegment::Code(std::mem::take(&mut cur_code)));
                        }
                        self.state = LexState::ByteNormalString;
                        segments.push(VisibleSegment::Literal(chars[i..=i + 1].iter().collect()));
                        code_tokens.push_str("\"\"");
                        i += 2;
                        continue;
                    }

                    // 6. Normal string "..."
                    if chars[i] == '"' {
                        if !cur_code.is_empty() {
                            segments.push(VisibleSegment::Code(std::mem::take(&mut cur_code)));
                        }
                        self.state = LexState::NormalString;
                        segments.push(VisibleSegment::Literal("\"".to_string()));
                        code_tokens.push_str("\"\"");
                        i += 1;
                        continue;
                    }

                    // 7. Byte char b'...'
                    if chars[i] == 'b' && i + 1 < chars.len() && chars[i + 1] == '\'' {
                        let is_simple_byte_char =
                            i + 3 < chars.len() && chars[i + 2] != '\\' && chars[i + 3] == '\'';
                        let is_esc_byte_char = i + 2 < chars.len() && chars[i + 2] == '\\';
                        if is_simple_byte_char {
                            if !cur_code.is_empty() {
                                segments.push(VisibleSegment::Code(std::mem::take(&mut cur_code)));
                            }
                            segments
                                .push(VisibleSegment::Literal(chars[i..=i + 3].iter().collect()));
                            code_tokens.push_str("b''");
                            i += 4;
                            continue;
                        } else if is_esc_byte_char {
                            let mut j = i + 3;
                            let mut found = false;
                            while j < chars.len() && j <= i + 6 {
                                if chars[j] == '\'' && chars[j - 1] != '\\'
                                    || (j > i + 3
                                        && chars[j] == '\''
                                        && chars[j - 2] == '\\'
                                        && chars[j - 1] == '\\')
                                {
                                    found = true;
                                    break;
                                }
                                j += 1;
                            }
                            if found {
                                if !cur_code.is_empty() {
                                    segments
                                        .push(VisibleSegment::Code(std::mem::take(&mut cur_code)));
                                }
                                segments
                                    .push(VisibleSegment::Literal(chars[i..=j].iter().collect()));
                                code_tokens.push_str("b''");
                                i = j + 1;
                                continue;
                            }
                        }
                    }

                    // 8. Char literal '...'
                    if chars[i] == '\'' {
                        let is_simple_char =
                            i + 2 < chars.len() && chars[i + 1] != '\\' && chars[i + 2] == '\'';
                        let is_esc_char = i + 1 < chars.len() && chars[i + 1] == '\\';
                        if is_simple_char {
                            if !cur_code.is_empty() {
                                segments.push(VisibleSegment::Code(std::mem::take(&mut cur_code)));
                            }
                            segments
                                .push(VisibleSegment::Literal(chars[i..=i + 2].iter().collect()));
                            code_tokens.push_str("''");
                            i += 3;
                            continue;
                        } else if is_esc_char {
                            let mut j = i + 2;
                            let mut found = false;
                            while j < chars.len() && j <= i + 10 {
                                if (chars[j] == '\'' && chars[j - 1] != '\\')
                                    || (j > i + 2
                                        && chars[j] == '\''
                                        && chars[j - 2] == '\\'
                                        && chars[j - 1] == '\\')
                                {
                                    found = true;
                                    break;
                                }
                                j += 1;
                            }
                            if found {
                                if !cur_code.is_empty() {
                                    segments
                                        .push(VisibleSegment::Code(std::mem::take(&mut cur_code)));
                                }
                                segments
                                    .push(VisibleSegment::Literal(chars[i..=j].iter().collect()));
                                code_tokens.push_str("''");
                                i = j + 1;
                                continue;
                            }
                        }
                        cur_code.push('\'');
                        code_tokens.push('\'');
                        i += 1;
                        continue;
                    }

                    // 9. Composite operators: <<, <=, >=, ->, =>
                    if chars[i] == '!' && i + 1 < chars.len() && chars[i + 1] == '=' {
                        cur_code.push_str("!=");
                        code_tokens.push_str("!=");
                        self.pending_macro_bang = false;
                        i += 2;
                        continue;
                    }
                    if chars[i] == '<'
                        && i + 2 < chars.len()
                        && chars[i + 1] == '<'
                        && chars[i + 2] == '='
                    {
                        cur_code.push_str("<<=");
                        code_tokens.push_str("<<=");
                        self.pending_macro_bang = false;
                        i += 3;
                        continue;
                    }
                    if chars[i] == '>'
                        && i + 2 < chars.len()
                        && chars[i + 1] == '>'
                        && chars[i + 2] == '='
                    {
                        cur_code.push_str(">>=");
                        code_tokens.push_str(">>=");
                        self.pending_macro_bang = false;
                        i += 3;
                        continue;
                    }
                    if chars[i] == '<' && i + 1 < chars.len() && chars[i + 1] == '<' {
                        let is_shift = ((!self.const_brace_stack.is_empty()
                            || self.expr_initializer_active)
                            && !code_tokens.trim_end().ends_with("::"))
                            || is_likely_comparison_less_than(&code_tokens);
                        if !is_shift {
                            self.angle_depth += 2;
                        }
                        cur_code.push_str("<<");
                        code_tokens.push_str("<<");
                        self.pending_macro_bang = false;
                        i += 2;
                        continue;
                    }
                    if chars[i] == '<' && i + 1 < chars.len() && chars[i + 1] == '=' {
                        cur_code.push_str("<=");
                        code_tokens.push_str("<=");
                        self.pending_macro_bang = false;
                        i += 2;
                        continue;
                    }
                    if chars[i] == '>' && i + 1 < chars.len() && chars[i + 1] == '=' {
                        cur_code.push_str(">=");
                        code_tokens.push_str(">=");
                        self.pending_macro_bang = false;
                        i += 2;
                        continue;
                    }
                    if chars[i] == '-' && i + 1 < chars.len() && chars[i + 1] == '>' {
                        cur_code.push_str("->");
                        code_tokens.push_str("->");
                        self.pending_macro_bang = false;
                        i += 2;
                        continue;
                    }
                    if chars[i] == '=' && i + 1 < chars.len() && chars[i + 1] == '>' {
                        cur_code.push_str("=>");
                        code_tokens.push_str("=>");
                        self.pending_macro_bang = false;
                        i += 2;
                        continue;
                    }

                    // 10. Structural tokens & depth tracking
                    match chars[i] {
                        '!' => {
                            self.pending_macro_bang = is_macro_bang(&code_tokens);
                            cur_code.push('!');
                            code_tokens.push('!');
                        }
                        '<' => {
                            self.pending_macro_bang = false;
                            // `for<'a>` (a higher-ranked trait bound) never
                            // has whitespace forcing the ` ` | `\t` | ... arm
                            // below to run between `for` and `<` - finalize
                            // here too, against the CURRENT (pre-increment)
                            // angle_depth: written in an impl header's own
                            // leading generic-bound list, this `for` sits
                            // inside the header's OUTER `impl<...>` frame, so
                            // angle_depth > 0 here and finalization is a
                            // no-op for it (the depth guard inside `finalize_
                            // header_keywords` never passes).
                            self.finalize_header_keywords(&code_tokens);
                            let in_const_expr =
                                !self.const_brace_stack.is_empty() || self.expr_initializer_active;
                            let is_comparison = if in_const_expr {
                                let t = code_tokens.trim_end();
                                !t.ends_with("::")
                            } else {
                                is_likely_comparison_less_than(&code_tokens)
                            };
                            if self.paren_depth == 0 && self.bracket_depth == 0 && !is_comparison {
                                self.angle_depth += 1;
                            }
                            cur_code.push('<');
                            code_tokens.push('<');
                        }
                        '>' => {
                            self.pending_macro_bang = false;
                            let angle_floor = self
                                .const_brace_stack
                                .last()
                                .map_or(0, |&(_, angle_depth)| angle_depth);
                            if self.paren_depth == 0
                                && self.bracket_depth == 0
                                && self.angle_depth > angle_floor
                            {
                                self.angle_depth -= 1;
                            }
                            cur_code.push('>');
                            code_tokens.push('>');
                        }
                        '(' => {
                            self.pending_macro_bang = false;
                            if self.paren_depth == 0
                                && self.angle_depth == 0
                                && self.bracket_depth == 0
                                && self.brace_depth == 0
                            {
                                has_top_level_open_paren = true;
                                top_level_open_paren_segs.push((segments.len(), cur_code.len()));
                            }
                            self.paren_depth += 1;
                            cur_code.push('(');
                            code_tokens.push('(');
                        }
                        ')' => {
                            self.pending_macro_bang = false;
                            if self.paren_depth == 1
                                && self.angle_depth == 0
                                && self.bracket_depth == 0
                                && self.brace_depth == 0
                            {
                                top_level_close_paren_segs.push((segments.len(), cur_code.len()));
                            }
                            if self.paren_depth > 0 {
                                self.paren_depth -= 1;
                            }
                            cur_code.push(')');
                            code_tokens.push(')');
                        }
                        '[' => {
                            self.pending_macro_bang = false;
                            self.bracket_depth += 1;
                            cur_code.push('[');
                            code_tokens.push('[');
                        }
                        ']' => {
                            self.pending_macro_bang = false;
                            if self.bracket_depth == 1 && first_outer_attribute_close_seg.is_none()
                            {
                                first_outer_attribute_close_seg =
                                    Some((segments.len(), cur_code.len()));
                            }
                            if self.bracket_depth > 0 {
                                self.bracket_depth -= 1;
                            }
                            cur_code.push(']');
                            code_tokens.push(']');
                        }
                        '{' => {
                            let is_macro_brace = self.pending_macro_bang;
                            self.pending_macro_bang = false;
                            if is_macro_brace {
                                self.macro_brace_stack.push(self.brace_depth);
                            } else {
                                if self.angle_depth > 0 || !self.const_brace_stack.is_empty() {
                                    self.const_brace_stack
                                        .push((self.brace_depth, self.angle_depth));
                                }
                                let is_top_level = self.paren_depth == 0
                                    && self.angle_depth == 0
                                    && self.bracket_depth == 0
                                    && self.brace_depth == 0;
                                if is_top_level {
                                    if !has_top_level_open_brace {
                                        has_top_level_open_brace = true;
                                        first_top_level_open_brace_seg =
                                            Some((segments.len(), cur_code.len()));
                                    }
                                    events.push(StructuralEvent {
                                        seg_idx: segments.len(),
                                        char_idx: cur_code.len(),
                                        kind: StructuralEventKind::TopLevelOpenBrace,
                                    });
                                } else if self.paren_depth == 0
                                    && self.angle_depth == 0
                                    && self.bracket_depth == 0
                                    && self.brace_depth == 1
                                {
                                    events.push(StructuralEvent {
                                        seg_idx: segments.len(),
                                        char_idx: cur_code.len(),
                                        kind: StructuralEventKind::BaselineOpenBrace,
                                    });
                                }
                            }
                            self.brace_depth += 1;
                            cur_code.push('{');
                            code_tokens.push('{');
                        }
                        '}' => {
                            self.pending_macro_bang = false;
                            let is_macro_close = self
                                .macro_brace_stack
                                .last()
                                .is_some_and(|&d| d == self.brace_depth.saturating_sub(1));
                            if is_macro_close {
                                self.macro_brace_stack.pop();
                            } else {
                                if self
                                    .const_brace_stack
                                    .last()
                                    .is_some_and(|&(brace_depth, _)| {
                                        brace_depth == self.brace_depth.saturating_sub(1)
                                    })
                                {
                                    self.const_brace_stack.pop();
                                }
                                if self.paren_depth == 0
                                    && self.angle_depth == 0
                                    && self.bracket_depth == 0
                                {
                                    if self.brace_depth == 1 {
                                        events.push(StructuralEvent {
                                            seg_idx: segments.len(),
                                            char_idx: cur_code.len(),
                                            kind: StructuralEventKind::TopLevelCloseBrace,
                                        });
                                        self.expr_initializer_active = false;
                                        self.pending_type_alias = false;
                                    } else if self.brace_depth == 2 {
                                        events.push(StructuralEvent {
                                            seg_idx: segments.len(),
                                            char_idx: cur_code.len(),
                                            kind: StructuralEventKind::MethodBodyClose,
                                        });
                                    }
                                }
                            }
                            if self.brace_depth > 0 {
                                self.brace_depth -= 1;
                            }
                            cur_code.push('}');
                            code_tokens.push('}');
                        }
                        ';' => {
                            self.pending_macro_bang = false;
                            if self.paren_depth == 0
                                && self.bracket_depth == 0
                                && self.brace_depth == 0
                            {
                                has_top_level_semicolon = true;
                                self.angle_depth = 0;
                                self.expr_initializer_active = false;
                                self.pending_type_alias = false;
                                events.push(StructuralEvent {
                                    seg_idx: segments.len(),
                                    char_idx: cur_code.len(),
                                    kind: StructuralEventKind::TopLevelSemicolon,
                                });
                            } else if self.paren_depth == 0
                                && self.bracket_depth == 0
                                && self.brace_depth == 1
                                && self.macro_brace_stack.is_empty()
                            {
                                self.expr_initializer_active = false;
                                self.pending_type_alias = false;
                                events.push(StructuralEvent {
                                    seg_idx: segments.len(),
                                    char_idx: cur_code.len(),
                                    kind: StructuralEventKind::BaselineSemicolon,
                                });
                            }
                            cur_code.push(';');
                            code_tokens.push(';');
                        }
                        ',' => {
                            self.pending_macro_bang = false;
                            if self.paren_depth == 0
                                && self.angle_depth == 0
                                && self.bracket_depth == 0
                                && self.brace_depth == 0
                            {
                                self.expr_initializer_active = false;
                                self.pending_type_alias = false;
                                events.push(StructuralEvent {
                                    seg_idx: segments.len(),
                                    char_idx: cur_code.len(),
                                    kind: StructuralEventKind::TopLevelComma,
                                });
                            } else if self.paren_depth == 0
                                && self.angle_depth == 0
                                && self.bracket_depth == 0
                                && self.brace_depth == 1
                                && self.macro_brace_stack.is_empty()
                            {
                                self.expr_initializer_active = false;
                                self.pending_type_alias = false;
                                events.push(StructuralEvent {
                                    seg_idx: segments.len(),
                                    char_idx: cur_code.len(),
                                    kind: StructuralEventKind::BaselineComma,
                                });
                            }
                            cur_code.push(',');
                            code_tokens.push(',');
                        }
                        '=' => {
                            self.pending_macro_bang = false;
                            if self.paren_depth == 0
                                && self.bracket_depth == 0
                                && self.angle_depth == 0
                                && (self.brace_depth == 0 || self.brace_depth == 1)
                                && !self.pending_type_alias
                            {
                                self.expr_initializer_active = true;
                            }
                            cur_code.push('=');
                            code_tokens.push('=');
                        }
                        ' ' | '\t' | '\r' | '\n' => {
                            if self.paren_depth == 0
                                && self.bracket_depth == 0
                                && self.angle_depth == 0
                                && (self.brace_depth == 0 || self.brace_depth == 1)
                                && ends_with_keyword(&code_tokens, "type")
                            {
                                self.pending_type_alias = true;
                            }
                            self.finalize_header_keywords(&code_tokens);
                            cur_code.push(chars[i]);
                            code_tokens.push(chars[i]);
                        }
                        c => {
                            self.pending_macro_bang = false;
                            cur_code.push(c);
                            code_tokens.push(c);
                        }
                    }
                    i += 1;
                }
                LexState::BlockComment(depth) => {
                    if chars[i] == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
                        self.state = LexState::BlockComment(depth + 1);
                        i += 2;
                    } else if chars[i] == '*' && i + 1 < chars.len() && chars[i + 1] == '/' {
                        if depth <= 1 {
                            self.state = LexState::Normal;
                            code_tokens.push(' ');
                        } else {
                            self.state = LexState::BlockComment(depth - 1);
                        }
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                LexState::NormalStringContinuation | LexState::ByteNormalStringContinuation => {
                    let is_byte = matches!(self.state, LexState::ByteNormalStringContinuation);
                    while i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
                        i += 1;
                    }
                    // Rust's escaped-newline continuation does not end at
                    // the next non-whitespace CHARACTER on the SAME
                    // physical line - it ends at the next non-whitespace
                    // SOURCE character, however many physical lines that
                    // takes (an entirely whitespace-only line in between
                    // contributes nothing to the literal, not even a
                    // newline). Only transition back to the literal state
                    // once a real character was actually found on this
                    // line; otherwise remain in the Continuation state so
                    // the NEXT physical line's own leading whitespace is
                    // also skipped as part of the SAME continuation, and
                    // so the caller's `was_escaped` check (which governs
                    // whether a synthetic newline is inserted between
                    // physical lines) stays true across every physical
                    // line the continuation actually spans, not just the
                    // first one.
                    if i < chars.len() {
                        self.state = if is_byte {
                            LexState::ByteNormalString
                        } else {
                            LexState::NormalString
                        };
                    }
                }
                LexState::NormalString | LexState::ByteNormalString => {
                    let is_byte = matches!(self.state, LexState::ByteNormalString);
                    let mut lit_str = String::new();
                    while i < chars.len() {
                        if chars[i] == '\\' {
                            if i + 1 < chars.len() {
                                lit_str.push(chars[i]);
                                lit_str.push(chars[i + 1]);
                                i += 2;
                            } else {
                                self.state = if is_byte {
                                    LexState::ByteNormalStringContinuation
                                } else {
                                    LexState::NormalStringContinuation
                                };
                                i += 1;
                                break;
                            }
                        } else if chars[i] == '"' {
                            lit_str.push('"');
                            self.state = LexState::Normal;
                            i += 1;
                            break;
                        } else {
                            lit_str.push(chars[i]);
                            i += 1;
                        }
                    }
                    if !lit_str.is_empty() {
                        if let Some(VisibleSegment::Literal(prev)) = segments.last_mut() {
                            prev.push_str(&lit_str);
                        } else {
                            segments.push(VisibleSegment::Literal(lit_str));
                        }
                    }
                }
                LexState::RawString(hashes) | LexState::RawByteString(hashes) => {
                    let mut lit_str = String::new();
                    while i < chars.len() {
                        if chars[i] == '"' {
                            let mut j = i + 1;
                            let mut match_hashes = 0;
                            while j < chars.len() && match_hashes < hashes && chars[j] == '#' {
                                match_hashes += 1;
                                j += 1;
                            }
                            if match_hashes == hashes {
                                lit_str.push('"');
                                for _ in 0..hashes {
                                    lit_str.push('#');
                                }
                                self.state = LexState::Normal;
                                i = j;
                                break;
                            }
                        }
                        lit_str.push(chars[i]);
                        i += 1;
                    }
                    if let Some(VisibleSegment::Literal(prev)) = segments.last_mut() {
                        prev.push_str(&lit_str);
                    } else {
                        segments.push(VisibleSegment::Literal(lit_str));
                    }
                }
            }
        }

        // `where`/`for` reaching the very end of a physical line with
        // nothing after them (a common rustfmt-style line wrap: `where`
        // alone on its own line, or the type after `for` continuing on
        // the next) never hits a trailing boundary character on THIS
        // line to trigger finalization via any of the other call sites.
        // Finalize once more here, against the fully-built `code_tokens`
        // for this call, so the keyword is still recognized when it is
        // the last thing scanned.
        self.finalize_header_keywords(&code_tokens);

        if !cur_code.is_empty() {
            segments.push(VisibleSegment::Code(cur_code));
        }

        ScannedLine {
            segments,
            code_tokens,
            events,
            has_top_level_semicolon,
            has_top_level_open_paren,
            top_level_open_paren_segs,
            top_level_close_paren_segs,
            has_top_level_open_brace,
            first_top_level_open_brace_seg,
            first_outer_attribute_close_seg,
            ends_in_string_literal: self.state.is_in_string_literal(),
            ends_in_escaped_continuation: self.state.is_escaped_continuation(),
        }
    }
}

fn parse_public_item(code_tokens: &str) -> Option<(&str, &str)> {
    let t = code_tokens.trim_start();
    if !t.starts_with("pub") {
        return None;
    }
    let after_pub = t[3..].trim_start();
    if after_pub.starts_with('(') {
        let mut depth = 0;
        let mut close_idx = None;
        for (i, c) in after_pub.char_indices() {
            if c == '(' {
                depth += 1;
            } else if c == ')' {
                depth -= 1;
                if depth == 0 {
                    close_idx = Some(i);
                    break;
                }
            }
        }
        let close_pos = close_idx?;
        let rest = after_pub[close_pos + 1..].trim_start();
        Some((t, rest))
    } else if t.starts_with("pub ") || t == "pub" {
        Some((t, after_pub))
    } else {
        None
    }
}

fn is_keyword_prefix<'a>(s: &'a str, kw: &str) -> Option<&'a str> {
    if let Some(rest) = s.strip_prefix(kw) {
        if rest.is_empty()
            || rest.starts_with(char::is_whitespace)
            || rest.starts_with('<')
            || rest.starts_with('(')
            || rest.starts_with('{')
            || rest.starts_with(':')
            || rest.starts_with(';')
        {
            return Some(rest.trim_start());
        }
    }
    None
}

fn is_outer_attribute_start(code_tokens: &str) -> bool {
    let t = code_tokens.trim_start();
    if let Some(rest) = t.strip_prefix('#') {
        let trimmed = rest.trim_start();
        trimmed.starts_with('[') || trimmed.is_empty()
    } else {
        false
    }
}

fn is_public_code(code_tokens: &str) -> bool {
    parse_public_item(code_tokens).is_some()
}

fn is_public_fn(code_tokens: &str) -> bool {
    if let Some((_, rest)) = parse_public_item(code_tokens) {
        let mut cur = rest.trim_start();
        loop {
            if let Some(after) = is_keyword_prefix(cur, "const") {
                cur = after;
                continue;
            }
            if let Some(after) = is_keyword_prefix(cur, "async") {
                cur = after;
                continue;
            }
            if let Some(after) = is_keyword_prefix(cur, "unsafe") {
                cur = after;
                continue;
            }
            if let Some(after) = is_keyword_prefix(cur, "extern") {
                let mut rem = after;
                if rem.starts_with("\"\"") {
                    rem = rem[2..].trim_start();
                } else if rem.starts_with('"') {
                    if let Some(close) = rem[1..].find('"') {
                        rem = rem[close + 2..].trim_start();
                    }
                }
                cur = rem;
                continue;
            }
            break;
        }
        is_keyword_prefix(cur, "fn").is_some()
    } else {
        false
    }
}

fn is_public_enum(code_tokens: &str) -> bool {
    if let Some((_, rest)) = parse_public_item(code_tokens) {
        is_keyword_prefix(rest, "enum").is_some()
    } else {
        false
    }
}

fn is_public_struct(code_tokens: &str) -> bool {
    parse_public_item(code_tokens)
        .is_some_and(|(_, rest)| is_keyword_prefix(rest, "struct").is_some())
}

fn has_public_tuple_struct_body_open(code_tokens: &str) -> bool {
    let Some((_, rest)) = parse_public_item(code_tokens) else {
        return false;
    };
    let Some(after_struct) = is_keyword_prefix(rest, "struct") else {
        return false;
    };
    CodeLexer::new()
        .scan_line(after_struct)
        .has_top_level_open_paren
}

fn is_public_union(code_tokens: &str) -> bool {
    parse_public_item(code_tokens)
        .is_some_and(|(_, rest)| is_keyword_prefix(rest, "union").is_some())
}

fn is_public_trait(code_tokens: &str) -> bool {
    parse_public_item(code_tokens).is_some_and(|(_, rest)| {
        let mut cur = rest.trim_start();
        if let Some(after) = is_keyword_prefix(cur, "unsafe") {
            cur = after;
        }
        is_keyword_prefix(cur, "trait").is_some()
    })
}

/// True when `code_tokens` starts an `impl` item (optionally `unsafe
/// impl`) - trait impl or inherent impl alike; `impl` blocks are never
/// visibility-qualified, so unlike every other `is_public_X` check this
/// does not go through `parse_public_item` at all.
fn is_impl_start(code_tokens: &str) -> bool {
    let mut cur = code_tokens.trim_start();
    if let Some(after) = is_keyword_prefix(cur, "unsafe") {
        cur = after;
    }
    is_keyword_prefix(cur, "impl").is_some()
}

/// True when `code_tokens` starts an `extern` BLOCK (optionally `unsafe
/// extern`, optionally with an ABI string) - `extern "C" { ... }`, never
/// a single foreign-function declaration (`extern "C" fn f();`, handled
/// by `is_public_fn`) or `extern crate ...;` (handled by
/// `is_public_extern_crate`), both of which lack the block's own `{`
/// immediately after the optional ABI string.
fn is_extern_block_start(code_tokens: &str) -> bool {
    let mut cur = code_tokens.trim_start();
    if let Some(after) = is_keyword_prefix(cur, "unsafe") {
        cur = after;
    }
    let Some(mut rest) = is_keyword_prefix(cur, "extern") else {
        return false;
    };
    rest = rest.trim_start();
    if rest.starts_with('"') {
        match rest[1..].find('"') {
            Some(close) => rest = rest[close + 2..].trim_start(),
            None => return false,
        }
    }
    rest.starts_with('{')
}

/// True when `code_tokens` is `extern`/`unsafe extern`, optionally
/// followed by a complete ABI string, with NOTHING ELSE after it yet -
/// not enough has been read to tell whether this is going to be an
/// extern BLOCK (`extern "C" { ... }`), a single foreign-function
/// declaration (`extern "C" fn f();`), or `extern crate foo;`. Mirrors
/// `is_incomplete_public_prefix`'s role for a partial `pub(...)`
/// visibility: it never classifies the construct itself, it only says
/// "read another physical line before deciding" - the outer dispatch
/// gate's own continuation loop keeps pulling lines while this is true,
/// so `is_extern_block_start`'s `{`-immediately-follows check (and
/// `is_public_fn`/`is_public_extern_crate` for the other two shapes)
/// always sees the deciding token even when a legally line-wrapped
/// header (the ABI string, or the whole header, on its own physical
/// line) puts it on a later line than the header started on.
fn is_incomplete_extern_prefix(code_tokens: &str) -> bool {
    let mut cur = code_tokens.trim_start();
    if let Some(after) = is_keyword_prefix(cur, "unsafe") {
        cur = after;
    }
    let Some(mut rest) = is_keyword_prefix(cur, "extern") else {
        return false;
    };
    rest = rest.trim_start();
    if rest.is_empty() {
        return true;
    }
    if rest.starts_with('"') {
        match rest[1..].find('"') {
            Some(close) => rest = rest[close + 2..].trim_start(),
            None => return true,
        }
    } else {
        return false;
    }
    rest.is_empty()
}

/// True when `code_tokens` is now a COMPLETE, private (non-`pub`) extern
/// declaration - `extern crate foo;` or `extern "C" fn f() { ... }` /
/// `extern "C" fn f();` - once `is_incomplete_extern_prefix`'s own
/// lookahead has read enough to rule out an extern BLOCK. Recognizing a
/// candidate is not the same as classifying it as public: this is exactly
/// that separation - candidate admission only ever means "keep reading,"
/// and once a NON-public extern form resolves, it must fall back to the
/// ordinary private-item path (consumed structurally, never inventoried)
/// rather than reach the public dispatcher's unsupported-item assertion.
/// A `pub`-prefixed extern crate/fn never reaches this: `is_incomplete_
/// extern_prefix` only ever admits a bare `extern`/`unsafe extern`
/// prefix, so a leading `pub` is already routed through the ordinary
/// public-item path instead.
fn is_private_extern_item(code_tokens: &str) -> bool {
    let mut cur = code_tokens.trim_start();
    if let Some(after) = is_keyword_prefix(cur, "unsafe") {
        cur = after;
    }
    let Some(mut rest) = is_keyword_prefix(cur, "extern") else {
        return false;
    };
    rest = rest.trim_start();
    if rest.starts_with('"') {
        match rest[1..].find('"') {
            Some(close) => rest = rest[close + 2..].trim_start(),
            None => return false,
        }
    }
    is_keyword_prefix(rest, "crate").is_some() || is_keyword_prefix(rest, "fn").is_some()
}

/// True when a FULLY-READ `impl` header (from `impl` up to, but not
/// including, its own body-opening `{`) declares an INHERENT impl
/// (`impl<T> S<T> { ... }`) rather than a trait impl (`impl<T> Trait<T>
/// for S<T> { ... }`). `header_saw_trait_for` is read live off the
/// `CodeLexer`'s own real-time depth-tracked state (see its doc comment)
/// while the header is scanned, not searched for afterward in already-
/// rendered text - a higher-ranked trait bound's `for<'a>`, wherever it
/// is legally written in the header, can never set it.
fn is_inherent_impl_header(header_saw_trait_for: bool) -> bool {
    !header_saw_trait_for
}

fn is_public_use(code_tokens: &str) -> bool {
    if let Some((_, rest)) = parse_public_item(code_tokens) {
        is_keyword_prefix(rest, "use").is_some()
    } else {
        false
    }
}

fn is_public_const_or_static(code_tokens: &str) -> bool {
    if let Some((_, rest)) = parse_public_item(code_tokens) {
        (is_keyword_prefix(rest, "const").is_some() || is_keyword_prefix(rest, "static").is_some())
            && !is_public_fn(code_tokens)
    } else {
        false
    }
}

fn is_public_type_alias(code_tokens: &str) -> bool {
    parse_public_item(code_tokens)
        .is_some_and(|(_, rest)| is_keyword_prefix(rest, "type").is_some())
}

fn is_public_mod(code_tokens: &str) -> bool {
    parse_public_item(code_tokens).is_some_and(|(_, rest)| is_keyword_prefix(rest, "mod").is_some())
}

fn is_public_extern_crate(code_tokens: &str) -> bool {
    parse_public_item(code_tokens).is_some_and(|(_, rest)| {
        is_keyword_prefix(rest, "extern")
            .is_some_and(|after_extern| is_keyword_prefix(after_extern, "crate").is_some())
    })
}

fn is_public_qualifiers_only(code_tokens: &str) -> bool {
    let Some((_, rest)) = parse_public_item(code_tokens) else {
        return false;
    };
    let mut cur = rest.trim_start();
    loop {
        if let Some(after) = is_keyword_prefix(cur, "const") {
            cur = after;
            continue;
        }
        if let Some(after) = is_keyword_prefix(cur, "async") {
            cur = after;
            continue;
        }
        if let Some(after) = is_keyword_prefix(cur, "unsafe") {
            cur = after;
            continue;
        }
        if let Some(after) = is_keyword_prefix(cur, "extern") {
            let mut rem = after;
            if rem.starts_with("\"\"") {
                rem = rem[2..].trim_start();
            } else if rem.starts_with('"') {
                if let Some(close) = rem[1..].find('"') {
                    rem = rem[close + 2..].trim_start();
                } else {
                    rem = "";
                }
            }
            cur = rem;
            continue;
        }
        break;
    }
    cur.trim().is_empty()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraitMemberKind {
    Method,
    AssociatedConst,
    AssociatedType,
    MacroInvocation,
    Other,
}

fn classify_trait_member(code_tokens: &str) -> TraitMemberKind {
    let mut cur = code_tokens.trim_start();
    if let Some((_, rest)) = parse_public_item(cur) {
        cur = rest.trim_start();
    }

    let mut is_const = false;
    let mut is_async = false;
    let mut is_unsafe = false;
    let mut is_extern = false;

    loop {
        if let Some(after) = is_keyword_prefix(cur, "const") {
            is_const = true;
            cur = after;
            continue;
        }
        if let Some(after) = is_keyword_prefix(cur, "async") {
            is_async = true;
            cur = after;
            continue;
        }
        if let Some(after) = is_keyword_prefix(cur, "unsafe") {
            is_unsafe = true;
            cur = after;
            continue;
        }
        if let Some(after) = is_keyword_prefix(cur, "extern") {
            is_extern = true;
            let mut rem = after;
            if rem.starts_with("\"\"") {
                rem = rem[2..].trim_start();
            } else if rem.starts_with('"') {
                if let Some(close) = rem[1..].find('"') {
                    rem = rem[close + 2..].trim_start();
                } else {
                    rem = "";
                }
            }
            cur = rem;
            continue;
        }
        break;
    }

    if is_keyword_prefix(cur, "fn").is_some() {
        TraitMemberKind::Method
    } else if is_const && !is_async && !is_unsafe && !is_extern {
        TraitMemberKind::AssociatedConst
    } else if is_keyword_prefix(cur, "type").is_some() {
        TraitMemberKind::AssociatedType
    } else if cur.contains('!') {
        TraitMemberKind::MacroInvocation
    } else {
        TraitMemberKind::Other
    }
}

/// Scans one braced item body (trait or inherent impl) for baseline
/// members - the SAME authoritative scanner both use, not a separate
/// parser per item kind. Reads the header up to the body's own `{`
/// (spanning further physical lines if needed), pushes it, then walks
/// the body member-by-member using the same `BaselineSemicolon`/
/// `BaselineComma`/`BaselineOpenBrace`/`TopLevelCloseBrace` structural
/// events every other collector in this file uses: a member ending in
/// `;` (an associated const/type) or opening a method body (skipped via
/// `MethodBodyClose`, never recursed into, exactly like a trait default
/// method) is captured with `emit_captured_fragment` - the same literal-
/// aware, no-blind-`normalize_ws` emitter every other capture path in
/// this file already uses.
///
/// `should_emit(member_code_tokens)` decides whether a fully-captured
/// member is actually pushed to the golden surface: trait members are
/// unconditionally public (`|_| true`), so this is the ONLY difference
/// from the trait path - inherent impl members pass a per-member `pub`-
/// visibility check instead. The classification that decides whether a
/// member is a METHOD (and must therefore have its body skipped,
/// regardless of whether it will ultimately be emitted) is untouched by
/// this predicate - a private inherent method's body is still never
/// recursed into, it is simply not pushed to `lines`.
/// A member boundary (right after a previous member's own terminator, or
/// the very start of a fresh line) may be immediately followed by another
/// member's attribute on that SAME physical line/remainder - e.g. `pub
/// const N: u32 = 1; #[deprecated] pub fn f() {}`. The only place that
/// knows how to peel an attribute off cleanly is the outer while loop's
/// own `capture_attribute` path, so a boundary found structurally mid-`sc`
/// (not just at the top of the while loop) must hand the rest of this
/// scanned line back to that SAME path rather than accumulating the
/// attribute's own text as if it were part of the next member's code -
/// this is the same peek `is_outer_attribute_start` already used at the
/// top of the while loop, just evaluated at `cur_pos` instead of position
/// (0, 0).
///
/// `sc` was produced by scanning its ENTIRE source string in one
/// `scan_line` call, so `item_lexer`'s own persistent state, by the time
/// this is called, reflects having walked all the way through `sc` - past
/// `cur_pos`, into whatever the split-off remainder itself contains. If a
/// split is returned, `item_lexer` is reset to `lexer_before_line` (the
/// checkpoint captured before `sc` was scanned) and replayed forward only
/// through the CONSUMED prefix (`sc`'s text from `region_start` - where
/// THIS member-processing region actually began scanning `sc` from,
/// which is NOT always `sc`'s own absolute start: when `sc` is the whole
/// opening physical line reused directly, `region_start` is the position
/// right after the construct's own `{`, not position (0, 0), since text
/// before that belongs to the construct's own header, never scanned by
/// `item_lexer` as body content at all - up to `cur_pos`), so its state
/// afterward represents exactly "read up to the boundary, nothing more" -
/// the same save-checkpoint/rescan-only-the-consumed-portion pattern
/// `capture_attribute` already uses for its own attribute boundary.
fn member_boundary_attribute(
    sc: &ScannedLine,
    region_start: Option<(usize, usize)>,
    cur_pos: Option<(usize, usize)>,
    item_lexer: &mut CodeLexer,
    lexer_before_line: &CodeLexer,
) -> Option<String> {
    let peek_code = sc.code_tokens_range(cur_pos, None);
    if !is_outer_attribute_start(&peek_code) {
        return None;
    }
    let mut remainder_text = sc.render_visible_range(cur_pos, None);
    if remainder_text.trim().is_empty() {
        return None;
    }
    // `sc.ends_in_escaped_continuation` covers the end of this WHOLE
    // scanned line, which is exactly the same endpoint this remainder
    // (from `cur_pos` to the end) also ends at. `render_visible_range`
    // never renders the trailing `\` a real escaped continuation ends
    // in, so it is manually restored here - this remainder is handed
    // back as plain text for a LATER, independent `scan_line` call
    // (via `capture_attribute`, then the main member loop) to replay,
    // and without it that later scan would see an ordinary unclosed
    // string instead, silently losing the continuation onto the next
    // physical line.
    if sc.ends_in_escaped_continuation {
        remainder_text.push('\\');
    }
    let consumed_prefix = sc.render_visible_range(region_start, cur_pos);
    *item_lexer = lexer_before_line.clone();
    item_lexer.scan_line(&consumed_prefix);
    Some(remainder_text)
}

#[allow(clippy::too_many_arguments)]
fn scan_baseline_member_body(
    lines: &mut Vec<String>,
    src_lines: &[&str],
    idx: &mut usize,
    item_lexer: &mut CodeLexer,
    same_line_remainder: &mut Option<String>,
    current_scanned: &ScannedLine,
    header_prefix: String,
    push_header: bool,
    should_emit: impl Fn(&str) -> bool,
) {
    let mut header = if current_scanned.has_top_level_open_brace {
        let up_to_brace = current_scanned.text_up_to_function_body_open_brace();
        let rendered_last = current_scanned.render_visible();
        if header_prefix.ends_with(&rendered_last) {
            let prefix_head = &header_prefix[..header_prefix.len() - rendered_last.len()];
            format!("{prefix_head}{up_to_brace}")
        } else {
            up_to_brace
        }
    } else {
        header_prefix
    };

    let mut body_open = current_scanned.has_top_level_open_brace;

    while !body_open && *idx + 1 < src_lines.len() {
        *idx += 1;
        let continuation = src_lines[*idx];
        if continuation.trim().is_empty() && item_lexer.state == LexState::Normal {
            continue;
        }
        let was_escaped = item_lexer.state.is_escaped_continuation();
        let continues_literal = item_lexer.state.is_in_string_literal();
        let sc = scan_logical_item_line(item_lexer, continuation, same_line_remainder);
        let rendered = sc.render_visible();
        body_open = sc.has_top_level_open_brace;
        if was_escaped {
            header.push_str(&rendered);
            continue;
        }
        if continues_literal {
            header.push('\n');
            header.push_str(&rendered);
            continue;
        }
        if rendered.trim().is_empty() {
            continue;
        }
        if !header.ends_with(' ') && !rendered.starts_with(' ') {
            header.push(' ');
        }
        header.push_str(&rendered);
    }

    if push_header {
        lines.push(header.clone());
    }

    let mut pending_member_attrs = Vec::new();
    let mut cur_member_text = String::new();
    let mut cur_member_code = String::new();
    let mut in_default_method_body = false;
    let mut body_closed = false;

    let emit = |lines: &mut Vec<String>,
                pending_attrs: &mut Vec<String>,
                cur_member_text: &mut String,
                cur_member_code: &mut String| {
        let trimmed_text = emit_captured_fragment(cur_member_text).unwrap_or_default();
        if !trimmed_text.is_empty() && should_emit(cur_member_code) {
            lines.append(pending_attrs);
            lines.push(trimmed_text);
        } else {
            pending_attrs.clear();
        }
        cur_member_text.clear();
        cur_member_code.clear();
    };

    // The opening physical line's own remainder (everything after the
    // body's `{`) is not processed with a second, attribute-unaware event
    // loop here - it is handed to the SAME main member-processing loop
    // below as its first `body_remainder`, so a leading attribute right
    // after the brace (`impl S { #[deprecated] pub fn f() {...} }`) goes
    // through the exact same `capture_attribute`/`pending_member_attrs`
    // path a later physical line already uses.
    let open_brace_next_pos = current_scanned
        .first_top_level_open_brace_seg
        .and_then(|open_brace_pos| next_pos(&current_scanned.segments, open_brace_pos));
    let opening_line_remainder = open_brace_next_pos
        .map(|start_pos| current_scanned.render_visible_range(Some(start_pos), None))
        .filter(|text| !text.trim().is_empty());
    // Whether the NEXT `scan_logical_item_line` call over the opening
    // line's own remainder would be a REPLAY of text already scanned
    // once, for real, by whichever scan produced `current_scanned` -
    // cleared the moment that remainder is consumed (attribute-captured
    // or handed to the main event loop below), since every later
    // `body_remainder` (a later physical line, or text split off by
    // `member_boundary_attribute`/`capture_attribute`) is either
    // genuinely fresh source text or already replayed correctly by its
    // own save-checkpoint/rescan-only-the-consumed-portion logic.
    let mut pending_opening_line_replay = opening_line_remainder.is_some();
    let mut body_remainder = opening_line_remainder;
    while !body_closed && (body_remainder.is_some() || *idx + 1 < src_lines.len()) {
        let owned_line = body_remainder.take();
        if owned_line.is_none() {
            *idx += 1;
            pending_opening_line_replay = false;
        }
        let line_text = owned_line.as_deref().unwrap_or(src_lines[*idx]);
        if line_text.trim().is_empty() && item_lexer.state == LexState::Normal {
            pending_opening_line_replay = false;
            continue;
        }

        if in_default_method_body {
            let lexer_before_line = item_lexer.clone();
            let sc = scan_logical_item_line(item_lexer, line_text, same_line_remainder);
            let mut cur_pos = None;
            for event in &sc.events {
                if event.kind == StructuralEventKind::MethodBodyClose {
                    in_default_method_body = false;
                    cur_pos = next_pos(&sc.segments, (event.seg_idx, event.char_idx));
                    break;
                }
            }
            if in_default_method_body {
                if item_lexer.brace_depth == 1 && item_lexer.macro_brace_stack.is_empty() {
                    in_default_method_body = false;
                } else if item_lexer.brace_depth == 0 {
                    break;
                }
                continue;
            }
            if cur_pos.is_none() {
                continue;
            }
            for event in &sc.events {
                let Some(pos) = cur_pos else { break };
                if event.seg_idx < pos.0 || (event.seg_idx == pos.0 && event.char_idx < pos.1) {
                    continue;
                }

                if cur_member_text.trim().is_empty() {
                    if let Some(remainder) = member_boundary_attribute(
                        &sc,
                        Some((0, 0)),
                        cur_pos,
                        item_lexer,
                        &lexer_before_line,
                    ) {
                        body_remainder = Some(remainder);
                        cur_pos = None;
                        break;
                    }
                }

                if event.kind == StructuralEventKind::TopLevelCloseBrace {
                    let (chunk_text, chunk_code) = if let Some(limit) =
                        prev_pos(&sc.segments, (event.seg_idx, event.char_idx))
                    {
                        (
                            sc.render_visible_range(cur_pos, Some(limit)),
                            sc.code_tokens_range(cur_pos, Some(limit)),
                        )
                    } else {
                        (String::new(), String::new())
                    };
                    if !chunk_text.trim().is_empty() {
                        if !cur_member_text.is_empty()
                            && !cur_member_text.ends_with(' ')
                            && !chunk_text.starts_with(' ')
                        {
                            cur_member_text.push(' ');
                            cur_member_code.push(' ');
                        }
                        cur_member_text.push_str(&chunk_text);
                        cur_member_code.push_str(&chunk_code);
                    }
                    emit(
                        lines,
                        &mut pending_member_attrs,
                        &mut cur_member_text,
                        &mut cur_member_code,
                    );
                    body_closed = true;
                    cur_pos = next_pos(&sc.segments, (event.seg_idx, event.char_idx));
                    break;
                }

                let chunk_text =
                    sc.render_visible_range(cur_pos, Some((event.seg_idx, event.char_idx)));
                let chunk_code =
                    sc.code_tokens_range(cur_pos, Some((event.seg_idx, event.char_idx)));
                if !chunk_text.trim().is_empty() {
                    if !cur_member_text.is_empty()
                        && !cur_member_text.ends_with(' ')
                        && !chunk_text.starts_with(' ')
                    {
                        cur_member_text.push(' ');
                        cur_member_code.push(' ');
                    }
                    cur_member_text.push_str(&chunk_text);
                    cur_member_code.push_str(&chunk_code);
                }
                cur_pos = next_pos(&sc.segments, (event.seg_idx, event.char_idx));

                if event.kind == StructuralEventKind::BaselineSemicolon {
                    emit(
                        lines,
                        &mut pending_member_attrs,
                        &mut cur_member_text,
                        &mut cur_member_code,
                    );
                } else if event.kind == StructuralEventKind::BaselineOpenBrace {
                    let is_method =
                        classify_trait_member(&cur_member_code) == TraitMemberKind::Method;
                    if is_method {
                        emit(
                            lines,
                            &mut pending_member_attrs,
                            &mut cur_member_text,
                            &mut cur_member_code,
                        );
                        in_default_method_body = true;
                    }
                }
            }
            if !body_closed && !in_default_method_body && cur_pos.is_some() {
                // The for-loop above only checks for a boundary attribute
                // AT an event; when the split point (right after this
                // member's own terminator) is the LAST thing `sc` has any
                // event for - the rest of `sc` (an attribute, then its
                // member) never produces a further structural event of
                // its own within this SAME scanned line - the loop exits
                // with nothing left to trigger that check, and this
                // trailing remainder would otherwise blindly append the
                // attribute's own text as if it were part of the member.
                // Check once more here, exactly like the loop's own
                // per-event check.
                if cur_member_text.trim().is_empty() {
                    if let Some(remainder) = member_boundary_attribute(
                        &sc,
                        Some((0, 0)),
                        cur_pos,
                        item_lexer,
                        &lexer_before_line,
                    ) {
                        body_remainder = Some(remainder);
                        continue;
                    }
                }
                let remainder_text = sc.render_visible_range(cur_pos, None);
                let remainder_code = sc.code_tokens_range(cur_pos, None);
                if !remainder_text.trim().is_empty() {
                    if !cur_member_text.is_empty()
                        && !cur_member_text.ends_with(' ')
                        && !remainder_text.starts_with(' ')
                    {
                        cur_member_text.push(' ');
                        cur_member_code.push(' ');
                    }
                    cur_member_text.push_str(&remainder_text);
                    cur_member_code.push_str(&remainder_code);
                }
            }
            continue;
        }

        if cur_member_text.trim().is_empty() && item_lexer.state == LexState::Normal {
            let mut check_lexer = item_lexer.clone();
            let check_sc = check_lexer.scan_line(line_text);
            if is_outer_attribute_start(&check_sc.code_tokens) {
                if pending_opening_line_replay {
                    // `capture_attribute` does its own internal replay
                    // (reset to a checkpoint, rescan only the attribute
                    // text) starting from `item_lexer`'s CURRENT state -
                    // which, for the untouched opening-line remainder,
                    // is still whatever the scan that produced `current_
                    // scanned` left it at (potentially past the body's
                    // own close, per the same P2-1 concern as above).
                    // Unlike the general member-processing loop, an
                    // attribute has no literal-continuation content of
                    // its own to lose, so re-deriving `item_lexer` from a
                    // fresh scan of `header` (this construct's own
                    // declaration up to and including its `{`) here is
                    // safe and gives `capture_attribute` the correct
                    // depth to start from.
                    let mut header_lexer = CodeLexer::new();
                    header_lexer.scan_line(&header);
                    *item_lexer = header_lexer;
                }
                let captured = capture_attribute(src_lines, idx, item_lexer, line_text);
                pending_member_attrs.push(captured.text);
                pending_opening_line_replay = false;
                if !captured.remainder.trim().is_empty() {
                    body_remainder = Some(captured.remainder);
                }
                continue;
            }
        }

        // `was_escaped`/`continues_literal` mean "the PREVIOUS iteration
        // left `item_lexer` mid-literal, so this iteration's own content
        // continues it." For the untouched opening-line remainder, there
        // is no previous iteration within this member-capture loop to
        // continue FROM - `item_lexer`'s current state instead reflects
        // whatever the scan that produced `current_scanned` left it at
        // (e.g. mid-raw-string, if the member's own literal opens but
        // does not close on that physical line), which describes this
        // remainder's own END, not something preceding it. Both must
        // read false here regardless.
        let (was_escaped, continues_literal) = if pending_opening_line_replay {
            (false, false)
        } else {
            (
                item_lexer.state.is_escaped_continuation(),
                item_lexer.state.is_in_string_literal(),
            )
        };
        // `lexer_before_line` is the checkpoint `member_boundary_
        // attribute` resets `item_lexer` to before replaying a consumed
        // prefix - it must represent "nothing of this member-processing
        // region has been read yet." For every other iteration that is
        // simply `item_lexer`'s current value (a genuinely fresh line or
        // split, not yet touched). For the untouched opening-line
        // remainder specifically, `item_lexer`'s current value is NOT
        // that checkpoint - like `header_lexer` below, it still reflects
        // the WHOLE physical line already having been scanned once, for
        // real, by whichever scan produced `current_scanned` (so it can
        // be mid-string, over-advanced past this region's own close, or
        // both) - re-deriving fresh from `header` here, exactly like the
        // attribute-check branch above already does, is required so a
        // boundary attribute found within `current_scanned` (reused
        // directly as `sc` below) replays its own consumed prefix from
        // the correct depth and literal state instead of whatever the
        // original whole-line scan happened to end at.
        let lexer_before_line = if pending_opening_line_replay {
            let mut header_lexer = CodeLexer::new();
            header_lexer.scan_line(&header);
            header_lexer
        } else {
            item_lexer.clone()
        };
        // The opening physical line was already scanned once, for real,
        // by whichever scan produced `current_scanned` - re-scanning its
        // OWN remainder text here would double-count any paren/bracket/
        // angle depth that member content past the body's own `{` also
        // opens (the P2-1 finding this whole function was reworked for),
        // and `render_visible_range` never round-trips an escaped-
        // newline continuation's own backslash back through a fresh
        // scan (it is intentionally omitted from rendered text, since it
        // is not part of the literal's actual value - see `LexState::
        // NormalString`'s handling of a trailing `\`), so a member whose
        // literal continuation crosses the body's own `{` would lose
        // that continuation entirely on replay. `current_scanned`'s own
        // events/segments already correctly reflect both concerns -
        // reuse them directly instead of replaying, sourcing `sc` from
        // `current_scanned` (with `cur_pos` starting right after the
        // body's own `{`, not at this text's own start) exactly once,
        // for the untouched opening-line remainder only.
        let use_opening_line_directly = pending_opening_line_replay;
        pending_opening_line_replay = false;
        let sc = if use_opening_line_directly {
            current_scanned.clone()
        } else {
            scan_logical_item_line(item_lexer, line_text, same_line_remainder)
        };
        let sc_start = if use_opening_line_directly {
            open_brace_next_pos
        } else {
            Some((0, 0))
        };

        let mut cur_pos = sc_start;
        for event in &sc.events {
            // When `sc` is `current_scanned` reused directly (the
            // opening-line-remainder case above), it carries events for
            // the WHOLE first physical line, including ones before
            // `cur_pos` (the construct's own opening brace) - skip them;
            // this is a no-op for every other case, since `cur_pos`
            // there always starts at this text's own (0, 0).
            if let Some(pos) = cur_pos {
                if event.seg_idx < pos.0 || (event.seg_idx == pos.0 && event.char_idx < pos.1) {
                    continue;
                }
            }

            // A method whose entire body opens AND closes on this same
            // physical line (e.g. `pub fn f() -> u32 { private_a() }`,
            // itself immediately followed by more members on this same
            // line or the next) must have that body-close event handled
            // HERE too, not only when a skip was already in progress at
            // the start of a physical line - otherwise `in_default_
            // method_body` would stay incorrectly true past this line,
            // discarding every member that follows.
            if in_default_method_body {
                if event.kind == StructuralEventKind::MethodBodyClose {
                    in_default_method_body = false;
                    cur_pos = next_pos(&sc.segments, (event.seg_idx, event.char_idx));
                }
                continue;
            }

            if cur_member_text.trim().is_empty() {
                if let Some(remainder) = member_boundary_attribute(
                    &sc,
                    sc_start,
                    cur_pos,
                    item_lexer,
                    &lexer_before_line,
                ) {
                    body_remainder = Some(remainder);
                    cur_pos = None;
                    break;
                }
            }

            if event.kind == StructuralEventKind::TopLevelCloseBrace {
                let (chunk_text, chunk_code) =
                    if let Some(limit) = prev_pos(&sc.segments, (event.seg_idx, event.char_idx)) {
                        (
                            sc.render_visible_range(cur_pos, Some(limit)),
                            sc.code_tokens_range(cur_pos, Some(limit)),
                        )
                    } else {
                        (String::new(), String::new())
                    };
                if was_escaped {
                    cur_member_text.push_str(&chunk_text);
                    cur_member_code.push_str(&chunk_code);
                } else if continues_literal {
                    cur_member_text.push('\n');
                    cur_member_text.push_str(&chunk_text);
                    cur_member_code.push_str(&chunk_code);
                } else if !chunk_text.trim().is_empty() {
                    if !cur_member_text.is_empty()
                        && !cur_member_text.ends_with(' ')
                        && !chunk_text.starts_with(' ')
                    {
                        cur_member_text.push(' ');
                        cur_member_code.push(' ');
                    }
                    cur_member_text.push_str(&chunk_text);
                    cur_member_code.push_str(&chunk_code);
                }
                emit(
                    lines,
                    &mut pending_member_attrs,
                    &mut cur_member_text,
                    &mut cur_member_code,
                );
                body_closed = true;
                cur_pos = next_pos(&sc.segments, (event.seg_idx, event.char_idx));
                break;
            }

            let chunk_text =
                sc.render_visible_range(cur_pos, Some((event.seg_idx, event.char_idx)));
            let chunk_code = sc.code_tokens_range(cur_pos, Some((event.seg_idx, event.char_idx)));
            if was_escaped {
                cur_member_text.push_str(&chunk_text);
                cur_member_code.push_str(&chunk_code);
            } else if continues_literal {
                cur_member_text.push('\n');
                cur_member_text.push_str(&chunk_text);
                cur_member_code.push_str(&chunk_code);
            } else if !chunk_text.trim().is_empty() {
                if !cur_member_text.is_empty()
                    && !cur_member_text.ends_with(' ')
                    && !chunk_text.starts_with(' ')
                {
                    cur_member_text.push(' ');
                    cur_member_code.push(' ');
                }
                cur_member_text.push_str(&chunk_text);
                cur_member_code.push_str(&chunk_code);
            }
            cur_pos = next_pos(&sc.segments, (event.seg_idx, event.char_idx));

            if event.kind == StructuralEventKind::BaselineSemicolon {
                emit(
                    lines,
                    &mut pending_member_attrs,
                    &mut cur_member_text,
                    &mut cur_member_code,
                );
            } else if event.kind == StructuralEventKind::BaselineOpenBrace {
                let is_method = classify_trait_member(&cur_member_code) == TraitMemberKind::Method;
                if is_method {
                    emit(
                        lines,
                        &mut pending_member_attrs,
                        &mut cur_member_text,
                        &mut cur_member_code,
                    );
                    in_default_method_body = true;
                }
            }
        }

        if !body_closed && !in_default_method_body && cur_pos.is_some() {
            // Same reasoning as the in_default_method_body sub-loop's own
            // trailing-remainder check above: the for-loop's per-event
            // attribute check never ran if this split point was the LAST
            // (or only) thing `sc` has an event for - check once more
            // here before blindly folding the rest of `sc` into the
            // current member's own text.
            if cur_member_text.trim().is_empty() {
                if let Some(remainder) = member_boundary_attribute(
                    &sc,
                    sc_start,
                    cur_pos,
                    item_lexer,
                    &lexer_before_line,
                ) {
                    body_remainder = Some(remainder);
                    continue;
                }
            }
            let remainder_text = sc.render_visible_range(cur_pos, None);
            let remainder_code = sc.code_tokens_range(cur_pos, None);
            if was_escaped {
                cur_member_text.push_str(&remainder_text);
                cur_member_code.push_str(&remainder_code);
            } else if continues_literal {
                cur_member_text.push('\n');
                cur_member_text.push_str(&remainder_text);
                cur_member_code.push_str(&remainder_code);
            } else if !remainder_text.trim().is_empty() {
                if !cur_member_text.is_empty()
                    && !cur_member_text.ends_with(' ')
                    && !remainder_text.starts_with(' ')
                {
                    cur_member_text.push(' ');
                    cur_member_code.push(' ');
                }
                cur_member_text.push_str(&remainder_text);
                cur_member_code.push_str(&remainder_code);
            }
        }
    }

    if item_lexer.state == LexState::Normal {
        item_lexer.reset_top_level_depths();
    }
}

fn is_incomplete_public_prefix(code_tokens: &str) -> bool {
    let t = code_tokens.trim_start();
    if let Some(after_pub) = is_keyword_prefix(t, "pub") {
        if after_pub.starts_with('(') {
            let mut depth = 0;
            for c in after_pub.chars() {
                if c == '(' {
                    depth += 1;
                } else if c == ')' {
                    depth -= 1;
                    if depth == 0 {
                        return false;
                    }
                }
            }
            depth > 0
        } else {
            after_pub.is_empty()
        }
    } else {
        false
    }
}

fn append_code_chunk(
    dest_text: &mut String,
    dest_code: &mut String,
    chunk_text: &str,
    chunk_code: &str,
    was_escaped: bool,
    continues_literal: bool,
) {
    if was_escaped {
        dest_text.push_str(chunk_text);
        dest_code.push_str(chunk_code);
    } else if continues_literal {
        dest_text.push('\n');
        dest_text.push_str(chunk_text);
        dest_code.push_str(chunk_code);
    } else if !chunk_text.trim().is_empty() {
        let needs_space = !dest_text.is_empty()
            && !dest_text.ends_with(' ')
            && !chunk_text.starts_with(' ')
            && !dest_text
                .chars()
                .last()
                .is_some_and(is_attached_opening_delimiter)
            && !chunk_text
                .chars()
                .next()
                .is_some_and(is_attached_punctuation_prefix);
        if needs_space {
            dest_text.push(' ');
            dest_code.push(' ');
        }
        dest_text.push_str(chunk_text);
        dest_code.push_str(chunk_code);
    }
}

/// Same accumulation as `append_code_chunk`, additionally threading through
/// the literal byte-ranges `chunk_text` carries (see
/// `render_visible_range_tagged`'s doc comment) so a later
/// `normalize_variant` call can tell which bytes of the accumulated
/// `dest_text` are literal without re-scanning it for quote characters.
/// `append_code_chunk` itself is untouched - this only wraps it and remaps
/// `chunk_literal_ranges` by the offset at which `chunk_text` actually
/// landed in `dest_text` (which can vary depending on `was_escaped`/
/// `continues_literal`/leading-glue-space decisions `append_code_chunk`
/// makes internally, so the offset is read back from `dest_text.len()`
/// after the call rather than predicted ahead of it).
#[allow(clippy::too_many_arguments)]
fn append_variant_chunk(
    dest_text: &mut String,
    dest_code: &mut String,
    dest_literal_ranges: &mut Vec<(usize, usize)>,
    chunk_text: &str,
    chunk_literal_ranges: &[(usize, usize)],
    chunk_code: &str,
    was_escaped: bool,
    continues_literal: bool,
) {
    append_code_chunk(
        dest_text,
        dest_code,
        chunk_text,
        chunk_code,
        was_escaped,
        continues_literal,
    );
    if chunk_literal_ranges.is_empty() {
        return;
    }
    let chunk_start_in_dest = dest_text.len() - chunk_text.len();
    // `append_code_chunk`'s `continues_literal` branch (taken whenever this
    // chunk continues a literal from the previous physical line, and that
    // previous line did not itself end in an escaped `\` continuation)
    // inserts one synthetic '\n' immediately before chunk_text, to
    // represent the real physical line break that occurred INSIDE the
    // literal (see requirement F: an actual runtime newline in a raw
    // literal spanning two physical lines). That byte is itself literal
    // content, not collapsible code whitespace - if left untagged,
    // normalize_variant's whitespace-collapse logic would silently turn a
    // genuine embedded newline into an ordinary space. Since the chunk
    // always begins already inside the continuing literal whenever this
    // branch fires, the chunk's own first literal range always starts at
    // its own byte 0; widen that one range left by one byte to also cover
    // the synthetic newline.
    let extend_left = !was_escaped
        && continues_literal
        && chunk_start_in_dest > 0
        && chunk_literal_ranges.first().is_some_and(|&(s, _)| s == 0);
    for (idx, &(s, e)) in chunk_literal_ranges.iter().enumerate() {
        let start = if idx == 0 && extend_left {
            chunk_start_in_dest - 1
        } else {
            chunk_start_in_dest + s
        };
        dest_literal_ranges.push((start, chunk_start_in_dest + e));
    }
}

fn emit_captured_fragment(text: &str) -> Option<String> {
    (!text.trim().is_empty()).then(|| text.to_owned())
}

/// Normalizes one captured enum-variant declaration for the golden surface.
///
/// `literal_ranges` are the byte ranges within `text` that came verbatim
/// from `VisibleSegment::Literal` content (see
/// `render_visible_range_tagged`'s own doc comment) - authoritative, not
/// re-derived here. A tagged `(char, is_literal)` stream is built from
/// `text` plus these ranges; the structural normalization below (comma/
/// paren/brace collapsing, whitespace collapsing) only ever inspects and
/// mutates `is_literal == false` entries. `is_literal == true` entries are
/// copied through completely unconditionally - never quote-scanned, never
/// whitespace-collapsed - so a literal's own content (an embedded `"`, an
/// arbitrary raw-string hash count, or genuine internal whitespace) can
/// never be misinterpreted as structural code, regardless of what
/// characters it contains. This replaces the previous per-call `in_str`/
/// `in_char` quote parser, which re-scanned already-classified text with
/// weaker rules than the authoritative lexer (it treated any `"` as a
/// normal-string boundary, so an embedded quote inside a raw string like
/// `r#"a"  b"#` was misread as closing the literal early, letting the
/// whitespace after it be collapsed as if it were code).
fn normalize_variant(text: &str, literal_ranges: &[(usize, usize)]) -> String {
    let is_literal_byte = |byte_idx: usize| {
        literal_ranges
            .iter()
            .any(|&(s, e)| byte_idx >= s && byte_idx < e)
    };
    let mut tagged: Vec<(char, bool)> = text
        .char_indices()
        .map(|(byte_idx, c)| (c, is_literal_byte(byte_idx)))
        .collect();

    while matches!(tagged.first(), Some((c, false)) if c.is_whitespace()) {
        tagged.remove(0);
    }
    while matches!(tagged.last(), Some((c, false)) if c.is_whitespace()) {
        tagged.pop();
    }
    if tagged.is_empty() {
        return String::new();
    }
    if !matches!(tagged.last(), Some((',', false))) {
        tagged.push((',', false));
    }

    let mut out = String::with_capacity(tagged.len());
    let mut i = 0;
    while i < tagged.len() {
        let (c, is_lit) = tagged[i];
        if is_lit {
            out.push(c);
            i += 1;
            continue;
        }

        if c == ',' {
            let mut j = i + 1;
            while j < tagged.len() && !tagged[j].1 && tagged[j].0.is_whitespace() {
                j += 1;
            }
            if j < tagged.len() && !tagged[j].1 && (tagged[j].0 == ')' || tagged[j].0 == '}') {
                if tagged[j].0 == '}' && !out.ends_with(' ') {
                    out.push(' ');
                }
                out.push(tagged[j].0);
                i = j + 1;
                continue;
            }
        }
        if c == '(' {
            out.push('(');
            i += 1;
            while i < tagged.len() && !tagged[i].1 && tagged[i].0.is_whitespace() {
                i += 1;
            }
            continue;
        }
        if c == ')' {
            if out.ends_with(' ') && !out.ends_with("()") {
                out.pop();
            }
            out.push(')');
            i += 1;
            continue;
        }
        if c == '{' {
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push('{');
            let mut j = i + 1;
            while j < tagged.len() && !tagged[j].1 && tagged[j].0.is_whitespace() {
                j += 1;
            }
            if j < tagged.len() && !(tagged[j].0 == '}' && !tagged[j].1) {
                out.push(' ');
            }
            i = j;
            continue;
        }
        if c == '}' {
            if !out.ends_with(' ') && !out.ends_with('{') {
                out.push(' ');
            }
            out.push('}');
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            if !out.is_empty()
                && !out.ends_with(' ')
                && !out
                    .chars()
                    .last()
                    .is_some_and(is_attached_opening_delimiter)
            {
                let mut j = i + 1;
                while j < tagged.len() && !tagged[j].1 && tagged[j].0.is_whitespace() {
                    j += 1;
                }
                if j < tagged.len() && (tagged[j].1 || !is_attached_punctuation_prefix(tagged[j].0))
                {
                    out.push(' ');
                }
                i = j;
                continue;
            }
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn normalized_public_surface(path: &str) -> String {
    let src = fs::read_to_string(path).unwrap_or_else(|err| panic!("read {path}: {err}"));
    normalized_public_surface_str(path, &src)
}

struct CapturedAttribute {
    text: String,
    remainder: String,
}

fn capture_attribute(
    src_lines: &[&str],
    idx: &mut usize,
    lexer: &mut CodeLexer,
    first_line: &str,
) -> CapturedAttribute {
    let mut attr_text = String::new();
    let mut line = first_line;
    loop {
        let lexer_before_line = lexer.clone();
        let continues_literal = lexer.state.is_in_string_literal();
        let scanned = lexer.scan_line(line);
        let attr_end = scanned.first_outer_attribute_close_seg;
        let rendered = attr_end.map_or_else(
            || scanned.render_visible(),
            |end| scanned.render_visible_range(None, Some(end)),
        );
        if continues_literal {
            attr_text.push('\n');
        } else if !attr_text.is_empty() && !attr_text.ends_with(' ') && !rendered.starts_with(' ') {
            attr_text.push(' ');
        }
        attr_text.push_str(&rendered);

        if let Some(end) = attr_end {
            let mut remainder = next_pos(&scanned.segments, end)
                .map_or_else(String::new, |start| {
                    scanned.render_visible_range(Some(start), None)
                });
            // Same reasoning as `member_boundary_attribute`: this
            // remainder (everything after the attribute's own `]`) is
            // handed back as plain text for a LATER, independent
            // `scan_line` call to replay, and `render_visible_range`
            // never renders a real escaped continuation's trailing `\` -
            // restore it here so that later scan doesn't misread an
            // unclosed literal spanning further physical lines as an
            // ordinary (non-continuing) unclosed string.
            if !remainder.is_empty() && scanned.ends_in_escaped_continuation {
                remainder.push('\\');
            }
            *lexer = lexer_before_line;
            lexer.scan_line(&rendered);
            return CapturedAttribute {
                text: attr_text,
                remainder,
            };
        }

        if *idx + 1 >= src_lines.len() {
            return CapturedAttribute {
                text: attr_text,
                remainder: String::new(),
            };
        }
        *idx += 1;
        line = src_lines[*idx];
    }
}

fn scan_logical_item_line(
    lexer: &mut CodeLexer,
    line: &str,
    same_line_remainder: &mut Option<String>,
) -> ScannedLine {
    scan_logical_item_line_with_terminator(lexer, line, same_line_remainder, None)
}

fn scan_logical_item_line_with_terminator(
    lexer: &mut CodeLexer,
    line: &str,
    same_line_remainder: &mut Option<String>,
    closes_with_brace: Option<bool>,
) -> ScannedLine {
    let lexer_before_line = lexer.clone();
    let scanned = lexer.scan_line(line);
    let public_braced_item = is_public_fn(&scanned.code_tokens)
        || is_public_enum(&scanned.code_tokens)
        || is_public_struct(&scanned.code_tokens)
        || is_public_union(&scanned.code_tokens)
        || is_public_trait(&scanned.code_tokens)
        || is_public_mod(&scanned.code_tokens);
    let public_semicolon_item = is_public_const_or_static(&scanned.code_tokens)
        || is_public_use(&scanned.code_tokens)
        || is_public_type_alias(&scanned.code_tokens)
        || is_public_extern_crate(&scanned.code_tokens);
    let has_balanced_top_level_braces = scanned.has_top_level_open_brace
        && scanned
            .events
            .iter()
            .any(|event| event.kind == StructuralEventKind::TopLevelCloseBrace);
    let closes_with_brace = closes_with_brace.unwrap_or({
        lexer_before_line.brace_depth > 0
            || (public_braced_item && scanned.has_top_level_open_brace)
            || (has_balanced_top_level_braces && !public_semicolon_item)
    });
    let boundary = scanned.events.iter().find_map(|event| {
        ((closes_with_brace && event.kind == StructuralEventKind::TopLevelCloseBrace)
            || (!closes_with_brace && event.kind == StructuralEventKind::TopLevelSemicolon))
            .then_some((event.seg_idx, event.char_idx))
    });
    let Some(boundary) = boundary else {
        return scanned;
    };
    let mut remainder = next_pos(&scanned.segments, boundary).map_or_else(String::new, |start| {
        scanned.render_visible_range(Some(start), None)
    });
    if remainder.trim().is_empty() {
        return scanned;
    }
    // Same reasoning as `member_boundary_attribute`/`capture_attribute`:
    // this remainder is handed back (via `same_line_remainder`) as plain
    // text for a LATER, independent `scan_logical_item_line` call to
    // replay, and `render_visible_range` never renders a real escaped
    // continuation's trailing `\` - restore it here so a member whose
    // own literal continues past this boundary onto the next physical
    // line is not misread as an ordinary (non-continuing) unclosed
    // string on that later scan.
    if scanned.ends_in_escaped_continuation {
        remainder.push('\\');
    }

    let consumed = scanned.render_visible_range(None, Some(boundary));
    *lexer = lexer_before_line;
    let scanned = lexer.scan_line(&consumed);
    *same_line_remainder = Some(remainder);
    scanned
}

fn normalized_public_surface_str(path: &str, src: &str) -> String {
    let src_lines: Vec<&str> = src.lines().collect();
    let mut lines = Vec::new();
    let mut pending_attrs = Vec::new();
    let mut same_line_remainder = None;
    let mut idx = 0usize;
    let mut file_lexer = CodeLexer::new();

    while idx < src_lines.len() {
        let owned_line = same_line_remainder.take();
        let raw_line = owned_line.as_deref().unwrap_or(src_lines[idx]);
        if raw_line.trim().is_empty() && file_lexer.state == LexState::Normal {
            idx += 1;
            continue;
        }

        let mut item_lexer = file_lexer.clone();
        let scanned = scan_logical_item_line(&mut item_lexer, raw_line, &mut same_line_remainder);

        if is_outer_attribute_start(&scanned.code_tokens) && file_lexer.state == LexState::Normal {
            item_lexer = file_lexer.clone();
            let captured = capture_attribute(&src_lines, &mut idx, &mut item_lexer, raw_line);
            pending_attrs.push(captured.text);
            if item_lexer.state == LexState::Normal {
                item_lexer.reset_top_level_depths();
            }
            file_lexer = item_lexer;
            if captured.remainder.trim().is_empty() {
                idx += 1;
            } else {
                same_line_remainder = Some(captured.remainder);
            }
            continue;
        }

        if is_public_code(&scanned.code_tokens)
            || is_incomplete_public_prefix(&scanned.code_tokens)
            || is_impl_start(&scanned.code_tokens)
            || is_extern_block_start(&scanned.code_tokens)
            || is_incomplete_extern_prefix(&scanned.code_tokens)
        {
            if is_impl_start(&scanned.code_tokens)
                || is_extern_block_start(&scanned.code_tokens)
                || is_incomplete_extern_prefix(&scanned.code_tokens)
            {
                // An impl block (trait impl or inherent impl) or an
                // extern block is never itself part of the public
                // declaration inventory - its own header is never pushed
                // either (see the branches below). A leading attribute
                // conceptually applies to the WHOLE block, not to
                // whichever public member happens to be captured next,
                // so it is dropped here rather than mis-attached -
                // matching this scanner's pre-existing behavior of not
                // seeing trait impls (and their attributes) at all.
                pending_attrs.clear();
            } else {
                lines.append(&mut pending_attrs);
            }

            let mut current_scanned = scanned;
            let mut prefix_text = current_scanned.render_visible();
            let mut combined_code_tokens = current_scanned.code_tokens.clone();

            while (is_incomplete_public_prefix(&combined_code_tokens)
                || is_incomplete_extern_prefix(&combined_code_tokens))
                && idx + 1 < src_lines.len()
            {
                idx += 1;
                let next_line = src_lines[idx];
                if next_line.trim().is_empty() && item_lexer.state == LexState::Normal {
                    continue;
                }
                let continues_literal = item_lexer.state.is_in_string_literal();
                let sc =
                    scan_logical_item_line(&mut item_lexer, next_line, &mut same_line_remainder);
                let rendered = sc.render_visible();
                if continues_literal {
                    prefix_text.push('\n');
                    prefix_text.push_str(&rendered);
                } else {
                    let trimmed_rendered = rendered.trim();
                    if trimmed_rendered.is_empty() && !sc.ends_in_string_literal {
                        continue;
                    }
                    if !prefix_text.ends_with(' ')
                        && !prefix_text.ends_with('(')
                        && !trimmed_rendered.starts_with(')')
                    {
                        prefix_text.push(' ');
                    }
                    prefix_text.push_str(trimmed_rendered);
                }
                let trimmed_code = sc.code_tokens.trim();
                if !trimmed_code.is_empty() {
                    if !combined_code_tokens.ends_with(' ')
                        && !combined_code_tokens.ends_with('(')
                        && !trimmed_code.starts_with(')')
                    {
                        combined_code_tokens.push(' ');
                    }
                    combined_code_tokens.push_str(trimmed_code);
                }
                current_scanned = sc;
            }

            while is_public_qualifiers_only(&combined_code_tokens) && idx + 1 < src_lines.len() {
                idx += 1;
                let next_line = src_lines[idx];
                if next_line.trim().is_empty() && item_lexer.state == LexState::Normal {
                    continue;
                }
                let continues_literal = item_lexer.state.is_in_string_literal();
                let sc =
                    scan_logical_item_line(&mut item_lexer, next_line, &mut same_line_remainder);
                let rendered = sc.render_visible();
                if continues_literal {
                    prefix_text.push('\n');
                    prefix_text.push_str(&rendered);
                } else {
                    if rendered.trim().is_empty() && !sc.ends_in_string_literal {
                        continue;
                    }
                    if !prefix_text.ends_with(' ') && !rendered.starts_with(' ') {
                        prefix_text.push(' ');
                    }
                    prefix_text.push_str(&rendered);
                }
                if !combined_code_tokens.ends_with(' ') && !sc.code_tokens.starts_with(' ') {
                    combined_code_tokens.push(' ');
                }
                combined_code_tokens.push_str(&sc.code_tokens);
                current_scanned = sc;
            }

            if is_public_fn(&combined_code_tokens) {
                let mut signature = if current_scanned.has_top_level_open_brace {
                    let up_to_brace = current_scanned.text_up_to_function_body_open_brace();
                    let rendered_last = current_scanned.render_visible();
                    if prefix_text.ends_with(&rendered_last) {
                        let prefix_head = &prefix_text[..prefix_text.len() - rendered_last.len()];
                        format!("{prefix_head}{up_to_brace}")
                    } else {
                        up_to_brace
                    }
                } else {
                    prefix_text
                };

                while !current_scanned.has_top_level_open_brace
                    && !current_scanned.has_top_level_semicolon
                    && idx + 1 < src_lines.len()
                {
                    idx += 1;
                    let continuation = src_lines[idx];
                    if continuation.trim().is_empty() && item_lexer.state == LexState::Normal {
                        continue;
                    }
                    let was_escaped = item_lexer.state.is_escaped_continuation();
                    let continues_literal = item_lexer.state.is_in_string_literal();
                    current_scanned = scan_logical_item_line(
                        &mut item_lexer,
                        continuation,
                        &mut same_line_remainder,
                    );
                    let rendered = current_scanned.text_up_to_function_body_open_brace();
                    if was_escaped {
                        signature.push_str(&rendered);
                        continue;
                    }
                    if continues_literal {
                        signature.push('\n');
                        signature.push_str(&rendered);
                        continue;
                    }
                    if rendered.trim().is_empty() {
                        continue;
                    }
                    if !signature.ends_with(' ') && !rendered.starts_with(' ') {
                        signature.push(' ');
                    }
                    signature.push_str(&rendered);
                }
                lines.push(signature);
                let mut body_closed = current_scanned
                    .events
                    .iter()
                    .any(|event| event.kind == StructuralEventKind::TopLevelCloseBrace);
                while current_scanned.has_top_level_open_brace
                    && !body_closed
                    && idx + 1 < src_lines.len()
                {
                    idx += 1;
                    current_scanned = scan_logical_item_line_with_terminator(
                        &mut item_lexer,
                        src_lines[idx],
                        &mut same_line_remainder,
                        Some(true),
                    );
                    body_closed = current_scanned
                        .events
                        .iter()
                        .any(|event| event.kind == StructuralEventKind::TopLevelCloseBrace);
                }
                if item_lexer.state == LexState::Normal {
                    item_lexer.reset_top_level_depths();
                }
                file_lexer = item_lexer;
                if same_line_remainder.is_none() {
                    idx += 1;
                }
                continue;
            }

            if is_public_enum(&combined_code_tokens) {
                let mut enum_decl = prefix_text;
                let mut body_open = current_scanned.has_top_level_open_brace;

                while !body_open && idx + 1 < src_lines.len() {
                    idx += 1;
                    let continuation = src_lines[idx];
                    if continuation.trim().is_empty() && item_lexer.state == LexState::Normal {
                        continue;
                    }
                    let was_escaped = item_lexer.state.is_escaped_continuation();
                    let continues_literal = item_lexer.state.is_in_string_literal();
                    let sc = scan_logical_item_line(
                        &mut item_lexer,
                        continuation,
                        &mut same_line_remainder,
                    );
                    let rendered = sc.render_visible();
                    if was_escaped {
                        enum_decl.push_str(&rendered);
                    } else if continues_literal {
                        enum_decl.push('\n');
                        enum_decl.push_str(&rendered);
                    } else {
                        if !rendered.trim().is_empty() {
                            if !enum_decl.ends_with(' ') && !rendered.starts_with(' ') {
                                enum_decl.push(' ');
                            }
                            enum_decl.push_str(&rendered);
                        }
                    }
                    body_open = sc.has_top_level_open_brace;
                    current_scanned = sc;
                }

                let header = if current_scanned.has_top_level_open_brace {
                    let up_to_brace = current_scanned.text_up_to_function_body_open_brace();
                    let rendered_last = current_scanned.render_visible();
                    if enum_decl.ends_with(&rendered_last) {
                        let prefix_head = &enum_decl[..enum_decl.len() - rendered_last.len()];
                        format!("{prefix_head}{up_to_brace}")
                    } else {
                        up_to_brace
                    }
                } else {
                    enum_decl
                };
                lines.push(header);

                let mut pending_variant_attrs = Vec::new();
                let mut cur_variant_text = String::new();
                let mut cur_variant_code = String::new();
                let mut cur_variant_literal_ranges: Vec<(usize, usize)> = Vec::new();
                let mut enum_body_closed = false;

                // Process remainder of opening line (if any)
                if let Some(open_brace_pos) = current_scanned.first_top_level_open_brace_seg {
                    if let Some(start_pos) = next_pos(&current_scanned.segments, open_brace_pos) {
                        let mut cur_pos = Some(start_pos);
                        for event in &current_scanned.events {
                            if event.kind == StructuralEventKind::TopLevelOpenBrace {
                                continue;
                            }
                            if event.seg_idx < start_pos.0
                                || (event.seg_idx == start_pos.0 && event.char_idx < start_pos.1)
                            {
                                continue;
                            }

                            if event.kind == StructuralEventKind::TopLevelCloseBrace {
                                let (chunk_text, chunk_literal_ranges, chunk_code) =
                                    if let Some(limit) = prev_pos(
                                        &current_scanned.segments,
                                        (event.seg_idx, event.char_idx),
                                    ) {
                                        let (text, ranges) = current_scanned
                                            .render_visible_range_tagged(cur_pos, Some(limit));
                                        (
                                            text,
                                            ranges,
                                            current_scanned.code_tokens_range(cur_pos, Some(limit)),
                                        )
                                    } else {
                                        (String::new(), Vec::new(), String::new())
                                    };
                                append_variant_chunk(
                                    &mut cur_variant_text,
                                    &mut cur_variant_code,
                                    &mut cur_variant_literal_ranges,
                                    &chunk_text,
                                    &chunk_literal_ranges,
                                    &chunk_code,
                                    false,
                                    false,
                                );
                                let trimmed_text = normalize_variant(
                                    &cur_variant_text,
                                    &cur_variant_literal_ranges,
                                );
                                if !trimmed_text.is_empty() {
                                    lines.append(&mut pending_variant_attrs);
                                    lines.push(trimmed_text);
                                } else {
                                    pending_variant_attrs.clear();
                                }
                                cur_variant_text.clear();
                                cur_variant_code.clear();
                                cur_variant_literal_ranges.clear();
                                enum_body_closed = true;
                                cur_pos = next_pos(
                                    &current_scanned.segments,
                                    (event.seg_idx, event.char_idx),
                                );
                                break;
                            }

                            let (chunk_text, chunk_literal_ranges) = current_scanned
                                .render_visible_range_tagged(
                                    cur_pos,
                                    Some((event.seg_idx, event.char_idx)),
                                );
                            let chunk_code = current_scanned
                                .code_tokens_range(cur_pos, Some((event.seg_idx, event.char_idx)));
                            append_variant_chunk(
                                &mut cur_variant_text,
                                &mut cur_variant_code,
                                &mut cur_variant_literal_ranges,
                                &chunk_text,
                                &chunk_literal_ranges,
                                &chunk_code,
                                false,
                                false,
                            );
                            cur_pos = next_pos(
                                &current_scanned.segments,
                                (event.seg_idx, event.char_idx),
                            );

                            if event.kind == StructuralEventKind::BaselineComma {
                                let trimmed_text = normalize_variant(
                                    &cur_variant_text,
                                    &cur_variant_literal_ranges,
                                );
                                if !trimmed_text.is_empty() {
                                    lines.append(&mut pending_variant_attrs);
                                    lines.push(trimmed_text);
                                } else {
                                    pending_variant_attrs.clear();
                                }
                                cur_variant_text.clear();
                                cur_variant_code.clear();
                                cur_variant_literal_ranges.clear();
                            }
                        }
                        if !enum_body_closed && cur_pos.is_some() {
                            let (remainder_text, remainder_literal_ranges) =
                                current_scanned.render_visible_range_tagged(cur_pos, None);
                            let remainder_code = current_scanned.code_tokens_range(cur_pos, None);
                            append_variant_chunk(
                                &mut cur_variant_text,
                                &mut cur_variant_code,
                                &mut cur_variant_literal_ranges,
                                &remainder_text,
                                &remainder_literal_ranges,
                                &remainder_code,
                                false,
                                false,
                            );
                        }
                    }
                }

                let mut variant_remainder = None;
                while !enum_body_closed
                    && (variant_remainder.is_some() || idx + 1 < src_lines.len())
                {
                    let owned_line = variant_remainder.take();
                    if owned_line.is_none() {
                        idx += 1;
                    }
                    let variant_line = owned_line.as_deref().unwrap_or(src_lines[idx]);
                    if variant_line.trim().is_empty() && item_lexer.state == LexState::Normal {
                        continue;
                    }
                    if cur_variant_text.trim().is_empty() && item_lexer.state == LexState::Normal {
                        let mut check_lexer = item_lexer.clone();
                        let check_sc = check_lexer.scan_line(variant_line);
                        if is_outer_attribute_start(&check_sc.code_tokens) {
                            let captured = capture_attribute(
                                &src_lines,
                                &mut idx,
                                &mut item_lexer,
                                variant_line,
                            );
                            pending_variant_attrs.push(captured.text);
                            if !captured.remainder.trim().is_empty() {
                                variant_remainder = Some(captured.remainder);
                            }
                            continue;
                        }
                    }

                    let was_escaped = item_lexer.state.is_escaped_continuation();
                    let continues_literal = item_lexer.state.is_in_string_literal();
                    let sc = scan_logical_item_line(
                        &mut item_lexer,
                        variant_line,
                        &mut same_line_remainder,
                    );

                    let mut cur_pos = Some((0, 0));
                    for event in &sc.events {
                        if event.kind == StructuralEventKind::TopLevelCloseBrace {
                            let (chunk_text, chunk_literal_ranges, chunk_code) =
                                if let Some(limit) =
                                    prev_pos(&sc.segments, (event.seg_idx, event.char_idx))
                                {
                                    let (text, ranges) =
                                        sc.render_visible_range_tagged(cur_pos, Some(limit));
                                    (text, ranges, sc.code_tokens_range(cur_pos, Some(limit)))
                                } else {
                                    (String::new(), Vec::new(), String::new())
                                };
                            append_variant_chunk(
                                &mut cur_variant_text,
                                &mut cur_variant_code,
                                &mut cur_variant_literal_ranges,
                                &chunk_text,
                                &chunk_literal_ranges,
                                &chunk_code,
                                was_escaped,
                                continues_literal,
                            );
                            let trimmed_text =
                                normalize_variant(&cur_variant_text, &cur_variant_literal_ranges);
                            if !trimmed_text.is_empty() {
                                lines.append(&mut pending_variant_attrs);
                                lines.push(trimmed_text);
                            } else {
                                pending_variant_attrs.clear();
                            }
                            cur_variant_text.clear();
                            cur_variant_code.clear();
                            cur_variant_literal_ranges.clear();
                            enum_body_closed = true;
                            cur_pos = next_pos(&sc.segments, (event.seg_idx, event.char_idx));
                            break;
                        }

                        let (chunk_text, chunk_literal_ranges) = sc.render_visible_range_tagged(
                            cur_pos,
                            Some((event.seg_idx, event.char_idx)),
                        );
                        let chunk_code =
                            sc.code_tokens_range(cur_pos, Some((event.seg_idx, event.char_idx)));
                        append_variant_chunk(
                            &mut cur_variant_text,
                            &mut cur_variant_code,
                            &mut cur_variant_literal_ranges,
                            &chunk_text,
                            &chunk_literal_ranges,
                            &chunk_code,
                            was_escaped,
                            continues_literal,
                        );
                        cur_pos = next_pos(&sc.segments, (event.seg_idx, event.char_idx));

                        if event.kind == StructuralEventKind::BaselineComma {
                            let trimmed_text =
                                normalize_variant(&cur_variant_text, &cur_variant_literal_ranges);
                            if !trimmed_text.is_empty() {
                                lines.append(&mut pending_variant_attrs);
                                lines.push(trimmed_text);
                            } else {
                                pending_variant_attrs.clear();
                            }
                            cur_variant_text.clear();
                            cur_variant_code.clear();
                            cur_variant_literal_ranges.clear();
                        }
                    }

                    if !enum_body_closed && cur_pos.is_some() {
                        let (remainder_text, remainder_literal_ranges) =
                            sc.render_visible_range_tagged(cur_pos, None);
                        let remainder_code = sc.code_tokens_range(cur_pos, None);
                        append_variant_chunk(
                            &mut cur_variant_text,
                            &mut cur_variant_code,
                            &mut cur_variant_literal_ranges,
                            &remainder_text,
                            &remainder_literal_ranges,
                            &remainder_code,
                            was_escaped,
                            continues_literal,
                        );
                    }
                }

                if item_lexer.state == LexState::Normal {
                    item_lexer.reset_top_level_depths();
                }
                file_lexer = item_lexer;
                if same_line_remainder.is_none() {
                    idx += 1;
                }
                continue;
            }

            if is_public_const_or_static(&combined_code_tokens) {
                let mut item = prefix_text;
                let mut prev_ended_in_string = current_scanned.ends_in_string_literal;
                let mut has_terminator = current_scanned.has_top_level_semicolon;

                while !has_terminator && idx + 1 < src_lines.len() {
                    idx += 1;
                    let next_line = src_lines[idx];
                    if next_line.trim().is_empty() && item_lexer.state == LexState::Normal {
                        continue;
                    }
                    let was_escaped = item_lexer.state.is_escaped_continuation();
                    let sc = scan_logical_item_line_with_terminator(
                        &mut item_lexer,
                        next_line,
                        &mut same_line_remainder,
                        Some(false),
                    );
                    has_terminator = sc.has_top_level_semicolon;
                    let rendered = sc.render_visible();
                    if rendered.trim().is_empty() && !sc.ends_in_string_literal {
                        prev_ended_in_string = sc.ends_in_string_literal;
                        continue;
                    }
                    if was_escaped {
                        item.push_str(&rendered);
                    } else if prev_ended_in_string {
                        item.push('\n');
                        item.push_str(&rendered);
                    } else {
                        if !item.ends_with(' ') && !rendered.starts_with(' ') {
                            item.push(' ');
                        }
                        item.push_str(&rendered);
                    }
                    prev_ended_in_string = sc.ends_in_string_literal;
                }

                lines.push(item);
                if item_lexer.state == LexState::Normal {
                    item_lexer.reset_top_level_depths();
                }
                file_lexer = item_lexer;
                if same_line_remainder.is_none() {
                    idx += 1;
                }
                continue;
            }

            if is_public_use(&combined_code_tokens) {
                let mut item = prefix_text;
                let mut is_done = current_scanned.has_top_level_semicolon;

                while !is_done && idx + 1 < src_lines.len() {
                    idx += 1;
                    let next_line = src_lines[idx];
                    if next_line.trim().is_empty() && item_lexer.state == LexState::Normal {
                        continue;
                    }
                    let was_escaped = item_lexer.state.is_escaped_continuation();
                    let continues_literal = item_lexer.state.is_in_string_literal();
                    let sc = scan_logical_item_line_with_terminator(
                        &mut item_lexer,
                        next_line,
                        &mut same_line_remainder,
                        Some(false),
                    );
                    is_done = sc.has_top_level_semicolon;
                    let rendered = sc.render_visible();
                    if was_escaped {
                        item.push_str(&rendered);
                        continue;
                    }
                    if continues_literal {
                        item.push('\n');
                        item.push_str(&rendered);
                        continue;
                    }
                    if rendered.trim().is_empty() && !sc.ends_in_string_literal {
                        continue;
                    }
                    if !item.ends_with(' ') && !rendered.starts_with(' ') {
                        item.push(' ');
                    }
                    item.push_str(&rendered);
                }

                lines.push(item);
                if item_lexer.state == LexState::Normal {
                    item_lexer.reset_top_level_depths();
                }
                file_lexer = item_lexer;
                if same_line_remainder.is_none() {
                    idx += 1;
                }
                continue;
            }

            if is_public_struct(&combined_code_tokens) || is_public_union(&combined_code_tokens) {
                while !current_scanned.has_top_level_semicolon
                    && !current_scanned.has_top_level_open_paren
                    && !current_scanned.has_top_level_open_brace
                    && idx + 1 < src_lines.len()
                {
                    idx += 1;
                    let continuation = src_lines[idx];
                    if continuation.trim().is_empty() && item_lexer.state == LexState::Normal {
                        continue;
                    }
                    let was_escaped = item_lexer.state.is_escaped_continuation();
                    let continues_literal = item_lexer.state.is_in_string_literal();
                    let sc = scan_logical_item_line(
                        &mut item_lexer,
                        continuation,
                        &mut same_line_remainder,
                    );
                    let rendered = sc.render_visible();
                    if was_escaped {
                        prefix_text.push_str(&rendered);
                    } else if continues_literal {
                        prefix_text.push('\n');
                        prefix_text.push_str(&rendered);
                    } else {
                        if rendered.trim().is_empty() && !sc.ends_in_string_literal {
                            continue;
                        }
                        let needs_space = !prefix_text.ends_with(' ')
                            && !rendered.starts_with(' ')
                            && !rendered.starts_with('(');
                        if needs_space {
                            prefix_text.push(' ');
                        }
                        prefix_text.push_str(&rendered);
                    }
                    if !combined_code_tokens.ends_with(' ') && !sc.code_tokens.starts_with(' ') {
                        combined_code_tokens.push(' ');
                    }
                    combined_code_tokens.push_str(&sc.code_tokens);
                    current_scanned = sc;
                }

                let is_tuple = has_public_tuple_struct_body_open(&combined_code_tokens);
                if current_scanned.has_top_level_semicolon && !is_tuple {
                    lines.push(prefix_text);
                    if item_lexer.state == LexState::Normal {
                        item_lexer.reset_top_level_depths();
                    }
                    file_lexer = item_lexer;
                    if same_line_remainder.is_none() {
                        idx += 1;
                    }
                    continue;
                }

                if let Some(open_pos) = is_tuple
                    .then(|| current_scanned.top_level_open_paren_segs.last().copied())
                    .flatten()
                {
                    let rendered_last = current_scanned.render_visible();
                    let header_line = current_scanned.render_visible_range(None, Some(open_pos));
                    let header = if prefix_text.ends_with(&rendered_last) {
                        let prefix_head = &prefix_text[..prefix_text.len() - rendered_last.len()];
                        format!("{prefix_head}{header_line}")
                    } else {
                        header_line
                    };
                    let mut tuple_body = String::new();
                    let mut body_start = next_pos(&current_scanned.segments, open_pos);
                    let mut opening_line = true;
                    let close_pos = loop {
                        if let Some(close_pos) = current_scanned
                            .top_level_close_paren_segs
                            .iter()
                            .copied()
                            .find(|&close| body_start.is_none_or(|start| close >= start))
                        {
                            if let Some(body_end) = prev_pos(&current_scanned.segments, close_pos) {
                                tuple_body.push_str(
                                    &current_scanned
                                        .render_visible_range(body_start, Some(body_end)),
                                );
                            }
                            break close_pos;
                        } else if !(opening_line && body_start.is_none()) {
                            tuple_body
                                .push_str(&current_scanned.render_visible_range(body_start, None));
                        }
                        assert!(
                            idx + 1 < src_lines.len(),
                            "unterminated public tuple struct in {path}"
                        );
                        tuple_body.push('\n');
                        idx += 1;
                        current_scanned = scan_logical_item_line(
                            &mut item_lexer,
                            src_lines[idx],
                            &mut same_line_remainder,
                        );
                        body_start = Some((0, 0));
                        opening_line = false;
                    };

                    let semicolon_pos = |sc: &ScannedLine| {
                        sc.events.iter().find_map(|event| {
                            (event.kind == StructuralEventKind::TopLevelSemicolon)
                                .then_some((event.seg_idx, event.char_idx))
                        })
                    };
                    let mut suffix = next_pos(&current_scanned.segments, close_pos).map_or_else(
                        String::new,
                        |start| {
                            current_scanned
                                .render_visible_range(Some(start), semicolon_pos(&current_scanned))
                        },
                    );
                    let mut suffix_code = String::new();
                    while semicolon_pos(&current_scanned).is_none() {
                        assert!(
                            idx + 1 < src_lines.len(),
                            "unterminated public tuple struct in {path}"
                        );
                        idx += 1;
                        let was_escaped = item_lexer.state.is_escaped_continuation();
                        let continues_literal = item_lexer.state.is_in_string_literal();
                        current_scanned = scan_logical_item_line(
                            &mut item_lexer,
                            src_lines[idx],
                            &mut same_line_remainder,
                        );
                        let rendered = current_scanned
                            .render_visible_range(None, semicolon_pos(&current_scanned));
                        let rendered_code = current_scanned
                            .code_tokens_range(None, semicolon_pos(&current_scanned));
                        // Same continuation-aware join every other multi-
                        // line accumulator in this file uses - a naive
                        // "always insert one space between chunks" join
                        // (the previous behavior here) cannot distinguish
                        // a literal genuinely continuing across physical
                        // lines from ordinary code wrapping, so it
                        // silently collapsed a real embedded newline
                        // inside a where-clause macro literal into a
                        // single space.
                        append_code_chunk(
                            &mut suffix,
                            &mut suffix_code,
                            &rendered,
                            &rendered_code,
                            was_escaped,
                            continues_literal,
                        );
                    }

                    let tuple_fields = normalized_public_surface_str(
                        path,
                        &format!("pub struct __Tuple {{\n{tuple_body}\n}}"),
                    );
                    let mut public_fields = tuple_fields
                        .lines()
                        .skip(2)
                        .filter(|line| !line.is_empty())
                        .map(str::to_owned)
                        .collect::<Vec<_>>();
                    if let Some(last) = public_fields.last_mut() {
                        last.truncate(last.trim_end().trim_end_matches(',').len());
                    }
                    let public_fields = public_fields.join(" ");
                    let suffix = emit_captured_fragment(&suffix).unwrap_or_default();
                    let separator = if suffix.starts_with(';') { "" } else { " " };
                    lines.push(format!("{header}{public_fields}){separator}{suffix}"));
                    if item_lexer.state == LexState::Normal {
                        item_lexer.reset_top_level_depths();
                    }
                    file_lexer = item_lexer;
                    if same_line_remainder.is_none() {
                        idx += 1;
                    }
                    continue;
                }

                // Braced struct / union
                let mut struct_header = if current_scanned.has_top_level_open_brace {
                    let up_to_brace = current_scanned.text_up_to_function_body_open_brace();
                    let rendered_last = current_scanned.render_visible();
                    if prefix_text.ends_with(&rendered_last) {
                        let prefix_head = &prefix_text[..prefix_text.len() - rendered_last.len()];
                        format!("{prefix_head}{up_to_brace}")
                    } else {
                        up_to_brace
                    }
                } else {
                    prefix_text
                };

                let mut body_open = current_scanned.has_top_level_open_brace;

                while !body_open && idx + 1 < src_lines.len() {
                    idx += 1;
                    let continuation = src_lines[idx];
                    if continuation.trim().is_empty() && item_lexer.state == LexState::Normal {
                        continue;
                    }
                    let was_escaped = item_lexer.state.is_escaped_continuation();
                    let continues_literal = item_lexer.state.is_in_string_literal();
                    let sc = scan_logical_item_line(
                        &mut item_lexer,
                        continuation,
                        &mut same_line_remainder,
                    );
                    let rendered = sc.render_visible();
                    body_open = sc.has_top_level_open_brace;
                    if was_escaped {
                        struct_header.push_str(&rendered);
                        continue;
                    }
                    if continues_literal {
                        struct_header.push('\n');
                        struct_header.push_str(&rendered);
                        continue;
                    }
                    if rendered.trim().is_empty() {
                        continue;
                    }
                    if !struct_header.ends_with(' ') && !rendered.starts_with(' ') {
                        struct_header.push(' ');
                    }
                    struct_header.push_str(&rendered);
                }

                lines.push(struct_header);

                let mut pending_field_attrs = Vec::new();
                let mut cur_field_text = String::new();
                let mut cur_field_code = String::new();
                let mut struct_body_closed = false;

                // Process remainder of opening line (if any)
                if let Some(open_brace_pos) = current_scanned.first_top_level_open_brace_seg {
                    if let Some(start_pos) = next_pos(&current_scanned.segments, open_brace_pos) {
                        let mut cur_pos = Some(start_pos);
                        for event in &current_scanned.events {
                            if event.kind == StructuralEventKind::TopLevelOpenBrace {
                                continue;
                            }
                            if event.seg_idx < start_pos.0
                                || (event.seg_idx == start_pos.0 && event.char_idx < start_pos.1)
                            {
                                continue;
                            }

                            if event.kind == StructuralEventKind::TopLevelCloseBrace {
                                let (chunk_text, chunk_code) = if let Some(limit) = prev_pos(
                                    &current_scanned.segments,
                                    (event.seg_idx, event.char_idx),
                                ) {
                                    (
                                        current_scanned.render_visible_range(cur_pos, Some(limit)),
                                        current_scanned.code_tokens_range(cur_pos, Some(limit)),
                                    )
                                } else {
                                    (String::new(), String::new())
                                };
                                append_code_chunk(
                                    &mut cur_field_text,
                                    &mut cur_field_code,
                                    &chunk_text,
                                    &chunk_code,
                                    false,
                                    false,
                                );
                                let trimmed_text =
                                    emit_captured_fragment(&cur_field_text).unwrap_or_default();
                                if !trimmed_text.is_empty() && is_public_code(&cur_field_code) {
                                    lines.append(&mut pending_field_attrs);
                                    lines.push(trimmed_text);
                                } else {
                                    pending_field_attrs.clear();
                                }
                                cur_field_text.clear();
                                cur_field_code.clear();
                                struct_body_closed = true;
                                cur_pos = next_pos(
                                    &current_scanned.segments,
                                    (event.seg_idx, event.char_idx),
                                );
                                break;
                            }

                            let chunk_text = current_scanned.render_visible_range(
                                cur_pos,
                                Some((event.seg_idx, event.char_idx)),
                            );
                            let chunk_code = current_scanned
                                .code_tokens_range(cur_pos, Some((event.seg_idx, event.char_idx)));
                            append_code_chunk(
                                &mut cur_field_text,
                                &mut cur_field_code,
                                &chunk_text,
                                &chunk_code,
                                false,
                                false,
                            );

                            if event.kind == StructuralEventKind::BaselineComma {
                                let trimmed_text =
                                    emit_captured_fragment(&cur_field_text).unwrap_or_default();
                                if !trimmed_text.is_empty() && is_public_code(&cur_field_code) {
                                    lines.append(&mut pending_field_attrs);
                                    lines.push(trimmed_text);
                                } else {
                                    pending_field_attrs.clear();
                                }
                                cur_field_text.clear();
                                cur_field_code.clear();
                                cur_pos = next_pos(
                                    &current_scanned.segments,
                                    (event.seg_idx, event.char_idx),
                                );
                            }
                        }
                        if !struct_body_closed && cur_pos.is_some() {
                            let remainder_text =
                                current_scanned.render_visible_range(cur_pos, None);
                            let remainder_code = current_scanned.code_tokens_range(cur_pos, None);
                            append_code_chunk(
                                &mut cur_field_text,
                                &mut cur_field_code,
                                &remainder_text,
                                &remainder_code,
                                false,
                                false,
                            );
                        }
                    }
                }

                let mut field_remainder = None;
                while !struct_body_closed
                    && (field_remainder.is_some() || idx + 1 < src_lines.len())
                {
                    let owned_line = field_remainder.take();
                    if owned_line.is_none() {
                        idx += 1;
                    }
                    let field_line = owned_line.as_deref().unwrap_or(src_lines[idx]);
                    if field_line.trim().is_empty() && item_lexer.state == LexState::Normal {
                        continue;
                    }
                    if cur_field_text.trim().is_empty() && item_lexer.state == LexState::Normal {
                        let mut check_lexer = item_lexer.clone();
                        let check_sc = check_lexer.scan_line(field_line);
                        if is_outer_attribute_start(&check_sc.code_tokens) {
                            let captured = capture_attribute(
                                &src_lines,
                                &mut idx,
                                &mut item_lexer,
                                field_line,
                            );
                            pending_field_attrs.push(captured.text);
                            if !captured.remainder.trim().is_empty() {
                                field_remainder = Some(captured.remainder);
                            }
                            continue;
                        }
                    }

                    let was_escaped = item_lexer.state.is_escaped_continuation();
                    let continues_literal = item_lexer.state.is_in_string_literal();
                    let sc = scan_logical_item_line(
                        &mut item_lexer,
                        field_line,
                        &mut same_line_remainder,
                    );

                    let mut cur_pos = Some((0, 0));
                    for event in &sc.events {
                        if event.kind == StructuralEventKind::TopLevelCloseBrace {
                            let (chunk_text, chunk_code) = if let Some(limit) =
                                prev_pos(&sc.segments, (event.seg_idx, event.char_idx))
                            {
                                (
                                    sc.render_visible_range(cur_pos, Some(limit)),
                                    sc.code_tokens_range(cur_pos, Some(limit)),
                                )
                            } else {
                                (String::new(), String::new())
                            };
                            append_code_chunk(
                                &mut cur_field_text,
                                &mut cur_field_code,
                                &chunk_text,
                                &chunk_code,
                                was_escaped,
                                continues_literal,
                            );
                            let trimmed_text =
                                emit_captured_fragment(&cur_field_text).unwrap_or_default();
                            if !trimmed_text.is_empty() && is_public_code(&cur_field_code) {
                                lines.append(&mut pending_field_attrs);
                                lines.push(trimmed_text);
                            } else {
                                pending_field_attrs.clear();
                            }
                            cur_field_text.clear();
                            cur_field_code.clear();
                            struct_body_closed = true;
                            cur_pos = next_pos(&sc.segments, (event.seg_idx, event.char_idx));
                            break;
                        }

                        let chunk_text =
                            sc.render_visible_range(cur_pos, Some((event.seg_idx, event.char_idx)));
                        let chunk_code =
                            sc.code_tokens_range(cur_pos, Some((event.seg_idx, event.char_idx)));
                        append_code_chunk(
                            &mut cur_field_text,
                            &mut cur_field_code,
                            &chunk_text,
                            &chunk_code,
                            was_escaped,
                            continues_literal,
                        );
                        cur_pos = next_pos(&sc.segments, (event.seg_idx, event.char_idx));

                        if event.kind == StructuralEventKind::BaselineComma {
                            let trimmed_text =
                                emit_captured_fragment(&cur_field_text).unwrap_or_default();
                            if !trimmed_text.is_empty() && is_public_code(&cur_field_code) {
                                lines.append(&mut pending_field_attrs);
                                lines.push(trimmed_text);
                            } else {
                                pending_field_attrs.clear();
                            }
                            cur_field_text.clear();
                            cur_field_code.clear();
                            cur_pos = next_pos(&sc.segments, (event.seg_idx, event.char_idx));
                        }
                    }

                    if !struct_body_closed && cur_pos.is_some() {
                        let remainder_text = sc.render_visible_range(cur_pos, None);
                        let remainder_code = sc.code_tokens_range(cur_pos, None);
                        append_code_chunk(
                            &mut cur_field_text,
                            &mut cur_field_code,
                            &remainder_text,
                            &remainder_code,
                            was_escaped,
                            continues_literal,
                        );
                    }
                }

                if item_lexer.state == LexState::Normal {
                    item_lexer.reset_top_level_depths();
                }
                file_lexer = item_lexer;
                if same_line_remainder.is_none() {
                    idx += 1;
                }
                continue;
            }

            if is_public_trait(&combined_code_tokens) {
                scan_baseline_member_body(
                    &mut lines,
                    &src_lines,
                    &mut idx,
                    &mut item_lexer,
                    &mut same_line_remainder,
                    &current_scanned,
                    prefix_text,
                    true,
                    |_member_code: &str| true,
                );
                file_lexer = item_lexer.clone();
                if same_line_remainder.is_none() {
                    idx += 1;
                }
                continue;
            }

            if is_impl_start(&combined_code_tokens) {
                while !current_scanned.has_top_level_open_brace && idx + 1 < src_lines.len() {
                    idx += 1;
                    let next_line = src_lines[idx];
                    if next_line.trim().is_empty() && item_lexer.state == LexState::Normal {
                        continue;
                    }
                    let continues_literal = item_lexer.state.is_in_string_literal();
                    let sc = scan_logical_item_line(
                        &mut item_lexer,
                        next_line,
                        &mut same_line_remainder,
                    );
                    let rendered = sc.render_visible();
                    if continues_literal {
                        prefix_text.push('\n');
                        prefix_text.push_str(&rendered);
                    } else {
                        let trimmed_rendered = rendered.trim();
                        if !trimmed_rendered.is_empty() || sc.ends_in_string_literal {
                            if !prefix_text.ends_with(' ')
                                && !prefix_text.ends_with('(')
                                && !trimmed_rendered.starts_with(')')
                            {
                                prefix_text.push(' ');
                            }
                            prefix_text.push_str(trimmed_rendered);
                        }
                    }
                    let trimmed_code = sc.code_tokens.trim();
                    if !trimmed_code.is_empty() {
                        if !combined_code_tokens.ends_with(' ')
                            && !combined_code_tokens.ends_with('(')
                            && !trimmed_code.starts_with(')')
                        {
                            combined_code_tokens.push(' ');
                        }
                        combined_code_tokens.push_str(trimmed_code);
                    }
                    current_scanned = sc;
                }

                // Trait impls (`impl Trait for Type { ... }`) never carry
                // explicit visibility on their own items (Rust forbids
                // `pub` there - visibility is inherited from the trait),
                // so they can never contribute an inherent public scope;
                // consume the body structurally (so its lines are not
                // independently re-evaluated) without emitting anything
                // from it, matching the pre-existing behavior where trait
                // impls were invisible to this scanner entirely.
                let is_inherent = is_inherent_impl_header(item_lexer.header_saw_trait_for);
                scan_baseline_member_body(
                    &mut lines,
                    &src_lines,
                    &mut idx,
                    &mut item_lexer,
                    &mut same_line_remainder,
                    &current_scanned,
                    prefix_text,
                    false,
                    |member_code: &str| is_inherent && is_public_code(member_code),
                );
                file_lexer = item_lexer.clone();
                if same_line_remainder.is_none() {
                    idx += 1;
                }
                continue;
            }

            if is_extern_block_start(&combined_code_tokens) {
                while !current_scanned.has_top_level_open_brace && idx + 1 < src_lines.len() {
                    idx += 1;
                    let next_line = src_lines[idx];
                    if next_line.trim().is_empty() && item_lexer.state == LexState::Normal {
                        continue;
                    }
                    let continues_literal = item_lexer.state.is_in_string_literal();
                    let sc = scan_logical_item_line(
                        &mut item_lexer,
                        next_line,
                        &mut same_line_remainder,
                    );
                    let rendered = sc.render_visible();
                    if continues_literal {
                        prefix_text.push('\n');
                        prefix_text.push_str(&rendered);
                    } else {
                        let trimmed_rendered = rendered.trim();
                        if !trimmed_rendered.is_empty() || sc.ends_in_string_literal {
                            if !prefix_text.ends_with(' ')
                                && !prefix_text.ends_with('(')
                                && !trimmed_rendered.starts_with(')')
                            {
                                prefix_text.push(' ');
                            }
                            prefix_text.push_str(trimmed_rendered);
                        }
                    }
                    current_scanned = sc;
                }

                // Foreign items inside an extern block have no method-
                // body-skip concern (declarations only, no bodies), and
                // no trait-impl-style ambiguity - every member is either
                // explicitly visible or not, exactly like an inherent
                // impl's own members. The block header itself is not
                // part of the public declaration inventory, same as impl.
                scan_baseline_member_body(
                    &mut lines,
                    &src_lines,
                    &mut idx,
                    &mut item_lexer,
                    &mut same_line_remainder,
                    &current_scanned,
                    prefix_text,
                    false,
                    |member_code: &str| is_public_code(member_code),
                );
                file_lexer = item_lexer.clone();
                if same_line_remainder.is_none() {
                    idx += 1;
                }
                continue;
            }

            if is_private_extern_item(&combined_code_tokens) {
                // A private (non-`pub`) extern crate or extern fn,
                // resolved once `is_incomplete_extern_prefix`'s own
                // lookahead read enough to rule out a block. Consumed
                // structurally like any other non-public item - never
                // inventoried - mirroring `is_public_fn`'s own signature-
                // then-body-skip shape for the fn case (~L2664) and the
                // semicolon-accumulation shape the public extern-crate/
                // type-alias path below uses for the crate case (~L3858),
                // minus every `lines.push`: nothing is captured, only
                // `idx`/`item_lexer` are advanced past it.
                let mut probe = combined_code_tokens.trim_start();
                if let Some(after) = is_keyword_prefix(probe, "unsafe") {
                    probe = after;
                }
                if let Some(mut rest) = is_keyword_prefix(probe, "extern") {
                    rest = rest.trim_start();
                    if rest.starts_with('"') {
                        if let Some(close) = rest[1..].find('"') {
                            rest = rest[close + 2..].trim_start();
                        }
                    }
                    probe = rest;
                }
                let is_fn = is_keyword_prefix(probe, "fn").is_some();

                if is_fn {
                    while !current_scanned.has_top_level_open_brace
                        && !current_scanned.has_top_level_semicolon
                        && idx + 1 < src_lines.len()
                    {
                        idx += 1;
                        let continuation = src_lines[idx];
                        if continuation.trim().is_empty() && item_lexer.state == LexState::Normal {
                            continue;
                        }
                        current_scanned = scan_logical_item_line(
                            &mut item_lexer,
                            continuation,
                            &mut same_line_remainder,
                        );
                    }
                    if current_scanned.has_top_level_open_brace {
                        let mut body_closed = current_scanned
                            .events
                            .iter()
                            .any(|event| event.kind == StructuralEventKind::TopLevelCloseBrace);
                        while !body_closed && idx + 1 < src_lines.len() {
                            idx += 1;
                            current_scanned = scan_logical_item_line_with_terminator(
                                &mut item_lexer,
                                src_lines[idx],
                                &mut same_line_remainder,
                                Some(true),
                            );
                            body_closed = current_scanned
                                .events
                                .iter()
                                .any(|event| event.kind == StructuralEventKind::TopLevelCloseBrace);
                        }
                    }
                } else {
                    let mut is_complete = current_scanned.has_top_level_semicolon;
                    while !is_complete && idx + 1 < src_lines.len() {
                        idx += 1;
                        let continuation = src_lines[idx];
                        if continuation.trim().is_empty() && item_lexer.state == LexState::Normal {
                            continue;
                        }
                        let sc = scan_logical_item_line_with_terminator(
                            &mut item_lexer,
                            continuation,
                            &mut same_line_remainder,
                            Some(false),
                        );
                        if sc.has_top_level_semicolon {
                            is_complete = true;
                        }
                    }
                }

                if item_lexer.state == LexState::Normal {
                    item_lexer.reset_top_level_depths();
                }
                file_lexer = item_lexer.clone();
                if same_line_remainder.is_none() {
                    idx += 1;
                }
                continue;
            }

            if is_public_mod(&combined_code_tokens) {
                let mut module_decl = prefix_text;
                while !current_scanned.has_top_level_open_brace
                    && !current_scanned.has_top_level_semicolon
                    && idx + 1 < src_lines.len()
                {
                    idx += 1;
                    let continuation = src_lines[idx];
                    let sc = scan_logical_item_line(
                        &mut item_lexer,
                        continuation,
                        &mut same_line_remainder,
                    );
                    let rendered = sc.render_visible();
                    if !rendered.trim().is_empty() {
                        if !module_decl.ends_with(' ') && !rendered.starts_with(' ') {
                            module_decl.push(' ');
                        }
                        module_decl.push_str(&rendered);
                    }
                    current_scanned = sc;
                }

                if current_scanned.has_top_level_semicolon {
                    lines.push(module_decl);
                } else if let Some(open_pos) = current_scanned.first_top_level_open_brace_seg {
                    let rendered_last = current_scanned.render_visible();
                    let header = if module_decl.ends_with(&rendered_last) {
                        let prefix_head = &module_decl[..module_decl.len() - rendered_last.len()];
                        format!(
                            "{prefix_head}{}",
                            current_scanned.text_up_to_function_body_open_brace()
                        )
                    } else {
                        current_scanned.text_up_to_function_body_open_brace()
                    };
                    lines.push(header);

                    let mut module_body = String::new();
                    let mut body_start = next_pos(&current_scanned.segments, open_pos);
                    let mut opening_line = true;
                    loop {
                        let close_pos = current_scanned
                            .events
                            .iter()
                            .find(|event| {
                                event.kind == StructuralEventKind::TopLevelCloseBrace
                                    && body_start.is_none_or(|start| {
                                        (event.seg_idx, event.char_idx) >= start
                                    })
                            })
                            .map(|event| (event.seg_idx, event.char_idx));
                        let body_end =
                            close_pos.and_then(|close| prev_pos(&current_scanned.segments, close));
                        if !(opening_line && body_start.is_none())
                            && !(close_pos.is_some() && body_end.is_none())
                        {
                            module_body.push_str(
                                &current_scanned.render_visible_range(body_start, body_end),
                            );
                        }
                        if close_pos.is_some() || idx + 1 >= src_lines.len() {
                            break;
                        }
                        module_body.push('\n');
                        idx += 1;
                        current_scanned = scan_logical_item_line(
                            &mut item_lexer,
                            src_lines[idx],
                            &mut same_line_remainder,
                        );
                        body_start = Some((0, 0));
                        opening_line = false;
                    }

                    let nested_surface = normalized_public_surface_str(path, &module_body);
                    lines.extend(
                        nested_surface
                            .lines()
                            .skip(1)
                            .filter(|line| !line.is_empty())
                            .map(str::to_owned),
                    );
                }

                if item_lexer.state == LexState::Normal {
                    item_lexer.reset_top_level_depths();
                }
                file_lexer = item_lexer;
                if same_line_remainder.is_none() {
                    idx += 1;
                }
                continue;
            }

            assert!(
                is_public_type_alias(&combined_code_tokens)
                    || is_public_extern_crate(&combined_code_tokens),
                "unsupported public item kind must fail closed: {combined_code_tokens}"
            );
            let is_item_done = |sc: &ScannedLine| sc.has_top_level_semicolon;
            let mut is_complete = is_item_done(&current_scanned);

            let mut item = prefix_text;

            while !is_complete && idx + 1 < src_lines.len() {
                idx += 1;
                let continuation = src_lines[idx];
                if continuation.trim().is_empty() && item_lexer.state == LexState::Normal {
                    continue;
                }
                let was_escaped = item_lexer.state.is_escaped_continuation();
                let continues_literal = item_lexer.state.is_in_string_literal();
                let sc = scan_logical_item_line_with_terminator(
                    &mut item_lexer,
                    continuation,
                    &mut same_line_remainder,
                    Some(false),
                );
                if is_item_done(&sc) {
                    is_complete = true;
                }
                let rendered = sc.render_visible();
                if was_escaped {
                    item.push_str(&rendered);
                    continue;
                }
                if continues_literal {
                    item.push('\n');
                    item.push_str(&rendered);
                    continue;
                }
                if rendered.trim().is_empty() && !sc.ends_in_string_literal {
                    continue;
                }
                if !item.ends_with(' ') && !rendered.starts_with(' ') {
                    item.push(' ');
                }
                item.push_str(&rendered);
            }

            lines.push(item);
            if item_lexer.state == LexState::Normal {
                item_lexer.reset_top_level_depths();
            }
            file_lexer = item_lexer;
            if same_line_remainder.is_none() {
                idx += 1;
            }
            continue;
        }

        // Advance file_lexer for non-public lines
        if item_lexer.state == LexState::Normal {
            item_lexer.reset_top_level_depths();
        }
        file_lexer = item_lexer;
        pending_attrs.clear();
        if same_line_remainder.is_none() {
            idx += 1;
        }
    }

    format!(
        "source: {}\n{}\n",
        path.replace('\\', "/"),
        lines.join("\n")
    )
}

fn normalize_snapshot_text(text: &str) -> String {
    text.replace("\r\n", "\n").trim_end().to_string()
}

fn update_mode() -> bool {
    std::env::var("SM_UPDATE_PUBLIC_API_SNAPSHOTS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[test]
fn public_api_inventory_matches_checked_in_contract_snapshots() {
    for (source, snapshot) in TARGETS {
        let actual = normalized_public_surface(source).trim_end().to_string();
        if update_mode() {
            fs::write(snapshot, format!("{actual}\n"))
                .unwrap_or_else(|err| panic!("write {snapshot}: {err}"));
            continue;
        }
        let expected =
            fs::read_to_string(snapshot).unwrap_or_else(|err| panic!("read {snapshot}: {err}"));
        assert_eq!(
            actual,
            normalize_snapshot_text(&expected),
            "public API inventory drifted for {source}; update snapshot only for intentional contract changes"
        );
    }
}

fn verification_code_contract_name(code: sm_verify::VerificationCode) -> &'static str {
    use sm_verify::VerificationCode;

    match code {
        VerificationCode::BadHeader => "BadHeader",
        VerificationCode::UnsupportedVersion => "UnsupportedVersion",
        VerificationCode::TruncatedFunction => "TruncatedFunction",
        VerificationCode::InvalidFunctionName => "InvalidFunctionName",
        VerificationCode::DuplicateFunction => "DuplicateFunction",
        VerificationCode::InvalidStringTable => "InvalidStringTable",
        VerificationCode::InvalidDebugSection => "InvalidDebugSection",
        VerificationCode::InvalidOwnershipSection => "InvalidOwnershipSection",
        VerificationCode::UnknownOpcode => "UnknownOpcode",
        VerificationCode::OperandOutOfBounds => "OperandOutOfBounds",
        VerificationCode::InvalidJumpTarget => "InvalidJumpTarget",
        VerificationCode::InvalidStringReference => "InvalidStringReference",
        VerificationCode::InvalidRegisterReference => "InvalidRegisterReference",
        VerificationCode::UnknownCallTarget => "UnknownCallTarget",
        VerificationCode::ResourceLimitExceeded => "ResourceLimitExceeded",
        VerificationCode::CapabilityViolation => "CapabilityViolation",
        VerificationCode::AmbiguousInstructionFraming => "AmbiguousInstructionFraming",
        VerificationCode::OpcodeRequiresNewerHeader => "OpcodeRequiresNewerHeader",
        VerificationCode::ReachableFunctionFallthrough => "ReachableFunctionFallthrough",
        VerificationCode::InvalidSignatureSection => "InvalidSignatureSection",
        VerificationCode::CallArgumentCountMismatch => "CallArgumentCountMismatch",
        VerificationCode::UndefinedRegisterRead => "UndefinedRegisterRead",
        VerificationCode::AnalysisStateLimitExceeded => "AnalysisStateLimitExceeded",
        VerificationCode::AnalysisWorkLimitExceeded => "AnalysisWorkLimitExceeded",
        VerificationCode::InvalidOwnershipAnchor => "InvalidOwnershipAnchor",
    }
}

#[test]
fn verification_code_variants_match_public_contract() {
    use sm_verify::VerificationCode;

    let variants = [
        VerificationCode::BadHeader,
        VerificationCode::UnsupportedVersion,
        VerificationCode::TruncatedFunction,
        VerificationCode::InvalidFunctionName,
        VerificationCode::DuplicateFunction,
        VerificationCode::InvalidStringTable,
        VerificationCode::InvalidDebugSection,
        VerificationCode::InvalidOwnershipSection,
        VerificationCode::UnknownOpcode,
        VerificationCode::OperandOutOfBounds,
        VerificationCode::InvalidJumpTarget,
        VerificationCode::InvalidStringReference,
        VerificationCode::InvalidRegisterReference,
        VerificationCode::UnknownCallTarget,
        VerificationCode::ResourceLimitExceeded,
        VerificationCode::CapabilityViolation,
        VerificationCode::AmbiguousInstructionFraming,
        VerificationCode::OpcodeRequiresNewerHeader,
        VerificationCode::ReachableFunctionFallthrough,
        VerificationCode::InvalidSignatureSection,
        VerificationCode::CallArgumentCountMismatch,
        VerificationCode::UndefinedRegisterRead,
        VerificationCode::AnalysisStateLimitExceeded,
        VerificationCode::AnalysisWorkLimitExceeded,
        VerificationCode::InvalidOwnershipAnchor,
    ];

    for variant in variants {
        assert_eq!(
            verification_code_contract_name(variant),
            format!("{variant:?}")
        );
    }
}

#[test]
fn prom_refs_forbidden_impl_surface_is_absent() {
    let path = "crates/prom-refs/src/lib.rs";
    let src = std::fs::read_to_string(path).unwrap();

    // 1. Remove comments
    let mut no_comments = String::with_capacity(src.len());
    let mut in_block_comment = false;
    let mut in_line_comment = false;
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        if in_block_comment {
            if c == '*' && chars.peek() == Some(&'/') {
                chars.next();
                in_block_comment = false;
            }
        } else if in_line_comment {
            if c == '\n' {
                in_line_comment = false;
                no_comments.push('\n');
            }
        } else if c == '/' && chars.peek() == Some(&'*') {
            chars.next();
            in_block_comment = true;
        } else if c == '/' && chars.peek() == Some(&'/') {
            chars.next();
            in_line_comment = true;
        } else {
            no_comments.push(c);
        }
    }

    // 2. Normalize whitespace
    let mut normalized = no_comments
        .replace("\n", " ")
        .replace("\r", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    // 3. Canonicalize known standard trait prefixes
    let prefixes = [
        "::core::convert::",
        "core::convert::",
        "::std::convert::",
        "std::convert::",
        "::core::ops::",
        "core::ops::",
        "::std::ops::",
        "std::ops::",
        "::core::borrow::",
        "core::borrow::",
        "::std::borrow::",
        "std::borrow::",
        "::core::default::",
        "core::default::",
        "::std::default::",
        "std::default::",
    ];
    for p in prefixes {
        normalized = normalized.replace(p, "");
    }

    let approved_types = [
        "ReferenceToken",
        "CapabilityRef",
        "ActorRef",
        "SessionRef",
        "ClientRef",
        "RevisionRef",
        "EpochRef",
    ];

    let wrappers = PROM_REF_WRAPPERS;

    for t in approved_types {
        let patterns = [
            format!("impl Default for {}", t),
            format!("impl From<u64> for {}", t),
            format!("impl Into<u64> for {}", t),
            format!("impl TryFrom<u64> for {}", t),
            format!("impl TryInto<u64> for {}", t),
            format!("impl From<{}> for u64", t),
            format!("impl Into<{}> for u64", t),
            format!("impl TryFrom<{}> for u64", t),
            format!("impl TryInto<{}> for u64", t),
            format!("impl Deref for {}", t),
            format!("impl AsRef<u64> for {}", t),
            format!("impl Borrow<u64> for {}", t),
        ];

        for pattern in patterns {
            assert!(
                !normalized.contains(&pattern),
                "Forbidden impl pattern found: {}",
                pattern
            );
        }
    }

    // Cross-domain conversions
    let mut generated_pairs = 0;
    let mut checked_patterns = 0;
    for src_wrapper in wrappers {
        for dest_wrapper in wrappers {
            if src_wrapper != dest_wrapper {
                generated_pairs += 1;
                let patterns = [
                    format!("impl From<{}> for {}", src_wrapper, dest_wrapper),
                    format!("impl Into<{}> for {}", dest_wrapper, src_wrapper),
                    format!("impl TryFrom<{}> for {}", src_wrapper, dest_wrapper),
                    format!("impl TryInto<{}> for {}", dest_wrapper, src_wrapper),
                ];
                for pattern in patterns {
                    checked_patterns += 1;
                    assert!(
                        !normalized.contains(&pattern),
                        "Forbidden cross-domain conversion pattern found: {}",
                        pattern
                    );
                }
            }
        }
    }
    assert_eq!(generated_pairs, 30, "Should generate 30 cross-domain pairs");
    assert_eq!(checked_patterns, 120, "Should check 120 patterns");

    // Forbidden inherent methods
    let forbidden_methods = [
        "pub fn raw",
        "pub const fn raw",
        "pub fn from_raw",
        "pub const fn from_raw",
        "pub fn resolve",
        "pub const fn resolve",
        "pub fn validate",
        "pub const fn validate",
        "pub fn verify",
        "pub const fn verify",
        "pub fn grant",
        "pub const fn grant",
        "pub fn admit",
        "pub const fn admit",
        "pub fn register",
        "pub const fn register",
        "pub fn serialize",
        "pub const fn serialize",
        "pub fn deserialize",
        "pub const fn deserialize",
    ];
    for method in forbidden_methods {
        assert!(
            !normalized.contains(method),
            "Forbidden public method declaration: {}",
            method
        );
    }

    // Note: this scan protects explicit source-level `impl` contracts only.
    // It does not constitute full semantic or compiler-level negative proof.
    // Alias-based or macro-generated implementations remain outside the textual guard’s proof boundary.
}

fn extract_define_ref_wrapper_names(source: &str) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;

    let mut no_comments_or_strings = String::new();

    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '/' {
            // line comment
            i += 2;
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '*' {
            // block comment
            i += 2;
            let mut depth = 1;
            while i < chars.len() && depth > 0 {
                if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '*' {
                    depth += 1;
                    i += 2;
                } else if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
        } else if chars[i] == '"' {
            // string literal
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i < chars.len() {
                i += 1;
            }
        } else if chars[i] == 'r'
            && i + 1 < chars.len()
            && (chars[i + 1] == '"' || chars[i + 1] == '#')
        {
            // raw string literal
            i += 1;
            let mut hash_count = 0;
            while i < chars.len() && chars[i] == '#' {
                hash_count += 1;
                i += 1;
            }
            if i < chars.len() && chars[i] == '"' {
                i += 1;
                loop {
                    if i >= chars.len() {
                        break;
                    }
                    if chars[i] == '"' {
                        let mut closing_hashes = 0;
                        let mut j = i + 1;
                        while j < chars.len() && chars[j] == '#' && closing_hashes < hash_count {
                            closing_hashes += 1;
                            j += 1;
                        }
                        if closing_hashes == hash_count {
                            i = j;
                            break;
                        }
                    }
                    i += 1;
                }
            }
        } else if chars[i] == '\'' {
            // character literal or lifetime
            no_comments_or_strings.push(chars[i]);
            i += 1;
            while i < chars.len() {
                let c = chars[i];
                if c == '\\' {
                    no_comments_or_strings.push(chars[i]);
                    if i + 1 < chars.len() {
                        no_comments_or_strings.push(chars[i + 1]);
                        i += 2;
                    } else {
                        i += 1;
                    }
                } else if c == '\'' {
                    no_comments_or_strings.push(chars[i]);
                    i += 1;
                    break;
                } else if !c.is_alphanumeric() && c != '_' {
                    // if it's a lifetime, there's no closing quote, so break on non-identifier
                    break;
                } else {
                    no_comments_or_strings.push(chars[i]);
                    i += 1;
                }
            }
        } else {
            no_comments_or_strings.push(chars[i]);
            i += 1;
        }
    }

    let clean_chars: Vec<char> = no_comments_or_strings.chars().collect();
    let mut i = 0;
    while i < clean_chars.len() {
        if clean_chars[i].is_alphabetic() || clean_chars[i] == '_' {
            let mut ident = String::new();
            while i < clean_chars.len()
                && (clean_chars[i].is_alphanumeric() || clean_chars[i] == '_')
            {
                ident.push(clean_chars[i]);
                i += 1;
            }
            if ident == "define_ref_wrapper" {
                let mut temp_i = i;
                while temp_i < clean_chars.len() && clean_chars[temp_i].is_whitespace() {
                    temp_i += 1;
                }
                if temp_i < clean_chars.len() && clean_chars[temp_i] == '!' {
                    temp_i += 1;
                    while temp_i < clean_chars.len() && clean_chars[temp_i].is_whitespace() {
                        temp_i += 1;
                    }
                    if temp_i < clean_chars.len()
                        && (clean_chars[temp_i] == '('
                            || clean_chars[temp_i] == '['
                            || clean_chars[temp_i] == '{')
                    {
                        temp_i += 1;
                        while temp_i < clean_chars.len() && clean_chars[temp_i].is_whitespace() {
                            temp_i += 1;
                        }

                        let mut arg = String::new();
                        while temp_i < clean_chars.len()
                            && (clean_chars[temp_i].is_alphanumeric() || clean_chars[temp_i] == '_')
                        {
                            arg.push(clean_chars[temp_i]);
                            temp_i += 1;
                        }

                        let is_valid_ident = !arg.is_empty()
                            && (arg.chars().next().unwrap().is_alphabetic()
                                || arg.starts_with('_'));
                        if !is_valid_ident {
                            return Err(format!(
                                "Invalid identifier in define_ref_wrapper! invocation: '{}'",
                                arg
                            ));
                        }

                        names.push(arg);
                        i = temp_i;
                    }
                }
            }
        } else {
            i += 1;
        }
    }

    Ok(names)
}

#[test]
fn prom_refs_wrapper_invocation_extractor_ignores_non_code_occurrences() {
    let source = r##"
        macro_rules! define_ref_wrapper {
            ($name:ident, $doc:expr) => { ... }
        }

        // define_ref_wrapper!(Fake1, ...);
        /* define_ref_wrapper!(Fake2, ...); */

        let a = "define_ref_wrapper!(Fake3, ...)";
        let b = r#"define_ref_wrapper!(Fake4, ...)"#;

        let _quote = '\'';
        let _plain = 'x';

        // These must remain ignored as non-code text around literal state transitions.
        let _text = "define_ref_wrapper!(FakeStringRef, \"fake\");";

        // Synthetic cases crossing or adjacent to character literal boundaries
        let _cross1 = 'd';efine_ref_wrapper!(FakeCross1, "...");
        let _cross2 = define_ref_wrapper'!';(FakeCross2, "...");

        define_ref_wrapper!(Real1, "doc");
        define_ref_wrapper! {
            Real2,
            "doc"
        }
        define_ref_wrapper![
            BracketRef,
            "real"
        ];
    "##;

    let names = extract_define_ref_wrapper_names(source).unwrap();
    assert_eq!(names, vec!["Real1", "Real2", "BracketRef"]);
}

#[test]
fn prom_refs_wrapper_invocations_match_contract() {
    let path = "crates/prom-refs/src/lib.rs";
    let src = fs::read_to_string(path).unwrap();

    let actual_wrappers = extract_define_ref_wrapper_names(&src).unwrap();

    let mut expected: Vec<String> = PROM_REF_WRAPPERS.iter().map(|s| s.to_string()).collect();
    let mut actual = actual_wrappers.clone();

    expected.sort();
    actual.sort();

    let mut actual_unique = actual.clone();
    actual_unique.dedup();

    if actual.len() != actual_unique.len() {
        println!("expected wrappers: {:?}", expected);
        println!("actual wrappers: {:?}", actual);
        println!(
            "missing wrappers: {:?}",
            expected
                .iter()
                .filter(|w| !actual.contains(w))
                .collect::<Vec<_>>()
        );
        println!(
            "unexpected wrappers: {:?}",
            actual
                .iter()
                .filter(|w| !expected.contains(w))
                .collect::<Vec<_>>()
        );
        println!("duplicate wrappers: {:?}", actual);
        panic!("duplicate wrappers found: {:?}", actual);
    }

    if actual != expected {
        println!("expected wrappers: {:?}", expected);
        println!("actual wrappers: {:?}", actual);

        let missing: Vec<_> = expected.iter().filter(|w| !actual.contains(w)).collect();
        let unexpected: Vec<_> = actual.iter().filter(|w| !expected.contains(w)).collect();

        println!("missing wrappers: {:?}", missing);
        println!("unexpected wrappers: {:?}", unexpected);
        println!("duplicate wrappers: []");

        panic!("Wrapper contract mismatch");
    }

    assert_eq!(actual.len(), 6, "actual invocation count = 6");
    assert_eq!(actual_unique.len(), 6, "actual unique wrapper count = 6");
    assert_eq!(expected.len(), 6, "expected wrapper count = 6");
}

#[test]
fn public_api_guard_captures_enum_variants_and_detects_drift() {
    let base_enum = r#"
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Demo {
            Alpha,
            Beta,
        }
    "#;
    let surface_base = normalized_public_surface_str("demo.rs", base_enum);
    assert!(
        surface_base.contains("Alpha"),
        "surface must contain variant Alpha"
    );
    assert!(
        surface_base.contains("Beta"),
        "surface must contain variant Beta"
    );

    // 1. Variant rename (Beta -> Gamma)
    let renamed = r#"
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Demo {
            Alpha,
            Gamma,
        }
    "#;
    assert_ne!(
        surface_base,
        normalized_public_surface_str("demo.rs", renamed),
        "variant rename must alter public API surface"
    );

    // 2. Variant addition
    let added = r#"
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Demo {
            Alpha,
            Beta,
            Gamma,
        }
    "#;
    assert_ne!(
        surface_base,
        normalized_public_surface_str("demo.rs", added),
        "variant addition must alter public API surface"
    );

    // 3. Variant removal
    let removed = r#"
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Demo {
            Alpha,
        }
    "#;
    assert_ne!(
        surface_base,
        normalized_public_surface_str("demo.rs", removed),
        "variant removal must alter public API surface"
    );

    // 4. Discriminant change
    let disc_1 = r#"
        pub enum Disc {
            A = 0x01,
            B = 0x02,
        }
    "#;
    let disc_2 = r#"
        pub enum Disc {
            A = 0x01,
            B = 0x03,
        }
    "#;
    let surface_disc_1 = normalized_public_surface_str("disc.rs", disc_1);
    let surface_disc_2 = normalized_public_surface_str("disc.rs", disc_2);
    assert!(
        surface_disc_1.contains("B = 0x02"),
        "discriminant must be captured"
    );
    assert_ne!(
        surface_disc_1, surface_disc_2,
        "discriminant change must alter public API surface"
    );

    // 5. Tuple variant field change
    let tuple_1 = r#"
        pub enum Tup {
            Val(u32),
        }
    "#;
    let tuple_2 = r#"
        pub enum Tup {
            Val(u64),
        }
    "#;
    let surface_tup_1 = normalized_public_surface_str("tup.rs", tuple_1);
    let surface_tup_2 = normalized_public_surface_str("tup.rs", tuple_2);
    assert!(
        surface_tup_1.contains("Val(u32)"),
        "tuple variant field must be captured"
    );
    assert_ne!(
        surface_tup_1, surface_tup_2,
        "tuple variant field shape change must alter public API surface"
    );

    // 6. Struct variant field change
    let struct_1 = r#"
        pub enum StructVar {
            Payload {
                tag: u16,
                count: u32,
            },
        }
    "#;
    let struct_2 = r#"
        pub enum StructVar {
            Payload {
                tag: u16,
                count: u64,
            },
        }
    "#;
    let surface_struct_1 = normalized_public_surface_str("struct_var.rs", struct_1);
    let surface_struct_2 = normalized_public_surface_str("struct_var.rs", struct_2);
    assert!(
        surface_struct_1.contains("count: u32"),
        "struct variant field must be captured"
    );
    assert_ne!(
        surface_struct_1, surface_struct_2,
        "struct variant field shape change must alter public API surface"
    );
}

#[test]
fn public_api_guard_captures_multiline_const_and_detects_value_drift() {
    let const_1 = r#"
        pub const HEADER_DEMO: Spec = Spec {
            epoch: 0,
            rev: 1,
            capabilities: CAP_A | CAP_B,
        };
    "#;
    let const_2 = r#"
        pub const HEADER_DEMO: Spec = Spec {
            epoch: 0,
            rev: 2,
            capabilities: CAP_A | CAP_B,
        };
    "#;
    let surface_const_1 = normalized_public_surface_str("const.rs", const_1);
    let surface_const_2 = normalized_public_surface_str("const.rs", const_2);
    assert!(
        surface_const_1.contains("rev: 1"),
        "const struct literal field value must be captured"
    );
    assert_ne!(
        surface_const_1, surface_const_2,
        "const struct literal field value drift must alter public API surface"
    );

    let arr_1 = r#"
        pub const BYTES: [u8; 3] = [
            1,
            2,
            3,
        ];
    "#;
    let arr_2 = r#"
        pub const BYTES: [u8; 3] = [
            1,
            9,
            3,
        ];
    "#;
    let surface_arr_1 = normalized_public_surface_str("arr.rs", arr_1);
    let surface_arr_2 = normalized_public_surface_str("arr.rs", arr_2);
    assert!(
        surface_arr_1.contains("2"),
        "const array element must be captured"
    );
    assert_ne!(
        surface_arr_1, surface_arr_2,
        "const array element drift must alter public API surface"
    );
}

#[test]
fn public_api_guard_does_not_treat_array_length_semicolon_as_function_terminator() {
    let returns_u32 = r#"
pub fn f(
    x: [u8; 3]
) -> u32
{
    private_impl_a()
}
"#;
    let returns_u64 = returns_u32.replace("-> u32", "-> u64");
    let different_private_body = returns_u32.replace("private_impl_a", "private_impl_b");

    let surface_u32 = normalized_public_surface_str("test.rs", returns_u32);
    let surface_u64 = normalized_public_surface_str("test.rs", &returns_u64);
    let surface_different_body = normalized_public_surface_str("test.rs", &different_private_body);

    assert_ne!(
        surface_u32, surface_u64,
        "return-type drift after an array-length semicolon must alter the public surface"
    );
    assert_eq!(
        surface_u32, surface_different_body,
        "private body drift must remain excluded from the public surface"
    );
}

#[test]
fn public_api_guard_does_not_treat_array_length_semicolon_as_const_terminator() {
    let value_3 = r#"
pub const X: [u8; 3] =
[
    1,
    2,
    3,
];
"#;
    let value_4 = value_3.replace("    3,", "    4,");

    assert_ne!(
        normalized_public_surface_str("test.rs", value_3),
        normalized_public_surface_str("test.rs", &value_4),
        "initializer drift after an array-type semicolon must alter the public surface"
    );
}

#[test]
fn public_api_guard_ignores_const_generic_braces_before_enum_body() {
    let variant_b = r#"
pub enum E<const N: usize = { 1 }>
{
    A,
    B,
}
"#;
    let renamed_variant = variant_b.replace("    A,", "    RenamedA,");
    let reformatted_const = variant_b.replace("{ 1 }", "{   1   }");

    assert_ne!(
        normalized_public_surface_str("test.rs", variant_b),
        normalized_public_surface_str("test.rs", &renamed_variant),
        "an enum body after a const-generic block must remain contract-bearing"
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", variant_b),
        normalized_public_surface_str("test.rs", &reformatted_const),
        "formatting-only whitespace inside const-generic code must remain normalized"
    );
}

#[test]
fn public_api_guard_ignores_nested_const_blocks_in_generic_public_items() {
    let u32_alias = r#"
pub type X = SomeType<
    { 1 },
    u32,
>;
"#;
    let u64_alias = u32_alias.replace("    u32,", "    u64,");

    assert_ne!(
        normalized_public_surface_str("test.rs", u32_alias),
        normalized_public_surface_str("test.rs", &u64_alias),
        "a nested const block must not hide a later type-alias component"
    );
}

#[test]
fn public_api_guard_does_not_overcapture_function_implementation_bodies() {
    let fn_impl_a = r#"
        pub fn compute_hash(seed: u64) -> u64 {
            let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
            state ^= state >> 30;
            state
        }
    "#;
    let fn_impl_b = r#"
        pub fn compute_hash(seed: u64) -> u64 {
            // completely different private implementation
            let state = seed.rotate_left(13);
            internal_helper(state);
            state
        }
    "#;
    let surface_a = normalized_public_surface_str("fn_demo.rs", fn_impl_a);
    let surface_b = normalized_public_surface_str("fn_demo.rs", fn_impl_b);

    assert_eq!(
        surface_a, surface_b,
        "changing private function body must not alter public API inventory"
    );
    assert!(
        !surface_a.contains("wrapping_add"),
        "function body internals must not be in public API inventory"
    );
    assert!(
        !surface_b.contains("rotate_left"),
        "function body internals must not be in public API inventory"
    );
}

#[test]
fn public_api_guard_handles_complex_lexical_literals_and_comments() {
    // A. Raw string const with braces and delimiters inside
    let src_a = r##"
        pub const X: &str = r#"{ not a Rust block }"#;
        pub const NEXT_A: u32 = 10;
    "##;
    let surface_a = normalized_public_surface_str("case_a.rs", src_a);
    assert!(
        surface_a.contains(r##"pub const X: &str = r#"{ not a Rust block }"#;"##),
        "raw string literal const must be captured: {surface_a}"
    );
    assert!(
        surface_a.contains("pub const NEXT_A: u32 = 10;"),
        "subsequent public const must be recognized: {surface_a}"
    );

    // B. Raw string with unmatched-looking delimiter
    let src_b = r##"
        pub const X: &str = r#"}"#;
        pub const NEXT_B: u32 = 20;
    "##;
    let surface_b = normalized_public_surface_str("case_b.rs", src_b);
    assert!(
        surface_b.contains(r##"pub const X: &str = r#"}"#;"##),
        "raw string with unmatched brace must be captured: {surface_b}"
    );
    assert!(
        surface_b.contains("pub const NEXT_B: u32 = 20;"),
        "subsequent public const must be recognized: {surface_b}"
    );

    // C. Higher-hash raw string
    let src_c = r###"
        pub const X: &str = r##"{ \" }"##;
        pub const NEXT_C: u32 = 30;
    "###;
    let surface_c = normalized_public_surface_str("case_c.rs", src_c);
    assert!(
        surface_c.contains(r###"pub const X: &str = r##"{ \" }"##;"###),
        "higher-hash raw string must be captured: {surface_c}"
    );
    assert!(
        surface_c.contains("pub const NEXT_C: u32 = 30;"),
        "subsequent public const must be recognized: {surface_c}"
    );

    // D. Raw byte string
    let src_d = r##"
        pub const X: &[u8] = br#"{ }"#;
        pub const NEXT_D: u32 = 40;
    "##;
    let surface_d = normalized_public_surface_str("case_d.rs", src_d);
    assert!(
        surface_d.contains(r##"pub const X: &[u8] = br#"{ }"#;"##),
        "raw byte string must be captured: {surface_d}"
    );
    assert!(
        surface_d.contains("pub const NEXT_D: u32 = 40;"),
        "subsequent public const must be recognized: {surface_d}"
    );

    // E. Block comment inside a public const
    let src_e = r#"
        pub const X: Spec = Spec {
            /* fake } delimiter */
            value: 1,
        };
        pub const NEXT_E: u32 = 50;
    "#;
    let surface_e = normalized_public_surface_str("case_e.rs", src_e);
    assert!(
        surface_e.contains("pub const X: Spec = Spec { value: 1, };"),
        "const with inline block comment must be captured: {surface_e}"
    );
    assert!(
        surface_e.contains("pub const NEXT_E: u32 = 50;"),
        "subsequent public const must be recognized: {surface_e}"
    );

    // F. Multi-line block comment
    let src_f = r#"
        pub const X: Spec = Spec {
            /*
               fake }
               fake {
            */
            value: 1,
        };
        pub const NEXT_F: u32 = 60;
    "#;
    let surface_f = normalized_public_surface_str("case_f.rs", src_f);
    assert!(
        surface_f.contains("pub const X: Spec = Spec { value: 1, };"),
        "const with multi-line block comment must be captured: {surface_f}"
    );
    assert!(
        surface_f.contains("pub const NEXT_F: u32 = 60;"),
        "subsequent public const must be recognized: {surface_f}"
    );

    // G. Public enum with raw strings and block comments
    let src_g = r#"
        pub enum LexEnum {
            Alpha,
            /*
               fake }
            */
            Beta,
        }
        pub const NEXT_G: u32 = 70;
    "#;
    let surface_g = normalized_public_surface_str("case_g.rs", src_g);
    assert!(
        surface_g.contains("Alpha"),
        "enum variant Alpha must be captured: {surface_g}"
    );
    assert!(
        surface_g.contains("Beta"),
        "enum variant Beta must be captured: {surface_g}"
    );
    assert!(
        surface_g.contains("pub const NEXT_G: u32 = 70;"),
        "subsequent public const after enum must be recognized: {surface_g}"
    );
}

#[test]
fn public_api_guard_captures_multiline_string_with_semicolons_and_raw_strings() {
    let old_src = r#"
pub const TEXT: &str = "first;
SECOND LINE";
pub const NEXT: u32 = 1;
"#;
    let new_src = r#"
pub const TEXT: &str = "first;
CHANGED PUBLIC VALUE";
pub const NEXT: u32 = 1;
"#;
    let old_surface = normalized_public_surface_str("test.rs", old_src);
    let new_surface = normalized_public_surface_str("test.rs", new_src);
    assert_ne!(
        old_surface, new_surface,
        "changing multiline string continuation after literal semicolon must change surface"
    );
    assert!(
        old_surface.contains("pub const TEXT: &str = \"first;\nSECOND LINE\";"),
        "full multiline string const must be captured: {old_surface}"
    );
    assert!(
        old_surface.contains("pub const NEXT: u32 = 1;"),
        "NEXT must remain separate public declaration: {old_surface}"
    );

    let old_raw = r##"
pub const RAW: &str = r#"first;
{ fake structural delimiter }
SECOND LINE"#;

pub const AFTER_RAW: u32 = 2;
"##;
    let new_raw = r##"
pub const RAW: &str = r#"first;
{ fake structural delimiter }
CHANGED PUBLIC VALUE"#;

pub const AFTER_RAW: u32 = 2;
"##;
    let old_raw_surface = normalized_public_surface_str("test.rs", old_raw);
    let new_raw_surface = normalized_public_surface_str("test.rs", new_raw);
    assert_ne!(
        old_raw_surface, new_raw_surface,
        "changing multiline raw string continuation after literal semicolon and fake braces must change surface"
    );
    assert!(
        old_raw_surface.contains("pub const AFTER_RAW: u32 = 2;"),
        "AFTER_RAW must remain separate public declaration: {old_raw_surface}"
    );
}

#[test]
fn public_api_guard_handles_char_and_byte_char_delimiters_without_depth_leak() {
    let src = r#"
pub const OPEN: char = '{';
pub const CLOSE: char = '}';
pub const BYTE_OPEN: u8 = b'{';
pub const BYTE_CLOSE: u8 = b'}';
pub const ESC_QUOTE: char = '\'';
pub const ESC_SLASH: char = '\\';
pub const ESC_NEWLINE: char = '\n';
pub const BYTE_ESC_QUOTE: u8 = b'\'';
pub const NEXT: u32 = 3;
"#;
    let surface = normalized_public_surface_str("test.rs", src);
    assert!(
        surface.contains("pub const OPEN: char = '{';"),
        "OPEN captured: {surface}"
    );
    assert!(
        surface.contains("pub const CLOSE: char = '}';"),
        "CLOSE captured: {surface}"
    );
    assert!(
        surface.contains("pub const BYTE_OPEN: u8 = b'{';"),
        "BYTE_OPEN captured: {surface}"
    );
    assert!(
        surface.contains("pub const BYTE_CLOSE: u8 = b'}';"),
        "BYTE_CLOSE captured: {surface}"
    );
    assert!(
        surface.contains(r"pub const ESC_QUOTE: char = '\'';"),
        "ESC_QUOTE captured: {surface}"
    );
    assert!(
        surface.contains(r"pub const ESC_SLASH: char = '\\';"),
        "ESC_SLASH captured: {surface}"
    );
    assert!(
        surface.contains(r"pub const ESC_NEWLINE: char = '\n';"),
        "ESC_NEWLINE captured: {surface}"
    );
    assert!(
        surface.contains(r"pub const BYTE_ESC_QUOTE: u8 = b'\'';"),
        "BYTE_ESC_QUOTE captured: {surface}"
    );
    assert!(
        surface.contains("pub const NEXT: u32 = 3;"),
        "NEXT captured: {surface}"
    );

    let changed_src = r#"
pub const OPEN: char = 'x';
pub const CLOSE: char = '}';
pub const BYTE_OPEN: u8 = b'{';
pub const BYTE_CLOSE: u8 = b'}';
pub const ESC_QUOTE: char = '\'';
pub const ESC_SLASH: char = '\\';
pub const ESC_NEWLINE: char = '\n';
pub const BYTE_ESC_QUOTE: u8 = b'\'';
pub const NEXT: u32 = 3;
"#;
    let changed_surface = normalized_public_surface_str("test.rs", changed_src);
    assert_ne!(
        surface, changed_surface,
        "changing char value must change surface"
    );
}

#[test]
fn public_api_guard_handles_single_line_enum_without_swallowing_function_body() {
    let src1 = r#"
pub enum State { N, F, T, S }

pub fn value() -> u32 {
    private_a()
}
"#;
    let src2 = r#"
pub enum State { N, F, T, S }

pub fn value() -> u32 {
    completely_different_private_body()
}
"#;
    let surface1 = normalized_public_surface_str("test.rs", src1);
    let surface2 = normalized_public_surface_str("test.rs", src2);
    assert_eq!(
        surface1, surface2,
        "single-line enum followed by function must not capture private function body: {surface1} vs {surface2}"
    );
    assert!(
        surface1.contains("State")
            && surface1.contains("N")
            && surface1.contains("F")
            && surface1.contains("T")
            && surface1.contains("S"),
        "enum variants must be captured: {surface1}"
    );
    assert!(
        surface1.contains("pub fn value() -> u32"),
        "function signature must be captured: {surface1}"
    );
    assert!(
        !surface1.contains("private_a"),
        "private body must not be captured: {surface1}"
    );
    assert!(
        !surface1.contains("completely_different_private_body"),
        "private body must not be captured: {surface1}"
    );
}

#[test]
fn public_api_guard_does_not_misclassify_keywords_inside_string_literals() {
    let src = r#"
pub const ENUM_TEXT: &str = " enum ";
pub const FN_TEXT: &str = " fn ";
pub const STRUCT_TEXT: &str = " struct ";
pub const NEXT: u32 = 4;
"#;
    let surface = normalized_public_surface_str("test.rs", src);
    assert!(
        surface.contains("pub const ENUM_TEXT: &str = \" enum \";"),
        "ENUM_TEXT captured: {surface}"
    );
    assert!(
        surface.contains("pub const FN_TEXT: &str = \" fn \";"),
        "FN_TEXT captured: {surface}"
    );
    assert!(
        surface.contains("pub const STRUCT_TEXT: &str = \" struct \";"),
        "STRUCT_TEXT captured: {surface}"
    );
    assert!(
        surface.contains("pub const NEXT: u32 = 4;"),
        "NEXT captured: {surface}"
    );
}

#[test]
fn public_api_guard_mutation_and_false_pass_matrix() {
    // A. Enum variant change -> MUST change surface
    let old_a = "pub enum E { A, B }";
    let new_a = "pub enum E { A, C }";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_a),
        normalized_public_surface_str("t.rs", new_a),
        "A. enum variant change must change surface"
    );

    // B. Enum discriminant change -> MUST change surface
    let old_b = "pub enum E { A = 1, B = 2 }";
    let new_b = "pub enum E { A = 1, B = 3 }";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_b),
        normalized_public_surface_str("t.rs", new_b),
        "B. enum discriminant change must change surface"
    );

    // C. Tuple variant payload type change -> MUST change surface
    let old_c = "pub enum E { A(u32) }";
    let new_c = "pub enum E { A(u64) }";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_c),
        normalized_public_surface_str("t.rs", new_c),
        "C. tuple variant payload type change must change surface"
    );

    // D. Struct variant field type change -> MUST change surface
    let old_d = "pub enum E { A { x: u32 } }";
    let new_d = "pub enum E { A { x: u64 } }";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_d),
        normalized_public_surface_str("t.rs", new_d),
        "D. struct variant field type change must change surface"
    );

    // E. HEADER-style const field value change -> MUST change surface
    let old_e = "pub const HEADER: Header = Header { magic: [1, 2, 3], version: 1 };";
    let new_e = "pub const HEADER: Header = Header { magic: [1, 2, 4], version: 1 };";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_e),
        normalized_public_surface_str("t.rs", new_e),
        "E. HEADER-style const field value change must change surface"
    );

    // F. Multiline string continuation change -> MUST change surface
    let old_f = "pub const S: &str = \"hello\nworld\";";
    let new_f = "pub const S: &str = \"hello\nmodified\";";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_f),
        normalized_public_surface_str("t.rs", new_f),
        "F. multiline string continuation change must change surface"
    );

    // G. Multiline raw-string continuation change -> MUST change surface
    let old_g = "pub const R: &str = r#\"hello\nworld\"#;";
    let new_g = "pub const R: &str = r#\"hello\nmodified\"#;";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_g),
        normalized_public_surface_str("t.rs", new_g),
        "G. multiline raw-string continuation change must change surface"
    );

    // H. Char literal value change -> MUST change surface
    let old_h = "pub const C: char = 'a';";
    let new_h = "pub const C: char = 'b';";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_h),
        normalized_public_surface_str("t.rs", new_h),
        "H. char literal value change must change surface"
    );

    // I. Private function body only change -> MUST NOT change surface
    let old_i = "pub fn f() -> u32 { private_impl_1(); 42 }";
    let new_i = "pub fn f() -> u32 { private_impl_2(); 99 }";
    assert_eq!(
        normalized_public_surface_str("t.rs", old_i),
        normalized_public_surface_str("t.rs", new_i),
        "I. private function body change must not change surface"
    );

    // J. Comments only change -> MUST NOT change surface
    let old_j = "// comment 1\npub const X: u32 = 1; /* inline comment */";
    let new_j = "// different comment\npub const X: u32 = 1; /* another comment */";
    assert_eq!(
        normalized_public_surface_str("t.rs", old_j),
        normalized_public_surface_str("t.rs", new_j),
        "J. comments change must not change surface"
    );

    // K. Whitespace-only formatting change -> MUST NOT change surface
    let old_k = "    pub   const   X:   u32   =   1;\n";
    let new_k = "pub const X: u32 = 1;\n";
    assert_eq!(
        normalized_public_surface_str("t.rs", old_k),
        normalized_public_surface_str("t.rs", new_k),
        "K. whitespace formatting change must not change surface"
    );

    // L. String literal double-space -> single-space: MUST DETECT
    let old_l = "pub const S: &str = \"a  b\";";
    let new_l = "pub const S: &str = \"a b\";";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_l),
        normalized_public_surface_str("t.rs", new_l),
        "L. string literal double-space vs single-space change must be detected"
    );

    // M. Multiline newline -> space: MUST DETECT
    let old_m = "pub const S: &str = \"a\nb\";";
    let new_m = "pub const S: &str = \"a b\";";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_m),
        normalized_public_surface_str("t.rs", new_m),
        "M. multiline literal newline vs space change must be detected"
    );

    // N. Raw-string indentation: MUST DETECT
    let old_n = "pub const R: &str = r#\"a\n  b\"#;";
    let new_n = "pub const R: &str = r#\"a\n b\"#;";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_n),
        normalized_public_surface_str("t.rs", new_n),
        "N. raw-string indentation change must be detected"
    );

    // O. Multiline pub-use member: MUST DETECT
    let old_o = "pub use demo::{\n    Alpha,\n    Beta,\n};";
    let new_o = "pub use demo::{\n    Alpha,\n    Gamma,\n};";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_o),
        normalized_public_surface_str("t.rs", new_o),
        "O. multiline pub-use member change must be detected"
    );

    // P. Const-generic signature value: MUST DETECT
    let old_p = "pub fn build() -> Foo<{ 32 }> {\n    private_a()\n}";
    let new_p = "pub fn build() -> Foo<{ 64 }> {\n    private_a()\n}";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_p),
        normalized_public_surface_str("t.rs", new_p),
        "P. const-generic signature value change must be detected"
    );

    // Q. Block comment token separator: declaration MUST remain inventoried
    let old_q = "pub/* comment */const X: u32 = 1;";
    let new_q = "pub/* comment */const X: u32 = 2;";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_q),
        normalized_public_surface_str("t.rs", new_q),
        "Q. block comment token separator declaration must detect value changes"
    );

    // R. pub(super) multiline const field: MUST DETECT
    let old_r = "pub(super) const HEADER: Spec = Spec {\n    rev: 1,\n};";
    let new_r = "pub(super) const HEADER: Spec = Spec {\n    rev: 2,\n};";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_r),
        normalized_public_surface_str("t.rs", new_r),
        "R. pub(super) multiline const field change must be detected"
    );

    // S. supported_headers ordered family mismatch: MUST DETECT
    let canonical_headers = sm_format::semcode_format::supported_headers();
    let mut reordered = canonical_headers.to_vec();
    reordered.swap(0, 1);
    assert_ne!(
        canonical_headers,
        &reordered[..],
        "S. reordering supported_headers must be detected"
    );
    let mut truncated = canonical_headers.to_vec();
    truncated.pop();
    assert_ne!(
        canonical_headers,
        &truncated[..],
        "S. removing header from supported_headers must be detected"
    );

    // T. const-generic comparison operator '>': MUST DETECT signature change
    let old_t = "pub fn build() -> Foo<{ if 1 > 0 { 32 } else { 64 } }> {\n    private_a()\n}";
    let new_t = "pub fn build() -> Foo<{ if 1 > 0 { 33 } else { 64 } }> {\n    private_a()\n}";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_t),
        normalized_public_surface_str("t.rs", new_t),
        "T. const-generic comparison operator '>' signature change must be detected"
    );
    let same_body_t =
        "pub fn build() -> Foo<{ if 1 > 0 { 32 } else { 64 } }> {\n    private_b()\n}";
    assert_eq!(
        normalized_public_surface_str("t.rs", old_t),
        normalized_public_surface_str("t.rs", same_body_t),
        "T. private body change must not change surface"
    );

    // U. const-generic shift operator '<<': MUST preserve correct function boundary
    let old_u = "pub fn build() -> Foo<{ 1 << 2 }> {\n    private_a()\n}\npub const NEXT: u32 = 1;";
    let new_u = "pub fn build() -> Foo<{ 1 << 3 }> {\n    private_a()\n}\npub const NEXT: u32 = 1;";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_u),
        normalized_public_surface_str("t.rs", new_u),
        "U. const-generic shift operator '<<' signature change must be detected"
    );
    let same_body_u =
        "pub fn build() -> Foo<{ 1 << 2 }> {\n    private_b()\n}\npub const NEXT: u32 = 1;";
    assert_eq!(
        normalized_public_surface_str("t.rs", old_u),
        normalized_public_surface_str("t.rs", same_body_u),
        "U. private body change must not change surface"
    );

    // V. combined function qualifiers: MUST classify as function
    let old_v = "pub const unsafe fn qualified() -> Foo<{ 32 }> {\n    private_a()\n}";
    let new_v = "pub const unsafe fn qualified() -> Foo<{ 64 }> {\n    private_a()\n}";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_v),
        normalized_public_surface_str("t.rs", new_v),
        "V. combined qualifiers signature change must be detected"
    );
    let same_body_v = "pub const unsafe fn qualified() -> Foo<{ 32 }> {\n    private_b()\n}";
    assert_eq!(
        normalized_public_surface_str("t.rs", old_v),
        normalized_public_surface_str("t.rs", same_body_v),
        "V. combined qualifiers private body change must not change surface"
    );

    // W. multiline cfg_attr predicate change: MUST DETECT
    let old_w =
        "#[cfg_attr(\n    feature = \"x\",\n    deprecated(note = \"old\")\n)]\npub fn api() {}";
    let new_w =
        "#[cfg_attr(\n    feature = \"y\",\n    deprecated(note = \"old\")\n)]\npub fn api() {}";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_w),
        normalized_public_surface_str("t.rs", new_w),
        "W. multiline cfg_attr predicate change must be detected"
    );

    // X. multiline attribute literal-value change: MUST DETECT
    let old_x =
        "#[cfg_attr(\n    feature = \"x\",\n    deprecated(note = \"old\")\n)]\npub fn api() {}";
    let new_x =
        "#[cfg_attr(\n    feature = \"x\",\n    deprecated(note = \"new\")\n)]\npub fn api() {}";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_x),
        normalized_public_surface_str("t.rs", new_x),
        "X. multiline attribute literal value change must be detected"
    );

    // Y. Normal multiline string trailing whitespace inside literal: MUST DETECT
    let old_y = format!("pub const S: &str = \"a{}{}\nb\";", ' ', ' ');
    let new_y = format!("pub const S: &str = \"a{}\nb\";", ' ');
    assert_ne!(
        normalized_public_surface_str("t.rs", &old_y),
        normalized_public_surface_str("t.rs", &new_y),
        "Y. normal multiline string trailing whitespace change must be detected"
    );

    // Z. Raw multiline string trailing whitespace inside literal: MUST DETECT
    let old_z = format!("pub const R: &str = r#\"a{}{}\nb\"#;", ' ', ' ');
    let new_z = format!("pub const R: &str = r#\"a{}\nb\"#;", ' ');
    assert_ne!(
        normalized_public_surface_str("t.rs", &old_z),
        normalized_public_surface_str("t.rs", &new_z),
        "Z. raw multiline string trailing whitespace change must be detected"
    );

    // AA. Literal whitespace inside struct const-generic default: MUST DETECT
    let old_aa = "pub struct S<const MSG: &'static str = \"a  b\">;";
    let new_aa = "pub struct S<const MSG: &'static str = \"a b\">;";
    assert_ne!(
        normalized_public_surface_str("t.rs", old_aa),
        normalized_public_surface_str("t.rs", new_aa),
        "AA. literal whitespace inside struct generic default must be detected"
    );
}

#[test]
fn supported_headers_match_canonical_contract() {
    use sm_format::semcode_format::*;

    let canonical_family: &[SemcodeHeaderSpec] = &[
        HEADER_V0, HEADER_V1, HEADER_V2, HEADER_V3, HEADER_V4, HEADER_V5, HEADER_V6, HEADER_V7,
        HEADER_V8, HEADER_V9, HEADER_V10, HEADER_V11, HEADER_V12, HEADER_V13, HEADER_V14,
        HEADER_V15, HEADER_V16, HEADER_V17, HEADER_V18, HEADER_V19, HEADER_V20, HEADER_V21,
    ];

    let actual = supported_headers();
    assert_eq!(
        actual, canonical_family,
        "supported_headers() must return exactly the canonical supported header family in canonical order"
    );

    for spec in actual {
        assert_eq!(
            header_spec_from_magic(&spec.magic),
            Some(*spec),
            "header_spec_from_magic must resolve canonical header for magic {:?}",
            spec.magic
        );
    }
}

#[test]
fn public_api_guard_handles_generic_angle_tokens_and_operators() {
    let fn_cmp_32 = r#"
pub fn build() -> Foo<{ if 1 > 0 { 32 } else { 64 } }> {
    private_a()
}
"#;
    let fn_cmp_33 = r#"
pub fn build() -> Foo<{ if 1 > 0 { 33 } else { 64 } }> {
    private_a()
}
"#;
    let fn_cmp_diff_body = r#"
pub fn build() -> Foo<{ if 1 > 0 { 32 } else { 64 } }> {
    private_b()
}
"#;
    let surf_cmp_32 = normalized_public_surface_str("test.rs", fn_cmp_32);
    let surf_cmp_33 = normalized_public_surface_str("test.rs", fn_cmp_33);
    let surf_cmp_diff_body = normalized_public_surface_str("test.rs", fn_cmp_diff_body);

    assert_ne!(
        surf_cmp_32, surf_cmp_33,
        "changing const-generic expression with '>' operator must alter surface"
    );
    assert_eq!(
        surf_cmp_32, surf_cmp_diff_body,
        "changing private body in function with const-generic '>' operator must not change surface"
    );
    assert!(
        !surf_cmp_32.contains("private_a"),
        "private body must not be captured: {surf_cmp_32}"
    );

    let fn_shift_1 = r#"
pub fn build() -> Foo<{ 1 << 2 }> {
    private_a()
}
pub const NEXT: u32 = 1;
"#;
    let fn_shift_2 = r#"
pub fn build() -> Foo<{ 1 << 3 }> {
    private_a()
}
pub const NEXT: u32 = 1;
"#;
    let fn_shift_diff_body = r#"
pub fn build() -> Foo<{ 1 << 2 }> {
    private_b()
}
pub const NEXT: u32 = 1;
"#;
    let surf_shift_1 = normalized_public_surface_str("test.rs", fn_shift_1);
    let surf_shift_2 = normalized_public_surface_str("test.rs", fn_shift_2);
    let surf_shift_diff_body = normalized_public_surface_str("test.rs", fn_shift_diff_body);

    assert_ne!(
        surf_shift_1, surf_shift_2,
        "changing const-generic expression with '<<' shift operator must alter surface"
    );
    assert_eq!(
        surf_shift_1, surf_shift_diff_body,
        "changing private body in function with '<<' shift operator must not change surface"
    );
    assert!(
        surf_shift_1.contains("pub const NEXT: u32 = 1;"),
        "following declaration must remain separate: {surf_shift_1}"
    );

    let fn_less = r#"
pub fn build() -> Foo<{ if 1 < 2 { 32 } else { 64 } }> {
    private_a()
}
"#;
    let surf_less = normalized_public_surface_str("test.rs", fn_less);
    assert!(
        surf_less.contains("pub fn build() -> Foo<{ if 1 < 2 { 32 } else { 64 } }> {"),
        "signature with '<' inside const block must be captured: {surf_less}"
    );
    assert!(
        !surf_less.contains("private_a"),
        "private body must not be captured: {surf_less}"
    );
}

#[test]
fn public_api_guard_treats_const_initializer_comparisons_as_top_level() {
    let private_a =
        "pub const LESS: bool = 1 < 2;\nfn hidden() -> u32 { 1 }\npub const NEXT: u32 = 7;";
    let private_b =
        "pub const LESS: bool = 1 < 2;\nfn hidden() -> u64 { 2 }\npub const NEXT: u32 = 7;";
    let changed =
        "pub const LESS: bool = 1 > 2;\nfn hidden() -> u32 { 1 }\npub const NEXT: u32 = 7;";

    let surface_a = normalized_public_surface_str("test.rs", private_a);
    let surface_b = normalized_public_surface_str("test.rs", private_b);
    assert_eq!(
        surface_a, surface_b,
        "private declarations after a const comparison must not alter the public surface"
    );
    assert_ne!(
        surface_a,
        normalized_public_surface_str("test.rs", changed),
        "changing the public const comparison must alter the public surface"
    );
    assert!(
        surface_a.contains("pub const NEXT: u32 = 7;"),
        "the public declaration after a const comparison must remain inventoried: {surface_a}"
    );
    assert!(
        !surface_a.contains("hidden"),
        "private declarations must not be captured: {surface_a}"
    );
}

#[test]
fn public_api_guard_handles_combined_function_qualifiers() {
    let fn_qual_32 = r#"
pub const unsafe fn qualified() -> Foo<{
    32
}> {
    private_a()
}
"#;
    let fn_qual_64 = r#"
pub const unsafe fn qualified() -> Foo<{
    64
}> {
    private_a()
}
"#;
    let fn_qual_diff_body = r#"
pub const unsafe fn qualified() -> Foo<{
    32
}> {
    private_b()
}
"#;
    let surf_qual_32 = normalized_public_surface_str("test.rs", fn_qual_32);
    let surf_qual_64 = normalized_public_surface_str("test.rs", fn_qual_64);
    let surf_qual_diff_body = normalized_public_surface_str("test.rs", fn_qual_diff_body);

    assert_ne!(
        surf_qual_32, surf_qual_64,
        "changing 32 -> 64 in combined qualifiers function must alter surface"
    );
    assert_eq!(
        surf_qual_32, surf_qual_diff_body,
        "changing only private_a -> private_b in combined qualifiers function must not alter surface"
    );
    assert!(
        !surf_qual_32.contains("private_a"),
        "private body must not be captured: {surf_qual_32}"
    );

    let fn_ffi = r#"
pub unsafe extern "C" fn ffi_entry() {
    private_impl()
}
"#;
    let surf_ffi = normalized_public_surface_str("test.rs", fn_ffi);
    assert!(
        surf_ffi.contains("pub unsafe extern \"C\" fn ffi_entry() {"),
        "pub unsafe extern \"C\" fn must be classified as function: {surf_ffi}"
    );
    assert!(
        !surf_ffi.contains("private_impl"),
        "private body must not be captured: {surf_ffi}"
    );
}

#[test]
fn public_api_guard_handles_multiline_outer_attributes() {
    let src_x_old = r#"
#[cfg_attr(
    feature = "x",
    deprecated(note = "old")
)]
pub fn api() {}
"#;
    let src_y_old = r#"
#[cfg_attr(
    feature = "y",
    deprecated(note = "old")
)]
pub fn api() {}
"#;
    let src_x_new = r#"
#[cfg_attr(
    feature = "x",
    deprecated(note = "new")
)]
pub fn api() {}
"#;
    let surf_x_old = normalized_public_surface_str("test.rs", src_x_old);
    let surf_y_old = normalized_public_surface_str("test.rs", src_y_old);
    let surf_x_new = normalized_public_surface_str("test.rs", src_x_new);

    assert_ne!(
        surf_x_old, surf_y_old,
        "changing attribute predicate feature x -> y must change surface"
    );
    assert_ne!(
        surf_x_old, surf_x_new,
        "changing attribute literal note old -> new must change surface"
    );

    let src_x_reformatted = r#"
#[cfg_attr(  feature = "x",   /* comment */  deprecated(note = "old")  )]
pub fn api() {}
"#;
    let surf_x_reformatted = normalized_public_surface_str("test.rs", src_x_reformatted);
    assert_eq!(
        surf_x_old, surf_x_reformatted,
        "formatting/comments-only changes to attribute must not change surface: {surf_x_old} vs {surf_x_reformatted}"
    );
}

#[test]
fn public_api_guard_preserves_newlines_inside_multiline_attribute_literals() {
    let multiline = "#[deprecated(note = \"a\nb\")]\npub fn api() {}";
    let single_line = "#[deprecated(note = \"a b\")]\npub fn api() {}";

    assert_ne!(
        normalized_public_surface_str("test.rs", multiline),
        normalized_public_surface_str("test.rs", single_line),
        "a literal newline in public attribute metadata must not normalize to a space"
    );
}

#[test]
fn public_api_guard_normalizes_multiline_enum_variant_attributes() {
    let single_line = r#"
pub enum E {
    #[cfg_attr( feature = "x", deprecated(note = "old") )]
    A,
}
"#;
    let multiline = r#"
pub enum E {
    #[cfg_attr(
        feature = "x",
        deprecated(note = "old")
    )]
    A,
}
"#;
    let changed = r#"
pub enum E {
    #[cfg_attr(
        feature = "x",
        deprecated(note = "new")
    )]
    A,
}
"#;

    assert_eq!(
        normalized_public_surface_str("test.rs", single_line),
        normalized_public_surface_str("test.rs", multiline),
        "formatting-only changes to an enum variant attribute must not alter the public surface"
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", multiline),
        normalized_public_surface_str("test.rs", changed),
        "contract changes inside a multiline enum variant attribute must alter the public surface"
    );
}

#[test]
fn public_api_guard_ignores_private_multiline_tuple_struct_fields() {
    let private_u32 = "pub struct S(\n    u32,\n);\npub const NEXT: u32 = 1;";
    let private_u64 = "pub struct S(\n    u64,\n);\npub const NEXT: u32 = 1;";

    let u32_surface = normalized_public_surface_str("test.rs", private_u32);
    let u64_surface = normalized_public_surface_str("test.rs", private_u64);
    assert_eq!(
        u32_surface, u64_surface,
        "changing a private tuple field type must not alter the public surface"
    );
    assert!(
        u32_surface.contains("pub const NEXT: u32 = 1;"),
        "the declaration after a tuple struct must remain inventoried: {u32_surface}"
    );
}

#[test]
fn public_api_guard_captures_public_multiline_tuple_struct_fields() {
    let public_u32 = "pub struct S(\n    pub u32,\n);";
    let public_u64 = "pub struct S(\n    pub u64,\n);";

    assert_ne!(
        normalized_public_surface_str("test.rs", public_u32),
        normalized_public_surface_str("test.rs", public_u64),
        "changing a public tuple field type must alter the public surface"
    );
}

#[test]
fn public_api_guard_preserves_literal_whitespace_and_raw_string_indentation() {
    let s_double_space = "pub const S: &str = \"a  b\";";
    let s_single_space = "pub const S: &str = \"a b\";";
    assert_ne!(
        normalized_public_surface_str("test.rs", s_double_space),
        normalized_public_surface_str("test.rs", s_single_space),
        "surfaces must differ for double space vs single space inside literal"
    );

    let s_multiline = "pub const S: &str = \"a\nb\";";
    assert_ne!(
        normalized_public_surface_str("test.rs", s_multiline),
        normalized_public_surface_str("test.rs", s_single_space),
        "surfaces must differ for multiline string vs single line space"
    );

    let r_indent_2 = "pub const R: &str = r#\"a\n  b\"#;";
    let r_indent_1 = "pub const R: &str = r#\"a\n b\"#;";
    assert_ne!(
        normalized_public_surface_str("test.rs", r_indent_2),
        normalized_public_surface_str("test.rs", r_indent_1),
        "surfaces must differ for raw string indentation change"
    );
}

#[test]
fn public_api_guard_captures_multiline_pub_use_reexports() {
    let old_use = r#"
pub use demo::{
    Alpha,
    Beta,
};
pub const NEXT: u32 = 10;
"#;
    let new_use = r#"
pub use demo::{
    Alpha,
    Gamma,
};
pub const NEXT: u32 = 10;
"#;
    let old_surf = normalized_public_surface_str("test.rs", old_use);
    let new_surf = normalized_public_surface_str("test.rs", new_use);
    assert_ne!(
        old_surf, new_surf,
        "multiline pub use re-export member change must alter surface"
    );
    assert!(
        old_surf.contains("Alpha") && old_surf.contains("Beta"),
        "re-exported members Alpha and Beta must be captured: {old_surf}"
    );
    assert!(
        old_surf.contains("pub const NEXT: u32 = 10;"),
        "following declaration must be inventoried: {old_surf}"
    );
}

#[test]
fn public_api_guard_handles_function_signature_const_generic_braces() {
    let fn_32 = r#"
pub fn build() -> Foo<{ 32 }> {
    private_a()
}
"#;
    let fn_64 = r#"
pub fn build() -> Foo<{ 64 }> {
    private_a()
}
"#;
    let fn_32_diff_body = r#"
pub fn build() -> Foo<{ 32 }> {
    private_b()
}
"#;
    let surf_32 = normalized_public_surface_str("test.rs", fn_32);
    let surf_64 = normalized_public_surface_str("test.rs", fn_64);
    let surf_32_diff_body = normalized_public_surface_str("test.rs", fn_32_diff_body);

    assert_ne!(
        surf_32, surf_64,
        "const-generic value change in return type must change surface"
    );
    assert_eq!(
        surf_32, surf_32_diff_body,
        "changing private body in function with const-generic return type must not change surface"
    );
    assert!(
        !surf_32.contains("private_a"),
        "private body must not be captured: {surf_32}"
    );
}

#[test]
fn public_api_guard_treats_comments_as_token_separators() {
    let src1 = "pub/* comment */const X: u32 = 1;";
    let src2 = "pub const X: u32 = 1;";
    let surf1 = normalized_public_surface_str("test.rs", src1);
    let surf2 = normalized_public_surface_str("test.rs", src2);
    assert_eq!(
        surf1, surf2,
        "comment separating pub and const must yield normalized declaration"
    );

    let src3 = "pub/* comment */const X: u32 = 2;";
    let surf3 = normalized_public_surface_str("test.rs", src3);
    assert_ne!(
        surf1, surf3,
        "value change with comment token separator must change surface"
    );

    let src_enum = "pub/*x*/ enum E { A, B }";
    let surf_enum = normalized_public_surface_str("test.rs", src_enum);
    assert!(
        surf_enum.contains("A") && surf_enum.contains("B"),
        "enum with comment separator must capture variants: {surf_enum}"
    );

    let src_crate = "pub(crate)/*x*/ const X: u32 = 1;";
    let surf_crate = normalized_public_surface_str("test.rs", src_crate);
    assert!(
        surf_crate.contains("pub(crate) const X: u32 = 1;"),
        "pub(crate) with comment separator must be captured: {surf_crate}"
    );
}

#[test]
fn public_api_guard_handles_all_pub_restricted_visibilities() {
    let src_super_1 = r#"
pub(super) const HEADER: Spec = Spec {
    rev: 1,
};
"#;
    let src_super_2 = r#"
pub(super) const HEADER: Spec = Spec {
    rev: 2,
};
"#;
    let surf_super_1 = normalized_public_surface_str("test.rs", src_super_1);
    let surf_super_2 = normalized_public_surface_str("test.rs", src_super_2);
    assert_ne!(
        surf_super_1, surf_super_2,
        "pub(super) multiline const change must be captured"
    );

    let src_in_path = "pub(in crate::module) static GLOBAL_DATA: u32 = 100;";
    let surf_in_path = normalized_public_surface_str("test.rs", src_in_path);
    assert!(
        surf_in_path.contains("pub(in crate::module) static GLOBAL_DATA: u32 = 100;"),
        "pub(in ...) restricted visibility static must be captured: {surf_in_path}"
    );
}

#[test]
fn public_api_guard_preserves_multiline_string_trailing_literal_whitespace() {
    // A. Normal multiline string trailing whitespace: 2 spaces vs 1 space before newline
    let s_two_spaces = format!("pub const S: &str = \"a{}{}\nb\";", ' ', ' ');
    let s_one_space = format!("pub const S: &str = \"a{}\nb\";", ' ');
    let surf_s_two = normalized_public_surface_str("test.rs", &s_two_spaces);
    let surf_s_one = normalized_public_surface_str("test.rs", &s_one_space);
    assert_ne!(
        surf_s_two, surf_s_one,
        "normal multiline string trailing whitespace inside literal must change surface"
    );

    // B. Raw multiline string trailing whitespace: 2 spaces vs 1 space before newline
    let r_two_spaces = format!("pub const R: &str = r#\"a{}{}\nb\"#;", ' ', ' ');
    let r_one_space = format!("pub const R: &str = r#\"a{}\nb\"#;", ' ');
    let surf_r_two = normalized_public_surface_str("test.rs", &r_two_spaces);
    let surf_r_one = normalized_public_surface_str("test.rs", &r_one_space);
    assert_ne!(
        surf_r_two, surf_r_one,
        "raw multiline string trailing whitespace inside literal must change surface"
    );
}

#[test]
fn public_api_guard_preserves_literal_whitespace_in_signatures_and_generic_items() {
    // C. Literal whitespace in public function signature
    let fn_lit_two_spaces = "pub fn build() -> Foo<\"a  b\"> {\n    private_a()\n}";
    let fn_lit_one_space = "pub fn build() -> Foo<\"a b\"> {\n    private_a()\n}";
    let fn_lit_diff_body = "pub fn build() -> Foo<\"a  b\"> {\n    private_b()\n}";
    let surf_fn_two = normalized_public_surface_str("test.rs", fn_lit_two_spaces);
    let surf_fn_one = normalized_public_surface_str("test.rs", fn_lit_one_space);
    let surf_fn_diff_body = normalized_public_surface_str("test.rs", fn_lit_diff_body);

    assert_ne!(
        surf_fn_two, surf_fn_one,
        "literal whitespace change in function signature must change surface"
    );
    assert_eq!(
        surf_fn_two, surf_fn_diff_body,
        "private body change must not change surface"
    );
    assert!(
        !surf_fn_two.contains("private_a"),
        "private body must not be captured in surface: {surf_fn_two}"
    );

    // D. Literal whitespace in struct, enum variant, and type alias
    let struct_two_spaces = "pub struct S<const MSG: &'static str = \"a  b\">;";
    let struct_one_space = "pub struct S<const MSG: &'static str = \"a b\">;";
    assert_ne!(
        normalized_public_surface_str("test.rs", struct_two_spaces),
        normalized_public_surface_str("test.rs", struct_one_space),
        "literal whitespace in struct const generic default must change surface"
    );

    let type_two_spaces = "pub type Msg = StaticMsg<\"a  b\">;";
    let type_one_space = "pub type Msg = StaticMsg<\"a b\">;";
    assert_ne!(
        normalized_public_surface_str("test.rs", type_two_spaces),
        normalized_public_surface_str("test.rs", type_one_space),
        "literal whitespace in type alias must change surface"
    );

    let enum_two_spaces = "pub enum E {\n    Variant = \"a  b\",\n}";
    let enum_one_space = "pub enum E {\n    Variant = \"a b\",\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", enum_two_spaces),
        normalized_public_surface_str("test.rs", enum_one_space),
        "literal whitespace in enum variant must change surface"
    );
}

#[test]
fn public_api_guard_preserves_literal_newlines_in_function_signatures() {
    let multiline = "pub fn f() -> type_macro!(\"a\nb\") {\n    private_a()\n}";
    let single_line = "pub fn f() -> type_macro!(\"a b\") {\n    private_a()\n}";
    let different_body = "pub fn f() -> type_macro!(\"a\nb\") {\n    private_b()\n}";

    let multiline_surface = normalized_public_surface_str("test.rs", multiline);
    assert_ne!(
        multiline_surface,
        normalized_public_surface_str("test.rs", single_line),
        "a literal newline in a public function signature must not normalize to a space"
    );
    assert_eq!(
        multiline_surface,
        normalized_public_surface_str("test.rs", different_body),
        "private function-body changes must not alter the public surface"
    );
    assert!(
        !multiline_surface.contains("private_a"),
        "private function bodies must not be captured: {multiline_surface}"
    );
}

#[test]
fn public_api_guard_normalizes_formatting_whitespace_outside_literals() {
    // E. Formatting outside literals remains normalized
    let src_spaces = "    pub   const   X:   u32   =   1;\n";
    let src_normal = "pub const X: u32 = 1;\n";
    assert_eq!(
        normalized_public_surface_str("test.rs", src_spaces),
        normalized_public_surface_str("test.rs", src_normal),
        "whitespace formatting outside literals must remain normalized"
    );

    let src_comments = "pub /* c1 */ const /* c2 */ X: u32 = 1;\n";
    assert_eq!(
        normalized_public_surface_str("test.rs", src_comments),
        normalized_public_surface_str("test.rs", src_normal),
        "comments-only differences outside literals must remain normalized"
    );
}

#[test]
fn public_api_guard_ignores_multiline_block_comments_inside_enums() {
    let src_short_comment = r#"
pub enum Mode {
    /* brief comment */
    Active,
    Inactive,
}
"#;
    let src_multiline_comment = r#"
pub enum Mode {
    /*
     * Detailed multiline
     * block comment with
     * several lines
     */
    Active,
    Inactive,
}
"#;
    let surf_short = normalized_public_surface_str("test.rs", src_short_comment);
    let surf_multiline = normalized_public_surface_str("test.rs", src_multiline_comment);
    assert_eq!(
        surf_short, surf_multiline,
        "multiline block comments inside enums must not produce spurious empty lines or alter snapshot: {surf_short} vs {surf_multiline}"
    );
}

#[test]
fn public_api_guard_handles_top_level_less_than_comparison() {
    let src_less_old = r#"
pub const LESS: bool = 1 < 2;
pub fn next_api() {
    private_a()
}
"#;
    let src_less_new = r#"
pub const LESS: bool = 1 < 3;
pub fn next_api() {
    private_a()
}
"#;
    let src_less_diff_body = r#"
pub const LESS: bool = 1 < 2;
pub fn next_api() {
    private_b()
}
"#;
    let surf_old = normalized_public_surface_str("test.rs", src_less_old);
    let surf_new = normalized_public_surface_str("test.rs", src_less_new);
    let surf_diff_body = normalized_public_surface_str("test.rs", src_less_diff_body);

    assert_ne!(
        surf_old, surf_new,
        "changing value in top-level '<' comparison const must alter surface"
    );
    assert_eq!(
        surf_old, surf_diff_body,
        "changing private body in next_api after top-level '<' const must not alter surface"
    );
    assert!(
        !surf_old.contains("private_a"),
        "private body of next_api must not be captured: {surf_old}"
    );
    assert!(
        surf_old.contains("pub const LESS: bool = 1 < 2;"),
        "const declaration must be captured intact: {surf_old}"
    );
    assert!(
        surf_old.contains("pub fn next_api() {"),
        "next function must be inventoried separately: {surf_old}"
    );
}

#[test]
fn public_api_guard_preserves_multiline_literal_newlines_in_signatures() {
    let fn_nl = "pub fn f() -> type_macro!(\"a\nb\") {\n    private_impl()\n}";
    let fn_sp = "pub fn f() -> type_macro!(\"a b\") {\n    private_impl()\n}";
    let fn_nl_diff_body = "pub fn f() -> type_macro!(\"a\nb\") {\n    different_private_impl()\n}";

    let surf_nl = normalized_public_surface_str("test.rs", fn_nl);
    let surf_sp = normalized_public_surface_str("test.rs", fn_sp);
    let surf_nl_diff_body = normalized_public_surface_str("test.rs", fn_nl_diff_body);

    assert_ne!(
        surf_nl, surf_sp,
        "multiline literal with newline in signature macro must produce different surface from space"
    );
    assert_eq!(
        surf_nl, surf_nl_diff_body,
        "private body change must not change surface"
    );
    assert!(
        !surf_nl.contains("private_impl"),
        "private body must not be captured: {surf_nl}"
    );
}

#[test]
fn public_api_guard_handles_multiline_qualifiers_on_functions() {
    let src_multiline_qual = r#"
pub unsafe extern
"C" fn f() {
    private_a()
}
"#;
    let src_diff_body = r#"
pub unsafe extern
"C" fn f() {
    private_b()
}
"#;
    let src_diff_qual = r#"
pub unsafe extern
"system" fn f() {
    private_a()
}
"#;
    let surf_base = normalized_public_surface_str("test.rs", src_multiline_qual);
    let surf_diff_body = normalized_public_surface_str("test.rs", src_diff_body);
    let surf_diff_qual = normalized_public_surface_str("test.rs", src_diff_qual);

    assert_eq!(
        surf_base, surf_diff_body,
        "private body changes in functions with multiline qualifiers must not alter surface: {surf_base} vs {surf_diff_body}"
    );
    assert_ne!(
        surf_base, surf_diff_qual,
        "qualifier changes across lines must alter public surface"
    );
    assert!(
        !surf_base.contains("private_a"),
        "private body must not be captured in multiline qualifier function: {surf_base}"
    );
    assert!(
        surf_base.contains("pub unsafe extern \"C\" fn f() {"),
        "signature must be captured properly: {surf_base}"
    );
}

#[test]
fn public_api_guard_distinguishes_braced_macros_from_function_and_item_bodies() {
    // 1. Braced macro in return type: payload change must alter surface; private body change must not
    let fn_braced_u32 = "pub fn f() -> type_macro! { u32 } {\n    private_a()\n}";
    let fn_braced_u64 = "pub fn f() -> type_macro! { u64 } {\n    private_a()\n}";
    let fn_braced_diff_body = "pub fn f() -> type_macro! { u32 } {\n    private_b()\n}";

    let surf_u32 = normalized_public_surface_str("test.rs", fn_braced_u32);
    let surf_u64 = normalized_public_surface_str("test.rs", fn_braced_u64);
    let surf_diff_body = normalized_public_surface_str("test.rs", fn_braced_diff_body);

    assert_ne!(
        surf_u32, surf_u64,
        "payload difference in braced macro must produce different public surface"
    );
    assert_eq!(
        surf_u32, surf_diff_body,
        "private body change after braced return type macro must not alter public surface"
    );
    assert!(
        !surf_u32.contains("private_a") && !surf_u32.contains("private_b"),
        "private body must not leak into surface: {surf_u32}"
    );
    assert!(
        surf_u32.contains("pub fn f() -> type_macro! { u32 } {"),
        "complete macro invocation and function header must be captured: {surf_u32}"
    );

    // 2. Path-qualified braced macro
    let fn_path_u32 = "pub fn f() -> foo::bar! { u32 } {\n    private_a()\n}";
    let fn_path_u64 = "pub fn f() -> foo::bar! { u64 } {\n    private_a()\n}";
    let surf_path_u32 = normalized_public_surface_str("test.rs", fn_path_u32);
    let surf_path_u64 = normalized_public_surface_str("test.rs", fn_path_u64);
    assert_ne!(
        surf_path_u32, surf_path_u64,
        "path-qualified braced macro payload difference must alter public surface"
    );
    assert!(
        !surf_path_u32.contains("private_a"),
        "private body must not leak: {surf_path_u32}"
    );

    // 3. Parenthesized macro form
    let fn_paren_u32 = "pub fn f() -> type_macro!(u32) {\n    private_a()\n}";
    let fn_paren_u64 = "pub fn f() -> type_macro!(u64) {\n    private_a()\n}";
    let fn_paren_diff_body = "pub fn f() -> type_macro!(u32) {\n    private_b()\n}";
    let surf_paren_u32 = normalized_public_surface_str("test.rs", fn_paren_u32);
    let surf_paren_u64 = normalized_public_surface_str("test.rs", fn_paren_u64);
    let surf_paren_diff_body = normalized_public_surface_str("test.rs", fn_paren_diff_body);
    assert_ne!(surf_paren_u32, surf_paren_u64);
    assert_eq!(surf_paren_u32, surf_paren_diff_body);
    assert!(!surf_paren_u32.contains("private_a"));

    // 4. Bracket-delimited macro form
    let fn_bracket_u32 = "pub fn f() -> type_macro![u32] {\n    private_a()\n}";
    let fn_bracket_u64 = "pub fn f() -> type_macro![u64] {\n    private_a()\n}";
    let surf_bracket_u32 = normalized_public_surface_str("test.rs", fn_bracket_u32);
    let surf_bracket_u64 = normalized_public_surface_str("test.rs", fn_bracket_u64);
    assert_ne!(surf_bracket_u32, surf_bracket_u64);
    assert!(!surf_bracket_u32.contains("private_a"));

    // 5. Existing plain function body detection
    let fn_plain = "pub fn f() -> u32 {\n    private_a()\n}";
    let surf_plain = normalized_public_surface_str("test.rs", fn_plain);
    assert!(surf_plain.contains("pub fn f() -> u32 {"));
    assert!(!surf_plain.contains("private_a"));

    // 6. Multiline qualifier + braced macro
    let fn_multi_qual_u32 =
        "pub unsafe extern\n\"C\" fn f() -> type_macro! { u32 } {\n    private_a()\n}";
    let fn_multi_qual_u64 =
        "pub unsafe extern\n\"C\" fn f() -> type_macro! { u64 } {\n    private_a()\n}";
    let fn_multi_qual_diff_qual =
        "pub unsafe extern\n\"system\" fn f() -> type_macro! { u32 } {\n    private_a()\n}";
    let fn_multi_qual_diff_body =
        "pub unsafe extern\n\"C\" fn f() -> type_macro! { u32 } {\n    private_b()\n}";

    let surf_mq_u32 = normalized_public_surface_str("test.rs", fn_multi_qual_u32);
    let surf_mq_u64 = normalized_public_surface_str("test.rs", fn_multi_qual_u64);
    let surf_mq_diff_qual = normalized_public_surface_str("test.rs", fn_multi_qual_diff_qual);
    let surf_mq_diff_body = normalized_public_surface_str("test.rs", fn_multi_qual_diff_body);

    assert_ne!(
        surf_mq_u32, surf_mq_u64,
        "payload change must alter surface"
    );
    assert_ne!(
        surf_mq_u32, surf_mq_diff_qual,
        "qualifier change must alter surface"
    );
    assert_eq!(
        surf_mq_u32, surf_mq_diff_body,
        "private body change must not alter surface"
    );
    assert!(!surf_mq_u32.contains("private_a"));

    // 7. Sibling paths: const, type alias, struct where clause
    let const_macro_u32 = "pub const X: type_macro! { u32 } = 10;";
    let const_macro_u64 = "pub const X: type_macro! { u64 } = 10;";
    let surf_c_u32 = normalized_public_surface_str("test.rs", const_macro_u32);
    let surf_c_u64 = normalized_public_surface_str("test.rs", const_macro_u64);
    assert_ne!(surf_c_u32, surf_c_u64);

    let type_macro_u32 = "pub type T = type_macro! { u32 };";
    let type_macro_u64 = "pub type T = type_macro! { u64 };";
    let surf_t_u32 = normalized_public_surface_str("test.rs", type_macro_u32);
    let surf_t_u64 = normalized_public_surface_str("test.rs", type_macro_u64);
    assert_ne!(surf_t_u32, surf_t_u64);

    let struct_where_macro =
        "pub struct Foo<T> where T: type_macro! { u32 } {\n    private_field: u32,\n}";
    let surf_s = normalized_public_surface_str("test.rs", struct_where_macro);
    assert!(surf_s.contains("pub struct Foo<T> where T: type_macro! { u32 } {"));
    assert!(!surf_s.contains("private_field"));
}

#[test]
fn public_api_guard_handles_unicode_macro_identifiers() {
    let fn_unicode_u32 = "pub fn f() -> Москва! { u32 } {\n    private_a()\n}";
    let fn_unicode_u64 = "pub fn f() -> Москва! { u64 } {\n    private_a()\n}";
    let fn_unicode_diff_body = "pub fn f() -> Москва! { u32 } {\n    private_b()\n}";

    let surf_u32 = normalized_public_surface_str("test.rs", fn_unicode_u32);
    let surf_u64 = normalized_public_surface_str("test.rs", fn_unicode_u64);
    let surf_diff_body = normalized_public_surface_str("test.rs", fn_unicode_diff_body);

    assert_ne!(
        surf_u32, surf_u64,
        "Unicode macro payload change must alter public surface: {surf_u32} vs {surf_u64}"
    );
    assert_eq!(
        surf_u32, surf_diff_body,
        "private body change after Unicode macro must not alter public surface"
    );
    assert!(
        !surf_u32.contains("private_a"),
        "private body must not leak into public surface: {surf_u32}"
    );
    assert!(
        surf_u32.contains("pub fn f() -> Москва! { u32 } {"),
        "Unicode macro and function header must be captured intact: {surf_u32}"
    );

    // Path-qualified Unicode macro
    let fn_path_u32 = "pub fn f() -> модуль::Макрос! { u32 } {\n    private_a()\n}";
    let fn_path_u64 = "pub fn f() -> модуль::Макрос! { u64 } {\n    private_a()\n}";
    let surf_path_u32 = normalized_public_surface_str("test.rs", fn_path_u32);
    let surf_path_u64 = normalized_public_surface_str("test.rs", fn_path_u64);
    assert_ne!(
        surf_path_u32, surf_path_u64,
        "path-qualified Unicode macro payload change must alter public surface"
    );
    assert!(
        !surf_path_u32.contains("private_a"),
        "private body must not leak: {surf_path_u32}"
    );
}

#[test]
fn public_api_guard_handles_trivia_separated_outer_attributes() {
    let canonical = "#[deprecated(note = \"old\")]\npub fn api() {}";
    let trivia_attr = "# /* comment */ [deprecated(note = \"old\")]\npub fn api() {}";
    let trivia_attr_new = "# /* comment */ [deprecated(note = \"new\")]\npub fn api() {}";

    let surf_canon = normalized_public_surface_str("test.rs", canonical);
    let surf_trivia = normalized_public_surface_str("test.rs", trivia_attr);
    let surf_trivia_new = normalized_public_surface_str("test.rs", trivia_attr_new);

    assert_eq!(
        surf_canon, surf_trivia,
        "trivia-separated attribute must normalize to same public contract as canonical attribute"
    );
    assert_ne!(
        surf_trivia, surf_trivia_new,
        "mutating note in trivia-separated attribute must alter public surface"
    );

    // Multiline cfg_attr with trivia
    let cfg_old = "# /*x*/ [cfg_attr(\n    feature = \"x\",\n    deprecated(note = \"old\")\n)]\npub fn api() {}";
    let cfg_new_feat = "# /*x*/ [cfg_attr(\n    feature = \"y\",\n    deprecated(note = \"old\")\n)]\npub fn api() {}";
    let cfg_new_note = "# /*x*/ [cfg_attr(\n    feature = \"x\",\n    deprecated(note = \"new\")\n)]\npub fn api() {}";

    let surf_cfg_old = normalized_public_surface_str("test.rs", cfg_old);
    let surf_cfg_new_feat = normalized_public_surface_str("test.rs", cfg_new_feat);
    let surf_cfg_new_note = normalized_public_surface_str("test.rs", cfg_new_note);

    assert_ne!(
        surf_cfg_old, surf_cfg_new_feat,
        "changing feature in cfg_attr must alter surface"
    );
    assert_ne!(
        surf_cfg_old, surf_cfg_new_note,
        "changing note in cfg_attr must alter surface"
    );

    // Enum variant attribute with trivia
    let enum_old = "pub enum E {\n    # /*x*/ [deprecated(note = \"old\")]\n    A,\n}";
    let enum_new = "pub enum E {\n    # /*x*/ [deprecated(note = \"new\")]\n    A,\n}";
    let enum_canon = "pub enum E {\n    #[deprecated(note = \"old\")]\n    A,\n}";

    let surf_e_old = normalized_public_surface_str("test.rs", enum_old);
    let surf_e_new = normalized_public_surface_str("test.rs", enum_new);
    let surf_e_canon = normalized_public_surface_str("test.rs", enum_canon);

    assert_eq!(
        surf_e_old, surf_e_canon,
        "enum variant trivia-separated attribute must match canonical"
    );
    assert_ne!(
        surf_e_old, surf_e_new,
        "mutating enum variant attribute note must alter surface"
    );
}

#[test]
fn public_api_guard_handles_trivia_separated_restricted_visibility() {
    let const_trivia_rev1 = "pub /*x*/ (crate) const HEADER: Spec = Spec {\n    rev: 1,\n};";
    let const_trivia_rev2 = "pub /*x*/ (crate) const HEADER: Spec = Spec {\n    rev: 2,\n};";
    let const_canon_rev1 = "pub(crate) const HEADER: Spec = Spec {\n    rev: 1,\n};";

    let surf_rev1 = normalized_public_surface_str("test.rs", const_trivia_rev1);
    let surf_rev2 = normalized_public_surface_str("test.rs", const_trivia_rev2);
    let surf_canon1 = normalized_public_surface_str("test.rs", const_canon_rev1);

    assert_eq!(
        surf_rev1, surf_canon1,
        "trivia-separated pub(crate) must normalize equal to canonical"
    );
    assert_ne!(
        surf_rev1, surf_rev2,
        "mutating multiline struct initializer in pub(crate) const must alter surface"
    );
    assert!(
        surf_rev1.contains("rev: 1"),
        "complete struct initializer must be captured in const: {surf_rev1}"
    );

    // pub(super) and pub(in path)
    let static_super_old = "pub /*x*/ ( super ) static GLOBAL: u32 = 100;";
    let static_super_new = "pub /*x*/ ( super ) static GLOBAL: u32 = 101;";
    let surf_super_old = normalized_public_surface_str("test.rs", static_super_old);
    let surf_super_new = normalized_public_surface_str("test.rs", static_super_new);
    assert_ne!(surf_super_old, surf_super_new);

    let static_in_path_old = "pub /*x*/ ( in crate::module ) static GLOBAL: u32 = 100;";
    let static_in_path_new = "pub /*x*/ ( in crate::module ) static GLOBAL: u32 = 101;";
    let surf_in_old = normalized_public_surface_str("test.rs", static_in_path_old);
    let surf_in_new = normalized_public_surface_str("test.rs", static_in_path_new);
    assert_ne!(surf_in_old, surf_in_new);
}

#[test]
fn public_api_guard_handles_mixed_width_unicode_macro_identifiers() {
    let fn_mixed_32 = "pub fn f() -> éx! { u32 } {\n    private_a()\n}";
    let fn_mixed_64 = "pub fn f() -> éx! { u64 } {\n    private_a()\n}";
    let fn_mixed_diff_body = "pub fn f() -> éx! { u32 } {\n    private_b()\n}";

    let surf_mixed_32 = normalized_public_surface_str("test.rs", fn_mixed_32);
    let surf_mixed_64 = normalized_public_surface_str("test.rs", fn_mixed_64);
    let surf_mixed_diff_body = normalized_public_surface_str("test.rs", fn_mixed_diff_body);

    assert_ne!(
        surf_mixed_32, surf_mixed_64,
        "payload mutation inside mixed-width Unicode macro must alter surface"
    );
    assert_eq!(
        surf_mixed_32, surf_mixed_diff_body,
        "private body change with mixed-width Unicode macro must not alter surface"
    );
    assert!(
        !surf_mixed_32.contains("private_a"),
        "private body must not be captured: {surf_mixed_32}"
    );

    // Other multi-byte / mixed-width shapes
    let fn_greek_32 = "pub fn f() -> alpha_β_gamma! { u32 } {\n    private_a()\n}";
    let fn_greek_64 = "pub fn f() -> alpha_β_gamma! { u64 } {\n    private_a()\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", fn_greek_32),
        normalized_public_surface_str("test.rs", fn_greek_64)
    );

    let fn_cjk_32 = "pub fn f() -> 名前_macro! { u32 } {\n    private_a()\n}";
    let fn_cjk_64 = "pub fn f() -> 名前_macro! { u64 } {\n    private_a()\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", fn_cjk_32),
        normalized_public_surface_str("test.rs", fn_cjk_64)
    );
}

#[test]
fn public_api_guard_handles_braced_public_items_and_member_visibility() {
    // 1. Single-line struct
    let struct_single_32 = "pub struct S { pub x: u32 }";
    let struct_single_64 = "pub struct S { pub x: u64 }";
    assert_ne!(
        normalized_public_surface_str("test.rs", struct_single_32),
        normalized_public_surface_str("test.rs", struct_single_64),
        "public struct field type change in single-line struct must alter surface"
    );

    let struct_private_32 = "pub struct S { private: u32, pub x: bool }";
    let struct_private_64 = "pub struct S { private: u64, pub x: bool }";
    assert_eq!(
        normalized_public_surface_str("test.rs", struct_private_32),
        normalized_public_surface_str("test.rs", struct_private_64),
        "private struct field type change in single-line struct must NOT alter surface"
    );

    // 2. Multiline struct
    let struct_multi_32 = "pub struct S {\n    pub x: u32,\n    private: InternalA,\n}";
    let struct_multi_64 = "pub struct S {\n    pub x: u64,\n    private: InternalA,\n}";
    let struct_multi_diff_priv = "pub struct S {\n    pub x: u32,\n    private: InternalB,\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", struct_multi_32),
        normalized_public_surface_str("test.rs", struct_multi_64),
        "public struct field type change in multiline struct must alter surface"
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", struct_multi_32),
        normalized_public_surface_str("test.rs", struct_multi_diff_priv),
        "private struct field type change in multiline struct must NOT alter surface"
    );

    // 3. Single-line and multiline trait
    let trait_req_32 = "pub trait T { fn f() -> u32; }";
    let trait_req_64 = "pub trait T { fn f() -> u64; }";
    assert_ne!(
        normalized_public_surface_str("test.rs", trait_req_32),
        normalized_public_surface_str("test.rs", trait_req_64),
        "required trait method return type change must alter surface"
    );

    let trait_def_body_a = "pub trait T { fn f() -> u32 { private_a() } }";
    let trait_def_body_b = "pub trait T { fn f() -> u32 { private_b() } }";
    let trait_def_sig_64 = "pub trait T { fn f() -> u64 { private_a() } }";
    assert_eq!(
        normalized_public_surface_str("test.rs", trait_def_body_a),
        normalized_public_surface_str("test.rs", trait_def_body_b),
        "trait default method private implementation change must NOT alter surface"
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", trait_def_body_a),
        normalized_public_surface_str("test.rs", trait_def_sig_64),
        "trait default method signature change must alter surface"
    );

    // 4. Trait associated items
    let trait_assoc_1 =
        "pub trait T {\n    type Item;\n    const N: usize;\n    fn req(&self) -> Self::Item;\n}";
    let trait_assoc_2 = "pub trait T {\n    type Item: Display;\n    const N: usize;\n    fn req(&self) -> Self::Item;\n}";
    let trait_assoc_3 =
        "pub trait T {\n    type Item;\n    const N: u32;\n    fn req(&self) -> Self::Item;\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", trait_assoc_1),
        normalized_public_surface_str("test.rs", trait_assoc_2),
        "trait associated type bound change must alter surface"
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", trait_assoc_1),
        normalized_public_surface_str("test.rs", trait_assoc_3),
        "trait associated const type change must alter surface"
    );

    // 5. Sibling union item
    let union_pub_32 = "pub union U { pub a: u32, private: u32 }";
    let union_pub_64 = "pub union U { pub a: u64, private: u32 }";
    let union_priv_diff = "pub union U { pub a: u32, private: u64 }";
    assert_ne!(
        normalized_public_surface_str("test.rs", union_pub_32),
        normalized_public_surface_str("test.rs", union_pub_64),
        "public union field type change must alter surface"
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", union_pub_32),
        normalized_public_surface_str("test.rs", union_priv_diff),
        "private union field type change must NOT alter surface"
    );
}

#[test]
fn public_api_guard_handles_escaped_newline_string_continuations() {
    let src_a = "pub const S: &str = \"a\\\n    b\";";
    let src_b = "pub const S: &str = \"a\\\n        b\";";
    let src_c = "pub const S: &str = \"a\\\n    c\";";

    assert!(src_a.contains("\\\n    "));
    assert!(src_b.contains("\\\n        "));
    assert!(src_c.contains("\\\n    "));

    let surf_a = normalized_public_surface_str("test.rs", src_a);
    let surf_b = normalized_public_surface_str("test.rs", src_b);
    let surf_c = normalized_public_surface_str("test.rs", src_c);

    assert_eq!(
        surf_a, surf_b,
        "indentation-only changes after escaped newline must produce identical public surfaces"
    );
    assert_ne!(
        surf_a, surf_c,
        "semantic payload change after escaped newline must alter public surface"
    );

    // Actual literal newline vs space must remain distinct
    let src_lit_nl = "pub const S: &str = \"a\nb\";";
    let src_lit_sp = "pub const S: &str = \"a b\";";
    assert_ne!(
        normalized_public_surface_str("test.rs", src_lit_nl),
        normalized_public_surface_str("test.rs", src_lit_sp),
        "actual literal newline must remain distinct from space"
    );

    // Raw strings must NOT normalize escaped newlines
    let src_raw_a = "pub const R: &str = r#\"a\\\n    b\"#;";
    let src_raw_b = "pub const R: &str = r#\"a\\\n        b\"#;";
    assert_ne!(
        normalized_public_surface_str("test.rs", src_raw_a),
        normalized_public_surface_str("test.rs", src_raw_b),
        "raw string indentation change must alter surface"
    );

    // Byte string continuation
    let src_byte_a = "pub const B: &[u8] = b\"a\\\n    b\";";
    let src_byte_b = "pub const B: &[u8] = b\"a\\\n        b\";";
    assert_eq!(
        normalized_public_surface_str("test.rs", src_byte_a),
        normalized_public_surface_str("test.rs", src_byte_b),
        "byte string escaped newline continuation must normalize indentation"
    );
}

#[test]
fn public_api_guard_handles_wrapped_struct_fields_p2_1() {
    // Regression A: Wrapped type
    let struct_wrapped_32 = "pub struct S {\n    pub value:\n        Vec<u32>,\n}";
    let struct_wrapped_64 = "pub struct S {\n    pub value:\n        Vec<u64>,\n}";
    assert!(struct_wrapped_32.contains("pub value:\n"));
    assert!(struct_wrapped_32.as_bytes().contains(&b'\n'));
    let surf_w32 = normalized_public_surface_str("test.rs", struct_wrapped_32);
    let surf_w64 = normalized_public_surface_str("test.rs", struct_wrapped_64);
    assert_ne!(
        surf_w32, surf_w64,
        "wrapped struct field type change must alter surface"
    );

    // Regression B: Formatting-only reflow
    let struct_oneline = "pub struct S {\n    pub value: Vec<u32>,\n}";
    let surf_oneline = normalized_public_surface_str("test.rs", struct_oneline);
    assert_eq!(
        surf_w32, surf_oneline,
        "wrapped struct field must match one-line struct field surface"
    );

    // Regression C: Private wrapped field
    let struct_priv_w32 =
        "pub struct S {\n    private:\n        Vec<u32>,\n    pub visible: bool,\n}";
    let struct_priv_w64 =
        "pub struct S {\n    private:\n        Vec<u64>,\n    pub visible: bool,\n}";
    assert!(struct_priv_w32.contains("private:\n"));
    assert!(struct_priv_w32.as_bytes().contains(&b'\n'));
    assert_eq!(
        normalized_public_surface_str("test.rs", struct_priv_w32),
        normalized_public_surface_str("test.rs", struct_priv_w64),
        "private wrapped struct field change must NOT alter surface"
    );

    // Regression D: Public field after private field containing comparison syntax
    let struct_cmp_32 =
        "pub struct S {\n    private: [(); (1 < 2) as usize],\n    pub visible: u32,\n}";
    let struct_cmp_64 =
        "pub struct S {\n    private: [(); (1 < 2) as usize],\n    pub visible: u64,\n}";
    let struct_cmp_diff_priv =
        "pub struct S {\n    private: [(); (2 < 3) as usize],\n    pub visible: u32,\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", struct_cmp_32),
        normalized_public_surface_str("test.rs", struct_cmp_64),
        "public field type after private comparison field must alter surface"
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", struct_cmp_32),
        normalized_public_surface_str("test.rs", struct_cmp_diff_priv),
        "private comparison field change must NOT alter surface"
    );

    // Regression E: Nested generic public field
    let struct_nested_32 =
        "pub struct S {\n    pub value:\n        Result<Vec<u32>, Option<[u8; 3]>>,\n}";
    let struct_nested_64 =
        "pub struct S {\n    pub value:\n        Result<Vec<u64>, Option<[u8; 3]>>,\n}";
    assert!(struct_nested_32.contains("pub value:\n"));
    assert!(struct_nested_32.as_bytes().contains(&b'\n'));
    assert_ne!(
        normalized_public_surface_str("test.rs", struct_nested_32),
        normalized_public_surface_str("test.rs", struct_nested_64),
        "nested generic wrapped field type change must alter surface"
    );

    // Regression F: Restricted visibility
    let struct_restr_32 = "pub struct S {\n    pub(crate) value:\n        Vec<u32>,\n}";
    let struct_restr_64 = "pub struct S {\n    pub(crate) value:\n        Vec<u64>,\n}";
    assert!(struct_restr_32.contains("pub(crate) value:\n"));
    assert!(struct_restr_32.as_bytes().contains(&b'\n'));
    assert_ne!(
        normalized_public_surface_str("test.rs", struct_restr_32),
        normalized_public_surface_str("test.rs", struct_restr_64),
        "restricted public wrapped field type change must alter surface"
    );
}

#[test]
fn public_api_guard_handles_inline_default_trait_bodies_p2_2() {
    // Regression A: Inline default method body isolation
    let trait_inline_a = "pub trait T {\n    fn f() -> u32 { private_a() }\n}";
    let trait_inline_b = "pub trait T {\n    fn f() -> u32 { private_b() }\n}";
    assert!(trait_inline_a.contains("{\n    fn f() -> u32 { private_a() }\n}"));
    assert!(trait_inline_a.as_bytes().contains(&b'\n'));
    assert_eq!(
        normalized_public_surface_str("test.rs", trait_inline_a),
        normalized_public_surface_str("test.rs", trait_inline_b),
        "inline default method body change in multiline trait must NOT alter surface"
    );

    // Regression B: Inline default method signature mutation
    let trait_inline_sig_64 = "pub trait T {\n    fn f() -> u64 { private_a() }\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", trait_inline_a),
        normalized_public_surface_str("test.rs", trait_inline_sig_64),
        "inline default method signature mutation must alter surface"
    );

    // Regression C: Required method
    let trait_req_32 = "pub trait T {\n    fn f() -> u32;\n}";
    let trait_req_64 = "pub trait T {\n    fn f() -> u64;\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", trait_req_32),
        normalized_public_surface_str("test.rs", trait_req_64),
        "required method signature mutation must alter surface"
    );

    // Regression D: Multiline default method
    let trait_multi_def_32 = "pub trait T {\n    fn f(\n        &self,\n        x: Vec<u32>,\n    ) -> u32\n    {\n        private_a()\n    }\n}";
    let trait_multi_def_64 = "pub trait T {\n    fn f(\n        &self,\n        x: Vec<u64>,\n    ) -> u32\n    {\n        private_a()\n    }\n}";
    let trait_multi_def_diff_body = "pub trait T {\n    fn f(\n        &self,\n        x: Vec<u32>,\n    ) -> u32\n    {\n        private_b()\n    }\n}";
    assert!(trait_multi_def_32.contains("fn f(\n"));
    assert!(trait_multi_def_32.as_bytes().contains(&b'\n'));
    assert_ne!(
        normalized_public_surface_str("test.rs", trait_multi_def_32),
        normalized_public_surface_str("test.rs", trait_multi_def_64),
        "multiline default method signature mutation must alter surface"
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", trait_multi_def_32),
        normalized_public_surface_str("test.rs", trait_multi_def_diff_body),
        "multiline default method body mutation must NOT alter surface"
    );

    // Regression E: Default body containing nested braces
    let trait_nested_body_a = "pub trait T {\n    fn f() -> u32 {\n        if private_cond() {\n            private_a()\n        } else {\n            private_b()\n        }\n    }\n}";
    let trait_nested_body_b = "pub trait T {\n    fn f() -> u32 {\n        if private_cond() {\n            private_c()\n        } else {\n            private_d()\n        }\n    }\n}";
    assert!(trait_nested_body_a.contains("if private_cond() {\n"));
    assert!(trait_nested_body_a.as_bytes().contains(&b'\n'));
    assert_eq!(
        normalized_public_surface_str("test.rs", trait_nested_body_a),
        normalized_public_surface_str("test.rs", trait_nested_body_b),
        "nested default method body mutation must NOT alter surface"
    );

    // Regression F: Type-position braced macro in trait method signature
    let trait_macro_32 = "pub trait T {\n    fn f() -> type_macro! { u32 } { private_a() }\n}";
    let trait_macro_64 = "pub trait T {\n    fn f() -> type_macro! { u64 } { private_a() }\n}";
    let trait_macro_diff_body =
        "pub trait T {\n    fn f() -> type_macro! { u32 } { private_b() }\n}";
    assert!(trait_macro_32.contains("type_macro! { u32 } { private_a() }\n"));
    assert!(trait_macro_32.as_bytes().contains(&b'\n'));
    assert_ne!(
        normalized_public_surface_str("test.rs", trait_macro_32),
        normalized_public_surface_str("test.rs", trait_macro_64),
        "type-position braced macro payload change in trait method must alter surface"
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", trait_macro_32),
        normalized_public_surface_str("test.rs", trait_macro_diff_body),
        "private body change with braced macro in trait method must NOT alter surface"
    );
}

#[test]
fn public_api_guard_handles_digit_suffixed_generic_identifiers_p2_3() {
    // Regression A: Digit-suffixed type
    let fn_vec2_1 = "pub fn f() -> Vec2<{1}> {\n    private_a()\n}";
    let fn_vec2_2 = "pub fn f() -> Vec2<{2}> {\n    private_a()\n}";
    let fn_vec2_diff_body = "pub fn f() -> Vec2<{1}> {\n    private_b()\n}";
    assert!(fn_vec2_1.contains("Vec2<{1}> {\n"));
    assert!(fn_vec2_1.as_bytes().contains(&b'\n'));
    assert_ne!(
        normalized_public_surface_str("test.rs", fn_vec2_1),
        normalized_public_surface_str("test.rs", fn_vec2_2),
        "const generic arg change in Vec2<{{1}}> must alter surface"
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", fn_vec2_1),
        normalized_public_surface_str("test.rs", fn_vec2_diff_body),
        "private body change with Vec2<{{1}}> must NOT alter surface"
    );

    // Regression B: Path-qualified digit-suffixed type
    let fn_math_vec2_1 = "pub fn f() -> math::Vec2<{1}> {\n    private_a()\n}";
    let fn_math_vec2_2 = "pub fn f() -> math::Vec2<{2}> {\n    private_a()\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", fn_math_vec2_1),
        normalized_public_surface_str("test.rs", fn_math_vec2_2),
        "const generic arg change in math::Vec2<{{1}}> must alter surface"
    );

    // Regression C: Nested generic
    let fn_opt_vec2_1 = "pub fn f() -> Option<Vec2<{1}>> {\n    private_a()\n}";
    let fn_opt_vec2_2 = "pub fn f() -> Option<Vec2<{2}>> {\n    private_a()\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", fn_opt_vec2_1),
        normalized_public_surface_str("test.rs", fn_opt_vec2_2),
        "nested generic arg change in Option<Vec2<{{1}}>> must alter surface"
    );

    // Regression D: Existing comparison
    let const_less_1 = "pub const LESS: bool = 1 < 2;\npub fn next() {}";
    let const_less_2 = "pub const LESS: bool = 1 < 3;\npub fn next() {}";
    assert_ne!(
        normalized_public_surface_str("test.rs", const_less_1),
        normalized_public_surface_str("test.rs", const_less_2)
    );

    // Regression E: Identifier expression comparison
    let const_ident_cmp_1 = "pub const LESS: bool = A < B;\npub fn next() {}";
    let const_ident_cmp_2 = "pub const LESS: bool = A < C;\npub fn next() {}";
    assert_ne!(
        normalized_public_surface_str("test.rs", const_ident_cmp_1),
        normalized_public_surface_str("test.rs", const_ident_cmp_2)
    );

    // Regression F: Const-generic expression containing comparison
    let fn_const_expr_cmp_1 =
        "pub fn f() -> Foo<{ if 1 < 2 { 32 } else { 64 } }> {\n    private_a()\n}";
    let fn_const_expr_cmp_2 =
        "pub fn f() -> Foo<{ if 1 < 2 { 32 } else { 128 } }> {\n    private_a()\n}";
    let fn_const_expr_cmp_diff_body =
        "pub fn f() -> Foo<{ if 1 < 2 { 32 } else { 64 } }> {\n    private_b()\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", fn_const_expr_cmp_1),
        normalized_public_surface_str("test.rs", fn_const_expr_cmp_2)
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", fn_const_expr_cmp_1),
        normalized_public_surface_str("test.rs", fn_const_expr_cmp_diff_body)
    );
}

#[test]
fn public_api_guard_handles_trait_associated_const_braces_blocker_1() {
    // Regression A: Braced associated const default mutation { 1 } vs { 2 }
    let trait_const_1 = "pub trait T {\n    const N: usize = { 1 };\n}";
    let trait_const_2 = "pub trait T {\n    const N: usize = { 2 };\n}";
    assert!(trait_const_1.contains("const N: usize = { 1 };"));
    assert!(trait_const_1.as_bytes().contains(&b'\n'));
    let surf_c1 = normalized_public_surface_str("test.rs", trait_const_1);
    let surf_c2 = normalized_public_surface_str("test.rs", trait_const_2);
    assert_ne!(
        surf_c1, surf_c2,
        "braced associated const default mutation {{ 1 }} -> {{ 2 }} must alter public surface"
    );

    // Regression B: Formatting-only change
    let trait_const_reflow = "pub trait T {\n    const N: usize = {\n        1\n    };\n}";
    assert!(trait_const_reflow.contains("{\n        1\n    };"));
    assert!(trait_const_reflow.as_bytes().contains(&b'\n'));
    let surf_reflow = normalized_public_surface_str("test.rs", trait_const_reflow);
    assert_eq!(
        surf_c1, surf_reflow,
        "formatting-only reflow in associated const must match surface"
    );

    // Regression C: Nested braced expression
    let trait_nested_const_1 =
        "pub trait T {\n    const N: usize = {\n        if true { 1 } else { 2 }\n    };\n}";
    let trait_nested_const_3 =
        "pub trait T {\n    const N: usize = {\n        if true { 3 } else { 2 }\n    };\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", trait_nested_const_1),
        normalized_public_surface_str("test.rs", trait_nested_const_3),
        "nested braced expression in associated const must alter surface on value mutation"
    );

    // Regression D: Default method remains isolated
    let trait_method_a = "pub trait T {\n    fn f() -> u32 { private_a() }\n}";
    let trait_method_b = "pub trait T {\n    fn f() -> u32 { private_b() }\n}";
    let trait_method_64 = "pub trait T {\n    fn f() -> u64 { private_a() }\n}";
    assert_eq!(
        normalized_public_surface_str("test.rs", trait_method_a),
        normalized_public_surface_str("test.rs", trait_method_b),
        "default method private implementation must remain isolated"
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", trait_method_a),
        normalized_public_surface_str("test.rs", trait_method_64),
        "default method signature change must alter surface"
    );
}

#[test]
fn public_api_guard_handles_const_generic_identifier_comparisons_blocker_2() {
    // Regression A: Identifier comparison inside const generic
    let fn_flag_cmp_1 = "pub fn f() -> Flag<{ A < B }> {\n    private_a()\n}";
    let fn_flag_cmp_2 = "pub fn f() -> Flag<{ A < C }> {\n    private_a()\n}";
    let fn_flag_cmp_diff_body = "pub fn f() -> Flag<{ A < B }> {\n    private_b()\n}";
    assert!(fn_flag_cmp_1.contains("Flag<{ A < B }>"));
    assert!(fn_flag_cmp_1.as_bytes().contains(&b'\n'));

    let surf_flag_1 = normalized_public_surface_str("test.rs", fn_flag_cmp_1);
    let surf_flag_2 = normalized_public_surface_str("test.rs", fn_flag_cmp_2);
    let surf_flag_diff_body = normalized_public_surface_str("test.rs", fn_flag_cmp_diff_body);

    assert_ne!(
        surf_flag_1, surf_flag_2,
        "operand mutation in Flag<{{ A < B }}> must alter surface"
    );
    assert_eq!(
        surf_flag_1, surf_flag_diff_body,
        "private body change with Flag<{{ A < B }}> must NOT alter surface"
    );
    assert!(
        !surf_flag_1.contains("private_a"),
        "private body must not leak into surface: {surf_flag_1}"
    );

    // Regression B: Numeric comparison inside const generic
    let fn_flag_num_1 = "pub fn f() -> Flag<{ 1 < 2 }> {\n    private_a()\n}";
    let fn_flag_num_2 = "pub fn f() -> Flag<{ 1 < 3 }> {\n    private_a()\n}";
    let fn_flag_num_diff_body = "pub fn f() -> Flag<{ 1 < 2 }> {\n    private_b()\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", fn_flag_num_1),
        normalized_public_surface_str("test.rs", fn_flag_num_2)
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", fn_flag_num_1),
        normalized_public_surface_str("test.rs", fn_flag_num_diff_body)
    );

    // Regression C: Nested generic inside const expression (turbofish)
    let fn_turbofish_1 =
        "pub fn f() -> Flag<{ core::mem::size_of::<u32>() < 8 }> {\n    private_a()\n}";
    let fn_turbofish_2 =
        "pub fn f() -> Flag<{ core::mem::size_of::<u64>() < 8 }> {\n    private_a()\n}";
    let fn_turbofish_diff_body =
        "pub fn f() -> Flag<{ core::mem::size_of::<u32>() < 8 }> {\n    private_b()\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", fn_turbofish_1),
        normalized_public_surface_str("test.rs", fn_turbofish_2)
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", fn_turbofish_1),
        normalized_public_surface_str("test.rs", fn_turbofish_diff_body)
    );
    assert!(!normalized_public_surface_str("test.rs", fn_turbofish_1).contains("private_a"));

    // Regression D: Existing digit-suffixed generic
    let fn_vec2_1 = "pub fn f() -> Vec2<{1}> {\n    private_a()\n}";
    let fn_vec2_2 = "pub fn f() -> Vec2<{2}> {\n    private_a()\n}";
    let fn_vec2_diff_body = "pub fn f() -> Vec2<{1}> {\n    private_b()\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", fn_vec2_1),
        normalized_public_surface_str("test.rs", fn_vec2_2)
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", fn_vec2_1),
        normalized_public_surface_str("test.rs", fn_vec2_diff_body)
    );

    // Regression E: Existing top-level const comparison
    let const_less_1 = "pub const LESS: bool = A < B;\npub fn next() {}";
    let const_less_2 = "pub const LESS: bool = A < C;\npub fn next() {}";
    assert_ne!(
        normalized_public_surface_str("test.rs", const_less_1),
        normalized_public_surface_str("test.rs", const_less_2)
    );

    // Regression F: Existing complex const generic
    let fn_complex_1 = "pub fn f() -> Foo<{ if 1 < 2 { 32 } else { 64 } }> {\n    private_a()\n}";
    let fn_complex_2 = "pub fn f() -> Foo<{ if 1 < 2 { 32 } else { 128 } }> {\n    private_a()\n}";
    let fn_complex_diff_body =
        "pub fn f() -> Foo<{ if 1 < 2 { 32 } else { 64 } }> {\n    private_b()\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", fn_complex_1),
        normalized_public_surface_str("test.rs", fn_complex_2)
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", fn_complex_1),
        normalized_public_surface_str("test.rs", fn_complex_diff_body)
    );
}

#[test]
fn public_api_guard_handles_generic_types_inside_const_expression_blocks_probe() {
    // Probe 1: Normal generic type Option<u8> inside const generic block
    let fn_opt_u8 =
        "pub fn f() -> Flag<{\n    let _: Option<u8> = None;\n    true\n}> {\n    private_a()\n}";
    let fn_opt_u16 =
        "pub fn f() -> Flag<{\n    let _: Option<u16> = None;\n    true\n}> {\n    private_a()\n}";
    let fn_opt_diff_body =
        "pub fn f() -> Flag<{\n    let _: Option<u8> = None;\n    true\n}> {\n    private_b()\n}";

    assert!(fn_opt_u8.contains("Option<u8>"));
    assert!(fn_opt_u8.as_bytes().contains(&b'\n'));

    let surf_u8 = normalized_public_surface_str("test.rs", fn_opt_u8);
    let surf_u16 = normalized_public_surface_str("test.rs", fn_opt_u16);
    let surf_diff_body = normalized_public_surface_str("test.rs", fn_opt_diff_body);

    assert_ne!(
        surf_u8, surf_u16,
        "type mutation in const generic block must alter public surface"
    );
    assert_eq!(
        surf_u8, surf_diff_body,
        "private body change must NOT alter public surface"
    );
    assert!(
        !surf_u8.contains("private_a"),
        "private body must not leak into public surface"
    );

    // Probe 2: Nested generic type Result<Option<u8>, Error>
    let fn_nested_u8 = "pub fn f() -> Flag<{\n    let _: Result<Option<u8>, Error> = Ok(None);\n    true\n}> {\n    private_a()\n}";
    let fn_nested_u16 = "pub fn f() -> Flag<{\n    let _: Result<Option<u16>, Error> = Ok(None);\n    true\n}> {\n    private_a()\n}";
    let fn_nested_diff_body = "pub fn f() -> Flag<{\n    let _: Result<Option<u8>, Error> = Ok(None);\n    true\n}> {\n    private_b()\n}";

    let surf_nest_u8 = normalized_public_surface_str("test.rs", fn_nested_u8);
    let surf_nest_u16 = normalized_public_surface_str("test.rs", fn_nested_u16);
    let surf_nest_diff = normalized_public_surface_str("test.rs", fn_nested_diff_body);

    assert_ne!(
        surf_nest_u8, surf_nest_u16,
        "nested generic type mutation in const block must alter public surface"
    );
    assert_eq!(
        surf_nest_u8, surf_nest_diff,
        "private body change with nested generic in const block must NOT alter public surface"
    );
    assert!(!surf_nest_u8.contains("private_a"));
}

#[test]
fn public_api_guard_handles_multiline_restricted_visibility_and_mutations() {
    let multi_vis_1 = "pub(in\n    crate) const X: u32 = 1;";
    let multi_vis_2 = "pub(in\n    crate) const X: u32 = 2;";
    let multi_vis_u64 = "pub(in\n    crate) const X: u64 = 1;";
    let single_vis = "pub(in crate) const X: u32 = 1;";

    assert!(multi_vis_1.contains("pub(in\n    crate)"));
    assert!(multi_vis_1.as_bytes().contains(&b'\n'));

    assert_ne!(
        normalized_public_surface_str("test.rs", multi_vis_1),
        normalized_public_surface_str("test.rs", multi_vis_2),
        "const value change with multiline pub(in crate) must alter public surface"
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", multi_vis_1),
        normalized_public_surface_str("test.rs", single_vis),
        "multiline pub(in crate) must normalize equivalently to single-line"
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", multi_vis_1),
        normalized_public_surface_str("test.rs", multi_vis_u64),
        "type change with multiline pub(in crate) must alter public surface"
    );

    let struct_const_1 = "pub(in\n    crate) const HEADER: Spec = Spec {\n    rev: 1,\n};";
    let struct_const_2 = "pub(in\n    crate) const HEADER: Spec = Spec {\n    rev: 2,\n};";
    assert_ne!(
        normalized_public_surface_str("test.rs", struct_const_1),
        normalized_public_surface_str("test.rs", struct_const_2),
        "braced const field mutation with multiline pub(in crate) must alter public surface"
    );

    let pub_crate_split = "pub(\n    crate\n) const X: u32 = 1;";
    let pub_crate_single = "pub(crate) const X: u32 = 1;";
    assert_eq!(
        normalized_public_surface_str("test.rs", pub_crate_split),
        normalized_public_surface_str("test.rs", pub_crate_single)
    );

    let pub_super = "pub(super) const X: u32 = 1;";
    let pub_self = "pub(self) const X: u32 = 1;";
    assert_ne!(
        normalized_public_surface_str("test.rs", pub_super),
        normalized_public_surface_str("test.rs", pub_self)
    );

    let pub_path_1 = "pub(in crate::module) const X: u32 = 1;";
    let pub_path_2 = "pub(in crate::other) const X: u32 = 1;";
    assert_ne!(
        normalized_public_surface_str("test.rs", pub_path_1),
        normalized_public_surface_str("test.rs", pub_path_2)
    );

    let pub_comment = "pub /*x*/ (\n    in crate\n) const X: u32 = 1;";
    assert_eq!(
        normalized_public_surface_str("test.rs", pub_comment),
        normalized_public_surface_str("test.rs", single_vis)
    );
}

#[test]
fn public_api_guard_handles_multiline_tuple_struct_headers_and_mutations() {
    // A. Private tuple field stays private
    let priv_u32 = "pub struct S\n(\n    u32,\n);";
    let priv_u64 = "pub struct S\n(\n    u64,\n);";
    assert!(priv_u32.contains("S\n("));
    assert!(priv_u32.as_bytes().contains(&b'\n'));
    assert_eq!(
        normalized_public_surface_str("test.rs", priv_u32),
        normalized_public_surface_str("test.rs", priv_u64),
        "private tuple field type mutation must NOT alter public surface"
    );

    // B. Public tuple field remains contract-bearing
    let pub_u32 = "pub struct S\n(\n    pub u32,\n);";
    let pub_u64 = "pub struct S\n(\n    pub u64,\n);";
    assert_ne!(
        normalized_public_surface_str("test.rs", pub_u32),
        normalized_public_surface_str("test.rs", pub_u64),
        "public tuple field type mutation MUST alter public surface"
    );

    // C. Restricted public tuple field
    let pub_crate = "pub struct S\n(\n    pub(crate) u32,\n);";
    assert_ne!(
        normalized_public_surface_str("test.rs", pub_crate),
        normalized_public_surface_str("test.rs", priv_u32),
        "restricted public tuple field must differ from private field"
    );

    // D. Following declaration remains separate
    let seq_u32 = "pub struct S\n(\n    u32,\n);\npub const NEXT: u32 = 7;";
    let seq_u64 = "pub struct S\n(\n    u64,\n);\npub const NEXT: u32 = 7;";
    let surf_seq_32 = normalized_public_surface_str("test.rs", seq_u32);
    let surf_seq_64 = normalized_public_surface_str("test.rs", seq_u64);
    assert_eq!(surf_seq_32, surf_seq_64);
    assert!(
        surf_seq_32.contains("pub const NEXT: u32 = 7;"),
        "following const must be inventoried separately: {surf_seq_32}"
    );

    // E. Canonical/reflow equivalence
    let same_line_open = "pub struct S(\n    u32,\n);";
    let split_line_open = "pub struct S\n(\n    u32,\n);";
    assert_eq!(
        normalized_public_surface_str("test.rs", same_line_open),
        normalized_public_surface_str("test.rs", split_line_open),
        "reflowed tuple struct opening delimiter must normalize equivalently"
    );
}

#[test]
fn public_api_guard_accumulates_and_normalizes_multiline_enum_variants() {
    // A. Multiline tuple variant formatting equivalence
    let tup_single = "pub enum E {\n    A(u32),\n}";
    let tup_multi = "pub enum E {\n    A(\n        u32,\n    ),\n}";
    let tup_diff = "pub enum E {\n    A(\n        u64,\n    ),\n}";

    assert_eq!(
        normalized_public_surface_str("test.rs", tup_single),
        normalized_public_surface_str("test.rs", tup_multi),
        "multiline tuple variant must normalize equivalently to single-line"
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", tup_multi),
        normalized_public_surface_str("test.rs", tup_diff),
        "tuple variant payload type change must alter public surface"
    );

    // B. Multiline struct variant formatting equivalence
    let str_single = "pub enum E {\n    A { x: u32, y: bool },\n}";
    let str_multi = "pub enum E {\n    A {\n        x: u32,\n        y: bool,\n    },\n}";
    let str_diff = "pub enum E {\n    A {\n        x: u64,\n        y: bool,\n    },\n}";

    assert_eq!(
        normalized_public_surface_str("test.rs", str_single),
        normalized_public_surface_str("test.rs", str_multi),
        "multiline struct variant must normalize equivalently to single-line"
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", str_multi),
        normalized_public_surface_str("test.rs", str_diff),
        "struct variant field type change must alter public surface"
    );

    // C. Discriminant equivalence
    let disc_single = "pub enum E {\n    A = 1,\n}";
    let disc_multi = "pub enum E {\n    A =\n        1,\n}";
    let disc_diff = "pub enum E {\n    A =\n        2,\n}";

    assert_eq!(
        normalized_public_surface_str("test.rs", disc_single),
        normalized_public_surface_str("test.rs", disc_multi),
        "multiline discriminant must normalize equivalently to single-line"
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", disc_multi),
        normalized_public_surface_str("test.rs", disc_diff),
        "discriminant value change must alter public surface"
    );

    // D. Outer attribute with trivia
    let attr_old =
        "pub enum E {\n    # /*x*/ [deprecated(note = \"old\")]\n    A(\n        u32,\n    ),\n}";
    let attr_new =
        "pub enum E {\n    # /*x*/ [deprecated(note = \"new\")]\n    A(\n        u32,\n    ),\n}";
    let attr_type_diff =
        "pub enum E {\n    # /*x*/ [deprecated(note = \"old\")]\n    A(\n        u64,\n    ),\n}";
    let attr_reformatted = "pub enum E {\n    #[deprecated(note = \"old\")]\n    A(u32),\n}";

    assert_ne!(
        normalized_public_surface_str("test.rs", attr_old),
        normalized_public_surface_str("test.rs", attr_new),
        "attribute literal note change on variant must alter public surface"
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", attr_old),
        normalized_public_surface_str("test.rs", attr_type_diff),
        "payload type change on attributed variant must alter public surface"
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", attr_old),
        normalized_public_surface_str("test.rs", attr_reformatted),
        "trivia and reflow on attributed variant must normalize equivalently"
    );
}

#[test]
fn public_api_guard_resumes_after_same_line_outer_attributes() {
    let u32_signature = "#[deprecated] pub fn f() -> u32 { private_a() }";
    let u64_signature = "#[deprecated] pub fn f() -> u64 { private_a() }";
    assert_ne!(
        normalized_public_surface_str("test.rs", u32_signature),
        normalized_public_surface_str("test.rs", u64_signature),
        "same-line item after an outer attribute must remain contract-bearing"
    );

    let different_body = "#[deprecated] pub fn f() -> u32 { private_b() }";
    assert_eq!(
        normalized_public_surface_str("test.rs", u32_signature),
        normalized_public_surface_str("test.rs", different_body),
        "an attributed function's private body must remain excluded"
    );

    let multiple_u32 = "#[deprecated] #[inline] pub fn f() -> u32 { 0 }";
    let multiple_u64 = "#[deprecated] #[inline] pub fn f() -> u64 { 0 }";
    let multiple_surface = normalized_public_surface_str("test.rs", multiple_u32);
    assert!(multiple_surface.contains("#[deprecated]"));
    assert!(multiple_surface.contains("#[inline]"));
    assert_ne!(
        multiple_surface,
        normalized_public_surface_str("test.rs", multiple_u64)
    );

    let multiline_u32 =
        "#[cfg_attr(\n    feature = \"x\",\n    deprecated\n)] pub fn f() -> u32 { 0 }";
    let multiline_u64 =
        "#[cfg_attr(\n    feature = \"x\",\n    deprecated\n)] pub fn f() -> u64 { 0 }";
    assert!(multiline_u32.contains(")] pub fn"));
    assert_ne!(
        normalized_public_surface_str("test.rs", multiline_u32),
        normalized_public_surface_str("test.rs", multiline_u64)
    );

    let variant_u32 = "pub enum E {\n    #[deprecated] A(u32),\n}";
    let variant_u64 = "pub enum E {\n    #[deprecated] A(u64),\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", variant_u32),
        normalized_public_surface_str("test.rs", variant_u64)
    );

    let public_field_u32 = "pub struct S {\n    #[deprecated] pub value: u32,\n}";
    let public_field_u64 = "pub struct S {\n    #[deprecated] pub value: u64,\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", public_field_u32),
        normalized_public_surface_str("test.rs", public_field_u64)
    );
    let private_field_u32 = "pub struct S {\n    #[deprecated] value: u32,\n}";
    let private_field_u64 = "pub struct S {\n    #[deprecated] value: u64,\n}";
    assert_eq!(
        normalized_public_surface_str("test.rs", private_field_u32),
        normalized_public_surface_str("test.rs", private_field_u64)
    );

    let trait_u32 = "pub trait T {\n    #[deprecated] fn f() -> u32;\n}";
    let trait_u64 = "pub trait T {\n    #[deprecated] fn f() -> u64;\n}";
    assert_ne!(
        normalized_public_surface_str("test.rs", trait_u32),
        normalized_public_surface_str("test.rs", trait_u64)
    );
}

#[test]
fn public_api_guard_continues_type_aliases_past_where_clause_commas() {
    let vec_target = "pub type Foo<T>\nwhere\n    T: Copy,\n= Vec<T>;";
    let option_target = "pub type Foo<T>\nwhere\n    T: Copy,\n= Option<T>;";
    assert_ne!(
        normalized_public_surface_str("test.rs", vec_target),
        normalized_public_surface_str("test.rs", option_target),
        "a where-clause comma must not hide the type-alias target"
    );
    let single_line = "pub type Foo<T> where T: Copy, = Vec<T>;";
    assert_eq!(
        normalized_public_surface_str("test.rs", vec_target),
        normalized_public_surface_str("test.rs", single_line),
        "equivalent type-alias formatting must normalize consistently"
    );

    let changed_bound = "pub type Foo<T>\nwhere\n    T: Clone,\n= Vec<T>;";
    assert_ne!(
        normalized_public_surface_str("test.rs", vec_target),
        normalized_public_surface_str("test.rs", changed_bound)
    );
    let multiple_bounds =
        "pub type Foo<T, U>\nwhere\n    T: Copy + Clone,\n    U: Send,\n= Result<T, U>;";
    let changed_multiple_bounds =
        "pub type Foo<T, U>\nwhere\n    T: Copy + Clone,\n    U: Sync,\n= Result<T, U>;";
    assert_ne!(
        normalized_public_surface_str("test.rs", multiple_bounds),
        normalized_public_surface_str("test.rs", changed_multiple_bounds)
    );

    let nested_target = "pub type Foo<T>\nwhere\n    T: Copy,\n= Result<Vec<T>, Option<T>>;";
    let changed_nested_target = "pub type Foo<T>\nwhere\n    T: Copy,\n= Result<Vec<T>, Box<T>>;";
    assert_ne!(
        normalized_public_surface_str("test.rs", nested_target),
        normalized_public_surface_str("test.rs", changed_nested_target)
    );

    let followed = "pub type Foo<T>\nwhere\n    T: Copy,\n= Vec<T>;\n\npub const NEXT: u32 = 1;";
    let followed_surface = normalized_public_surface_str("test.rs", followed);
    assert!(followed_surface.contains("= Vec<T>;"));
    assert!(followed_surface.contains("pub const NEXT: u32 = 1;"));
}

#[test]
fn public_api_guard_recurses_into_inline_public_modules() {
    let u32_signature = "pub mod m { pub fn f() -> u32 { private_a() } }";
    let u64_signature = "pub mod m { pub fn f() -> u64 { private_a() } }";
    assert_ne!(
        normalized_public_surface_str("test.rs", u32_signature),
        normalized_public_surface_str("test.rs", u64_signature),
        "nested public contracts in inline modules must remain contract-bearing"
    );

    let different_body = "pub mod m { pub fn f() -> u32 { private_b() } }";
    assert_eq!(
        normalized_public_surface_str("test.rs", u32_signature),
        normalized_public_surface_str("test.rs", different_body)
    );

    let hidden_u32 = "pub mod m { fn hidden() -> u32 { 1 } pub fn visible() -> bool { true } }";
    let hidden_u64 = "pub mod m { fn hidden() -> u64 { 2 } pub fn visible() -> bool { true } }";
    assert_eq!(
        normalized_public_surface_str("test.rs", hidden_u32),
        normalized_public_surface_str("test.rs", hidden_u64)
    );

    let public_field_u32 = "pub mod m { pub struct S { pub x: u32, private: u32, } }";
    let public_field_u64 = "pub mod m { pub struct S { pub x: u64, private: u32, } }";
    assert_ne!(
        normalized_public_surface_str("test.rs", public_field_u32),
        normalized_public_surface_str("test.rs", public_field_u64)
    );
    let private_field_u64 = "pub mod m { pub struct S { pub x: u32, private: u64, } }";
    assert_eq!(
        normalized_public_surface_str("test.rs", public_field_u32),
        normalized_public_surface_str("test.rs", private_field_u64)
    );

    let nested_u32 = "pub mod outer { pub mod inner { pub fn f() -> u32 { private_a() } } }";
    let nested_u64 = "pub mod outer { pub mod inner { pub fn f() -> u64 { private_a() } } }";
    assert_ne!(
        normalized_public_surface_str("test.rs", nested_u32),
        normalized_public_surface_str("test.rs", nested_u64)
    );

    let external = normalized_public_surface_str("test.rs", "pub mod external;");
    assert!(external.contains("pub mod external;"));
    assert!(!external.contains("private_a"));
}

#[test]
fn public_api_guard_preserves_literal_whitespace_in_public_fields() {
    let surface = |src| normalized_public_surface_str("test.rs", src);
    for (two_spaces, one_space, label) in [
        (
            "pub struct S { pub x: Ty!(\"a  b\"), }",
            "pub struct S { pub x: Ty!(\"a b\"), }",
            "named field",
        ),
        (
            "pub struct S { pub x: Ty!(r#\"a  b\"#), }",
            "pub struct S { pub x: Ty!(r#\"a b\"#), }",
            "raw string field",
        ),
        (
            "pub struct S { pub x: Ty!(b\"a  b\"), }",
            "pub struct S { pub x: Ty!(b\"a b\"), }",
            "byte string field",
        ),
        (
            "pub struct S(pub Ty!(\"a  b\"));",
            "pub struct S(pub Ty!(\"a b\"));",
            "tuple field",
        ),
        (
            "pub enum E { V(Ty!(\"a  b\")) }",
            "pub enum E { V(Ty!(\"a b\")) }",
            "enum payload",
        ),
        (
            "pub union U { pub x: Ty!(\"a  b\"), }",
            "pub union U { pub x: Ty!(\"a b\"), }",
            "union field",
        ),
        (
            "pub struct S { pub x: Ty!(\"a\nb\"), }",
            "pub struct S { pub x: Ty!(\"a b\"), }",
            "literal newline",
        ),
    ] {
        let two_spaces = surface(two_spaces);
        let one_space = surface(one_space);
        assert_ne!(
            two_spaces, one_space,
            "{label} literal collapsed; two={two_spaces:?}, one={one_space:?}"
        );
    }

    assert_eq!(
        surface("pub struct S { hidden: Ty!(\"a  b\"), pub visible: u32, }"),
        surface("pub struct S { hidden: Ty!(\"a b\"), pub visible: u32, }")
    );
    assert_eq!(
        surface("pub struct S { pub x: Result<Ty!(\"a  b\"), u32>, }"),
        surface("pub struct S {\n pub   x: Result<Ty!(\"a  b\"),    u32>,\n}")
    );
    assert_ne!(
        surface("pub fn f<T>() where T: Bound<Ty!(\"a  b\")> {}"),
        surface("pub fn f<T>() where T: Bound<Ty!(\"a b\")> {}")
    );
}

#[test]
fn public_api_guard_preserves_literal_whitespace_in_trait_members() {
    let surface = |src| normalized_public_surface_str("test.rs", src);
    for (two_spaces, one_space, label) in [
        (
            "pub trait T { const S: &'static str = \"a  b\"; }",
            "pub trait T { const S: &'static str = \"a b\"; }",
            "associated const",
        ),
        (
            "pub trait T { const S: &'static str = r#\"a  b\"#; }",
            "pub trait T { const S: &'static str = r#\"a b\"#; }",
            "raw associated const",
        ),
        (
            "pub trait T { const S: &'static str = \"a\nb\"; }",
            "pub trait T { const S: &'static str = \"a b\"; }",
            "associated const newline",
        ),
        (
            "pub trait T { fn f() -> Ty!(\"a  b\"); }",
            "pub trait T { fn f() -> Ty!(\"a b\"); }",
            "method return type",
        ),
        (
            "pub trait T { type Item: Bound<Ty!(\"a  b\")>; }",
            "pub trait T { type Item: Bound<Ty!(\"a b\")>; }",
            "associated type bound",
        ),
    ] {
        let two_spaces = surface(two_spaces);
        let one_space = surface(one_space);
        assert_ne!(
            two_spaces, one_space,
            "{label} literal collapsed; two={two_spaces:?}, one={one_space:?}"
        );
    }

    let body_a = "pub trait T { fn f() -> Ty!(\"a  b\") { private_a() } }";
    let body_b = "pub trait T { fn f() -> Ty!(\"a  b\") { private_b() } }";
    assert_eq!(surface(body_a), surface(body_b));
    assert_ne!(
        surface(body_a),
        surface("pub trait T { fn f() -> Ty!(\"a b\") { private_a() } }")
    );
}

#[test]
fn public_api_guard_resumes_after_inline_module_on_same_line() {
    let surface = |src| normalized_public_surface_str("test.rs", src);
    for (u32_source, u64_source, label) in [
        (
            "pub mod m {} pub fn f() -> u32 { 0 }",
            "pub mod m {} pub fn f() -> u64 { 0 }",
            "empty module",
        ),
        (
            "pub mod m { pub const X: u32 = 1; } pub fn f() -> u32 { 0 }",
            "pub mod m { pub const X: u32 = 1; } pub fn f() -> u64 { 0 }",
            "module member",
        ),
        (
            "pub mod m {} #[deprecated] pub fn f() -> u32 { 0 }",
            "pub mod m {} #[deprecated] pub fn f() -> u64 { 0 }",
            "following attribute",
        ),
        (
            "pub mod a {} pub mod b { pub fn f() -> u32 { 0 } }",
            "pub mod a {} pub mod b { pub fn f() -> u64 { 0 } }",
            "second module",
        ),
        (
            "pub mod a {\n pub mod b {}\n} pub fn outer() -> u32 { 0 }",
            "pub mod a {\n pub mod b {}\n} pub fn outer() -> u64 { 0 }",
            "nested module",
        ),
    ] {
        let u32_surface = surface(u32_source);
        let u64_surface = surface(u64_source);
        assert_ne!(
            u32_surface, u64_surface,
            "{label} remainder was lost; u32={u32_surface:?}, u64={u64_surface:?}"
        );
    }

    assert_ne!(
        surface("pub mod m { pub const X: u32 = 1; } pub fn f() -> u32 { 0 }"),
        surface("pub mod m { pub const X: u32 = 2; } pub fn f() -> u32 { 0 }")
    );
    assert_eq!(
        surface("pub mod m { pub const X: u32 = 1; } pub fn f() -> u32 { private_a() }"),
        surface("pub mod m { pub const X: u32 = 1; } pub fn f() -> u32 { private_b() }")
    );
    assert_eq!(
        surface("pub mod m {} fn hidden() -> u32 { 0 }"),
        surface("pub mod m {} fn hidden() -> u64 { 0 }")
    );
    let multiple = surface("pub mod m {} pub const A: u32 = 1; pub fn f() -> u32 { 0 }");
    assert!(multiple.contains("pub const A: u32 = 1;"));
    assert!(multiple.contains("pub fn f() -> u32 {"));
}

#[test]
fn public_api_guard_resumes_after_same_line_balanced_public_items() {
    let surface = |src| normalized_public_surface_str("test.rs", src);
    for (one_source, two_source, label) in [
        (
            "pub struct S {\n} pub const X: u32 = 1;",
            "pub struct S {\n} pub const X: u32 = 2;",
            "struct",
        ),
        (
            "pub union U {\n x: u32\n} pub const X: u32 = 1;",
            "pub union U {\n x: u32\n} pub const X: u32 = 2;",
            "union",
        ),
        (
            "pub trait T {\n} pub const X: u32 = 1;",
            "pub trait T {\n} pub const X: u32 = 2;",
            "trait",
        ),
        (
            "pub enum E {\n} pub const X: u32 = 1;",
            "pub enum E {\n} pub const X: u32 = 2;",
            "enum",
        ),
        (
            "pub fn first() {\n} pub const X: u32 = 1;",
            "pub fn first() {\n} pub const X: u32 = 2;",
            "function",
        ),
    ] {
        let u32_surface = surface(one_source);
        let u64_surface = surface(two_source);
        assert_ne!(
            u32_surface, u64_surface,
            "{label} remainder was lost; u32={u32_surface:?}, u64={u64_surface:?}"
        );
    }
}

#[test]
fn public_api_guard_preserves_outer_generic_depth_across_right_shift() {
    let second_2 = "pub fn f() -> Foo<{ 8 >> 1 }, { 2 }> { private_a() }";
    let second_3 = "pub fn f() -> Foo<{ 8 >> 1 }, { 3 }> { private_a() }";
    let surface_2 = normalized_public_surface_str("test.rs", second_2);
    let surface_3 = normalized_public_surface_str("test.rs", second_3);

    assert_ne!(
        surface_2, surface_3,
        "a following const-generic argument must remain contract-bearing; surface: {surface_2:?}"
    );

    let different_body = "pub fn f() -> Foo<{ 8 >> 1 }, { 2 }> { private_b() }";
    assert_eq!(
        surface_2,
        normalized_public_surface_str("test.rs", different_body)
    );
    let shifted_16 = "pub fn f() -> Foo<{ 16 >> 1 }, { 2 }> { private_a() }";
    assert_ne!(
        surface_2,
        normalized_public_surface_str("test.rs", shifted_16)
    );

    let nested_u32 = "pub fn f() -> Outer<Inner<u32>> { private_a() }";
    let nested_u64 = "pub fn f() -> Outer<Inner<u64>> { private_a() }";
    let nested_body = "pub fn f() -> Outer<Inner<u32>> { private_b() }";
    assert_ne!(
        normalized_public_surface_str("test.rs", nested_u32),
        normalized_public_surface_str("test.rs", nested_u64)
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", nested_u32),
        normalized_public_surface_str("test.rs", nested_body)
    );

    let inner_u8 = "pub fn f() -> Foo<{ core::mem::size_of::<Option<Result<u8, u16>>>() }, { 2 }> { private_a() }";
    let inner_u32 = "pub fn f() -> Foo<{ core::mem::size_of::<Option<Result<u32, u16>>>() }, { 2 }> { private_a() }";
    let inner_body = "pub fn f() -> Foo<{ core::mem::size_of::<Option<Result<u8, u16>>>() }, { 2 }> { private_b() }";
    assert_ne!(
        normalized_public_surface_str("test.rs", inner_u8),
        normalized_public_surface_str("test.rs", inner_u32)
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", inner_u8),
        normalized_public_surface_str("test.rs", inner_body)
    );

    let left_8 = "pub fn f() -> Foo<{ 8 << 1 }, { 2 }> { private_a() }";
    let left_16 = "pub fn f() -> Foo<{ 16 << 1 }, { 2 }> { private_a() }";
    let left_second = "pub fn f() -> Foo<{ 8 << 1 }, { 3 }> { private_a() }";
    let left_body = "pub fn f() -> Foo<{ 8 << 1 }, { 2 }> { private_b() }";
    assert_ne!(
        normalized_public_surface_str("test.rs", left_8),
        normalized_public_surface_str("test.rs", left_16)
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", left_8),
        normalized_public_surface_str("test.rs", left_second)
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", left_8),
        normalized_public_surface_str("test.rs", left_body)
    );

    let shift_assign_1 = "pub fn f() -> Foo<{ let mut x = 8; x >>= 1; x }, { 2 }> { private_a() }";
    let shift_assign_2 = "pub fn f() -> Foo<{ let mut x = 8; x >>= 2; x }, { 2 }> { private_a() }";
    let left_assign_1 = "pub fn f() -> Foo<{ let mut x = 8; x <<= 1; x }, { 2 }> { private_a() }";
    let left_assign_2 = "pub fn f() -> Foo<{ let mut x = 8; x <<= 2; x }, { 2 }> { private_a() }";
    assert_ne!(
        normalized_public_surface_str("test.rs", shift_assign_1),
        normalized_public_surface_str("test.rs", shift_assign_2)
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", left_assign_1),
        normalized_public_surface_str("test.rs", left_assign_2)
    );

    let qualified_2 = "pub fn f<T>() -> Foo<<T as Trait>::Assoc, { 2 }> { private_a() }";
    let qualified_3 = "pub fn f<T>() -> Foo<<T as Trait>::Assoc, { 3 }> { private_a() }";
    assert_ne!(
        normalized_public_surface_str("test.rs", qualified_2),
        normalized_public_surface_str("test.rs", qualified_3),
        "adjacent generic openings must not be classified as a left shift"
    );
}

#[test]
fn public_api_guard_retains_tuple_struct_where_clause() {
    let copy = "pub struct S<T>(pub T) where T: Copy;";
    let clone = "pub struct S<T>(pub T) where T: Clone;";
    let copy_surface = normalized_public_surface_str("test.rs", copy);
    let clone_surface = normalized_public_surface_str("test.rs", clone);

    assert_ne!(
        copy_surface, clone_surface,
        "tuple-struct where clauses must remain contract-bearing; surface: {copy_surface:?}"
    );

    let multiline_copy = "pub struct S<T>(\n    pub T,\n)\nwhere\n    T: Copy;";
    let multiline_clone = "pub struct S<T>(\n    pub T,\n)\nwhere\n    T: Clone;";
    assert!(multiline_copy.as_bytes().contains(&b'\n'));
    assert_ne!(
        normalized_public_surface_str("test.rs", multiline_copy),
        normalized_public_surface_str("test.rs", multiline_clone)
    );
    assert_eq!(
        copy_surface,
        normalized_public_surface_str("test.rs", multiline_copy)
    );

    let public_option = "pub struct S<T>(pub Option<T>) where T: Copy;";
    assert_ne!(
        copy_surface,
        normalized_public_surface_str("test.rs", public_option)
    );

    let private_t = "pub struct S<T>(T) where T: Copy;";
    let private_option = "pub struct S<T>(Option<T>) where T: Copy;";
    let private_clone = "pub struct S<T>(T) where T: Clone;";
    assert_eq!(
        normalized_public_surface_str("test.rs", private_t),
        normalized_public_surface_str("test.rs", private_option)
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", private_t),
        normalized_public_surface_str("test.rs", private_clone)
    );

    let predicates = "pub struct Pair<T, U>(pub T, pub U)\nwhere\n    T: Copy,\n    U: Clone;";
    let changed_t = "pub struct Pair<T, U>(pub T, pub U)\nwhere\n    T: Send,\n    U: Clone;";
    let changed_u = "pub struct Pair<T, U>(pub T, pub U)\nwhere\n    T: Copy,\n    U: Sync;";
    assert_ne!(
        normalized_public_surface_str("test.rs", predicates),
        normalized_public_surface_str("test.rs", changed_t)
    );
    assert_ne!(
        normalized_public_surface_str("test.rs", predicates),
        normalized_public_surface_str("test.rs", changed_u)
    );

    let associated_u32 = "pub struct Iter<T>(pub T) where T: Iterator<Item = u32>;";
    let associated_u64 = "pub struct Iter<T>(pub T) where T: Iterator<Item = u64>;";
    assert_ne!(
        normalized_public_surface_str("test.rs", associated_u32),
        normalized_public_surface_str("test.rs", associated_u64)
    );

    let followed = "pub struct S<T>(pub T)\nwhere\n    T: Copy;\n\npub const NEXT: u32 = 1;";
    let followed_surface = normalized_public_surface_str("test.rs", followed);
    assert!(followed_surface.contains("where T: Copy;"));
    assert!(followed_surface.contains("pub const NEXT: u32 = 1;"));

    let plain_public_u32 = "pub struct Plain(pub u32);";
    let plain_public_u64 = "pub struct Plain(pub u64);";
    let plain_private_u32 = "pub struct Private(u32);";
    let plain_private_u64 = "pub struct Private(u64);";
    assert_ne!(
        normalized_public_surface_str("test.rs", plain_public_u32),
        normalized_public_surface_str("test.rs", plain_public_u64)
    );
    assert_eq!(
        normalized_public_surface_str("test.rs", plain_private_u32),
        normalized_public_surface_str("test.rs", plain_private_u64)
    );

    for visibility in ["pub(crate)", "pub(super)", "pub(in crate)"] {
        let restricted_copy = format!("pub struct Restricted<T>({visibility} T) where T: Copy;");
        let restricted_clone = format!("pub struct Restricted<T>({visibility} T) where T: Clone;");
        assert_ne!(
            normalized_public_surface_str("test.rs", &restricted_copy),
            normalized_public_surface_str("test.rs", &restricted_clone)
        );
    }
}

#[test]
fn public_api_guard_preserves_raw_strings_in_enum_variants() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // A-E: whitespace-inside-a-raw-literal mutations that must NOT collapse
    // to the same surface, across hash counts, embedded quotes, and raw
    // byte strings.
    for (two_spaces, one_space, label) in [
        (
            "pub enum E { A = m!(r#\"a\"  b\"#), }",
            "pub enum E { A = m!(r#\"a\" b\"#), }",
            "A: exact Codex case - one hash, embedded quote right after open",
        ),
        (
            "pub enum E { A = m!(r#\"a\"quoted\"  b\"#), }",
            "pub enum E { A = m!(r#\"a\"quoted\" b\"#), }",
            "B: embedded normal quote mid-literal does not terminate it",
        ),
        (
            "pub enum E { A = m!(r##\"a\"#  b\"##), }",
            "pub enum E { A = m!(r##\"a\"# b\"##), }",
            "C: two hashes - embedded \"# does not close r##...\"##",
        ),
        (
            "pub enum E { A = m!(r###\"a  b\"###), }",
            "pub enum E { A = m!(r###\"a b\"###), }",
            "D: three hashes",
        ),
        (
            "pub enum E { A = m!(br#\"a\"  b\"#), }",
            "pub enum E { A = m!(br#\"a\" b\"#), }",
            "E: raw byte string, one hash, embedded quote",
        ),
        (
            "pub enum E { A = m!(br##\"a\"#  b\"##), }",
            "pub enum E { A = m!(br##\"a\"# b\"##), }",
            "E: raw byte string, two hashes",
        ),
    ] {
        let two = surface(two_spaces);
        let one = surface(one_space);
        assert_ne!(
            two, one,
            "{label} literal whitespace collapsed; two={two:?}, one={one:?}"
        );
    }

    // F: an actual runtime newline inside the raw literal (a genuine `\n`
    // byte in the simulated source, spanning two physical lines) must
    // differ from a single-space collapse of the same content.
    let newline_variant = "pub enum E {\n    A = m!(r#\"a\nb\"#),\n}";
    assert!(
        newline_variant.contains("a\nb"),
        "test fixture must contain a genuine runtime newline inside the raw literal"
    );
    let space_variant = "pub enum E {\n    A = m!(r#\"a b\"#),\n}";
    assert_ne!(
        surface(newline_variant),
        surface(space_variant),
        "an actual newline inside a raw literal spanning two physical lines was collapsed"
    );

    // G: code-only formatting around the macro call may normalize; the raw
    // payload itself must remain byte-for-byte significant.
    let tight = "pub enum E { A = m!(r#\"a  b\"#), }";
    let spread = "pub enum E {\n    A  =  m!( r#\"a  b\"#  ),\n}";
    assert_eq!(
        surface(tight),
        surface(spread),
        "code-only formatting around an unchanged raw literal must still normalize equal"
    );

    // Sibling enum shapes: normalize_variant() is shared by unit, tuple,
    // and struct variants, and by attributed variants - the raw-string fix
    // must not be an explicit-discriminant-only special case.
    for (two_spaces, one_space, label) in [
        (
            "pub enum E { A(Ty!(r#\"a\"  b\"#)), }",
            "pub enum E { A(Ty!(r#\"a\" b\"#)), }",
            "tuple variant payload",
        ),
        (
            "pub enum E { A { x: Ty!(r#\"a\"  b\"#) }, }",
            "pub enum E { A { x: Ty!(r#\"a\" b\"#) }, }",
            "struct variant field",
        ),
        (
            "pub enum E {\n    #[deprecated]\n    A = m!(r#\"a\"  b\"#),\n}",
            "pub enum E {\n    #[deprecated]\n    A = m!(r#\"a\" b\"#),\n}",
            "attributed variant",
        ),
    ] {
        let two = surface(two_spaces);
        let one = surface(one_space);
        assert_ne!(
            two, one,
            "{label} literal whitespace collapsed; two={two:?}, one={one:?}"
        );
    }
}

#[test]
fn public_api_guard_string_continuation_survives_whitespace_only_lines() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 2A: false-negative collision. Built via explicit `\n`/`\\` construction
    // (not Rust source-literal continuation), so the scanner receives the
    // intended bytes verbatim: a real backslash immediately before a real
    // LF, then a whitespace-only physical line, then another LF before the
    // resuming content.
    let escaped_then_blank = "pub const S: &str = \"foo\\\n    \nbar\";";
    assert!(
        escaped_then_blank.contains("foo\\\n"),
        "fixture must contain a real backslash immediately before a real LF"
    );
    assert!(
        escaped_then_blank.contains("\\\n    \n"),
        "fixture must contain a whitespace-only physical line between the two LFs"
    );
    let real_newline = "pub const S: &str = \"foo\nbar\";";
    assert_ne!(
        surface(escaped_then_blank),
        surface(real_newline),
        "an escaped continuation across a whitespace-only line collided with a genuine embedded newline"
    );

    // 2B: semantic equivalence matrix - all represent the payload "foobar".
    let plain = "pub const S: &str = \"foobar\";";
    let one_line_tight = "pub const S: &str = \"foo\\\nbar\";";
    let one_line_indented = "pub const S: &str = \"foo\\\n    bar\";";
    let two_lines_blank_then_indented = "pub const S: &str = \"foo\\\n    \n        bar\";";
    for (variant, label) in [
        (
            one_line_tight,
            "one continuation line, no leading whitespace",
        ),
        (
            one_line_indented,
            "one continuation line, with leading whitespace",
        ),
        (
            two_lines_blank_then_indented,
            "whitespace-only line then an indented resuming line",
        ),
    ] {
        assert_eq!(
            surface(plain),
            surface(variant),
            "{label} did not canonicalize to the same payload as a plain literal"
        );
    }

    // A genuine embedded newline must remain distinct from every
    // escaped-continuation form above.
    assert_ne!(surface(plain), surface(real_newline));
    assert_ne!(surface(one_line_tight), surface(real_newline));

    // 2C: byte strings - repeat the critical collision and equivalence
    // cases under Rust's byte-string continuation semantics (identical to
    // normal strings).
    let byte_escaped_then_blank = "pub const S: &[u8] = b\"foo\\\n    \nbar\";";
    let byte_real_newline = "pub const S: &[u8] = b\"foo\nbar\";";
    let byte_plain = "pub const S: &[u8] = b\"foobar\";";
    assert_ne!(
        surface(byte_escaped_then_blank),
        surface(byte_real_newline),
        "byte-string continuation across a whitespace-only line collided with a genuine embedded newline"
    );
    assert_eq!(
        surface(byte_plain),
        surface(byte_escaped_then_blank),
        "byte-string continuation across a whitespace-only line did not canonicalize to the plain payload"
    );

    // 2D: C strings. CodeLexer has no dedicated C-string grammar - `c"..."`
    // is recognized as ordinary code (`c`) immediately followed by a plain
    // string literal (`"..."`), which is exactly the right boundary for
    // capture purposes (the `c` prefix affects the literal's TYPE, not its
    // payload), so the same NormalString/NormalStringContinuation fix
    // covers it with no separate grammar.
    let c_string_escaped_then_blank = "pub const S: &core::ffi::CStr = c\"foo\\\n    \nbar\";";
    let c_string_real_newline = "pub const S: &core::ffi::CStr = c\"foo\nbar\";";
    let c_string_plain = "pub const S: &core::ffi::CStr = c\"foobar\";";
    assert_ne!(
        surface(c_string_escaped_then_blank),
        surface(c_string_real_newline),
        "C-string continuation across a whitespace-only line collided with a genuine embedded newline"
    );
    assert_eq!(
        surface(c_string_plain),
        surface(c_string_escaped_then_blank),
        "C-string continuation across a whitespace-only line did not canonicalize to the plain payload"
    );

    // Raw (C) strings have no escape-continuation semantics at all - every
    // byte between the delimiters is literal, already guaranteed by the
    // raw-string literal-preservation fix. One regression per shape
    // confirms the `c`/`cr`/`cr#` prefix does not disturb that boundary.
    assert_ne!(
        surface("pub const S: &str = r#\"a  b\"#;"),
        surface("pub const S: &str = r#\"a b\"#;"),
        "raw string literal whitespace must remain unaffected by continuation handling"
    );
    assert_ne!(
        surface("pub const S: &core::ffi::CStr = cr#\"a  b\"#;"),
        surface("pub const S: &core::ffi::CStr = cr#\"a b\"#;"),
        "raw C-string literal whitespace must remain unaffected by continuation handling"
    );
}

#[test]
fn public_api_guard_enum_discriminant_comparison_does_not_poison_generic_depth() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    let base = "const A0: i32 = 1;\nconst B0: i32 = 2;\n\npub enum E {\n    \
                A = { if A0 < B0 { 1 } else { 2 } },\n    B = 3,\n}\n";
    let mutated_following_variant = "const A0: i32 = 1;\nconst B0: i32 = 2;\n\npub enum E {\n    \
                A = { if A0 < B0 { 1 } else { 2 } },\n    B = 4,\n}\n";
    let mutated_block_value = "const A0: i32 = 1;\nconst B0: i32 = 2;\n\npub enum E {\n    \
                A = { if A0 < B0 { 9 } else { 2 } },\n    B = 3,\n}\n";

    assert_ne!(
        surface(base),
        surface(mutated_following_variant),
        "mutating the FOLLOWING variant B's discriminant must change the surface - if it does \
         not, the `<` in A's discriminant left angle_depth stale past A's own declaration"
    );
    assert_ne!(
        surface(base),
        surface(mutated_block_value),
        "mutating the block-expression value inside A's own discriminant must change the surface"
    );
}

#[test]
fn public_api_guard_trait_associated_const_comparison_does_not_poison_following_members() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    let base = "const A0: i32 = 1;\nconst B0: i32 = 2;\nconst C0: i32 = 3;\n\n\
                pub trait T {\n    const LESS: bool = A0 < B0;\n    fn f() -> u32;\n}\n\n\
                pub const NEXT: u32 = 7;\n";
    let mutated_comparison_rhs = "const A0: i32 = 1;\nconst B0: i32 = 2;\nconst C0: i32 = 3;\n\n\
                pub trait T {\n    const LESS: bool = A0 < C0;\n    fn f() -> u32;\n}\n\n\
                pub const NEXT: u32 = 7;\n";
    let mutated_method_sig = "const A0: i32 = 1;\nconst B0: i32 = 2;\nconst C0: i32 = 3;\n\n\
                pub trait T {\n    const LESS: bool = A0 < B0;\n    fn f() -> u64;\n}\n\n\
                pub const NEXT: u32 = 7;\n";
    let mutated_next = "const A0: i32 = 1;\nconst B0: i32 = 2;\nconst C0: i32 = 3;\n\n\
                pub trait T {\n    const LESS: bool = A0 < B0;\n    fn f() -> u32;\n}\n\n\
                pub const NEXT: u32 = 8;\n";

    assert_ne!(
        surface(base),
        surface(mutated_comparison_rhs),
        "A0 < B0 -> A0 < C0 must change the surface (the comparison's own operand is part of \
         the public contract)"
    );
    assert_ne!(
        surface(base),
        surface(mutated_method_sig),
        "the FOLLOWING required method's signature (u32 -> u64) must still be inventoried and \
         change the surface - proving `<` in the preceding associated const did not leave a \
         stale angle_depth that swallows or misparses this method"
    );
    assert_ne!(
        surface(base),
        surface(mutated_next),
        "the top-level declaration AFTER the trait body must remain separately inventoried and \
         change the surface"
    );
    let base_surface = surface(base);
    assert!(
        base_surface.contains("fn f() -> u32;"),
        "required method must remain inventoried: {base_surface:?}"
    );
    assert!(
        base_surface.contains("pub const NEXT: u32 = 7;"),
        "NEXT must remain a separate top-level public declaration: {base_surface:?}"
    );

    // Default method: signature mutation differs, private body mutation does not.
    let default_body_a = "const A0: i32 = 1;\nconst B0: i32 = 2;\n\n\
                pub trait T {\n    const LESS: bool = A0 < B0;\n\n    \
                fn f() -> u32 {\n        private_a()\n    }\n}\n";
    let default_body_b = "const A0: i32 = 1;\nconst B0: i32 = 2;\n\n\
                pub trait T {\n    const LESS: bool = A0 < B0;\n\n    \
                fn f() -> u32 {\n        private_b()\n    }\n}\n";
    let default_sig_u64 = "const A0: i32 = 1;\nconst B0: i32 = 2;\n\n\
                pub trait T {\n    const LESS: bool = A0 < B0;\n\n    \
                fn f() -> u64 {\n        private_a()\n    }\n}\n";
    assert_eq!(
        surface(default_body_a),
        surface(default_body_b),
        "default method body-only mutation (private_a -> private_b) must not change the surface"
    );
    assert_ne!(
        surface(default_body_a),
        surface(default_sig_u64),
        "default method signature mutation (u32 -> u64) must change the surface"
    );
}

#[test]
fn public_api_guard_const_expression_operator_matrix_preserves_generics() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // Every relational/shift operator, in an enum discriminant context,
    // must not corrupt generic depth for the following variant.
    for (op, a_val, b_val) in [
        ("<", "3", "4"),
        (">", "4", "3"),
        ("<=", "3", "4"),
        (">=", "4", "3"),
        ("<<", "1", "2"),
        (">>", "8", "4"),
    ] {
        let build = |b: &str| {
            format!(
                "const A0: i32 = {a_val};\nconst B0: i32 = {b_val};\n\npub enum E {{\n    \
                 A = {{ if A0 {op} B0 {{ 1 }} else {{ 2 }} }},\n    B = {b},\n}}\n"
            )
        };
        assert_ne!(
            surface(&build("3")),
            surface(&build("4")),
            "operator `{op}` in a discriminant expression must not poison depth past A's own \
             declaration - the following variant B must still be independently inventoried"
        );
    }
    // Compound-assignment shift operators, in an enum discriminant context
    // (the trait-associated-const-block-then-method shape below hits a
    // separate, pre-existing scanner limitation unrelated to this fix -
    // see the confirmed-pre-existing note in the commit message; this
    // matrix intentionally sticks to the already-proven-robust enum
    // construction for these two operators).
    for (op, a_val, b_val) in [("<<=", "1", "2"), (">>=", "8", "4")] {
        let build = |b: &str| {
            format!(
                "const A0: i32 = {a_val};\nconst B0: i32 = {b_val};\n\npub enum E {{\n    \
                 A = {{ let mut x = A0; x {op} B0; x as isize }},\n    B = {b},\n}}\n"
            )
        };
        assert_ne!(
            surface(&build("3")),
            surface(&build("4")),
            "compound-assignment operator `{op}` in a discriminant expression must not poison \
             depth past A's own declaration - the following variant B must still be \
             independently inventoried"
        );
    }

    // Existing generic shapes must remain unaffected by this round's change.
    let generic_cases = [
        "pub fn f() -> Foo<Bar<Baz>> { g() }",
        "pub struct S<const N: Vec2<{ 1 }>>;",
        "pub fn f<T: Trait>() -> <T as Trait>::Assoc { g() }",
        "pub const N: usize = core::mem::size_of::<Option<Result<u8, u16>>>();",
    ];
    for src in generic_cases {
        // Must not panic and must produce a stable, non-empty surface -
        // the acceptance bar here is that generic syntax keeps parsing as
        // generics (asserted precisely by the existing dedicated
        // regressions for each shape); this is a smoke check that this
        // round's change does not disturb them when run adjacently.
        let rendered = surface(src);
        assert!(!rendered.trim().is_empty(), "must still parse: {src}");
    }

    // Turbofish specifically inside an enum discriminant and inside a
    // trait associated-const initializer must still open/close a real
    // generic frame (not be treated as a comparison).
    let turbofish_enum_a = "pub enum E {\n    \
                A = core::mem::size_of::<Option<u32>>() as isize,\n    B = 1,\n}\n";
    let turbofish_enum_b = "pub enum E {\n    \
                A = core::mem::size_of::<Option<u64>>() as isize,\n    B = 1,\n}\n";
    assert_ne!(
        surface(turbofish_enum_a),
        surface(turbofish_enum_b),
        "turbofish generic argument inside an enum discriminant must remain significant"
    );
    let turbofish_enum_b_mut = "pub enum E {\n    \
                A = core::mem::size_of::<Option<u32>>() as isize,\n    B = 2,\n}\n";
    assert_ne!(
        surface(turbofish_enum_a),
        surface(turbofish_enum_b_mut),
        "the following variant B must remain independently inventoried after a turbofish-\
         bearing discriminant"
    );

    let turbofish_trait_a = "pub trait T {\n    \
                const N: usize = core::mem::size_of::<Option<u32>>();\n    fn f() -> u32;\n}\n";
    let turbofish_trait_b = "pub trait T {\n    \
                const N: usize = core::mem::size_of::<Option<u64>>();\n    fn f() -> u32;\n}\n";
    assert_ne!(
        surface(turbofish_trait_a),
        surface(turbofish_trait_b),
        "turbofish generic argument inside a trait associated-const initializer must remain \
         significant"
    );
}

#[test]
fn public_api_guard_captures_single_line_inherent_impl_public_method() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 3A: exact public method regression.
    let base = "pub struct S;\nimpl S { pub fn f(&self) -> u32 { private_a() } }\n";
    let sig_u64 = "pub struct S;\nimpl S { pub fn f(&self) -> u64 { private_a() } }\n";
    let body_b = "pub struct S;\nimpl S { pub fn f(&self) -> u32 { private_b() } }\n";
    assert_ne!(
        surface(base),
        surface(sig_u64),
        "a single-line inherent impl's public method signature must be inventoried and change \
         the surface on mutation"
    );
    assert_eq!(
        surface(base),
        surface(body_b),
        "a private method-body-only mutation must not change the surface"
    );
    let base_surface = surface(base);
    assert!(
        base_surface.contains("pub fn f(&self) -> u32"),
        "public method must be captured: {base_surface:?}"
    );
    assert!(
        !base_surface.contains("impl S"),
        "the impl header itself must not appear as its own contract line: {base_surface:?}"
    );
}

#[test]
fn public_api_guard_inherent_impl_formatting_equivalence() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 3B: single-line and multiline impl formatting must be equivalent.
    let single_line = "pub struct S;\nimpl S { pub fn f(&self) -> u32 { private_a() } }\n";
    let multiline =
        "pub struct S;\nimpl S {\n    pub fn f(&self) -> u32 {\n        private_a()\n    }\n}\n";
    assert_eq!(
        surface(single_line),
        surface(multiline),
        "single-line and multiline inherent impl formatting must produce the same public \
         contract surface"
    );
}

#[test]
fn public_api_guard_inherent_impl_associated_const() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 3C: associated const.
    let base = "pub struct S;\nimpl S { pub const N: u32 = 1; }\n";
    let value_2 = "pub struct S;\nimpl S { pub const N: u32 = 2; }\n";
    let type_u64 = "pub struct S;\nimpl S { pub const N: u64 = 1; }\n";
    assert_ne!(
        surface(base),
        surface(value_2),
        "associated const value mutation must change the surface"
    );
    assert_ne!(
        surface(base),
        surface(type_u64),
        "associated const type mutation must change the surface"
    );
}

#[test]
fn public_api_guard_inherent_impl_multiple_same_line_members() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 3D: multiple associated items on one line - both public contracts
    // inventoried, mutating either changes the surface, private method
    // body mutation does not.
    let base = "pub struct S;\n\
                impl S { pub const N: u32 = 1; pub fn f(&self) -> u32 { private_a() } }\n";
    let base_surface = surface(base);
    assert!(
        base_surface.contains("pub const N: u32 = 1;"),
        "associated const must be inventoried: {base_surface:?}"
    );
    assert!(
        base_surface.contains("pub fn f(&self) -> u32"),
        "method must be inventoried: {base_surface:?}"
    );

    let mutated_const = "pub struct S;\n\
                impl S { pub const N: u32 = 2; pub fn f(&self) -> u32 { private_a() } }\n";
    assert_ne!(
        base_surface,
        surface(mutated_const),
        "mutating the const among multiple same-line members must change the surface"
    );

    let mutated_fn_sig = "pub struct S;\n\
                impl S { pub const N: u32 = 1; pub fn f(&self) -> u64 { private_a() } }\n";
    assert_ne!(
        base_surface,
        surface(mutated_fn_sig),
        "mutating the method signature among multiple same-line members must change the surface"
    );

    let mutated_fn_body = "pub struct S;\n\
                impl S { pub const N: u32 = 1; pub fn f(&self) -> u32 { private_b() } }\n";
    assert_eq!(
        base_surface,
        surface(mutated_fn_body),
        "mutating only the private method body among multiple same-line members must not \
         change the surface"
    );
}

#[test]
fn public_api_guard_inherent_impl_private_members_stay_hidden() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 3E: private inherent members.
    let base = "pub struct S;\nimpl S {\n    \
                fn hidden(&self) -> u32 { private_a() }\n    \
                pub fn visible(&self) -> bool { true }\n}\n";
    let hidden_sig_u64 = "pub struct S;\nimpl S {\n    \
                fn hidden(&self) -> u64 { private_a() }\n    \
                pub fn visible(&self) -> bool { true }\n}\n";
    let hidden_body_b = "pub struct S;\nimpl S {\n    \
                fn hidden(&self) -> u32 { private_b() }\n    \
                pub fn visible(&self) -> bool { true }\n}\n";
    assert_eq!(
        surface(base),
        surface(hidden_sig_u64),
        "a private inherent method's own signature mutation must not change the surface"
    );
    assert_eq!(
        surface(base),
        surface(hidden_body_b),
        "a private inherent method's own body mutation must not change the surface"
    );
    let base_surface = surface(base);
    assert!(
        !base_surface.contains("hidden"),
        "private inherent member must never leak into the surface: {base_surface:?}"
    );
    assert!(
        base_surface.contains("pub fn visible(&self) -> bool"),
        "public inherent member must still be captured: {base_surface:?}"
    );
}

#[test]
fn public_api_guard_generic_inherent_impl() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 3F: generic inherent impl.
    let base = "pub struct S<T>(pub T);\nimpl<T> S<T> {\n    \
                pub fn get(&self) -> &T {\n        private_a()\n    }\n}\n";
    let sig_mut = "pub struct S<T>(pub T);\nimpl<T> S<T> {\n    \
                pub fn get(&self) -> &mut T {\n        private_a()\n    }\n}\n";
    let body_b = "pub struct S<T>(pub T);\nimpl<T> S<T> {\n    \
                pub fn get(&self) -> &T {\n        private_b()\n    }\n}\n";
    assert_ne!(
        surface(base),
        surface(sig_mut),
        "generic inherent impl method signature mutation must change the surface"
    );
    assert_eq!(
        surface(base),
        surface(body_b),
        "generic inherent impl method private body mutation must not change the surface"
    );
}

#[test]
fn public_api_guard_trait_impl_never_invents_public_methods() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 3G: trait impl discipline - Rust forbids explicit `pub` inside a
    // trait impl, so none of its methods can ever legally be "public" on
    // their own; the new inherent-impl logic must not treat a trait impl
    // as an inherent public scope and must not invent a contract line
    // for anything inside it.
    let base = "pub struct S;\npub trait Greet {\n    fn hello(&self) -> u32;\n}\n\
                impl Greet for S {\n    fn hello(&self) -> u32 { 0 }\n}\n";
    let mutated_body = "pub struct S;\npub trait Greet {\n    fn hello(&self) -> u32;\n}\n\
                impl Greet for S {\n    fn hello(&self) -> u32 { 1 }\n}\n";
    let mutated_sig = "pub struct S;\npub trait Greet {\n    fn hello(&self) -> u32;\n}\n\
                impl Greet for S {\n    fn hello(&self) -> u64 { 0 }\n}\n";
    assert_eq!(
        surface(base),
        surface(mutated_body),
        "a trait impl method body mutation must not change the surface - nothing inside a \
         trait impl is part of the inherent public contract"
    );
    assert_eq!(
        surface(base),
        surface(mutated_sig),
        "a trait impl method signature mutation must not change the surface either - trait \
         impl methods can never carry explicit visibility"
    );
    let base_surface = surface(base);
    assert!(
        !base_surface.contains("impl Greet"),
        "the trait impl header itself must never appear: {base_surface:?}"
    );
    // "hello" appears exactly once - from the trait's own required
    // method declaration - never a second, invented time from inside
    // the trait impl body.
    assert_eq!(
        base_surface.matches("hello").count(),
        1,
        "\"hello\" must appear exactly once (the trait's own required method), never again from \
         an invented trait-impl line: {base_surface:?}"
    );
    // The trait's OWN required method remains correctly inventoried -
    // this finding is about trait IMPLS, not trait DECLARATIONS.
    assert!(
        base_surface.contains("fn hello(&self) -> u32;"),
        "the trait declaration's own required method must remain inventoried: {base_surface:?}"
    );
}

#[test]
fn public_api_guard_extern_block_public_foreign_items() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 3H: extern block sibling audit. `unsafe extern "C" { ... }` is the
    // Rust 2024 syntax for extern blocks; foreign items inside may carry
    // explicit visibility, and the same single-line-vs-multiline
    // formatting-dependence risk applies as for inherent impls.
    let single_line = "unsafe extern \"C\" { pub fn foreign_api(x: u32) -> u32; }\n";
    let single_line_mut = "unsafe extern \"C\" { pub fn foreign_api(x: u32) -> u64; }\n";
    assert_ne!(
        surface(single_line),
        surface(single_line_mut),
        "a public foreign item inside a single-line extern block must be inventoried and \
         change the surface on mutation"
    );
    let multiline = "unsafe extern \"C\" {\n    pub fn foreign_api(x: u32) -> u32;\n}\n";
    assert_eq!(
        surface(single_line),
        surface(multiline),
        "single-line and multiline extern block formatting must produce the same surface"
    );
    let private_item = "unsafe extern \"C\" { fn hidden_api(x: u32) -> u32; }\n";
    let private_item_mut = "unsafe extern \"C\" { fn hidden_api(x: u32) -> u64; }\n";
    assert_eq!(
        surface(private_item),
        surface(private_item_mut),
        "a private (non-pub) foreign item must not surface or change the output"
    );
    let base_surface = surface(single_line);
    assert!(
        base_surface.contains("pub fn foreign_api(x: u32) -> u32;"),
        "public foreign item must be captured: {base_surface:?}"
    );
}

#[test]
fn public_api_guard_tuple_struct_where_suffix_literal_probe() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // Section 4: mandatory sibling probe - tuple-struct where-suffix
    // literals. Reports its own PASS/FAIL rather than assuming either.
    let real_newline = "pub struct S<T>(pub T)\nwhere\n    T: Bound<Ty!(\"a\nb\")>;\n";
    let one_space = "pub struct S<T>(pub T)\nwhere\n    T: Bound<Ty!(\"a b\")>;\n";
    assert_ne!(
        surface(real_newline),
        surface(one_space),
        "tuple-struct where-suffix literal newline vs space collapsed - this sibling probe \
         FAILED and must be promoted to a fixed P2 in this same batch"
    );

    let escaped_continuation =
        "pub struct S<T>(pub T)\nwhere\n    T: Bound<Ty!(\"a\\\n    b\")>;\n";
    let plain = "pub struct S<T>(pub T)\nwhere\n    T: Bound<Ty!(\"ab\")>;\n";
    assert_eq!(
        surface(escaped_continuation),
        surface(plain),
        "tuple-struct where-suffix escaped continuation must canonicalize like every other \
         escaped-continuation literal in this file"
    );
}

// ====================================================================
// Round 4: opening-line impl-member attributes (P2-1)
// ====================================================================

#[test]
fn public_api_guard_recognizes_attributes_before_opening_line_impl_members() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 1A: exact pre-fix regression.
    let base = "pub struct S;\nimpl S { #[deprecated] pub fn f(&self) -> u32 { 0 } }\n";
    let sig_u64 = "pub struct S;\nimpl S { #[deprecated] pub fn f(&self) -> u64 { 0 } }\n";
    assert_ne!(
        surface(base),
        surface(sig_u64),
        "an attributed public method on the impl's own opening line must be inventoried and \
         change the surface on signature mutation"
    );
    let body_only = "pub struct S;\nimpl S { #[deprecated] pub fn f(&self) -> u32 { 1 } }\n";
    assert_eq!(
        surface(base),
        surface(body_only),
        "body-only mutation of an attributed opening-line public method must not change the \
         surface"
    );

    // 1B: attribute with a literal - no blind literal normalization.
    let lit_two = "pub struct S;\n\
                impl S { #[deprecated(note = \"a  b\")] pub fn f(&self) -> u32 { 0 } }\n";
    let lit_one = "pub struct S;\n\
                impl S { #[deprecated(note = \"a b\")] pub fn f(&self) -> u32 { 0 } }\n";
    assert_ne!(
        surface(lit_two),
        surface(lit_one),
        "an attribute literal on an opening-line impl member must preserve its own whitespace, \
         not be blindly normalized"
    );

    // Multiline attribute before an opening-line-adjacent member; formatting-
    // equivalent one-line/multiline forms must match, and the public member
    // must still be captured.
    let multiline_attr = "pub struct S;\nimpl S {\n    #[deprecated(\n        note = \"x\"\n    )]\n    pub fn f(&self) -> u32 { 0 }\n}\n";
    let multiline_attr_surface = surface(multiline_attr);
    assert!(
        multiline_attr_surface.contains("pub fn f(&self) -> u32"),
        "public member after a multiline attribute must be captured: {multiline_attr_surface:?}"
    );

    // Multiple attributes must both remain associated with the public member.
    let multi_attr =
        "pub struct S;\nimpl S {\n    #[cfg(feature = \"x\")]\n    #[deprecated]\n    \
                pub fn f(&self) -> u32 { 0 }\n}\n";
    let multi_attr_surface = surface(multi_attr);
    assert!(
        multi_attr_surface.contains("cfg"),
        "first of two attributes must be retained: {multi_attr_surface:?}"
    );
    assert!(
        multi_attr_surface.contains("deprecated"),
        "second of two attributes must be retained: {multi_attr_surface:?}"
    );
    assert!(
        multi_attr_surface.contains("pub fn f(&self) -> u32"),
        "public member after two attributes must be captured: {multi_attr_surface:?}"
    );

    // Attribute + PRIVATE member: neither the member nor its attribute may
    // leak as an orphan public-contract line.
    let private_attr =
        "pub struct S;\nimpl S {\n    #[deprecated]\n    fn hidden(&self) -> u32 { 0 }\n}\n";
    let private_attr_surface = surface(private_attr);
    assert!(
        !private_attr_surface.contains("hidden"),
        "a private member must not leak: {private_attr_surface:?}"
    );
    assert!(
        !private_attr_surface.contains("deprecated"),
        "a private member's own attribute must not leak as an orphan line: \
         {private_attr_surface:?}"
    );

    // Attribute after a PREVIOUS same-line member.
    let after_prev = "pub struct S;\nimpl S {\n    pub const N: u32 = 1;\n    \
                #[deprecated] pub fn f(&self) -> u32 { 0 }\n}\n";
    let after_prev_surface = surface(after_prev);
    assert!(
        after_prev_surface.contains("pub const N: u32 = 1;"),
        "the earlier member must remain inventoried: {after_prev_surface:?}"
    );
    assert!(
        after_prev_surface.contains("pub fn f(&self) -> u32"),
        "the attributed later member must also be inventoried: {after_prev_surface:?}"
    );

    // Entire impl, including the attribute, on ONE physical line.
    let all_one_line =
        "pub struct S;\nimpl S { pub const N: u32 = 1; #[deprecated] pub fn f(&self) -> u32 { 0 } }\n";
    let all_one_line_surface = surface(all_one_line);
    assert!(
        all_one_line_surface.contains("pub const N: u32 = 1;"),
        "first member on a fully single-line impl must be inventoried: {all_one_line_surface:?}"
    );
    assert!(
        all_one_line_surface.contains("pub fn f(&self) -> u32"),
        "attributed second member on a fully single-line impl must be inventoried: \
         {all_one_line_surface:?}"
    );
}

// ====================================================================
// Round 4: HRTB vs impl-level `for` (P2-2)
// ====================================================================

#[test]
fn public_api_guard_distinguishes_hrtb_from_impl_level_for() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 2A: exact pre-fix regression.
    let base = "pub struct S<T>(pub T);\n\
                impl<T: for<'a> Fn(&'a str)> S<T> {\n    pub fn f(&self) -> u32 { 0 }\n}\n";
    let mutated = "pub struct S<T>(pub T);\n\
                impl<T: for<'a> Fn(&'a str)> S<T> {\n    pub fn f(&self) -> u64 { 0 }\n}\n";
    assert_ne!(
        surface(base),
        surface(mutated),
        "an inherent impl whose generic bound contains an HRTB `for<'a>` must still be \
         classified as inherent, and its public method must be inventoried"
    );

    // 2B: required inherent-impl matrix - all must classify INHERENT.
    for (src, label) in [
        (
            "pub struct S;\nimpl S {\n    pub fn f(&self) -> u32 { 0 }\n}\n",
            "plain inherent impl",
        ),
        (
            "pub struct S<T>(pub T);\nimpl<T> S<T> {\n    pub fn f(&self) -> u32 { 0 }\n}\n",
            "generic inherent impl",
        ),
        (
            "pub struct S<const N: usize>;\nimpl<const N: usize> S<N> {\n    \
             pub fn f(&self) -> u32 { 0 }\n}\n",
            "const-generic inherent impl",
        ),
        (
            "pub struct S<T>(pub T);\nimpl<T: for<'a> Fn(&'a str)> S<T> {\n    \
             pub fn f(&self) -> u32 { 0 }\n}\n",
            "HRTB generic-bound inherent impl",
        ),
        (
            "pub struct S<T>(pub T);\nimpl<T> S<T>\nwhere\n    T: for<'a> Fn(&'a str),\n{\n    \
             pub fn f(&self) -> u32 { 0 }\n}\n",
            "HRTB where-clause inherent impl",
        ),
    ] {
        let out = surface(src);
        assert!(
            out.contains("pub fn f(&self) -> u32"),
            "{label} must classify as INHERENT and inventory its public method: {out:?}"
        );
    }

    // 2C: required trait-impl NEGATIVE matrix - none of these may leak a
    // method as an invented inherent public item, and their `for`-bearing
    // header must never itself appear.
    for (src, label) in [
        (
            "pub struct S;\npub trait Trait { fn f(&self) -> u32; }\n\
             impl Trait for S {\n    fn f(&self) -> u32 { 0 }\n}\n",
            "plain trait impl",
        ),
        (
            "pub struct S<T>(pub T);\npub trait Trait { fn f(&self) -> u32; }\n\
             impl<T> Trait for S<T> {\n    fn f(&self) -> u32 { 0 }\n}\n",
            "generic trait impl",
        ),
        (
            "pub struct S<T>(pub T);\npub trait Trait<U> { fn f(&self) -> u32; }\n\
             impl<T> Trait<T> for S<T> {\n    fn f(&self) -> u32 { 0 }\n}\n",
            "generic trait-with-param impl",
        ),
        (
            "pub struct S;\npub unsafe trait Trait { fn f(&self) -> u32; }\n\
             unsafe impl Trait for S {\n    fn f(&self) -> u32 { 0 }\n}\n",
            "unsafe trait impl",
        ),
        (
            "pub struct S<T>(pub T);\npub trait Trait { fn f(&self) -> u32; }\n\
             impl<T> Trait for S<T>\nwhere\n    T: for<'a> Fn(&'a str),\n{\n    \
             fn f(&self) -> u32 { 0 }\n}\n",
            "trait impl with where-clause HRTB",
        ),
    ] {
        let out = surface(src);
        assert!(
            !out.contains("impl") || !out.contains("for"),
            "{label}: the trait impl header itself must never appear verbatim: {out:?}"
        );
        assert_eq!(
            out.matches("fn f").count(),
            1,
            "{label}: \"fn f\" must appear exactly once (the trait's own required method), \
             never a second, invented time from inside the trait impl: {out:?}"
        );
    }
}

// ====================================================================
// Round 4: incomplete extern-block headers (P2-3)
// ====================================================================

#[test]
fn public_api_guard_recognizes_extern_block_before_continuation_brace() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // 3A: exact pre-fix regression - brace on the line AFTER the ABI string.
    let base = "unsafe extern \"C\"\n{\n    pub fn f() -> u32;\n}\n";
    let mutated = "unsafe extern \"C\"\n{\n    pub fn f() -> u64;\n}\n";
    assert_ne!(
        surface(base),
        surface(mutated),
        "an extern block whose opening brace is on the line after the ABI string must still \
         inventory its public foreign item and change the surface on mutation"
    );

    // 3B: layout matrix - all formatting-equivalent.
    let same_line = "unsafe extern \"C\" { pub fn f() -> u32; }\n";
    let brace_after_header_tight = "unsafe extern \"C\"\n{ pub fn f() -> u32; }\n";
    let fully_multiline = "unsafe extern \"C\"\n{\n    pub fn f() -> u32;\n}\n";
    assert_eq!(
        surface(same_line),
        surface(brace_after_header_tight),
        "same-line vs brace-on-next-line extern headers must be formatting-equivalent"
    );
    assert_eq!(
        surface(same_line),
        surface(fully_multiline),
        "same-line vs fully-multiline extern headers must be formatting-equivalent"
    );

    // 3C: multiple foreign items - mutating either changes the surface.
    let multi = "unsafe extern \"C\" {\n    pub fn f() -> u32;\n    pub static X: u32;\n}\n";
    let multi_mut_fn = "unsafe extern \"C\" {\n    pub fn f() -> u64;\n    pub static X: u32;\n}\n";
    let multi_mut_static =
        "unsafe extern \"C\" {\n    pub fn f() -> u32;\n    pub static X: u64;\n}\n";
    assert_ne!(surface(multi), surface(multi_mut_fn));
    assert_ne!(surface(multi), surface(multi_mut_static));
    let multi_surface = surface(multi);
    assert!(multi_surface.contains("pub fn f() -> u32;"));
    assert!(multi_surface.contains("pub static X: u32;"));

    // 3E: ABI variants - not hard-coded to "C" only.
    for abi in ["\"C\"", "\"system\"", "\"C-unwind\""] {
        let src = format!("unsafe extern {abi} {{ pub fn f() -> u32; }}\n");
        let mutated = format!("unsafe extern {abi} {{ pub fn f() -> u64; }}\n");
        assert_ne!(
            surface(&src),
            surface(&mutated),
            "extern block with ABI {abi} must inventory its public foreign item"
        );
    }

    // 3F: a single foreign FUNCTION DECLARATION (not a block) must remain
    // unaffected, and an ordinary `pub extern "C" fn` definition (with a
    // real body) must not enter block-style member recursion.
    let plain_extern_fn = "pub extern \"C\" fn f() -> u32 { 0 }\n";
    let plain_extern_fn_mut = "pub extern \"C\" fn f() -> u64 { 0 }\n";
    assert_ne!(
        surface(plain_extern_fn),
        surface(plain_extern_fn_mut),
        "a plain `pub extern \"C\" fn` definition must still be captured on its own terms"
    );
    let plain_extern_fn_surface = surface(plain_extern_fn);
    assert!(
        plain_extern_fn_surface.contains("pub extern \"C\" fn f() -> u32"),
        "plain extern fn definition must be captured normally: {plain_extern_fn_surface:?}"
    );

    // extern crate must remain unaffected.
    let extern_crate_a = "extern crate foo;\npub struct S;\n";
    let extern_crate_b = "extern crate bar;\npub struct S;\n";
    assert_eq!(
        surface(extern_crate_a),
        surface(extern_crate_b),
        "extern crate declarations are not part of the public contract inventory and must not \
         be affected by extern-block recognition"
    );
}

// ====================================================================
// Round-3 final audit: reflow across the impl-for / extern-block / same-
// line-member-boundary changes above must not change classification -
// each of these mirrors an already-required shape but adds a legal line
// wrap (or a construct combination) the primary tests above did not
// separately cover, per the final "logical-header reflow" and "same-
// line member-state" audits.
// ====================================================================

#[test]
fn public_api_guard_final_audit_reflow_and_same_line_symmetry() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // R1: trait impl with `for` itself wrapped onto its own line must
    // still classify as a TRAIT impl (never invent an inherent method),
    // matching 2C's negative matrix but with `for` line-wrapped instead
    // of on the same line as the trait name.
    let wrapped_for = "pub struct S;\npub trait Trait { fn f(&self) -> u32; }\n\
                        impl Trait\n    for\n    S\n{\n    fn f(&self) -> u32 { 0 }\n}\n";
    let wrapped_for_surface = surface(wrapped_for);
    assert!(
        !wrapped_for_surface.contains("impl"),
        "impl header must never itself be inventoried: {wrapped_for_surface:?}"
    );
    assert_eq!(
        wrapped_for_surface.matches("fn f").count(),
        1,
        "a trait impl with `for` wrapped onto its own line must not invent a second, inherent \
         `fn f`: {wrapped_for_surface:?}"
    );

    // R2: unsafe impl, `for` and the Self type both wrapped, HRTB in the
    // where-clause too - combines every reflow point in one header.
    let wrapped_unsafe = "pub struct S;\npub unsafe trait Trait { fn f(&self) -> u32; }\n\
                           unsafe impl Trait\n    for S\nwhere\n    S: Sized,\n{\n    \
                           fn f(&self) -> u32 { 0 }\n}\n";
    let wrapped_unsafe_surface = surface(wrapped_unsafe);
    assert_eq!(
        wrapped_unsafe_surface.matches("fn f").count(),
        1,
        "unsafe trait impl with `for` and Self both wrapped must not invent a second, inherent \
         `fn f`: {wrapped_unsafe_surface:?}"
    );

    // R3: an outer attribute directly before an extern block header must
    // be dropped, not mis-attached to the block's first foreign item -
    // the same rule already required for impl blocks (is_impl_start
    // branch) must hold symmetrically for is_incomplete_extern_prefix.
    let extern_attr =
        "#[cfg(feature = \"ffi\")]\nunsafe extern \"C\" {\n    pub fn f() -> u32;\n}\n";
    let extern_attr_surface = surface(extern_attr);
    assert!(
        !extern_attr_surface.contains("cfg"),
        "an attribute before an extern block header must not be mis-attached to its first \
         foreign item: {extern_attr_surface:?}"
    );
    assert!(
        extern_attr_surface.contains("pub fn f() -> u32;"),
        "the foreign item itself must still be inventoried: {extern_attr_surface:?}"
    );

    // R4: extern block, entirely on one physical line, with an
    // attributed SECOND foreign item following an unattributed first one
    // - the same same-line-member-boundary path already required for
    // impl bodies (public_api_guard_recognizes_attributes_before_opening
    // _line_impl_members) must hold symmetrically for extern blocks,
    // since both route through the same `scan_baseline_member_body`.
    let extern_one_line =
        "unsafe extern \"C\" { pub static X: u32; #[link_name = \"g\"] pub fn f() -> u32; }\n";
    let extern_one_line_surface = surface(extern_one_line);
    assert!(
        extern_one_line_surface.contains("pub static X: u32;"),
        "first (unattributed) foreign item on a fully single-line extern block must be \
         inventoried: {extern_one_line_surface:?}"
    );
    assert!(
        extern_one_line_surface.contains("pub fn f() -> u32;"),
        "second (attributed) foreign item on a fully single-line extern block must be \
         inventoried, not swallowed as part of the first: {extern_one_line_surface:?}"
    );
    assert!(
        extern_one_line_surface.contains("link_name"),
        "the second item's own attribute must be captured: {extern_one_line_surface:?}"
    );

    // R5: `pub(crate)`/`pub(in path)` restricted-visibility members
    // inside an inherent impl, formatting-independent (all on one line
    // vs multiline), must remain equivalent - the brace_depth-round-trip
    // fix must not be specific to `pub fn`/`pub const`.
    let restricted_one_line = "pub struct S;\nimpl S { pub(crate) fn f() -> u32 { 0 } }\n";
    let restricted_multiline =
        "pub struct S;\nimpl S {\n    pub(crate) fn f() -> u32 {\n        0\n    }\n}\n";
    assert_eq!(
        surface(restricted_one_line),
        surface(restricted_multiline),
        "a fully single-line inherent impl with a pub(crate) member must be formatting-\
         equivalent to its multiline form"
    );

    // R6: a non-pub (private) foreign item inside a fully single-line
    // extern block must stay hidden, symmetric with R5's inherent-impl
    // check - `should_emit` for extern blocks is plain `is_public_code`
    // (no `is_inherent` gate), so this exercises a different `should_
    // emit` closure than every other single-line test above.
    let extern_private_one_line =
        "unsafe extern \"C\" { fn hidden() -> u32; pub fn f() -> u32; }\n";
    let extern_private_one_line_surface = surface(extern_private_one_line);
    assert!(
        !extern_private_one_line_surface.contains("hidden"),
        "a private foreign item on a fully single-line extern block must not leak: \
         {extern_private_one_line_surface:?}"
    );
    assert!(
        extern_private_one_line_surface.contains("pub fn f() -> u32;"),
        "the public foreign item following it must still be inventoried: \
         {extern_private_one_line_surface:?}"
    );
}

// ====================================================================
// Round 4: fresh P2s from the exact-head review of 3ad0177a -
// opening-line member replay state, private extern lookahead leaking
// into the public dispatcher, and `where` separated from its bound list
// by a block comment.
// ====================================================================

#[test]
fn public_api_guard_replays_opening_line_members_from_body_open_state() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // P2-1A: exact Codex case - a member signature that starts on the
    // impl's own opening line and continues onto a later physical line.
    let base = "pub struct S;\n\nimpl S { pub fn f(\n    &self,\n) -> u32 { 0 } }\n";
    let mutated = "pub struct S;\n\nimpl S { pub fn f(\n    &self,\n) -> u64 { 0 } }\n";
    assert_ne!(
        surface(base),
        surface(mutated),
        "a member signature that opens on the impl's own line and continues onto a later \
         physical line must be inventoried and change the surface on mutation"
    );

    // P2-1B: formatting equivalence - opening-line-start vs impl-brace-
    // on-its-own-line, same multiline signature either way.
    let normal_form =
        "pub struct S;\n\nimpl S {\n    pub fn f(\n        &self,\n    ) -> u32 { 0 }\n}\n";
    assert_eq!(
        surface(base),
        surface(normal_form),
        "an opening-line member whose signature continues onto another physical line must be \
         formatting-equivalent to the same member with the impl's own brace on its own line"
    );

    // P2-1C: private method body mutation on the opening-line, multiline
    // form must not change the surface.
    let body_only = "pub struct S;\n\nimpl S { pub fn f(\n    &self,\n) -> u32 { 1 } }\n";
    assert_eq!(
        surface(base),
        surface(body_only),
        "a private body-only mutation on an opening-line member with a multiline signature must \
         not change the surface"
    );

    // P2-1D: replay matrix - opening-line member fragments ending in a
    // still-open delimiter/literal/clause, continuing onto the next
    // physical line. Each checks: contract mutation -> DIFFERENT,
    // formatting-only reflow -> EQUAL, private body mutation -> EQUAL.
    struct Case {
        label: &'static str,
        base: &'static str,
        mutated: &'static str,
        reflow: &'static str,
        body_mut: &'static str,
    }
    let cases = [
        Case {
            label: "open paren (generic return type)",
            base: "pub struct S;\nimpl S { pub fn f() -> Vec<\n    u32,\n> { Vec::new() } }\n",
            mutated: "pub struct S;\nimpl S { pub fn f() -> Vec<\n    u64,\n> { Vec::new() } }\n",
            reflow: "pub struct S;\nimpl S {\n    pub fn f() -> Vec<\n        u32,\n    > { \
                      Vec::new() }\n}\n",
            body_mut: "pub struct S;\nimpl S { pub fn f() -> Vec<\n    u32,\n> { \
                        Vec::with_capacity(1) } }\n",
        },
        Case {
            label: "open bracket (array return type)",
            base: "pub struct S;\nimpl S { pub fn f() -> [\n    u32; 2\n] { [1, 2] } }\n",
            mutated: "pub struct S;\nimpl S { pub fn f() -> [\n    u64; 2\n] { [1, 2] } }\n",
            reflow: "pub struct S;\nimpl S {\n    pub fn f() -> [\n        u32; 2\n    ] { [1, \
                      2] }\n}\n",
            body_mut: "pub struct S;\nimpl S { pub fn f() -> [\n    u32; 2\n] { [3, 4] } }\n",
        },
        Case {
            label: "escaped-continuation normal string literal",
            base: "pub struct S;\nimpl S { pub const N: &str = \"abc\\\n    def\"; }\n",
            mutated: "pub struct S;\nimpl S { pub const N: &str = \"abc\\\n    xyz\"; }\n",
            reflow: "pub struct S;\nimpl S {\n    pub const N: &str = \"abc\\\n        def\";\n}\n",
            // A const's own value IS its public signature - there is no
            // private body to mutate independently, unlike every other
            // case in this matrix, so `body_mut` is set equal to `base`
            // (trivially satisfying the same-surface check).
            body_mut: "pub struct S;\nimpl S { pub const N: &str = \"abc\\\n    def\"; }\n",
        },
        Case {
            label: "raw string literal spanning a real newline",
            base: "pub struct S;\nimpl S { pub const N: &str = r#\"\nraw\n\"#; }\n",
            mutated: "pub struct S;\nimpl S { pub const N: &str = r#\"\nRAW\n\"#; }\n",
            reflow: "pub struct S;\nimpl S {\n    pub const N: &str = r#\"\nraw\n\"#;\n}\n",
            // Same reasoning as above - a const has no private body.
            body_mut: "pub struct S;\nimpl S { pub const N: &str = r#\"\nraw\n\"#; }\n",
        },
        Case {
            label: "type-position macro with paren delimiter",
            base: "pub struct S;\nimpl S { pub fn f() -> type_macro!(\n    u32,\n) { 0 } }\n",
            mutated: "pub struct S;\nimpl S { pub fn f() -> type_macro!(\n    u64,\n) { 0 } }\n",
            reflow: "pub struct S;\nimpl S {\n    pub fn f() -> type_macro!(\n        u32,\n    ) \
                      { 0 }\n}\n",
            body_mut: "pub struct S;\nimpl S { pub fn f() -> type_macro!(\n    u32,\n) { 1 } }\n",
        },
        Case {
            label: "where clause on an opening-line method signature",
            base: "pub struct S;\nimpl S { pub fn f<T>() -> u32\nwhere\n    T: Clone,\n{ 0 } }\n",
            mutated: "pub struct S;\nimpl S { pub fn f<T>() -> u64\nwhere\n    T: Clone,\n{ 0 } \
                       }\n",
            reflow: "pub struct S;\nimpl S {\n    pub fn f<T>() -> u32\n    where\n        T: \
                      Clone,\n    { 0 }\n}\n",
            body_mut: "pub struct S;\nimpl S { pub fn f<T>() -> u32\nwhere\n    T: Clone,\n{ 1 } \
                        }\n",
        },
    ];
    for case in cases {
        let base_surface = surface(case.base);
        assert_ne!(
            base_surface,
            surface(case.mutated),
            "{}: a public contract mutation must change the surface: {base_surface:?}",
            case.label
        );
        assert_eq!(
            base_surface,
            surface(case.reflow),
            "{}: formatting-only reflow must not change the surface: {base_surface:?}",
            case.label
        );
        assert_eq!(
            base_surface,
            surface(case.body_mut),
            "{}: a private body-only mutation must not change the surface: {base_surface:?}",
            case.label
        );
    }

    // P2-1E: associated const whose type continues onto the next line.
    let const_base = "pub struct S;\nimpl S { pub const N:\n    u32 = 1;\n}\n";
    let const_mutated = "pub struct S;\nimpl S { pub const N:\n    u32 = 2;\n}\n";
    let const_surface = surface(const_base);
    assert!(
        const_surface.contains("pub const N: u32 = 1;"),
        "an associated const whose type continues onto the next physical line must be \
         inventoried: {const_surface:?}"
    );
    assert_ne!(
        const_surface,
        surface(const_mutated),
        "mutating the const's value must change the surface: {const_surface:?}"
    );
}

#[test]
fn public_api_guard_extern_lookahead_falls_back_to_private_path() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // P2-2A: `extern` then `crate foo;` on the next line - a complete,
    // PRIVATE (not part of the public contract) item once resolved. Must
    // not panic and must not depend on the crate name.
    let crate_a = "extern\ncrate foo;\npub struct S;\n";
    let crate_b = "extern\ncrate bar;\npub struct S;\n";
    assert_eq!(
        surface(crate_a),
        surface(crate_b),
        "extern crate declarations are not part of the public contract, including when `extern` \
         and `crate` land on different physical lines"
    );

    // P2-2B: `extern` then `"C" fn hidden() {}` on the next line - a
    // complete, PRIVATE (non-pub) foreign-function definition once
    // resolved. Must not panic and a private-only mutation must not
    // change the surface.
    let fn_a = "extern\n\"C\" fn hidden() -> u32 { 0 }\npub struct S;\n";
    let fn_b = "extern\n\"C\" fn hidden() -> u64 { 1 }\npub struct S;\n";
    assert_eq!(
        surface(fn_a),
        surface(fn_b),
        "a private extern fn definition split across `extern` and its ABI/signature must not \
         leak into or otherwise affect the public contract"
    );

    // P2-2C: the public counterpart must remain unaffected by the new
    // lookahead gate - split across lines the same way.
    let pub_a = "pub extern\n\"C\" fn visible() -> u32 { 0 }\n";
    let pub_b = "pub extern\n\"C\" fn visible() -> u64 { 0 }\n";
    assert_ne!(
        surface(pub_a),
        surface(pub_b),
        "a public extern fn definition split across `extern` and its ABI/signature must still \
         be inventoried and change the surface on mutation: {:?}",
        surface(pub_a)
    );
    assert!(
        surface(pub_a).contains("extern \"C\" fn visible() -> u32"),
        "the public extern fn must actually be captured, not merely fail to panic: {:?}",
        surface(pub_a)
    );

    // P2-2D: private extern fn immediately followed by a genuine public
    // item - the fallback to the private path must not corrupt scanning
    // of what comes after it.
    let followed = "extern\ncrate foo;\npub const AFTER: u32 = 1;\n";
    let followed_surface = surface(followed);
    assert!(
        followed_surface.contains("pub const AFTER: u32 = 1;"),
        "an item following a lookahead-resolved private extern crate must still be inventoried: \
         {followed_surface:?}"
    );
}

#[test]
fn public_api_guard_recognizes_where_before_a_block_comment() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // P2-3A: exact Codex case.
    let base =
        "pub struct S<T>(pub T);\nimpl<T> S<T> where/**/T: for<'a> Fn(&'a str) {\n    pub fn f(&self) -> u32 { 0 }\n}\n";
    let mutated =
        "pub struct S<T>(pub T);\nimpl<T> S<T> where/**/T: for<'a> Fn(&'a str) {\n    pub fn f(&self) -> u64 { 0 }\n}\n";
    assert_ne!(
        surface(base),
        surface(mutated),
        "an inherent impl whose `where` is immediately followed by a block comment must still \
         classify as inherent and inventory its public method"
    );

    // P2-3B: trivia matrix - every shape below must classify identically
    // (same public inventory) for an inherent impl.
    let shapes = [
        ("where T:", "impl<T> S<T> where T: for<'a> Fn(&'a str) {"),
        ("where\\nT:", "impl<T> S<T> where\nT: for<'a> Fn(&'a str) {"),
        (
            "where/**/T:",
            "impl<T> S<T> where/**/T: for<'a> Fn(&'a str) {",
        ),
        (
            "where /* comment */ T:",
            "impl<T> S<T> where /* comment */ T: for<'a> Fn(&'a str) {",
        ),
        (
            "where\\n/* comment */\\nT:",
            "impl<T> S<T> where\n/* comment */\nT: for<'a> Fn(&'a str) {",
        ),
    ];
    let reference = surface(&format!(
        "pub struct S<T>(pub T);\n{}\n    pub fn f(&self) -> u32 {{ 0 }}\n}}\n",
        shapes[0].1
    ));
    for (label, header) in &shapes {
        let src =
            format!("pub struct S<T>(pub T);\n{header}\n    pub fn f(&self) -> u32 {{ 0 }}\n}}\n");
        assert_eq!(
            reference,
            surface(&src),
            "{label}: every `where`-before-HRTB trivia shape must classify the impl identically"
        );
    }

    // P2-3C: trait-impl counterpart must remain classified as a trait
    // impl (never invent an inherent public method) with the same
    // where/block-comment trivia.
    let trait_impl_base = "pub struct S;\npub trait Trait { fn f(&self) -> u32; }\nimpl Trait \
                            for S where/**/S: Sized {\n    fn f(&self) -> u32 { 0 }\n}\n";
    let trait_impl_surface = surface(trait_impl_base);
    assert_eq!(
        trait_impl_surface.matches("fn f").count(),
        1,
        "a trait impl with `where` immediately followed by a block comment must not invent a \
         second, inherent `fn f`: {trait_impl_surface:?}"
    );
    assert!(
        !trait_impl_surface.contains("impl"),
        "the impl header itself must never be inventoried: {trait_impl_surface:?}"
    );
}

// ====================================================================
// Round 4 mandatory member-replay audit: a member-boundary split (or an
// attribute captured off it) whose OWN remainder itself ends in an
// escaped-newline continuation must not lose that continuation on
// replay - the same P2-1 backslash-loss finding, reached through a
// different split point than the opening line. Confirmed pre-existing
// on unmodified 3ad0177a (both members vanished entirely, not just the
// second), not a regression from this round's other fixes.
// ====================================================================

#[test]
fn public_api_guard_member_boundary_split_preserves_escaped_continuation() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // Exact regression: an attribute-boundary split (found mid-`sc`,
    // after an earlier same-line member) immediately followed by a
    // member whose own string literal continues onto the next physical
    // line.
    let base = "pub struct S;\nimpl S { pub const N: u32 = 1; #[deprecated] pub const M: &str = \
                \"abc\\\n    def\"; }\n";
    let mutated = "pub struct S;\nimpl S { pub const N: u32 = 1; #[deprecated] pub const M: &str \
                    = \"abc\\\n    xyz\"; }\n";
    let base_surface = surface(base);
    assert!(
        base_surface.contains("pub const N: u32 = 1;"),
        "the member BEFORE the attribute boundary must still be inventoried: {base_surface:?}"
    );
    assert!(
        base_surface.contains("pub const M: &str = \"abcdef\";"),
        "the attributed member's own escaped continuation must survive the boundary split, not \
         be silently dropped along with the whole member: {base_surface:?}"
    );
    assert_ne!(
        base_surface,
        surface(mutated),
        "mutating the value past the boundary split must change the surface: {base_surface:?}"
    );

    // Same shape, but the split point is `capture_attribute`'s OWN
    // internal remainder (single member, attribute right after the
    // body's own `{`) rather than `member_boundary_attribute`'s.
    let opening = "pub struct S;\nimpl S { #[deprecated] pub const M: &str = \"abc\\\n    def\"; \
                    }\n";
    let opening_mutated = "pub struct S;\nimpl S { #[deprecated] pub const M: &str = \"abc\\\n    \
                            xyz\"; }\n";
    let opening_surface = surface(opening);
    assert!(
        opening_surface.contains("pub const M: &str = \"abcdef\";"),
        "an opening-line attribute's own remainder ending in an escaped continuation must \
         survive: {opening_surface:?}"
    );
    assert_ne!(
        opening_surface,
        surface(opening_mutated),
        "mutating the value must change the surface: {opening_surface:?}"
    );
}

#[test]
fn public_api_guard_same_line_remainder_preserves_escaped_continuation() {
    let surface = |src: &str| normalized_public_surface_str("test.rs", src);

    // A same-line boundary (`scan_logical_item_line_with_terminator`'s
    // own `same_line_remainder` split, for a second logical item on the
    // same physical line as an earlier one) whose own remainder ends in
    // an escaped-newline continuation must not lose it - the same class
    // of finding as the member-boundary-attribute case above, reached
    // through the top-level same-line-item path instead.
    let base = "pub struct S; pub const M: &str = \"abc\\\n    def\";\n";
    let mutated = "pub struct S; pub const M: &str = \"abc\\\n    xyz\";\n";
    let base_surface = surface(base);
    assert!(
        base_surface.contains("pub const M: &str = \"abcdef\";"),
        "the second same-line item's own escaped continuation must survive the boundary split: \
         {base_surface:?}"
    );
    assert_ne!(
        base_surface,
        surface(mutated),
        "mutating the value must change the surface: {base_surface:?}"
    );
}
