use crate::exobyte_format::{write_i32_le, write_u16_le, write_u32_le, Opcode, MAGIC};
use std::collections::HashMap;

pub const EXOCODE_EBNF: &str = r#"
Program      = { Function } ;
Function     = "fn" Ident "(" [ Param { "," Param } ] ")" [ "->" Type ] Block ;
Param        = Ident ":" Type ;
Type         = "quad" | "bool" | "i32" | "u32" | "fx" ;
Block        = "{" { Stmt } "}" ;
Stmt         = LetStmt | IfStmt | MatchStmt | ReturnStmt | ExprStmt ;
LetStmt      = "let" Ident [ ":" Type ] "=" Expr ";" ;
IfStmt       = "if" Expr Block [ "else" ( Block | IfStmt ) ] ;
MatchStmt    = "match" Expr "{" MatchArm { MatchArm } "}" ;
MatchArm     = ( "N" | "F" | "T" | "S" | "_" ) "=>" Block ;
ReturnStmt   = "return" [ Expr ] ";" ;
ExprStmt     = Expr ";" ;
Expr         = Impl ;
Impl         = Or { "->" Or } ;                // '->' is quad-only
Or           = And { "||" And } ;              // quad or bool
And          = Eq { "&&" Eq } ;                // quad or bool
Eq           = Unary { ("==" | "!=") Unary } ; // returns bool
Unary        = [ "!" ] Primary ;
Primary      = QuadLit | BoolLit | Num | Ident | Call | "(" Expr ")" ;
Call         = Ident "(" [ Expr { "," Expr } ] ")" ;
QuadLit      = "N" | "F" | "T" | "S" ;
BoolLit      = "true" | "false" ;
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    Quad,
    Bool,
    I32,
    U32,
    Fx,
    Unit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuadVal {
    N,
    F,
    T,
    S,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    AndAnd,
    OrOr,
    Implies,
    Eq,
    Ne,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    QuadLiteral(QuadVal),
    BoolLiteral(bool),
    Num(i64),
    Var(String),
    Call(String, Vec<Expr>),
    Unary(UnaryOp, Box<Expr>),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<Type>,
        value: Expr,
    },
    If {
        condition: Expr,
        then_block: Vec<Stmt>,
        else_block: Vec<Stmt>,
    },
    Match {
        scrutinee: Expr,
        arms: Vec<MatchArm>,
        default: Vec<Stmt>,
    },
    Return(Option<Expr>),
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchArm {
    pub pat: QuadVal,
    pub block: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub ret: Type,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub functions: Vec<Function>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSig {
    pub params: Vec<Type>,
    pub ret: Type,
}

pub type FnTable = HashMap<String, FnSig>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    KwFn,
    KwLet,
    KwIf,
    KwElse,
    KwReturn,
    KwMatch,
    KwTrue,
    KwFalse,
    TyQuad,
    TyBool,
    TyI32,
    TyU32,
    TyFx,
    QuadN,
    QuadF,
    QuadT,
    QuadS,
    Ident,
    Num,
    AndAnd,
    OrOr,
    Bang,
    Arrow,
    FatArrow,
    EqEq,
    Ne,
    Assign,
    LBrace,
    RBrace,
    LParen,
    RParen,
    Semi,
    Comma,
    Colon,
    Underscore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
    pub pos: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrontendError {
    pub pos: usize,
    pub message: String,
}

impl core::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "at {}: {}", self.pos, self.message)
    }
}

impl std::error::Error for FrontendError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeEnv {
    scopes: Vec<HashMap<String, Type>>,
}

impl ScopeEnv {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn with_params(params: &[(String, Type)]) -> Self {
        let mut env = Self::new();
        for (name, ty) in params {
            env.insert(name.clone(), *ty);
        }
        env
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            let _ = self.scopes.pop();
        }
    }

    pub fn insert(&mut self, name: String, ty: Type) {
        if let Some(last) = self.scopes.last_mut() {
            last.insert(name, ty);
        }
    }

    pub fn get(&self, name: &str) -> Option<Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(t) = scope.get(name) {
                return Some(*t);
            }
        }
        None
    }
}

impl Default for ScopeEnv {
    fn default() -> Self {
        Self::new()
    }
}

pub fn build_fn_table(functions: &[Function]) -> Result<FnTable, FrontendError> {
    let mut out = HashMap::new();
    for f in functions {
        if out.contains_key(&f.name) {
            return Err(FrontendError {
                pos: 0,
                message: format!("duplicate function '{}'", f.name),
            });
        }
        out.insert(
            f.name.clone(),
            FnSig {
                params: f.params.iter().map(|(_, t)| *t).collect(),
                ret: f.ret,
            },
        );
    }
    Ok(out)
}

pub fn lex(input: &str) -> Result<Vec<Token>, FrontendError> {
    let bytes = input.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        let start = i;
        let push = |kind: TokenKind, text: &str, pos: usize, out: &mut Vec<Token>| {
            out.push(Token {
                kind,
                text: text.to_string(),
                pos,
            });
        };

        match c {
            b'{' => {
                push(TokenKind::LBrace, "{", i, &mut out);
                i += 1;
            }
            b'}' => {
                push(TokenKind::RBrace, "}", i, &mut out);
                i += 1;
            }
            b'(' => {
                push(TokenKind::LParen, "(", i, &mut out);
                i += 1;
            }
            b')' => {
                push(TokenKind::RParen, ")", i, &mut out);
                i += 1;
            }
            b';' => {
                push(TokenKind::Semi, ";", i, &mut out);
                i += 1;
            }
            b',' => {
                push(TokenKind::Comma, ",", i, &mut out);
                i += 1;
            }
            b':' => {
                push(TokenKind::Colon, ":", i, &mut out);
                i += 1;
            }
            b'_' => {
                push(TokenKind::Underscore, "_", i, &mut out);
                i += 1;
            }
            b'!' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    push(TokenKind::Ne, "!=", i, &mut out);
                    i += 2;
                } else {
                    push(TokenKind::Bang, "!", i, &mut out);
                    i += 1;
                }
            }
            b'=' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    push(TokenKind::EqEq, "==", i, &mut out);
                    i += 2;
                } else if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    push(TokenKind::FatArrow, "=>", i, &mut out);
                    i += 2;
                } else {
                    push(TokenKind::Assign, "=", i, &mut out);
                    i += 1;
                }
            }
            b'&' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'&' {
                    push(TokenKind::AndAnd, "&&", i, &mut out);
                    i += 2;
                } else {
                    return Err(FrontendError {
                        pos: i,
                        message: "expected '&&'".to_string(),
                    });
                }
            }
            b'|' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                    push(TokenKind::OrOr, "||", i, &mut out);
                    i += 2;
                } else {
                    return Err(FrontendError {
                        pos: i,
                        message: "expected '||'".to_string(),
                    });
                }
            }
            b'-' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    push(TokenKind::Arrow, "->", i, &mut out);
                    i += 2;
                } else {
                    return Err(FrontendError {
                        pos: i,
                        message: "expected '->'".to_string(),
                    });
                }
            }
            d if d.is_ascii_digit() => {
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                push(TokenKind::Num, &input[start..i], start, &mut out);
            }
            a if a.is_ascii_alphabetic() => {
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let text = &input[start..i];
                let kind = match text {
                    "fn" => TokenKind::KwFn,
                    "let" => TokenKind::KwLet,
                    "if" => TokenKind::KwIf,
                    "else" => TokenKind::KwElse,
                    "return" => TokenKind::KwReturn,
                    "match" => TokenKind::KwMatch,
                    "true" => TokenKind::KwTrue,
                    "false" => TokenKind::KwFalse,
                    "quad" => TokenKind::TyQuad,
                    "bool" => TokenKind::TyBool,
                    "i32" => TokenKind::TyI32,
                    "u32" => TokenKind::TyU32,
                    "fx" => TokenKind::TyFx,
                    "N" => TokenKind::QuadN,
                    "F" => TokenKind::QuadF,
                    "T" => TokenKind::QuadT,
                    "S" => TokenKind::QuadS,
                    _ => TokenKind::Ident,
                };
                push(kind, text, start, &mut out);
            }
            _ => {
                return Err(FrontendError {
                    pos: i,
                    message: format!("unexpected character '{}'", c as char),
                })
            }
        }
    }

    Ok(out)
}

pub fn parse_function(input: &str) -> Result<Function, FrontendError> {
    let tokens = lex(input)?;
    let mut p = Parser { tokens, idx: 0 };
    let f = p.parse_function()?;
    if p.idx != p.tokens.len() {
        return Err(FrontendError {
            pos: p.pos(),
            message: "unexpected trailing tokens after function".to_string(),
        });
    }
    Ok(f)
}

pub fn parse_program(input: &str) -> Result<Program, FrontendError> {
    let tokens = lex(input)?;
    let mut p = Parser { tokens, idx: 0 };
    p.parse_program()
}

pub fn type_check_function(func: &Function) -> Result<(), FrontendError> {
    let mut table = HashMap::new();
    table.insert(
        func.name.clone(),
        FnSig {
            params: func.params.iter().map(|(_, t)| *t).collect(),
            ret: func.ret,
        },
    );
    type_check_function_with_table(func, &table)
}

pub fn type_check_program(p: &Program) -> Result<(), FrontendError> {
    let table = build_fn_table(&p.functions)?;
    let main_sig = table.get("main").ok_or(FrontendError {
        pos: 0,
        message: "program must define fn main()".to_string(),
    })?;
    if !main_sig.params.is_empty() || main_sig.ret != Type::Unit {
        return Err(FrontendError {
            pos: 0,
            message: "main must have signature fn main()".to_string(),
        });
    }
    for f in &p.functions {
        type_check_function_with_table(f, &table)?;
    }
    Ok(())
}

pub fn type_check_function_with_table(
    func: &Function,
    table: &FnTable,
) -> Result<(), FrontendError> {
    let mut env = ScopeEnv::with_params(&func.params);
    for stmt in &func.body {
        check_stmt(stmt, &mut env, func.ret, table)?;
    }
    Ok(())
}

fn check_stmt(
    stmt: &Stmt,
    env: &mut ScopeEnv,
    ret_ty: Type,
    table: &FnTable,
) -> Result<(), FrontendError> {
    match stmt {
        Stmt::Let { name, ty, value } => {
            let vt = infer_expr_type(value, env, table)?;
            let final_ty = if let Some(ann) = ty {
                if *ann != vt {
                    return Err(FrontendError {
                        pos: 0,
                        message: format!("type mismatch in let '{}': {:?} vs {:?}", name, ann, vt),
                    });
                }
                *ann
            } else {
                vt
            };
            env.insert(name.clone(), final_ty);
            Ok(())
        }
        Stmt::If {
            condition,
            then_block,
            else_block,
        } => {
            let ct = infer_expr_type(condition, env, table)?;
            if ct != Type::Bool {
                return Err(FrontendError {
                    pos: 0,
                    message: "if condition must be bool; explicit compare is required for quad"
                        .to_string(),
                });
            }

            let mut then_env = env.clone();
            then_env.push_scope();
            for s in then_block {
                check_stmt(s, &mut then_env, ret_ty, table)?;
            }
            then_env.pop_scope();

            let mut else_env = env.clone();
            else_env.push_scope();
            for s in else_block {
                check_stmt(s, &mut else_env, ret_ty, table)?;
            }
            else_env.pop_scope();
            Ok(())
        }
        Stmt::Match {
            scrutinee,
            arms,
            default,
        } => {
            let st = infer_expr_type(scrutinee, env, table)?;
            if st != Type::Quad {
                return Err(FrontendError {
                    pos: 0,
                    message: "match is allowed only for quad scrutinee".to_string(),
                });
            }
            if default.is_empty() {
                return Err(FrontendError {
                    pos: 0,
                    message: "match requires default arm '_'".to_string(),
                });
            }

            for arm in arms {
                let mut arm_env = env.clone();
                arm_env.push_scope();
                for s in &arm.block {
                    check_stmt(s, &mut arm_env, ret_ty, table)?;
                }
                arm_env.pop_scope();
            }

            let mut def_env = env.clone();
            def_env.push_scope();
            for s in default {
                check_stmt(s, &mut def_env, ret_ty, table)?;
            }
            def_env.pop_scope();
            Ok(())
        }
        Stmt::Return(v) => {
            let got = if let Some(e) = v {
                infer_expr_type(e, env, table)?
            } else {
                Type::Unit
            };
            if got != ret_ty {
                return Err(FrontendError {
                    pos: 0,
                    message: format!("return type mismatch: expected {:?}, got {:?}", ret_ty, got),
                });
            }
            Ok(())
        }
        Stmt::Expr(e) => {
            let _ = infer_expr_type(e, env, table)?;
            Ok(())
        }
    }
}

fn infer_expr_type(expr: &Expr, env: &ScopeEnv, table: &FnTable) -> Result<Type, FrontendError> {
    match expr {
        Expr::QuadLiteral(_) => Ok(Type::Quad),
        Expr::BoolLiteral(_) => Ok(Type::Bool),
        Expr::Num(_) => Ok(Type::I32),
        Expr::Var(v) => env.get(v).ok_or(FrontendError {
            pos: 0,
            message: format!("unknown variable '{}'", v),
        }),
        Expr::Call(name, args) => {
            let sig = table.get(name).ok_or(FrontendError {
                pos: 0,
                message: format!("unknown function '{}'", name),
            })?;
            if sig.params.len() != args.len() {
                return Err(FrontendError {
                    pos: 0,
                    message: format!(
                        "function '{}' expects {} args, got {}",
                        name,
                        sig.params.len(),
                        args.len()
                    ),
                });
            }
            for (i, arg) in args.iter().enumerate() {
                let at = infer_expr_type(arg, env, table)?;
                if at != sig.params[i] {
                    return Err(FrontendError {
                        pos: 0,
                        message: format!(
                            "arg {} for '{}' has type {:?}, expected {:?}",
                            i, name, at, sig.params[i]
                        ),
                    });
                }
            }
            Ok(sig.ret)
        }
        Expr::Unary(UnaryOp::Not, inner) => {
            let t = infer_expr_type(inner, env, table)?;
            match t {
                Type::Quad | Type::Bool => Ok(t),
                _ => Err(FrontendError {
                    pos: 0,
                    message: format!("operator ! unsupported for {:?}", t),
                }),
            }
        }
        Expr::Binary(l, op, r) => {
            let lt = infer_expr_type(l, env, table)?;
            let rt = infer_expr_type(r, env, table)?;
            match op {
                BinaryOp::Eq | BinaryOp::Ne => {
                    if lt == rt {
                        Ok(Type::Bool)
                    } else {
                        Err(FrontendError {
                            pos: 0,
                            message: format!("cannot compare {:?} and {:?}", lt, rt),
                        })
                    }
                }
                BinaryOp::AndAnd | BinaryOp::OrOr => {
                    if lt != rt {
                        return Err(FrontendError {
                            pos: 0,
                            message: format!("operator type mismatch: {:?} vs {:?}", lt, rt),
                        });
                    }
                    match lt {
                        Type::Quad | Type::Bool => Ok(lt),
                        _ => Err(FrontendError {
                            pos: 0,
                            message: format!("operator unsupported for {:?}", lt),
                        }),
                    }
                }
                BinaryOp::Implies => {
                    if lt == Type::Quad && rt == Type::Quad {
                        Ok(Type::Quad)
                    } else {
                        Err(FrontendError {
                            pos: 0,
                            message: "operator '->' is allowed only for quad".to_string(),
                        })
                    }
                }
            }
        }
    }
}

struct Parser {
    tokens: Vec<Token>,
    idx: usize,
}

impl Parser {
    fn parse_program(&mut self) -> Result<Program, FrontendError> {
        let mut functions = Vec::new();
        while self.idx < self.tokens.len() {
            functions.push(self.parse_function()?);
        }
        Ok(Program { functions })
    }

    fn parse_function(&mut self) -> Result<Function, FrontendError> {
        self.expect(TokenKind::KwFn, "expected 'fn'")?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LParen, "expected '('")?;
        let mut params = Vec::new();
        if !self.check(TokenKind::RParen) {
            loop {
                let pname = self.expect_ident()?;
                self.expect(TokenKind::Colon, "expected ':'")?;
                let pty = self.parse_type()?;
                params.push((pname, pty));
                if self.eat(TokenKind::Comma) {
                    continue;
                }
                break;
            }
        }
        self.expect(TokenKind::RParen, "expected ')'")?;
        let ret = if self.eat(TokenKind::Arrow) {
            self.parse_type()?
        } else {
            Type::Unit
        };
        let body = self.parse_block()?;
        Ok(Function {
            name,
            params,
            ret,
            body,
        })
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, FrontendError> {
        self.expect(TokenKind::LBrace, "expected '{'")?;
        let mut out = Vec::new();
        while !self.check(TokenKind::RBrace) {
            out.push(self.parse_stmt()?);
        }
        self.expect(TokenKind::RBrace, "expected '}'")?;
        Ok(out)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, FrontendError> {
        if self.eat(TokenKind::KwLet) {
            let name = self.expect_ident()?;
            let ty = if self.eat(TokenKind::Colon) {
                Some(self.parse_type()?)
            } else {
                None
            };
            self.expect(TokenKind::Assign, "expected '='")?;
            let value = self.parse_expr()?;
            self.expect(TokenKind::Semi, "expected ';'")?;
            return Ok(Stmt::Let { name, ty, value });
        }
        if self.eat(TokenKind::KwIf) {
            let condition = self.parse_expr()?;
            let then_block = self.parse_block()?;
            let else_block = if self.eat(TokenKind::KwElse) {
                if self.eat(TokenKind::KwIf) {
                    let nested = self.parse_if_after_kw_if()?;
                    vec![nested]
                } else {
                    self.parse_block()?
                }
            } else {
                Vec::new()
            };
            return Ok(Stmt::If {
                condition,
                then_block,
                else_block,
            });
        }
        if self.eat(TokenKind::KwMatch) {
            let scrutinee = self.parse_expr()?;
            self.expect(TokenKind::LBrace, "expected '{' after match expr")?;
            let mut arms = Vec::new();
            let mut default: Option<Vec<Stmt>> = None;
            while !self.check(TokenKind::RBrace) {
                if self.eat(TokenKind::Underscore) {
                    self.expect(TokenKind::FatArrow, "expected '=>' after '_'")?;
                    let block = self.parse_block()?;
                    default = Some(block);
                    continue;
                }
                let pat = if self.eat(TokenKind::QuadN) {
                    QuadVal::N
                } else if self.eat(TokenKind::QuadF) {
                    QuadVal::F
                } else if self.eat(TokenKind::QuadT) {
                    QuadVal::T
                } else if self.eat(TokenKind::QuadS) {
                    QuadVal::S
                } else {
                    return Err(FrontendError {
                        pos: self.pos(),
                        message: "expected match pattern N|F|T|S|_".to_string(),
                    });
                };
                self.expect(TokenKind::FatArrow, "expected '=>' after match pattern")?;
                let block = self.parse_block()?;
                arms.push(MatchArm { pat, block });
            }
            self.expect(TokenKind::RBrace, "expected '}' after match arms")?;
            return Ok(Stmt::Match {
                scrutinee,
                arms,
                default: default.unwrap_or_default(),
            });
        }
        if self.eat(TokenKind::KwReturn) {
            if self.eat(TokenKind::Semi) {
                return Ok(Stmt::Return(None));
            }
            let expr = self.parse_expr()?;
            self.expect(TokenKind::Semi, "expected ';'")?;
            return Ok(Stmt::Return(Some(expr)));
        }
        let expr = self.parse_expr()?;
        self.expect(TokenKind::Semi, "expected ';'")?;
        Ok(Stmt::Expr(expr))
    }

    fn parse_if_after_kw_if(&mut self) -> Result<Stmt, FrontendError> {
        let condition = self.parse_expr()?;
        let then_block = self.parse_block()?;
        let else_block = if self.eat(TokenKind::KwElse) {
            if self.eat(TokenKind::KwIf) {
                let nested = self.parse_if_after_kw_if()?;
                vec![nested]
            } else {
                self.parse_block()?
            }
        } else {
            Vec::new()
        };
        Ok(Stmt::If {
            condition,
            then_block,
            else_block,
        })
    }

    fn parse_expr(&mut self) -> Result<Expr, FrontendError> {
        self.parse_impl()
    }

    fn parse_impl(&mut self) -> Result<Expr, FrontendError> {
        let mut left = self.parse_or()?;
        while self.eat(TokenKind::Arrow) {
            let right = self.parse_or()?;
            left = Expr::Binary(Box::new(left), BinaryOp::Implies, Box::new(right));
        }
        Ok(left)
    }

    fn parse_or(&mut self) -> Result<Expr, FrontendError> {
        let mut left = self.parse_and()?;
        while self.eat(TokenKind::OrOr) {
            let right = self.parse_and()?;
            left = Expr::Binary(Box::new(left), BinaryOp::OrOr, Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, FrontendError> {
        let mut left = self.parse_eq()?;
        while self.eat(TokenKind::AndAnd) {
            let right = self.parse_eq()?;
            left = Expr::Binary(Box::new(left), BinaryOp::AndAnd, Box::new(right));
        }
        Ok(left)
    }

    fn parse_eq(&mut self) -> Result<Expr, FrontendError> {
        let mut left = self.parse_unary()?;
        loop {
            if self.eat(TokenKind::EqEq) {
                let right = self.parse_unary()?;
                left = Expr::Binary(Box::new(left), BinaryOp::Eq, Box::new(right));
                continue;
            }
            if self.eat(TokenKind::Ne) {
                let right = self.parse_unary()?;
                left = Expr::Binary(Box::new(left), BinaryOp::Ne, Box::new(right));
                continue;
            }
            break;
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, FrontendError> {
        if self.eat(TokenKind::Bang) {
            let inner = self.parse_unary()?;
            return Ok(Expr::Unary(UnaryOp::Not, Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, FrontendError> {
        if self.eat(TokenKind::LParen) {
            let e = self.parse_expr()?;
            self.expect(TokenKind::RParen, "expected ')'")?;
            return Ok(e);
        }
        if self.eat(TokenKind::QuadN) {
            return Ok(Expr::QuadLiteral(QuadVal::N));
        }
        if self.eat(TokenKind::QuadF) {
            return Ok(Expr::QuadLiteral(QuadVal::F));
        }
        if self.eat(TokenKind::QuadT) {
            return Ok(Expr::QuadLiteral(QuadVal::T));
        }
        if self.eat(TokenKind::QuadS) {
            return Ok(Expr::QuadLiteral(QuadVal::S));
        }
        if self.eat(TokenKind::KwTrue) {
            return Ok(Expr::BoolLiteral(true));
        }
        if self.eat(TokenKind::KwFalse) {
            return Ok(Expr::BoolLiteral(false));
        }
        if self.check(TokenKind::Num) {
            let text = self.advance().text;
            let n = text.parse::<i64>().map_err(|_| FrontendError {
                pos: 0,
                message: "invalid number".to_string(),
            })?;
            return Ok(Expr::Num(n));
        }
        if self.check(TokenKind::Ident) {
            let name = self.advance().text;
            if self.eat(TokenKind::LParen) {
                let mut args = Vec::new();
                if !self.check(TokenKind::RParen) {
                    loop {
                        args.push(self.parse_expr()?);
                        if self.eat(TokenKind::Comma) {
                            continue;
                        }
                        break;
                    }
                }
                self.expect(TokenKind::RParen, "expected ')'")?;
                return Ok(Expr::Call(name, args));
            }
            return Ok(Expr::Var(name));
        }
        Err(FrontendError {
            pos: self.pos(),
            message: "expected primary expression".to_string(),
        })
    }

    fn parse_type(&mut self) -> Result<Type, FrontendError> {
        if self.eat(TokenKind::TyQuad) {
            return Ok(Type::Quad);
        }
        if self.eat(TokenKind::TyBool) {
            return Ok(Type::Bool);
        }
        if self.eat(TokenKind::TyI32) {
            return Ok(Type::I32);
        }
        if self.eat(TokenKind::TyU32) {
            return Ok(Type::U32);
        }
        if self.eat(TokenKind::TyFx) {
            return Ok(Type::Fx);
        }
        Err(FrontendError {
            pos: self.pos(),
            message: "expected type".to_string(),
        })
    }

    fn check(&self, kind: TokenKind) -> bool {
        self.tokens
            .get(self.idx)
            .map(|t| t.kind == kind)
            .unwrap_or(false)
    }

    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.idx += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: TokenKind, msg: &str) -> Result<(), FrontendError> {
        if self.eat(kind) {
            Ok(())
        } else {
            Err(FrontendError {
                pos: self.pos(),
                message: msg.to_string(),
            })
        }
    }

    fn expect_ident(&mut self) -> Result<String, FrontendError> {
        if self.check(TokenKind::Ident) {
            Ok(self.advance().text)
        } else {
            Err(FrontendError {
                pos: self.pos(),
                message: "expected identifier".to_string(),
            })
        }
    }

    fn advance(&mut self) -> Token {
        let t = self.tokens[self.idx].clone();
        self.idx += 1;
        t
    }

    fn pos(&self) -> usize {
        self.tokens.get(self.idx).map(|t| t.pos).unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrInstr {
    Label {
        name: String,
    },
    LoadQ {
        dst: u16,
        val: QuadVal,
    },
    LoadBool {
        dst: u16,
        val: bool,
    },
    LoadI32 {
        dst: u16,
        val: i32,
    },
    LoadVar {
        dst: u16,
        name: String,
    },
    StoreVar {
        name: String,
        src: u16,
    },
    QAnd {
        dst: u16,
        lhs: u16,
        rhs: u16,
    },
    QOr {
        dst: u16,
        lhs: u16,
        rhs: u16,
    },
    QNot {
        dst: u16,
        src: u16,
    },
    QImpl {
        dst: u16,
        lhs: u16,
        rhs: u16,
    },
    BoolAnd {
        dst: u16,
        lhs: u16,
        rhs: u16,
    },
    BoolOr {
        dst: u16,
        lhs: u16,
        rhs: u16,
    },
    BoolNot {
        dst: u16,
        src: u16,
    },
    CmpEq {
        dst: u16,
        lhs: u16,
        rhs: u16,
    },
    CmpNe {
        dst: u16,
        lhs: u16,
        rhs: u16,
    },
    Jmp {
        label: String,
    },
    JmpIf {
        cond: u16,
        label: String,
    },
    Call {
        dst: Option<u16>,
        name: String,
        args: Vec<u16>,
    },
    Ret {
        src: Option<u16>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrFunction {
    pub name: String,
    pub instrs: Vec<IrInstr>,
}

pub fn lower_expr_to_ir(
    expr: &Expr,
    var_types: &HashMap<String, Type>,
    fn_table: &FnTable,
) -> Result<Vec<IrInstr>, FrontendError> {
    let mut out = Vec::new();
    let mut next = 0u16;
    let mut env = ScopeEnv::new();
    for (name, ty) in var_types {
        env.insert(name.clone(), *ty);
    }
    let _ = lower_expr(expr, &mut next, &mut out, &env, fn_table)?;
    Ok(out)
}

pub fn lower_function_to_ir(
    func: &Function,
    fn_table: &FnTable,
) -> Result<IrFunction, FrontendError> {
    type_check_function_with_table(func, fn_table)?;

    let mut ctx = LoweringCtx::new();
    let mut env = ScopeEnv::with_params(&func.params);
    for stmt in &func.body {
        lower_stmt(stmt, &mut ctx, &mut env, func.ret, fn_table)?;
    }

    if !ctx.ends_with_ret() {
        if func.ret == Type::Unit {
            ctx.instrs.push(IrInstr::Ret { src: None });
        } else {
            return Err(FrontendError {
                pos: 0,
                message: format!(
                    "function '{}' may exit without returning {:?}",
                    func.name, func.ret
                ),
            });
        }
    }

    Ok(IrFunction {
        name: func.name.clone(),
        instrs: ctx.instrs,
    })
}

pub fn compile_program_to_ir(input: &str) -> Result<Vec<IrFunction>, FrontendError> {
    let program = parse_program(input)?;
    let fn_table = build_fn_table(&program.functions)?;
    type_check_program(&program)?;
    let mut out = Vec::new();
    for f in &program.functions {
        out.push(lower_function_to_ir(f, &fn_table)?);
    }
    Ok(out)
}

pub fn validate_ir(f: &IrFunction) -> Result<(), FrontendError> {
    let mut labels: HashMap<String, usize> = HashMap::new();
    let mut has_ret = false;

    for (idx, instr) in f.instrs.iter().enumerate() {
        if let IrInstr::Label { name } = instr {
            if labels.insert(name.clone(), idx).is_some() {
                return Err(FrontendError {
                    pos: idx,
                    message: format!("duplicate label '{}' in '{}'", name, f.name),
                });
            }
        }
        if matches!(instr, IrInstr::Ret { .. }) {
            has_ret = true;
        }
    }

    if !has_ret {
        return Err(FrontendError {
            pos: 0,
            message: format!("function '{}' has no RET", f.name),
        });
    }

    for (idx, instr) in f.instrs.iter().enumerate() {
        match instr {
            IrInstr::Jmp { label } | IrInstr::JmpIf { label, .. } => {
                if !labels.contains_key(label) {
                    return Err(FrontendError {
                        pos: idx,
                        message: format!("jump to unknown label '{}' in '{}'", label, f.name),
                    });
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub fn compile_program_to_exobyte(input: &str) -> Result<Vec<u8>, FrontendError> {
    let ir = compile_program_to_ir(input)?;
    for f in &ir {
        validate_ir(f)?;
    }
    emit_exobyte(&ir)
}

fn emit_exobyte(funcs: &[IrFunction]) -> Result<Vec<u8>, FrontendError> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    for f in funcs {
        let name_bytes = f.name.as_bytes();
        write_u16_le(
            &mut out,
            u16::try_from(name_bytes.len()).map_err(|_| FrontendError {
                pos: 0,
                message: "function name too long".to_string(),
            })?,
        );
        out.extend_from_slice(name_bytes);
        let code = emit_exobyte_function(f)?;
        write_u32_le(
            &mut out,
            u32::try_from(code.len()).map_err(|_| FrontendError {
                pos: 0,
                message: "function code too large".to_string(),
            })?,
        );
        out.extend_from_slice(&code);
    }
    Ok(out)
}

fn emit_exobyte_function(f: &IrFunction) -> Result<Vec<u8>, FrontendError> {
    let mut interner = StringInterner::new();
    for instr in &f.instrs {
        match instr {
            IrInstr::LoadVar { name, .. } => {
                let _ = interner.id(name)?;
            }
            IrInstr::StoreVar { name, .. } => {
                let _ = interner.id(name)?;
            }
            IrInstr::Call { name, .. } => {
                let _ = interner.id(name)?;
            }
            _ => {}
        }
    }

    let mut label_pc: HashMap<String, u32> = HashMap::new();
    let mut pc: u32 = 0;
    for instr in &f.instrs {
        match instr {
            IrInstr::Label { name } => {
                label_pc.insert(name.clone(), pc);
            }
            _ => {
                pc = pc
                    .checked_add(encoded_size(instr).ok_or(FrontendError {
                        pos: 0,
                        message: "label has no encoded size".to_string(),
                    })? as u32)
                    .ok_or(FrontendError {
                        pos: 0,
                        message: "bytecode size overflow".to_string(),
                    })?;
            }
        }
    }

    let mut code = Vec::new();
    interner.emit_table(&mut code)?;
    for instr in &f.instrs {
        emit_instr(instr, &label_pc, &interner, &mut code)?;
    }
    Ok(code)
}

fn encoded_size(instr: &IrInstr) -> Option<usize> {
    let s = match instr {
        IrInstr::Label { .. } => return None,
        IrInstr::LoadQ { .. } => 1 + 2 + 1,
        IrInstr::LoadBool { .. } => 1 + 2 + 1,
        IrInstr::LoadI32 { .. } => 1 + 2 + 4,
        IrInstr::LoadVar { .. } => 1 + 2 + 2,
        IrInstr::StoreVar { .. } => 1 + 2 + 2,
        IrInstr::QAnd { .. }
        | IrInstr::QOr { .. }
        | IrInstr::QImpl { .. }
        | IrInstr::BoolAnd { .. }
        | IrInstr::BoolOr { .. }
        | IrInstr::CmpEq { .. }
        | IrInstr::CmpNe { .. } => 1 + 2 + 2 + 2,
        IrInstr::QNot { .. } | IrInstr::BoolNot { .. } => 1 + 2 + 2,
        IrInstr::Jmp { .. } => 1 + 4,
        IrInstr::JmpIf { .. } => 1 + 2 + 4,
        IrInstr::Call { args, .. } => 1 + 1 + 2 + 2 + 2 + (args.len() * 2),
        IrInstr::Ret { src: Some(_) } => 1 + 1 + 2,
        IrInstr::Ret { src: None } => 1 + 1,
    };
    Some(s)
}

fn emit_instr(
    instr: &IrInstr,
    label_pc: &HashMap<String, u32>,
    interner: &StringInterner,
    out: &mut Vec<u8>,
) -> Result<(), FrontendError> {
    match instr {
        IrInstr::Label { .. } => {}
        IrInstr::LoadQ { dst, val } => {
            out.push(Opcode::LoadQ.byte());
            write_u16_le(out, *dst);
            out.push(match val {
                QuadVal::N => 0,
                QuadVal::F => 1,
                QuadVal::T => 2,
                QuadVal::S => 3,
            });
        }
        IrInstr::LoadBool { dst, val } => {
            out.push(Opcode::LoadBool.byte());
            write_u16_le(out, *dst);
            out.push(if *val { 1 } else { 0 });
        }
        IrInstr::LoadI32 { dst, val } => {
            out.push(Opcode::LoadI32.byte());
            write_u16_le(out, *dst);
            write_i32_le(out, *val);
        }
        IrInstr::LoadVar { dst, name } => {
            out.push(Opcode::LoadVar.byte());
            write_u16_le(out, *dst);
            write_u16_le(out, interner.lookup(name)?);
        }
        IrInstr::StoreVar { name, src } => {
            out.push(Opcode::StoreVar.byte());
            write_u16_le(out, interner.lookup(name)?);
            write_u16_le(out, *src);
        }
        IrInstr::QAnd { dst, lhs, rhs } => emit_3reg(Opcode::QAnd, *dst, *lhs, *rhs, out),
        IrInstr::QOr { dst, lhs, rhs } => emit_3reg(Opcode::QOr, *dst, *lhs, *rhs, out),
        IrInstr::QNot { dst, src } => emit_2reg(Opcode::QNot, *dst, *src, out),
        IrInstr::QImpl { dst, lhs, rhs } => emit_3reg(Opcode::QImpl, *dst, *lhs, *rhs, out),
        IrInstr::BoolAnd { dst, lhs, rhs } => emit_3reg(Opcode::BoolAnd, *dst, *lhs, *rhs, out),
        IrInstr::BoolOr { dst, lhs, rhs } => emit_3reg(Opcode::BoolOr, *dst, *lhs, *rhs, out),
        IrInstr::BoolNot { dst, src } => emit_2reg(Opcode::BoolNot, *dst, *src, out),
        IrInstr::CmpEq { dst, lhs, rhs } => emit_3reg(Opcode::CmpEq, *dst, *lhs, *rhs, out),
        IrInstr::CmpNe { dst, lhs, rhs } => emit_3reg(Opcode::CmpNe, *dst, *lhs, *rhs, out),
        IrInstr::Jmp { label } => {
            out.push(Opcode::Jmp.byte());
            let addr = *label_pc.get(label).ok_or(FrontendError {
                pos: 0,
                message: format!("unknown label '{}'", label),
            })?;
            write_u32_le(out, addr);
        }
        IrInstr::JmpIf { cond, label } => {
            out.push(Opcode::JmpIf.byte());
            write_u16_le(out, *cond);
            let addr = *label_pc.get(label).ok_or(FrontendError {
                pos: 0,
                message: format!("unknown label '{}'", label),
            })?;
            write_u32_le(out, addr);
        }
        IrInstr::Call { dst, name, args } => {
            out.push(Opcode::Call.byte());
            match dst {
                Some(r) => {
                    out.push(1);
                    write_u16_le(out, *r);
                }
                None => {
                    out.push(0);
                    write_u16_le(out, 0);
                }
            }
            write_u16_le(out, interner.lookup(name)?);
            write_u16_le(
                out,
                u16::try_from(args.len()).map_err(|_| FrontendError {
                    pos: 0,
                    message: "too many call args".to_string(),
                })?,
            );
            for a in args {
                write_u16_le(out, *a);
            }
        }
        IrInstr::Ret { src } => {
            out.push(Opcode::Ret.byte());
            match src {
                Some(r) => {
                    out.push(1);
                    write_u16_le(out, *r);
                }
                None => {
                    out.push(0);
                }
            }
        }
    }
    Ok(())
}

fn emit_3reg(op: Opcode, dst: u16, lhs: u16, rhs: u16, out: &mut Vec<u8>) {
    out.push(op.byte());
    write_u16_le(out, dst);
    write_u16_le(out, lhs);
    write_u16_le(out, rhs);
}

fn emit_2reg(op: Opcode, dst: u16, src: u16, out: &mut Vec<u8>) {
    out.push(op.byte());
    write_u16_le(out, dst);
    write_u16_le(out, src);
}

#[derive(Debug, Default)]
struct StringInterner {
    ids: HashMap<String, u16>,
    by_id: Vec<String>,
}

impl StringInterner {
    fn new() -> Self {
        Self::default()
    }

    fn id(&mut self, s: &str) -> Result<u16, FrontendError> {
        if let Some(id) = self.ids.get(s) {
            return Ok(*id);
        }
        let id = u16::try_from(self.by_id.len()).map_err(|_| FrontendError {
            pos: 0,
            message: "string table overflow".to_string(),
        })?;
        self.ids.insert(s.to_string(), id);
        self.by_id.push(s.to_string());
        Ok(id)
    }

    fn lookup(&self, s: &str) -> Result<u16, FrontendError> {
        self.ids.get(s).copied().ok_or(FrontendError {
            pos: 0,
            message: format!("string '{}' not interned", s),
        })
    }

    fn emit_table(&self, out: &mut Vec<u8>) -> Result<(), FrontendError> {
        write_u16_le(
            out,
            u16::try_from(self.by_id.len()).map_err(|_| FrontendError {
                pos: 0,
                message: "string table too large".to_string(),
            })?,
        );
        for s in &self.by_id {
            let b = s.as_bytes();
            write_u16_le(
                out,
                u16::try_from(b.len()).map_err(|_| FrontendError {
                    pos: 0,
                    message: "string too long".to_string(),
                })?,
            );
            out.extend_from_slice(b);
        }
        Ok(())
    }
}

fn lower_expr(
    expr: &Expr,
    next: &mut u16,
    out: &mut Vec<IrInstr>,
    env: &ScopeEnv,
    fn_table: &FnTable,
) -> Result<(u16, Type), FrontendError> {
    match expr {
        Expr::QuadLiteral(v) => {
            let r = alloc(next);
            out.push(IrInstr::LoadQ { dst: r, val: *v });
            Ok((r, Type::Quad))
        }
        Expr::BoolLiteral(v) => {
            let r = alloc(next);
            out.push(IrInstr::LoadBool { dst: r, val: *v });
            Ok((r, Type::Bool))
        }
        Expr::Num(n) => {
            let r = alloc(next);
            let val = i32::try_from(*n).map_err(|_| FrontendError {
                pos: 0,
                message: format!("numeric literal {} does not fit in i32", n),
            })?;
            out.push(IrInstr::LoadI32 { dst: r, val });
            Ok((r, Type::I32))
        }
        Expr::Var(name) => {
            let ty = env.get(name).ok_or(FrontendError {
                pos: 0,
                message: format!("unknown variable '{}'", name),
            })?;
            let r = alloc(next);
            out.push(IrInstr::LoadVar {
                dst: r,
                name: name.clone(),
            });
            Ok((r, ty))
        }
        Expr::Call(name, args) => {
            let sig = fn_table.get(name).ok_or(FrontendError {
                pos: 0,
                message: format!("unknown function '{}'", name),
            })?;
            if sig.params.len() != args.len() {
                return Err(FrontendError {
                    pos: 0,
                    message: format!(
                        "function '{}' expects {} args, got {}",
                        name,
                        sig.params.len(),
                        args.len()
                    ),
                });
            }
            let mut regs = Vec::new();
            for (i, arg) in args.iter().enumerate() {
                let (r, t) = lower_expr(arg, next, out, env, fn_table)?;
                if t != sig.params[i] {
                    return Err(FrontendError {
                        pos: 0,
                        message: format!(
                            "arg {} for '{}' has type {:?}, expected {:?}",
                            i, name, t, sig.params[i]
                        ),
                    });
                }
                regs.push(r);
            }
            if sig.ret == Type::Unit {
                return Err(FrontendError {
                    pos: 0,
                    message: format!(
                        "unit-returning call '{}' cannot be used as expression value",
                        name
                    ),
                });
            }
            let r = alloc(next);
            out.push(IrInstr::Call {
                dst: Some(r),
                name: name.clone(),
                args: regs,
            });
            Ok((r, sig.ret))
        }
        Expr::Unary(UnaryOp::Not, inner) => {
            let (src, ty) = lower_expr(inner, next, out, env, fn_table)?;
            let dst = alloc(next);
            match ty {
                Type::Quad => out.push(IrInstr::QNot { dst, src }),
                Type::Bool => out.push(IrInstr::BoolNot { dst, src }),
                _ => {
                    return Err(FrontendError {
                        pos: 0,
                        message: format!("operator ! unsupported for {:?}", ty),
                    })
                }
            }
            Ok((dst, ty))
        }
        Expr::Binary(left, op, right) => {
            let (lr, lt) = lower_expr(left, next, out, env, fn_table)?;
            let (rr, rt) = lower_expr(right, next, out, env, fn_table)?;
            if lt != rt {
                return Err(FrontendError {
                    pos: 0,
                    message: format!("operator type mismatch: {:?} vs {:?}", lt, rt),
                });
            }
            let dst = alloc(next);
            match op {
                BinaryOp::AndAnd => match lt {
                    Type::Quad => out.push(IrInstr::QAnd {
                        dst,
                        lhs: lr,
                        rhs: rr,
                    }),
                    Type::Bool => out.push(IrInstr::BoolAnd {
                        dst,
                        lhs: lr,
                        rhs: rr,
                    }),
                    _ => {
                        return Err(FrontendError {
                            pos: 0,
                            message: format!("operator && unsupported for {:?}", lt),
                        })
                    }
                },
                BinaryOp::OrOr => match lt {
                    Type::Quad => out.push(IrInstr::QOr {
                        dst,
                        lhs: lr,
                        rhs: rr,
                    }),
                    Type::Bool => out.push(IrInstr::BoolOr {
                        dst,
                        lhs: lr,
                        rhs: rr,
                    }),
                    _ => {
                        return Err(FrontendError {
                            pos: 0,
                            message: format!("operator || unsupported for {:?}", lt),
                        })
                    }
                },
                BinaryOp::Implies => {
                    if lt != Type::Quad {
                        return Err(FrontendError {
                            pos: 0,
                            message: "operator '->' is allowed only for quad".to_string(),
                        });
                    }
                    out.push(IrInstr::QImpl {
                        dst,
                        lhs: lr,
                        rhs: rr,
                    });
                    return Ok((dst, Type::Quad));
                }
                BinaryOp::Eq => {
                    out.push(IrInstr::CmpEq {
                        dst,
                        lhs: lr,
                        rhs: rr,
                    });
                    return Ok((dst, Type::Bool));
                }
                BinaryOp::Ne => {
                    out.push(IrInstr::CmpNe {
                        dst,
                        lhs: lr,
                        rhs: rr,
                    });
                    return Ok((dst, Type::Bool));
                }
            }
            Ok((dst, lt))
        }
    }
}

fn lower_stmt(
    stmt: &Stmt,
    ctx: &mut LoweringCtx,
    env: &mut ScopeEnv,
    ret_ty: Type,
    fn_table: &FnTable,
) -> Result<(), FrontendError> {
    match stmt {
        Stmt::Let { name, ty, value } => {
            let (reg, vty) = lower_expr(value, &mut ctx.next_reg, &mut ctx.instrs, env, fn_table)?;
            let final_ty = if let Some(ann) = ty { *ann } else { vty };
            env.insert(name.clone(), final_ty);
            ctx.instrs.push(IrInstr::StoreVar {
                name: name.clone(),
                src: reg,
            });
            Ok(())
        }
        Stmt::Expr(expr) => {
            lower_expr_stmt(expr, ctx, env, fn_table)?;
            Ok(())
        }
        Stmt::Return(v) => {
            match v {
                Some(e) => {
                    let (reg, ty) =
                        lower_expr(e, &mut ctx.next_reg, &mut ctx.instrs, env, fn_table)?;
                    if ty != ret_ty {
                        return Err(FrontendError {
                            pos: 0,
                            message: format!(
                                "return type mismatch in lowering: expected {:?}, got {:?}",
                                ret_ty, ty
                            ),
                        });
                    }
                    ctx.instrs.push(IrInstr::Ret { src: Some(reg) });
                }
                None => {
                    if ret_ty != Type::Unit {
                        return Err(FrontendError {
                            pos: 0,
                            message: format!(
                                "return without value in non-unit function ({:?})",
                                ret_ty
                            ),
                        });
                    }
                    ctx.instrs.push(IrInstr::Ret { src: None });
                }
            }
            Ok(())
        }
        Stmt::If {
            condition,
            then_block,
            else_block,
        } => {
            let (cond_reg, cond_ty) =
                lower_expr(condition, &mut ctx.next_reg, &mut ctx.instrs, env, fn_table)?;
            if cond_ty != Type::Bool {
                return Err(FrontendError {
                    pos: 0,
                    message: "if condition must be bool".to_string(),
                });
            }

            let id = ctx.next_if_id();
            let then_label = format!("if_{}_then", id);
            let else_label = format!("if_{}_else", id);
            let end_label = format!("if_{}_end", id);

            ctx.instrs.push(IrInstr::JmpIf {
                cond: cond_reg,
                label: then_label.clone(),
            });
            ctx.instrs.push(IrInstr::Jmp {
                label: else_label.clone(),
            });

            ctx.instrs.push(IrInstr::Label { name: then_label });
            let mut then_env = env.clone();
            then_env.push_scope();
            for s in then_block {
                lower_stmt(s, ctx, &mut then_env, ret_ty, fn_table)?;
            }
            then_env.pop_scope();
            ctx.instrs.push(IrInstr::Jmp {
                label: end_label.clone(),
            });

            ctx.instrs.push(IrInstr::Label { name: else_label });
            let mut else_env = env.clone();
            else_env.push_scope();
            for s in else_block {
                lower_stmt(s, ctx, &mut else_env, ret_ty, fn_table)?;
            }
            else_env.pop_scope();
            ctx.instrs.push(IrInstr::Jmp {
                label: end_label.clone(),
            });

            ctx.instrs.push(IrInstr::Label { name: end_label });
            Ok(())
        }
        Stmt::Match {
            scrutinee,
            arms,
            default,
        } => {
            let (scr_reg, scr_ty) =
                lower_expr(scrutinee, &mut ctx.next_reg, &mut ctx.instrs, env, fn_table)?;
            if scr_ty != Type::Quad {
                return Err(FrontendError {
                    pos: 0,
                    message: "match scrutinee must be quad".to_string(),
                });
            }
            if default.is_empty() {
                return Err(FrontendError {
                    pos: 0,
                    message: "match requires default arm '_'".to_string(),
                });
            }

            let mid = ctx.next_if_id();
            let end_label = format!("match_{}_end", mid);
            let default_label = format!("match_{}_default", mid);
            let arm_labels: Vec<String> = (0..arms.len())
                .map(|i| format!("match_{}_arm_{}", mid, i))
                .collect();

            for (i, arm) in arms.iter().enumerate() {
                let lit_reg = alloc(&mut ctx.next_reg);
                ctx.instrs.push(IrInstr::LoadQ {
                    dst: lit_reg,
                    val: arm.pat,
                });
                let cmp_reg = alloc(&mut ctx.next_reg);
                ctx.instrs.push(IrInstr::CmpEq {
                    dst: cmp_reg,
                    lhs: scr_reg,
                    rhs: lit_reg,
                });
                ctx.instrs.push(IrInstr::JmpIf {
                    cond: cmp_reg,
                    label: arm_labels[i].clone(),
                });
            }
            ctx.instrs.push(IrInstr::Jmp {
                label: default_label.clone(),
            });

            for (i, arm) in arms.iter().enumerate() {
                ctx.instrs.push(IrInstr::Label {
                    name: arm_labels[i].clone(),
                });
                let mut arm_env = env.clone();
                arm_env.push_scope();
                for s in &arm.block {
                    lower_stmt(s, ctx, &mut arm_env, ret_ty, fn_table)?;
                }
                arm_env.pop_scope();
                ctx.instrs.push(IrInstr::Jmp {
                    label: end_label.clone(),
                });
            }

            ctx.instrs.push(IrInstr::Label {
                name: default_label,
            });
            let mut def_env = env.clone();
            def_env.push_scope();
            for s in default {
                lower_stmt(s, ctx, &mut def_env, ret_ty, fn_table)?;
            }
            def_env.pop_scope();
            ctx.instrs.push(IrInstr::Jmp {
                label: end_label.clone(),
            });

            ctx.instrs.push(IrInstr::Label { name: end_label });
            Ok(())
        }
    }
}

fn lower_expr_stmt(
    expr: &Expr,
    ctx: &mut LoweringCtx,
    env: &ScopeEnv,
    fn_table: &FnTable,
) -> Result<(), FrontendError> {
    if let Expr::Call(name, args) = expr {
        let sig = fn_table.get(name).ok_or(FrontendError {
            pos: 0,
            message: format!("unknown function '{}'", name),
        })?;
        if sig.params.len() != args.len() {
            return Err(FrontendError {
                pos: 0,
                message: format!(
                    "function '{}' expects {} args, got {}",
                    name,
                    sig.params.len(),
                    args.len()
                ),
            });
        }
        let mut regs = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            let (r, t) = lower_expr(arg, &mut ctx.next_reg, &mut ctx.instrs, env, fn_table)?;
            if t != sig.params[i] {
                return Err(FrontendError {
                    pos: 0,
                    message: format!("arg {} for '{}' type mismatch", i, name),
                });
            }
            regs.push(r);
        }
        let dst = if sig.ret == Type::Unit {
            None
        } else {
            Some(alloc(&mut ctx.next_reg))
        };
        ctx.instrs.push(IrInstr::Call {
            dst,
            name: name.clone(),
            args: regs,
        });
        return Ok(());
    }

    let _ = lower_expr(expr, &mut ctx.next_reg, &mut ctx.instrs, env, fn_table)?;
    Ok(())
}

#[derive(Debug, Default)]
struct LoweringCtx {
    next_reg: u16,
    next_label_id: u32,
    instrs: Vec<IrInstr>,
}

impl LoweringCtx {
    fn new() -> Self {
        Self {
            next_reg: 0,
            next_label_id: 0,
            instrs: Vec::new(),
        }
    }

    fn next_if_id(&mut self) -> u32 {
        let id = self.next_label_id;
        self.next_label_id += 1;
        id
    }

    fn ends_with_ret(&self) -> bool {
        matches!(self.instrs.last(), Some(IrInstr::Ret { .. }))
    }
}

#[inline]
fn alloc(next: &mut u16) -> u16 {
    let out = *next;
    *next += 1;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexer_recognizes_core_tokens() {
        let tokens = lex("fn x(a: quad) -> quad { let c = !a && T; }").expect("lex");
        assert!(tokens.iter().any(|t| t.kind == TokenKind::KwFn));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::AndAnd));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Bang));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Arrow));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::QuadT));
    }

    #[test]
    fn typecheck_rejects_if_quad_condition() {
        let src = r#"
			fn evaluate(state: quad) -> quad {
				if state { return T; }
				return F;
			}
		"#;
        let f = parse_function(src).expect("parse");
        let err = type_check_function(&f).expect_err("must fail");
        assert!(err.message.contains("if condition must be bool"));
    }

    #[test]
    fn typecheck_accepts_explicit_quad_compare() {
        let src = r#"
			fn evaluate(state: quad) -> quad {
				if state == T { return T; }
				return S;
			}
		"#;
        let f = parse_function(src).expect("parse");
        type_check_function(&f).expect("must pass");
    }

    #[test]
    fn typecheck_rejects_bool_implies() {
        let src = r#"
			fn f(a: bool, b: bool) -> bool {
				return a -> b;
			}
		"#;
        let f = parse_function(src).expect("parse");
        let err = type_check_function(&f).expect_err("must fail");
        assert!(err.message.contains("allowed only for quad"));
    }

    #[test]
    fn if_scopes_do_not_leak() {
        let src = r#"
			fn f(a: bool) -> bool {
				if a { let t = true; }
				return t;
			}
		"#;
        let f = parse_function(src).expect("parse");
        let err = type_check_function(&f).expect_err("must fail");
        assert!(err.message.contains("unknown variable 't'"));
    }

    #[test]
    fn lower_num_uses_load_i32() {
        let expr = Expr::Num(42);
        let ir = lower_expr_to_ir(&expr, &HashMap::new(), &HashMap::new()).expect("lower");
        assert!(ir
            .iter()
            .any(|i| matches!(i, IrInstr::LoadI32 { val: 42, .. })));
    }

    #[test]
    fn lower_bool_ops_use_bool_instrs() {
        let expr = Expr::Binary(
            Box::new(Expr::BoolLiteral(true)),
            BinaryOp::AndAnd,
            Box::new(Expr::Unary(
                UnaryOp::Not,
                Box::new(Expr::BoolLiteral(false)),
            )),
        );
        let ir = lower_expr_to_ir(&expr, &HashMap::new(), &HashMap::new()).expect("lower");
        assert!(ir.iter().any(|i| matches!(i, IrInstr::BoolNot { .. })));
        assert!(ir.iter().any(|i| matches!(i, IrInstr::BoolAnd { .. })));
    }

    #[test]
    fn call_is_typed_via_fn_table() {
        let expr = Expr::Call("idq".to_string(), vec![Expr::QuadLiteral(QuadVal::T)]);
        let mut fns = HashMap::new();
        fns.insert(
            "idq".to_string(),
            FnSig {
                params: vec![Type::Quad],
                ret: Type::Quad,
            },
        );
        let ir = lower_expr_to_ir(&expr, &HashMap::new(), &fns).expect("lower");
        assert!(ir
            .iter()
            .any(|i| matches!(i, IrInstr::Call { dst: Some(_), .. })));
    }

    #[test]
    fn parse_program_two_functions() {
        let src = r#"
			fn a() -> quad { return T; }
			fn main() { return; }
		"#;
        let p = parse_program(src).expect("parse");
        assert_eq!(p.functions.len(), 2);
        assert_eq!(p.functions[0].name, "a");
        assert_eq!(p.functions[1].name, "main");
    }

    #[test]
    fn typecheck_program_requires_main() {
        let src = r#"
			fn helper() -> quad { return T; }
		"#;
        let p = parse_program(src).expect("parse");
        let err = type_check_program(&p).expect_err("must fail");
        assert!(err.message.contains("must define fn main()"));
    }

    #[test]
    fn lower_if_emits_labels_and_jumps() {
        let src = r#"
			fn main() {
				let a = true;
				if a { return; } else { return; }
			}
		"#;
        let p = parse_program(src).expect("parse");
        let table = build_fn_table(&p.functions).expect("table");
        let ir = lower_function_to_ir(&p.functions[0], &table).expect("lower");
        assert!(ir.instrs.iter().any(|i| matches!(i, IrInstr::Label { .. })));
        assert!(ir.instrs.iter().any(|i| matches!(i, IrInstr::JmpIf { .. })));
        assert!(ir.instrs.iter().any(|i| matches!(i, IrInstr::Jmp { .. })));
    }

    #[test]
    fn lower_let_storevar() {
        let src = r#"
			fn main() {
				let x: quad = T;
				return;
			}
		"#;
        let p = parse_program(src).expect("parse");
        let table = build_fn_table(&p.functions).expect("table");
        let ir = lower_function_to_ir(&p.functions[0], &table).expect("lower");
        assert!(ir.instrs.iter().any(|i| matches!(
            i,
            IrInstr::StoreVar { name, .. } if name == "x"
        )));
    }

    #[test]
    fn exprstmt_allows_unit_call() {
        let src = r#"
			fn printq(q: quad) { return; }
			fn main() {
				let x: quad = T;
				printq(x);
				return;
			}
		"#;
        let p = parse_program(src).expect("parse");
        let table = build_fn_table(&p.functions).expect("table");
        type_check_program(&p).expect("typecheck");
        let ir = lower_function_to_ir(&p.functions[1], &table).expect("lower");
        assert!(ir.instrs.iter().any(|i| matches!(
            i,
            IrInstr::Call {
                dst: None,
                name,
                ..
            } if name == "printq"
        )));

        let bad_expr = Expr::Call("printq".to_string(), vec![Expr::QuadLiteral(QuadVal::T)]);
        let err = lower_expr_to_ir(&bad_expr, &HashMap::new(), &table).expect_err("must fail");
        assert!(err.message.contains("cannot be used as expression value"));
    }

    #[test]
    fn match_requires_default() {
        let src = r#"
			fn main() {
				let q: quad = T;
				match q {
					T => { return; }
				}
			}
		"#;
        let p = parse_program(src).expect("parse");
        let err = type_check_program(&p).expect_err("must fail");
        assert!(err.message.contains("match requires default"));
    }

    #[test]
    fn match_only_quad() {
        let src = r#"
			fn main() {
				let b: bool = true;
				match b {
					T => { return; }
					_ => { return; }
				}
			}
		"#;
        let p = parse_program(src).expect("parse");
        let err = type_check_program(&p).expect_err("must fail");
        assert!(err.message.contains("match is allowed only for quad"));
    }

    #[test]
    fn lower_match_emits_chain_and_end_label() {
        let src = r#"
			fn main() {
				let q: quad = T;
				match q {
					N => { return; }
					T => { return; }
					_ => { return; }
				}
			}
		"#;
        let p = parse_program(src).expect("parse");
        let table = build_fn_table(&p.functions).expect("table");
        let ir = lower_function_to_ir(&p.functions[0], &table).expect("lower");
        assert!(ir.instrs.iter().any(|i| matches!(i, IrInstr::CmpEq { .. })));
        assert!(ir.instrs.iter().any(|i| matches!(i, IrInstr::JmpIf { .. })));
        assert!(ir.instrs.iter().any(|i| matches!(
            i,
            IrInstr::Label { name } if name.contains("match_") && name.contains("_end")
        )));
    }

    #[test]
    fn else_if_parses_and_lowers() {
        let src = r#"
			fn main() {
				let a: bool = true;
				let b: bool = false;
				if a { return; } else if b { return; } else { return; }
			}
		"#;
        let p = parse_program(src).expect("parse");
        let table = build_fn_table(&p.functions).expect("table");
        let ir = lower_function_to_ir(&p.functions[0], &table).expect("lower");
        let jmp_ifs = ir
            .instrs
            .iter()
            .filter(|i| matches!(i, IrInstr::JmpIf { .. }))
            .count();
        assert!(jmp_ifs >= 2);
    }

    #[test]
    fn emit_exobyte_has_header_and_functions() {
        let src = r#"
			fn helper() -> quad { return T; }
			fn main() { return; }
		"#;
        let bytes = compile_program_to_exobyte(src).expect("compile");
        assert!(bytes.len() >= 8);
        assert_eq!(&bytes[0..8], &MAGIC);
    }

    #[test]
    fn validate_ir_rejects_unknown_label() {
        let f = IrFunction {
            name: "main".to_string(),
            instrs: vec![
                IrInstr::LoadBool { dst: 0, val: true },
                IrInstr::Jmp {
                    label: "missing".to_string(),
                },
                IrInstr::Ret { src: None },
            ],
        };
        let err = validate_ir(&f).expect_err("must fail");
        assert!(err.message.contains("unknown label"));
    }
}
