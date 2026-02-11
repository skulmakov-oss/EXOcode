pub const MAGIC: [u8; 8] = *b"EXOBYTE0";

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    LoadQ = 0x01,
    LoadBool = 0x02,
    LoadI32 = 0x03,
    LoadVar = 0x04,
    StoreVar = 0x05,
    QAnd = 0x10,
    QOr = 0x11,
    QNot = 0x12,
    QImpl = 0x13,
    BoolAnd = 0x14,
    BoolOr = 0x15,
    BoolNot = 0x16,
    CmpEq = 0x20,
    CmpNe = 0x21,
    Jmp = 0x30,
    JmpIf = 0x31,
    Call = 0x40,
    Ret = 0x41,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExobyteFormatError {
    UnexpectedEof,
    InvalidUtf8,
    UnknownOpcode(u8),
}

impl core::fmt::Display for ExobyteFormatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ExobyteFormatError::UnexpectedEof => write!(f, "unexpected EOF"),
            ExobyteFormatError::InvalidUtf8 => write!(f, "invalid utf8"),
            ExobyteFormatError::UnknownOpcode(v) => write!(f, "unknown opcode 0x{:02x}", v),
        }
    }
}

impl std::error::Error for ExobyteFormatError {}

impl Opcode {
    pub fn byte(self) -> u8 {
        self as u8
    }

    pub fn from_byte(v: u8) -> Result<Self, ExobyteFormatError> {
        match v {
            x if x == Self::LoadQ as u8 => Ok(Self::LoadQ),
            x if x == Self::LoadBool as u8 => Ok(Self::LoadBool),
            x if x == Self::LoadI32 as u8 => Ok(Self::LoadI32),
            x if x == Self::LoadVar as u8 => Ok(Self::LoadVar),
            x if x == Self::StoreVar as u8 => Ok(Self::StoreVar),
            x if x == Self::QAnd as u8 => Ok(Self::QAnd),
            x if x == Self::QOr as u8 => Ok(Self::QOr),
            x if x == Self::QNot as u8 => Ok(Self::QNot),
            x if x == Self::QImpl as u8 => Ok(Self::QImpl),
            x if x == Self::BoolAnd as u8 => Ok(Self::BoolAnd),
            x if x == Self::BoolOr as u8 => Ok(Self::BoolOr),
            x if x == Self::BoolNot as u8 => Ok(Self::BoolNot),
            x if x == Self::CmpEq as u8 => Ok(Self::CmpEq),
            x if x == Self::CmpNe as u8 => Ok(Self::CmpNe),
            x if x == Self::Jmp as u8 => Ok(Self::Jmp),
            x if x == Self::JmpIf as u8 => Ok(Self::JmpIf),
            x if x == Self::Call as u8 => Ok(Self::Call),
            x if x == Self::Ret as u8 => Ok(Self::Ret),
            _ => Err(ExobyteFormatError::UnknownOpcode(v)),
        }
    }
}

pub fn write_u16_le(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn write_u32_le(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn write_i32_le(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn read_u8(bytes: &[u8], i: &mut usize) -> Result<u8, ExobyteFormatError> {
    if *i >= bytes.len() {
        return Err(ExobyteFormatError::UnexpectedEof);
    }
    let v = bytes[*i];
    *i += 1;
    Ok(v)
}

pub fn read_u16_le(bytes: &[u8], i: &mut usize) -> Result<u16, ExobyteFormatError> {
    if *i + 2 > bytes.len() {
        return Err(ExobyteFormatError::UnexpectedEof);
    }
    let v = u16::from_le_bytes([bytes[*i], bytes[*i + 1]]);
    *i += 2;
    Ok(v)
}

pub fn read_u32_le(bytes: &[u8], i: &mut usize) -> Result<u32, ExobyteFormatError> {
    if *i + 4 > bytes.len() {
        return Err(ExobyteFormatError::UnexpectedEof);
    }
    let v = u32::from_le_bytes([bytes[*i], bytes[*i + 1], bytes[*i + 2], bytes[*i + 3]]);
    *i += 4;
    Ok(v)
}

pub fn read_i32_le(bytes: &[u8], i: &mut usize) -> Result<i32, ExobyteFormatError> {
    Ok(read_u32_le(bytes, i)? as i32)
}

pub fn read_utf8(bytes: &[u8], i: &mut usize, len: usize) -> Result<String, ExobyteFormatError> {
    if *i + len > bytes.len() {
        return Err(ExobyteFormatError::UnexpectedEof);
    }
    let s = std::str::from_utf8(&bytes[*i..*i + len])
        .map_err(|_| ExobyteFormatError::InvalidUtf8)?
        .to_string();
    *i += len;
    Ok(s)
}
