//! EXOcode Core — Quadro Logic Engine
//!
//! Packed Quadro logic:
//! - 64-bit register stores 32 quadits (2 bits each).
//! - Quadit encoding: N=00, F=01, T=10, S=11
//! - All masks exposed by this crate are LSB-aligned: only even bit positions.
//!
//! Features:
//! - "simd": Enables AVX2/NEON optimizations.
//! - "std": Enables runtime CPU feature detection (x86_64).
//! - "bench": Enables internal micro-benchmarks.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "simd", allow(dead_code))]

#[cfg(feature = "std")]
extern crate std;

use core::fmt;

#[cfg(feature = "std")]
pub mod exobyte_format;
#[cfg(feature = "std")]
pub mod exobyte_vm;
#[cfg(feature = "std")]
pub mod frontend;
#[cfg(feature = "std")]
pub mod language;
#[cfg(feature = "std")]
pub mod parser;

// ------------------------------
// Public constants (quadit states)
// ------------------------------
pub const N: u8 = 0b00;
pub const F: u8 = 0b01;
pub const T: u8 = 0b10;
pub const S: u8 = 0b11;

// ------------------------------
// Bit-plane masks
// ------------------------------
pub const LSB_MASK: u64 = 0x5555_5555_5555_5555; // 0101...
pub const MSB_MASK: u64 = 0xAAAA_AAAA_AAAA_AAAA; // 1010...

#[allow(unused)]
#[inline(always)]
fn debug_assert_lsb_aligned(mask: u64) {
    debug_assert_eq!(
        mask & MSB_MASK,
        0,
        "mask must be LSB-aligned (no bits in MSB positions)"
    );
}

#[inline(always)]
fn debug_assert_valid_state(state: u8) {
    debug_assert!(state <= 0b11, "invalid quadit state");
}

/// Expand LSB-aligned mask (1 bit per quadit) into 2-bit-per-quadit mask.
#[inline(always)]
fn expand_mask2(mask_lsb: u64) -> u64 {
    debug_assert_lsb_aligned(mask_lsb);
    mask_lsb | (mask_lsb << 1)
}

/// Count quadits selected by an LSB mask.
#[inline(always)]
pub fn popcount_quadits(mask_lsb: u64) -> u32 {
    debug_assert_lsb_aligned(mask_lsb);
    mask_lsb.count_ones()
}

/// Iterate LSB-aligned mask indices (0..32) where bit is set.
#[inline]
pub fn iter_mask_indices(mask_lsb: u64) -> MaskIndexIter {
    debug_assert_lsb_aligned(mask_lsb);
    MaskIndexIter { mask_lsb }
}

pub struct MaskIndexIter {
    mask_lsb: u64,
}

impl Iterator for MaskIndexIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.mask_lsb == 0 {
            return None;
        }
        let tz = self.mask_lsb.trailing_zeros();
        self.mask_lsb &= self.mask_lsb - 1;
        Some((tz / 2) as usize)
    }
}

/// 64-bit packed register containing 32 quadits (2 bits each).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct QuadroReg(u64);

impl QuadroReg {
    #[inline]
    pub fn new() -> Self {
        Self(0)
    }
    #[inline]
    pub fn from_raw(val: u64) -> Self {
        Self(val)
    }
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }

    // --------------------------
    // Safe quadit access
    // --------------------------
    #[inline]
    pub fn try_get(self, index: usize) -> Option<u8> {
        if index >= 32 {
            return None;
        }
        let shift = index * 2;
        Some(((self.0 >> shift) & 0b11) as u8)
    }

    #[inline]
    pub fn try_set(&mut self, index: usize, state: u8) -> Result<(), &'static str> {
        if index >= 32 {
            return Err("Index out of bounds");
        }
        if state > 0b11 {
            return Err("Invalid quadit state");
        }
        unsafe { self.set_unchecked(index, state) };
        Ok(())
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(self, index: usize) -> u8 {
        let shift = index * 2;
        ((self.0 >> shift) & 0b11) as u8
    }

    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, index: usize, state: u8) {
        debug_assert!(index < 32);
        debug_assert_valid_state(state);
        let shift = index * 2;
        let clear_mask = !(0b11_u64 << shift);
        self.0 = (self.0 & clear_mask) | ((state as u64) << shift);
    }

    // --------------------------
    // SWAR bit-plane analysis
    // --------------------------
    #[inline(always)]
    fn analyze_bits(self) -> (u64, u64) {
        let lsb = self.0 & LSB_MASK;
        let msb = (self.0 & MSB_MASK) >> 1;
        (lsb, msb)
    }

    #[inline]
    pub fn masks_all(self) -> QuadMasks {
        let (lsb, msb) = self.analyze_bits();
        let m_n = ((!lsb) & (!msb)) & LSB_MASK;
        let m_f = (lsb & (!msb)) & LSB_MASK;
        let m_t = (msb & (!lsb)) & LSB_MASK;
        let m_s = (msb & lsb) & LSB_MASK;

        debug_assert_lsb_aligned(m_n);
        debug_assert_lsb_aligned(m_f);
        debug_assert_lsb_aligned(m_t);
        debug_assert_lsb_aligned(m_s);

        QuadMasks {
            n: m_n,
            f: m_f,
            t: m_t,
            s: m_s,
        }
    }

    #[inline]
    pub fn mask_null(self) -> u64 {
        let (lsb, msb) = self.analyze_bits();
        ((!lsb) & (!msb)) & LSB_MASK
    }

    #[inline]
    pub fn mask_strict_false(self) -> u64 {
        let (lsb, msb) = self.analyze_bits();
        (lsb & (!msb)) & LSB_MASK
    }

    #[inline]
    pub fn mask_strict_true(self) -> u64 {
        let (lsb, msb) = self.analyze_bits();
        (msb & (!lsb)) & LSB_MASK
    }

    #[inline]
    pub fn mask_super(self) -> u64 {
        let (lsb, msb) = self.analyze_bits();
        (msb & lsb) & LSB_MASK
    }

    #[inline]
    pub fn mask_non_null(self) -> u64 {
        (!self.mask_null()) & LSB_MASK
    }

    // --------------------------
    // Gates
    // --------------------------
    #[must_use]
    #[inline]
    pub fn merge(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    #[must_use]
    #[inline]
    pub fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
    #[must_use]
    #[inline]
    pub fn raw_delta(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    #[must_use]
    #[inline]
    pub fn inverse(self) -> Self {
        let even = (self.0 & MSB_MASK) >> 1;
        let odd = (self.0 & LSB_MASK) << 1;
        Self(even | odd)
    }

    // --------------------------
    // Mask-based writes
    // --------------------------
    #[inline]
    pub fn set_by_mask(mut self, mask_lsb: u64, state: u8) -> Self {
        debug_assert_lsb_aligned(mask_lsb);
        debug_assert_valid_state(state);

        let mask2 = expand_mask2(mask_lsb);
        self.0 &= !mask2;

        let pattern = match state {
            N => 0,
            F => mask_lsb,
            T => mask_lsb << 1,
            S => mask2,
            _ => unreachable!("invalid quadit state"),
        };

        self.0 |= pattern;
        self
    }

    #[inline]
    pub fn try_set_by_mask(self, mask_lsb: u64, state: u8) -> Result<Self, &'static str> {
        if (mask_lsb & MSB_MASK) != 0 {
            return Err("mask must be LSB-aligned");
        }
        if state > 0b11 {
            return Err("invalid quadit state");
        }
        Ok(self.set_by_mask(mask_lsb, state))
    }

    #[inline]
    pub fn clear_by_mask(self, mask_lsb: u64) -> Self {
        self.set_by_mask(mask_lsb, N)
    }

    #[inline]
    pub fn force_super(self, mask_lsb: u64) -> Self {
        self.set_by_mask(mask_lsb, S)
    }

    // --------------------------
    // Events / Delta
    // --------------------------
    #[inline]
    pub fn calc_delta(prev: Self, current: Self) -> StateDelta {
        let prev_m = prev.masks_all();
        let curr_m = current.masks_all();

        let entered_true = ((!prev_m.t) & curr_m.t) & LSB_MASK;
        let left_true = (prev_m.t & (!curr_m.t)) & LSB_MASK;

        let entered_false = ((!prev_m.f) & curr_m.f) & LSB_MASK;
        let left_false = (prev_m.f & (!curr_m.f)) & LSB_MASK;

        let entered_super = ((!prev_m.s) & curr_m.s) & LSB_MASK;
        let left_super = (prev_m.s & (!curr_m.s)) & LSB_MASK;

        debug_assert_lsb_aligned(entered_true);
        debug_assert_lsb_aligned(left_true);
        debug_assert_lsb_aligned(entered_false);
        debug_assert_lsb_aligned(left_false);
        debug_assert_lsb_aligned(entered_super);
        debug_assert_lsb_aligned(left_super);

        StateDelta {
            entered_true,
            left_true,
            entered_false,
            left_false,
            entered_super,
            left_super,
        }
    }
}

/// State masks (LSB-aligned).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct QuadMasks {
    pub n: u64,
    pub f: u64,
    pub t: u64,
    pub s: u64,
}

impl fmt::Debug for QuadMasks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuadMasks")
            .field("n", &format_args!("{:#018x}", self.n))
            .field("f", &format_args!("{:#018x}", self.f))
            .field("t", &format_args!("{:#018x}", self.t))
            .field("s", &format_args!("{:#018x}", self.s))
            .finish()
    }
}

/// Transition masks (LSB-aligned, Array-of-Structures layout).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StateDelta {
    pub entered_true: u64,
    pub left_true: u64,
    pub entered_false: u64,
    pub left_false: u64,
    pub entered_super: u64,
    pub left_super: u64,
}

impl fmt::Debug for StateDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateDelta")
            .field("entered_true", &format_args!("{:#018x}", self.entered_true))
            .field("left_true", &format_args!("{:#018x}", self.left_true))
            .field(
                "entered_false",
                &format_args!("{:#018x}", self.entered_false),
            )
            .field("left_false", &format_args!("{:#018x}", self.left_false))
            .field(
                "entered_super",
                &format_args!("{:#018x}", self.entered_super),
            )
            .field("left_super", &format_args!("{:#018x}", self.left_super))
            .finish()
    }
}

/// Delta results in SoA (Structure of Arrays) layout.
#[derive(Clone, Copy)]
pub struct DeltaSoA<const N: usize> {
    pub entered_true: [u64; N],
    pub left_true: [u64; N],
    pub entered_false: [u64; N],
    pub left_false: [u64; N],
    pub entered_super: [u64; N],
    pub left_super: [u64; N],
}

impl<const N: usize> DeltaSoA<N> {
    #[inline]
    pub fn new() -> Self {
        Self {
            entered_true: [0; N],
            left_true: [0; N],
            entered_false: [0; N],
            left_false: [0; N],
            entered_super: [0; N],
            left_super: [0; N],
        }
    }

    #[inline]
    pub fn to_aos(&self) -> [StateDelta; N] {
        let mut out = [StateDelta {
            entered_true: 0,
            left_true: 0,
            entered_false: 0,
            left_false: 0,
            entered_super: 0,
            left_super: 0,
        }; N];

        for i in 0..N {
            out[i] = StateDelta {
                entered_true: self.entered_true[i],
                left_true: self.left_true[i],
                entered_false: self.entered_false[i],
                left_false: self.left_false[i],
                entered_super: self.entered_super[i],
                left_super: self.left_super[i],
            };
        }
        out
    }
}

impl fmt::Debug for QuadroReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QReg[")?;
        for i in (0..32).rev() {
            let val = self.try_get(i).unwrap_or(N);
            match val {
                N => write!(f, ".")?,
                F => write!(f, "F")?,
                T => write!(f, "T")?,
                S => write!(f, "S")?,
                _ => write!(f, "?")?,
            }
        }
        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// SIMD Engine: AVX2 (x86_64) & NEON (AArch64)
// Feature-gated: enable with `--features simd`
// ---------------------------------------------------------------------------
#[cfg(feature = "simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64"))]
pub mod simd_opt {
    use super::{QuadroReg, LSB_MASK, MSB_MASK};

    #[cfg(debug_assertions)]
    use super::debug_assert_lsb_aligned;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    use core::arch::x86_64::*;

    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::*;

    #[inline]
    pub fn bulk_merge(dest: &mut [QuadroReg], src: &[QuadroReg]) -> usize {
        let len = dest.len().min(src.len());

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if cfg!(target_feature = "avx2") {
                unsafe {
                    return bulk_merge_avx2(dest, src, len);
                }
            }
            #[cfg(feature = "std")]
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    return bulk_merge_avx2(dest, src, len);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            return bulk_merge_neon(dest, src, len);
        }

        #[allow(unreachable_code)]
        {
            let _ = len;
            0
        }
    }

    #[inline]
    pub fn bulk_intersect(dest: &mut [QuadroReg], src: &[QuadroReg]) -> usize {
        let len = dest.len().min(src.len());

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if cfg!(target_feature = "avx2") {
                unsafe {
                    return bulk_intersect_avx2(dest, src, len);
                }
            }
            #[cfg(feature = "std")]
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    return bulk_intersect_avx2(dest, src, len);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            return bulk_intersect_neon(dest, src, len);
        }

        #[allow(unreachable_code)]
        {
            let _ = len;
            0
        }
    }

    #[inline]
    pub fn bulk_inverse(dest: &mut [QuadroReg]) -> usize {
        let len = dest.len();

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if cfg!(target_feature = "avx2") {
                unsafe {
                    return bulk_inverse_avx2(dest, len);
                }
            }
            #[cfg(feature = "std")]
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    return bulk_inverse_avx2(dest, len);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            return bulk_inverse_neon(dest, len);
        }

        #[allow(unreachable_code)]
        {
            let _ = len;
            0
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn bulk_calc_delta(
        prev: &[QuadroReg],
        curr: &[QuadroReg],
        out_et: &mut [u64],
        out_lt: &mut [u64],
        out_ef: &mut [u64],
        out_lf: &mut [u64],
        out_es: &mut [u64],
        out_ls: &mut [u64],
    ) -> usize {
        let len = prev.len();
        debug_assert!(len <= curr.len());
        debug_assert!(out_et.len() >= len);
        debug_assert!(out_lt.len() >= len);
        debug_assert!(out_ef.len() >= len);
        debug_assert!(out_lf.len() >= len);
        debug_assert!(out_es.len() >= len);
        debug_assert!(out_ls.len() >= len);

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if cfg!(target_feature = "avx2") {
                unsafe {
                    return bulk_calc_delta_avx2(
                        prev, curr, out_et, out_lt, out_ef, out_lf, out_es, out_ls, len,
                    );
                }
            }
            #[cfg(feature = "std")]
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    return bulk_calc_delta_avx2(
                        prev, curr, out_et, out_lt, out_ef, out_lf, out_es, out_ls, len,
                    );
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            return bulk_calc_delta_neon(
                prev, curr, out_et, out_lt, out_ef, out_lf, out_es, out_ls, len,
            );
        }

        #[allow(unreachable_code)]
        {
            let _ = len;
            0
        }
    }

    // --- AVX2 (4x u64) ---
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2")]
    unsafe fn bulk_merge_avx2(dest: &mut [QuadroReg], src: &[QuadroReg], len: usize) -> usize {
        let mut i = 0;
        while i + 4 <= len {
            let a = _mm256_loadu_si256(dest.as_ptr().add(i) as *const __m256i);
            let b = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(
                dest.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_or_si256(a, b),
            );
            i += 4;
        }
        i
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2")]
    unsafe fn bulk_intersect_avx2(dest: &mut [QuadroReg], src: &[QuadroReg], len: usize) -> usize {
        let mut i = 0;
        while i + 4 <= len {
            let a = _mm256_loadu_si256(dest.as_ptr().add(i) as *const __m256i);
            let b = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(
                dest.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_and_si256(a, b),
            );
            i += 4;
        }
        i
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2")]
    unsafe fn bulk_inverse_avx2(dest: &mut [QuadroReg], len: usize) -> usize {
        let mask_lsb = _mm256_set1_epi64x(LSB_MASK as i64);
        let mask_msb = _mm256_set1_epi64x(MSB_MASK as i64);

        let mut i = 0;
        while i + 4 <= len {
            let v = _mm256_loadu_si256(dest.as_ptr().add(i) as *const __m256i);
            let even = _mm256_and_si256(v, mask_msb);
            let down = _mm256_srli_epi64(even, 1);
            let odd = _mm256_and_si256(v, mask_lsb);
            let up = _mm256_slli_epi64(odd, 1);
            _mm256_storeu_si256(
                dest.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_or_si256(down, up),
            );
            i += 4;
        }
        i
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn bulk_calc_delta_avx2(
        prev: &[QuadroReg],
        curr: &[QuadroReg],
        out_et: &mut [u64],
        out_lt: &mut [u64],
        out_ef: &mut [u64],
        out_lf: &mut [u64],
        out_es: &mut [u64],
        out_ls: &mut [u64],
        len: usize,
    ) -> usize {
        let mask_lsb = _mm256_set1_epi64x(LSB_MASK as i64);
        let mask_msb = _mm256_set1_epi64x(MSB_MASK as i64);

        let mut i = 0;
        while i + 4 <= len {
            let p = _mm256_loadu_si256(prev.as_ptr().add(i) as *const __m256i);
            let c = _mm256_loadu_si256(curr.as_ptr().add(i) as *const __m256i);

            let p_lsb = _mm256_and_si256(p, mask_lsb);
            let c_lsb = _mm256_and_si256(c, mask_lsb);
            let p_msb = _mm256_srli_epi64(_mm256_and_si256(p, mask_msb), 1);
            let c_msb = _mm256_srli_epi64(_mm256_and_si256(c, mask_msb), 1);

            let p_t = _mm256_andnot_si256(p_lsb, p_msb);
            let c_t = _mm256_andnot_si256(c_lsb, c_msb);
            let p_f = _mm256_andnot_si256(p_msb, p_lsb);
            let c_f = _mm256_andnot_si256(c_msb, c_lsb);
            let p_s = _mm256_and_si256(p_lsb, p_msb);
            let c_s = _mm256_and_si256(c_lsb, c_msb);

            let r_et = _mm256_andnot_si256(p_t, c_t);
            let r_lt = _mm256_andnot_si256(c_t, p_t);
            let r_ef = _mm256_andnot_si256(p_f, c_f);
            let r_lf = _mm256_andnot_si256(c_f, p_f);
            let r_es = _mm256_andnot_si256(p_s, c_s);
            let r_ls = _mm256_andnot_si256(c_s, p_s);

            _mm256_storeu_si256(out_et.as_mut_ptr().add(i) as *mut __m256i, r_et);
            _mm256_storeu_si256(out_lt.as_mut_ptr().add(i) as *mut __m256i, r_lt);
            _mm256_storeu_si256(out_ef.as_mut_ptr().add(i) as *mut __m256i, r_ef);
            _mm256_storeu_si256(out_lf.as_mut_ptr().add(i) as *mut __m256i, r_lf);
            _mm256_storeu_si256(out_es.as_mut_ptr().add(i) as *mut __m256i, r_es);
            _mm256_storeu_si256(out_ls.as_mut_ptr().add(i) as *mut __m256i, r_ls);

            i += 4;
        }

        // Debug verification for produced SIMD span
        #[cfg(debug_assertions)]
        {
            for k in 0..i {
                debug_assert_lsb_aligned(out_et[k]);
                debug_assert_lsb_aligned(out_lt[k]);
                debug_assert_lsb_aligned(out_ef[k]);
                debug_assert_lsb_aligned(out_lf[k]);
                debug_assert_lsb_aligned(out_es[k]);
                debug_assert_lsb_aligned(out_ls[k]);
                debug_assert_eq!(
                    (out_et[k] | out_lt[k] | out_ef[k] | out_lf[k] | out_es[k] | out_ls[k])
                        & MSB_MASK,
                    0
                );
            }
        }

        i
    }

    // --- NEON (2x u64) ---
    #[cfg(target_arch = "aarch64")]
    unsafe fn bulk_merge_neon(dest: &mut [QuadroReg], src: &[QuadroReg], len: usize) -> usize {
        let mut i = 0;
        while i + 2 <= len {
            let a = vld1q_u64(dest.as_ptr().add(i) as *const u64);
            let b = vld1q_u64(src.as_ptr().add(i) as *const u64);
            vst1q_u64(dest.as_mut_ptr().add(i) as *mut u64, vorrq_u64(a, b));
            i += 2;
        }
        i
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn bulk_intersect_neon(dest: &mut [QuadroReg], src: &[QuadroReg], len: usize) -> usize {
        let mut i = 0;
        while i + 2 <= len {
            let a = vld1q_u64(dest.as_ptr().add(i) as *const u64);
            let b = vld1q_u64(src.as_ptr().add(i) as *const u64);
            vst1q_u64(dest.as_mut_ptr().add(i) as *mut u64, vandq_u64(a, b));
            i += 2;
        }
        i
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn bulk_inverse_neon(dest: &mut [QuadroReg], len: usize) -> usize {
        let mask_lsb = vdupq_n_u64(LSB_MASK);
        let mask_msb = vdupq_n_u64(MSB_MASK);

        let mut i = 0;
        while i + 2 <= len {
            let v = vld1q_u64(dest.as_ptr().add(i) as *const u64);
            let even = vandq_u64(v, mask_msb);
            let down = vshrq_n_u64(even, 1);
            let odd = vandq_u64(v, mask_lsb);
            let up = vshlq_n_u64(odd, 1);
            vst1q_u64(dest.as_mut_ptr().add(i) as *mut u64, vorrq_u64(down, up));
            i += 2;
        }
        i
    }

    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn bulk_calc_delta_neon(
        prev: &[QuadroReg],
        curr: &[QuadroReg],
        out_et: &mut [u64],
        out_lt: &mut [u64],
        out_ef: &mut [u64],
        out_lf: &mut [u64],
        out_es: &mut [u64],
        out_ls: &mut [u64],
        len: usize,
    ) -> usize {
        let mask_lsb = vdupq_n_u64(LSB_MASK);
        let mask_msb = vdupq_n_u64(MSB_MASK);

        let mut i = 0;
        while i + 2 <= len {
            let pv = vld1q_u64(prev.as_ptr().add(i) as *const u64);
            let cv = vld1q_u64(curr.as_ptr().add(i) as *const u64);

            let p_lsb = vandq_u64(pv, mask_lsb);
            let c_lsb = vandq_u64(cv, mask_lsb);
            let p_msb = vshrq_n_u64(vandq_u64(pv, mask_msb), 1);
            let c_msb = vshrq_n_u64(vandq_u64(cv, mask_msb), 1);

            let p_t = vbicq_u64(p_msb, p_lsb);
            let c_t = vbicq_u64(c_msb, c_lsb);
            let p_f = vbicq_u64(p_lsb, p_msb);
            let c_f = vbicq_u64(c_lsb, c_msb);
            let p_s = vandq_u64(p_lsb, p_msb);
            let c_s = vandq_u64(c_lsb, c_msb);

            let r_et = vbicq_u64(c_t, p_t);
            let r_lt = vbicq_u64(p_t, c_t);
            let r_ef = vbicq_u64(c_f, p_f);
            let r_lf = vbicq_u64(p_f, c_f);
            let r_es = vbicq_u64(c_s, p_s);
            let r_ls = vbicq_u64(p_s, c_s);

            vst1q_u64(out_et.as_mut_ptr().add(i) as *mut u64, r_et);
            vst1q_u64(out_lt.as_mut_ptr().add(i) as *mut u64, r_lt);
            vst1q_u64(out_ef.as_mut_ptr().add(i) as *mut u64, r_ef);
            vst1q_u64(out_lf.as_mut_ptr().add(i) as *mut u64, r_lf);
            vst1q_u64(out_es.as_mut_ptr().add(i) as *mut u64, r_es);
            vst1q_u64(out_ls.as_mut_ptr().add(i) as *mut u64, r_ls);

            i += 2;
        }

        #[cfg(debug_assertions)]
        {
            for k in 0..i {
                debug_assert_lsb_aligned(out_et[k]);
                debug_assert_lsb_aligned(out_lt[k]);
                debug_assert_lsb_aligned(out_ef[k]);
                debug_assert_lsb_aligned(out_lf[k]);
                debug_assert_lsb_aligned(out_es[k]);
                debug_assert_lsb_aligned(out_ls[k]);
                debug_assert_eq!(
                    (out_et[k] | out_lt[k] | out_ef[k] | out_lf[k] | out_es[k] | out_ls[k])
                        & MSB_MASK,
                    0
                );
            }
        }

        i
    }
}

// ---------------------------------------------------------------------------
// Bank (multi-register) utilities
// ---------------------------------------------------------------------------
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct QuadroBank<const NREG: usize> {
    pub regs: [QuadroReg; NREG],
}

impl<const NREG: usize> QuadroBank<NREG> {
    #[inline]
    pub fn new() -> Self {
        Self {
            regs: [QuadroReg::new(); NREG],
        }
    }

    #[inline]
    pub fn merge_inplace(&mut self, other: &Self) {
        #[cfg(all(
            feature = "simd",
            any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
        ))]
        let start = simd_opt::bulk_merge(&mut self.regs, &other.regs);

        #[cfg(not(all(
            feature = "simd",
            any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
        )))]
        let start = 0;

        for i in start..NREG {
            self.regs[i] = self.regs[i].merge(other.regs[i]);
        }
    }

    #[inline]
    pub fn intersect_inplace(&mut self, other: &Self) {
        #[cfg(all(
            feature = "simd",
            any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
        ))]
        let start = simd_opt::bulk_intersect(&mut self.regs, &other.regs);

        #[cfg(not(all(
            feature = "simd",
            any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
        )))]
        let start = 0;

        for i in start..NREG {
            self.regs[i] = self.regs[i].intersect(other.regs[i]);
        }
    }

    #[inline]
    pub fn inverse_inplace(&mut self) {
        #[cfg(all(
            feature = "simd",
            any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
        ))]
        let start = simd_opt::bulk_inverse(&mut self.regs);

        #[cfg(not(all(
            feature = "simd",
            any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
        )))]
        let start = 0;

        for i in start..NREG {
            self.regs[i] = self.regs[i].inverse();
        }
    }

    #[inline]
    pub fn calc_deltas_soa(prev: &Self, curr: &Self) -> DeltaSoA<NREG> {
        let mut soa = DeltaSoA::<NREG>::new();

        #[cfg(all(
            feature = "simd",
            any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
        ))]
        let start = simd_opt::bulk_calc_delta(
            &prev.regs,
            &curr.regs,
            &mut soa.entered_true,
            &mut soa.left_true,
            &mut soa.entered_false,
            &mut soa.left_false,
            &mut soa.entered_super,
            &mut soa.left_super,
        );

        #[cfg(not(all(
            feature = "simd",
            any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
        )))]
        let start = 0;

        for i in start..NREG {
            let d = QuadroReg::calc_delta(prev.regs[i], curr.regs[i]);
            soa.entered_true[i] = d.entered_true;
            soa.left_true[i] = d.left_true;
            soa.entered_false[i] = d.entered_false;
            soa.left_false[i] = d.left_false;
            soa.entered_super[i] = d.entered_super;
            soa.left_super[i] = d.left_super;
        }

        soa
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------
#[cfg(all(feature = "bench", feature = "std"))]
pub mod bench {
    use super::*;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    pub struct BenchResult {
        pub iterations: u64,
        pub elapsed: Duration,
        pub ops_per_sec: f64,
    }

    #[inline]
    fn finish(start: Instant, iters: u64) -> BenchResult {
        let elapsed = start.elapsed();
        let secs = elapsed.as_secs_f64().max(1e-12);
        BenchResult {
            iterations: iters,
            elapsed,
            ops_per_sec: (iters as f64) / secs,
        }
    }

    pub fn bench_bank_soa<const NREG: usize>(iters: u64) -> BenchResult {
        let mut a = QuadroBank::<NREG>::new();
        let mut b = QuadroBank::<NREG>::new();

        for i in 0..NREG {
            a.regs[i] = QuadroReg::from_raw((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            b.regs[i] = QuadroReg::from_raw((i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        }

        // Warmup
        for _ in 0..256 {
            let soa = QuadroBank::<NREG>::calc_deltas_soa(&a, &b);
            black_box(soa.entered_true[0]);
        }

        let start = Instant::now();
        let mut acc = 0u64;

        for _ in 0..iters {
            let soa = QuadroBank::<NREG>::calc_deltas_soa(&a, &b);
            acc ^= black_box(soa.entered_true[0]);
        }

        black_box(acc);
        finish(start, iters)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_simd_correctness_and_alignment() {
        const NREG: usize = 17;
        let mut prev = QuadroBank::<NREG>::new();
        let mut curr = QuadroBank::<NREG>::new();

        for i in 0..NREG {
            let p_val = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let c_val = (i as u64)
                .wrapping_mul(0xD1B5_4A32_D192_ED03)
                .rotate_left(13);
            prev.regs[i] = QuadroReg::from_raw(p_val);
            curr.regs[i] = QuadroReg::from_raw(c_val);
        }

        let soa = QuadroBank::<NREG>::calc_deltas_soa(&prev, &curr);

        for i in 0..NREG {
            let scalar = QuadroReg::calc_delta(prev.regs[i], curr.regs[i]);

            assert_eq!(soa.entered_true[i], scalar.entered_true);
            assert_eq!(soa.left_true[i], scalar.left_true);
            assert_eq!(soa.entered_false[i], scalar.entered_false);
            assert_eq!(soa.left_false[i], scalar.left_false);
            assert_eq!(soa.entered_super[i], scalar.entered_super);
            assert_eq!(soa.left_super[i], scalar.left_super);

            let all = soa.entered_true[i]
                | soa.left_true[i]
                | soa.entered_false[i]
                | soa.left_false[i]
                | soa.entered_super[i]
                | soa.left_super[i];

            assert_eq!(all & MSB_MASK, 0, "MSB leakage detected");
        }
    }

    #[test]
    fn test_inverse_truth_table_on_single_quadit() {
        let r_t = QuadroReg::new().set_by_mask(1, T);
        assert_eq!(r_t.inverse().try_get(0), Some(F));

        let r_f = QuadroReg::new().set_by_mask(1, F);
        assert_eq!(r_f.inverse().try_get(0), Some(T));

        let r_n = QuadroReg::new().set_by_mask(1, N);
        assert_eq!(r_n.inverse().try_get(0), Some(N));

        let r_s = QuadroReg::new().set_by_mask(1, S);
        assert_eq!(r_s.inverse().try_get(0), Some(S));
    }

    #[test]
    fn test_inverse_inplace_bank_matches_scalar() {
        const N: usize = 9;
        let mut bank = QuadroBank::<N>::new();

        for i in 0..N {
            bank.regs[i] = QuadroReg::from_raw((i as u64).wrapping_mul(0x1234_5678_9ABC_DEF1));
        }

        let mut expected = bank;
        for i in 0..N {
            expected.regs[i] = expected.regs[i].inverse();
        }

        bank.inverse_inplace();

        for i in 0..N {
            assert_eq!(bank.regs[i].raw(), expected.regs[i].raw());
        }
    }

    #[test]
    fn test_try_set_and_try_set_by_mask_validation() {
        let mut reg = QuadroReg::new();

        assert_eq!(reg.try_set(32, T), Err("Index out of bounds"));
        assert_eq!(reg.try_set(0, 4), Err("Invalid quadit state"));
        assert!(reg.try_set(31, S).is_ok());
        assert_eq!(reg.try_get(31), Some(S));

        let misaligned_mask = 0b10_u64;
        assert_eq!(
            reg.try_set_by_mask(misaligned_mask, T),
            Err("mask must be LSB-aligned")
        );
        assert_eq!(reg.try_set_by_mask(1, 4), Err("invalid quadit state"));
    }

    #[test]
    fn test_masks_partition_and_non_null_consistency() {
        let mut reg = QuadroReg::new();
        reg = reg.set_by_mask(1_u64 << (0 * 2), N);
        reg = reg.set_by_mask(1_u64 << (1 * 2), F);
        reg = reg.set_by_mask(1_u64 << (2 * 2), T);
        reg = reg.set_by_mask(1_u64 << (3 * 2), S);

        let m = reg.masks_all();
        let union = m.n | m.f | m.t | m.s;

        assert_eq!(m.n & m.f, 0);
        assert_eq!(m.n & m.t, 0);
        assert_eq!(m.n & m.s, 0);
        assert_eq!(m.f & m.t, 0);
        assert_eq!(m.f & m.s, 0);
        assert_eq!(m.t & m.s, 0);
        assert_eq!(union, LSB_MASK);
        assert_eq!(reg.mask_non_null(), (m.f | m.t | m.s) & LSB_MASK);
    }

    #[test]
    fn test_popcount_and_iter_mask_indices() {
        let mask = (1_u64 << (0 * 2)) | (1_u64 << (3 * 2)) | (1_u64 << (31 * 2));
        let indices: Vec<usize> = iter_mask_indices(mask).collect();

        assert_eq!(popcount_quadits(mask), 3);
        assert_eq!(indices, vec![0, 3, 31]);
    }

    #[test]
    fn test_calc_delta_controlled_transitions() {
        let mut prev = QuadroReg::new();
        prev = prev.set_by_mask(1_u64 << (0 * 2), T);
        prev = prev.set_by_mask(1_u64 << (1 * 2), F);
        prev = prev.set_by_mask(1_u64 << (2 * 2), S);

        let mut curr = QuadroReg::new();
        curr = curr.set_by_mask(1_u64 << (0 * 2), F);
        curr = curr.set_by_mask(1_u64 << (1 * 2), T);
        curr = curr.set_by_mask(1_u64 << (2 * 2), N);
        curr = curr.set_by_mask(1_u64 << (3 * 2), S);

        let d = QuadroReg::calc_delta(prev, curr);
        let b0 = 1_u64 << (0 * 2);
        let b1 = 1_u64 << (1 * 2);
        let b2 = 1_u64 << (2 * 2);
        let b3 = 1_u64 << (3 * 2);

        assert_eq!(d.entered_true, b1);
        assert_eq!(d.left_true, b0);
        assert_eq!(d.entered_false, b0);
        assert_eq!(d.left_false, b1);
        assert_eq!(d.entered_super, b3);
        assert_eq!(d.left_super, b2);
    }
}
