use crate::error::BankError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct Swizzle {
    pub b: u8,
    pub m: u8,
    pub s: u8,
}

impl Swizzle {
    pub fn new(b: u8, m: u8, s: u8) -> Result<Self, BankError> {
        if b == 0 {
            return Err(BankError::InvalidSwizzle {
                raw: format!("{b},{m},{s}"),
                reason: "B must be greater than 0".to_string(),
            });
        }
        if s == 0 {
            return Err(BankError::InvalidSwizzle {
                raw: format!("{b},{m},{s}"),
                reason: "S must be greater than 0".to_string(),
            });
        }
        if s < b {
            return Err(BankError::InvalidSwizzle {
                raw: format!("{b},{m},{s}"),
                reason: format!("S must be >= B, got S={s} < B={b}"),
            });
        }
        let total_bits = (m as u32)
            .checked_add(s as u32)
            .and_then(|ms| ms.checked_add(b as u32))
            .ok_or_else(|| BankError::InvalidSwizzle {
                raw: format!("{b},{m},{s}"),
                reason: "bit shift overflow".to_string(),
            })?;
        if total_bits > 64 {
            return Err(BankError::InvalidSwizzle {
                raw: format!("{b},{m},{s}"),
                reason: format!("bit range M + S + B = {total_bits} exceeds 64 bits"),
            });
        }
        Ok(Self { b, m, s })
    }

    pub fn apply(&self, offset: u64) -> Result<u64, BankError> {
        let b = self.b;
        let m = self.m;
        let s = self.s;

        let base_mask = if b >= 64 {
            u64::MAX
        } else {
            (1u64.checked_shl(b as u32).unwrap_or(0)).wrapping_sub(1)
        };

        let target_mask =
            base_mask
                .checked_shl(m as u32)
                .ok_or_else(|| BankError::InvalidSwizzle {
                    raw: format!("{b},{m},{s}"),
                    reason: "shift overflow in target mask".to_string(),
                })?;

        let source_shift =
            (m as u32)
                .checked_add(s as u32)
                .ok_or_else(|| BankError::InvalidSwizzle {
                    raw: format!("{b},{m},{s}"),
                    reason: "shift overflow in source mask".to_string(),
                })?;

        if source_shift >= 64 && b > 0 {
            return Err(BankError::InvalidSwizzle {
                raw: format!("{b},{m},{s}"),
                reason: "shift overflow in source mask".to_string(),
            });
        }

        let source_mask =
            base_mask
                .checked_shl(source_shift)
                .ok_or_else(|| BankError::InvalidSwizzle {
                    raw: format!("{b},{m},{s}"),
                    reason: "shift overflow in source mask".to_string(),
                })?;

        let _ = target_mask; // Validates target mask shape
        let source = (offset & source_mask) >> (s as u32);
        Ok(offset ^ source)
    }

    pub fn target_bits(&self) -> Vec<u32> {
        let m = self.m as u32;
        let b = self.b as u32;
        (m..(m + b)).collect()
    }

    pub fn source_bits(&self) -> Vec<u32> {
        let start = (self.m as u32) + (self.s as u32);
        let b = self.b as u32;
        (start..(start + b)).collect()
    }

    pub fn untouched_bits(&self) -> Vec<u32> {
        (0..(self.m as u32)).collect()
    }

    pub fn operation_str(&self) -> String {
        let t_lo = self.m;
        let t_hi = self.m + self.b - 1;
        let s_lo = self.m + self.s;
        let s_hi = self.m + self.s + self.b - 1;
        if self.b == 1 {
            format!("bits[{t_lo}] ^= bits[{s_lo}]")
        } else {
            format!("bits[{t_hi}:{t_lo}] ^= bits[{s_hi}:{s_lo}]")
        }
    }

    pub fn patterns(&self) -> u64 {
        1u64.checked_shl(self.b as u32).unwrap_or(u64::MAX)
    }

    pub fn to_array(&self) -> [u8; 3] {
        [self.b, self.m, self.s]
    }
}

impl FromStr for Swizzle {
    type Err = BankError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let cleaned = s
            .trim()
            .trim_start_matches('<')
            .trim_end_matches('>')
            .trim_start_matches('(')
            .trim_end_matches(')');
        let parts: Vec<&str> = cleaned
            .split([',', ' ', ';'])
            .filter(|p| !p.is_empty())
            .collect();

        if parts.len() != 3 {
            return Err(BankError::InvalidSwizzle {
                raw: s.to_string(),
                reason: format!("expected 3 components B,M,S, got {}", parts.len()),
            });
        }

        let p0 = parts.first().copied().unwrap_or("");
        let p1 = parts.get(1).copied().unwrap_or("");
        let p2 = parts.get(2).copied().unwrap_or("");

        // Check for negative shift
        if let Ok(signed_s) = p2.parse::<i64>()
            && signed_s <= 0
        {
            return Err(BankError::UnsupportedNegativeShift { raw: s.to_string() });
        }

        let b = p0.parse::<u8>().map_err(|e| BankError::InvalidSwizzle {
            raw: s.to_string(),
            reason: format!("failed to parse B as u8: {e}"),
        })?;

        let m = p1.parse::<u8>().map_err(|e| BankError::InvalidSwizzle {
            raw: s.to_string(),
            reason: format!("failed to parse M as u8: {e}"),
        })?;

        let s_val = p2.parse::<u8>().map_err(|e| BankError::InvalidSwizzle {
            raw: s.to_string(),
            reason: format!("failed to parse S as u8: {e}"),
        })?;

        Self::new(b, m, s_val)
    }
}
