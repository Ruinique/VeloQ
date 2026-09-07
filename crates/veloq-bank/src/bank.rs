use crate::error::BankError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct BankModel {
    pub bank_count: u32,
    pub bank_width_bytes: u32,
}

impl BankModel {
    pub fn new(bank_count: u32, bank_width_bytes: u32) -> Result<Self, BankError> {
        if bank_count == 0 {
            return Err(BankError::InvalidBankModel {
                reason: "bank count must be greater than 0".to_string(),
            });
        }
        if bank_width_bytes == 0 {
            return Err(BankError::InvalidBankModel {
                reason: "bank width bytes must be greater than 0".to_string(),
            });
        }
        Ok(Self {
            bank_count,
            bank_width_bytes,
        })
    }

    pub fn nvidia() -> Self {
        Self {
            bank_count: 32,
            bank_width_bytes: 4,
        }
    }

    pub fn amd() -> Self {
        Self {
            bank_count: 32,
            bank_width_bytes: 4,
        }
    }

    pub fn from_arch_or_override(
        arch: Option<&str>,
        banks: Option<u32>,
        bank_width: Option<u32>,
    ) -> Result<Self, BankError> {
        let mut model = match arch.map(|a| a.trim().to_ascii_lowercase()).as_deref() {
            Some("nvidia") | None => Self::nvidia(),
            Some("amd") | Some("amd-cdna") | Some("cdna") | Some("rdna") => Self::amd(),
            Some(other) => {
                return Err(BankError::InvalidBankModel {
                    reason: format!("unknown arch `{other}`: expected `nvidia` or `amd`"),
                });
            }
        };

        if let Some(b) = banks {
            if b == 0 {
                return Err(BankError::InvalidBankModel {
                    reason: "bank count must be greater than 0".to_string(),
                });
            }
            model.bank_count = b;
        }

        if let Some(w) = bank_width {
            if w == 0 {
                return Err(BankError::InvalidBankModel {
                    reason: "bank width must be greater than 0".to_string(),
                });
            }
            model.bank_width_bytes = w;
        }

        Ok(model)
    }

    pub fn bank_for_byte_address(&self, byte_address: u64) -> u32 {
        let width = self.bank_width_bytes as u64;
        let count = self.bank_count as u64;
        ((byte_address / width) % count) as u32
    }
}
