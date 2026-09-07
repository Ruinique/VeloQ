use crate::error::BankError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    F16,
    BF16,
    F32,
    F64,
    I8,
    I16,
    I32,
}

impl DataType {
    pub fn size_bytes(&self) -> u64 {
        match self {
            Self::I8 => 1,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 => 8,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
        }
    }
}

impl FromStr for DataType {
    type Err = BankError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "half" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::BF16),
            "f32" | "fp32" | "float" => Ok(Self::F32),
            "f64" | "fp64" | "double" => Ok(Self::F64),
            "i8" | "int8" => Ok(Self::I8),
            "i16" | "int16" => Ok(Self::I16),
            "i32" | "int32" => Ok(Self::I32),
            _ => Err(BankError::InvalidDataType { raw: s.to_string() }),
        }
    }
}
