use crate::error::BankError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct Layout2D {
    pub shape: [u64; 2],
    pub stride: [u64; 2],
}

impl Layout2D {
    pub fn new(shape: [u64; 2], stride: [u64; 2]) -> Result<Self, BankError> {
        let rows = shape.first().copied().unwrap_or(0);
        let cols = shape.get(1).copied().unwrap_or(0);
        if rows == 0 || cols == 0 {
            return Err(BankError::InvalidShape {
                raw: format!("{rows}x{cols}"),
                reason: "shape dimensions must be non-zero".to_string(),
            });
        }
        Ok(Self { shape, stride })
    }

    pub fn logical_offset(&self, coord: [u64; 2]) -> Result<u64, BankError> {
        let shape_rows = self.shape.first().copied().unwrap_or(0);
        let shape_cols = self.shape.get(1).copied().unwrap_or(0);
        let row = coord.first().copied().unwrap_or(0);
        let col = coord.get(1).copied().unwrap_or(0);

        if row >= shape_rows || col >= shape_cols {
            return Err(BankError::InvalidCoordinate {
                row,
                col,
                shape_rows,
                shape_cols,
            });
        }

        let stride_row = self.stride.first().copied().unwrap_or(0);
        let stride_col = self.stride.get(1).copied().unwrap_or(0);

        let row_offset = row
            .checked_mul(stride_row)
            .ok_or(BankError::InvalidCoordinate {
                row,
                col,
                shape_rows,
                shape_cols,
            })?;

        let col_offset = col
            .checked_mul(stride_col)
            .ok_or(BankError::InvalidCoordinate {
                row,
                col,
                shape_rows,
                shape_cols,
            })?;

        row_offset
            .checked_add(col_offset)
            .ok_or(BankError::InvalidCoordinate {
                row,
                col,
                shape_rows,
                shape_cols,
            })
    }
}

pub fn parse_shape(s: &str) -> Result<[u64; 2], BankError> {
    let cleaned = s.trim().trim_start_matches('[').trim_end_matches(']');
    let parts: Vec<&str> = cleaned
        .split(['x', 'X', ',', ' '])
        .filter(|p| !p.is_empty())
        .collect();

    if parts.len() != 2 {
        return Err(BankError::InvalidShape {
            raw: s.to_string(),
            reason: format!("expected 2 dimensions (e.g. 8x64), got {}", parts.len()),
        });
    }

    let p0 = parts.first().copied().unwrap_or("");
    let p1 = parts.get(1).copied().unwrap_or("");

    let d0 = p0.parse::<u64>().map_err(|e| BankError::InvalidShape {
        raw: s.to_string(),
        reason: format!("invalid row dimension `{p0}`: {e}"),
    })?;

    let d1 = p1.parse::<u64>().map_err(|e| BankError::InvalidShape {
        raw: s.to_string(),
        reason: format!("invalid column dimension `{p1}`: {e}"),
    })?;

    if d0 == 0 || d1 == 0 {
        return Err(BankError::InvalidShape {
            raw: s.to_string(),
            reason: "dimensions must be greater than 0".to_string(),
        });
    }

    Ok([d0, d1])
}

pub fn parse_stride_2d(s: &str) -> Result<[u64; 2], BankError> {
    let cleaned = s.trim().trim_start_matches('[').trim_end_matches(']');
    let parts: Vec<&str> = cleaned
        .split([',', 'x', 'X', ' '])
        .filter(|p| !p.is_empty())
        .collect();

    if parts.len() != 2 {
        return Err(BankError::InvalidStride {
            raw: s.to_string(),
            reason: format!("expected 2 stride values (e.g. 64,1), got {}", parts.len()),
        });
    }

    let p0 = parts.first().copied().unwrap_or("");
    let p1 = parts.get(1).copied().unwrap_or("");

    let s0 = p0.parse::<u64>().map_err(|e| BankError::InvalidStride {
        raw: s.to_string(),
        reason: format!("invalid row stride `{p0}`: {e}"),
    })?;

    let s1 = p1.parse::<u64>().map_err(|e| BankError::InvalidStride {
        raw: s.to_string(),
        reason: format!("invalid column stride `{p1}`: {e}"),
    })?;

    Ok([s0, s1])
}

pub fn parse_stride_1d(s: &str) -> Result<u64, BankError> {
    let cleaned = s.trim();
    cleaned
        .parse::<u64>()
        .map_err(|e| BankError::InvalidStride {
            raw: s.to_string(),
            reason: format!("invalid element stride `{cleaned}`: {e}"),
        })
}
