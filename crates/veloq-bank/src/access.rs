use crate::error::BankError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AccessEntry {
    pub lane: u32,
    pub coord: [u64; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AccessFileFormat {
    Object { accesses: Vec<AccessEntry> },
    List(Vec<AccessEntry>),
}

pub fn load_accesses_from_file(path: &Path) -> Result<Vec<AccessEntry>, BankError> {
    let content = fs::read_to_string(path).map_err(|e| BankError::InvalidAccessFile {
        path: path.to_path_buf(),
        reason: format!("io error: {e}"),
    })?;

    let parsed: AccessFileFormat =
        serde_json::from_str(&content).map_err(|e| BankError::InvalidAccessFile {
            path: path.to_path_buf(),
            reason: format!("json parse error: {e}"),
        })?;

    let accesses = match parsed {
        AccessFileFormat::Object { accesses } => accesses,
        AccessFileFormat::List(list) => list,
    };

    if accesses.is_empty() {
        return Err(BankError::EmptyAccessSet);
    }

    Ok(accesses)
}

pub fn parse_inline_coords(s: &str) -> Result<Vec<AccessEntry>, BankError> {
    let mut accesses = Vec::new();
    let entries: Vec<&str> = s
        .split([';', '\n'])
        .map(|entry| entry.trim())
        .filter(|entry| !entry.is_empty())
        .collect();

    for raw in entries {
        // Expected: lane:row,col or lane:row,col:group
        let parts: Vec<&str> = raw.split(':').map(|p| p.trim()).collect();
        if parts.len() < 2 {
            return Err(BankError::InvalidAccessPattern {
                reason: format!("invalid entry `{raw}`: expected `lane:row,col`"),
            });
        }

        let lane_str = parts.first().copied().unwrap_or("");
        let lane = lane_str
            .parse::<u32>()
            .map_err(|e| BankError::InvalidAccessPattern {
                reason: format!("invalid lane `{lane_str}`: {e}"),
            })?;

        let coord_str = parts.get(1).copied().unwrap_or("");
        let coord_parts: Vec<&str> = coord_str
            .split(['x', 'X', ','])
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();

        if coord_parts.len() != 2 {
            return Err(BankError::InvalidAccessPattern {
                reason: format!("invalid coordinate `{coord_str}`: expected `row,col`"),
            });
        }

        let r_str = coord_parts.first().copied().unwrap_or("");
        let c_str = coord_parts.get(1).copied().unwrap_or("");

        let row = r_str
            .parse::<u64>()
            .map_err(|e| BankError::InvalidAccessPattern {
                reason: format!("invalid row `{r_str}`: {e}"),
            })?;

        let col = c_str
            .parse::<u64>()
            .map_err(|e| BankError::InvalidAccessPattern {
                reason: format!("invalid col `{c_str}`: {e}"),
            })?;

        let group = if let Some(g_str) = parts.get(2) {
            let g = g_str
                .parse::<u32>()
                .map_err(|e| BankError::InvalidAccessPattern {
                    reason: format!("invalid group `{g_str}`: {e}"),
                })?;
            Some(g)
        } else {
            None
        };

        accesses.push(AccessEntry {
            lane,
            coord: [row, col],
            group,
        });
    }

    if accesses.is_empty() {
        return Err(BankError::EmptyAccessSet);
    }

    Ok(accesses)
}
