use crate::access::AccessEntry;
use crate::bank::BankModel;
use crate::dtype::DataType;
use crate::error::BankError;
use crate::layout::Layout2D;
use crate::swizzle::Swizzle;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LaneTraceRow {
    pub key: String,
    pub lane: u32,
    pub coord: [u64; 2],
    pub logical_offset: u64,
    pub physical_offset: u64,
    pub byte_address: u64,
    pub bank: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<u32>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub broadcast_like: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct BankSummary {
    pub access_count: usize,
    pub unique_banks: usize,
    pub conflicting_bank_count: usize,
    pub conflicting_access_count: usize,
    pub max_conflict_degree: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SwizzleReasoning {
    pub untouched_bits: Vec<u32>,
    pub target_bits: Vec<u32>,
    pub source_bits: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stride_boundary_bit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_aligned_to_stride_boundary: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CompareRow {
    pub key: String,
    pub swizzle: [u8; 3],
    pub unique_banks: usize,
    pub conflicting_bank_count: usize,
    pub max_conflict_degree: usize,
    pub total_conflicting_accesses: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchCandidateRow {
    pub key: String,
    pub rank: usize,
    pub swizzle: [u8; 3],
    pub max_conflict_degree: usize,
    pub total_conflicting_accesses: usize,
    pub unique_banks: usize,
    pub source_bits: Vec<u32>,
    pub target_bits: Vec<u32>,
    pub reasoning: SwizzleReasoning,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExplainData {
    pub swizzle: Swizzle,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub element_stride: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stride_log2: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub element_size_bytes: Option<u64>,
    pub untouched_bits: Vec<u32>,
    pub target_bits: Vec<u32>,
    pub source_bits: Vec<u32>,
    pub operation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_aligned_to_stride_boundary: Option<bool>,
    pub patterns: u64,
}

pub fn explain_swizzle(
    swizzle: Swizzle,
    element_stride: Option<u64>,
    dtype: Option<DataType>,
) -> ExplainData {
    let stride_log2 = element_stride.and_then(|s| {
        if s > 0 && s.is_power_of_two() {
            Some(s.trailing_zeros())
        } else {
            None
        }
    });

    let element_size_bytes = dtype.map(|d| d.size_bytes());

    let source_start = (swizzle.m as u32) + (swizzle.s as u32);
    let source_aligned = stride_log2.map(|log2| log2 == source_start);

    ExplainData {
        swizzle,
        element_stride,
        stride_log2,
        element_size_bytes,
        untouched_bits: swizzle.untouched_bits(),
        target_bits: swizzle.target_bits(),
        source_bits: swizzle.source_bits(),
        operation: swizzle.operation_str(),
        source_aligned_to_stride_boundary: source_aligned,
        patterns: swizzle.patterns(),
    }
}

pub fn build_reasoning(swizzle: &Swizzle, stride_row: u64) -> SwizzleReasoning {
    let stride_log2 = if stride_row > 0 && stride_row.is_power_of_two() {
        Some(stride_row.trailing_zeros())
    } else {
        None
    };

    let source_start = (swizzle.m as u32) + (swizzle.s as u32);
    let source_aligned = stride_log2.map(|log2| log2 == source_start);

    SwizzleReasoning {
        untouched_bits: swizzle.untouched_bits(),
        target_bits: swizzle.target_bits(),
        source_bits: swizzle.source_bits(),
        stride_boundary_bit: stride_log2,
        source_aligned_to_stride_boundary: source_aligned,
    }
}

pub fn trace_accesses(
    layout: &Layout2D,
    dtype: DataType,
    swizzle: Option<Swizzle>,
    bank_model: &BankModel,
    accesses: &[AccessEntry],
) -> Result<(Vec<LaneTraceRow>, BankSummary), BankError> {
    if accesses.is_empty() {
        return Err(BankError::EmptyAccessSet);
    }

    struct EvaluatedAccess {
        lane: u32,
        coord: [u64; 2],
        logical_offset: u64,
        physical_offset: u64,
        byte_address: u64,
        bank: u32,
        group: u32,
    }

    let mut evaluated = Vec::with_capacity(accesses.len());

    for entry in accesses {
        let logical_offset = layout.logical_offset(entry.coord)?;
        let physical_offset = match swizzle {
            Some(s) => s.apply(logical_offset)?,
            None => logical_offset,
        };
        let byte_address = physical_offset
            .checked_mul(dtype.size_bytes())
            .ok_or_else(|| BankError::InvalidCoordinate {
                row: entry.coord.first().copied().unwrap_or(0),
                col: entry.coord.get(1).copied().unwrap_or(0),
                shape_rows: layout.shape.first().copied().unwrap_or(0),
                shape_cols: layout.shape.get(1).copied().unwrap_or(0),
            })?;
        let bank = bank_model.bank_for_byte_address(byte_address);
        let group = entry.group.unwrap_or(0);

        evaluated.push(EvaluatedAccess {
            lane: entry.lane,
            coord: entry.coord,
            logical_offset,
            physical_offset,
            byte_address,
            bank,
            group,
        });
    }

    // Group-by (group, bank) to detect conflicts and broadcasts
    // Key: (group, bank), Value: Map of byte_address -> count
    let mut group_bank_addrs: BTreeMap<(u32, u32), BTreeMap<u64, usize>> = BTreeMap::new();
    let mut unique_banks_set: BTreeSet<u32> = BTreeSet::new();

    for ev in &evaluated {
        unique_banks_set.insert(ev.bank);
        let addr_map = group_bank_addrs.entry((ev.group, ev.bank)).or_default();
        let counter = addr_map.entry(ev.byte_address).or_insert(0);
        *counter = counter.saturating_add(1);
    }

    // Analyze conflict metrics
    let mut max_conflict_degree: usize = if evaluated.is_empty() { 0 } else { 1 };
    let mut conflicting_banks_set: BTreeSet<(u32, u32)> = BTreeSet::new();

    for (&(group, bank), addr_map) in &group_bank_addrs {
        let distinct_count = addr_map.len();
        if distinct_count > max_conflict_degree {
            max_conflict_degree = distinct_count;
        }
        if distinct_count >= 2 {
            conflicting_banks_set.insert((group, bank));
        }
    }

    let mut conflicting_access_count: usize = 0;
    let mut rows = Vec::with_capacity(evaluated.len());

    for ev in evaluated {
        let addr_map = group_bank_addrs.get(&(ev.group, ev.bank));
        let hits_for_this_addr = addr_map
            .and_then(|m| m.get(&ev.byte_address))
            .copied()
            .unwrap_or(0);

        // Broadcast: same bank + same byte address hit by multiple lanes
        let broadcast_like = hits_for_this_addr > 1;

        let is_conflict_bank = conflicting_banks_set.contains(&(ev.group, ev.bank));
        if is_conflict_bank {
            conflicting_access_count = conflicting_access_count.saturating_add(1);
        }

        rows.push(LaneTraceRow {
            key: format!("lane:{}", ev.lane),
            lane: ev.lane,
            coord: ev.coord,
            logical_offset: ev.logical_offset,
            physical_offset: ev.physical_offset,
            byte_address: ev.byte_address,
            bank: ev.bank,
            group: if ev.group == 0 && accesses.iter().all(|a| a.group.is_none()) {
                None
            } else {
                Some(ev.group)
            },
            broadcast_like,
        });
    }

    let summary = BankSummary {
        access_count: accesses.len(),
        unique_banks: unique_banks_set.len(),
        conflicting_bank_count: conflicting_banks_set.len(),
        conflicting_access_count,
        max_conflict_degree,
    };

    Ok((rows, summary))
}

pub fn compare_swizzles(
    layout: &Layout2D,
    dtype: DataType,
    bank_model: &BankModel,
    accesses: &[AccessEntry],
    swizzles: &[Swizzle],
) -> Result<Vec<CompareRow>, BankError> {
    if accesses.is_empty() {
        return Err(BankError::EmptyAccessSet);
    }

    let mut rows = Vec::with_capacity(swizzles.len());

    for swizzle in swizzles {
        let (_, summary) = trace_accesses(layout, dtype, Some(*swizzle), bank_model, accesses)?;
        rows.push(CompareRow {
            key: format!("swizzle:{},{},{}", swizzle.b, swizzle.m, swizzle.s),
            swizzle: swizzle.to_array(),
            unique_banks: summary.unique_banks,
            conflicting_bank_count: summary.conflicting_bank_count,
            max_conflict_degree: summary.max_conflict_degree,
            total_conflicting_accesses: summary.conflicting_access_count,
        });
    }

    // Sort criteria:
    // 1. max_conflict_degree asc
    // 2. total_conflicting_accesses asc
    // 3. unique_banks desc
    rows.sort_by(|a, b| {
        a.max_conflict_degree
            .cmp(&b.max_conflict_degree)
            .then_with(|| {
                a.total_conflicting_accesses
                    .cmp(&b.total_conflicting_accesses)
            })
            .then_with(|| b.unique_banks.cmp(&a.unique_banks))
            .then_with(|| a.swizzle.cmp(&b.swizzle))
    });

    Ok(rows)
}

#[expect(
    clippy::too_many_arguments,
    reason = "search arguments encapsulate ranges and model parameters"
)]
pub fn search_swizzles(
    layout: &Layout2D,
    dtype: DataType,
    bank_model: &BankModel,
    accesses: &[AccessEntry],
    b_range: (u8, u8),
    m_range: (u8, u8),
    s_range: (u8, u8),
    limit: usize,
) -> Result<Vec<SearchCandidateRow>, BankError> {
    if accesses.is_empty() {
        return Err(BankError::EmptyAccessSet);
    }

    let (b_min, b_max) = b_range;
    let (m_min, m_max) = m_range;
    let (s_min, s_max) = s_range;

    let b_count = if b_max >= b_min {
        (b_max - b_min + 1) as usize
    } else {
        0
    };
    let m_count = if m_max >= m_min {
        (m_max - m_min + 1) as usize
    } else {
        0
    };
    let s_count = if s_max >= s_min {
        (s_max - s_min + 1) as usize
    } else {
        0
    };

    let total_cartesian = b_count.saturating_mul(m_count).saturating_mul(s_count);
    const MAX_SEARCH_SPACE: usize = 100_000;
    if total_cartesian > MAX_SEARCH_SPACE {
        return Err(BankError::SearchSpaceTooLarge {
            total: total_cartesian,
            max: MAX_SEARCH_SPACE,
        });
    }

    let mut valid_candidates = Vec::new();
    for b in b_min..=b_max {
        if b == 0 {
            continue;
        }
        for m in m_min..=m_max {
            for s in s_min..=s_max {
                if s < b || s == 0 {
                    continue;
                }
                if let Ok(sw) = Swizzle::new(b, m, s) {
                    valid_candidates.push(sw);
                }
            }
        }
    }

    let row_stride = layout.stride.first().copied().unwrap_or(0);
    let mut results = Vec::with_capacity(valid_candidates.len());

    for sw in &valid_candidates {
        let (_, summary) = trace_accesses(layout, dtype, Some(*sw), bank_model, accesses)?;
        let reasoning = build_reasoning(sw, row_stride);

        results.push(SearchCandidateRow {
            key: format!("swizzle:{},{},{}", sw.b, sw.m, sw.s),
            rank: 0, // Filled after sorting
            swizzle: sw.to_array(),
            max_conflict_degree: summary.max_conflict_degree,
            total_conflicting_accesses: summary.conflicting_access_count,
            unique_banks: summary.unique_banks,
            source_bits: sw.source_bits(),
            target_bits: sw.target_bits(),
            reasoning,
        });
    }

    // Sort candidates
    results.sort_by(|a, b| {
        a.max_conflict_degree
            .cmp(&b.max_conflict_degree)
            .then_with(|| {
                a.total_conflicting_accesses
                    .cmp(&b.total_conflicting_accesses)
            })
            .then_with(|| b.unique_banks.cmp(&a.unique_banks))
            .then_with(|| a.swizzle.cmp(&b.swizzle))
    });

    // Assign rank and apply limit
    let take_count = results.len().min(limit);
    let mut final_rows = Vec::with_capacity(take_count);
    for (idx, mut row) in results.into_iter().enumerate() {
        if idx >= limit {
            break;
        }
        row.rank = idx.saturating_add(1);
        final_rows.push(row);
    }

    Ok(final_rows)
}
