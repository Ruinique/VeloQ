use crate::analyze::{BankSummary, CompareRow, ExplainData, LaneTraceRow, SearchCandidateRow};
use crate::error::BankError;
use schemars::JsonSchema;
use serde::Serialize;
use veloq_core::{
    Envelope, EnvelopeError, OutputFormat, SourceRef,
    tabular::{TabularView, emit_csv, emit_table},
};

pub const BANK_SOURCE: SourceRef = SourceRef {
    kind: "bank",
    version: "v0",
};

#[derive(Debug, Serialize, JsonSchema)]
pub struct BankTraceResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<LaneTraceRow>,
    pub summary: BankSummary,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct BankCompareResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<CompareRow>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct BankSearchResponse {
    pub count: usize,
    pub total_matched: usize,
    pub rows: Vec<SearchCandidateRow>,
}

pub fn render_explain(data: ExplainData, fmt: OutputFormat) -> Result<(), BankError> {
    match fmt {
        OutputFormat::Json => {
            let env = Envelope::new(BANK_SOURCE, "bank.explain", None, None, None, data);
            let json = env.to_json_pretty().map_err(|e| BankError::JsonError {
                reason: e.to_string(),
            })?;
            println!("{json}");
            Ok(())
        }
        OutputFormat::Table => {
            let sw = &data.swizzle;
            println!("Swizzle<{},{},{}>\n", sw.b, sw.m, sw.s);

            let untouched_str = if sw.m == 0 {
                "none".to_string()
            } else if sw.m == 1 {
                "bits [0]".to_string()
            } else {
                format!("bits [{}:0]", sw.m - 1)
            };
            println!("untouched : {untouched_str}");

            let t_lo = sw.m;
            let t_hi = sw.m + sw.b - 1;
            let target_str = if sw.b == 1 {
                format!("bits [{t_lo}]")
            } else {
                format!("bits [{t_hi}:{t_lo}]")
            };
            println!("target    : {target_str}");

            let s_lo = sw.m + sw.s;
            let s_hi = sw.m + sw.s + sw.b - 1;
            let source_str = if sw.b == 1 {
                format!("bits [{s_lo}]")
            } else {
                format!("bits [{s_hi}:{s_lo}]")
            };
            println!("source    : {source_str}\n");

            println!("operation:\n{}\n", data.operation);

            if let Some(stride) = data.element_stride {
                if let Some(log2) = data.stride_log2 {
                    println!("stride = {stride} = 2^{log2}");
                } else {
                    println!("stride = {stride}");
                }
            }
            println!("source starts at bit {s_lo}");
            Ok(())
        }
        OutputFormat::Csv => {
            let mut view = TabularView::new(["field", "value"]);
            let sw = &data.swizzle;
            view.push_row(vec![
                "swizzle".to_string(),
                format!("{},{},{}", sw.b, sw.m, sw.s),
            ]);
            view.push_row(vec!["operation".to_string(), data.operation]);
            view.push_row(vec!["patterns".to_string(), data.patterns.to_string()]);
            if let Some(stride) = data.element_stride {
                view.push_row(vec!["element_stride".to_string(), stride.to_string()]);
            }
            if let Some(log2) = data.stride_log2 {
                view.push_row(vec!["stride_log2".to_string(), log2.to_string()]);
            }
            if let Some(sz) = data.element_size_bytes {
                view.push_row(vec!["element_size_bytes".to_string(), sz.to_string()]);
            }
            emit_csv(&view, "bank.explain", "static").map_err(|e| BankError::OutputError {
                format: "csv",
                reason: e.to_string(),
            })
        }
    }
}

pub fn render_trace(
    rows: Vec<LaneTraceRow>,
    summary: BankSummary,
    fmt: OutputFormat,
) -> Result<(), BankError> {
    match fmt {
        OutputFormat::Json => {
            let resp = BankTraceResponse {
                count: rows.len(),
                total_matched: rows.len(),
                rows,
                summary,
            };
            let env = Envelope::new(BANK_SOURCE, "bank.trace", None, None, None, resp);
            let json = env.to_json_pretty().map_err(|e| BankError::JsonError {
                reason: e.to_string(),
            })?;
            println!("{json}");
            Ok(())
        }
        OutputFormat::Csv | OutputFormat::Table => {
            let mut view = TabularView::new([
                "key",
                "lane",
                "coord",
                "logical_offset",
                "physical_offset",
                "byte_address",
                "bank",
                "broadcast_like",
            ]);
            view.push_meta("access_count", summary.access_count.to_string());
            view.push_meta("unique_banks", summary.unique_banks.to_string());
            view.push_meta(
                "conflicting_bank_count",
                summary.conflicting_bank_count.to_string(),
            );
            view.push_meta(
                "conflicting_access_count",
                summary.conflicting_access_count.to_string(),
            );
            view.push_meta(
                "max_conflict_degree",
                summary.max_conflict_degree.to_string(),
            );

            for r in &rows {
                let row_val = r.coord.first().copied().unwrap_or(0);
                let col_val = r.coord.get(1).copied().unwrap_or(0);
                view.push_row(vec![
                    r.key.clone(),
                    r.lane.to_string(),
                    format!("{row_val},{col_val}"),
                    r.logical_offset.to_string(),
                    r.physical_offset.to_string(),
                    r.byte_address.to_string(),
                    r.bank.to_string(),
                    r.broadcast_like.to_string(),
                ]);
            }

            if matches!(fmt, OutputFormat::Csv) {
                emit_csv(&view, "bank.trace", "static").map_err(|e| BankError::OutputError {
                    format: "csv",
                    reason: e.to_string(),
                })
            } else {
                emit_table(&view, "bank.trace", "static").map_err(|e| BankError::OutputError {
                    format: "table",
                    reason: e.to_string(),
                })
            }
        }
    }
}

pub fn render_compare(rows: Vec<CompareRow>, fmt: OutputFormat) -> Result<(), BankError> {
    match fmt {
        OutputFormat::Json => {
            let count = rows.len();
            let resp = BankCompareResponse {
                count,
                total_matched: count,
                rows,
            };
            let env = Envelope::new(BANK_SOURCE, "bank.compare", None, None, None, resp);
            let json = env.to_json_pretty().map_err(|e| BankError::JsonError {
                reason: e.to_string(),
            })?;
            println!("{json}");
            Ok(())
        }
        OutputFormat::Csv | OutputFormat::Table => {
            let mut view = TabularView::new([
                "key",
                "swizzle",
                "unique_banks",
                "conflicting_bank_count",
                "max_conflict_degree",
                "total_conflicting_accesses",
            ]);
            for r in &rows {
                let b = r.swizzle.first().copied().unwrap_or(0);
                let m = r.swizzle.get(1).copied().unwrap_or(0);
                let s = r.swizzle.get(2).copied().unwrap_or(0);
                view.push_row(vec![
                    r.key.clone(),
                    format!("{b},{m},{s}"),
                    r.unique_banks.to_string(),
                    r.conflicting_bank_count.to_string(),
                    r.max_conflict_degree.to_string(),
                    r.total_conflicting_accesses.to_string(),
                ]);
            }

            if matches!(fmt, OutputFormat::Csv) {
                emit_csv(&view, "bank.compare", "static").map_err(|e| BankError::OutputError {
                    format: "csv",
                    reason: e.to_string(),
                })
            } else {
                emit_table(&view, "bank.compare", "static").map_err(|e| BankError::OutputError {
                    format: "table",
                    reason: e.to_string(),
                })
            }
        }
    }
}

pub fn render_search(rows: Vec<SearchCandidateRow>, fmt: OutputFormat) -> Result<(), BankError> {
    match fmt {
        OutputFormat::Json => {
            let count = rows.len();
            let resp = BankSearchResponse {
                count,
                total_matched: count,
                rows,
            };
            let env = Envelope::new(BANK_SOURCE, "bank.search", None, None, None, resp);
            let json = env.to_json_pretty().map_err(|e| BankError::JsonError {
                reason: e.to_string(),
            })?;
            println!("{json}");
            Ok(())
        }
        OutputFormat::Csv | OutputFormat::Table => {
            let mut view = TabularView::new([
                "rank",
                "key",
                "swizzle",
                "max_conflict_degree",
                "total_conflicting_accesses",
                "unique_banks",
                "source_bits",
                "target_bits",
            ]);
            for r in &rows {
                let b = r.swizzle.first().copied().unwrap_or(0);
                let m = r.swizzle.get(1).copied().unwrap_or(0);
                let s = r.swizzle.get(2).copied().unwrap_or(0);
                view.push_row(vec![
                    r.rank.to_string(),
                    r.key.clone(),
                    format!("{b},{m},{s}"),
                    r.max_conflict_degree.to_string(),
                    r.total_conflicting_accesses.to_string(),
                    r.unique_banks.to_string(),
                    format!("{:?}", r.source_bits),
                    format!("{:?}", r.target_bits),
                ]);
            }

            if matches!(fmt, OutputFormat::Csv) {
                emit_csv(&view, "bank.search", "static").map_err(|e| BankError::OutputError {
                    format: "csv",
                    reason: e.to_string(),
                })
            } else {
                emit_table(&view, "bank.search", "static").map_err(|e| BankError::OutputError {
                    format: "table",
                    reason: e.to_string(),
                })
            }
        }
    }
}

pub fn emit_error(err: &BankError, command: Option<&str>, fmt: OutputFormat) {
    let env = EnvelopeError::from_diagnostic(
        Some(BANK_SOURCE),
        command.map(|c| format!("bank.{c}")),
        None,
        None,
        err,
    );
    if !matches!(fmt, OutputFormat::Json) {
        eprintln!("veloq bank: {err}");
    }
    if let Ok(s) = env.to_json_pretty() {
        println!("{s}");
    }
}
