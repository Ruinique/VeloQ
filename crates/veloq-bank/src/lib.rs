pub mod access;
pub mod analyze;
pub mod bank;
pub mod cli;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod output;
pub mod swizzle;

pub use access::{AccessEntry, load_accesses_from_file, parse_inline_coords};
pub use analyze::{
    BankSummary, CompareRow, ExplainData, LaneTraceRow, SearchCandidateRow, SwizzleReasoning,
    compare_swizzles, explain_swizzle, search_swizzles, trace_accesses,
};
pub use bank::BankModel;
pub use cli::{cli, run};
pub use dtype::DataType;
pub use error::BankError;
pub use layout::Layout2D;
pub use output::BANK_SOURCE;
pub use swizzle::Swizzle;
