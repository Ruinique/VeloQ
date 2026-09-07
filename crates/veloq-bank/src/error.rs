use std::borrow::Cow;
use std::path::PathBuf;
use thiserror::Error;
use veloq_core::{ErrorCode, VeloqDiagnostic};

#[derive(Debug, Error)]
pub enum BankError {
    #[error("invalid swizzle `{raw}`: {reason}")]
    InvalidSwizzle { raw: String, reason: String },

    #[error("unsupported negative shift in swizzle `{raw}`: S must be positive (S >= B > 0) in v0")]
    UnsupportedNegativeShift { raw: String },

    #[error("invalid shape `{raw}`: {reason}")]
    InvalidShape { raw: String, reason: String },

    #[error("invalid stride `{raw}`: {reason}")]
    InvalidStride { raw: String, reason: String },

    #[error("coordinate [{row}, {col}] out of bounds for shape [{shape_rows}, {shape_cols}]")]
    InvalidCoordinate {
        row: u64,
        col: u64,
        shape_rows: u64,
        shape_cols: u64,
    },

    #[error("failed to read access pattern file `{path}`: {reason}")]
    InvalidAccessFile { path: PathBuf, reason: String },

    #[error("invalid access pattern specification: {reason}")]
    InvalidAccessPattern { reason: String },

    #[error("invalid bank model: {reason}")]
    InvalidBankModel { reason: String },

    #[error("access pattern is empty: provide at least one lane coordinate")]
    EmptyAccessSet,

    #[error(
        "search space too large ({total} candidates exceed maximum {max}): narrow search ranges"
    )]
    SearchSpaceTooLarge { total: usize, max: usize },

    #[error("invalid data type `{raw}`: expected f16, bf16, f32, f64, i8, i16, i32")]
    InvalidDataType { raw: String },

    #[error("output format `{format}` error: {reason}")]
    OutputError {
        format: &'static str,
        reason: String,
    },

    #[error("json serialization error: {reason}")]
    JsonError { reason: String },

    #[error("missing required argument: {arg}")]
    MissingArgument { arg: &'static str },
}

impl VeloqDiagnostic for BankError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::InvalidSwizzle { .. } => ErrorCode::new("bank.invalid-swizzle"),
            Self::UnsupportedNegativeShift { .. } => {
                ErrorCode::new("bank.unsupported-negative-shift")
            }
            Self::InvalidShape { .. } => ErrorCode::new("bank.invalid-shape"),
            Self::InvalidStride { .. } => ErrorCode::new("bank.invalid-stride"),
            Self::InvalidCoordinate { .. } => ErrorCode::new("bank.invalid-coordinate"),
            Self::InvalidAccessFile { .. } | Self::InvalidAccessPattern { .. } => {
                ErrorCode::new("bank.invalid-access-file")
            }
            Self::InvalidBankModel { .. } => ErrorCode::new("bank.invalid-bank-model"),
            Self::EmptyAccessSet => ErrorCode::new("bank.empty-access-set"),
            Self::SearchSpaceTooLarge { .. } => ErrorCode::new("bank.search-space-too-large"),
            Self::InvalidDataType { .. } => ErrorCode::new("bank.invalid-dtype"),
            Self::OutputError { .. } => ErrorCode::new("bank.output-error"),
            Self::JsonError { .. } => ErrorCode::new("bank.json-error"),
            Self::MissingArgument { .. } => ErrorCode::new("bank.missing-argument"),
        }
    }

    fn hint(&self) -> Option<Cow<'_, str>> {
        match self {
            Self::InvalidSwizzle { .. } => Some(Cow::Borrowed(
                "expected format `B,M,S` where B > 0, M >= 0, S >= B > 0 (e.g. `2,4,2`)",
            )),
            Self::UnsupportedNegativeShift { .. } => Some(Cow::Borrowed(
                "v0 only supports positive shifts S > 0 with S >= B",
            )),
            Self::InvalidShape { .. } => Some(Cow::Borrowed(
                "expected format `ROWSxCOLS` or `ROWS,COLS` (e.g. `8x64`)",
            )),
            Self::InvalidStride { .. } => Some(Cow::Borrowed(
                "expected format `ROW_STRIDE,COL_STRIDE` (e.g. `64,1`) or single element stride for explain",
            )),
            Self::InvalidCoordinate {
                shape_rows,
                shape_cols,
                ..
            } => Some(Cow::Owned(format!(
                "coordinates must satisfy row < {shape_rows} and col < {shape_cols}"
            ))),
            Self::InvalidAccessFile { .. } => Some(Cow::Borrowed(
                "ensure the JSON file exists and contains `{\"accesses\": [{\"lane\": 0, \"coord\": [0, 0]}]}`",
            )),
            Self::EmptyAccessSet => Some(Cow::Borrowed(
                "provide `--access <path>` with non-empty accesses or inline coordinates via `--coords '0:0,0;1:1,0'`",
            )),
            Self::SearchSpaceTooLarge { max, .. } => Some(Cow::Owned(format!(
                "narrow `--b`, `--m`, or `--s` ranges so candidate count is <= {max}"
            ))),
            Self::InvalidDataType { .. } => Some(Cow::Borrowed(
                "supported dtypes: f16, bf16, f32, f64, i8, i16, i32",
            )),
            _ => None,
        }
    }
}
