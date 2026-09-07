use crate::access::{load_accesses_from_file, parse_inline_coords};
use crate::analyze::{compare_swizzles, explain_swizzle, search_swizzles, trace_accesses};
use crate::bank::BankModel;
use crate::dtype::DataType;
use crate::error::BankError;
use crate::layout::{Layout2D, parse_shape, parse_stride_1d, parse_stride_2d};
use crate::output::{emit_error, render_compare, render_explain, render_search, render_trace};
use crate::swizzle::Swizzle;
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::Path;
use std::str::FromStr;
use veloq_core::OutputFormat;

pub fn cli() -> Command {
    Command::new("bank")
        .about("Static bank-conflict and layout swizzle reasoning")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("explain")
                .about("Explain the bit-level operation of a CuTe-style swizzle")
                .arg(
                    Arg::new("swizzle")
                        .long("swizzle")
                        .required(true)
                        .value_name("B,M,S")
                        .help("Swizzle parameters B,M,S (e.g. 2,4,2)"),
                )
                .arg(
                    Arg::new("stride")
                        .long("stride")
                        .value_name("ELEMENTS")
                        .help("Optional element stride along swizzled dimension"),
                )
                .arg(
                    Arg::new("dtype")
                        .long("dtype")
                        .value_name("TYPE")
                        .help("Data type (e.g. f16, bf16, f32, f64, i8, i16, i32)"),
                ),
        )
        .subcommand(
            Command::new("trace")
                .about("Trace lane coordinates to physical offsets, byte addresses, and banks")
                .arg(
                    Arg::new("shape")
                        .long("shape")
                        .required(true)
                        .value_name("ROWSxCOLS")
                        .help("2D tensor shape (e.g. 8x64)"),
                )
                .arg(
                    Arg::new("stride")
                        .long("stride")
                        .required(true)
                        .value_name("ROW_STRIDE,COL_STRIDE")
                        .help("2D element stride (e.g. 64,1)"),
                )
                .arg(
                    Arg::new("dtype")
                        .long("dtype")
                        .required(true)
                        .value_name("TYPE")
                        .help("Data type (e.g. f16, bf16, f32, f64, i8, i16, i32)"),
                )
                .arg(
                    Arg::new("swizzle")
                        .long("swizzle")
                        .value_name("B,M,S")
                        .help("Optional CuTe swizzle B,M,S (identity if omitted)"),
                )
                .arg(
                    Arg::new("arch")
                        .long("arch")
                        .value_name("ARCH")
                        .default_value("nvidia")
                        .help("Target GPU architecture preset (nvidia, amd)"),
                )
                .arg(
                    Arg::new("banks")
                        .long("banks")
                        .value_name("N")
                        .value_parser(clap::value_parser!(u32))
                        .help("Override bank count (default: 32)"),
                )
                .arg(
                    Arg::new("bank-width")
                        .long("bank-width")
                        .value_name("BYTES")
                        .value_parser(clap::value_parser!(u32))
                        .help("Override bank width in bytes (default: 4)"),
                )
                .arg(
                    Arg::new("access")
                        .long("access")
                        .value_name("PATH")
                        .help("Path to access pattern JSON file"),
                )
                .arg(
                    Arg::new("coords")
                        .long("coords")
                        .value_name("LANE_COORDS")
                        .help("Inline coordinates (e.g. '0:0,0;1:1,0;2:2,0;3:3,0')"),
                ),
        )
        .subcommand(
            Command::new("compare")
                .about("Compare bank conflicts across candidate swizzles for an access pattern")
                .arg(
                    Arg::new("shape")
                        .long("shape")
                        .required(true)
                        .value_name("ROWSxCOLS")
                        .help("2D tensor shape (e.g. 8x64)"),
                )
                .arg(
                    Arg::new("stride")
                        .long("stride")
                        .required(true)
                        .value_name("ROW_STRIDE,COL_STRIDE")
                        .help("2D element stride (e.g. 64,1)"),
                )
                .arg(
                    Arg::new("dtype")
                        .long("dtype")
                        .required(true)
                        .value_name("TYPE")
                        .help("Data type (e.g. f16, bf16, f32, f64, i8, i16, i32)"),
                )
                .arg(
                    Arg::new("arch")
                        .long("arch")
                        .value_name("ARCH")
                        .default_value("nvidia")
                        .help("Target GPU architecture preset (nvidia, amd)"),
                )
                .arg(
                    Arg::new("banks")
                        .long("banks")
                        .value_name("N")
                        .value_parser(clap::value_parser!(u32))
                        .help("Override bank count (default: 32)"),
                )
                .arg(
                    Arg::new("bank-width")
                        .long("bank-width")
                        .value_name("BYTES")
                        .value_parser(clap::value_parser!(u32))
                        .help("Override bank width in bytes (default: 4)"),
                )
                .arg(
                    Arg::new("access")
                        .long("access")
                        .value_name("PATH")
                        .help("Path to access pattern JSON file"),
                )
                .arg(
                    Arg::new("coords")
                        .long("coords")
                        .value_name("LANE_COORDS")
                        .help("Inline coordinates (e.g. '0:0,0;1:1,0;2:2,0;3:3,0')"),
                )
                .arg(
                    Arg::new("swizzle")
                        .long("swizzle")
                        .required(true)
                        .action(ArgAction::Append)
                        .value_name("B,M,S")
                        .help("Candidate swizzle B,M,S (can specify multiple times)"),
                ),
        )
        .subcommand(
            Command::new("search")
                .about("Search parameter space for optimal CuTe swizzles")
                .arg(
                    Arg::new("shape")
                        .long("shape")
                        .required(true)
                        .value_name("ROWSxCOLS")
                        .help("2D tensor shape (e.g. 8x64)"),
                )
                .arg(
                    Arg::new("stride")
                        .long("stride")
                        .required(true)
                        .value_name("ROW_STRIDE,COL_STRIDE")
                        .help("2D element stride (e.g. 64,1)"),
                )
                .arg(
                    Arg::new("dtype")
                        .long("dtype")
                        .required(true)
                        .value_name("TYPE")
                        .help("Data type (e.g. f16, bf16, f32, f64, i8, i16, i32)"),
                )
                .arg(
                    Arg::new("arch")
                        .long("arch")
                        .value_name("ARCH")
                        .default_value("nvidia")
                        .help("Target GPU architecture preset (nvidia, amd)"),
                )
                .arg(
                    Arg::new("banks")
                        .long("banks")
                        .value_name("N")
                        .value_parser(clap::value_parser!(u32))
                        .help("Override bank count (default: 32)"),
                )
                .arg(
                    Arg::new("bank-width")
                        .long("bank-width")
                        .value_name("BYTES")
                        .value_parser(clap::value_parser!(u32))
                        .help("Override bank width in bytes (default: 4)"),
                )
                .arg(
                    Arg::new("access")
                        .long("access")
                        .value_name("PATH")
                        .help("Path to access pattern JSON file"),
                )
                .arg(
                    Arg::new("coords")
                        .long("coords")
                        .value_name("LANE_COORDS")
                        .help("Inline coordinates (e.g. '0:0,0;1:1,0;2:2,0;3:3,0')"),
                )
                .arg(
                    Arg::new("b")
                        .long("b")
                        .value_name("MIN:MAX")
                        .default_value("1:4")
                        .help("Search range for B (e.g. 1:4)"),
                )
                .arg(
                    Arg::new("m")
                        .long("m")
                        .value_name("MIN:MAX")
                        .default_value("2:5")
                        .help("Search range for M (e.g. 2:5)"),
                )
                .arg(
                    Arg::new("s")
                        .long("s")
                        .value_name("MIN:MAX")
                        .default_value("1:5")
                        .help("Search range for S (e.g. 1:5)"),
                )
                .arg(
                    Arg::new("limit")
                        .long("limit")
                        .value_name("N")
                        .default_value("20")
                        .value_parser(clap::value_parser!(usize))
                        .help("Maximum candidates to report (default: 20)"),
                ),
        )
}

pub fn run(matches: &ArgMatches, fmt: OutputFormat) -> Result<i32, BankError> {
    let (subcmd, sub_m) = match matches.subcommand() {
        Some((cmd, m)) => (cmd, m),
        None => return Ok(0),
    };

    let res = match subcmd {
        "explain" => run_explain(sub_m, fmt),
        "trace" => run_trace(sub_m, fmt),
        "compare" => run_compare(sub_m, fmt),
        "search" => run_search(sub_m, fmt),
        _ => Err(BankError::MissingArgument {
            arg: "unknown subcommand",
        }),
    };

    match res {
        Ok(()) => Ok(0),
        Err(err) => {
            emit_error(&err, Some(subcmd), fmt);
            Ok(1)
        }
    }
}

fn run_explain(matches: &ArgMatches, fmt: OutputFormat) -> Result<(), BankError> {
    let sw_raw = matches
        .get_one::<String>("swizzle")
        .ok_or(BankError::MissingArgument { arg: "swizzle" })?;
    let swizzle = Swizzle::from_str(sw_raw)?;

    let stride = matches
        .get_one::<String>("stride")
        .map(|s| parse_stride_1d(s))
        .transpose()?;

    let dtype = matches
        .get_one::<String>("dtype")
        .map(|d| DataType::from_str(d))
        .transpose()?;

    let data = explain_swizzle(swizzle, stride, dtype);
    render_explain(data, fmt)
}

fn load_access_pattern(matches: &ArgMatches) -> Result<Vec<crate::access::AccessEntry>, BankError> {
    if let Some(path_str) = matches.get_one::<String>("access") {
        let p = Path::new(path_str);
        load_accesses_from_file(p)
    } else if let Some(coords_str) = matches.get_one::<String>("coords") {
        parse_inline_coords(coords_str)
    } else {
        Err(BankError::EmptyAccessSet)
    }
}

fn load_bank_model(matches: &ArgMatches) -> Result<BankModel, BankError> {
    let arch = matches.get_one::<String>("arch").map(|s| s.as_str());
    let banks = matches.get_one::<u32>("banks").copied();
    let bank_width = matches.get_one::<u32>("bank-width").copied();
    BankModel::from_arch_or_override(arch, banks, bank_width)
}

fn run_trace(matches: &ArgMatches, fmt: OutputFormat) -> Result<(), BankError> {
    let shape_str = matches
        .get_one::<String>("shape")
        .ok_or(BankError::MissingArgument { arg: "shape" })?;
    let shape = parse_shape(shape_str)?;

    let stride_str = matches
        .get_one::<String>("stride")
        .ok_or(BankError::MissingArgument { arg: "stride" })?;
    let stride = parse_stride_2d(stride_str)?;
    let layout = Layout2D::new(shape, stride)?;

    let dtype_str = matches
        .get_one::<String>("dtype")
        .ok_or(BankError::MissingArgument { arg: "dtype" })?;
    let dtype = DataType::from_str(dtype_str)?;

    let swizzle = matches
        .get_one::<String>("swizzle")
        .map(|s| Swizzle::from_str(s))
        .transpose()?;

    let bank_model = load_bank_model(matches)?;
    let accesses = load_access_pattern(matches)?;

    let (rows, summary) = trace_accesses(&layout, dtype, swizzle, &bank_model, &accesses)?;
    render_trace(rows, summary, fmt)
}

fn run_compare(matches: &ArgMatches, fmt: OutputFormat) -> Result<(), BankError> {
    let shape_str = matches
        .get_one::<String>("shape")
        .ok_or(BankError::MissingArgument { arg: "shape" })?;
    let shape = parse_shape(shape_str)?;

    let stride_str = matches
        .get_one::<String>("stride")
        .ok_or(BankError::MissingArgument { arg: "stride" })?;
    let stride = parse_stride_2d(stride_str)?;
    let layout = Layout2D::new(shape, stride)?;

    let dtype_str = matches
        .get_one::<String>("dtype")
        .ok_or(BankError::MissingArgument { arg: "dtype" })?;
    let dtype = DataType::from_str(dtype_str)?;

    let swizzle_strs: Vec<&String> = matches
        .get_many::<String>("swizzle")
        .ok_or(BankError::MissingArgument { arg: "swizzle" })?
        .collect();

    let mut swizzles = Vec::with_capacity(swizzle_strs.len());
    for s in swizzle_strs {
        swizzles.push(Swizzle::from_str(s)?);
    }

    let bank_model = load_bank_model(matches)?;
    let accesses = load_access_pattern(matches)?;

    let rows = compare_swizzles(&layout, dtype, &bank_model, &accesses, &swizzles)?;
    render_compare(rows, fmt)
}

fn parse_range(s: &str, name: &'static str) -> Result<(u8, u8), BankError> {
    let cleaned = s.trim();
    if let Some((low_s, high_s)) = cleaned.split_once(':') {
        let low = low_s
            .trim()
            .parse::<u8>()
            .map_err(|e| BankError::InvalidSwizzle {
                raw: s.to_string(),
                reason: format!("invalid {name} range start `{low_s}`: {e}"),
            })?;
        let high = high_s
            .trim()
            .parse::<u8>()
            .map_err(|e| BankError::InvalidSwizzle {
                raw: s.to_string(),
                reason: format!("invalid {name} range end `{high_s}`: {e}"),
            })?;
        Ok((low, high))
    } else if let Some((low_s, high_s)) = cleaned.split_once("..=") {
        let low = low_s
            .trim()
            .parse::<u8>()
            .map_err(|e| BankError::InvalidSwizzle {
                raw: s.to_string(),
                reason: format!("invalid {name} range start `{low_s}`: {e}"),
            })?;
        let high = high_s
            .trim()
            .parse::<u8>()
            .map_err(|e| BankError::InvalidSwizzle {
                raw: s.to_string(),
                reason: format!("invalid {name} range end `{high_s}`: {e}"),
            })?;
        Ok((low, high))
    } else {
        let val = cleaned
            .parse::<u8>()
            .map_err(|e| BankError::InvalidSwizzle {
                raw: s.to_string(),
                reason: format!("invalid {name} value `{cleaned}`: {e}"),
            })?;
        Ok((val, val))
    }
}

fn run_search(matches: &ArgMatches, fmt: OutputFormat) -> Result<(), BankError> {
    let shape_str = matches
        .get_one::<String>("shape")
        .ok_or(BankError::MissingArgument { arg: "shape" })?;
    let shape = parse_shape(shape_str)?;

    let stride_str = matches
        .get_one::<String>("stride")
        .ok_or(BankError::MissingArgument { arg: "stride" })?;
    let stride = parse_stride_2d(stride_str)?;
    let layout = Layout2D::new(shape, stride)?;

    let dtype_str = matches
        .get_one::<String>("dtype")
        .ok_or(BankError::MissingArgument { arg: "dtype" })?;
    let dtype = DataType::from_str(dtype_str)?;

    let b_str = matches
        .get_one::<String>("b")
        .map(|s| s.as_str())
        .unwrap_or("1:4");
    let m_str = matches
        .get_one::<String>("m")
        .map(|s| s.as_str())
        .unwrap_or("2:5");
    let s_str = matches
        .get_one::<String>("s")
        .map(|s| s.as_str())
        .unwrap_or("1:5");

    let b_range = parse_range(b_str, "b")?;
    let m_range = parse_range(m_str, "m")?;
    let s_range = parse_range(s_str, "s")?;

    let limit = matches.get_one::<usize>("limit").copied().unwrap_or(20);

    let bank_model = load_bank_model(matches)?;
    let accesses = load_access_pattern(matches)?;

    let rows = search_swizzles(
        &layout,
        dtype,
        &bank_model,
        &accesses,
        b_range,
        m_range,
        s_range,
        limit,
    )?;
    render_search(rows, fmt)
}
