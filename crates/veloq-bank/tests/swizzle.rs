use std::str::FromStr;
use veloq_bank::{BankError, Swizzle};

#[test]
fn test_swizzle_2_4_2_semantics() -> Result<(), anyhow::Error> {
    let sw = Swizzle::new(2, 4, 2)?;
    assert_eq!(sw.target_bits(), vec![4, 5]);
    assert_eq!(sw.source_bits(), vec![6, 7]);
    assert_eq!(sw.untouched_bits(), vec![0, 1, 2, 3]);
    assert_eq!(sw.operation_str(), "bits[5:4] ^= bits[7:6]");
    assert_eq!(sw.patterns(), 4);

    assert_eq!(sw.apply(0)?, 0);
    assert_eq!(sw.apply(64)?, 80);
    assert_eq!(sw.apply(128)?, 160);
    assert_eq!(sw.apply(192)?, 240);

    Ok(())
}

#[test]
fn test_swizzle_parsing_and_invariants() -> Result<(), anyhow::Error> {
    assert!(Swizzle::from_str("2,4,2").is_ok());
    assert!(Swizzle::from_str("<2,4,2>").is_ok());

    // S < B should fail
    match Swizzle::from_str("3,2,1") {
        Err(BankError::InvalidSwizzle { .. }) => {}
        other => anyhow::bail!("expected InvalidSwizzle, got {other:?}"),
    }

    // B == 0 should fail
    match Swizzle::from_str("0,2,2") {
        Err(BankError::InvalidSwizzle { .. }) => {}
        other => anyhow::bail!("expected InvalidSwizzle, got {other:?}"),
    }

    // Negative S should fail with UnsupportedNegativeShift
    match Swizzle::from_str("2,4,-2") {
        Err(BankError::UnsupportedNegativeShift { .. }) => {}
        other => anyhow::bail!("expected UnsupportedNegativeShift, got {other:?}"),
    }

    Ok(())
}
