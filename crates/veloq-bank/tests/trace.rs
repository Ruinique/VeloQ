use veloq_bank::{AccessEntry, BankModel, DataType, Layout2D, Swizzle, trace_accesses};

#[test]
fn test_case_a_no_swizzle_all_conflict() -> Result<(), anyhow::Error> {
    let layout = Layout2D::new([8, 64], [64, 1])?;
    let dtype = DataType::F16;
    let bank_model = BankModel::nvidia();
    let accesses = vec![
        AccessEntry {
            lane: 0,
            coord: [0, 0],
            group: None,
        },
        AccessEntry {
            lane: 1,
            coord: [1, 0],
            group: None,
        },
        AccessEntry {
            lane: 2,
            coord: [2, 0],
            group: None,
        },
        AccessEntry {
            lane: 3,
            coord: [3, 0],
            group: None,
        },
    ];

    let (rows, summary) = trace_accesses(&layout, dtype, None, &bank_model, &accesses)?;

    assert_eq!(rows.len(), 4);
    assert_eq!(rows.first().map(|r| r.logical_offset), Some(0));
    assert_eq!(rows.get(1).map(|r| r.logical_offset), Some(64));
    assert_eq!(rows.get(2).map(|r| r.logical_offset), Some(128));
    assert_eq!(rows.get(3).map(|r| r.logical_offset), Some(192));

    for row in &rows {
        assert_eq!(row.bank, 0);
        assert!(!row.broadcast_like);
    }

    assert_eq!(summary.access_count, 4);
    assert_eq!(summary.unique_banks, 1);
    assert_eq!(summary.conflicting_bank_count, 1);
    assert_eq!(summary.conflicting_access_count, 4);
    assert_eq!(summary.max_conflict_degree, 4);

    Ok(())
}

#[test]
fn test_case_b_swizzle_2_4_2_resolves_conflict() -> Result<(), anyhow::Error> {
    let layout = Layout2D::new([8, 64], [64, 1])?;
    let dtype = DataType::F16;
    let bank_model = BankModel::nvidia();
    let swizzle = Swizzle::new(2, 4, 2)?;
    let accesses = vec![
        AccessEntry {
            lane: 0,
            coord: [0, 0],
            group: None,
        },
        AccessEntry {
            lane: 1,
            coord: [1, 0],
            group: None,
        },
        AccessEntry {
            lane: 2,
            coord: [2, 0],
            group: None,
        },
        AccessEntry {
            lane: 3,
            coord: [3, 0],
            group: None,
        },
    ];

    let (rows, summary) = trace_accesses(&layout, dtype, Some(swizzle), &bank_model, &accesses)?;

    assert_eq!(rows.len(), 4);
    assert_eq!(rows.first().map(|r| r.physical_offset), Some(0));
    assert_eq!(rows.get(1).map(|r| r.physical_offset), Some(80));
    assert_eq!(rows.get(2).map(|r| r.physical_offset), Some(160));
    assert_eq!(rows.get(3).map(|r| r.physical_offset), Some(240));

    assert_eq!(rows.first().map(|r| r.bank), Some(0));
    assert_eq!(rows.get(1).map(|r| r.bank), Some(8));
    assert_eq!(rows.get(2).map(|r| r.bank), Some(16));
    assert_eq!(rows.get(3).map(|r| r.bank), Some(24));

    assert_eq!(summary.access_count, 4);
    assert_eq!(summary.unique_banks, 4);
    assert_eq!(summary.conflicting_bank_count, 0);
    assert_eq!(summary.conflicting_access_count, 0);
    assert_eq!(summary.max_conflict_degree, 1);

    Ok(())
}

#[test]
fn test_case_c_swizzle_induces_conflict_from_unconflicted() -> Result<(), anyhow::Error> {
    let layout = Layout2D::new([8, 64], [64, 1])?;
    let dtype = DataType::F16;
    let bank_model = BankModel::nvidia();
    let swizzle = Swizzle::new(2, 4, 2)?;
    let accesses = vec![
        AccessEntry {
            lane: 0,
            coord: [0, 0],
            group: None,
        },
        AccessEntry {
            lane: 1,
            coord: [1, 16],
            group: None,
        },
        AccessEntry {
            lane: 2,
            coord: [2, 32],
            group: None,
        },
        AccessEntry {
            lane: 3,
            coord: [3, 48],
            group: None,
        },
    ];

    // Without swizzle: offsets 0, 80, 160, 240 -> banks 0, 8, 16, 24 (no conflict)
    let (orig_rows, orig_summary) = trace_accesses(&layout, dtype, None, &bank_model, &accesses)?;
    assert_eq!(orig_rows.first().map(|r| r.logical_offset), Some(0));
    assert_eq!(orig_rows.get(1).map(|r| r.logical_offset), Some(80));
    assert_eq!(orig_rows.get(2).map(|r| r.logical_offset), Some(160));
    assert_eq!(orig_rows.get(3).map(|r| r.logical_offset), Some(240));
    assert_eq!(orig_summary.max_conflict_degree, 1);

    // With swizzle: offsets 0, 64, 128, 192 -> all bank 0 (conflict degree 4!)
    let (sw_rows, sw_summary) =
        trace_accesses(&layout, dtype, Some(swizzle), &bank_model, &accesses)?;
    assert_eq!(sw_rows.first().map(|r| r.physical_offset), Some(0));
    assert_eq!(sw_rows.get(1).map(|r| r.physical_offset), Some(64));
    assert_eq!(sw_rows.get(2).map(|r| r.physical_offset), Some(128));
    assert_eq!(sw_rows.get(3).map(|r| r.physical_offset), Some(192));

    for row in &sw_rows {
        assert_eq!(row.bank, 0);
    }
    assert_eq!(sw_summary.max_conflict_degree, 4);
    assert_eq!(sw_summary.conflicting_bank_count, 1);
    assert_eq!(sw_summary.conflicting_access_count, 4);

    Ok(())
}

#[test]
fn test_case_d_broadcast_not_counted_as_conflict() -> Result<(), anyhow::Error> {
    let layout = Layout2D::new([8, 64], [64, 1])?;
    let dtype = DataType::F16;
    let bank_model = BankModel::nvidia();
    // Two lanes accessing the exact same coordinate (0, 0)
    let accesses = vec![
        AccessEntry {
            lane: 0,
            coord: [0, 0],
            group: None,
        },
        AccessEntry {
            lane: 1,
            coord: [0, 0],
            group: None,
        },
    ];

    let (rows, summary) = trace_accesses(&layout, dtype, None, &bank_model, &accesses)?;

    assert_eq!(rows.len(), 2);
    for row in &rows {
        assert_eq!(row.bank, 0);
        assert_eq!(row.byte_address, 0);
        assert!(row.broadcast_like);
    }

    assert_eq!(summary.access_count, 2);
    assert_eq!(summary.unique_banks, 1);
    assert_eq!(summary.conflicting_bank_count, 0);
    assert_eq!(summary.conflicting_access_count, 0);
    assert_eq!(summary.max_conflict_degree, 1);

    Ok(())
}
