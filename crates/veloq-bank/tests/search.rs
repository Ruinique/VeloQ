use veloq_bank::{AccessEntry, BankModel, DataType, Layout2D, search_swizzles};

#[test]
fn test_search_swizzles() -> Result<(), anyhow::Error> {
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

    let rows = search_swizzles(
        &layout,
        dtype,
        &bank_model,
        &accesses,
        (1, 4),
        (2, 5),
        (1, 5),
        10,
    )?;

    assert!(!rows.is_empty());
    assert!(rows.len() <= 10);

    let top = rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("no search rows returned"))?;
    assert_eq!(top.rank, 1);
    assert_eq!(top.max_conflict_degree, 1);
    assert_eq!(top.total_conflicting_accesses, 0);

    // Verify reasoning
    assert_eq!(top.reasoning.stride_boundary_bit, Some(6));

    Ok(())
}
