use veloq_bank::{AccessEntry, BankModel, DataType, Layout2D, Swizzle, compare_swizzles};

#[test]
fn test_compare_swizzles_ranking() -> Result<(), anyhow::Error> {
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

    let sw_333 = Swizzle::new(3, 3, 3)?;
    let sw_242 = Swizzle::new(2, 4, 2)?;

    let rows = compare_swizzles(&layout, dtype, &bank_model, &accesses, &[sw_333, sw_242])?;
    assert_eq!(rows.len(), 2);

    // swizzle 2,4,2 should be top because it has max_conflict_degree = 1
    let best = rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("no compare rows returned"))?;
    assert_eq!(best.swizzle, [2, 4, 2]);
    assert_eq!(best.max_conflict_degree, 1);
    assert_eq!(best.total_conflicting_accesses, 0);
    assert_eq!(best.unique_banks, 4);

    Ok(())
}
