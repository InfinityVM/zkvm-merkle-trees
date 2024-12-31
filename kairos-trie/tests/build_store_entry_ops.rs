use proptest::prelude::*;

use kairos_trie::KeyHash;
use trie_test_utils::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    #[test]
    fn prop_end_to_end_entry_ops_big(
        batches in arb_batches(1..50_000usize, 1..100_000usize, 1000, 10_000)) {
        end_to_end_entry_ops(batches);
    }

    #[test]
    fn prop_end_to_end_entry_ops(
        batches in arb_batches(1..5000usize, 1..100_000usize, 1000, 10_000)) {
        end_to_end_entry_ops(batches);
    }

    #[test]
    fn prop_end_to_end_entry_ops_small(
        batches in arb_batches(1..500usize, 1..100_000usize, 10_000, 10_000)) {
        end_to_end_entry_ops(batches);
    }

    #[test]
    fn prop_end_to_end_entry_ops_tiny(
        batches in arb_batches(1..50usize, 1..100_000usize, 10_000, 10_000)) {
        end_to_end_entry_ops(batches);
    }
}

#[test]
fn leaf_prefix_insert() {
    let failed = vec![vec![
        Operation::Insert(KeyHash([1, 0, 0, 0, 0, 0, 0, 0]), 0u64.to_le_bytes()),
        Operation::Insert(KeyHash([1, 0, 0, 0, 0, 0, 0, 1]), 0u64.to_le_bytes()),
    ]];

    end_to_end_entry_ops(failed);
}

#[test]
fn leaf_prefix_insert_at_root() {
    let failed = vec![vec![
        Operation::Insert(KeyHash([1, 0, 0, 0, 0, 0, 0, 0]), 0u64.to_le_bytes()),
        Operation::Insert(KeyHash([1, 0, 0, 0, 0, 0, 0, 1]), 0u64.to_le_bytes()),
        Operation::Insert(KeyHash([0, 0, 0, 0, 0, 0, 0, 0]), 0u64.to_le_bytes()),
    ]];

    end_to_end_entry_ops(failed);
}

#[test]
fn leaf_prefix_insert_entry_insert_at_root() {
    let failed = vec![vec![
        Operation::Insert(KeyHash([1, 0, 0, 0, 0, 0, 0, 0]), 0u64.to_le_bytes()),
        Operation::Insert(KeyHash([1, 0, 0, 0, 0, 0, 0, 1]), 0u64.to_le_bytes()),
        Operation::EntryInsert(KeyHash([0, 0, 0, 0, 0, 0, 0, 0]), 0u64.to_le_bytes()),
    ]];

    end_to_end_entry_ops(failed);
}

#[test]
fn leaf_prefix_entry_insert() {
    let failed = vec![vec![
        Operation::EntryInsert(KeyHash([1, 0, 0, 0, 0, 0, 0, 0]), 0u64.to_le_bytes()),
        Operation::EntryInsert(KeyHash([1, 0, 0, 0, 0, 0, 0, 1]), 0u64.to_le_bytes()),
    ]];

    end_to_end_entry_ops(failed);
}

#[test]
fn leaf_prefix_entry_or_insert() {
    let failed = vec![vec![
        Operation::EntryOrInsert(KeyHash([1, 0, 0, 0, 0, 0, 0, 0]), 0u64.to_le_bytes()),
        Operation::EntryOrInsert(KeyHash([1, 0, 0, 0, 0, 0, 0, 1]), 0u64.to_le_bytes()),
    ]];

    end_to_end_entry_ops(failed);
}

#[test]
fn insert_remove_with_gets() {
    let operations = vec![vec![
        Operation::Insert(KeyHash([1, 0, 0, 0, 0, 0, 0, 0]), 0u64.to_le_bytes()),
        Operation::Insert(KeyHash([2, 0, 0, 0, 0, 0, 0, 0]), 1u64.to_le_bytes()),
        Operation::Insert(KeyHash([3, 0, 0, 0, 0, 0, 0, 0]), 2u64.to_le_bytes()),
        Operation::Get(KeyHash([1, 0, 0, 0, 0, 0, 0, 0])),
        Operation::Remove(KeyHash([2, 0, 0, 0, 0, 0, 0, 0])),
        Operation::Get(KeyHash([2, 0, 0, 0, 0, 0, 0, 0])),
        Operation::Insert(KeyHash([4, 0, 0, 0, 0, 0, 0, 0]), 3u64.to_le_bytes()),
        Operation::Insert(KeyHash([5, 0, 0, 0, 0, 0, 0, 0]), 4u64.to_le_bytes()),
        Operation::Get(KeyHash([3, 0, 0, 0, 0, 0, 0, 0])),
        Operation::Remove(KeyHash([1, 0, 0, 0, 0, 0, 0, 0])),
        Operation::Get(KeyHash([1, 0, 0, 0, 0, 0, 0, 0])),
        Operation::Insert(KeyHash([6, 0, 0, 0, 0, 0, 0, 0]), 5u64.to_le_bytes()),
        Operation::Insert(KeyHash([7, 0, 0, 0, 0, 0, 0, 0]), 6u64.to_le_bytes()),
        Operation::Get(KeyHash([4, 0, 0, 0, 0, 0, 0, 0])),
        Operation::Remove(KeyHash([5, 0, 0, 0, 0, 0, 0, 0])),
        Operation::Get(KeyHash([5, 0, 0, 0, 0, 0, 0, 0])),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn insert_remove_multiple_batches() {
    let operations = vec![
        vec![
            Operation::Insert(KeyHash([1, 0, 0, 0, 0, 0, 0, 0]), 0u64.to_le_bytes()),
            Operation::Insert(KeyHash([2, 0, 0, 0, 0, 0, 0, 0]), 1u64.to_le_bytes()),
            Operation::Remove(KeyHash([1, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([3, 0, 0, 0, 0, 0, 0, 0]), 2u64.to_le_bytes()),
            Operation::Insert(KeyHash([4, 0, 0, 0, 0, 0, 0, 0]), 3u64.to_le_bytes()),
            Operation::Remove(KeyHash([2, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([5, 0, 0, 0, 0, 0, 0, 0]), 4u64.to_le_bytes()),
            Operation::Insert(KeyHash([6, 0, 0, 0, 0, 0, 0, 0]), 5u64.to_le_bytes()),
            Operation::Remove(KeyHash([3, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([7, 0, 0, 0, 0, 0, 0, 0]), 6u64.to_le_bytes()),
        ],
        vec![
            Operation::Insert(KeyHash([8, 0, 0, 0, 0, 0, 0, 0]), 7u64.to_le_bytes()),
            Operation::Insert(KeyHash([9, 0, 0, 0, 0, 0, 0, 0]), 8u64.to_le_bytes()),
            Operation::Remove(KeyHash([4, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([10, 0, 0, 0, 0, 0, 0, 0]), 9u64.to_le_bytes()),
            Operation::Insert(KeyHash([11, 0, 0, 0, 0, 0, 0, 0]), 10u64.to_le_bytes()),
            Operation::Remove(KeyHash([5, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([12, 0, 0, 0, 0, 0, 0, 0]), 11u64.to_le_bytes()),
            Operation::Insert(KeyHash([13, 0, 0, 0, 0, 0, 0, 0]), 12u64.to_le_bytes()),
            Operation::Remove(KeyHash([6, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([14, 0, 0, 0, 0, 0, 0, 0]), 13u64.to_le_bytes()),
        ],
        vec![
            Operation::Insert(KeyHash([15, 0, 0, 0, 0, 0, 0, 0]), 14u64.to_le_bytes()),
            Operation::Insert(KeyHash([16, 0, 0, 0, 0, 0, 0, 0]), 15u64.to_le_bytes()),
            Operation::Remove(KeyHash([7, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([17, 0, 0, 0, 0, 0, 0, 0]), 16u64.to_le_bytes()),
            Operation::Insert(KeyHash([18, 0, 0, 0, 0, 0, 0, 0]), 17u64.to_le_bytes()),
            Operation::Remove(KeyHash([8, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([19, 0, 0, 0, 0, 0, 0, 0]), 18u64.to_le_bytes()),
            Operation::Insert(KeyHash([20, 0, 0, 0, 0, 0, 0, 0]), 19u64.to_le_bytes()),
            Operation::Remove(KeyHash([9, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([21, 0, 0, 0, 0, 0, 0, 0]), 20u64.to_le_bytes()),
        ],
        vec![
            Operation::Insert(KeyHash([22, 0, 0, 0, 0, 0, 0, 0]), 21u64.to_le_bytes()),
            Operation::Insert(KeyHash([23, 0, 0, 0, 0, 0, 0, 0]), 22u64.to_le_bytes()),
            Operation::Remove(KeyHash([10, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([24, 0, 0, 0, 0, 0, 0, 0]), 23u64.to_le_bytes()),
            Operation::Insert(KeyHash([25, 0, 0, 0, 0, 0, 0, 0]), 24u64.to_le_bytes()),
            Operation::Remove(KeyHash([11, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([26, 0, 0, 0, 0, 0, 0, 0]), 25u64.to_le_bytes()),
            Operation::Insert(KeyHash([27, 0, 0, 0, 0, 0, 0, 0]), 26u64.to_le_bytes()),
            Operation::Remove(KeyHash([12, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([28, 0, 0, 0, 0, 0, 0, 0]), 27u64.to_le_bytes()),
        ],
        vec![
            Operation::Insert(KeyHash([29, 0, 0, 0, 0, 0, 0, 0]), 28u64.to_le_bytes()),
            Operation::Insert(KeyHash([30, 0, 0, 0, 0, 0, 0, 0]), 29u64.to_le_bytes()),
            Operation::Remove(KeyHash([13, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([31, 0, 0, 0, 0, 0, 0, 0]), 30u64.to_le_bytes()),
            Operation::Insert(KeyHash([32, 0, 0, 0, 0, 0, 0, 0]), 31u64.to_le_bytes()),
            Operation::Remove(KeyHash([14, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([33, 0, 0, 0, 0, 0, 0, 0]), 32u64.to_le_bytes()),
            Operation::Insert(KeyHash([34, 0, 0, 0, 0, 0, 0, 0]), 33u64.to_le_bytes()),
            Operation::Remove(KeyHash([15, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Insert(KeyHash([35, 0, 0, 0, 0, 0, 0, 0]), 34u64.to_le_bytes()),
        ],
    ];

    end_to_end_entry_ops(operations);
}
#[test]
fn mixed_operations_multiple_batches() {
    let operations = vec![
        vec![
            Operation::Insert(KeyHash([1, 0, 0, 0, 0, 0, 0, 0]), 0u64.to_le_bytes()),
            Operation::Insert(KeyHash([2, 0, 0, 0, 0, 0, 0, 0]), 1u64.to_le_bytes()),
            Operation::Get(KeyHash([1, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Remove(KeyHash([2, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Get(KeyHash([2, 0, 0, 0, 0, 0, 0, 0])),
            Operation::EntryInsert(KeyHash([3, 0, 0, 0, 0, 0, 0, 0]), 2u64.to_le_bytes()),
            Operation::EntryGet(KeyHash([3, 0, 0, 0, 0, 0, 0, 0])),
        ],
        vec![
            Operation::Insert(KeyHash([4, 0, 0, 0, 0, 0, 0, 0]), 3u64.to_le_bytes()),
            Operation::Insert(KeyHash([5, 0, 0, 0, 0, 0, 0, 0]), 4u64.to_le_bytes()),
            Operation::Get(KeyHash([4, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Remove(KeyHash([5, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Get(KeyHash([5, 0, 0, 0, 0, 0, 0, 0])),
            Operation::EntryAndModifyOrInsert(
                KeyHash([6, 0, 0, 0, 0, 0, 0, 0]),
                5u64.to_le_bytes(),
            ),
            Operation::EntryOrInsert(KeyHash([6, 0, 0, 0, 0, 0, 0, 0]), 6u64.to_le_bytes()),
        ],
        vec![
            Operation::Insert(KeyHash([7, 0, 0, 0, 0, 0, 0, 0]), 7u64.to_le_bytes()),
            Operation::Insert(KeyHash([8, 0, 0, 0, 0, 0, 0, 0]), 8u64.to_le_bytes()),
            Operation::Get(KeyHash([7, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Remove(KeyHash([8, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Get(KeyHash([8, 0, 0, 0, 0, 0, 0, 0])),
            Operation::EntryInsert(KeyHash([9, 0, 0, 0, 0, 0, 0, 0]), 9u64.to_le_bytes()),
            Operation::EntryGet(KeyHash([9, 0, 0, 0, 0, 0, 0, 0])),
        ],
        vec![
            Operation::Insert(KeyHash([10, 0, 0, 0, 0, 0, 0, 0]), 10u64.to_le_bytes()),
            Operation::Insert(KeyHash([11, 0, 0, 0, 0, 0, 0, 0]), 11u64.to_le_bytes()),
            Operation::Get(KeyHash([10, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Remove(KeyHash([11, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Get(KeyHash([11, 0, 0, 0, 0, 0, 0, 0])),
            Operation::EntryAndModifyOrInsert(
                KeyHash([12, 0, 0, 0, 0, 0, 0, 0]),
                12u64.to_le_bytes(),
            ),
            Operation::EntryOrInsert(KeyHash([12, 0, 0, 0, 0, 0, 0, 0]), 13u64.to_le_bytes()),
        ],
        vec![
            Operation::Insert(KeyHash([13, 0, 0, 0, 0, 0, 0, 0]), 14u64.to_le_bytes()),
            Operation::Insert(KeyHash([14, 0, 0, 0, 0, 0, 0, 0]), 15u64.to_le_bytes()),
            Operation::Get(KeyHash([13, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Remove(KeyHash([14, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Get(KeyHash([14, 0, 0, 0, 0, 0, 0, 0])),
            Operation::EntryInsert(KeyHash([15, 0, 0, 0, 0, 0, 0, 0]), 16u64.to_le_bytes()),
            Operation::EntryGet(KeyHash([15, 0, 0, 0, 0, 0, 0, 0])),
        ],
        vec![
            Operation::Insert(KeyHash([16, 0, 0, 0, 0, 0, 0, 0]), 17u64.to_le_bytes()),
            Operation::Insert(KeyHash([17, 0, 0, 0, 0, 0, 0, 0]), 18u64.to_le_bytes()),
            Operation::Get(KeyHash([16, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Remove(KeyHash([17, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Get(KeyHash([17, 0, 0, 0, 0, 0, 0, 0])),
            Operation::EntryAndModifyOrInsert(
                KeyHash([18, 0, 0, 0, 0, 0, 0, 0]),
                19u64.to_le_bytes(),
            ),
            Operation::EntryOrInsert(KeyHash([18, 0, 0, 0, 0, 0, 0, 0]), 20u64.to_le_bytes()),
        ],
        vec![
            Operation::Insert(KeyHash([19, 0, 0, 0, 0, 0, 0, 0]), 21u64.to_le_bytes()),
            Operation::Insert(KeyHash([20, 0, 0, 0, 0, 0, 0, 0]), 22u64.to_le_bytes()),
            Operation::Get(KeyHash([19, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Remove(KeyHash([20, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Get(KeyHash([20, 0, 0, 0, 0, 0, 0, 0])),
            Operation::EntryInsert(KeyHash([21, 0, 0, 0, 0, 0, 0, 0]), 23u64.to_le_bytes()),
            Operation::EntryGet(KeyHash([21, 0, 0, 0, 0, 0, 0, 0])),
        ],
        vec![
            Operation::Insert(KeyHash([22, 0, 0, 0, 0, 0, 0, 0]), 24u64.to_le_bytes()),
            Operation::Insert(KeyHash([23, 0, 0, 0, 0, 0, 0, 0]), 25u64.to_le_bytes()),
            Operation::Get(KeyHash([22, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Remove(KeyHash([23, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Get(KeyHash([23, 0, 0, 0, 0, 0, 0, 0])),
            Operation::EntryAndModifyOrInsert(
                KeyHash([24, 0, 0, 0, 0, 0, 0, 0]),
                26u64.to_le_bytes(),
            ),
            Operation::EntryOrInsert(KeyHash([24, 0, 0, 0, 0, 0, 0, 0]), 27u64.to_le_bytes()),
        ],
        vec![
            Operation::Insert(KeyHash([25, 0, 0, 0, 0, 0, 0, 0]), 28u64.to_le_bytes()),
            Operation::Insert(KeyHash([26, 0, 0, 0, 0, 0, 0, 0]), 29u64.to_le_bytes()),
            Operation::Get(KeyHash([25, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Remove(KeyHash([26, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Get(KeyHash([26, 0, 0, 0, 0, 0, 0, 0])),
            Operation::EntryInsert(KeyHash([27, 0, 0, 0, 0, 0, 0, 0]), 30u64.to_le_bytes()),
            Operation::EntryGet(KeyHash([27, 0, 0, 0, 0, 0, 0, 0])),
        ],
        vec![
            Operation::Insert(KeyHash([28, 0, 0, 0, 0, 0, 0, 0]), 31u64.to_le_bytes()),
            Operation::Insert(KeyHash([29, 0, 0, 0, 0, 0, 0, 0]), 32u64.to_le_bytes()),
            Operation::Get(KeyHash([28, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Remove(KeyHash([29, 0, 0, 0, 0, 0, 0, 0])),
            Operation::Get(KeyHash([29, 0, 0, 0, 0, 0, 0, 0])),
            Operation::EntryAndModifyOrInsert(
                KeyHash([30, 0, 0, 0, 0, 0, 0, 0]),
                33u64.to_le_bytes(),
            ),
            Operation::EntryOrInsert(KeyHash([30, 0, 0, 0, 0, 0, 0, 0]), 34u64.to_le_bytes()),
        ],
    ];

    end_to_end_entry_ops(operations);
}

#[test]
fn matching_prefix_test() {
    let operations = vec![vec![
        Operation::Insert(KeyHash([1, 1, 1, 0, 0, 0, 0, 0]), [1, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 1, 0, 0, 0, 0, 0, 0]), [2, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([0, 0, 0, 0, 0, 0, 0, 0]), [3, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([5, 0, 0, 0, 0, 0, 0, 0]), [4, 0, 0, 0, 0, 0, 0, 0]),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn matching_prefix_test_2() {
    let operations = vec![vec![
        Operation::Insert(KeyHash([1, 1, 1, 0, 0, 0, 0, 0]), [1, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 1, 0, 0, 0, 0, 0, 0]), [2, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 0, 0, 0, 0, 0, 0, 0]), [3, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([2, 0, 0, 0, 0, 0, 0, 0]), [4, 0, 0, 0, 0, 0, 0, 0]),
    ]];

    end_to_end_entry_ops(operations);
}
#[test]
fn matching_prefix_test_3() {
    let operations = vec![vec![
        Operation::Insert(KeyHash([1, 1, 1, 0, 0, 0, 0, 0]), [1, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 1, 1, 0, 0, 1, 0, 0]), [2, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 1, 0, 0, 0, 0, 0, 0]), [3, 0, 0, 0, 0, 0, 0, 0]),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn matching_prefix_test_4() {
    let operations = vec![vec![
        Operation::Insert(KeyHash([1, 1, 1, 1, 1, 0, 0, 0]), [1, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 1, 1, 1, 0, 0, 0, 0]), [2, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 0, 0, 0, 0, 0, 0, 0]), [3, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 1, 0, 0, 0, 0, 0, 0]), [4, 0, 0, 0, 0, 0, 0, 0]),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn shift_error_test_1() {
    let operations = vec![vec![
        Operation::EntryOrInsert(
            KeyHash([2147483648, 0, 0, 0, 0, 0, 0, 0]),
            [0, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::EntryInsert(KeyHash([0, 0, 0, 0, 0, 0, 0, 0]), [0, 0, 0, 0, 0, 0, 0, 0]),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn additional_test_case_1() {
    let operations = vec![vec![
        Operation::Insert(KeyHash([1, 1, 1, 1, 1, 1, 1, 0]), [1, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 1, 1, 1, 1, 1, 0, 0]), [2, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 1, 1, 1, 1, 0, 0, 0]), [3, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 1, 0, 0, 0, 0, 0, 0]), [4, 0, 0, 0, 0, 0, 0, 0]),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn additional_test_case_2() {
    let operations = vec![
        vec![
            Operation::Insert(
                KeyHash([
                    3082270391, 3072898999, 3082270647, 3082205111, 3082270647, 3082270647,
                    3082270647, 3082270647,
                ]),
                [1, 0, 0, 0, 0, 0, 0, 0],
            ),
            Operation::Insert(
                KeyHash([
                    3082270647, 3082270647, 3082270647, 3082270647, 3082270647, 3082270647,
                    3082270647, 3082270647,
                ]),
                [2, 0, 0, 0, 0, 0, 0, 0],
            ),
        ],
        vec![
            Operation::Insert(
                KeyHash([
                    3082234039, 3082270647, 3082270647, 3082270647, 3082270647, 3082270647,
                    3082270391, 3072898999,
                ]),
                [3, 0, 0, 0, 0, 0, 0, 0],
            ),
            Operation::Insert(
                KeyHash([
                    3082270647, 3082270647, 4294967223, 3086952447, 3082270647, 3082244279,
                    3082270647, 3082270647,
                ]),
                [4, 0, 0, 0, 0, 0, 0, 0],
            ),
        ],
        vec![
            Operation::Insert(
                KeyHash([
                    3082234039, 3082270647, 3082270647, 3082270647, 3082270647, 3082270647,
                    3082270647, 3082270647,
                ]),
                [5, 0, 0, 0, 0, 0, 0, 0],
            ),
            Operation::Insert(
                KeyHash([
                    3082234039, 3082270647, 3082270647, 3082270647, 3082270504, 3082270647,
                    3082205111, 3072898999,
                ]),
                [6, 0, 0, 0, 0, 0, 0, 0],
            ),
            Operation::Insert(
                KeyHash([
                    3082270647, 4290230199, 3082270647, 3075520439, 3082270647, 3082270647, 29111,
                    0,
                ]),
                [7, 0, 0, 0, 0, 0, 0, 0],
            ),
        ],
    ];

    end_to_end_entry_ops(operations);
}

#[test]
fn additional_test_case_3() {
    let operations = vec![vec![
        Operation::Insert(KeyHash([1, 2, 3, 4, 5, 0, 0, 0]), [1, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 2, 3, 4, 5, 0, 2, 2]), [2, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(KeyHash([1, 2, 2, 2, 0, 2, 0, 0]), [3, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Remove(KeyHash([1, 2, 2, 2, 0, 2, 0, 0])),
        Operation::Insert(KeyHash([0, 0, 0, 0, 0, 0, 0, 0]), [4, 0, 0, 0, 0, 0, 0, 0]),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn additional_test_case_4() {
    let operations = vec![vec![
        Operation::Insert(
            KeyHash([1, 1, 1, 4294508571, 2490367, 0, 1, 1]),
            [1, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Insert(KeyHash([1, 1, 1, 1, 2, 1, 3, 4]), [2, 0, 0, 0, 0, 0, 0, 0]),
        Operation::Insert(
            KeyHash([1, 1, 1, 27, 2424832, 0, 1, 1]),
            [3, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Remove(KeyHash([1, 1, 1, 1, 2, 1, 3, 4])),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn test_max_min_key_entries() {
    let operations = vec![vec![
        Operation::EntryInsert(
            KeyHash([
                4294967295, 4294967295, 2686975, 4294901760, 150953727, 4278190080, 4294967295,
                4294967295,
            ]),
            [255, 255, 255, 255, 255, 255, 255, 255],
        ),
        Operation::EntryOrInsert(KeyHash([0, 0, 0, 0, 0, 0, 0, 0]), [0, 0, 0, 0, 0, 0, 0, 0]),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn additional_test_case_5() {
    let operations = vec![vec![
        Operation::Insert(
            KeyHash([
                3082270391, 1212696759, 3074967624, 3082270647, 3082205111, 683128759, 3082270647,
                3082270647,
            ]),
            [1, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Insert(
            KeyHash([
                3082270647, 3082270391, 3072898999, 3082270647, 364885943, 353635078, 3474199814,
                4144637948,
            ]),
            [2, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Insert(
            KeyHash([
                3082270647, 3082270647, 3082270647, 3082270647, 3082289151, 3082270391, 3072898999,
                3082270647,
            ]),
            [3, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::EntryInsert(
            KeyHash([
                3082270647, 3082270647, 3082270647, 3082270647, 183, 0, 3082270464, 4294967295,
            ]),
            [4, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Insert(
            KeyHash([3082270647, 1907865527, 0, 0, 0, 0, 0, 0]),
            [5, 0, 0, 0, 0, 0, 0, 0],
        ),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn additional_test_case_6() {
    let operations = vec![vec![
        Operation::Insert(
            KeyHash([
                2139062143, 2139062143, 2141552511, 2139062143, 2139062143, 2139062143, 252645135,
                252645135,
            ]),
            [1, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Insert(
            KeyHash([
                2139062143, 2139062143, 1518305151, 2138537855, 2139062143, 2139062143, 2139062143,
                2139062143,
            ]),
            [2, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Insert(
            KeyHash([
                2139062143, 2139062143, 2139062143, 2139062143, 2139062143, 2139062143, 2139062143,
                2139062143,
            ]),
            [3, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Insert(
            KeyHash([
                2139062143, 2139062143, 2139062143, 2139062143, 11, 4294967293, 0, 2139062143,
            ]),
            [4, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Remove(KeyHash([
            2139062143, 2139062143, 1518305151, 2138537855, 2139062143, 2139062143, 2139062143,
            2139062143,
        ])),
        Operation::Insert(
            KeyHash([
                2139062143, 2139062143, 2139062143, 11, 4294967293, 2139052671, 2139062143, 1408895,
            ]),
            [5, 0, 0, 0, 0, 0, 0, 0],
        ),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn additional_test_case_7() {
    let operations = vec![vec![
        Operation::Insert(
            KeyHash([
                3082234039, 12040119, 3065493431, 3082270647, 3082270504, 3082270647, 3082270647,
                1431680951,
            ]),
            [1, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Insert(
            KeyHash([
                3082234039, 3082270647, 3082270647, 3082270647, 3082270647, 3082270647, 3082270391,
                3072898999,
            ]),
            [2, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Insert(
            KeyHash([
                3082234039, 3082270647, 3082270647, 3082270647, 2139062199, 2139062143, 2139062143,
                763330431,
            ]),
            [3, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Insert(
            KeyHash([
                2139062143, 2139062143, 2139062143, 2139062143, 2139062143, 2139062143, 2139062143,
                2088730495,
            ]),
            [4, 0, 0, 0, 0, 0, 0, 0],
        ),
        Operation::Remove(KeyHash([
            3082234039, 12040119, 3065493431, 3082270647, 3082270504, 3082270647, 3082270647,
            1431680951,
        ])),
        Operation::Get(KeyHash([3082234039, 0, 0, 0, 0, 0, 0, 0])),
    ]];

    end_to_end_entry_ops(operations);
}

#[test]
fn additional_test_case_8() {
    let operations = vec![vec![
        Operation::EntryInsert(
            KeyHash([
                286331153, 286331153, 286331153, 286331153, 286331153, 286331153, 286331153,
                286331153,
            ]),
            [17, 17, 17, 17, 17, 17, 17, 17],
        ),
        Operation::EntryInsert(
            KeyHash([
                286331153, 286331153, 286331153, 4294967057, 4294967294, 4294967295, 4294967295,
                3087007743,
            ]),
            [183, 183, 59, 183, 183, 183, 72, 183],
        ),
        Operation::EntryInsert(
            KeyHash([
                4294967055, 4294967295, 4294967295, 4294967295, 4294967295, 4294948721, 4290183167,
                4294966783,
            ]),
            [255, 255, 255, 255, 255, 255, 255, 255],
        ),
        Operation::EntryInsert(
            KeyHash([
                4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295,
                4294967039,
            ]),
            [48, 255, 255, 255, 255, 255, 255, 255],
        ),
        Operation::EntryInsert(
            KeyHash([
                4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967039, 788529151,
                4294967295,
            ]),
            [255, 255, 255, 255, 255, 255, 255, 255],
        ),
        Operation::EntryInsert(
            KeyHash([
                3077701631, 4227858431, 4261412790, 4294967295, 4294967295, 4294967295, 4294967295,
                4294967295,
            ]),
            [255, 255, 255, 255, 255, 255, 255, 255],
        ),
        Operation::EntryInsert(
            KeyHash([
                1162167621, 1162167621, 1162167621, 1162167621, 1162167621, 1162167621, 1162167621,
                1162167621,
            ]),
            [69, 69, 69, 69, 69, 69, 69, 69],
        ),
        Operation::Remove(KeyHash([
            1162167621, 1162167621, 1162167621, 1162167621, 1162167621, 1162167621, 1162167621,
            1162167621,
        ])),
        Operation::Remove(KeyHash([
            1162167621, 3244574021, 3153781694, 4294912025, 4294966783, 3082270647, 3082287031,
            1236776887,
        ])),
    ]];

    end_to_end_entry_ops(operations);
}
