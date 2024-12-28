use btree_test_utils::*;
use proptest::prelude::*;

#[test]
fn test_empty() {
    end_to_end_ops(vec![vec![]]);
}

#[test]
fn test_single_insert() {
    end_to_end_ops(vec![vec![Operation::Insert(0, 0)]]);
}

#[test]
fn test_two_inserts() {
    end_to_end_ops(vec![vec![Operation::Insert(0, 0), Operation::Insert(1, 1)]]);
}

#[test]
fn test_get_and_delete_ops() {
    let batches = vec![
        vec![Operation::Insert(8, 0), Operation::Insert(7, 0)],
        vec![
            Operation::Insert(1, 2493376526),
            Operation::Insert(5, 2836387748),
            Operation::Insert(3, 357313916),
            Operation::Delete(2),
            Operation::Delete(1),
            Operation::Insert(3, 1588217466),
            Operation::Delete(1),
            Operation::Delete(3),
            Operation::Get(5),
            Operation::Delete(4),
            Operation::Insert(3, 1643695574),
            Operation::Delete(6),
            Operation::Delete(3),
            Operation::Get(3),
            Operation::Delete(0),
            Operation::GetLastKeyValue,
            Operation::Get(1),
            Operation::Get(4),
            Operation::Insert(3, 3418225359),
            Operation::Insert(6, 733456367),
            Operation::Delete(2),
            Operation::Delete(2),
            Operation::Delete(1),
            Operation::GetLastKeyValue,
            Operation::GetLastKeyValue,
            Operation::GetLastKeyValue,
            Operation::Delete(2),
            Operation::Insert(0, 1649480804),
            Operation::GetLastKeyValue,
            Operation::Insert(1, 3462688671),
            Operation::GetLastKeyValue,
            Operation::Get(0),
            Operation::GetLastKeyValue,
            Operation::Delete(0),
            Operation::GetFirstKeyValue,
            Operation::Insert(5, 585243167),
            Operation::GetFirstKeyValue,
            Operation::GetFirstKeyValue,
            Operation::GetFirstKeyValue,
            Operation::GetLastKeyValue,
            Operation::Delete(4),
            Operation::Get(0),
            Operation::GetLastKeyValue,
            Operation::Delete(0),
        ],
        vec![
            Operation::GetLastKeyValue,
            Operation::GetLastKeyValue,
            Operation::GetLastKeyValue,
            Operation::Get(0),
            Operation::GetLastKeyValue,
            Operation::Get(1),
            Operation::GetFirstKeyValue,
            Operation::Delete(8),
            Operation::GetLastKeyValue,
            Operation::Delete(0),
            Operation::GetLastKeyValue,
            Operation::Delete(1),
            Operation::Delete(1),
            Operation::GetLastKeyValue,
            Operation::Insert(2, 720712201),
            Operation::GetFirstKeyValue,
            Operation::Insert(0, 2060025109),
        ],
    ];
    end_to_end_ops(batches);
}

#[test]
fn test_minimal_failing_input() {
    let batches = vec![
        vec![
            Operation::Insert(6, 0),
            Operation::Insert(3, 0),
            Operation::Insert(0, 2709613295),
            Operation::Insert(1, 1673088535),
            Operation::Insert(2, 894537384),
            Operation::Insert(0, 3368456274),
            Operation::Insert(7, 3999566415),
            Operation::Insert(4, 1111448046),
            Operation::Delete(3),
        ],
        vec![Operation::Delete(6)],
    ];
    end_to_end_ops(batches);
}

#[test]
fn test_minimal_failing_input_2() {
    let batches = vec![
        vec![
            Operation::Insert(5, 0),
            Operation::Insert(1, 0),
            Operation::Insert(3, 0),
            Operation::Insert(0, 0),
            Operation::Insert(2, 0),
            Operation::Insert(6, 0),
            Operation::Insert(4, 0),
        ],
        vec![Operation::Delete(0)],
        vec![Operation::Delete(1), Operation::Insert(0, 0)],
    ];
    end_to_end_ops(batches);
}

#[test]
fn test_minimal_failing_input_3() {
    let batches = vec![
        vec![Operation::Insert(2, 0), Operation::Delete(2)],
        vec![Operation::GetFirstKeyValue],
    ];
    end_to_end_ops(batches);
}

proptest! {
    #[test]
    fn prop_end_to_end_ops(
        // This is pretty, but turns out to be horribly compared to the old strategy
        batches in prop::collection::vec(prop::collection::vec(any_with::<Operation>(100), 1..100), 1..10)) {
        end_to_end_ops(batches);
    }

    #[test]
    fn prop_end_to_end_entry_ops_single_batch(
        batch in prop::collection::vec(any_with::<Operation>(1000), 1..1000)) {
        end_to_end_ops(vec![batch]);
    }

    #[test]
    fn prop_end_to_end_ops_old_strategy(
        batches in arb_batches(100_000, 1..100_000, 1000, 10_000)) {
        end_to_end_ops(batches);
    }


    #[test]
    fn prop_end_to_end_ops_old_strategy_small_tree(
        batches in arb_batches(1000, 1..100_000, 1000, 10_000)) {
        end_to_end_ops(batches);
    }


    #[test]
    fn prop_end_to_end_ops_old_strategy_tiny_tree(
        batches in arb_batches(100, 1..100_000, 1000, 10_000)) {
        end_to_end_ops(batches);
    }
}
