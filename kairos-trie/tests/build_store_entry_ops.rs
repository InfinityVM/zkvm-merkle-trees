mod utils;
use std::{collections::HashMap, rc::Rc};

use proptest::prelude::*;

use kairos_trie::{
    stored::{memory_db::MemoryDb, merkle::SnapshotBuilder},
    KeyHash, Transaction, TrieRoot,
};
use utils::operations::*;

fn end_to_end_entry_ops(batches: Vec<Vec<Operation>>) {
    // The persistent backing, likely rocksdb
    let db = Rc::new(MemoryDb::<[u8; 8]>::empty());

    // An empty trie root
    let mut prior_root_hash = TrieRoot::default();

    // used as a reference for trie behavior
    let mut hash_map = HashMap::new();

    for batch in batches.iter() {
        eprintln!("Batch size: {}", batch.len());
        // We build a snapshot on the server.
        let (new_root_hash, snapshot) =
            run_against_snapshot_builder(batch, prior_root_hash, db.clone(), &mut hash_map);

        // We verify the snapshot in a zkVM
        run_against_snapshot(batch, snapshot, new_root_hash, prior_root_hash);

        // After a batch is verified in an on chain zkVM the contract would update's its root hash
        prior_root_hash = new_root_hash;
    }

    // After all batches are applied, the trie and the hashmap should be in sync
    let txn = Transaction::from_snapshot_builder(
        SnapshotBuilder::<_, [u8; 8]>::empty(db).with_trie_root_hash(prior_root_hash),
    );

    // Check that the trie and the hashmap are in sync
    for (k, v) in hash_map.iter() {
        let ret_v = txn.get(k).unwrap().unwrap();
        assert_eq!(v, ret_v);
    }
}

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
