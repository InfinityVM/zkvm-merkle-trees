#![allow(unused)]

use std::{
    collections::{hash_map, BTreeMap, HashMap},
    rc::Rc,
};

use kairos_trie::DigestHasher;
use proptest::{prelude::*, sample::SizeRange};

use sha2::Sha256;
use zkvm_merkle_btree::{
    db::MemoryDb,
    node::NodeHash,
    snapshot::{Snapshot, SnapshotBuilder, VerifiedSnapshot},
    store::Store,
    transaction::MerkleBTreeTxn,
};

#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Insert(u32, u32),
    Get(u32),
    Delete(u32),
    GetFirstKeyValue,
    GetLastKeyValue,
}

prop_compose! {
    pub fn key_range(max_key: u32) (key in 0..max_key) -> u32 {
        key
    }
}

impl Arbitrary for Operation {
    type Parameters = u32;
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(max_key: Self::Parameters) -> Self::Strategy {
        key_range(max_key)
            .prop_flat_map(|max_key| {
                prop_oneof![
                    (0..=max_key, any::<u32>()).prop_map(|(k, v)| Operation::Insert(k, v)),
                    (0..=max_key).prop_map(Operation::Get),
                    (0..=max_key).prop_map(Operation::Delete),
                    Just(Operation::GetFirstKeyValue),
                    Just(Operation::GetLastKeyValue)
                ]
            })
            .boxed()
    }
}

prop_compose! {
    pub fn arb_operations(max_key_count: u32, op_count: impl Into<SizeRange>)
                         (ops in prop::collection::vec(
                              (1..6u8,
                               0..max_key_count,
                               any::<u32>()
                              ),
                              op_count
                            )
                         ) -> Vec<Operation> {
    ops.into_iter().filter_map(|(op, key, value)| {
        match op {
            1 => Some(Operation::Get(key)),
            2 => Some(Operation::Insert(key, value)),
            3 => Some(Operation::Delete(key)),
            4 => Some(Operation::GetFirstKeyValue),
            5 => Some(Operation::GetLastKeyValue),
            _ => unreachable!(),
        }}).collect()
    }
}

prop_compose! {
    pub fn arb_batches(max_key_count: u32, op_count: impl Into<SizeRange>, max_batch_count: usize, max_batch_size: usize)
                      (
                          ops in arb_operations(max_key_count, op_count),
                          windows in prop::collection::vec(0..max_batch_size, 0..max_batch_count - 1)
                      ) -> Vec<Vec<Operation>> {
                          arb_batches_inner(ops, windows)
    }
}

fn arb_batches_inner(ops: Vec<Operation>, windows: Vec<usize>) -> Vec<Vec<Operation>> {
    let mut batches = Vec::new();
    let mut start = 0;

    // Partition the operations into batches
    for window_size in windows {
        if start + window_size > ops.len() {
            break;
        }

        batches.push(ops[start..start + window_size].to_vec());

        start += window_size;
    }

    if start < ops.len() {
        batches.push(ops[start..].to_vec());
    }

    batches
}

// Code like this runs in the server.
pub fn run_against_snapshot_builder(
    batch: &[Operation],
    old_root_hash: NodeHash,
    db: Rc<MemoryDb<u32, u32>>,
    btree: &mut BTreeMap<u32, u32>,
) -> (NodeHash, Snapshot<u32, u32>) {
    let mut txn = MerkleBTreeTxn::new_snapshot_builder_txn(old_root_hash, db);

    for op in batch {
        let (old, new) = trie_op(op, &mut txn);
        let (old_bt, new_bt) = btree_op(op, btree);
        assert_eq!(old, old_bt);
        assert_eq!(new, new_bt);
    }

    let new_root_hash = txn.commit(&mut DigestHasher::<Sha256>::default()).unwrap();
    let snapshot = txn.build_initial_snapshot();
    (new_root_hash, snapshot)
}

/// Code like this would run in a zkVM
pub fn run_against_snapshot(
    batch: &[Operation],
    snapshot: Snapshot<u32, u32>,
    new_root_hash: NodeHash,
    old_root_hash: NodeHash,
) {
    // Does the contract's expected old root hash match the submitted snapshot?

    let verified_snapshot =
        VerifiedSnapshot::verify_snapshot(&snapshot, &mut DigestHasher::<Sha256>::default())
            .unwrap();

    assert_eq!(old_root_hash, verified_snapshot.root_hash());

    // Create a transaction against the snapshot at the old root hash
    // let mut txn = MerkleBTreeTxn::from_verified_snapshot(verified_snapshot);
    let mut txn = MerkleBTreeTxn::from_verified_snapshot_ref(&verified_snapshot);

    // Apply the operations to the transaction
    for op in batch {
        trie_op(op, &mut txn);
    }

    // Calculate the new root hash
    let root_hash = txn
        .calc_root_hash(&mut DigestHasher::<Sha256>::default())
        .unwrap();

    // Check that the new root hash matches the submitted new root hash
    // This last bit is actually unnecessary, but it's a good sanity check
    assert_eq!(root_hash, new_root_hash);
}

fn trie_op<S: Store<Key = u32, Value = u32>>(
    op: &Operation,
    txn: &mut MerkleBTreeTxn<S>,
) -> (Option<u32>, Option<u32>) {
    match op {
        Operation::Insert(key, value) => {
            let old = txn.insert(*key, *value).unwrap();
            (old, Some(*value))
        }
        Operation::Get(key) => {
            let value = txn.get(key).unwrap();
            (value, value)
        }
        Operation::Delete(key) => {
            let old = txn.remove(key).unwrap();
            (old, None)
        }
        Operation::GetFirstKeyValue => {
            let value = txn.first_key_value().unwrap().map(|(_, v)| *v);
            (value, value)
        }
        Operation::GetLastKeyValue => {
            let value = txn.last_key_value().unwrap().map(|(_, v)| *v);
            (value, value)
        }
    }
}

fn btree_op(op: &Operation, btree: &mut BTreeMap<u32, u32>) -> (Option<u32>, Option<u32>) {
    match op {
        Operation::Insert(key, value) => {
            let old = btree.insert(*key, *value);
            (old, Some(*value))
        }
        Operation::Get(key) => {
            let value = btree.get(key).copied();
            (value, value)
        }
        Operation::Delete(key) => {
            let old = btree.remove(key);
            (old, None)
        }
        Operation::GetFirstKeyValue => {
            let value = btree.first_key_value().map(|(_, v)| *v);
            (value, value)
        }
        Operation::GetLastKeyValue => {
            let value = btree.last_key_value().map(|(_, v)| *v);
            (value, value)
        }
    }
}

fn end_to_end_ops(batches: Vec<Vec<Operation>>) {
    // The persistent backing, likely rocksdb
    let db = Rc::new(MemoryDb::<u32, u32>::default());

    // An empty trie root
    let mut prior_root_hash = NodeHash::default();

    // used as a reference for B-tree behavior
    let mut btree = BTreeMap::new();

    for batch in batches.iter() {
        eprintln!("Batch size: {}", batch.len());
        // We build a snapshot on the server.
        let (new_root_hash, snapshot) =
            run_against_snapshot_builder(batch, prior_root_hash, db.clone(), &mut btree);

        // We verify the snapshot in a zkVM
        run_against_snapshot(batch, snapshot, new_root_hash, prior_root_hash);

        // After a batch is verified in an on chain zkVM the contract would update's its root hash
        prior_root_hash = new_root_hash;
    }

    // After all batches are applied, the B-tree and the btree should be in sync
    let txn = MerkleBTreeTxn::new_snapshot_builder_txn(prior_root_hash, db);

    // Check that the B-tree and the btree are in sync
    for (k, v) in btree.iter() {
        let ret_v = txn.get(k).unwrap().unwrap();
        assert_eq!(v, &ret_v);
    }

    // Verify first and last key/value pairs match
    if let Some((first_k, first_v)) = btree.first_key_value() {
        let (ret_k, ret_v) = txn.first_key_value().unwrap().unwrap();
        assert_eq!(first_k, ret_k);
        assert_eq!(first_v, ret_v);
    }

    if let Some((last_k, last_v)) = btree.last_key_value() {
        let (ret_k, ret_v) = txn.last_key_value().unwrap().unwrap();
        assert_eq!(last_k, ret_k);
        assert_eq!(last_v, ret_v);
    }
}

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