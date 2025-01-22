use std::{collections::BTreeMap, ops::RangeBounds, rc::Rc};

use proptest::{prelude::*, sample::SizeRange};

use sha2::Sha256;
use zkvm_merkle_btree::{db::MemoryDb, *};

#[derive(Debug, Clone, Copy, arbitrary::Arbitrary)]
pub enum Operation {
    Insert(u32, u32),
    Get(u32),
    Delete(u32),
    GetFirstKeyValue,
    GetLastKeyValue,
    IterAll,
    IterRange(Option<u32>, Option<u32>),
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
                    50 => (0..=max_key, any::<u32>()).prop_map(|(k, v)| Operation::Insert(k, v)),
                    20 => (0..=max_key).prop_map(Operation::Get),
                    20 => (0..=max_key).prop_map(Operation::Delete),
                    5 => Just(Operation::GetFirstKeyValue),
                    5 => Just(Operation::GetLastKeyValue),
                    5 => Just(Operation::IterAll),
                    10 => (proptest::option::of(0..10000u32), proptest::option::of(0..10000u32)).prop_map(|(start, end)| Operation::IterRange(start, end)),
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

pub fn arb_batches_inner(ops: Vec<Operation>, windows: Vec<usize>) -> Vec<Vec<Operation>> {
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

fn iter_test<S: Store<Key = u32, Value = u32>>(
    range: impl RangeBounds<u32> + Clone,
    txn: &Transaction<S>,
    std: &BTreeMap<u32, u32>,
) {
    itertools::assert_equal(
        txn.range(range.clone()).enumerate(),
        std.range(range).map(Ok).enumerate(),
    );
}

// Code like this runs in the server.
pub fn run_against_snapshot_builder(
    batch: &[Operation],
    old_root_hash: NodeHash,
    db: Rc<MemoryDb<u32, u32>>,
    std_btree: &mut BTreeMap<u32, u32>,
) -> (NodeHash, Snapshot<u32, u32>) {
    let mut txn_btree = Transaction::new_snapshot_builder_txn(old_root_hash, db);

    for op in batch {
        match *op {
            Operation::Insert(k, v) => {
                let res_txn = txn_btree.insert(k, v).unwrap();
                let res_std = std_btree.insert(k, v);
                assert_eq!(
                    res_txn, res_std,
                    "insert failed for key: {}, value: {}, merkle btree returned {:?}, btreemap returned {:?}",
                    k, v, res_txn, res_std
                );
            }
            Operation::Get(k) => {
                let res_txn = txn_btree.get(&k).unwrap().cloned();
                let res_std = std_btree.get(&k).cloned();
                assert_eq!(
                    res_txn, res_std,
                    "get failed for key: {}, merkle btree returned {:?}, btreemap returned {:?}",
                    k, res_txn, res_std
                );
            }
            Operation::Delete(k) => {
                let res_txn = txn_btree.remove(&k).unwrap();
                let res_std = std_btree.remove(&k);
                assert_eq!(
                    res_txn, res_std,
                    "delete failed for key: {}, merkle btree returned {:?}, btreemap returned {:?}",
                    k, res_txn, res_std
                );
            }
            Operation::GetFirstKeyValue => {
                let res_txn = txn_btree.first_key_value().unwrap();
                let res_std = std_btree.first_key_value();
                assert_eq!(
                    res_txn, res_std,
                    "get first key value failed, merkle btree returned {:?}, btreemap returned {:?}",
                    res_txn, res_std
                );
            }
            Operation::GetLastKeyValue => {
                let res_txn = txn_btree.last_key_value().unwrap();
                let res_std = std_btree.last_key_value();
                assert_eq!(
                    res_txn, res_std,
                    "get last key value failed, merkle btree returned {:?}, btreemap returned {:?}",
                    res_txn, res_std
                );
            }
            Operation::IterAll => {
                itertools::assert_equal(
                    txn_btree.iter().enumerate(),
                    std_btree.iter().map(Ok).enumerate(),
                );
            }
            Operation::IterRange(start, end) => {
                // We slightly differ from the std btree in that we don't panic when the start is greater than the end
                if let (Some(start), Some(end)) = (start, end) {
                    if start > end {
                        assert!(txn_btree.range(start..end).next().is_none());

                        continue;
                    }
                }

                match (start, end) {
                    (Some(start), Some(end)) => iter_test(start..end, &txn_btree, std_btree),
                    (Some(start), None) => iter_test(start.., &txn_btree, std_btree),
                    (None, Some(end)) => iter_test(..end, &txn_btree, std_btree),
                    (None, None) => iter_test(.., &txn_btree, std_btree),
                };
            }
        }
    }

    let new_root_hash = txn_btree
        .commit(&mut DigestHasher::<Sha256>::default())
        .unwrap();
    let snapshot = txn_btree.build_initial_snapshot();
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
    let mut txn = Transaction::from_verified_snapshot_ref(&verified_snapshot);

    // Apply the operations to the transaction
    for op in batch {
        merkle_btree_op(op, &mut txn);
    }

    // Calculate the new root hash
    let root_hash = txn
        .calc_root_hash(&mut DigestHasher::<Sha256>::default())
        .unwrap();

    // Check that the new root hash matches the submitted new root hash
    // This last bit is actually unnecessary, but it's a good sanity check
    assert_eq!(root_hash, new_root_hash);
}

fn merkle_btree_op<S: Store<Key = u32, Value = u32>>(op: &Operation, txn: &mut Transaction<S>) {
    match op {
        Operation::Insert(key, value) => {
            txn.insert(*key, *value).unwrap();
        }
        Operation::Get(key) => {
            txn.get(key).unwrap();
        }
        Operation::Delete(key) => {
            txn.remove(key).unwrap();
        }
        Operation::GetFirstKeyValue => {
            txn.first_key_value().unwrap();
        }
        Operation::GetLastKeyValue => {
            txn.last_key_value().unwrap();
        }

        Operation::IterAll => {
            txn.iter().for_each(|r| {
                r.unwrap();
            });
        }
        Operation::IterRange(start, end) => {
            match (*start, *end) {
                (Some(start), Some(end)) => txn.range(start..end).for_each(|r| {
                    r.unwrap();
                }),
                (Some(start), None) => txn.range(start..).for_each(|r| {
                    r.unwrap();
                }),
                (None, Some(end)) => txn.range(..end).for_each(|r| {
                    r.unwrap();
                }),
                (None, None) => txn.range::<u32, _>(..).for_each(|r| {
                    r.unwrap();
                }),
            };
        }
    }
}

pub fn end_to_end_ops(batches: Vec<Vec<Operation>>) {
    // The persistent backing, likely rocksdb
    let db = Rc::new(MemoryDb::<u32, u32>::default());

    // An empty trie root
    let mut prior_root_hash = NodeHash::default();

    // used as a reference for B-tree behavior
    let mut btree = BTreeMap::new();

    for batch in batches.iter() {
        // We build a snapshot on the server.
        let (new_root_hash, snapshot) =
            run_against_snapshot_builder(batch, prior_root_hash, db.clone(), &mut btree);

        // We verify the snapshot in a zkVM
        run_against_snapshot(batch, snapshot, new_root_hash, prior_root_hash);

        // After a batch is verified in an on chain zkVM the contract would update's its root hash
        prior_root_hash = new_root_hash;
    }

    // After all batches are applied, the B-tree and the btree should be in sync
    let txn = Transaction::new_snapshot_builder_txn(prior_root_hash, db);

    // Check that the B-tree and the btree are in sync
    for (k, v) in btree.iter() {
        let ret_v = txn.get(k).unwrap().unwrap();
        assert_eq!(v, ret_v);
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
