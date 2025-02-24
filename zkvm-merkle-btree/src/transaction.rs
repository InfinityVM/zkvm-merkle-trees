use alloc::sync::Arc;
use core::mem;
use smallvec::SmallVec;
use std::{
    borrow::Borrow,
    marker::PhantomData,
    ops::{Bound, RangeBounds, RangeFull},
};

use arrayvec::ArrayVec;
use kairos_trie::{PortableHash, PortableHasher};

use crate::{
    db::{DatabaseGet, DatabaseSet},
    errors::BTreeError,
    node::{
        InnerNode, InnerNodeSnapshot, InnerOuter, LeafNode, Node, NodeHash, NodeRef, BTREE_ORDER,
        EMPTY_TREE_ROOT_HASH,
    },
    snapshot::{Snapshot, SnapshotBuilder, VerifiedSnapshot},
    store::{Idx, Store},
};

/// A transaction against a merkle b+tree.
#[derive(Clone)]
pub struct Transaction<S: Store> {
    pub data_store: S,
    current_root: Option<NodeRef<S::Key, S::Value>>,
}

/// # Safety
///
/// This impl is needed to work around `SnapshotBuilder` having a `RefCell`.
/// This impl would not be safe on `SnapshotBuilder<K, V, Db>` directly.
///
/// The safety of this impl is reliant on the following:
/// - SnapshotBuilder` is directly owned by the `Transaction` and not behind an `Rc` or `Arc`.
/// - The RefCell in SnapshotBuilder is the only thing that prevents the `SnapshotBuilder` from having an auto impl of Send.
/// - Everything in the `RefCell` is append only, and the data types are `Send + Clone`.
/// - The references returned by the `SnapshotBuilder::get` are only ever short lived (enforced by the lifetime).
///   The lifetime returned by `SnapshotBuilder::get` enforces that reference cannot outlive SnapshotBuilder.
///   Given this the if SnapshotBuilder moves the lifetime is invalidated by the borrow checker.
///
/// Given these conditions we can clone and move or directly move a `Transaction<SnapshotBuilder<K, V, Db>>` to another thread.
unsafe impl<K, V, Db> Send for Transaction<SnapshotBuilder<K, V, Db>>
where
    K: Send + Ord + Clone + PortableHash,
    V: Send + Clone + PortableHash,
    Db: Send + DatabaseGet<K, V> + DatabaseSet<K, V>,
{
}

impl<S: Store> Transaction<S> {
    pub fn new(data_store: S) -> Self {
        Self {
            current_root: data_store.get_store_root_idx().map(NodeRef::Stored),
            data_store,
        }
    }

    /// Calculate the root hash of the trie.
    ///
    /// Caller must ensure that the hasher is reset before calling this method.
    #[inline]
    pub fn calc_root_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
    ) -> Result<NodeHash, BTreeError> {
        self.calc_root_hash_inner(hasher, &mut |_, _| Ok(()), &mut |_, _, _| Ok(()))
    }

    fn calc_root_hash_inner(
        &self,
        hasher: &mut impl PortableHasher<32>,
        on_modified_leaf: &mut impl FnMut(
            &NodeHash,
            &LeafNode<S::Key, S::Value>,
        ) -> Result<(), BTreeError>,
        on_modified_branch: &mut impl FnMut(
            &NodeHash,
            &InnerNode<S::Key, S::Value>,
            &ArrayVec<NodeHash, { BTREE_ORDER * 2 }>,
        ) -> Result<(), BTreeError>,
    ) -> Result<NodeHash, BTreeError> {
        match &self.current_root {
            // We allow two versions of an empty tree as an optimization to reuse the allocated memory.
            None => Ok(EMPTY_TREE_ROOT_HASH),
            Some(NodeRef::Leaf(leaf)) if leaf.keys.is_empty() => Ok(EMPTY_TREE_ROOT_HASH),

            Some(node_ref) => Self::calc_root_hash_node(
                #[cfg(debug_assertions)]
                0,
                hasher,
                &self.data_store,
                node_ref,
                on_modified_leaf,
                on_modified_branch,
            ),
        }
    }

    #[inline]
    fn calc_root_hash_node(
        #[cfg(debug_assertions)] depth: usize,
        hasher: &mut impl PortableHasher<32>,
        data_store: &S,
        node_ref: &NodeRef<S::Key, S::Value>,
        on_modified_leaf: &mut impl FnMut(
            &NodeHash,
            &LeafNode<S::Key, S::Value>,
        ) -> Result<(), BTreeError>,
        on_modified_branch: &mut impl FnMut(
            &NodeHash,
            &InnerNode<S::Key, S::Value>,
            &ArrayVec<NodeHash, { BTREE_ORDER * 2 }>,
        ) -> Result<(), BTreeError>,
    ) -> Result<NodeHash, BTreeError> {
        match node_ref {
            NodeRef::Inner(node) => {
                const MAX_CHILDREN: usize = BTREE_ORDER * 2;
                debug_assert!(MAX_CHILDREN == InnerNode::<S::Key, S::Value>::max_children());
                #[cfg(debug_assertions)]
                {
                    node.assert_inner_invariants();
                    debug_assert!(node.keys.len() == node.children.len() - 1);
                    debug_assert!(
                        depth == 0
                            || node.children.len() >= InnerNode::<S::Key, S::Value>::min_children()
                    );
                }

                let mut child_hashes = ArrayVec::<NodeHash, MAX_CHILDREN>::new();

                for child in &node.children {
                    let child_hash = Self::calc_root_hash_node(
                        #[cfg(debug_assertions)]
                        (depth + 1),
                        hasher,
                        data_store,
                        child,
                        on_modified_leaf,
                        on_modified_branch,
                    )?;
                    child_hashes.push(child_hash);
                }

                node.portable_hash_iter(hasher, child_hashes.iter());
                let hash = hasher.finalize_reset();

                on_modified_branch(&hash, node, &child_hashes)?;

                Ok(hash)
            }
            NodeRef::Leaf(leaf) => {
                #[cfg(debug_assertions)]
                {
                    leaf.assert_leaf_invariants();
                    debug_assert!(leaf.keys.len() == leaf.children.len());
                    debug_assert!(
                        depth == 0
                            || leaf.children.len() >= Node::<S::Key, S::Value>::min_children()
                    );
                }

                leaf.portable_hash(hasher);
                let hash = hasher.finalize_reset();

                on_modified_leaf(&hash, leaf)?;

                Ok(hash)
            }
            NodeRef::Stored(stored_idx) => data_store.calc_subtree_hash(hasher, *stored_idx),
        }
    }

    pub fn get(&self, key: &S::Key) -> Result<Option<&S::Value>, BTreeError> {
        match &self.current_root {
            None => Ok(None),
            Some(NodeRef::Inner(node)) => Self::get_inner(&self.data_store, node, key),
            Some(NodeRef::Leaf(leaf)) => match leaf.keys.binary_search(key) {
                Ok(idx) => Ok(Some(&leaf.children[idx])),
                Err(_) => Ok(None),
            },
            Some(NodeRef::Stored(idx)) => Self::get_stored(&self.data_store, *idx, key),
        }
    }

    /// Get the value associated with the key.
    ///
    /// Caller must ensure the parent_node is not empty node, and has one more child than keys.
    /// The root node could be an empty outer node, but a Root node should never be an inner empty node.
    fn get_inner<'s>(
        data_store: &'s S,
        mut parent_node: &'s InnerNode<S::Key, S::Value>,
        key: &S::Key,
    ) -> Result<Option<&'s S::Value>, BTreeError> {
        loop {
            let idx = match parent_node.keys.binary_search(key) {
                Ok(equal_key_idx) => equal_key_idx,
                Err(idx) => idx,
            };

            match &parent_node.children[idx] {
                NodeRef::Inner(child) => {
                    parent_node = child;
                }
                NodeRef::Leaf(leaf) => match leaf.keys.binary_search(key) {
                    Ok(idx) => return Ok(Some(&leaf.children[idx])),
                    Err(_) => return Ok(None),
                },
                NodeRef::Stored(idx) => {
                    return Self::get_stored(data_store, *idx, key);
                }
            }
        }
    }

    /// Get the value associated with the key from the `Store` (`Snapshot` or `SnapshotBuilder`).
    fn get_stored<'s>(
        data_store: &'s S,
        mut stored_idx: Idx,
        key: &S::Key,
    ) -> Result<Option<&'s S::Value>, BTreeError> {
        loop {
            let node = data_store.get(stored_idx)?;

            match node {
                InnerOuter::Inner(node) => {
                    // This invariant on inner nodes ensures Err(idx) is a valid index in children.
                    debug_assert!(node.keys.len() == node.children.len() - 1);
                    let idx = match node.keys.binary_search(key) {
                        Ok(idx) | Err(idx) => idx,
                    };

                    stored_idx = node.children[idx]
                }
                InnerOuter::Outer(leaf) => match leaf.keys.binary_search(key) {
                    Ok(idx) => return Ok(Some(&leaf.children[idx])),
                    Err(_) => return Ok(None),
                },
            }
        }
    }

    pub fn insert(&mut self, key: S::Key, value: S::Value) -> Result<Option<S::Value>, BTreeError> {
        let (middle_key, new_right): (
            S::Key,
            InnerOuter<Arc<Node<S::Key, NodeRef<S::Key, S::Value>>>, Arc<Node<S::Key, S::Value>>>,
        ) = 'handle_split: {
            match &mut self.current_root {
                None => {
                    self.current_root = Some(NodeRef::Leaf(Arc::new(Node {
                        keys: ArrayVec::from_iter([key]),
                        children: ArrayVec::from_iter([value]),
                    })));

                    return Ok(None);
                }
                Some(NodeRef::Stored(stored_idx)) => {
                    let node = self.data_store.get(*stored_idx)?.into();
                    self.current_root = Some(node);
                    return self.insert(key, value);
                }
                Some(NodeRef::Inner(node)) => {
                    let node = Arc::make_mut(node);

                    match Self::insert_inner(&self.data_store, node, key, value)? {
                        Insert_::Inserted => return Ok(None),
                        Insert_::Replaced(old_value) => return Ok(Some(old_value)),
                        Insert_::SplitNode {
                            middle_key,
                            right_node,
                        } => break 'handle_split (middle_key, InnerOuter::Inner(right_node)),
                    }
                }
                Some(NodeRef::Leaf(leaf)) => {
                    let leaf = Arc::make_mut(leaf);
                    match leaf.keys.binary_search(&key) {
                        Ok(idx) => {
                            return Ok(Some(mem::replace(&mut leaf.children[idx], value)));
                        }
                        Err(idx) if !leaf.is_full() => {
                            leaf.keys.insert(idx, key);
                            leaf.children.insert(idx, value);
                            return Ok(None);
                        }
                        Err(idx) => {
                            let (middle_key, new_right_leaf) = leaf.insert_split(idx, key, value);
                            break 'handle_split (middle_key, InnerOuter::Outer(new_right_leaf));
                        }
                    }
                }
            }
        };

        // A split occurred
        let left_node = mem::replace(
            &mut self.current_root,
            Some(NodeRef::Inner(Arc::new(InnerNode {
                keys: ArrayVec::from_iter([middle_key]),
                children: ArrayVec::new(),
            }))),
        );

        match &mut self.current_root {
            Some(NodeRef::Inner(node)) => {
                let node = Arc::make_mut(node);
                // Only a empty tree could have a null root node
                node.children.push(left_node.unwrap());
                node.children.push(new_right.into());

                #[cfg(debug_assertions)]
                node.assert_inner_invariants();

                Ok(None)
            }
            _ => unreachable!("The current root should always be an inner node"),
        }
    }

    fn insert_inner<'s>(
        data_store: &'s S,
        parent_node: &'s mut InnerNode<S::Key, S::Value>,
        key: S::Key,
        value: S::Value,
    ) -> Result<Insert_<S::Key, S::Value>, BTreeError> {
        // This invariant on inner nodes ensures Err(idx) is a valid index in children.
        debug_assert!(parent_node.keys.len() == parent_node.children.len() - 1);
        let idx = match parent_node.keys.binary_search(&key) {
            Ok(equal_key_idx) => equal_key_idx,
            Err(idx) => idx,
        };

        // This loop is only used to restart the logic after attaching a stored node.
        'handle_stored: loop {
            // We do this hideous nested labels to satisfy the borrow checker.
            let (middle_key, new_right) = 'handle_split: {
                match &mut parent_node.children[idx] {
                    NodeRef::Stored(stored_idx) => {
                        let node = data_store.get(*stored_idx)?.into();

                        parent_node.children[idx] = node;
                        continue 'handle_stored;
                    }
                    NodeRef::Inner(child) => {
                        match Self::insert_inner(data_store, Arc::make_mut(child), key, value)? {
                            Insert_::SplitNode {
                                middle_key,
                                right_node,
                            } => break 'handle_split (middle_key, InnerOuter::Inner(right_node)),
                            r => return Ok(r),
                        }
                    }
                    NodeRef::Leaf(leaf) => {
                        let leaf = Arc::make_mut(leaf);
                        match leaf.keys.binary_search(&key) {
                            Ok(idx) => {
                                return Ok(Insert_::Replaced(mem::replace(
                                    &mut leaf.children[idx],
                                    value,
                                )));
                            }
                            Err(idx) => {
                                if !leaf.is_full() {
                                    leaf.keys.insert(idx, key);
                                    leaf.children.insert(idx, value);
                                    return Ok(Insert_::Inserted);
                                } else {
                                    let (middle_key, new_right_leaf) =
                                        leaf.insert_split(idx, key, value);

                                    break 'handle_split (
                                        middle_key,
                                        InnerOuter::Outer(new_right_leaf),
                                    );
                                }
                            }
                        };
                    }
                }
            };

            // A split occurred
            return Ok(parent_node.handle_split(idx, middle_key, new_right));
        }
    }

    pub fn remove(&mut self, key: &S::Key) -> Result<Option<S::Value>, BTreeError> {
        match &mut self.current_root {
            None => Ok(None),
            Some(NodeRef::Stored(stored_idx)) => {
                let node = self.data_store.get(*stored_idx)?.into();
                self.current_root = Some(node);
                self.remove(key)
            }
            Some(NodeRef::Inner(node)) => {
                let node = Arc::make_mut(node);
                match Self::remove_inner(&self.data_store, node, key)? {
                    Remove::NotPresent => Ok(None),
                    Remove::Removed(value) => Ok(Some(value)),
                    Remove::Underflow(value) => {
                        if node.keys.is_empty() {
                            debug_assert!(node.children.len() == 1);
                            // handle_underflow removed the last key so the last child is the new root
                            self.current_root = Some(node.children.pop().unwrap());
                        }

                        Ok(Some(value))
                    }
                }
            }
            Some(NodeRef::Leaf(leaf)) => {
                let leaf = Arc::make_mut(leaf);
                match leaf.keys.binary_search(key) {
                    Ok(idx) => {
                        let value = leaf.children.remove(idx);
                        leaf.keys.remove(idx);

                        Ok(Some(value))
                    }
                    Err(_) => Ok(None),
                }
            }
        }
    }

    pub fn remove_inner(
        data_store: &S,
        parent_node: &mut InnerNode<S::Key, S::Value>,
        key: &S::Key,
    ) -> Result<Remove<S>, BTreeError> {
        // This invariant on inner nodes ensures Err(idx) is a valid index in children.
        debug_assert!(parent_node.keys.len() == parent_node.children.len() - 1);
        let idx = match parent_node.keys.binary_search(key) {
            Ok(idx) | Err(idx) => idx,
        };

        match &mut parent_node.children[idx] {
            NodeRef::Inner(child) => {
                let child = Arc::make_mut(child);
                match Self::remove_inner(data_store, child, key)? {
                    Remove::NotPresent => Ok(Remove::NotPresent),
                    Remove::Removed(value) => Ok(Remove::Removed(value)),
                    Remove::Underflow(value) => {
                        Self::handle_underflow(data_store, parent_node, idx, value)
                    }
                }
            }
            NodeRef::Leaf(leaf) => {
                let leaf = Arc::make_mut(leaf);
                match leaf.keys.binary_search(key) {
                    Ok(leaf_idx) => {
                        let value = leaf.children.remove(leaf_idx);
                        leaf.keys.remove(leaf_idx);

                        if leaf.children.len() < Node::<S::Key, S::Value>::min_children() {
                            Self::handle_underflow(data_store, parent_node, idx, value)
                        } else {
                            Ok(Remove::Removed(value))
                        }
                    }
                    Err(_) => Ok(Remove::NotPresent),
                }
            }
            NodeRef::Stored(stored_idx) => {
                let node = data_store.get(*stored_idx)?.into();
                parent_node.children[idx] = node;

                Self::remove_inner(data_store, parent_node, key)
            }
        }
    }

    fn handle_underflow(
        data_store: &S,
        parent_node: &mut InnerNode<S::Key, S::Value>,
        idx: usize,
        value: S::Value,
    ) -> Result<Remove<S>, BTreeError> {
        if let Err(()) = parent_node.merge_or_balance(idx) {
            if idx == 0 {
                let stored_idx = parent_node.children[1].stored().unwrap();
                parent_node.children[1] = NodeRef::from(data_store.get(stored_idx)?);
            } else {
                let hash_idx = parent_node.children[idx - 1].stored().unwrap();
                parent_node.children[idx - 1] = NodeRef::from(data_store.get(hash_idx)?);
            }

            // unwrap is fine because we just attached a stored sibling parent_node to the tree
            parent_node.merge_or_balance(idx).unwrap()
        };

        if parent_node.is_too_small() {
            Ok(Remove::Underflow(value))
        } else {
            Ok(Remove::Removed(value))
        }
    }

    pub fn first_key_value(&self) -> Result<Option<(&S::Key, &S::Value)>, BTreeError> {
        let mut node = match &self.current_root {
            None => return Ok(None),
            Some(NodeRef::Inner(node)) => node,
            Some(NodeRef::Leaf(leaf)) => {
                if leaf.keys.is_empty() {
                    return Ok(None);
                } else {
                    return Ok(Some((&leaf.keys[0], &leaf.children[0])));
                }
            }
            Some(NodeRef::Stored(stored_idx)) => {
                return self.first_key_value_stored(*stored_idx);
            }
        };

        loop {
            match &node.children[0] {
                NodeRef::Inner(child) => {
                    node = child;
                }
                NodeRef::Leaf(leaf) => {
                    return Ok(Some((&leaf.keys[0], &leaf.children[0])));
                }
                NodeRef::Stored(stored_idx) => {
                    return self.first_key_value_stored(*stored_idx);
                }
            }
        }
    }

    fn first_key_value_stored(
        &self,
        mut stored_idx: Idx,
    ) -> Result<Option<(&S::Key, &S::Value)>, BTreeError> {
        loop {
            let node = self.data_store.get(stored_idx)?;

            match node {
                InnerOuter::Inner(node) => {
                    stored_idx = node.children[0];
                }
                InnerOuter::Outer(leaf) => {
                    return Ok(Some((&leaf.keys[0], &leaf.children[0])));
                }
            }
        }
    }

    pub fn last_key_value(&self) -> Result<Option<(&S::Key, &S::Value)>, BTreeError> {
        let mut node = match &self.current_root {
            None => return Ok(None),
            Some(NodeRef::Inner(node)) => node,
            Some(NodeRef::Leaf(leaf)) => {
                if leaf.keys.is_empty() {
                    return Ok(None);
                } else {
                    return Ok(Some((
                        leaf.keys.last().unwrap(),
                        leaf.children.last().unwrap(),
                    )));
                }
            }
            Some(NodeRef::Stored(stored_idx)) => {
                return self.last_key_value_stored(*stored_idx);
            }
        };

        loop {
            match &node.children[node.keys.len()] {
                NodeRef::Inner(child) => {
                    node = child;
                }
                NodeRef::Leaf(leaf) => {
                    return Ok(Some((
                        leaf.keys.last().unwrap(),
                        leaf.children.last().unwrap(),
                    )));
                }
                NodeRef::Stored(stored_idx) => {
                    return self.last_key_value_stored(*stored_idx);
                }
            }
        }
    }

    fn last_key_value_stored(
        &self,
        mut stored_idx: Idx,
    ) -> Result<Option<(&S::Key, &S::Value)>, BTreeError> {
        loop {
            let node = self.data_store.get(stored_idx)?;

            match node {
                InnerOuter::Inner(node) => {
                    stored_idx = *node.children.last().unwrap();
                }
                InnerOuter::Outer(leaf) => {
                    return Ok(Some((
                        leaf.keys.last().unwrap(),
                        leaf.children.last().unwrap(),
                    )));
                }
            }
        }
    }

    pub fn range<K, R>(&self, range: R) -> Range<'_, S, K, R>
    where
        R: RangeBounds<K>,
        K: Borrow<S::Key>,
    {
        Range {
            txn: self,
            range,
            stack: SmallVec::new(),
            current_leaf: None,
            phantom: PhantomData,
        }
    }

    pub fn iter(&self) -> Iter<'_, S> {
        Iter {
            range: self.range(..),
        }
    }
}

impl<K: Ord + Clone + PortableHash, V: Clone + PortableHash, Db: DatabaseGet<K, V>>
    Transaction<SnapshotBuilder<K, V, Db>>
{
    pub fn new_snapshot_builder_txn(root: NodeHash, db: Db) -> Self {
        debug_assert!(EMPTY_TREE_ROOT_HASH == NodeHash::default());

        if root == EMPTY_TREE_ROOT_HASH {
            Self {
                data_store: SnapshotBuilder::new(root, db),
                current_root: None,
            }
        } else {
            Self {
                data_store: SnapshotBuilder::new(root, db),
                current_root: Some(NodeRef::Stored(0)),
            }
        }
    }

    /// Builds a snapshot of the tree before the transaction.
    /// The `Snapshot` is not a complete representation of the tree.
    /// The `Snapshot` only contains information about the parts of the tree touched by the transaction.
    /// Because of this, two `Snapshot`s of the same tree may not be equal if the transactions differ.
    ///
    /// Note: All operations including get affect the contents of the snapshot.
    #[inline]
    pub fn build_initial_snapshot(&self) -> Snapshot<K, V> {
        self.data_store.build_initial_snapshot()
    }
}

impl<'s, S: Store + AsRef<Snapshot<S::Key, S::Value>>> Transaction<&'s VerifiedSnapshot<S>> {
    /// Create a `Transaction` from a borrowed `VerifiedSnapshot`.
    #[inline]
    pub fn from_verified_snapshot_ref(snapshot: &'s VerifiedSnapshot<S>) -> Self {
        Transaction {
            current_root: snapshot.root_node_ref(),
            data_store: snapshot,
        }
    }
}

impl<S: Store + AsRef<Snapshot<S::Key, S::Value>>> Transaction<VerifiedSnapshot<S>> {
    #[inline]
    pub fn from_verified_snapshot_owned(snapshot: VerifiedSnapshot<S>) -> Self {
        Self {
            current_root: snapshot.root_node_ref(),
            data_store: snapshot,
        }
    }
}

impl<K: Clone + PortableHash + Ord, V: Clone + PortableHash, Db: DatabaseSet<K, V>>
    Transaction<SnapshotBuilder<K, V, Db>>
{
    #[inline]
    pub fn from_snapshot_builder(snapshot_builder: SnapshotBuilder<K, V, Db>) -> Self {
        Self {
            current_root: Some(NodeRef::Stored(0)),
            data_store: snapshot_builder,
        }
    }

    /// Write modified nodes to the database and return the root hash.
    /// Calling this method will write all modified nodes to the database.
    /// Calling this method again will rewrite the nodes to the database.
    ///
    /// Caching writes is the responsibility of the `DatabaseSet` implementation.
    ///
    /// Caller must ensure that the hasher is reset before calling this method.
    #[inline]
    pub fn commit(&self, hasher: &mut impl PortableHasher<32>) -> Result<NodeHash, BTreeError> {
        let on_modified_leaf = &mut |hash: &NodeHash, leaf: &LeafNode<K, V>| {
            self.data_store
                .db
                .set(hash, InnerOuter::Outer(leaf.clone()))
                .map_err(|e| BTreeError::from(e.to_string()))
        };

        let on_modified_branch =
            &mut |hash: &NodeHash,
                  node: &InnerNode<K, V>,
                  child_hashes: &ArrayVec<_, { BTREE_ORDER * 2 }>| {
                let node = InnerOuter::Inner(Node {
                    keys: node.keys.clone(),
                    children: ArrayVec::from_iter(child_hashes.iter().cloned()),
                });

                self.data_store
                    .db
                    .set(hash, node)
                    .map_err(|e| BTreeError::from(e.to_string()))
            };

        self.calc_root_hash_inner(hasher, on_modified_leaf, on_modified_branch)
    }
}

pub(crate) enum Insert_<K, V> {
    Inserted,
    Replaced(V),
    /// The key should be the right node's first key.
    SplitNode {
        middle_key: K,
        right_node: Arc<InnerNode<K, V>>,
    },
}

pub enum Remove<S: Store> {
    NotPresent,
    Removed(S::Value),
    /// Removing caused a node to be smaller than the minimum size.
    Underflow(S::Value),
}

pub struct Iter<'s, S: Store> {
    range: Range<'s, S, S::Key, RangeFull>,
}

impl<'s, S: Store> Iterator for Iter<'s, S> {
    type Item = Result<(&'s S::Key, &'s S::Value), BTreeError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next()
    }
}

enum StackItem<'s, S: Store> {
    Node(&'s InnerNode<S::Key, S::Value>),
    Stored(&'s InnerNodeSnapshot<S::Key>),
}

/// An iterator over a range of keys in the tree.
pub struct Range<'s, S: Store, K: Borrow<S::Key>, R: RangeBounds<K>> {
    txn: &'s Transaction<S>,
    /// The range of keys to iterate over.
    range: R,
    /// The stack of nodes we have visited.
    /// (child index descended into, parent node)
    stack: SmallVec<[(usize, StackItem<'s, S>); 32]>,
    /// The index key to be visited in the current leaf node,
    /// and current leaf node we are iterating over.
    // TODO: remove the Arc and cloning with the redesign of Store
    current_leaf: Option<(usize, &'s LeafNode<S::Key, S::Value>)>,
    phantom: PhantomData<K>,
}

#[allow(clippy::needless_lifetimes)]
impl<'s, S: Store, K: Borrow<S::Key>, R: RangeBounds<K>> Range<'s, S, K, R> {
    fn bound_to_idx<V>(&self, node: &'s Node<S::Key, V>) -> usize {
        match self.range.start_bound() {
            Bound::Included(key) => node.keys.binary_search(key.borrow()).unwrap_or_else(|i| i),
            Bound::Excluded(key) => {
                let start = key.borrow();
                match node.keys.binary_search(start) {
                    // If the excluded starting key is in an inner node
                    // the all greater keys must be to the right of it
                    Ok(i) => i + 1,
                    Err(i) => i,
                }
            }
            Bound::Unbounded => 0,
        }
    }

    fn setup_stack(&mut self) -> Result<(), BTreeError> {
        debug_assert!(self.stack.is_empty());
        debug_assert!(self.current_leaf.is_none());

        match &self.txn.current_root {
            None => Ok(()),
            Some(NodeRef::Inner(node)) => {
                let mut parent = node;
                loop {
                    let idx = self.bound_to_idx(parent);
                    self.stack.push((idx, StackItem::Node(parent)));

                    match &parent.children[idx] {
                        NodeRef::Inner(child) => {
                            parent = child;
                        }
                        NodeRef::Leaf(leaf) => {
                            let idx = self.bound_to_idx(leaf);
                            self.current_leaf = Some((idx, leaf));
                            return Ok(());
                        }
                        NodeRef::Stored(stored_idx) => {
                            self.setup_stack_stored(*stored_idx)?;
                            return Ok(());
                        }
                    }
                }
            }
            Some(NodeRef::Leaf(leaf)) => {
                let idx = self.bound_to_idx(leaf);
                self.current_leaf = Some((idx, leaf));
                Ok(())
            }
            Some(NodeRef::Stored(stored_idx)) => self.setup_stack_stored(*stored_idx),
        }
    }

    fn setup_stack_stored(&mut self, mut stored_idx: Idx) -> Result<(), BTreeError> {
        loop {
            let Ok(node) = self.txn.data_store.get(stored_idx) else {
                return Err(BTreeError::from("stored node not found"));
            };

            match node {
                InnerOuter::Inner(node) => {
                    let idx = self.bound_to_idx(node);
                    self.stack.push((idx, StackItem::Stored(node)));
                    stored_idx = node.children[idx];
                }
                InnerOuter::Outer(leaf) => {
                    let idx = self.bound_to_idx(leaf);
                    self.current_leaf = Some((idx, leaf));
                    return Ok(());
                }
            }
        }
    }
}

impl<'s, S: Store, K: Borrow<S::Key>, R: RangeBounds<K>> Iterator for Range<'s, S, K, R> {
    type Item = Result<(&'s S::Key, &'s S::Value), BTreeError>;

    // The question mark makes it harder to read since it's a Option<Result<...>>
    #[allow(clippy::question_mark)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.txn.current_root.is_none() {
            return None;
        }

        // this is a mess, but it's going to be rewritten anyway
        if self.current_leaf.is_none() {
            match (self.range.start_bound(), self.range.end_bound()) {
                (Bound::Included(start), Bound::Included(end)) if start.borrow() > end.borrow() => {
                    return None
                }
                (Bound::Excluded(start), Bound::Excluded(end))
                    if start.borrow() >= end.borrow() =>
                {
                    return None
                }
                _ => {}
            }

            if let Err(e) = self.setup_stack() {
                return Some(Err(e));
            }
        }

        let (idx, leaf) = self.current_leaf.as_mut().unwrap();
        if let Some(child_key) = leaf.keys.get(*idx) {
            match self.range.end_bound() {
                Bound::Included(key) if child_key <= key.borrow() => {
                    let child = &leaf.children[*idx];
                    *idx += 1;
                    return Some(Ok((child_key, child)));
                }
                Bound::Excluded(key) if child_key < key.borrow() => {
                    let child = &leaf.children[*idx];
                    *idx += 1;
                    return Some(Ok((child_key, child)));
                }
                Bound::Unbounded => {
                    let child = &leaf.children[*idx];
                    *idx += 1;

                    return Some(Ok((child_key, child)));
                }
                _ => return None,
            }
        } else {
            // We have reached the end of this leaf node.
            // Pop until we find a parent node with a next child index.
            loop {
                let Some((parent_idx, parent_node)) = self.stack.last_mut() else {
                    return None;
                };

                let children_len = match parent_node {
                    StackItem::Node(node) => node.children.len(),
                    StackItem::Stored(node) => node.children.len(),
                };

                if *parent_idx + 1 == children_len {
                    match self.stack.pop() {
                        Some(_) => continue,
                        None => return None,
                    }
                } else {
                    *parent_idx += 1;
                    break;
                }
            }
        }

        loop {
            let Some((parent_idx, parent_node)) = self.stack.last() else {
                return None;
            };

            match parent_node {
                StackItem::Node(node) => match &node.children[*parent_idx] {
                    NodeRef::Inner(child) => {
                        self.stack.push((0, StackItem::Node(child)));
                    }
                    NodeRef::Leaf(leaf) => {
                        self.current_leaf = Some((0, leaf));
                        return self.next();
                    }
                    NodeRef::Stored(stored_idx) => {
                        let Ok(node) = self.txn.data_store.get(*stored_idx) else {
                            return Some(Err(BTreeError::from("stored node not found")));
                        };

                        match node {
                            InnerOuter::Inner(node) => {
                                self.stack.push((0, StackItem::Stored(node)));
                            }
                            InnerOuter::Outer(leaf) => {
                                self.current_leaf = Some((0, leaf));
                                return self.next();
                            }
                        }
                    }
                },
                StackItem::Stored(node) => {
                    match self.txn.data_store.get(node.children[*parent_idx]) {
                        Ok(InnerOuter::Inner(node)) => {
                            self.stack.push((0, StackItem::Stored(node)));
                        }
                        Ok(InnerOuter::Outer(leaf)) => {
                            self.current_leaf = Some((0, leaf));
                            return self.next();
                        }
                        Err(e) => {
                            return Some(Err(BTreeError::from(e.to_string())));
                        }
                    }
                }
            }
        }
    }
}

// TODO merge these tests with the tests in btree-test-utils
#[cfg(test)]
mod test {
    use std::ops::RangeBounds;

    use alloc::collections::btree_map::BTreeMap;

    use proptest::prelude::*;

    use crate::{db::MemoryDb, transaction::Transaction, Store};

    #[derive(Clone, Debug)]
    enum Op {
        Insert(u32, u32),
        Get(u32),
        Delete(u32),
        GetFirstKeyValue,
        GetLastKeyValue,
        IterAll,
        IterRange(Option<u32>, Option<u32>),
    }

    fn iter_test<S: Store<Key = u32, Value = u32>>(
        range: impl RangeBounds<u32> + Clone,
        txn: &Transaction<S>,
        std: &BTreeMap<u32, u32>,
    ) {
        let mut txn_iter = txn.range(range.clone()).enumerate();

        for res_std in std.range(range).enumerate() {
            let (i_txn, res_txn) = txn_iter.next().expect("to few elements");
            let res_txn = res_txn.expect("txn error");

            assert_eq!((i_txn, res_txn), res_std);
        }
    }

    fn run_operations(mut operations: Vec<Op>) {
        let mut txn_btree =
            Transaction::new_snapshot_builder_txn(Default::default(), MemoryDb::default());
        let mut std_btree = BTreeMap::new();

        operations.push(Op::IterAll);

        for op in operations {
            match op {
                Op::Insert(k, v) => {
                    let res_txn = txn_btree.insert(k, v).unwrap();
                    let res_std = std_btree.insert(k, v);
                    assert_eq!(res_txn, res_std,
                        "insert failed for key: {}, value: {}, merkle btree returned {:?}, btreemap returned {:?}",
                        k, v, res_txn, res_std);
                }
                Op::Get(k) => {
                    let res_txn = txn_btree.get(&k).unwrap().cloned();
                    let res_std = std_btree.get(&k).cloned();
                    assert_eq!(res_txn, res_std,
                        "get failed for key: {}, merkle btree returned {:?}, btreemap returned {:?}",
                        k, res_txn, res_std);
                }
                Op::Delete(k) => {
                    let res_txn = txn_btree.remove(&k).unwrap();
                    let res_std = std_btree.remove(&k);
                    assert_eq!(res_txn, res_std,
                        "delete failed for key: {}, merkle btree returned {:?}, btreemap returned {:?}",
                        k, res_txn, res_std);
                }
                Op::GetFirstKeyValue => {
                    let res_txn = txn_btree.first_key_value().unwrap();
                    let res_std = std_btree.first_key_value();
                    assert_eq!(res_txn, res_std,
                        "get first key value failed, merkle btree returned {:?}, btreemap returned {:?}",
                        res_txn, res_std);
                }
                Op::GetLastKeyValue => {
                    let res_txn = txn_btree.last_key_value().unwrap();
                    let res_std = std_btree.last_key_value();
                    assert_eq!(res_txn, res_std,
                        "get last key value failed, merkle btree returned {:?}, btreemap returned {:?}",
                        res_txn, res_std);
                }
                Op::IterAll => {
                    let mut txn_iter = txn_btree.iter().enumerate();

                    for res_std in std_btree.iter().enumerate() {
                        let (i_txn, res_txn) = txn_iter.next().expect("to few elements");
                        let res_txn = res_txn.expect("txn error");

                        assert_eq!((i_txn, res_txn), res_std);
                    }
                }
                Op::IterRange(start, end) => {
                    if let (Some(start), Some(end)) = (start, end) {
                        if start > end {
                            assert!(txn_btree.range(start..end).next().is_none());

                            return;
                        }
                    }

                    match (start, end) {
                        (Some(start), Some(end)) => iter_test(start..end, &txn_btree, &std_btree),
                        (Some(start), None) => iter_test(start.., &txn_btree, &std_btree),
                        (None, Some(end)) => iter_test(..end, &txn_btree, &std_btree),
                        (None, None) => iter_test(.., &txn_btree, &std_btree),
                    };
                }
            }
        }
    }

    #[test]
    fn test_hardcoded_1_insert() {
        let operations = vec![Op::Insert(0, 0), Op::Insert(1, 0), Op::Insert(2, 0)];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_2_insert() {
        let operations = vec![
            Op::Insert(15, 0),
            Op::Insert(16, 0),
            Op::Insert(17, 0),
            Op::Insert(18, 0),
            Op::Insert(19, 0),
            Op::Insert(20, 0),
            Op::Insert(0, 0),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_3_insert() {
        let operations = vec![
            Op::Insert(0, 0),
            Op::Insert(1, 0),
            Op::Insert(1551649896, 0),
            Op::Insert(2, 0),
            Op::Insert(1551649897, 0),
            Op::Insert(1551649898, 0),
            Op::Insert(3, 0),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_4_insert() {
        let operations = vec![
            Op::Insert(1551649896, 0),
            Op::Insert(1551649897, 0),
            Op::Insert(0, 0),
            Op::Insert(0, 0), // Duplicate insert
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_5_insert_insert() {
        let operations = vec![Op::Insert(0, 0), Op::Insert(1, 0), Op::Insert(1, 0)];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_5_insert_delete() {
        let operations = vec![Op::Insert(0, 0), Op::Insert(1, 0), Op::Delete(1)];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_6_insert_delete_get() {
        let operations = vec![
            Op::Insert(1, 10),
            Op::Insert(2, 20),
            Op::Insert(3, 30),
            Op::Delete(2),
            Op::Get(2),
            Op::Delete(1),
            Op::Delete(3),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_7_insert_delete() {
        let operations = vec![
            Op::Insert(0, 0),
            Op::Insert(1, 0),
            Op::Insert(2, 0),
            Op::Insert(3, 0),
            Op::Insert(257, 0),
            Op::Insert(63, 0),
            Op::Insert(4, 0),
            Op::Insert(5, 0),
            Op::Insert(64, 0),
            Op::Insert(65, 0),
            Op::Insert(66, 0),
            Op::Insert(67, 0),
            Op::Insert(6, 0),
            Op::Delete(257),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_8_insert_delete_get() {
        let operations = vec![
            Op::Insert(100, 1000),
            Op::Insert(200, 2000),
            Op::Insert(300, 3000),
            Op::Delete(200),
            Op::Get(200),
            Op::Insert(400, 4000),
            Op::Insert(500, 5000),
            Op::Delete(100),
            Op::Get(100),
            Op::Delete(300),
            Op::Get(300),
            Op::Delete(400),
            Op::Get(400),
            Op::Delete(500),
            Op::Get(500),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_9_insert_delete() {
        let operations = vec![
            Op::Insert(887, 0),
            Op::Insert(0, 0),
            Op::Insert(1, 0),
            Op::Insert(119, 0),
            Op::Insert(2, 0),
            Op::Delete(887),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_10_insert_delete() {
        let operations = vec![
            Op::Insert(0, 0),
            Op::Insert(0, 0),
            Op::Insert(422, 0),
            Op::Insert(422, 0),
            Op::Insert(423, 0),
            Op::Insert(424, 0),
            Op::Insert(425, 0),
            Op::Delete(422),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_11_insert_delete() {
        let operations = vec![
            Op::Insert(0, 0),
            Op::Insert(1, 0),
            Op::Insert(2, 0),
            Op::Insert(3, 0),
            Op::Insert(4, 0),
            Op::Insert(580, 0),
            Op::Insert(581, 0),
            Op::Insert(582, 0),
            Op::Insert(5, 0),
            Op::Insert(527, 0),
            Op::Insert(6, 0),
            Op::Delete(527),
            Op::Insert(527, 0),
            Op::Insert(6, 0),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_12_insert_delete() {
        let operations = vec![
            Op::Insert(0, 0),
            Op::Insert(0, 0),
            Op::Insert(513, 0),
            Op::Insert(0, 0),
            Op::Insert(1, 0),
            Op::Insert(2, 0),
            Op::Insert(617, 0),
            Op::Insert(618, 0),
            Op::Insert(0, 0),
            Op::Insert(0, 0),
            Op::Insert(514, 0),
            Op::Insert(467, 0),
            Op::Insert(3, 0),
            Op::Insert(4, 0),
            Op::Insert(467, 0),
            Op::Insert(512, 0),
            Op::Insert(505, 0),
            Op::Insert(5, 0),
            Op::Delete(512),
            Op::Delete(505),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_13_first_last_key_value() {
        let operations = vec![
            Op::GetFirstKeyValue,
            Op::GetLastKeyValue,
            Op::Insert(100, 1),
            Op::GetFirstKeyValue,
            Op::GetLastKeyValue,
            Op::Insert(50, 2),
            Op::GetFirstKeyValue,
            Op::GetLastKeyValue,
            Op::Insert(150, 3),
            Op::GetFirstKeyValue,
            Op::GetLastKeyValue,
            Op::Delete(50),
            Op::GetFirstKeyValue,
            Op::GetLastKeyValue,
            Op::Delete(150),
            Op::GetFirstKeyValue,
            Op::GetLastKeyValue,
            Op::Delete(100),
            Op::GetFirstKeyValue,
            Op::GetLastKeyValue,
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_14_insert_delete() {
        let operations = vec![
            Op::Insert(4182, 0),
            Op::Insert(0, 0),
            Op::Insert(1, 0),
            Op::Insert(0, 0),
            Op::Insert(2197, 0),
            Op::Insert(0, 0),
            Op::Insert(2, 0),
            Op::Insert(2198, 0),
            Op::Insert(4126, 0),
            Op::Insert(4126, 0),
            Op::Insert(4126, 0),
            Op::Insert(4126, 0),
            Op::Insert(4127, 0),
            Op::Insert(3953, 0),
            Op::Insert(3952, 0),
            Op::Insert(3700, 0),
            Op::Insert(3947, 0),
            Op::Insert(2199, 0),
            Op::Delete(3947),
            Op::Delete(3952),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_15_insert_delete_first() {
        let operations = vec![
            Op::Insert(23, 0),
            Op::Insert(8, 0),
            Op::Insert(24, 0),
            Op::Insert(25, 0),
            Op::Insert(26, 0),
            Op::Delete(8),
            Op::Delete(23),
            Op::GetFirstKeyValue,
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_16_insert_delete_last_key() {
        let operations = vec![
            Op::Insert(3, 0),
            Op::Insert(9952, 0),
            Op::Insert(9970, 0),
            Op::Insert(0, 0),
            Op::Insert(9982, 0),
            Op::Insert(4, 0),
            Op::Insert(5, 0),
            Op::Delete(9982),
            Op::Delete(9970),
            Op::GetLastKeyValue,
            Op::IterRange(None, None),
            Op::IterRange(Some(0), None),
            Op::IterRange(None, Some(0)),
            Op::IterRange(Some(0), Some(0)),
            Op::IterRange(Some(1), Some(1)),
            Op::IterRange(Some(1), Some(10)),
            Op::IterRange(Some(10), None),
        ];
        run_operations(operations);
    }

    #[test]
    fn test_hardcoded_17_insert_delete_last_key_iter() {
        let operations = vec![Op::Insert(1, 0), Op::Insert(2, 0), Op::IterAll];
        run_operations(operations);
    }

    proptest! {
        #[test]
        fn test_merkle_btree_txn_against_btreemap(operations in proptest::collection::vec(
            prop_oneof![
                100 => (0..10000u32, any::<u32>()).prop_map(|(k, v)| Op::Insert(k, v)),
                50 => (0..10000u32).prop_map(Op::Get),
                50 => (0..10000u32).prop_map(Op::Delete),
                20 => Just(Op::GetFirstKeyValue),
                20 => Just(Op::GetLastKeyValue),
                10 => Just(Op::IterAll),
                15 => (proptest::option::of(0..10000u32), proptest::option::of(0..10000u32)).prop_map(|(start, end)| Op::IterRange(start, end)),
            ],
            1..10_000
        )) {
            run_operations(operations);
        }
    }
}
