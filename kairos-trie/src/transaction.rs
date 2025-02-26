pub(crate) mod nodes;

use alloc::borrow::Cow;
use alloc::{boxed::Box, format};
use core::{iter, mem};

use crate::stored::merkle::VerifiedSnapshot;
use crate::stored::DatabaseGet;
use crate::{stored, KeyHash, NodeHash, PortableHash, PortableHasher};
use crate::{
    stored::{
        merkle::{Snapshot, SnapshotBuilder},
        DatabaseSet, Store,
    },
    TrieError,
};

use self::nodes::{
    Branch, KeyPosition, KeyPositionAdjacent, Leaf, Node, NodeRef, StoredLeafRef, TrieRoot,
};

pub struct Transaction<S: Store> {
    pub data_store: S,
    current_root: TrieRoot<NodeRef<S::Value>>,
}

impl<Db: DatabaseSet<V>, V: Clone + PortableHash> Transaction<SnapshotBuilder<Db, V>> {
    /// Write modified nodes to the database and return the root hash.
    /// Calling this method will write all modified nodes to the database.
    /// Calling this method again will rewrite the nodes to the database.
    ///
    /// Caching writes is the responsibility of the `DatabaseSet` implementation.
    ///
    /// Caller must ensure that the hasher is reset before calling this method.
    #[inline]
    pub fn commit(
        &self,
        hasher: &mut impl PortableHasher<32>,
    ) -> Result<TrieRoot<NodeHash>, TrieError> {
        let store_modified_branch =
            &mut |hash: &NodeHash, branch: &Branch<NodeRef<V>>, left: NodeHash, right: NodeHash| {
                let branch = Branch {
                    left,
                    right,
                    mask: branch.mask,
                    prior_word: branch.prior_word,
                    prefix: branch.prefix.clone(),
                };

                self.data_store
                    .db()
                    .set(*hash, Node::Branch(branch))
                    .map_err(|e| format!("Error writing branch {hash} to database: {e}").into())
            };

        let store_modified_leaf = &mut |hash: &NodeHash, leaf: &Leaf<V>| {
            self.data_store
                .db()
                .set(*hash, Node::Leaf(leaf.clone()))
                .map_err(|e| format!("Error writing leaf {hash} to database: {e}").into())
        };

        let root_hash =
            self.calc_root_hash_inner(hasher, store_modified_branch, store_modified_leaf)?;
        Ok(root_hash)
    }
}

impl<S: Store> Transaction<S> {
    /// Caller must ensure that the hasher is reset before calling this method.
    #[inline]
    pub fn calc_root_hash_inner(
        &self,
        hasher: &mut impl PortableHasher<32>,
        on_modified_branch: &mut impl FnMut(
            &NodeHash,
            &Branch<NodeRef<S::Value>>,
            NodeHash,
            NodeHash,
        ) -> Result<(), TrieError>,
        on_modified_leaf: &mut impl FnMut(&NodeHash, &Leaf<S::Value>) -> Result<(), TrieError>,
    ) -> Result<TrieRoot<NodeHash>, TrieError> {
        let root_hash = match &self.current_root {
            TrieRoot::Empty => return Ok(TrieRoot::Empty),
            TrieRoot::Node(node_ref) => Self::calc_root_hash_node(
                hasher,
                &self.data_store,
                node_ref,
                on_modified_leaf,
                on_modified_branch,
            )?,
        };

        Ok(TrieRoot::Node(root_hash))
    }

    /// Calculate the root hash of the trie.
    ///
    /// Caller must ensure that the hasher is reset before calling this method.
    #[inline]
    pub fn calc_root_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
    ) -> Result<TrieRoot<NodeHash>, TrieError> {
        self.calc_root_hash_inner(hasher, &mut |_, _, _, _| Ok(()), &mut |_, _| Ok(()))
    }

    #[inline]
    fn calc_root_hash_node(
        hasher: &mut impl PortableHasher<32>,
        data_store: &S,
        node_ref: &NodeRef<S::Value>,
        on_modified_leaf: &mut impl FnMut(&NodeHash, &Leaf<S::Value>) -> Result<(), TrieError>,
        on_modified_branch: &mut impl FnMut(
            &NodeHash,
            &Branch<NodeRef<S::Value>>,
            NodeHash,
            NodeHash,
        ) -> Result<(), TrieError>,
    ) -> Result<NodeHash, TrieError> {
        // TODO use a stack instead of recursion
        match node_ref {
            NodeRef::ModBranch(branch) => {
                let left = Self::calc_root_hash_node(
                    hasher,
                    data_store,
                    &branch.left,
                    on_modified_leaf,
                    on_modified_branch,
                )?;
                let right = Self::calc_root_hash_node(
                    hasher,
                    data_store,
                    &branch.right,
                    on_modified_leaf,
                    on_modified_branch,
                )?;

                let hash = branch.hash_branch(hasher, &left, &right);
                on_modified_branch(&hash, branch, left, right)?;
                Ok(hash)
            }
            NodeRef::ModLeaf(leaf) => {
                let hash = leaf.hash_leaf(hasher);

                on_modified_leaf(&hash, leaf)?;
                Ok(hash)
            }
            NodeRef::Stored(stored_idx) => data_store
                .calc_subtree_hash(hasher, *stored_idx)
                .map_err(|e| {
                    format!(
                        "Error in `calc_root_hash_node`: {e} at {file}:{line}:{column}",
                        file = file!(),
                        line = line!(),
                        column = column!()
                    )
                    .into()
                }),
        }
    }
}

impl<Db: 'static + DatabaseGet<V>, V: Clone + PortableHash> Transaction<SnapshotBuilder<Db, V>> {
    /// This method is like standard `Transaction::get` but won't affect the Transaction or any Snapshot built from it.
    /// You should use this method to check precondition before modifying the Transaction.
    ///
    /// If you use the normal get, and then don't modify the trie,
    /// you will still have added the merkle path to the snapshot, which is unnecessary.
    #[inline]
    pub fn get_exclude_from_txn<'s>(
        &'s self,
        key_hash: &KeyHash,
    ) -> Result<Option<Cow<'s, V>>, TrieError> {
        match &self.current_root {
            TrieRoot::Empty => Ok(None),
            TrieRoot::Node(node_ref) => {
                Self::get_node_exclude_from_txn(&self.data_store, node_ref, key_hash)
            }
        }
    }

    #[inline]
    fn get_node_exclude_from_txn<'root, 's: 'root>(
        data_store: &'s SnapshotBuilder<Db, V>,
        mut node_ref: &'root NodeRef<V>,
        key_hash: &KeyHash,
    ) -> Result<Option<Cow<'root, V>>, TrieError> {
        loop {
            match node_ref {
                NodeRef::ModBranch(branch) => match branch.key_position(key_hash) {
                    KeyPosition::Left => node_ref = &branch.left,
                    KeyPosition::Right => node_ref = &branch.right,
                    KeyPosition::Adjacent(_) => return Ok(None),
                },
                NodeRef::ModLeaf(leaf) => {
                    if leaf.key_hash == *key_hash {
                        return Ok(Some(Cow::Borrowed(&leaf.value)));
                    } else {
                        return Ok(None);
                    }
                }
                NodeRef::Stored(stored_idx) => {
                    let stored_hash = data_store
                        .get_node_hash(*stored_idx)
                        .map_err(|e| format!("Error in `get_node_exclude_from_txn`: {e}"))?;

                    return Self::get_stored_node_exclude_from_txn(
                        data_store.db(),
                        stored_hash,
                        key_hash,
                    )
                    .map(|v| v.map(Cow::Owned));
                }
            }
        }
    }

    #[inline]
    fn get_stored_node_exclude_from_txn(
        database: &Db,
        mut stored_hash: NodeHash,
        key_hash: &KeyHash,
    ) -> Result<Option<V>, TrieError> {
        loop {
            let node = database
                .get(&stored_hash)
                .map_err(|e| format!("Error in `get_stored_node_exclude_from_txn`: {e}"))?;

            match node {
                Node::Branch(branch) => match branch.key_position(key_hash) {
                    KeyPosition::Left => stored_hash = branch.left,
                    KeyPosition::Right => stored_hash = branch.right,
                    KeyPosition::Adjacent(_) => return Ok(None),
                },
                Node::Leaf(leaf) => {
                    if leaf.key_hash == *key_hash {
                        return Ok(Some(leaf.value));
                    } else {
                        return Ok(None);
                    }
                }
            }
        }
    }
}

impl<S: Store> Transaction<S> {
    #[inline]
    pub fn get(&self, key_hash: &KeyHash) -> Result<Option<&S::Value>, TrieError> {
        match &self.current_root {
            TrieRoot::Empty => Ok(None),
            TrieRoot::Node(node_ref) => Self::get_node(&self.data_store, node_ref, key_hash),
        }
    }

    #[inline]
    fn get_node<'root, 's: 'root>(
        data_store: &'s S,
        mut node_ref: &'root NodeRef<S::Value>,
        key_hash: &KeyHash,
    ) -> Result<Option<&'root S::Value>, TrieError> {
        let mut last_bit_idx = 0;
        let mut last_word_idx = 0;
        loop {
            match node_ref {
                NodeRef::ModBranch(branch) => {
                    debug_assert!(branch.mask.word_idx() >= last_word_idx);
                    debug_assert!(branch.mask.bit_idx >= last_bit_idx);
                    debug_assert_eq!(
                        branch.prefix.len(),
                        branch.mask.word_idx().saturating_sub(last_word_idx + 1)
                    );

                    last_bit_idx = branch.mask.bit_idx;
                    last_word_idx = branch.mask.word_idx();

                    match branch.key_position(key_hash) {
                        KeyPosition::Left => node_ref = &branch.left,
                        KeyPosition::Right => node_ref = &branch.right,
                        KeyPosition::Adjacent(_) => return Ok(None),
                    }
                }
                NodeRef::ModLeaf(leaf) => {
                    if leaf.key_hash == *key_hash {
                        return Ok(Some(&leaf.value));
                    } else {
                        return Ok(None);
                    }
                }
                NodeRef::Stored(stored_idx) => {
                    return Self::get_stored_node(data_store, *stored_idx, key_hash);
                }
            }
        }
    }

    #[inline]
    fn get_stored_node<'s>(
        data_store: &'s S,
        mut stored_idx: stored::Idx,
        key_hash: &KeyHash,
    ) -> Result<Option<&'s S::Value>, TrieError> {
        loop {
            let node = data_store
                .get_node(stored_idx)
                .map_err(|e| format!("Error in `get_stored_node`: {e}"))?;

            match node {
                Node::Branch(branch) => match branch.key_position(key_hash) {
                    KeyPosition::Left => stored_idx = branch.left,
                    KeyPosition::Right => stored_idx = branch.right,
                    KeyPosition::Adjacent(_) => return Ok(None),
                },
                Node::Leaf(leaf) => {
                    if leaf.key_hash == *key_hash {
                        break;
                    } else {
                        return Ok(None);
                    }
                }
            }
        }

        match data_store
            .get_node(stored_idx)
            .map_err(|e| format!("Error in `get_stored_node`: {e}"))?
        {
            Node::Leaf(leaf) => Ok(Some(&leaf.value)),
            _ => unreachable!("Prior loop only breaks on a leaf"),
        }
    }

    #[inline]
    pub fn insert(&mut self, key_hash: &KeyHash, value: S::Value) -> Result<(), TrieError> {
        match &mut self.current_root {
            TrieRoot::Empty => {
                self.current_root = TrieRoot::Node(NodeRef::ModLeaf(Box::new(Leaf {
                    key_hash: *key_hash,
                    value,
                })));
                Ok(())
            }
            TrieRoot::Node(node_ref) => {
                Self::insert_node(&mut self.data_store, node_ref, key_hash, value)
            }
        }
    }

    #[inline(always)]
    fn insert_node<'root, 's: 'root>(
        data_store: &'s mut S,
        mut node_ref: &'root mut NodeRef<S::Value>,
        key_hash: &KeyHash,
        value: S::Value,
    ) -> Result<(), TrieError> {
        let mut last_bit_idx = 0;
        let mut last_word_idx = 0;

        loop {
            match node_ref {
                NodeRef::ModBranch(branch) => {
                    debug_assert!(branch.mask.word_idx() >= last_word_idx);
                    debug_assert!(branch.mask.bit_idx >= last_bit_idx);
                    debug_assert_eq!(
                        branch.prefix.len(),
                        branch.mask.word_idx().saturating_sub(last_word_idx + 1)
                    );

                    last_bit_idx = branch.mask.bit_idx;
                    last_word_idx = branch.mask.word_idx();

                    match branch.key_position(key_hash) {
                        KeyPosition::Left => {
                            node_ref = &mut branch.left;
                            continue;
                        }
                        KeyPosition::Right => {
                            node_ref = &mut branch.right;
                            continue;
                        }
                        KeyPosition::Adjacent(pos) => {
                            branch.new_adjacent_leaf(
                                pos,
                                Box::new(Leaf {
                                    key_hash: *key_hash,
                                    value,
                                }),
                            );

                            return Ok(());
                        }
                    }
                }
                NodeRef::ModLeaf(leaf) => {
                    if leaf.key_hash == *key_hash {
                        leaf.value = value;

                        return Ok(());
                    } else {
                        let old_leaf = mem::replace(node_ref, NodeRef::temp_null_stored());
                        let NodeRef::ModLeaf(old_leaf) = old_leaf else {
                            unreachable!("We just matched a ModLeaf");
                        };
                        let new_leaf = Box::new(Leaf {
                            key_hash: *key_hash,
                            value,
                        });

                        let (new_branch, _) =
                            Branch::new_from_leafs(last_word_idx, old_leaf, new_leaf);

                        *node_ref = NodeRef::ModBranch(new_branch);
                        return Ok(());
                    }
                }
                NodeRef::Stored(stored_idx) => {
                    let new_node = data_store.get_node(*stored_idx).map_err(|e| {
                        format!("Error at `{}:{}:{}`: `{e}`", file!(), line!(), column!())
                    })?;
                    match new_node {
                        Node::Branch(new_branch) => {
                            *node_ref = NodeRef::ModBranch(Box::new(Branch {
                                left: NodeRef::Stored(new_branch.left),
                                right: NodeRef::Stored(new_branch.right),
                                mask: new_branch.mask,
                                prior_word: new_branch.prior_word,
                                prefix: new_branch.prefix.clone(),
                            }));

                            continue;
                        }
                        Node::Leaf(leaf) => {
                            if leaf.key_hash == *key_hash {
                                *node_ref = NodeRef::ModLeaf(Box::new(Leaf {
                                    key_hash: *key_hash,
                                    value,
                                }));

                                return Ok(());
                            } else {
                                let (new_branch, _) = Branch::new_from_leafs(
                                    last_word_idx,
                                    StoredLeafRef::new(leaf, *stored_idx),
                                    Box::new(Leaf {
                                        key_hash: *key_hash,
                                        value,
                                    }),
                                );

                                *node_ref = NodeRef::ModBranch(new_branch);
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }
    }

    #[inline]
    pub fn remove(&mut self, key_hash: &KeyHash) -> Result<Option<S::Value>, TrieError> {
        match &mut self.current_root {
            TrieRoot::Empty => Ok(None),
            TrieRoot::Node(node_ref @ NodeRef::ModBranch(_)) => {
                Self::remove_node(&self.data_store, node_ref, key_hash)
            }
            TrieRoot::Node(NodeRef::Stored(stored_idx)) => {
                let stored_hash = self.data_store.get_node(*stored_idx).map_err(|e| {
                    format!(
                        "Error in `remove` at {file}:{line}:{column}: could not get stored node: {e}",
                        file = file!(),
                        line = line!(),
                        column = column!(),
                    )
                })?;

                match stored_hash {
                    Node::Branch(branch) => {
                        self.current_root = TrieRoot::Node(NodeRef::ModBranch(Box::new(
                            Branch::from_stored(branch),
                        )));

                        let TrieRoot::Node(node_ref) = &mut self.current_root else {
                            unreachable!("We just set the root to a ModBranch");
                        };
                        Self::remove_node(&self.data_store, node_ref, key_hash)
                    }
                    Node::Leaf(leaf) => {
                        if leaf.key_hash == *key_hash {
                            self.current_root = TrieRoot::Empty;
                            Ok(Some(leaf.value.clone()))
                        } else {
                            Ok(None)
                        }
                    }
                }
            }
            TrieRoot::Node(NodeRef::ModLeaf(leaf)) => {
                if leaf.key_hash == *key_hash {
                    let TrieRoot::Node(NodeRef::ModLeaf(leaf)) =
                        mem::replace(&mut self.current_root, TrieRoot::Empty)
                    else {
                        unreachable!("We just matched a ModLeaf");
                    };

                    Ok(Some(leaf.value))
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Remove a leaf from the trie returning the value if it exists.
    ///
    /// Caller must ensure that this method is never called with parent_node_ref pointing to a leaf or a stored leaf.
    #[inline(always)]
    fn remove_node(
        data_store: &S,
        mut parent_node_ref: &mut NodeRef<S::Value>,
        key_hash: &KeyHash,
    ) -> Result<Option<S::Value>, TrieError> {
        let mut grandparent_word_idx = 0usize;

        // This label break is sadly necessary to satisfy the borrow checker.
        // We only break to this label if we have found a leaf with an equal key_hash to remove.
        let (parent_branch, leaf_key_position) = 'leaf: {
            loop {
                let key_position = match parent_node_ref {
                    NodeRef::ModLeaf(_) => unreachable!("A leaf should never be a parent"),
                    NodeRef::Stored(stored_idx) => {
                        // This checks if a leaf with the key_hash under this stored node.
                        // We perform this extra partial traversal to avoid marking nodes as modified unnecessarily.
                        // Future work may remove this partial traversal by tracking node status in each node.
                        if Self::get_stored_node(data_store, *stored_idx, key_hash)?.is_none() {
                            return Ok(None);
                        };

                        let node = data_store.get_node(*stored_idx).map_err(|e| {
                        format!(
                            "Error in `remove_node` at {file}:{line}:{column}: could not get stored node: {e}",
                            file = file!(),
                            line = line!(),
                            column = column!(),
                        )
                    })?;

                        match node {
                            Node::Branch(branch) => {
                                *parent_node_ref =
                                    NodeRef::ModBranch(Box::new(Branch::from_stored(branch)));

                                continue;
                            }
                            Node::Leaf(_) => {
                                unreachable!("A stored leaf should never be a parent");
                            }
                        }
                    }
                    NodeRef::ModBranch(parent_branch) => {
                        let key_position = parent_branch.key_position(key_hash);
                        let matched_child = match key_position {
                            KeyPosition::Left => &mut parent_branch.left,
                            KeyPosition::Right => &mut parent_branch.right,
                            KeyPosition::Adjacent(_) => return Ok(None),
                        };

                        match matched_child {
                            NodeRef::Stored(stored_idx) => {
                                if Self::get_stored_node(data_store, *stored_idx, key_hash)?
                                    .is_none()
                                {
                                    return Ok(None);
                                };

                                let node = data_store.get_node(*stored_idx).map_err(|e| {
                                format!(
                                    "Error in `remove_node` at {file}:{line}:{column}: could not get stored node: {e}",
                                    file = file!(),
                                    line = line!(),
                                    column = column!(),
                                )
                            })?;

                                match node {
                                    Node::Branch(branch) => {
                                        *matched_child = NodeRef::ModBranch(Box::new(
                                            Branch::from_stored(branch),
                                        ));

                                        key_position
                                    }
                                    Node::Leaf(leaf) => {
                                        // This check is technically unnecessary
                                        // since we already checked that the leaf exists under a parent stored node.
                                        debug_assert_eq!(leaf.key_hash, *key_hash);
                                        if leaf.key_hash == *key_hash {
                                            break 'leaf (parent_branch, key_position);
                                        } else {
                                            return Ok(None);
                                        }
                                    }
                                }
                            }
                            NodeRef::ModBranch(_) => {
                                // This code path continues below the match
                                // It is the equivalent of this assignment and continue.
                                // The assignment to parent_node_ref must be done outside the match to satisfy the borrow checker.
                                //
                                // parent_node_ref = matched_child;
                                // continue;
                                key_position
                            }
                            NodeRef::ModLeaf(leaf) => {
                                if leaf.key_hash == *key_hash {
                                    break 'leaf (parent_branch, key_position);
                                } else {
                                    return Ok(None);
                                }
                            }
                        }
                    }
                };

                // This code path is only taken if the matched_child is a ModBranch
                // See the comment there.
                let NodeRef::ModBranch(branch) = parent_node_ref else {
                    unreachable!("We just matched a ModBranch");
                };

                grandparent_word_idx = branch.mask.word_idx();

                match key_position {
                    KeyPosition::Left => parent_node_ref = &mut branch.left,
                    KeyPosition::Right => parent_node_ref = &mut branch.right,
                    KeyPosition::Adjacent(_) => return Ok(None),
                }
            }
        };

        // This code is the continuation of the 'leaf labeled break.
        // At this point we know the parent_branch has the leaf we want to remove at leaf_key_position.
        let (matched, unmatched) = match leaf_key_position {
            KeyPosition::Left => (&mut parent_branch.left, &mut parent_branch.right),
            KeyPosition::Right => (&mut parent_branch.right, &mut parent_branch.left),
            KeyPosition::Adjacent(_) => return Ok(None),
        };

        let matched = mem::replace(matched, NodeRef::temp_null_stored());
        let mut unmatched = mem::replace(unmatched, NodeRef::temp_null_stored());

        let leaf_value = match matched {
            NodeRef::ModLeaf(leaf) => {
                debug_assert_eq!(leaf.key_hash, *key_hash);
                leaf.value
            }
            NodeRef::Stored(stored_idx) => {
                let node = data_store.get_node(stored_idx).map_err(|e| {
                    format!(
                        "Error in `remove_node` at {file}:{line}:{column}: could not get stored node: {e}",
                        file = file!(),
                        line = line!(),
                        column = column!(),
                    )
                })?;

                match node {
                    Node::Leaf(leaf) => {
                        debug_assert_eq!(leaf.key_hash, *key_hash);
                        leaf.value.clone()
                    }
                    _ => unreachable!("We just matched a leaf"),
                }
            }
            NodeRef::ModBranch(_) => unreachable!("We are in the leaf labeled break"),
        };

        Self::adjust_sibling_prefix(
            data_store,
            &mut unmatched,
            parent_branch,
            grandparent_word_idx,
        )?;

        *parent_node_ref = mem::replace(&mut unmatched, NodeRef::temp_null_stored());

        Ok(Some(leaf_value))
    }

    // This method adds the parent's prefix and prior_word to the unmatched_child if needed.
    fn adjust_sibling_prefix(
        data_store: &S,
        unmatched_child: &mut NodeRef<S::Value>,
        parent_branch: &Branch<NodeRef<S::Value>>,
        grandparent_word_idx: usize,
    ) -> Result<(), String> {
        let parent_prefix = &parent_branch.prefix;
        let parent_prior_word = parent_branch.prior_word;

        // check if we need to add the prefix and prior_word to the unmatched_child
        let parent_branch_word_idx = parent_branch.mask.word_idx();

        // There cannot be a prefix if the word_idx is 0 or 1
        // word_idx 0 prefix is 0
        // word_idx 1 prefix is prior_word
        if parent_branch.mask.word_idx() == 0 {
            debug_assert!(parent_prefix.is_empty());
            debug_assert!(parent_prior_word == 0);
            debug_assert!(grandparent_word_idx == 0);

            #[cfg(debug_assertions)]
            if let NodeRef::ModBranch(branch) = unmatched_child {
                let word_idx = branch.mask.word_idx();
                if word_idx == 0 {
                    debug_assert!(branch.prefix.is_empty());
                    debug_assert!(branch.prior_word == 0);
                } else if word_idx == 1 {
                    debug_assert!(branch.prefix.is_empty());
                } else {
                    debug_assert_eq!(
                        branch.prefix.len(),
                        word_idx.saturating_sub(parent_branch_word_idx + 1)
                    )
                }
            }

            Ok(())
        } else if parent_branch_word_idx == grandparent_word_idx {
            debug_assert!(parent_prefix.is_empty());
            Ok(())
        } else {
            let new_prefix = |word_idx, prefix: &[u32]| {
                if word_idx == parent_branch_word_idx {
                    debug_assert!(prefix.is_empty());
                    parent_prefix.to_vec()
                } else {
                    parent_prefix
                        .iter()
                        .chain(iter::once(&parent_prior_word))
                        .chain(prefix)
                        .copied()
                        .collect()
                }
            };

            match unmatched_child {
                NodeRef::ModLeaf(_) => Ok(()),
                // We don't need to add a parent prefix or prior_word
                // the prior_word is already the same
                // and the prefix is empty
                NodeRef::ModBranch(branch)
                    if parent_branch_word_idx == branch.mask.word_idx()
                        && parent_prefix.is_empty() =>
                {
                    Ok(())
                }
                NodeRef::ModBranch(branch) => {
                    branch.prefix = new_prefix(branch.mask.word_idx(), &branch.prefix);
                    Ok(())
                }

                // This creates an unessary read and maybe write of the stored node.
                // We could avoid this by using extendion nodes instead of prefixes.
                NodeRef::Stored(stored_idx) => {
                    let node = data_store.get_node(*stored_idx).map_err(|e| {
                        format!(
                            "Error in `remove_node` at {file}:{line}:{column}: could not get stored node: {e}",
                            file = file!(),
                            line = line!(),
                            column = column!(),
                            )
                        })?;

                    match node {
                        Node::Leaf(_) => Ok(()),
                        Node::Branch(branch)
                            if parent_branch_word_idx == branch.mask.word_idx()
                                && parent_prefix.is_empty() =>
                        {
                            Ok(())
                        }
                        Node::Branch(branch) => {
                            *unmatched_child = NodeRef::ModBranch(Box::new(Branch {
                                left: NodeRef::Stored(branch.left),
                                right: NodeRef::Stored(branch.right),
                                mask: branch.mask,
                                prior_word: branch.prior_word,
                                prefix: new_prefix(branch.mask.word_idx(), &branch.prefix),
                            }));
                            Ok(())
                        }
                    }
                }
            }
        }
    }
}

impl<S: Store> Transaction<S> {
    /// This method allows for getting, inserting, and updating a entry in the trie with a single lookup.
    /// We match the standard library's `Entry` API for the most part.
    ///
    /// Note: Use of `entry` renders the trie path even if the entry is not modified.
    /// This incurs allocations, now and unnecessary rehashing later when calculating the root hash.
    /// For this reason you should prefer `get` if you have a high probability of not modifying the entry.
    #[inline]
    pub fn entry<'txn>(
        &'txn mut self,
        key_hash: &KeyHash,
    ) -> Result<Entry<'txn, S::Value>, TrieError> {
        let mut key_position = KeyPositionAdjacent::PrefixOfWord(usize::MAX);

        match self.current_root {
            TrieRoot::Empty => Ok(Entry::VacantEmptyTrie(VacantEntryEmptyTrie {
                root: &mut self.current_root,
                key_hash: *key_hash,
            })),
            TrieRoot::Node(ref mut root) => {
                let mut node_ref = root;
                let mut last_word_idx = 0;
                let mut last_bit_idx = 0;

                let last_word_idx = loop {
                    let go_right = match &*node_ref {
                        NodeRef::ModBranch(branch) => {
                            debug_assert!(branch.mask.word_idx() >= last_word_idx);
                            debug_assert!(branch.mask.bit_idx >= last_bit_idx);
                            last_bit_idx = branch.mask.bit_idx;

                            last_word_idx = branch.mask.word_idx();
                            match branch.key_position(key_hash) {
                                KeyPosition::Left => false,
                                KeyPosition::Right => true,
                                KeyPosition::Adjacent(pos) => {
                                    key_position = pos;
                                    break last_word_idx;
                                }
                            }
                        }
                        NodeRef::ModLeaf(_) => break last_word_idx,
                        NodeRef::Stored(idx) => {
                            let loaded_node = self.data_store.get_node(*idx).map_err(|e| {
                                format!(
                                    "Error in `entry` at {file}:{line}:{column}: could not get stored node: {e}",
                                    file = file!(),
                                    line = line!(),
                                    column = column!(),
                                )
                            })?;

                            match loaded_node {
                                Node::Branch(branch) => {
                                    // Connect the new branch to the trie.
                                    *node_ref =
                                        NodeRef::ModBranch(Box::new(Branch::from_stored(branch)));
                                }
                                Node::Leaf(leaf) => {
                                    *node_ref = NodeRef::ModLeaf(Box::new(leaf.clone()));
                                }
                            }
                            continue;
                        }
                    };

                    match (go_right, node_ref) {
                        (true, NodeRef::ModBranch(ref mut branch)) => {
                            node_ref = &mut branch.right;
                        }
                        (false, NodeRef::ModBranch(ref mut branch)) => {
                            node_ref = &mut branch.left;
                        }
                        _ => unreachable!("We just matched a ModBranch"),
                    }
                };

                // This convoluted return makes the borrow checker happy.
                if let NodeRef::ModLeaf(leaf) = &*node_ref {
                    if leaf.key_hash != *key_hash {
                        // This is a logical null
                        // TODO we should break VacantEntry into two types VacantEntryBranch and VacantEntryLeaf
                        debug_assert_eq!(
                            key_position,
                            KeyPositionAdjacent::PrefixOfWord(usize::MAX)
                        );

                        return Ok(Entry::Vacant(VacantEntry {
                            parent: node_ref,
                            key_hash: *key_hash,
                            key_position,
                            last_word_idx,
                        }));
                    }
                };

                if let NodeRef::ModBranch(_) = &*node_ref {
                    Ok(Entry::Vacant(VacantEntry {
                        parent: node_ref,
                        key_hash: *key_hash,
                        key_position,
                        last_word_idx,
                    }))
                } else if let NodeRef::ModLeaf(leaf) = &mut *node_ref {
                    Ok(Entry::Occupied(OccupiedEntry { leaf }))
                } else {
                    unreachable!("prior loop only breaks on a leaf or branch");
                }
            }
        }
    }

    #[inline]
    pub fn print_modified_tree(&self)
    where
        S::Value: core::fmt::Debug,
    {
        match &self.current_root {
            TrieRoot::Empty => println!("Empty"),
            TrieRoot::Node(node) => Self::print_modified_node(node, 0),
        }
    }

    fn print_modified_node(node: &NodeRef<S::Value>, depth: usize)
    where
        S::Value: core::fmt::Debug,
    {
        let indent = "  ".repeat(depth);
        match node {
            NodeRef::ModBranch(branch) => {
                println!("{}Branch {{", indent);
                println!("{}  mask:", indent);
                println!(
                    "{}      word_idx: {}, bit_idx: {}, relative_bit_idx: {}",
                    indent,
                    branch.mask.word_idx(),
                    branch.mask.bit_idx,
                    branch.mask.relative_bit_idx()
                );
                println!(
                    "{}      left_prefix: {:032b}",
                    indent, branch.mask.left_prefix
                );
                println!(
                    "{}      right_prefix: {:032b}",
                    indent,
                    branch.mask.right_prefix()
                );
                println!(
                    "{}      prefix_mask: {:032b}",
                    indent,
                    branch.mask.prefix_mask()
                );
                println!(
                    "{}      discriminant_bit_mask: {:032b}",
                    indent,
                    branch.mask.discriminant_bit_mask()
                );
                println!(
                    "{}      prefix_discriminant_mask: {:032b}",
                    indent,
                    branch.mask.prefix_discriminant_mask()
                );
                println!("{}  prior_word: {:032b},", indent, branch.prior_word);
                println!(
                    "{}  prefix: [{}],",
                    indent,
                    branch
                        .prefix
                        .iter()
                        .map(|w| format!("{:032b}", w))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                println!("{}  left: ", indent);
                Self::print_modified_node(&branch.left, depth + 2);
                println!("{}  right: ", indent);
                Self::print_modified_node(&branch.right, depth + 2);
                println!("{}}}", indent);
            }
            NodeRef::ModLeaf(leaf) => {
                let key_str: String = leaf.key_hash.0.iter().fold(String::new(), |mut s, word| {
                    s.push_str(&format!("{:032b} ", word));
                    s
                });
                println!(
                    "{}Leaf {{ key: 0x{}, value: {:?} }}",
                    indent, key_str, leaf.value
                );
            }
            NodeRef::Stored(_) => {
                println!("{}..stored..", indent);
            }
        }
    }
}

impl<Db: DatabaseGet<V>, V: PortableHash + Clone> Transaction<SnapshotBuilder<Db, V>> {
    /// An alias for `SnapshotBuilder::new_with_db`.
    ///
    /// Builds a snapshot of the trie before the transaction.
    /// The `Snapshot` is not a complete representation of the trie.
    /// The `Snapshot` only contains information about the parts of the trie touched by the transaction.
    /// Because of this, two `Snapshot`s of the same trie may not be equal if the transactions differ.
    ///
    /// Note: All operations including get affect the contents of the snapshot.
    #[inline]
    pub fn build_initial_snapshot(&self) -> Snapshot<V> {
        self.data_store.build_initial_snapshot()
    }

    #[inline]
    pub fn from_snapshot_builder(builder: SnapshotBuilder<Db, V>) -> Self {
        Transaction {
            current_root: builder.trie_root(),
            data_store: builder,
        }
    }
}

impl<Db: DatabaseGet<V>, V: PortableHash + Clone> TryFrom<SnapshotBuilder<Db, V>>
    for Transaction<SnapshotBuilder<Db, V>>
{
    type Error = TrieError;

    #[inline]
    fn try_from(value: SnapshotBuilder<Db, V>) -> Result<Self, Self::Error> {
        Ok(Transaction::from_snapshot_builder(value))
    }
}

impl<'s, V: PortableHash + Clone> Transaction<&'s Snapshot<V>> {
    /// Create a `Transaction` from a borrowed `Snapshot`.
    #[inline]
    pub fn from_unverified_snapshot_ref(snapshot: &'s Snapshot<V>) -> Result<Self, TrieError> {
        Ok(Transaction {
            current_root: snapshot.trie_root()?,
            data_store: snapshot,
        })
    }
}

impl<V: PortableHash + Clone> Transaction<Snapshot<V>> {
    /// Create a `Transaction` from a owned `Snapshot`.
    #[inline]
    pub fn from_unverified_snapshot(snapshot: Snapshot<V>) -> Result<Self, TrieError> {
        Ok(Transaction {
            current_root: snapshot.trie_root()?,
            data_store: snapshot,
        })
    }
}

impl<'s, V: PortableHash + Clone> TryFrom<&'s Snapshot<V>> for Transaction<&'s Snapshot<V>> {
    type Error = TrieError;

    #[inline]
    fn try_from(value: &'s Snapshot<V>) -> Result<Self, Self::Error> {
        Self::from_unverified_snapshot_ref(value)
    }
}

impl<V: PortableHash + Clone> TryFrom<Snapshot<V>> for Transaction<Snapshot<V>> {
    type Error = TrieError;

    #[inline]
    fn try_from(value: Snapshot<V>) -> Result<Self, Self::Error> {
        Self::from_unverified_snapshot(value)
    }
}

impl<'s, S: Store + AsRef<Snapshot<S::Value>>> Transaction<&'s VerifiedSnapshot<S>> {
    /// Create a `Transaction` from a borrowed `VerifiedSnapshot`.
    #[inline]
    pub fn from_verified_snapshot_ref(snapshot: &'s VerifiedSnapshot<S>) -> Self {
        Transaction {
            current_root: snapshot.trie_root(),
            data_store: snapshot,
        }
    }
}

impl<S: Store + AsRef<Snapshot<S::Value>>> Transaction<VerifiedSnapshot<S>> {
    /// Create a `Transaction` from a owned `VerifiedSnapshot`.
    #[inline]
    pub fn from_verified_snapshot(snapshot: VerifiedSnapshot<S>) -> Self {
        Transaction {
            current_root: snapshot.trie_root(),
            data_store: snapshot,
        }
    }
}

impl<'s, S: Store + AsRef<Snapshot<S::Value>>> From<&'s VerifiedSnapshot<S>>
    for Transaction<&'s VerifiedSnapshot<S>>
{
    #[inline]
    fn from(value: &'s VerifiedSnapshot<S>) -> Self {
        Transaction::from_verified_snapshot_ref(value)
    }
}

impl<S: Store + AsRef<Snapshot<S::Value>>> From<VerifiedSnapshot<S>>
    for Transaction<VerifiedSnapshot<S>>
{
    #[inline]
    fn from(value: VerifiedSnapshot<S>) -> Self {
        Transaction::from_verified_snapshot(value)
    }
}

pub enum Entry<'a, V> {
    /// A Leaf
    Occupied(OccupiedEntry<'a, V>),
    /// The first Branch that proves the key is not in the trie.
    Vacant(VacantEntry<'a, V>),
    VacantEmptyTrie(VacantEntryEmptyTrie<'a, V>),
}

impl<'a, V> Entry<'a, V> {
    #[inline]
    pub fn get(&self) -> Option<&V> {
        match self {
            Entry::Occupied(OccupiedEntry { leaf }) => Some(&leaf.value),
            _ => None,
        }
    }

    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut V> {
        match self {
            Entry::Occupied(OccupiedEntry { leaf }) => Some(&mut leaf.value),
            _ => None,
        }
    }

    #[inline]
    pub fn into_mut(self) -> Option<&'a mut V> {
        match self {
            Entry::Occupied(OccupiedEntry { leaf }) => Some(&mut leaf.value),
            _ => None,
        }
    }

    /// Prefer `Transaction::insert` over `Entry::insert` if you are not using any other `Entry` methods.
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        match self {
            Entry::Occupied(mut o) => {
                o.insert(value);
                o.into_mut()
            }
            Entry::VacantEmptyTrie(entry) => entry.insert(value),
            Entry::Vacant(entry) => entry.insert(value),
        }
    }

    #[inline]
    pub fn or_insert(self, value: V) -> &'a mut V {
        self.or_insert_with(|| value)
    }

    #[inline]
    pub fn or_insert_with<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        self.or_insert_with_key(|_| default())
    }

    #[inline]
    pub fn or_insert_with_key<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce(&KeyHash) -> V,
    {
        match self {
            Entry::Occupied(o) => &mut o.leaf.value,
            Entry::VacantEmptyTrie(entry) => {
                let value = default(entry.key());
                entry.insert(value)
            }
            Entry::Vacant(entry) => {
                let value = default(entry.key());
                entry.insert(value)
            }
        }
    }

    #[inline]
    pub fn key(&self) -> &KeyHash {
        match self {
            Entry::Occupied(OccupiedEntry { leaf }) => &leaf.key_hash,
            Entry::Vacant(VacantEntry { key_hash, .. })
            | Entry::VacantEmptyTrie(VacantEntryEmptyTrie { key_hash, .. }) => key_hash,
        }
    }
    #[inline]
    pub fn and_modify<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match self {
            Entry::Occupied(OccupiedEntry { ref mut leaf }) => {
                f(&mut leaf.value);
                self
            }
            _ => self,
        }
    }

    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        #[allow(clippy::unwrap_or_default)]
        self.or_insert_with(Default::default)
    }
}

pub struct OccupiedEntry<'a, V> {
    /// This always points to a Leaf.
    /// It may be a ModLeaf or a stored Leaf.
    leaf: &'a mut Leaf<V>,
}

impl<'a, V> OccupiedEntry<'a, V> {
    #[inline]
    pub fn key(&self) -> &KeyHash {
        &self.leaf.key_hash
    }

    #[inline]
    pub fn get(&self) -> &V {
        &self.leaf.value
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.leaf.value
    }

    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        &mut self.leaf.value
    }

    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(&mut self.leaf.value, value)
    }
}

pub struct VacantEntry<'a, V> {
    parent: &'a mut NodeRef<V>,
    key_hash: KeyHash,
    key_position: KeyPositionAdjacent,
    // The word index of the last branch we traversed.
    last_word_idx: usize,
}

impl<'a, V> VacantEntry<'a, V> {
    #[inline]
    pub fn key(&self) -> &KeyHash {
        &self.key_hash
    }

    #[inline]
    pub fn into_key(self) -> KeyHash {
        self.key_hash
    }

    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        let VacantEntry {
            parent,
            key_hash,
            key_position,
            last_word_idx,
        } = self;
        if let NodeRef::ModBranch(branch) = parent {
            debug_assert_eq!(branch.mask.word_idx(), last_word_idx);

            let leaf =
                branch.new_adjacent_leaf_ret(key_position, Box::new(Leaf { key_hash, value }));
            return &mut leaf.value;
        };

        let owned_parent = mem::replace(parent, NodeRef::temp_null_stored());
        match owned_parent {
            NodeRef::ModLeaf(old_leaf) => {
                let (new_branch, new_leaf_is_right) = Branch::new_from_leafs(
                    last_word_idx,
                    old_leaf,
                    Box::new(Leaf { key_hash, value }),
                );

                *parent = NodeRef::ModBranch(new_branch);

                match parent {
                    NodeRef::ModBranch(branch) => {
                        let leaf = if new_leaf_is_right {
                            &mut branch.right
                        } else {
                            &mut branch.left
                        };

                        match leaf {
                            NodeRef::ModLeaf(ref mut leaf) => &mut leaf.value,
                            _ => {
                                unreachable!("new_from_leafs returns the location of the new leaf")
                            }
                        }
                    }
                    _ => unreachable!("new_from_leafs returns a ModBranch"),
                }
            }
            _ => {
                unreachable!("`entry` ensures VacantEntry should never point to a Stored node")
            }
        }
    }
}

pub struct VacantEntryEmptyTrie<'a, V> {
    root: &'a mut TrieRoot<NodeRef<V>>,
    key_hash: KeyHash,
}

impl<'a, V> VacantEntryEmptyTrie<'a, V> {
    #[inline]
    pub fn key(&self) -> &KeyHash {
        &self.key_hash
    }

    #[inline]
    pub fn into_key(self) -> KeyHash {
        self.key_hash
    }

    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        let VacantEntryEmptyTrie { root, key_hash } = self;
        *root = TrieRoot::Node(NodeRef::ModLeaf(Box::new(Leaf { key_hash, value })));

        match root {
            TrieRoot::Node(NodeRef::ModLeaf(leaf)) => &mut leaf.value,
            _ => unreachable!("We just set root to a ModLeaf"),
        }
    }
}
