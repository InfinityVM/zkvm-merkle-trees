use core::fmt::{Debug, Display};

use kairos_trie::{PortableHash, PortableHasher};

use crate::node::{NodeHash, NodeOrLeaf};

pub type Idx = u32;

pub trait Store {
    type Error: Display + Debug;
    type Key: Ord + Clone + PortableHash + Debug;
    type Value: Clone + PortableHash + Debug;

    fn calc_subtree_hash(
        &self,
        hasher: &mut impl PortableHasher<32>,
        hash_idx: Idx,
    ) -> Result<NodeHash, Self::Error>;

    fn get(&self, hash_idx: Idx) -> Result<NodeOrLeaf<Self::Key, Self::Value>, Self::Error>;
}
