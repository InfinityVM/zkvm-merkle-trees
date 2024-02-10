use std::hash::Hash;

use alloc::{collections::BTreeMap, fmt::Debug, string::String};

use crate::Leaf;

#[allow(clippy::type_complexity)]
pub type NodeRef<S> = Node<<S as Store>::BranchRef, <S as Store>::LeafRef>;

pub type Branch<S> = crate::Branch<NodeRef<S>>;

pub trait Store {
    /// The hash of a node or leaf.
    /// Alternatively, this could be a reference or an index that uniquely identifies a node or leaf
    type BranchRef: Ref;
    type LeafRef: Ref;
    type Error: Into<String>;

    fn get_branch(&self, hash: Self::BranchRef) -> Result<Branch<Self>, Error>;
    fn get_leaf(&self, hash: Self::LeafRef) -> Result<&Leaf, Error>;
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Node<B, L> {
    Branch(B),
    Leaf(L),
}

pub trait Ref: Copy + Clone + Eq + Ord + Hash + Debug {}

pub enum Error {
    NodeNotFound,
}

impl From<Error> for String {
    fn from(err: Error) -> String {
        match err {
            Error::NodeNotFound => "Node not found".into(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BranchHash(pub [u8; 32]);
impl Ref for BranchHash {}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct LeafHash(pub [u8; 32]);
impl Ref for LeafHash {}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct MemoryStore {
    // TODO: use a indexmap
    branches: BTreeMap<BranchHash, Branch<Self>>,
    leaves: BTreeMap<LeafHash, Leaf>,
}

impl Store for MemoryStore {
    type BranchRef = BranchHash;
    type LeafRef = LeafHash;
    type Error = Error;

    fn get_branch(&self, hash: Self::BranchRef) -> Result<Branch<Self>, Error> {
        self.branches.get(&hash).cloned().ok_or(Error::NodeNotFound)
    }

    fn get_leaf(&self, hash: Self::LeafRef) -> Result<&Leaf, Error> {
        self.leaves.get(&hash).ok_or(Error::NodeNotFound)
    }
}