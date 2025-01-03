// #![no_std]
#![allow(clippy::type_complexity)]

extern crate alloc;

pub mod db;
pub mod node;
pub mod snapshot;
pub mod store;
pub mod transaction;

pub mod prelude {
    pub use crate::node::NodeHash;
    pub use crate::snapshot::{Snapshot, SnapshotBuilder, VerifiedSnapshot};
    pub use crate::store::Store;
    pub use crate::transaction::MerkleBTreeTxn;
    pub use kairos_trie::{DigestHasher, PortableHash, PortableHasher, PortableUpdate};
}
