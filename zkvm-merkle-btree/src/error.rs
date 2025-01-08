use alloc::{boxed::Box, string::String};
use core::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BTreeErr<DbGetError, DbSetError> {
    /// The Store was wrong in some way.
    /// Maybe the Snapshot is not valid.
    /// Maybe it's being used with a different tree.
    /// Something went very wrong.
    StoreError(Box<str>),
    /// The database failed to get a Node or Leaf.
    DbGetError(DbGetError),
    /// The database failed to set a Node or Leaf.
    DbSetError(DbSetError),
}

impl<DbGetError: Display, DbSetError: Display> Display for BTreeErr<DbGetError, DbSetError> {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            BTreeErr::StoreError(s) => write!(f, "StoreError: {}", s),
            BTreeErr::DbGetError(e) => write!(f, "DbGetError: {}", e),
            BTreeErr::DbSetError(e) => write!(f, "DbSetError: {}", e),
        }
    }
}

impl<DbGetError, DbSetError> From<String> for BTreeErr<DbGetError, DbSetError> {
    #[inline]
    fn from(s: String) -> Self {
        Self::StoreError(s.into_boxed_str())
    }
}
