use alloc::{
    boxed::Box,
    string::{String, ToString},
};
use core::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BTreeError(Box<str>);

impl BTreeError {
    #[inline]
    pub fn display(&self) -> &str {
        &self.0
    }
}

impl Display for BTreeError {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for BTreeError {
    #[inline]
    fn from(s: &str) -> Self {
        Self(s.into())
    }
}

impl From<String> for BTreeError {
    #[inline]
    fn from(s: String) -> Self {
        Self(s.into_boxed_str())
    }
}

impl From<&String> for BTreeError {
    #[inline]
    fn from(s: &String) -> Self {
        Self(s.clone().into_boxed_str())
    }
}

impl From<&BTreeError> for String {
    #[inline]
    fn from(e: &BTreeError) -> Self {
        e.0.to_string()
    }
}

impl From<BTreeError> for String {
    #[inline]
    fn from(e: BTreeError) -> Self {
        e.0.to_string()
    }
}
