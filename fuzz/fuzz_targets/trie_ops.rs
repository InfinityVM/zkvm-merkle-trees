#![no_main]

use libfuzzer_sys::fuzz_target;
use trie_test_utils::{end_to_end_entry_ops, Operation};

fuzz_target!(|batches: Vec<Vec<Operation>>| {
    end_to_end_entry_ops(batches);
});

