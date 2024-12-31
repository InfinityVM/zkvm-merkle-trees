use trie_test_utils::{Operation, end_to_end_entry_ops};

fn main() {
    afl::fuzz!(|ops: Vec<Vec<Operation>>| {
        end_to_end_entry_ops(ops);
    });
}
