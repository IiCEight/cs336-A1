

from cs336_basics.config import logger_config
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.tokenizer.train_tokenizer import save_vocabulary_merges, train_bpe


def verify_tokenizer(original_tokenizer, vocab_filepath: str, merges_filepath: str):
    """
    Verifies that saving, loading, encoding, and decoding all work perfectly.
    """
    print("Starting Tokenizer Verification...\n")
    
    # ---------------------------------------------------------
    # Step 1: Save the original tokenizer
    # ---------------------------------------------------------
    print("1. Saving tokenizer...")
    save_vocabulary_merges(
        original_tokenizer.vocabulary, 
        original_tokenizer.merges, 
        vocab_filepath, 
        merges_filepath
    )
    
    # ---------------------------------------------------------
    # Step 2: Load into a new instance
    # ---------------------------------------------------------
    print("2. Loading new tokenizer from files...")
    loaded_tokenizer = original_tokenizer.__class__.from_files_remapped(
        vocab_filepath, 
        merges_filepath, 
        original_tokenizer.special_tokens
    )
    
# ---------------------------------------------------------
    # Step 3: Verify Internal State (Strict Equality)
    # ---------------------------------------------------------
    print("3. Verifying internal state...")
    
    # Check Vocabularies
    if original_tokenizer.vocabulary != loaded_tokenizer.vocabulary:
        print("❌ Vocabulary mismatch found! Diagnosing...")
        orig_keys = set(original_tokenizer.vocabulary.keys())
        load_keys = set(loaded_tokenizer.vocabulary.keys())
        
        if orig_keys != load_keys:
            print(f"Missing keys in loaded: {orig_keys - load_keys}")
            print(f"Extra keys in loaded: {load_keys - orig_keys}")
        else:
            for k in orig_keys:
                v_orig = original_tokenizer.vocabulary[k]
                v_load = loaded_tokenizer.vocabulary[k]
                if v_orig != v_load:
                    print(f"Mismatch at Token ID {k}:")
                    print(f"  Original : {repr(v_orig)} (Type: {type(v_orig)})")
                    print(f"  Loaded   : {repr(v_load)} (Type: {type(v_load)})")
                    break # Stop at the first mismatch
        raise AssertionError("Vocabularies do not match!")
        
    assert original_tokenizer.merges == loaded_tokenizer.merges, "❌ Merges do not match!"
    assert original_tokenizer.special_tokens == loaded_tokenizer.special_tokens, "❌ Special tokens do not match!"
    print("   ✅ Internal state matches perfectly.")
    # ---------------------------------------------------------
    # Step 4: Verify Encoding / Decoding (The Stress Test)
    # ---------------------------------------------------------
    print("4. Running encode/decode stress tests...")
    
    # We deliberately include emojis, multiple spaces, tabs, and newlines 
    # to trigger the byte-fallback logic.
    with open("/home/isaber/iSaber/cs336-A1/tests/fixtures/tinystories_sample_5M.txt", "r", encoding="utf-8") as f:
        test_strings = [line.strip() for line in f.readlines()]  # Test on the first 1000 lines
    
    for text in test_strings:
        # 1. Do both tokenizers output the exact same IDs?
        orig_ids = original_tokenizer.encode(text)
        loaded_ids = loaded_tokenizer.encode(text)
        assert orig_ids == loaded_ids, f"❌ Encoding mismatch for text: {text!r}"
        
        # 2. Does decoding those IDs reconstruct the EXACT original text?
        # (This is the Golden Rule of Tokenizers)
        decoded_text = loaded_tokenizer.decode(loaded_ids)
        assert decoded_text == text, f"❌ Decode mismatch! Expected {text!r}, got {decoded_text!r}"

    print("   ✅ All encode/decode tests passed.")
    print("\n🎉 SUCCESS! Your tokenizer is 100% correct and safely serialized.")


if __name__ == "__main__":

    logger_config.set_up_logger("WARNING")  # Ensure logging is configured for the test

    special_tokens = ["<|endoftext|>"]


    tokenizer = None
    train_dataset_path = "/home/isaber/iSaber/cs336-A1/tests/fixtures/tinystories_sample_5M.txt"
    vocabulary, merges = train_bpe(train_dataset_path, 500, special_tokens)
    # Save the them to your data folder
    save_vocabulary_merges(
        vocabulary, 
        merges, 
        "./test_vocab.json", 
        "./test_merges.txt"
    )

    tokenizer = Tokenizer(vocabulary, merges, special_tokens)
    verify_tokenizer(tokenizer, "./test_vocab.json", "./test_merges.txt")