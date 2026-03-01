# use typer to parse command line arguments and parse Traceback stack
import os
from typing import Annotated
from loguru import logger
import numpy as np
import typer



from cs336_basics.config import loggerConfig
from cs336_basics.load.data_loader import open_dataset
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.tokenizer.train_tokenizer import save_vocabulary_merges, train_bpe


app = typer.Typer(
    pretty_exceptions_show_locals=False,  # This hides the long list of variables
    # pretty_exceptions_short=True         # This makes the traceback even more concise
)

@app.command()
def main(
    train_dataset_path: Annotated[
        str, typer.Option(help="path to the train dataset")
    ] = "../data/TinyStoriesV2-GPT4-train.txt",
    valid_dataset_path: Annotated[
        str, typer.Option(help="path to the valid dataset")
    ] = "../data/TinyStoriesV2-GPT4-valid.txt",
    vocab_filepath : Annotated[
        str, typer.Option(help="path to saved vocabulary")
    ] = "./data/tinystories_vocab.json",
    merges_filepath : Annotated[
        str, typer.Option(help="path to saved merges")
    ] = "./data/tinystories_merges.txt",
    train_tokens_filepath: Annotated[
        str, typer.Option(help="path to saved training tokens") 
    ] = "./data/train_token.bin",
    valid_tokens_filepath: Annotated[
        str, typer.Option(help="path to saved validation tokens")
    ] = "./data/valid_token.bin",
    vocab_size: Annotated[
        int, typer.Option(help="Size of the tokenizer vocabulary (e.g., 50257)")
    ] = 10000,

    level: Annotated[str, typer.Option("-l", help="Logging level")] = "INFO"
    ):
    
    loggerConfig.setUpLogger(level)

    special_tokens = ["<|endoftext|>"]


    tokenizer = None

    vocabulary, merges = train_bpe(train_dataset_path,vocab_size, special_tokens)
    # Save the them to your data folder
    save_vocabulary_merges(
        vocabulary, 
        merges, 
        vocab_filepath, 
        merges_filepath
    )

    tokenizer = Tokenizer(vocabulary, merges, special_tokens)
    
    logger.info("Loading train and valid dataset...")
    train_text, valid_text = open_dataset(train_dataset_path, valid_dataset_path)

    # Encode and save to memmap
    for split, text in [("train", train_text), ("valid", valid_text)]:
        logger.info(f"Encoding {split} text...")
        tokens = tokenizer.encode(text)
        
        logger.info(f"Saving {split} to np.memmap...")
        
        filepath = train_tokens_filepath if split == "train" else valid_tokens_filepath

        # Create the memory-mapped array on disk
        mmap = np.memmap(filepath, dtype=np.uint16, mode='w+', shape=(len(tokens),))
        
        # Write the tokens into the array
        mmap[:] = tokens
        
        # CRITICAL: flush() forces the OS to write the data from RAM to the hard drive
        mmap.flush()
        logger.info(f"Successfully saved {split}.bin with {len(tokens)} tokens.")

if __name__ == "__main__":
    app()
