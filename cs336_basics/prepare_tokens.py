# use typer to parse command line arguments and parse Traceback stack
import os
from typing import Annotated
from loguru import logger
import numpy as np
import typer



from cs336_basics.config import logger_config
from cs336_basics.load.data_loader import chunked_file_reader, open_dataset
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
    ] = "./data/TinyStoriesV2-GPT4-train.txt",
    valid_dataset_path: Annotated[
        str, typer.Option(help="path to the valid dataset")
    ] = "./data/TinyStoriesV2-GPT4-valid.txt",
    vocab_filepath : Annotated[
        str, typer.Option(help="path to saved vocabulary")
    ] = "./data/tinystories_vocab.json",
    merges_filepath : Annotated[
        str, typer.Option(help="path to saved merges")
    ] = "./data/tinystories_merges.txt",
    train_tokens_filepath: Annotated[
        str, typer.Option(help="path to saved training tokens") 
    ] = "./data/train_token_origin.bin",
    valid_tokens_filepath: Annotated[
        str, typer.Option(help="path to saved validation tokens")
    ] = "./data/valid_token_origin.bin",
    chunks_num: Annotated[
        int, typer.Option(help="number of chunks to split the dataset into")
    ] = 10,
    vocab_size: Annotated[
        int, typer.Option(help="Size of the tokenizer vocabulary (e.g., 50257)")
    ] = 10000,

    level: Annotated[str, typer.Option("-l", help="Logging level")] = "INFO"
    ):
    
    logger_config.set_up_logger(level)

    special_tokens = ["<|endoftext|>"]


    vocabulary, merges = train_bpe(train_dataset_path,vocab_size, special_tokens)
    # Save the them to your data folder
    save_vocabulary_merges(
        vocabulary, 
        merges, 
        vocab_filepath, 
        merges_filepath
    )

    tokenizer = Tokenizer(vocabulary, merges, special_tokens)
    # tokenizer = Tokenizer.from_files_remapped(vocab_filepath, merges_filepath, special_tokens)
    
    # train_text, valid_text = open_dataset(train_dataset_path, valid_dataset_path)

    # Encode and save to memmap
    for split, filepath in [("train", train_dataset_path), ("valid", valid_dataset_path)]:
        logger.info(f"Encoding {split} text...")

        chunks_generator = chunked_file_reader(filepath, chunks_num)
        
        tokens_generator = tokenizer.encode_iterable(chunks_generator)
        output_path = train_tokens_filepath if split == "train" else valid_tokens_filepath
        with open(output_path, 'wb') as f_out:
            batch = []
            total_tokens = 0
            
            # Iterate over the lazy tokens
            for token_id in tokens_generator:
                batch.append(token_id)
                
                # Write to disk in batches of 1,000,000 to keep it incredibly fast
                if len(batch) >= 10_000_000:
                    # Convert python ints to uint16 array, then to raw bytes, and write
                    byte_data = np.array(batch, dtype=np.uint16).tobytes()
                    f_out.write(byte_data)
                    
                    total_tokens += len(batch)
                    batch.clear() # Clear the RAM!
                    logger.info(f"  ... encoded {total_tokens} tokens so far")
                    
            # Write any remaining tokens in the final partial batch
            if batch:
                byte_data = np.array(batch, dtype=np.uint16).tobytes()
                f_out.write(byte_data)
                total_tokens += len(batch)

        logger.info(f"Successfully saved {split}.bin with {total_tokens} tokens total.")

if __name__ == "__main__":
    app()
