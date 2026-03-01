# use typer to parse command line arguments and parse Traceback stack
from typing import Annotated
import typer

from cs336_basics.config import loggerConfig
from cs336_basics.constant.constant import ONE_BYTES_SIZE
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.tokenizer.train_tokenizer import train_bpe


app = typer.Typer(
    pretty_exceptions_show_locals=False,  # This hides the long list of variables
    # pretty_exceptions_short=True         # This makes the traceback even more concise
)


@app.command()
def main(
    dataset_path: Annotated[
        str, typer.Option(help="path to the dataset")
    ] = "../autodl-tmp/SEED",
    checkpoint_dir : Annotated[str, typer.Option()] = "./",
    device: Annotated[str, typer.Option(help="device to run the model on")] = "cpu",
    vocab_size: Annotated[
        int, typer.Option(help="Size of the tokenizer vocabulary (e.g., 50257)")
    ] = "cuda",
    context_length: Annotated[
        int, typer.Option(help="Maximum sequence length of one sample")
    ] = 1,
    num_layers: Annotated[
        int, typer.Option(help="Number of Transformer blocks")
    ] = 1,
    d_model: Annotated[
        int, typer.Option(help="Dimensionality of the embeddings and hidden states.")
    ] = 1,
    num_heads: Annotated[
        int, typer.Option(help="Number of attention heads.")
    ] = 1,
    batch_size: Annotated[int, typer.Option(help="batch size for training")] = 32,
    epochs: Annotated[int, typer.Option(help="Total number of training epochs")] = 20,
    level: Annotated[str, typer.Option("-l", help="Logging level")] = "DEBUG"
    ):
    
    loggerConfig.setUpLogger(level)

    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    input_path = "./data/temp.txt"
    input_path = "/home/saber/cs336-A1/tests/fixtures/tinystories_sample_5M.txt"

    vocab_path = "/home/saber/cs336-A1/tests/fixtures/gpt2_vocab.json"
    merges_path = "/home/saber/cs336-A1/tests/fixtures/gpt2_merges.txt"

    special_tokens = ["<|endoftext|>"]
    merge_times = 10000

    # vocabulary, merges = train_bpe(input_path, ONE_BYTES_SIZE + merge_times + len(special_tokens), special_tokens)

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    text = "Hello, how are you?"

    # with open(input_path, "r") as f:
    #     text = f.read()

    tokens = tokenizer.encode(text)

    text = tokenizer.decode(tokens)


if __name__ == "__main__":
    app()
