# use typer to parse command line arguments and parse Traceback stack
from typing import Annotated
import typer

from cs336_basics.config import loggerConfig
from cs336_basics.constant.constant import ONE_BYTES_SIZE
from cs336_basics.tokenizer.BPETokenizer import train_bpe


app = typer.Typer(
    pretty_exceptions_show_locals=False,  # This hides the long list of variables
    # pretty_exceptions_short=True         # This makes the traceback even more concise
)

@app.command()
def main(level: Annotated[str, typer.Option("-l", help="Logging level")] = "DEBUG"):
    loggerConfig.setUpLogger(level)
    
    input_path = "./data/overlap.txt"
    input_path = "./data/temp.txt"
    input_path = "/home/saber/cs336-A1/tests/fixtures/tinystories_sample_5M.txt"
    special_tokens = [b"<|endoftext|>"]
    merge_times = 1000

    train_bpe(input_path, ONE_BYTES_SIZE + merge_times + len(special_tokens), special_tokens)
    

if __name__ == "__main__":
    app()
