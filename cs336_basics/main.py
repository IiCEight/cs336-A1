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
    
    input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = [b"<|endoftext|>"]

    train_bpe(input_path, ONE_BYTES_SIZE + 1000, special_tokens)
    

if __name__ == "__main__":
    app()
