import torch
from cs336_basics.module.transformer_lm import TransformerLM
from cs336_basics.tokenizer.tokenizer import Tokenizer

# use typer to parse command line arguments and parse Traceback stack
from typing import Annotated

import typer

from cs336_basics.config import logger_config


app = typer.Typer(
    pretty_exceptions_show_locals=False,  # This hides the long list of variables
    # pretty_exceptions_short=True         # This makes the traceback even more concise
)

@app.command()
def main(
    prompt : Annotated[
        str, typer.Argument()
    ] = "hello",
    vocab_filepath : Annotated[
        str, typer.Option(help="path to saved vocabulary")
    ] = "./data/tinystories_vocab.json",
    merges_filepath : Annotated[
        str, typer.Option(help="path to saved merges")
    ] = "./data/tinystories_merges.txt",
    checkpoint_path : Annotated[
        str, typer.Option(help="path to saved merges")
    ] = "./checkpoint/best_checkpoint_epoch_X.pt",
    temperature : Annotated[float, typer.Option(help="batch size for training")] = 1.0,
    max_new_tokens : Annotated[int, typer.Option(help="max tokens the model generate")] = 200,
    vocab_size: Annotated[
        int, typer.Option(help="Size of the tokenizer vocabulary (e.g., 50257)")
    ] = 10000,
    context_length: Annotated[
        int, typer.Option(help="Maximum sequence length of one sample")
    ] = 256,
    num_layers: Annotated[
        int, typer.Option(help="Number of Transformer blocks")
    ] = 4,
    d_model: Annotated[
        int, typer.Option(help="Dimensionality of the embeddings and hidden states.")
    ] = 512,
    d_ff : Annotated[
        int, typer.Option(
            help="Dimensionality of the feed-forward inner layer (section 3.3).")
    ] = 1344,
    rope_theta : Annotated[
        float, typer.Option(help="The RoPE $\Theta$ parameter.")
    ] = 10000,
    num_heads: Annotated[
        int, typer.Option(help="Number of attention heads.")
    ] = 16,
    device: Annotated[str, typer.Option(help="device to run the model on")] = "cpu",
    level: Annotated[str, typer.Option("-l", help="Logging level")] = "INFO",
    ):

    logger_config.setUpLogger(level)

    # 1. Recreate the Tokenizer
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    # 2. Recreate the Model Architecture (must match your training config exactly)
    model = TransformerLM(
        vocab_size, context_length, num_layers, 
        d_model, num_heads, d_ff, rope_theta, device
        )

    # 3. Load the Trained Weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) # Adjust key if your save_checkpoint used a different name
    model.to(device)
    model.eval() # CRITICAL: turn off dropout!

    # 4. Tokenize the input prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 5. The Autoregressive Loop
    print(f"--- Generating story from prompt: '{prompt}' ---\n")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop the context if it exceeds your model's maximum context_length
            idx_cond = input_tensor[:, -context_length:]
            
            # Forward pass to get logits
            logits = model(idx_cond)
            
            # Pluck out the logits for the very last token in the sequence
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature scaling to control randomness (higher = more random)
            next_token_logits = next_token_logits / temperature
            
            # Convert logits to probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the predicted token to the running sequence
            input_tensor = torch.cat((input_tensor, next_token), dim=1)
            
            # Optional: Stop early if the model generates the end-of-text token
            if next_token.item() == tokenizer.encode("<|endoftext|>")[0]:
                break

    # 6. Decode the final sequence back to text
    generated_ids = input_tensor[0].tolist()
    final_text = tokenizer.decode(generated_ids)
    
    return final_text

if __name__ == "__main__":
    app()