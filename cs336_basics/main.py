# use typer to parse command line arguments and parse Traceback stack
import os
from typing import Annotated
from loguru import logger
import torch
import typer

import numpy as np
import wandb

from cs336_basics.checkpoint.check_point import save_checkpoint
from cs336_basics.config import loggerConfig
from cs336_basics.load.data_loader import get_batch_data, open_dataset
from cs336_basics.module.transformer_lm import TransformerLM
from cs336_basics.optimizer.adamw import AdamW
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.tokenizer.train_tokenizer import train_bpe
from cs336_basics.utils.cross_entropy import cross_entropy
from cs336_basics.utils.gradient_clipping import gradient_clipping
from cs336_basics.utils.learning_rate_scheduling import lr_cosine_schedule


app = typer.Typer(
    pretty_exceptions_show_locals=False,  # This hides the long list of variables
    # pretty_exceptions_short=True         # This makes the traceback even more concise
)

@app.command()
def main(
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
    checkpoint_dir : Annotated[str, typer.Option()] = "./checkpoint/",
    epochs: Annotated[int, typer.Option(help="Total number of training epochs")] = 50000,
    epochs_per_checkpoint : Annotated[
        int, typer.Option(help="The number of epochs per checkpoint.")
    ] = 5000,
    epochs_per_evaluation : Annotated[
        int, 
        typer.Option(help="The number of epochs per evaluation of validation dataset.")
    ] = 1000,
    batch_size: Annotated[int, typer.Option(help="batch size for training")] = 64,
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
    learning_rate : Annotated[
        float, typer.Option(help="The learning rate.")
    ] = 5e-4,
    num_heads: Annotated[
        int, typer.Option(help="Number of attention heads.")
    ] = 16,
    max_l2_norm_of_gradient : Annotated[
        float, typer.Option(help="Used for gradient clipping.")
    ] = 1.0,
    warmup_iters: Annotated[int, typer.Option(help="Number of warmup steps.")] = 1000,
    cosine_cycle_iters: Annotated[int, typer.Option(help="Number of cosine cycle steps.")] = 50000,
    min_learning_rate: Annotated[float, typer.Option(help="The minimum learning rate.")] = 5e-5,
    max_learning_rate: Annotated[float, typer.Option(help="The maximum learning rate.")] = 5e-4,
    beta1: Annotated[float, typer.Option(help="AdamW beta1 parameter.")] = 0.9,
    beta2: Annotated[float, typer.Option(help="AdamW beta2 parameter.")] = 0.95,
    eps: Annotated[float, typer.Option(help="AdamW epsilon parameter.")] = 1e-8,
    weight_decay: Annotated[float, typer.Option(help="AdamW weight decay.")] = 0.1,
    device: Annotated[str, typer.Option(help="device to run the model on")] = "cpu",
    level: Annotated[str, typer.Option("-l", help="Logging level")] = "INFO",
    wandb_project: Annotated[
        str, typer.Option(help="WandB project name (leave empty to disable)")
    ] = "tinystories-gpt", # <-- 2. New CLI argument
    ):
    
    loggerConfig.setUpLogger(level)

    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    input_path = "./data/temp.txt"
    input_path = "/home/saber/cs336-A1/tests/fixtures/tinystories_sample_5M.txt"

    vocab_path = "/home/saber/cs336-A1/tests/fixtures/gpt2_vocab.json"
    merges_path = "/home/saber/cs336-A1/tests/fixtures/gpt2_merges.txt"

    special_tokens = ["<|endoftext|>"]



    logger.info("Loading pre-trained Tokenizer...")
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    logger.info("Mounting memory-mapped datasets...")
    # mode='r' is crucial! It ensures the training loop can read the data, 
    # but strictly prevents you from accidentally overwriting it.
    train_tokens = np.memmap(train_tokens_filepath, dtype=np.uint16, mode='r')
    valid_tokens = np.memmap(valid_tokens_filepath, dtype=np.uint16, mode='r')

    model = TransformerLM(
        vocab_size, context_length, num_layers, 
        d_model, num_heads, d_ff, rope_theta, device
        )
    
    model.to(device)

    if wandb_project:
        wandb.init(
            project=wandb_project,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "vocab_size": vocab_size,
                "context_length": context_length,
                "num_layers": num_layers,
                "d_model": d_model,
                "d_ff": d_ff,
                "learning_rate": learning_rate,
                "num_heads": num_heads,
                "max_grad_norm": max_l2_norm_of_gradient,
                "warmup_iters":  warmup_iters,
                "cosine_cycle_iters":  cosine_cycle_iters,
                "min_learning_rate": min_learning_rate ,
                "max_learning_rate":  max_learning_rate,
                "beta1": beta1,
                "beta2": beta2,
                "eps": eps,
                "weight_decay": weight_decay
            }
        )

    criterion = cross_entropy
    optimizer = AdamW(model.parameters(), learning_rate, (beta1, beta2), eps, weight_decay)


    model.train()
    best_valid_loss = float('inf')
    for epoch in range(1, epochs + 1):
        loss4log= 0.0
        data_batch, label_batch = get_batch_data(train_tokens, batch_size, context_length, device)

        data_batch = data_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()

        outputs = model(data_batch)

        loss = criterion(outputs, label_batch)

        loss.backward()
        
        gradient_clipping(model.parameters(), max_l2_norm_of_gradient)

        optimizer.step()
        
        new_lr = lr_cosine_schedule(epoch, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        # update learning rate.
        for group in optimizer.param_groups:
            group["lr"] = new_lr

        loss4log = loss.item()

        if epoch % 50 == 0:
            logger.info("Epoch [{}/{}], Avg Loss: {:.4f}", epoch, epochs, loss4log)
            if wandb_project:
                    wandb.log({
                        "train/loss": loss4log, 
                        "train/learning_rate": new_lr, 
                        "step": epoch
                    })

        if epoch % epochs_per_checkpoint == 0:
            logger.info("Saving checkpoint: epoch {}", epoch)
            file_name = f"checkpoint_epoch_{epoch}.pt"
            save_path = os.path.join(checkpoint_dir, file_name)
            save_checkpoint(model, optimizer, epoch, save_path)

        if epoch % epochs_per_evaluation == 0:
            logger.info("Starting evaluation on validation set.....")
            valid_loss = evaluate(model, criterion, valid_tokens, batch_size, context_length, device)
            logger.info("Valid Loss: {}", valid_loss)

            if wandb_project:
                wandb.log({
                    "val/loss": valid_loss,
                    "step": epoch
                })

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                logger.info("--> New best validation loss! Saving best checkpoint...")
                file_name = f"best_checkpoint_epoch_{epoch}.pt"
                save_path = os.path.join(checkpoint_dir, file_name)
                save_checkpoint(model, optimizer, epoch, save_path)
    
    # 6. Finish the wandb run
    if wandb_project:
        wandb.finish()


def evaluate(model, criterion, dataset, batch_size, 
         context_length, device, eval_iters=200):
    model.eval() # Set model to evaluation mode
    
    total_loss = 0.0
    
    with torch.no_grad():
        for epoch in range(1, eval_iters + 1):
            data_batch, label_batch = get_batch_data(dataset, batch_size, context_length, device)
            data_batch = data_batch.to(device)
            label_batch = label_batch.to(device)
            outputs = model(data_batch)

            loss = criterion(outputs, label_batch)

            total_loss += loss.item()

    # CRITICAL: Switch the model back to training mode!
    model.train()

    # Return the average
    return total_loss / eval_iters


if __name__ == "__main__":
    app()
