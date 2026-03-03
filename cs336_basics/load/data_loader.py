from collections.abc import Iterable

import numpy.typing as npt
import torch

from cs336_basics.utils.split_data import find_chunk_boundaries

def get_batch_data(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, **sample** language modeling input sequences and their corresponding
    labels from the dataset.
    """
    len_dataset = dataset.shape[0]
    positions = torch.randint(low=0, high=len_dataset - context_length - 1, size = (batch_size,))

    data_batch = torch.stack(
            [torch.tensor(dataset[i : i + context_length]) for i in positions]
        )
    label_batch = torch.stack(
            [torch.tensor(dataset[i + 1 : i + context_length + 1]) for i in positions]
        )
    
    # 3. Ensure they are the correct data type (long integers for token IDs)
    data_batch = data_batch.long()
    label_batch = label_batch.long()
    
    # 4. Move them to the requested hardware device
    data_batch = data_batch.to(device)
    label_batch = label_batch.to(device)
    
    return data_batch, label_batch

def open_dataset(train_dataset_path:str, valid_dataset_path)->str:
    with open(train_dataset_path) as f:
        train_data = f.read()

    with open(valid_dataset_path) as f:
        valid_data = f.read()

    return train_data, valid_data

def chunked_file_reader(filepath: str, chunk_num:int)->Iterable[str]:
    """Reads a large file in chunks to prevent memory blowouts."""

    boundaries = find_chunk_boundaries(filepath, chunk_num, split_special_token=b"<|endoftext|>")

    with open(filepath, 'rb') as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # Jump to the start of the chunk
            f.seek(start)
            
            # Read exactly the number of bytes in this chunk
            raw_bytes = f.read(end - start)
            
            # Decode the raw bytes into a standard Python string
            # errors='ignore' ensures a stray broken byte won't crash your 3-hour job
            chunk_str = raw_bytes.decode('utf-8', errors='ignore')
            
            # Yield the chunk to your tokenizer pipeline!
            yield chunk_str