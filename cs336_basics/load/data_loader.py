import numpy.typing as npt
import torch

def get_batch_data(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, **sample** language modeling input sequences and their corresponding
    labels from the dataset.
    """
    len_dataset = dataset.shape[0]
    positions = torch.randint(low=0, high=len_dataset - context_length, size = (batch_size,))

    data_batch = torch.stack(
            [torch.from_numpy(dataset[i : i + context_length]) for i in positions]
        )
    label_batch = torch.stack(
            [torch.from_numpy(dataset[i + 1 : i + context_length + 1]) for i in positions]
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