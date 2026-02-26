
import torch


array = torch.ones((10,3,4))
index = torch.tensor([[2, 3],
                      [4, 5],
                      [6, 7]])

result = array[index]

# Shape of arrary torch.Size([10, 3, 4]), 
# shape of index torch.Size([3, 2]), 
# shape of result torch.Size([3, 2, 3, 4])
print(f"Shape of arrary {array.shape}, shape of index {index.shape}, shape of result {result.shape}")