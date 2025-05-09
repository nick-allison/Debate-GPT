import torch

class TextDataset:
    """
    Splits data into train/val sets. Yields random context chunks.
    """
    def __init__(self, encoded_data, context_size=128, batch_size=32, split_factor=0.9):
        self.context_size = context_size
        self.batch_size = batch_size
        self.data = encoded_data
        n = int(len(encoded_data) * split_factor)
        self.train_data, self.val_data = self.data[:n], self.data[n:]

    def get_batch(self, split, device="cpu"):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.context_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.context_size] for i in ix])
        y = torch.stack([data[i+1:i+self.context_size+1] for i in ix])
        return x.to(device), y.to(device)
