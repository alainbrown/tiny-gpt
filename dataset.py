import torch

class Dataset:

  def __init__(self, tokens, context):
    self.tokens = tokens
    self.context = context

  def __getitem__(self, index):
    x = self.tokens[index : index + self.context]
    y = self.tokens[index + 1 : index + self.context + 1]

    return (
        torch.tensor(x, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )

  def __len__(self):
    return len(self.tokens) - self.context
