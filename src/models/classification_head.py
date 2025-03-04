import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Model head for classification tasks (computes logits).
    Arguments:
        num_classes: number of output classes.
        input_dim: dimension of head input.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features=input_dim, out_features=self.num_classes)
    
    def forward(self, x):
        return self.fc(x)
