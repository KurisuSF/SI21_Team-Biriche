import torch.nn as nn
import torch
from pathlib import Path

file_path = Path(__file__).parent.absolute()

#TODO: Define la red neuronal
class Network(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=32, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * (input_dim//4) * (input_dim//4), 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def main():
    net = Network(3, 43)
    print(net)
    torch.rand(1, 3, 32, 32)
    print(net(torch.rand(1, 3, 32, 32)))


if __name__ == "__main__":
    main()
