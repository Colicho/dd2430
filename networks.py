import torch.nn as nn

class SiameseNetworkSimple(nn.Module):
    def __init__(self):
        super(SiameseNetworkSimple, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(21, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        output = self.fc(x)
        return output

class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(48, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 7)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.convolutional(x)
        x = self.flatten(x)
        x = self.fc(x)
    
        return x
    