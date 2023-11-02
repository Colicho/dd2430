import torch.nn as nn

class SiameseNetworkSimple(nn.Module):
    def __init__(self):
        super(SiameseNetworkSimple, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(21, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        output = self.fc(x)
        return output
    

class SiameseNetworkComplex(nn.Module):
    def __init__(self):
        super(SiameseNetworkComplex, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(21, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        output = self.fc(x)
        return output
    