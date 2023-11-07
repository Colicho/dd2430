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
    